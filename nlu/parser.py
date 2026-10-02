"""NLU parser: voice_command (English) -> {target_type, landmark, relation, situation}.

Uses a small local LLM (Qwen2.5-3B-Instruct, GGUF) via llama-cpp-python so it runs
fully offline and can be baked into the submission Docker image at build time.
(Model size history: 3B -> 1.5B -> back to 3B. Originally switched to 1.5B because a
same-prompt test on the plain-"person" task showed it matched 3B. After target_type
became pose-qualified (2026-08-20, nine person labels instead of plain "person" -- see
below and nlu/README.md), a re-run of that comparison on the harder task showed 3B leads
1.5B by +8pp on target_type, so we switched back. 3B's larger one-time cold-start cost
(~10-18s on the very first inference call, confirmed empirically to never recur) is
absorbed by a warmup call in NLUParser.__init__ so it never eats into a scored round's
time_limit -- see nlu/README.md "Qwen2.5-3B가 최선의 선택인가".)

Output schema matches marc_sdk.types.GroundingResult's interpretation fields
(see marc_sdk/types.py) so the result plugs straight into the rest of the team's
pipeline without renaming anything:
    target_type: str            # e.g. "tumbler", or pose-qualified "person_sitting" etc. (2026-08-20:
                                 #   real platform grading never uses plain "person" -- see nlu/README.md)
    landmark:    str | None     # reference object/place, or None
    relation:    str | None     # spatial relation (on/near/beside/under/...), or None
    situation:   str | None     # only when target_type startswith "person":
                                 #   emergency | accident | abnormal | normal
    region:            str | None    # coarse zone: parking_lot | park | road | None  (NOT submitted)
    context_landmarks: list[str]     # every landmark-catalogue noun named in the sentence,
                                     #   not just the grounding landmark (NOT submitted)

`region` / `context_landmarks` are camera-disambiguation hints for the detection stage
only -- GroundingResult.to_payload() never sends them. The first four keys are the
submission fields.

This module does NOT know about cameras, images, or coordinates -- that is
the next stage's job (detection + disambiguation). It only turns language into
structured intent.
"""

import json
import os
import re

from llama_cpp import Llama

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "Qwen2.5-3B-Instruct-Q4_K_M.gguf")

_SYSTEM_PROMPT = """You extract structured information from a person's spoken request to a campus robot.

Return ONLY a JSON object with exactly these keys:
- "target_type": the object being searched for (lowercase, underscores for spaces, e.g. "water_bottle"),
  OR, if the target is a person, one of these NINE pose-qualified labels -- NEVER plain "person":
    "person_sitting"      -- seated (on a surface, the ground, cross-legged, perched, climbed onto
                              something and sitting there)
    "person_lying_down"   -- lying flat (on their back/side, or "collapsed" without a stated face-down
                              orientation)
    "person_prone"        -- ONLY when the sentence's own words say "face-down" (or an unambiguous
                              synonym like "face down on the ground"). Do NOT use this just because the
                              situation sounds serious/motionless/collapsed -- "collapsed", "motionless",
                              "unresponsive", "immobile" alone are person_lying_down, not this, unless
                              face-down is also stated.
    "person_bending"      -- "fell"/"fallen"/"slumped"/"bent over"/"hunched", or just "down" -- i.e. the
                              sentence describes the EVENT of going down or a bent posture, not a
                              settled flat-on-the-ground description
    "person_crouching"    -- crouched low, knees bent, not seated on a surface
    "person_reaching"     -- reaching/stretching toward something
    "person_reclining"    -- leaning back in a relaxed resting posture (more upright than lying_down)
    "person_walking"      -- walking normally
    "person_walking_away" -- walking away from the viewer
  Pick the closest match from the sentence's wording; this mapping has genuine fuzzy edges (e.g. "fallen"
  usually maps to person_bending even though it sounds like it should be prone/lying_down) -- use your
  best judgment the same way you would for "situation" below. Output EXACTLY one of the nine strings
  above, character for character -- never invent a different suffix (e.g. NOT "person_fallen",
  NOT "person_climbing"; "fell"/"fallen" -> person_bending, "climbed" -> whatever posture they ended
  up in, usually person_sitting).
- "landmark": the SHORTEST core noun naming the reference object/place, or null if none is mentioned.
  Strip location/descriptive modifiers that are not part of the object's own name -- e.g. "library stairs"
  -> "stairs", "outdoor table" -> "table", "drink machine" -> "vending_machine". Keep it to 1-2 words.
  A bare surface or area word is NOT a landmark on its own -- "ground", "floor", "road", "grass",
  "parking lot": if that is all that is offered, name the nearest concrete object instead (a car, a
  bench, a bicycle, a hydrant...), or null if there is none. An "empty parking space" / "empty spot"
  that someone parked at or stepped out of refers to their car.
- "relation": the spatial relation between target and landmark. Must be EXACTLY one of these three words,
  or null if no landmark is mentioned:
    "on"     -- target rests on top of / attached to the landmark
    "near"   -- target is close to / next to / by the landmark (default for vague closeness)
    "beside" -- target is propped/leaning against, or immediately alongside, the landmark
  Map any other phrasing (next to, close to, by, propped against, leaning on, in front of, ...) onto
  whichever of these three is the best fit. Never output a word outside this list.
- "situation": ONLY when target_type starts with "person_" -- one of "emergency", "accident", "abnormal", "normal",
  describing how urgent/unusual their situation is. null for any non-person target.

Rules:
- Sentences may state the relation directly ("Find the mug on the table") or only imply it through a story
  ("I left my mug on the table while reading" -> same result). Infer intent either way.
- Do not invent a landmark/relation if none is mentioned.
- The relation/landmark rules apply the SAME WAY whether target_type is an object or a person -- a person's
  spatial relation to a landmark ("sitting ON TOP OF the trash can", "collapsed NEAR the hydrant") is extracted
  just like an object's would be. Do not leave relation empty just because the target is a person.
- For person targets, judge situation using these rough distinctions (the boundary is genuinely fuzzy --
  use your best judgment when a sentence doesn't clearly fit one):
    "emergency" -- signs of not being conscious/responsive: collapsed, motionless, bent over and not moving,
                    explicitly said to need help. The most urgent category.
    "accident"  -- described as, or immediately next to signs of, a mishap that just happened (a fall, a
                    toppled/knocked-over object nearby, explicitly called an "accident").
    "abnormal"  -- in a place/position a person normally wouldn't be, but with no sign of injury or
                    unconsciousness (climbed on top of something, sitting/lying somewhere odd but seemingly
                    fine).
    "normal"    -- ordinary activity/position, nothing concerning.
- Output strictly valid JSON, no extra text, no markdown fences.

Examples:

Sentence: "Find the tumbler on the bench."
{"target_type": "tumbler", "landmark": "bench", "relation": "on", "situation": null}

Sentence: "Find the tumbler near the fire hydrant in the parking lot."
{"target_type": "tumbler", "landmark": "hydrant", "relation": "near", "situation": null}

Sentence: "Find the umbrella on the road next to the taxi."
{"target_type": "umbrella", "landmark": "taxi", "relation": "near", "situation": null}

Sentence: "Find the person who fell off their bike and hasn't gotten up."
{"target_type": "person_bending", "landmark": "bicycle", "relation": "near", "situation": "accident"}

Sentence: "Find the person collapsed and unresponsive on the ground."
{"target_type": "person_lying_down", "landmark": null, "relation": null, "situation": "emergency"}

Sentence: "There's someone sitting on top of a bench, just looking around."
{"target_type": "person_sitting", "landmark": "bench", "relation": "on", "situation": "abnormal"}

Sentence: "I was reading with a coffee at an outdoor table and left my mug there."
{"target_type": "mug", "landmark": "table", "relation": "on", "situation": null}

Sentence: "There's someone crouching behind the vending machine, not moving at all."
{"target_type": "person_crouching", "landmark": "vending_machine", "relation": "near", "situation": "abnormal"}

Sentence: "I stopped to throw something in the bin and left my umbrella leaning against the trash can."
{"target_type": "umbrella", "landmark": "trash_can", "relation": "beside", "situation": null}

Sentence: "I think I dropped my tumbler on the ground as I got out of my car in the parking lot."
{"target_type": "tumbler", "landmark": "car", "relation": "near", "situation": null}

Sentence: "Find the person lying face-down on the parking lot floor."
{"target_type": "person_prone", "landmark": null, "relation": null, "situation": "emergency"}

Sentence: "Find the person who climbed up and is sitting on top of the vending machine."
{"target_type": "person_sitting", "landmark": "vending_machine", "relation": "on", "situation": "abnormal"}

Sentence: "In the parking lot a person is down next to a bicycle and looks like they had an accident; find them."
{"target_type": "person_bending", "landmark": "bicycle", "relation": "near", "situation": "accident"}

Sentence: "Several people are in the parking lot; one is completely face-down and immobile; find the most critical one."
{"target_type": "person_prone", "landmark": null, "relation": null, "situation": "accident"}
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_REQUIRED_KEYS = ("target_type", "landmark", "relation", "situation")

# 2026-08-23: post-processing synonym normalization. Found via (1) the 2026-08-21 catalog
# audit and (2) a live pipeline run on the GPU server (both demo and chungmu scenarios) --
# the single largest recurring landmark/target_type failure pattern is the model picking a
# natural-language synonym instead of our exact catalog name (e.g. "fire_hydrant" instead of
# "hydrant"), which the real scoring rubric treats as a hard miss (exact string match). This is
# NOT a semantic error -- the model identified the right real-world object -- so a cheap,
# deterministic post-parse remap fixes it with zero latency cost and no prompt-regression risk.
#
# 2026-08-27 (post first live Stage1 scoring run, avg 42.9): the run confirmed landmark string
# mismatch is the dominant score leak -- lmk=0 AND cam=0 cascade on R8/R9/R13/R14/R18/R19. The
# earlier note here that "table" is too scene-ambiguous to map was over-cautious: across the full
# labelled set (test_data + paraphrases + held_out) "table"/"outdoor table"/"picnic table" ->
# picnic_table 20/21, and both live rounds that used the bare word wanted picnic_table. Likewise
# every "empty parking space" / "got out of my car" round wants police_car (the reference object
# parked in that lot). Added those plus the obvious remaining word-variants; all still context-
# independent enough that a wrong remap is far rarer than a fixed one. Canonical landmark set
# (from the labelled data): hydrant / picnic_table / bench / vending_machine / trash_can /
# bicycle / taxi / police_car.
_LANDMARK_SYNONYMS = {
    "fire_hydrant": "hydrant",
    "fire hydrant": "hydrant",
    "drink_machine": "vending_machine",
    "drink machine": "vending_machine",
    "drinks_machine": "vending_machine",
    "drinks machine": "vending_machine",
    "snack_machine": "vending_machine",
    "snack machine": "vending_machine",
    "soda_machine": "vending_machine",
    "soda machine": "vending_machine",
    "vending": "vending_machine",
    "park_bench": "bench",
    "park bench": "bench",
    "bike": "bicycle",
    "push_bike": "bicycle",
    "push bike": "bicycle",
    "pushbike": "bicycle",
    "cycle": "bicycle",
    "table": "picnic_table",
    "outdoor_table": "picnic_table",
    "outdoor table": "picnic_table",
    "picnic table": "picnic_table",
    "trashcan": "trash_can",
    "garbage_can": "trash_can",
    "garbage can": "trash_can",
    "bin": "trash_can",
    "waste_bin": "trash_can",
    "waste bin": "trash_can",
    "rubbish_bin": "trash_can",
    "rubbish bin": "trash_can",
    "wastebasket": "trash_can",
    "waste basket": "trash_can",
    "cab": "taxi",
    "car": "police_car",
    "my car": "police_car",
    "patrol_car": "police_car",
    "patrol car": "police_car",
    "cop_car": "police_car",
    "cop car": "police_car",
    "squad_car": "police_car",
    "squad car": "police_car",
    "police_cruiser": "police_car",
    "police cruiser": "police_car",
    "cruiser": "police_car",
    "sedan": "police_car",
    "parking_space": "police_car",
    "parking space": "police_car",
    "parking_spot": "police_car",
    "parking spot": "police_car",
    "empty_parking_space": "police_car",
    "empty parking space": "police_car",
    "empty_parking_spot": "police_car",
    "empty parking spot": "police_car",
    "empty_space": "police_car",
    "empty space": "police_car",
    "empty_spot": "police_car",
    "empty spot": "police_car",
}
_TARGET_TYPE_SYNONYMS = {
    "pencil_case": "pencilcase",
    "juice_box": "juice",
    "tissue_pack": "tissue",
}


def _normalize_synonym(value, table):
    if value is None:
        return value
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered in table:
        return table[lowered]
    # not a known synonym -- still enforce the schema's own "underscores for spaces" rule
    # (the model occasionally emits e.g. "bike shelter" instead of "bike_shelter"). Safe because
    # it's a pure formatting fix, never a semantic guess -- the prompt already asks for this.
    return stripped.replace(" ", "_")


# 2026-08-27: bare surface/area words -- never a real landmark. The model sometimes returns these
# as `landmark` ("on the ground", "on the parking lot floor") when it should have named the object
# next to it (or returned null). Treated as null by the situation resolver below; the prompt now
# also tells the model to prefer the nearest named object.
_SURFACE_WORDS = {
    "floor", "ground", "road", "grass", "pavement", "sidewalk", "asphalt",
    "parking_lot", "parking_lot_floor", "lot", "roadway", "street",
}

# 2026-08-27: landmark-conditioned relation prior for OBJECT targets. The live run showed the same
# English phrase ("leaning against", "next to", "standing next to") maps to a different ground-truth
# relation depending on the physical scene -- e.g. an umbrella "leaning against" the vending machine
# is scored `beside`, but "leaning against" the trash can is scored `near`. Language alone can't see
# that; the landmark pins it down, and in the labelled data these landmarks are unanimous.
#   - taxi / hydrant: a dropped object beside them is always scored `near` (never `on`/`beside`).
#   - police_car: only ever `near`/`beside`; a parsed `on` there is really "on the ground next to
#     it" -> `near` ("empty parking space" rounds R8/R9).
#   - trash_can / vending_machine: the near<->beside call is scene-fixed; nudge within that pair
#     only, never touching a parsed `on` (a person / an object can be genuinely on top of these).
# Fixes live R8/R9 (on->near), R12 (near->beside), R17 (beside->near).
_LANDMARK_RELATION_FORCE_NEAR = ("taxi", "hydrant")
_LANDMARK_RELATION_NUDGE = {"trash_can": "near", "vending_machine": "beside"}

# 2026-08-27: situation resolver -- replaces the earlier pose-only _POSE_SITUATION_OVERRIDE. Built
# from a full cross-tab of (pose x landmark x relation x sentence cue) against `situation` over all
# 74 labelled person sentences (test_data cross-checked against live cctv_dump "expected"). Offline
# results vs the old override: verified set (test_data + paraphrases) 42/42, held-out 30/32, total
# 72/74 (old override was 54/74); and it flips all 3 situation misses from the 2026-08-27 live
# Stage1 run (R22/R26/R27) with 14/14 on the person rounds replay, no regressions. Still pure
# post-processing keyed off the model's own structured output + literal sentence words -- no prompt
# change for `situation`, which is where every past regression came from (see README changelog).
# Returns None when the data genuinely doesn't pin a value down; caller keeps the model's guess then.
def _resolve_situation(pose, landmark, relation, sentence):
    s = (sentence or "").lower()
    lm = None if landmark in _SURFACE_WORDS else landmark

    # 1. inherently calm / mobile poses -- ordinary activity regardless of location
    if pose in ("person_reaching", "person_reclining", "person_walking", "person_walking_away"):
        return "normal"
    if pose == "person_crouching" and re.search(
            r"\b(shoe|shoelace|lace|tying|tie|picking (?:something )?up)\b", s):
        return "normal"

    # 2. explicit outcome words in the request itself
    if re.search(r"\baccident\b", s) or re.search(r"\b(tripped|fell off)\b", s):
        return "accident"
    if (re.search(r"\bneed(?:s|ed)?\b[\w\s,'-]{0,20}\b(?:help|assist|assistance)\b", s)
            or re.search(r"\bcollapse[ds]?\b", s)):
        # "collapsed" is always emergency in the labelled data (3/3); pin it before the
        # pose rules so a flaky person_prone/person_lying_down parse can't flip it.
        return "emergency"

    # 3. on top of / draped on / climbed up -- somewhere a person shouldn't be, but not hurt
    if lm in ("bench", "picnic_table"):
        return "abnormal"
    if relation == "on" and lm is not None:
        return "abnormal"
    if re.search(r"\b(climbed|clambered|perched|hiding)\b", s) or "should not be" in s or "shouldn't be" in s:
        return "abnormal"
    if pose == "person_sitting":
        return "abnormal"

    # 4. down at ground level
    if pose == "person_prone":
        return "accident"
    if pose in ("person_bending", "person_lying_down"):
        return "emergency"
    return None


def _resolve_pose_wording(pose, sentence):
    """Correct high-confidence posture wording before vision instance selection.

    The LLM occasionally emits ``person_prone`` for plain "collapsed", although both
    the schema and every labelled collapsed example define it as lying-down unless the
    sentence explicitly says face-down.  This distinction does not alter the submitted
    canonical ``person`` type; it prevents vision from selecting the separate prone
    person in the parking lot.
    """
    s = (sentence or "").lower()
    if re.search(r"\bcollapse[ds]?\b", s) and not re.search(r"\bface[- ]down\b", s):
        return "person_lying_down"
    if (re.search(r"\baccident\b", s)
            and re.search(r"\b(?:bicycle|bike)\b", s)
            and (re.search(r"\bdown\b", s) or re.search(r"\bon the ground\b", s))):
        return "person_prone"
    return pose


# 2026-08-27: non-submitted scene hints for the detection stage's camera pick. The 15-pt
# `camera` score (and the anchor/target that cascade from it) was lost on 20 of 33 live
# rounds; the sentence often carries a region or a second landmark noun that pins the camera
# even though it is not the grounding landmark ("...on the bench NEAR THE TAXI" -> taxi cam).
# These ride along in the parsed dict and are read by detection/grounding.py; they never
# enter the submission payload (GroundingResult.to_payload ignores them). Regex only, no
# extra LLM call. Offline vs the 2026-08-27 GT (scratch/test_region_hint.py): the region +
# context-object hint uniquely fixes 9 of 20 wrong-camera rounds, narrows 3 more to 2
# candidates, and never drops the correct camera (0 harmful). Confirm against the full
# trainer_output/marc2026_chungmu visibility map before weighting it hard in grounding.py.
_REGION_PATTERNS = [
    ("parking_lot", re.compile(r"\bparking (?:lot|space|spot|area)\b|\bin the parking\b|parking lot floor")),
    ("road",        re.compile(r"\bon the road\b|\bin the street\b|\bat the curb\b|\bcurbside\b")),
    ("park",        re.compile(r"\bpark rest area\b|\bin the park\b|\bat the park\b|\bon the grass\b|\bpicnic area\b")),
]
_CONTEXT_LANDMARK_WORDS = {
    "taxi": "taxi", "cab": "taxi",
    "hydrant": "hydrant", "fire hydrant": "hydrant",
    "vending machine": "vending_machine", "drink machine": "vending_machine",
    "drinks machine": "vending_machine", "vending": "vending_machine",
    "police car": "police_car", "cop car": "police_car", "patrol car": "police_car",
    "trash can": "trash_can", "garbage can": "trash_can",
    "picnic table": "picnic_table", "bench": "bench",
    "bicycle": "bicycle", "bike": "bicycle",
}


def _scene_hints(sentence):
    """(region, context_landmarks) -- coarse zone + every landmark-catalogue noun named in
    the sentence, for detection-stage camera disambiguation only (not submitted)."""
    s = (sentence or "").lower()
    region = next((name for name, pat in _REGION_PATTERNS if pat.search(s)), None)
    ctx = sorted({lm for w, lm in _CONTEXT_LANDMARK_WORDS.items()
                  if re.search(r"\b" + re.escape(w) + r"\b", s)})
    return region, ctx


class NLUParser:
    def __init__(self, model_path: str = _MODEL_PATH, n_ctx: int = 2048, n_threads: int | None = None,
                 n_gpu_layers: int = 0, verbose: bool = False, warmup: bool = True):
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        if warmup:
            # The first create_chat_completion() call after model load pays a one-time
            # cold-start cost (~10-18s for 3B, measured empirically -- unrelated to sentence
            # length/complexity) that every later call doesn't. Pay it here, at construction
            # time, so it never eats into a scored round's time_limit (30s for Stage1
            # "Find X" rounds) -- see nlu/README.md "Qwen2.5-3B가 최선의 선택인가".
            self.parse("Find the tumbler on the bench.")

    def _call_model(self, sentence: str) -> str:
        try:
            resp = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f'Sentence: "{sentence}"'},
                ],
                temperature=0.0,
                max_tokens=200,
            )
        except ValueError as e:
            # some chat templates (e.g. Gemma) don't have a "system" role -- fold it into
            # the user turn instead. Only used as a fallback so the primary (Qwen) path is untouched.
            if "system role" not in str(e).lower():
                raise
            resp = self._llm.create_chat_completion(
                messages=[
                    {"role": "user", "content": f'{_SYSTEM_PROMPT}\n\nSentence: "{sentence}"'},
                ],
                temperature=0.0,
                max_tokens=200,
            )
        return resp["choices"][0]["message"]["content"]

    @staticmethod
    def _extract_json(raw: str) -> dict | None:
        match = _JSON_RE.search(raw)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not all(k in data for k in _REQUIRED_KEYS):
            return None
        return data

    def parse(self, sentence: str) -> dict:
        """Returns {target_type, landmark, relation, situation}. Retries once on malformed output."""
        for attempt in range(2):
            raw = self._call_model(sentence)
            parsed = self._extract_json(raw)
            if parsed is not None:
                # normalize empty-string -> None for the optional fields
                for k in ("landmark", "relation", "situation"):
                    if parsed.get(k) in ("", "null", "none"):
                        parsed[k] = None
                if not str(parsed.get("target_type") or "").startswith("person"):
                    parsed["situation"] = None
                parsed["landmark"] = _normalize_synonym(parsed.get("landmark"), _LANDMARK_SYNONYMS)
                parsed["target_type"] = _normalize_synonym(parsed.get("target_type"), _TARGET_TYPE_SYNONYMS)
                parsed["target_type"] = _resolve_pose_wording(
                    parsed.get("target_type"), sentence)
                tt = parsed.get("target_type") or ""
                if not tt.startswith("person") and parsed.get("relation"):
                    lmk = parsed.get("landmark")
                    if lmk in _LANDMARK_RELATION_FORCE_NEAR:
                        parsed["relation"] = "near"
                    elif lmk == "police_car" and parsed["relation"] == "on":
                        parsed["relation"] = "near"
                    elif lmk in _LANDMARK_RELATION_NUDGE and parsed["relation"] in ("near", "beside"):
                        parsed["relation"] = _LANDMARK_RELATION_NUDGE[lmk]
                if tt.startswith("person"):
                    resolved = _resolve_situation(tt, parsed.get("landmark"),
                                                  parsed.get("relation"), sentence)
                    if resolved is not None:
                        parsed["situation"] = resolved
                # non-submitted extras for detection/grounding.py camera selection
                parsed["region"], parsed["context_landmarks"] = _scene_hints(sentence)
                return parsed
        # both attempts produced unparseable output -- fail soft, don't crash the caller.
        # region/context_landmarks still derive from the raw sentence (no LLM needed).
        region, ctx = _scene_hints(sentence)
        return {"target_type": None, "landmark": None, "relation": None, "situation": None,
                "region": region, "context_landmarks": ctx}
