"""Language-guided visual grounding using the NLU, detection, and geometry modules."""

import logging

from marc_sdk import GroundingResult

log = logging.getLogger("participant_app")


class VLAGrounding:
    """Team dongdong's language, visual grounding, and geometry pipeline.

    Three stages, one per teammate -- this class is the seam where they meet:
      1. Language (은재)  -- nlu.parser.NLUParser: voice_command -> {target_type, landmark,
         relation, situation}. Implemented, see nlu/.
      2. Detection + disambiguation (은서) -- detection.grounding.VisualGrounding: picks the
         camera and the right instance among candidates, and returns world coordinates.
         Implemented, see detection/.
      3. Geometry (효빈) -- geometry.camera.CameraGeometry, called *by* stage 2 rather than
         after it (a candidate can only be chosen once its coordinates are known). See
         geometry/.

    The stages degrade independently: if detection finds nothing, or geometry is not ready,
    the language fields are still submitted on their own. Those are half the Stage 1 score
    ("객체·상황 해석" / "랜드마크와 공간 관계"), so a partial answer is worth much more than
    a crash or an empty submission.
    """

    def __init__(self, client=None):
        # imports are local so this module stays importable without llama-cpp-python/torch
        # installed until the pipeline is constructed.
        from nlu.parser import NLUParser
        self._nlu = NLUParser()
        self._client = client
        self._vision = None
        # Set once vision is known to be unavailable so _ensure_vision() stops retrying and
        # the "language-only" warning is logged once, not every round.
        self._vision_failed = client is None
        if client is None:
            # Not fatal, but it silently costs the coordinate half of the score, so say
            # so loudly rather than letting a language-only run look healthy.
            log.warning("[real-vla] constructed without a client -- detection and geometry "
                        "are DISABLED, only language fields will be submitted. "
                        "Pass the MARCClient: VLAGrounding(self.client)")
        # VisualGrounding is built lazily on the first _result() call, NOT here:
        # DemoParticipant constructs this object before client.connect() has created the
        # rclpy node, and VisualGrounding -> CameraGeometry subscribes to /tf_static the
        # moment it is constructed, which needs that node. It exists by the first mission.

    def _ensure_vision(self):
        """Build the detection + geometry stage once, on first use.

        Deferred from __init__ because CameraGeometry subscribes to /tf_static as soon as
        it is constructed, and DemoParticipant builds this object before client.connect()
        has created the rclpy node. Tried at most once; on failure the round falls back to
        language-only answers.
        """
        if self._vision is not None or self._vision_failed:
            return
        try:
            from detection.grounding import VisualGrounding
            self._vision = VisualGrounding(self._client)
        except Exception as e:  # noqa: BLE001 -- torch/weights missing, node not ready, ...
            self._vision_failed = True
            log.warning("[real-vla] vision stage unavailable (%s) -- "
                        "submitting language-only answers", e)

    def _result(self, voice_command: str, camera_ids: list = None) -> GroundingResult:
        parsed = self._nlu.parse(voice_command)
        # Keep the NLU submission fields and detector routing on the same canonical
        # catalogue spellings. The parser already handles these aliases; this is the
        # boundary guard for alternate parsers and future prompt changes.
        from detection.images import normalize_detection_label
        parsed["target_type"] = normalize_detection_label(parsed.get("target_type"))
        parsed["landmark"] = normalize_detection_label(parsed.get("landmark"))
        log.info("[real-vla] parsed %r -> %s", voice_command, parsed)

        camera_id = (camera_ids or [""])[0]
        anchor_coord = [0.0, 0.0, 0.0]
        target_coord = [0.0, 0.0, 0.0]
        landmark = parsed.get("landmark") or ""

        self._ensure_vision()
        if self._vision is not None:
            try:
                found = self._vision.ground(parsed, camera_ids)
            except Exception as e:  # noqa: BLE001 -- perception must never kill a round
                log.exception("[real-vla] vision stage failed: %s", e)
                found = None
            if found:
                camera_id = found["camera_id"] or camera_id
                if found["target_coord"]:
                    target_coord = found["target_coord"]
                    anchor_coord = found["anchor_coord"] or found["target_coord"]
                # Detection can name a landmark the sentence never mentioned (12 of the
                # 39 practice commands are like that), so let it fill the blank in.
                landmark = found["landmark"] or landmark

        return GroundingResult(
            camera_id=camera_id,
            # The NLU pose label is for visual instance selection.  The Stage 1 API
            # requires the canonical submission label "person" for every SAR problem.
            target_type=("person" if (parsed.get("target_type") or "").startswith("person_")
                         else parsed.get("target_type") or ""),
            anchor_coord=anchor_coord,
            target_coord=target_coord,
            landmark=landmark,
            relation=parsed.get("relation"),
            situation=parsed.get("situation"),
        )

    def process(self, voice_command: str, camera_ids: list = None) -> GroundingResult:
        return self._result(voice_command, camera_ids)

    def process_stage2(self, task_description: str) -> GroundingResult:
        return self._result(task_description)
