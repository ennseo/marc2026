"""Hand-written sentences NOT in test_data.py's 39 known examples.

Purpose: check that the parser generalizes to unseen vocabulary/phrasing,
not just memorized patterns. Uses object/landmark words that never appear
in the 39 known examples.

expected_* fields are our own best-guess ground truth (there is no official
answer for these -- eyeball the parser's output against them).

NOTE (2026-08-15): expected_relation for person entries is now extracted the same
way as for objects (not forced to null) -- see nlu/README.md. All person entries
here happen to have a landmark stated in the text, so all of them get a real
on/near/beside value now, none stay null.

NOTE (2026-08-20): expected_target_type for person entries is now pose-qualified
(person_sitting/person_bending/person_lying_down/person_prone) instead of plain
"person" -- see test_data.py's 2026-08-20 note for why (real platform grading never
uses plain "person"). Unlike test_data.py's known-39, there is no live scoring data
for these hand-written sentences, so the pose assigned to each is our own best-guess
inference from the wording (by analogy to the confirmed known-39 mappings), not
independently confirmed. Treat these 11 as reasonable estimates, not ground truth.

NOTE (2026-08-21): +16 more added (40 -> 56). Cross-checked the confirmed 34-class
catalog (trainer_output/marc2026_chungmu) against every target_type/landmark used so
far across test_data.py + this file, and found normal_car/sports_car had NEVER been
used as a landmark (their `kind` is "landmark", confirmed via the same trainer_output
scan -- they can only ever be landmarks, never targets, same for bicycle/kick_scooter)
despite being 2 of the 34 catalog classes. Also ran a systematic per-class usage count
across all of test_data.py + paraphrases.py + this file and found person_reaching had
ZERO uses anywhere -- not one test case, and not even a few-shot example in parser.py
(only the field description mentions it). This batch fills the normal_car/sports_car
gap, closes the person_reaching blind spot, adds a few new person-pose+landmark
pairings not tested before, and 2 fully-invented target+landmark pairs
(flashlight/dumpster, umbrella/guardrail) to keep the pure-vocabulary-generalization
ratio roughly in line with the 2026-08-19 batch. One entry (person_crouching) was
rewritten after a leakage re-check found it echoed a few-shot example's exact
structure too closely -- see its inline comment. Same caveat as above: best-guess
labels, not independently confirmed.

NOTE (2026-08-24): +15 more added (56 -> 71), 3 each for the 5 person poses
(crouching/reclining/walking/walking_away/reaching) that the 2026-08-21 audit found had
only 1 held-out example apiece. Same best-guess-label caveat applies.
"""

HELD_OUT_EXAMPLES = [
    {
        "sentence": 'Find the backpack under the bus stop bench.',
        "expected_target_type": 'backpack',
        "expected_landmark": 'bench',
        "expected_relation": 'near',
        "expected_situation": None,
    },
    {
        "sentence": 'Find the water bottle on the bike rack.',
        "expected_target_type": 'water_bottle',
        "expected_landmark": 'bike_rack',
        "expected_relation": 'on',
        "expected_situation": None,
    },
    {
        "sentence": 'I was charging my laptop near the bulletin board and forgot it there.',
        "expected_target_type": 'laptop',
        "expected_landmark": 'bulletin_board',
        "expected_relation": 'near',
        "expected_situation": None,
    },
    {
        "sentence": 'After class I left my notebook on the stairs by the library entrance.',
        "expected_target_type": 'notebook',
        "expected_landmark": 'stairs',
        "expected_relation": 'on',
        "expected_situation": None,
    },
    {
        "sentence": 'Find the person slumped over at the bus stop.',
        "expected_target_type": 'person_bending',
        "expected_landmark": 'bus_stop',
        "expected_relation": 'near',
        "expected_situation": 'emergency',
    },
    {
        "sentence": 'Someone is lying motionless near the bike rack; check on them.',
        "expected_target_type": 'person_lying_down',
        "expected_landmark": 'bike_rack',
        "expected_relation": 'near',
        "expected_situation": 'emergency',
    },
    {
        "sentence": 'Find the badge clipped to the bulletin board.',
        "expected_target_type": 'badge',
        "expected_landmark": 'bulletin_board',
        "expected_relation": 'on',
        "expected_situation": None,
    },
    {
        # NOTE (2026-08-16): originally this was "There are a few students studying by the library
        # stairs, and one has been sitting oddly still for a while..." which turned out to be a
        # near copy of one of parser.py's few-shot prompt examples -- not a real generalization
        # test. Replaced with an unrelated scene/vocabulary to close that leak. See README "few-shot
        # 예시와 테스트셋 중복(데이터 누출) 점검".
        "sentence": "A student has been perched cross-legged on the roof of the bike shelter for the last ten minutes, just staring at nothing.",
        "expected_target_type": 'person_sitting',
        "expected_landmark": 'bike_shelter',
        "expected_relation": 'on',
        "expected_situation": 'abnormal',
    },
    {
        "sentence": 'Find the keys on the parking meter.',
        "expected_target_type": 'keys',
        "expected_landmark": 'parking_meter',
        "expected_relation": 'on',
        "expected_situation": None,
    },
    {
        "sentence": 'Find the wallet beside the fountain.',
        "expected_target_type": 'wallet',
        "expected_landmark": 'fountain',
        "expected_relation": 'beside',
        "expected_situation": None,
    },
    {
        "sentence": 'Find the phone near the elevator.',
        "expected_target_type": 'phone',
        "expected_landmark": 'elevator',
        "expected_relation": 'near',
        "expected_situation": None,
    },
    {
        "sentence": 'I left my scarf hanging on the lamppost.',
        "expected_target_type": 'scarf',
        "expected_landmark": 'lamppost',
        "expected_relation": 'on',
        "expected_situation": None,
    },
    {
        "sentence": 'I was waiting by the recycling bin and forgot my gloves there.',
        "expected_target_type": 'gloves',
        "expected_landmark": 'recycling_bin',
        "expected_relation": 'near',
        "expected_situation": None,
    },
    {
        "sentence": 'Find the jacket draped over the planter box.',
        "expected_target_type": 'jacket',
        "expected_landmark": 'planter_box',
        "expected_relation": 'on',
        "expected_situation": None,
    },
    {
        "sentence": 'I was sitting by the fountain and left my tablet there.',
        "expected_target_type": 'tablet',
        "expected_landmark": 'fountain',
        "expected_relation": 'near',
        "expected_situation": None,
    },
    {
        "sentence": 'Find the charger plugged in near the vending area.',
        "expected_target_type": 'charger',
        "expected_landmark": 'vending_area',
        "expected_relation": 'near',
        "expected_situation": None,
    },
    {
        "sentence": 'Find the umbrella propped against the gate.',
        "expected_target_type": 'umbrella',
        "expected_landmark": 'gate',
        "expected_relation": 'beside',
        "expected_situation": None,
    },
    {
        "sentence": "There's a badge that fell near the elevator, could you find it?",
        "expected_target_type": 'badge',
        "expected_landmark": 'elevator',
        "expected_relation": 'near',
        "expected_situation": None,
    },
    {
        "sentence": 'Find the person collapsed by the parking meter.',
        "expected_target_type": 'person_lying_down',
        "expected_landmark": 'parking_meter',
        "expected_relation": 'near',
        "expected_situation": 'emergency',
    },
    {
        "sentence": "There's someone who climbed up onto the lamppost.",
        "expected_target_type": 'person_sitting',
        "expected_landmark": 'lamppost',
        "expected_relation": 'on',
        "expected_situation": 'abnormal',
    },
    {
        "sentence": 'Find the person sitting motionless near the fountain.',
        "expected_target_type": 'person_sitting',
        "expected_landmark": 'fountain',
        "expected_relation": 'near',
        "expected_situation": 'emergency',
    },
    {
        "sentence": 'Someone tripped and fell near the recycling bins; check on them.',
        "expected_target_type": 'person_bending',
        "expected_landmark": 'recycling_bin',
        "expected_relation": 'near',
        "expected_situation": 'accident',
    },
    {
        "sentence": 'Find the person lying on the ground by the gate.',
        "expected_target_type": 'person_lying_down',
        "expected_landmark": 'gate',
        "expected_relation": 'near',
        "expected_situation": 'emergency',
    },
    {
        "sentence": "I noticed someone lying face-down near the planter box, they haven't moved in a while.",
        "expected_target_type": 'person_prone',
        "expected_landmark": 'planter_box',
        "expected_relation": 'near',
        "expected_situation": 'emergency',
    },

    # ---- 2026-08-19 batch: grounded in objects/landmarks actually confirmed via the
    # marc2026_chungmu dataset-gen run (trainer_output/*/*.json `class` fields), not
    # invented -- see nlu/next-gpu-session-checklist.md ("CCTV 데이터 기반 새 held-out
    # 문장 작성"). Object/landmark PAIRS below do not appear in test_data.py's 39 known
    # examples, so this still checks combo generalization, not memorization. ~75% real
    # catalog items (both sides), ~25% mix in a genuinely novel object/landmark word
    # (last 4) for pure-vocabulary generalization the way the original 24 above do.
    {
        "sentence": 'Find the juice box near the postbox.',
        "expected_target_type": 'juice',
        "expected_landmark": 'postbox',
        "expected_relation": 'near',
        "expected_situation": None,
    },
    {
        "sentence": "There's a cola can left on the ground next to the kick scooter.",
        "expected_target_type": 'cola_can',
        "expected_landmark": 'kick_scooter',
        "expected_relation": 'near',
        "expected_situation": None,
    },
    {
        "sentence": 'Find the disposable cup someone left on the picnic table.',
        "expected_target_type": 'disposable_cup',
        "expected_landmark": 'picnic_table',
        "expected_relation": 'on',
        "expected_situation": None,
    },
    {
        "sentence": 'I think I dropped a tissue pack near the bench.',
        "expected_target_type": 'tissue',
        "expected_landmark": 'bench',
        "expected_relation": 'near',
        "expected_situation": None,
    },
    {
        "sentence": 'Find the cracker box sitting beside the police car.',
        "expected_target_type": 'cracker_box',
        "expected_landmark": 'police_car',
        "expected_relation": 'beside',
        "expected_situation": None,
    },
    {
        "sentence": 'Find the sunblock left on the SUV.',
        "expected_target_type": 'sunblock',
        "expected_landmark": 'suv',
        "expected_relation": 'on',
        "expected_situation": None,
    },
    {
        "sentence": 'I left my mug near the taxi.',
        "expected_target_type": 'mug',
        "expected_landmark": 'taxi',
        "expected_relation": 'near',
        "expected_situation": None,
    },
    {
        "sentence": 'Find the umbrella propped beside the fire hydrant.',
        "expected_target_type": 'umbrella',
        "expected_landmark": 'hydrant',
        "expected_relation": 'beside',
        "expected_situation": None,
    },
    {
        "sentence": 'Someone left a tumbler by the postbox.',
        "expected_target_type": 'tumbler',
        "expected_landmark": 'postbox',
        "expected_relation": 'near',
        "expected_situation": None,
    },
    {
        "sentence": 'Find the pencil case on top of the vending machine.',
        "expected_target_type": 'pencilcase',
        "expected_landmark": 'vending_machine',
        "expected_relation": 'on',
        "expected_situation": None,
    },
    {
        "sentence": "There's someone down near the kick scooter, looks like they fell off it.",
        "expected_target_type": 'person_bending',
        "expected_landmark": 'kick_scooter',
        "expected_relation": 'near',
        "expected_situation": 'accident',
    },
    {
        "sentence": 'Find the person slumped over near the postbox.',
        "expected_target_type": 'person_bending',
        "expected_landmark": 'postbox',
        "expected_relation": 'near',
        "expected_situation": 'emergency',
    },
    {
        # novel object, real landmark (bike_rack: known from problems.yaml lf/hf ids, not
        # yet in our partial chungmu sample -- still a real catalog landmark, not invented).
        "sentence": 'I left my earbuds case on the bike rack.',
        "expected_target_type": 'earbuds_case',
        "expected_landmark": 'bike_rack',
        "expected_relation": 'on',
        "expected_situation": None,
    },
    {
        "sentence": 'Find the skateboard leaning against the vending machine.',
        "expected_target_type": 'skateboard',
        "expected_landmark": 'vending_machine',
        "expected_relation": 'near',
        "expected_situation": None,
    },
    {
        # fully novel (both sides) -- pure language generalization, no simulator grounding.
        "sentence": 'Find the thermos next to the newspaper stand.',
        "expected_target_type": 'thermos',
        "expected_landmark": 'newspaper_stand',
        "expected_relation": 'near',
        "expected_situation": None,
    },
    {
        "sentence": 'Someone left their gym bag beside the recycling station.',
        "expected_target_type": 'gym_bag',
        "expected_landmark": 'recycling_station',
        "expected_relation": 'beside',
        "expected_situation": None,
    },

    # ---- 2026-08-21 batch: fills the normal_car/sports_car landmark gap (confirmed via
    # trainer_output that these 2 of the 34 catalog classes had never been used anywhere),
    # a few new person-pose+landmark pairings, and 2 fully novel target+landmark pairs for
    # pure vocabulary generalization -- see docstring note above.
    {
        "sentence": 'I left my sunblock on the hood of the sedan.',
        "expected_target_type": 'sunblock',
        "expected_landmark": 'normal_car',
        "expected_relation": 'on',
        "expected_situation": None,
    },
    {
        "sentence": "There's a tissue pack that fell near the sports car.",
        "expected_target_type": 'tissue',
        "expected_landmark": 'sports_car',
        "expected_relation": 'near',
        "expected_situation": None,
    },
    {
        "sentence": 'Find the badge that fell near the sedan in the parking lot.',
        "expected_target_type": 'badge',
        "expected_landmark": 'normal_car',
        "expected_relation": 'near',
        "expected_situation": None,
    },
    {
        "sentence": 'Find the cracker box propped beside the sports car.',
        "expected_target_type": 'cracker_box',
        "expected_landmark": 'sports_car',
        "expected_relation": 'beside',
        "expected_situation": None,
    },
    {
        "sentence": 'I left my wallet next to the bicycle in the parking lot.',
        "expected_target_type": 'wallet',
        "expected_landmark": 'bicycle',
        "expected_relation": 'near',
        "expected_situation": None,
    },
    {
        "sentence": 'Find the keys sitting on top of the kick scooter.',
        "expected_target_type": 'keys',
        "expected_landmark": 'kick_scooter',
        "expected_relation": 'on',
        "expected_situation": None,
    },
    {
        "sentence": 'I left my charger near the sedan while loading groceries.',
        "expected_target_type": 'charger',
        "expected_landmark": 'normal_car',
        "expected_relation": 'near',
        "expected_situation": None,
    },
    {
        "sentence": "I left my sunglasses on the roof of the police car.",
        "expected_target_type": 'sunglasses',
        "expected_landmark": 'police_car',
        "expected_relation": 'on',
        "expected_situation": None,
    },
    {
        # NOTE (2026-08-21): originally "crouching behind the fire hydrant, not moving" --
        # too close to parser.py's own few-shot example ("crouching behind the vending
        # machine, not moving at all") in wording/structure -- caught during a leakage
        # re-check right after writing this batch, replaced before ever being used to
        # measure anything. See README "few-shot 예시와 테스트셋 중복" section.
        "sentence": 'Someone is crouched down beside the fire hydrant, retying their shoe.',
        "expected_target_type": 'person_crouching',
        "expected_landmark": 'hydrant',
        "expected_relation": 'beside',
        "expected_situation": 'normal',
    },
    {
        "sentence": 'Someone is reclining on the bench, looking relaxed.',
        "expected_target_type": 'person_reclining',
        "expected_landmark": 'bench',
        "expected_relation": 'on',
        "expected_situation": 'normal',
    },
    {
        "sentence": 'A person is walking away from the SUV toward the exit.',
        "expected_target_type": 'person_walking_away',
        "expected_landmark": 'suv',
        "expected_relation": 'near',
        "expected_situation": 'normal',
    },
    {
        "sentence": "There's a person walking beside the sports car toward the entrance.",
        "expected_target_type": 'person_walking',
        "expected_landmark": 'sports_car',
        "expected_relation": 'beside',
        "expected_situation": 'normal',
    },
    {
        "sentence": 'There\'s a person lying motionless near the trash can, someone should check on them.',
        "expected_target_type": 'person_lying_down',
        "expected_landmark": 'trash_can',
        "expected_relation": 'near',
        "expected_situation": 'emergency',
    },
    {
        # person_reaching had ZERO coverage anywhere before this -- not in test_data.py,
        # paraphrases.py, or held_out_sentences.py, AND not in parser.py's few-shot examples
        # either (only the field description mentions it). Found during a systematic
        # catalog-coverage audit; added specifically to close that blind spot.
        "sentence": 'There\'s a person reaching into the open trunk of the SUV, grabbing something.',
        "expected_target_type": 'person_reaching',
        "expected_landmark": 'suv',
        "expected_relation": 'near',
        "expected_situation": 'normal',
    },
    {
        # fully novel (both sides) -- pure language generalization, no simulator grounding.
        "sentence": 'I think I dropped my flashlight near the dumpster.',
        "expected_target_type": 'flashlight',
        "expected_landmark": 'dumpster',
        "expected_relation": 'near',
        "expected_situation": None,
    },
    {
        "sentence": 'Find the umbrella propped against the guardrail.',
        "expected_target_type": 'umbrella',
        "expected_landmark": 'guardrail',
        "expected_relation": 'beside',
        "expected_situation": None,
    },

    # ---- 2026-08-24 batch: the 2026-08-21 catalog audit flagged that 5 of the 9 person
    # poses (crouching/reclining/walking/walking_away/reaching) had exactly ONE held-out
    # example each -- "statistically almost no basis" for those numbers. This batch adds
    # 3 more per pose (4 total each now) so the reported accuracy for these poses actually
    # means something. All landmarks are real catalog items (kind-constraint checked:
    # landmark-kind only, never object-kind), no (pose, landmark) pair repeats one already
    # used elsewhere in this file for that same pose, and none are structurally close to
    # parser.py's few-shot examples (checked by hand against all 13).
    {
        "sentence": 'A person is crouched low next to the postbox, tying their shoelace.',
        "expected_target_type": 'person_crouching',
        "expected_landmark": 'postbox',
        "expected_relation": 'near',
        "expected_situation": 'normal',
    },
    {
        "sentence": 'Someone is crouching down beside the taxi, picking something up off the ground.',
        "expected_target_type": 'person_crouching',
        "expected_landmark": 'taxi',
        "expected_relation": 'beside',
        "expected_situation": 'normal',
    },
    {
        "sentence": 'Behind the trash can, someone is crouching low as if hiding from something.',
        "expected_target_type": 'person_crouching',
        "expected_landmark": 'trash_can',
        "expected_relation": 'near',
        "expected_situation": 'abnormal',
    },
    {
        "sentence": 'A person is reclining against the police car, looking relaxed with their eyes closed.',
        "expected_target_type": 'person_reclining',
        "expected_landmark": 'police_car',
        "expected_relation": 'beside',
        "expected_situation": 'normal',
    },
    {
        "sentence": 'Someone is reclining on top of the picnic table, taking in the sun.',
        "expected_target_type": 'person_reclining',
        "expected_landmark": 'picnic_table',
        "expected_relation": 'on',
        "expected_situation": 'normal',
    },
    {
        "sentence": "There's a person reclining right next to the kick scooter, resting on the grass.",
        "expected_target_type": 'person_reclining',
        "expected_landmark": 'kick_scooter',
        "expected_relation": 'near',
        "expected_situation": 'normal',
    },
    {
        "sentence": 'A person is walking near the bicycle parked by the entrance.',
        "expected_target_type": 'person_walking',
        "expected_landmark": 'bicycle',
        "expected_relation": 'near',
        "expected_situation": 'normal',
    },
    {
        "sentence": 'Someone is walking by the postbox, checking their phone.',
        "expected_target_type": 'person_walking',
        "expected_landmark": 'postbox',
        "expected_relation": 'near',
        "expected_situation": 'normal',
    },
    {
        "sentence": "There's a person walking alongside the taxi as it waits at the curb.",
        "expected_target_type": 'person_walking',
        "expected_landmark": 'taxi',
        "expected_relation": 'beside',
        "expected_situation": 'normal',
    },
    {
        "sentence": 'A person is walking away from the vending machine toward the parking lot.',
        "expected_target_type": 'person_walking_away',
        "expected_landmark": 'vending_machine',
        "expected_relation": 'near',
        "expected_situation": 'normal',
    },
    {
        "sentence": 'Someone is walking away from the police car, heading down the sidewalk.',
        "expected_target_type": 'person_walking_away',
        "expected_landmark": 'police_car',
        "expected_relation": 'near',
        "expected_situation": 'normal',
    },
    {
        "sentence": "There's a person walking away from the picnic table, back turned to the camera.",
        "expected_target_type": 'person_walking_away',
        "expected_landmark": 'picnic_table',
        "expected_relation": 'near',
        "expected_situation": 'normal',
    },
    {
        "sentence": 'Someone is reaching into the trash can, digging around for something.',
        "expected_target_type": 'person_reaching',
        "expected_landmark": 'trash_can',
        "expected_relation": 'near',
        "expected_situation": 'normal',
    },
    {
        "sentence": 'A person is reaching toward the vending machine, trying to grab a drink from the return slot.',
        "expected_target_type": 'person_reaching',
        "expected_landmark": 'vending_machine',
        "expected_relation": 'near',
        "expected_situation": 'normal',
    },
    {
        "sentence": "There's someone reaching down beside the bicycle to grab something they dropped.",
        "expected_target_type": 'person_reaching',
        "expected_landmark": 'bicycle',
        "expected_relation": 'beside',
        "expected_situation": 'normal',
    },
]
