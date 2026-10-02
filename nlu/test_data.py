"""Known example sentences pulled from the starter kit demo data (marc2026_demo scenario).

39 sentences with target_type/landmark/relation/situation.
This is a SAMPLE for testing generalization, not the full vocabulary -- do not hardcode against it.
See nlu/held_out_sentences.py for hand-written sentences NOT in this set (generalization check).
See nlu/paraphrases.py for paraphrased variants of these 33 Stage1 sentences (same labels, different wording).

NOTE (2026-08-15): the starter kit answer table itself says its answers are "GT + a few random wrong axes
mixed in" (so the baseline mock demo does not always score 1.0). We found 3 entries where target_type
was one of these intentionally-wrong values (all corrupted to "cola_can" even though the sentence
says umbrella/tumbler) -- corrected here to match what the sentence actually says. Do not treat
the starter kit answer table target_type/landmark/relation as blindly authoritative; cross-check against the
literal sentence when something looks inconsistent, same as we did for these 3.

NOTE (2026-08-15, relation for person targets): the original file always set relation=null when
target_type is "person", even when a relation is stated ("sitting ON TOP OF the trash can"). We
decided this convention is not worth forcing (parser.py extracts relation for person targets the
same way as for objects now -- see nlu/README.md) so 11 of the 14 person entries here were updated
to the relation actually stated in the text. The remaining 3 (landmark not textually grounded --
hydrant/bicycle guessed from the scene, not the sentence) keep relation=null since no relation can
be read off the text either way.

NOTE (2026-08-20): cross-checked all 39 against real platform scoring data (cctv_dump/round_*/info.json
"expected", captured from two independent live sessions -- see nlu/README.md). Found and fixed real
label errors beyond what static analysis could catch:
  - person target_type is NEVER just "person" in real grading -- it's always pose-qualified
    (person_sitting/person_bending/person_lying_down/person_prone/...). All 13 confirmed person
    entries fixed here (1 more, round 29 above, has no live data in either dump -- inferred by
    analogy, flagged in-line, not independently confirmed).
  - landmark/relation have 0% scoring weight for person-type problems in the real rubric, confirmed
    directly from the weights field. So the relation values kept below for person entries are not
    scored either way -- kept as our best textual read (useful for detection candidate disambiguation
    even though not separately graded), not forced back to null.
  - 3 landmark errors and 2 relation errors on object-type entries were genuine mislabels (not
    visual-only-landmark cases) -- one of them (round 16) had the landmark stated outright in the
    sentence text ("vending machine") and we simply had it wrong. Fixed.
  - 3 situation errors, all should have been "abnormal".
"""

KNOWN_EXAMPLES = [{'landmark': 'picnic_table',
  'relation': 'on',
  'sentence': 'Find the mug on the picnic table.',
  'situation': None,
  'target_type': 'mug'},
 {'landmark': 'bench',
  'relation': 'on',
  'sentence': 'Find the tumbler on the bench.',
  'situation': None,
  'target_type': 'tumbler'},
 {'landmark': 'hydrant',
  'relation': 'near',
  'sentence': 'Find the tumbler near the fire hydrant in the parking lot.',
  'situation': None,
  'target_type': 'tumbler'},
 {'landmark': 'picnic_table',
  'relation': 'beside',
  'sentence': 'Find the sunblock beside the picnic table.',
  'situation': None,
  'target_type': 'sunblock'},
 {'landmark': 'vending_machine',
  'relation': 'beside',
  'sentence': 'Find the umbrella beside the vending machine.',
  'situation': None,
  'target_type': 'umbrella'},
 {'landmark': 'trash_can',
  'relation': 'near',
  'sentence': 'Find the umbrella next to the trash can.',
  'situation': None,
  'target_type': 'umbrella'},
 {'landmark': 'picnic_table',
  'relation': 'near',
  'sentence': 'Find the pencil case near the picnic table.',
  'situation': None,
  'target_type': 'pencilcase'},
 {'landmark': 'police_car',
  'relation': 'near',
  'sentence': 'Find the umbrella dropped in the empty parking space.',
  'situation': None,
  'target_type': 'umbrella'},
 {'landmark': 'police_car',
  'relation': 'near',
  'sentence': 'Find the tumbler dropped in the empty parking space.',
  'situation': None,
  'target_type': 'tumbler'},
 {'landmark': 'taxi',
  'relation': 'near',
  'sentence': 'Find the tumbler on the road next to the taxi.',
  'situation': None,
  'target_type': 'tumbler'},
 {'landmark': 'taxi',
  'relation': 'near',
  'sentence': 'Find the umbrella on the road next to the taxi.',
  'situation': None,
  'target_type': 'umbrella'},
 {'landmark': 'vending_machine',
  'relation': 'beside',
  'sentence': 'After working out I grabbed a drink from the machine and left my umbrella standing '
              'next to it.',
  'situation': None,
  'target_type': 'umbrella'},
 {'landmark': 'picnic_table',
  'relation': 'near',
  'sentence': 'I was taking notes at an outdoor table and left my pencil case behind.',
  'situation': None,
  'target_type': 'pencilcase'},
 {'landmark': 'picnic_table',
  'relation': 'on',
  'sentence': 'I was reading with a coffee at an outdoor table and left my mug there.',
  'situation': None,
  'target_type': 'mug'},
 {'landmark': 'picnic_table',
  'relation': 'beside',
  'sentence': 'I was resting in the sun at the park and left the sunblock I had been using.',
  'situation': None,
  'target_type': 'sunblock'},
 {'landmark': 'vending_machine',
  'relation': 'beside',
  'sentence': 'After grabbing a drink from the vending machine I left my umbrella propped against '
              'its side.',
  'situation': None,
  'target_type': 'umbrella'},
 {'landmark': 'trash_can',
  'relation': 'near',
  'sentence': 'I stopped to throw something in the bin and left my umbrella leaning against the '
              'trash can.',
  'situation': None,
  'target_type': 'umbrella'},
 {'landmark': 'police_car',
  'relation': 'near',
  'sentence': 'I think I dropped my tumbler on the ground as I got out of my car in the parking '
              'lot.',
  'situation': None,
  'target_type': 'tumbler'},
 {'landmark': 'police_car',
  'relation': 'near',
  'sentence': 'Getting out of my car I dropped my umbrella in the empty space next to it; find it '
              'in the parking lot.',
  'situation': None,
  'target_type': 'umbrella'},
 {'landmark': 'trash_can',
  'relation': 'on',
  'sentence': 'Find the person sitting on top of the trash can in the parking lot.',
  'situation': 'abnormal',
  'target_type': 'person_sitting'},
 {'landmark': 'vending_machine',
  'relation': 'on',
  'sentence': 'Find the person sitting on top of the vending machine.',
  'situation': 'abnormal',
  'target_type': 'person_sitting'},
 {'landmark': 'bench',
  'relation': 'on',
  'sentence': 'Find the person lying face-down on the bench near the taxi.',
  'situation': 'abnormal',
  'target_type': 'person_bending'},
 {'landmark': 'picnic_table',
  'relation': 'on',
  'sentence': 'Find the person lying on top of the picnic table.',
  'situation': 'abnormal',
  'target_type': 'person_lying_down'},
 {'landmark': 'hydrant',
  'relation': None,
  'sentence': 'Find the person collapsed on the parking lot floor.',
  'situation': 'emergency',
  'target_type': 'person_lying_down'},
 {'landmark': 'bicycle',
  'relation': None,
  'sentence': 'Find the person lying face-down and motionless in the parking lot.',
  'situation': 'accident',
  'target_type': 'person_prone'},
 {'landmark': 'bicycle',
  'relation': 'near',
  'sentence': 'Find the person fallen on the ground next to the toppled bicycle.',
  'situation': 'emergency',
  'target_type': 'person_bending'},
 {'landmark': 'trash_can',
  'relation': 'near',
  'sentence': 'Find the person bent over and motionless near the trash can in the park rest area.',
  'situation': 'emergency',
  'target_type': 'person_bending'},
 {'landmark': 'trash_can',
  'relation': 'near',
  'sentence': 'People are relaxing at the park and one is even lying on the grass; find the one '
              'near the trash can who is bent over and not moving and actually needs help.',
  'situation': 'emergency',
  'target_type': 'person_bending'},
 {'landmark': 'bicycle',
  'relation': 'near',
  'sentence': 'In the parking lot a person is down next to a bicycle and looks like they had an '
              'accident; find them.',
  'situation': 'accident',
  'target_type': 'person_prone'},
 {'landmark': 'vending_machine',
  'relation': 'on',
  'sentence': 'Someone is up where people should not be; find the person who climbed on top of the '
              'drink machine.',
  'situation': 'abnormal',
  'target_type': 'person_sitting'},
 {'landmark': 'trash_can',
  'relation': 'on',
  'sentence': 'In the parking lot, find the person who climbed up and is sitting on top of the '
              'trash can.',
  'situation': 'abnormal',
  'target_type': 'person_sitting'},
 {'landmark': 'bench',
  'relation': 'on',
  'sentence': 'Near the taxi, find the person lying down on top of the bench.',
  'situation': 'abnormal',
  'target_type': 'person_bending'},
 {'landmark': 'bicycle',
  'relation': None,
  'sentence': 'Several people are in the parking lot; one is completely face-down and immobile; '
              'find the most critical one.',
  'situation': 'accident',
  'target_type': 'person_prone'},
 {'landmark': 'police_car',
  'relation': 'near',
  'sentence': 'Pick up the tumbler dropped on the parking lot floor and bring it to the owner.',
  'situation': None,
  'target_type': 'tumbler'},
 {'landmark': 'police_car',
  'relation': 'near',
  'sentence': 'Pick up the umbrella in the empty parking space and bring it to the owner.',
  'situation': None,
  'target_type': 'umbrella'},
 {'landmark': 'vending_machine',
  'relation': 'beside',
  'sentence': 'Pick up the umbrella beside the vending machine and bring it to the owner.',
  'situation': None,
  'target_type': 'umbrella'},
 {'landmark': 'taxi',
  'relation': 'near',
  'sentence': 'Pick up the tumbler on the road next to the taxi and bring it to the owner.',
  'situation': None,
  'target_type': 'tumbler'},
 {'landmark': 'taxi',
  'relation': 'near',
  'sentence': 'Pick up the umbrella on the road next to the taxi and bring it to the owner.',
  'situation': None,
  'target_type': 'umbrella'},
 {'landmark': 'picnic_table',
  'relation': 'near',
  'sentence': 'Pick up the pencil case near the picnic table and bring it to the owner.',
  'situation': None,
  'target_type': 'pencilcase'}]
