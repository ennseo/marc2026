import unittest
from types import SimpleNamespace

import numpy as np

from detection.grounding import VisualGrounding


def _det(label, confidence, x):
    return (label, confidence, (x - 1.0, 2.0, x + 1.0, 8.0))


class _Client:
    def __init__(self, camera_ids):
        self.camera_ids = camera_ids

    def list_cctv(self):
        return list(self.camera_ids)

    def get_cctv_image(self, camera_id):
        value = self.camera_ids.index(camera_id) + 1
        image = np.full((16, 32, 3), value, dtype=np.uint8)
        return SimpleNamespace(data=image.tobytes(), height=16, width=32,
                               encoding="rgb8")


class _Detector:
    def __init__(self, camera_ids, detections):
        self.camera_ids = camera_ids
        self.detections = detections

    def detect(self, image, labels):
        camera_id = self.camera_ids[int(image[0, 0, 0]) - 1]
        return [d for d in self.detections.get(camera_id, []) if d[0] in labels]


class _CropDetector:
    def __init__(self, detections):
        self.detections = detections
        self.calls = 0

    def detect(self, image, labels):
        self.calls += 1
        return [d for d in self.detections if d[0] in labels]


class _SequenceDetector:
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.calls = 0

    def detect(self, image, labels):
        index = min(self.calls, len(self.sequence) - 1)
        self.calls += 1
        return [d for d in self.sequence[index] if d[0] in labels]


class _Geometry:
    def wait_ready(self):
        return True

    def unproject(self, camera_id, u, v, on_landmark=None, as_landmark=None):
        return [float(u), 0.0, 0.0]


class _PoseRefiner:
    def __init__(self, pixel):
        self.pixel = pixel
        self.calls = []

    def reference_pixel(self, image, bbox, pose):
        self.calls.append((tuple(bbox), pose))
        return {"pixel": self.pixel, "policy": "test", "bbox_iou": 1.0,
                "ankle_conf": [1.0, 1.0]}


def _ground(camera_ids, detections, parsed):
    grounding = VisualGrounding(
        _Client(camera_ids),
        detector=_Detector(camera_ids, detections),
        geom=_Geometry(),
    )
    return grounding.ground(parsed)


class LandmarkFallbackTest(unittest.TestCase):
    def test_missing_target_retries_latest_frame_and_replaces_fallback(self):
        detector = _SequenceDetector([
            [_det("picnic_table", 0.9, 10.0)],
            [_det("mug", 0.8, 10.2), _det("picnic_table", 0.9, 10.0)],
        ])
        grounding = VisualGrounding(
            _Client(["cam_a"]), detector=detector, geom=_Geometry(),
            missing_target_retries=1,
        )

        result = grounding.ground({
            "target_type": "mug", "landmark": "picnic_table", "relation": "on",
        })

        self.assertEqual(detector.calls, 2)
        self.assertTrue(result["target_detected"])
        self.assertAlmostEqual(result["target_coord"][0], 10.2)

    def test_semantic_area_landmark_searches_for_concrete_visual_anchor(self):
        result = _ground(
            ["cam_a"],
            {"cam_a": [
                _det("sunblock", 0.8, 10.5),
                _det("picnic_table", 0.9, 10.0),
            ]},
            {"target_type": "sunblock", "landmark": "park", "relation": "on",
             "region": "park"},
        )

        self.assertEqual(result["camera_id"], "cam_a")
        self.assertEqual(result["landmark"], "picnic_table")
        self.assertAlmostEqual(result["target_coord"][0], 10.5)
        self.assertAlmostEqual(result["anchor_coord"][0], 10.0)

    def test_person_on_support_uses_anchor_xy_instead_of_bbox_or_pose(self):
        pose = _PoseRefiner((20.0, 8.0))
        grounding = VisualGrounding(
            _Client(["cam_a"]),
            detector=_Detector(
                ["cam_a"], {"cam_a": [
                    _det("person_lying_down", 0.8, 10.4),
                    _det("picnic_table", 0.9, 10.0),
                ]}),
            geom=_Geometry(), pose_refiner=pose,
        )

        result = grounding.ground({
            "target_type": "person",
            "detection_target_type": "person_lying_down",
            "landmark": "picnic_table",
            "relation": "on",
            "situation": "abnormal",
        })

        self.assertEqual(result["target_coord"], [10.0, 0.0, 0.0])
        self.assertEqual(result["pose_refinement"]["policy"], "support_anchor_xy")
        self.assertEqual(pose.calls, [])

    def test_prone_person_uses_directional_bbox_reference(self):
        grounding = VisualGrounding(
            _Client(["cam_a"]),
            detector=_Detector(
                ["cam_a"], {"cam_a": [_det("person_prone", 0.9, 4.0)]}),
            geom=_Geometry(),
        )

        result = grounding.ground({
            "target_type": "person",
            "detection_target_type": "person_prone",
            "landmark": None,
            "relation": None,
            "situation": "accident",
        })

        self.assertEqual(result["target_coord"], [3.5, 0.0, 0.0])
        self.assertEqual(result["pose_refinement"]["policy"], "prone_bbox_left_75")

    def test_bending_near_trash_uses_ankle_with_small_world_offset(self):
        pose = _PoseRefiner((7.0, 8.0))
        grounding = VisualGrounding(
            _Client(["cam_a"]),
            detector=_Detector(
                ["cam_a"], {"cam_a": [
                    _det("person_bending", 0.9, 4.0),
                    _det("trash_can", 0.8, 3.0),
                ]}),
            geom=_Geometry(), pose_refiner=pose,
        )

        result = grounding.ground({
            "target_type": "person",
            "detection_target_type": "person_bending",
            "landmark": "trash_can",
            "relation": "near",
            "situation": "emergency",
        })

        self.assertEqual(result["target_coord"], [7.384, -0.167, 0.172])
        self.assertEqual(result["pose_refinement"]["policy"],
                         "ankle_trash_world_offset")
        self.assertEqual(pose.calls[-1][1], "person_bending")

    def test_pose_refinement_runs_after_person_instance_selection(self):
        pose = _PoseRefiner((7.0, 8.0))
        grounding = VisualGrounding(
            _Client(["cam_a"]),
            detector=_Detector(
                ["cam_a"], {"cam_a": [_det("person_lying_down", 0.9, 4.0)]}),
            geom=_Geometry(),
            pose_refiner=pose,
        )

        result = grounding.ground({
            "target_type": "person",
            "detection_target_type": "person_lying_down",
            "landmark": None,
            "relation": None,
        })

        self.assertEqual(result["camera_id"], "cam_a")
        self.assertEqual(result["target_coord"], [7.0, 0.0, 0.0])
        self.assertEqual(result["pose_refinement"]["policy"], "test")
        self.assertEqual(pose.calls, [((3.0, 2.0, 5.0, 8.0), "person_lying_down")])

    def test_roi_detector_remaps_crop_box_and_only_runs_for_enabled_class(self):
        roi = _CropDetector([("tumbler", 0.8, (10.0, 20.0, 20.0, 30.0))])
        grounding = VisualGrounding(
            _Client(["cam_a"]),
            detector=_Detector(["cam_a"], {}),
            geom=_Geometry(),
            roi_detector=roi,
            roi_classes={"tumbler"},
            roi_crop_sizes=(40,),
        )
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        landmark = [("bench", 0.9, (50.0, 40.0, 70.0, 60.0))]

        detections = grounding._crop_targets(image, "tumbler", landmark)

        self.assertEqual(detections, [
            ("tumbler", 0.8, (50.0, 50.0, 60.0, 60.0)),
        ])
        self.assertEqual(roi.calls, 1)
        self.assertEqual(grounding._crop_targets(image, "person", landmark), [])
        self.assertEqual(roi.calls, 1)

    def test_person_on_2d_prefers_supported_overlap(self):
        supported = VisualGrounding._person_on_2d_score(
            (8.0, 2.0, 12.0, 9.0), (7.0, 7.0, 13.0, 12.0))
        separate = VisualGrounding._person_on_2d_score(
            (1.0, 2.0, 4.0, 9.0), (7.0, 7.0, 13.0, 12.0))

        self.assertGreater(supported, 1.0)
        self.assertLess(separate, 0.0)

    def test_missing_target_preserves_named_landmark(self):
        result = _ground(
            ["cam_a"],
            {"cam_a": [_det("bench", 0.9, 10.0)]},
            {"target_type": "tumbler", "landmark": "bench", "relation": "on"},
        )

        self.assertEqual(result["camera_id"], "cam_a")
        self.assertEqual(result["anchor_coord"], [10.0, 0.0, 0.0])
        self.assertEqual(result["target_coord"], result["anchor_coord"])
        self.assertFalse(result["target_detected"])
        self.assertEqual(result["fallback_reason"], "no_relation_compatible_target")

    def test_target_without_named_landmark_does_not_beat_landmark_fallback(self):
        result = _ground(
            ["cam_target_only", "cam_landmark"],
            {
                "cam_target_only": [_det("umbrella", 0.99, 4.0)],
                "cam_landmark": [_det("police_car", 0.8, 20.0)],
            },
            {"target_type": "umbrella", "landmark": "police_car", "relation": "near"},
        )

        self.assertEqual(result["camera_id"], "cam_landmark")
        self.assertFalse(result["target_detected"])

    def test_on_relation_rejects_distant_wrong_instance(self):
        result = _ground(
            ["cam_a"],
            {"cam_a": [_det("tumbler", 0.99, 5.0), _det("bench", 0.9, 0.0)]},
            {"target_type": "tumbler", "landmark": "bench", "relation": "on"},
        )

        self.assertFalse(result["target_detected"])
        self.assertEqual(result["target_coord"], [0.0, 0.0, 0.0])

    def test_relation_compatible_target_beats_fallback(self):
        result = _ground(
            ["cam_a"],
            {"cam_a": [_det("tumbler", 0.8, 0.2), _det("bench", 0.9, 0.0)]},
            {"target_type": "tumbler", "landmark": "bench", "relation": "on"},
        )

        self.assertTrue(result["target_detected"])
        self.assertAlmostEqual(result["target_coord"][0], 0.2)

    def test_pose_specific_detection_label_is_kept_inside_vision(self):
        result = _ground(
            ["cam_wrong_pose", "cam_prone"],
            {
                # YOLO canonicalises every non-requested person pose to "person" while
                # preserving the explicitly requested pose label.
                "cam_wrong_pose": [_det("person", 0.99, 3.0)],
                "cam_prone": [_det("person_prone", 0.7, 4.0)],
            },
            {"target_type": "person", "detection_target_type": "person_prone",
             "landmark": None, "relation": None, "situation": "accident"},
        )

        self.assertEqual(result["camera_id"], "cam_prone")

    def test_detection_relation_is_used_without_changing_submission_relation(self):
        result = _ground(
            ["cam_near", "cam_on"],
            {
                "cam_near": [
                    _det("person_sitting", 0.99, 4.0),
                    _det("trash_can", 0.9, 0.0),
                ],
                "cam_on": [
                    _det("person_sitting", 0.7, 10.2),
                    _det("trash_can", 0.8, 10.0),
                ],
            },
            {"target_type": "person", "detection_target_type": "person_sitting",
             "landmark": "trash_can", "relation": None, "detection_relation": "on",
             "situation": "abnormal"},
        )

        self.assertEqual(result["camera_id"], "cam_on")

    def test_person_near_does_not_reward_an_ambiguous_exact_distance_band(self):
        result = _ground(
            ["cam_requested_pose", "cam_exact_near"],
            {
                # 1.5 m is classified as beside, but overlaps the measured near range.
                "cam_requested_pose": [
                    _det("person_bending", 0.95, 1.5),
                    _det("trash_can", 0.80, 0.0),
                ],
                # The old +1 exact-near reward selected this lower-confidence person.
                "cam_exact_near": [
                    _det("person_bending", 0.93, 14.0),
                    _det("trash_can", 0.99, 10.0),
                ],
            },
            {"target_type": "person", "detection_target_type": "person_bending",
             "landmark": "trash_can", "relation": None,
             "detection_relation": "near", "situation": "emergency"},
        )

        self.assertEqual(result["camera_id"], "cam_requested_pose")

    def test_person_camera_uses_secondary_context_landmark(self):
        result = _ground(
            ["cam_high_confidence", "cam_with_taxi"],
            {
                "cam_high_confidence": [
                    _det("person_lying_down", 0.99, 2.0),
                    _det("bench", 0.9, 0.0),
                ],
                "cam_with_taxi": [
                    _det("person_lying_down", 0.75, 12.0),
                    _det("bench", 0.8, 10.0),
                    _det("taxi", 0.9, 14.0),
                ],
            },
            {"target_type": "person", "detection_target_type": "person_lying_down",
             "landmark": "bench", "relation": None, "situation": "abnormal",
             "context_landmarks": ["bench", "taxi"]},
        )

        self.assertEqual(result["camera_id"], "cam_with_taxi")

    def test_person_camera_prefers_closer_landmark_pair_continuously(self):
        result = _ground(
            ["cam_far", "cam_close"],
            {
                "cam_far": [
                    _det("person_bending", 0.99, 7.0),
                    _det("bicycle", 0.9, 0.0),
                ],
                "cam_close": [
                    _det("person_bending", 0.80, 11.0),
                    _det("bicycle", 0.8, 10.0),
                ],
            },
            {"target_type": "person", "detection_target_type": "person_bending",
             "landmark": "bicycle", "relation": "near", "situation": "emergency"},
        )

        self.assertEqual(result["camera_id"], "cam_close")

    def test_person_landmark_only_camera_beats_unpaired_target_without_coordinate(self):
        result = _ground(
            ["cam_unpaired_person", "cam_named_landmark"],
            {
                "cam_unpaired_person": [
                    _det("person_sitting", 0.99, 4.0),
                ],
                "cam_named_landmark": [
                    _det("vending_machine", 0.80, 12.0),
                ],
            },
            {"target_type": "person", "detection_target_type": "person_sitting",
             "landmark": "vending_machine", "relation": "on",
             "situation": "abnormal"},
        )

        self.assertEqual(result["camera_id"], "cam_named_landmark")
        self.assertIsNone(result["target_coord"])
        self.assertEqual(result["anchor_coord"], [12.0, 0.0, 0.0])
        self.assertFalse(result["target_detected"])
        self.assertEqual(result["fallback_reason"],
                         "person_missing_named_landmark_visible")


if __name__ == "__main__":
    unittest.main()
