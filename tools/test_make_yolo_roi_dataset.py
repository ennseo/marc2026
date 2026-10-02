import json
import os
import tempfile
import unittest

from PIL import Image

from tools.make_yolo_dataset import is_val
from tools.make_yolo_roi_dataset import (
    CropSpec,
    choose_positive_crop,
    clipped_box,
    main,
    sample_negatives,
    sample_positives,
    square_crop,
)


def _spec(scene_key, ordinal, positive, target_classes=()):
    return CropSpec(
        json_path=f"source_{ordinal}.json",
        image_path=f"source_{ordinal}.png",
        scene_key=scene_key,
        source_stem=f"cam_{ordinal}",
        anchor_class="bench",
        anchor_instance=f"bench_{ordinal}",
        bounds=(0, 0, 64, 64),
        requested_positive=positive,
        requested_target_classes=target_classes,
    )


class RoiGeometryTest(unittest.TestCase):
    def test_square_crop_shifts_at_frame_edges_without_shrinking(self):
        self.assertEqual(square_crop(10, 10, 40, 100, 80), (0, 0, 40, 40))
        self.assertEqual(square_crop(95, 75, 40, 100, 80), (60, 40, 100, 80))

    def test_clipped_box_requires_requested_retained_area(self):
        crop = (20, 20, 80, 80)
        self.assertEqual(clipped_box((10, 30, 30, 50), crop, 0.5),
                         (0.0, 10.0, 10.0, 30.0))
        self.assertIsNone(clipped_box((10, 30, 30, 50), crop, 0.51))

    def test_positive_crop_is_landmark_centred_and_uses_smallest_fitting_size(self):
        target = {"class": "tumbler", "bbox": [490, 200, 510, 220]}
        landmark = {
            "class": "bench", "instance_id": "bench_0", "kind": "landmark",
            "bbox": [180, 190, 220, 230],
        }
        selected, bounds = choose_positive_crop(
            target, [landmark], (384, 512, 768), 1280, 720, 0.5)

        self.assertIs(selected, landmark)
        self.assertEqual(bounds, (0, 0, 512, 512))
        self.assertIsNotNone(clipped_box(target["bbox"], bounds, 0.5))

    def test_negative_sampling_is_deterministic_and_split_local(self):
        train_key = next(f"scenario/scene_{i:05d}" for i in range(1000)
                         if not is_val(f"scenario/scene_{i:05d}"))
        val_key = next(f"scenario/scene_{i:05d}" for i in range(1000)
                       if is_val(f"scenario/scene_{i:05d}"))
        positives = [_spec(train_key, 0, True), _spec(val_key, 1, True)]
        negatives = ([_spec(train_key, i, False) for i in range(2, 7)] +
                     [_spec(val_key, i, False) for i in range(7, 12)])

        first = sample_negatives(positives, negatives, 1.0)
        second = sample_negatives(positives, list(reversed(negatives)), 1.0)

        self.assertEqual([item.stable_key for item in first],
                         [item.stable_key for item in second])
        self.assertEqual(len(first), 2)
        self.assertEqual(sum(not is_val(item.scene_key) for item in first), 1)
        self.assertEqual(sum(is_val(item.scene_key) for item in first), 1)

    def test_positive_sampling_caps_each_class_per_split_deterministically(self):
        train_key = next(f"scenario/scene_{i:05d}" for i in range(1000)
                         if not is_val(f"scenario/scene_{i:05d}"))
        val_key = next(f"scenario/scene_{i:05d}" for i in range(1000)
                       if is_val(f"scenario/scene_{i:05d}"))
        positives = []
        for ordinal in range(10):
            positives.append(_spec(train_key, ordinal, True, ("tumbler",)))
            positives.append(_spec(val_key, ordinal + 20, True, ("tumbler",)))

        first = sample_positives(positives, 5)
        second = sample_positives(list(reversed(positives)), 5)

        self.assertEqual([item.stable_key for item in first],
                         [item.stable_key for item in second])
        self.assertEqual(sum(not is_val(item.scene_key) for item in first), 4)
        self.assertEqual(sum(is_val(item.scene_key) for item in first), 1)


class RoiDatasetEndToEndTest(unittest.TestCase):
    def test_builds_crop_labels_yaml_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            src = os.path.join(temp, "trainer_output")
            scene = os.path.join(src, "scenario", "scene_00000")
            dst = os.path.join(temp, "roi")
            os.makedirs(scene)
            Image.new("RGB", (128, 96), (30, 40, 50)).save(
                os.path.join(scene, "rig_a.png"))
            payload = {
                "image": {"width": 128, "height": 96},
                "detections": [
                    {
                        "class": "bench", "instance_id": "bench_0",
                        "kind": "landmark", "bbox": [20, 25, 60, 65],
                        "occlusion": 0.0,
                    },
                    {
                        "class": "tumbler", "instance_id": "tumbler_0",
                        "kind": "object", "bbox": [66, 42, 72, 54],
                        "occlusion": 0.1,
                    },
                    {
                        "class": "umbrella", "instance_id": "umbrella_hidden",
                        "kind": "object", "bbox": [80, 40, 90, 50],
                        "occlusion": 0.95,
                    },
                ],
            }
            with open(os.path.join(scene, "rig_a.json"), "w", encoding="utf-8") as handle:
                json.dump(payload, handle)

            result = main([
                src, dst,
                "--crop-sizes", "64,96",
                "--negative-ratio", "0",
                "--target-classes", "tumbler,umbrella,pencilcase",
            ])

            self.assertEqual(result, 0)
            self.assertTrue(os.path.isfile(os.path.join(dst, "data.yaml")))
            with open(os.path.join(dst, "roi_manifest.json"), encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(len(manifest), 1)
            self.assertTrue(manifest[0]["positive"])
            self.assertEqual(manifest[0]["target_classes"], ["tumbler"])
            label_path = manifest[0]["file"].replace("images/", "labels/")
            label_path = os.path.splitext(label_path)[0] + ".txt"
            with open(os.path.join(dst, *label_path.split("/")), encoding="utf-8") as handle:
                lines = [line for line in handle.read().splitlines() if line]
            class_ids = {int(line.split()[0]) for line in lines}
            self.assertIn(0, class_ids)   # bench
            self.assertIn(16, class_ids)  # tumbler
            self.assertNotIn(17, class_ids)  # highly occluded umbrella

    def test_refuses_nonempty_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            src = os.path.join(temp, "trainer_output")
            dst = os.path.join(temp, "roi")
            os.makedirs(src)
            os.makedirs(dst)
            with open(os.path.join(dst, "stale.txt"), "w", encoding="utf-8") as handle:
                handle.write("old run")

            with self.assertRaises(SystemExit):
                main([src, dst])


if __name__ == "__main__":
    unittest.main()
