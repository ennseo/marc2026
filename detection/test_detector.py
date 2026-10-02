"""Dependency-free routing tests for the YOLO/OWLv2 hybrid detector."""

import unittest

from detection.detector import HybridDetector
from detection.images import normalize_detection_label


class FakeDetector:
    def __init__(self, supported=()):
        self.supported = set(supported)
        self.calls = []

    def supports(self, label):
        return label in self.supported

    def detect(self, image, labels):
        self.calls.append(list(labels))
        return [(label, 0.8 if label in self.supported else 0.4, (0, 0, 1, 1))
                for label in labels]


class HybridDetectorTest(unittest.TestCase):
    def test_known_labels_use_only_yolo(self):
        yolo = FakeDetector({"bench", "mug"})
        made = []
        hybrid = HybridDetector(yolo, owl_factory=lambda: made.append(True))

        got = hybrid.detect(None, ["mug", "bench"])

        self.assertEqual(yolo.calls, [["mug", "bench"]])
        self.assertEqual(made, [])
        self.assertEqual([d[0] for d in got], ["mug", "bench"])

    def test_mixed_request_routes_only_unknown_label_to_owl(self):
        yolo = FakeDetector({"bench"})
        owl = FakeDetector({"master_chef_can"})
        hybrid = HybridDetector(yolo, owl_factory=lambda: owl)

        got = hybrid.detect(None, ["master_chef_can", "bench"])

        self.assertEqual(yolo.calls, [["bench"]])
        self.assertEqual(owl.calls, [["master_chef_can"]])
        self.assertEqual({d[0] for d in got}, {"master_chef_can", "bench"})

    def test_alias_is_normalized_before_routing(self):
        yolo = FakeDetector({"hydrant"})
        hybrid = HybridDetector(yolo, owl_factory=lambda: self.fail("OWL loaded"))

        hybrid.detect(None, ["fire hydrant"])

        self.assertEqual(yolo.calls, [["hydrant"]])
        self.assertEqual(normalize_detection_label("pencil case"), "pencilcase")

    def test_failed_owl_initialization_is_not_retried_per_camera(self):
        yolo = FakeDetector()
        attempts = []

        def fail():
            attempts.append(1)
            raise RuntimeError("offline cache missing")

        hybrid = HybridDetector(yolo, owl_factory=fail)
        self.assertEqual(hybrid.detect(None, ["unknown_a"]), [])
        self.assertEqual(hybrid.detect(None, ["unknown_b"]), [])
        self.assertEqual(len(attempts), 1)


if __name__ == "__main__":
    unittest.main()
