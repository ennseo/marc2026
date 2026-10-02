"""Detectors and pose refinement for visual grounding.

YOLODetector handles labels represented in the fine-tuned checkpoint.
HybridDetector routes other labels to a lazily loaded OwlV2Detector; it does
not retry known YOLO labels after a miss. PersonPoseRefiner supplies optional
keypoint-based reference pixels after camera and person-instance selection.

Object detectors expose:
    detect(image, labels) -> [(label, confidence, (x0, y0, x1, y1))]

Images are HxWx3 uint8 RGB arrays. Labels use canonical catalogue names;
OWLv2 converts them to text prompts using detection.images.DETECT_PROMPT.
"""

import logging
import os

import numpy as np

from .images import DETECT_FLOOR, DETECT_PROMPT, normalize_detection_label

log = logging.getLogger("participant_app")

# Bundled runtime checkpoints. Public-copy hashes are in weights/manifest.json.
DEFAULT_YOLO_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "weights", "chungmu_yolo11s_1280_best.pt")
DEFAULT_ROI_YOLO_WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "weights", "chungmu_roi_10cls_yolo11s_1280_best.pt")
DEFAULT_PERSON_POSE_WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "weights", "yolo11n-pose.pt")

# The ROI checkpoint retains all 34 class heads. Only these ten target classes
# receive the crop-trained second look.
ROI_TARGET_CLASSES = frozenset({
    "mug", "tumbler", "umbrella", "sunblock", "pencilcase",
    "juice", "cola_can", "cracker_box", "tissue", "disposable_cup",
})


class PersonPoseRefiner:
    """Find a safer ground-reference pixel for selected lying/bending people.

    Pose runs only after grounding has selected a camera and person instance. It
    therefore cannot change camera ranking, and any missing/ambiguous keypoint falls
    back to the established bbox bottom-centre coordinate.
    """

    _LEFT_HIP, _RIGHT_HIP = 11, 12
    _LEFT_ANKLE, _RIGHT_ANKLE = 15, 16
    SUPPORTED_POSES = frozenset({"person_lying_down", "person_bending"})

    def __init__(self, weights=DEFAULT_PERSON_POSE_WEIGHTS, device=None, imgsz=640,
                 keypoint_conf=0.25, min_bbox_iou=0.5, crop_scale=2.0):
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"pose weights not found: {weights}")
        self.weights = weights
        self.device = device
        self.imgsz = int(imgsz)
        self.keypoint_conf = float(keypoint_conf)
        self.min_bbox_iou = float(min_bbox_iou)
        self.crop_scale = float(crop_scale)
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
            log.info("[detect] person pose refiner ready (%s)", self.weights)
        return self._model

    @staticmethod
    def _iou(a, b):
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        union = ((a[2] - a[0]) * (a[3] - a[1]) +
                 (b[2] - b[0]) * (b[3] - b[1]) - inter)
        return inter / union if union > 0 else 0.0

    def reference_pixel(self, image, bbox, pose):
        if pose not in self.SUPPORTED_POSES or image is None or bbox is None:
            return None
        height, width = image.shape[:2]
        x0, y0, x1, y1 = map(float, bbox)
        if x1 <= x0 or y1 <= y0:
            return None
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        crop_w = (x1 - x0) * self.crop_scale
        crop_h = (y1 - y0) * self.crop_scale
        left = max(0, int(round(cx - crop_w / 2.0)))
        top = max(0, int(round(cy - crop_h / 2.0)))
        right = min(width, int(round(cx + crop_w / 2.0)))
        bottom = min(height, int(round(cy + crop_h / 2.0)))
        if right - left < 16 or bottom - top < 16:
            return None

        crop = image[top:bottom, left:right]
        result = self._load().predict(
            source=np.ascontiguousarray(crop[:, :, ::-1]),
            conf=0.05, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if result.boxes is None or result.keypoints is None:
            return None

        best = None
        for index, local_box in enumerate(result.boxes.xyxy.cpu().numpy()):
            pose_box = (float(local_box[0] + left), float(local_box[1] + top),
                        float(local_box[2] + left), float(local_box[3] + top))
            overlap = self._iou(pose_box, bbox)
            if best is None or overlap > best[0]:
                best = (overlap, index)
        if best is None or best[0] < self.min_bbox_iou:
            return None

        overlap, index = best
        xy = result.keypoints.xy[index].cpu().numpy().astype(float)
        conf = result.keypoints.conf[index].cpu().numpy().astype(float)
        xy[:, 0] += left
        xy[:, 1] += top
        ankle_ids = (self._LEFT_ANKLE, self._RIGHT_ANKLE)
        if any(conf[idx] < self.keypoint_conf for idx in ankle_ids):
            return None
        ankle = np.mean(xy[list(ankle_ids)], axis=0)
        pixel = ankle
        if pose == "person_lying_down":
            hip_ids = (self._LEFT_HIP, self._RIGHT_HIP)
            if any(conf[idx] < self.keypoint_conf for idx in hip_ids):
                return None
            hip = np.mean(xy[list(hip_ids)], axis=0)
            axis = ankle - hip
            if float(np.linalg.norm(axis)) < 2.0:
                return None
            pixel = ankle + 0.5 * axis
        if not (0.0 <= pixel[0] < width and 0.0 <= pixel[1] < height):
            return None
        return {
            "pixel": (float(pixel[0]), float(pixel[1])),
            "bbox_iou": float(overlap),
            "ankle_conf": [float(conf[idx]) for idx in ankle_ids],
            "policy": "foot_out_50" if pose == "person_lying_down" else "ankle",
        }


class YOLODetector:
    """Ultralytics YOLO adapter for the common ``detect(image, labels)`` interface.

    The trained model has pose-specific person classes while Stage 1 asks for the
    canonical target type ``person``.  When ``person`` is requested, all
    ``person_*`` predictions are therefore returned under that canonical label.
    """

    def __init__(self, weights, threshold=0.25, device=None, imgsz=1280):
        from ultralytics import YOLO

        if not os.path.isfile(weights):
            raise FileNotFoundError(f"YOLO weights not found: {weights}")
        self.model = YOLO(weights)
        self.threshold = threshold
        self.device = device
        self.imgsz = imgsz
        names = self.model.names
        self.names = dict(names) if isinstance(names, dict) else dict(enumerate(names))
        log.info("[detect] YOLO ready (%s, threshold %.2f, imgsz %d)",
                 weights, self.threshold, self.imgsz)

    def supports(self, label):
        """Whether this checkpoint can serve ``label`` without open-vocabulary help."""
        label = normalize_detection_label(label)
        if label == "person":
            return any(name.startswith("person_") for name in self.names.values())
        return label in self.names.values()

    def detect(self, image, labels):
        wanted = {normalize_detection_label(label) for label in labels if label}
        # Avoid running class heads that this grounding request cannot use.  A requested
        # `person` expands to every pose-specific class in the trained dataset.
        class_ids = []
        output_label = {}
        for idx, name in self.names.items():
            if name in wanted:
                class_ids.append(idx)
                output_label[idx] = name
            elif "person" in wanted and name.startswith("person_"):
                class_ids.append(idx)
                output_label[idx] = "person"
        if not class_ids:
            return []

        # Ultralytics expects BGR ndarray input, while this pipeline decodes RGB.
        # Convert explicitly: swapped channels can silently reduce detection confidence.
        result = self.model.predict(
            source=np.ascontiguousarray(image[:, :, ::-1]),
            conf=self.threshold,
            imgsz=self.imgsz,
            device=self.device,
            classes=class_ids,
            verbose=False,
        )[0]
        out = []
        if result.boxes is not None:
            for box, score, cls in zip(result.boxes.xyxy, result.boxes.conf,
                                       result.boxes.cls):
                idx = int(cls.item())
                label = output_label.get(idx)
                if label is None:
                    continue
                out.append((label, float(score.item()),
                            tuple(float(v) for v in box.tolist())))
        out.sort(key=lambda d: -d[1])
        return out


class HybridDetector:
    """YOLO fast path plus lazy OWLv2 routing for genuinely unseen labels.

    This deliberately does *not* retry a known YOLO class with OWLv2 after a miss. On
    CPU that can exceed the 30-second Stage 1 budget across six cameras, and a miss is
    not evidence that OWLv2 will do better. The open-vocabulary path is used only when
    the checkpoint has no head for a requested canonical label.
    """

    def __init__(self, yolo, owl_factory=None):
        self.yolo = yolo
        self._owl_factory = owl_factory or OwlV2Detector
        self._owl = None
        self._owl_failed = False

    def supports(self, label):
        # The hybrid can attempt any non-empty text label through OWLv2.
        return bool(normalize_detection_label(label))

    @staticmethod
    def _unique(labels):
        return list(dict.fromkeys(normalize_detection_label(label)
                                  for label in labels if label))

    def _open_vocab(self):
        if self._owl is None and not self._owl_failed:
            try:
                self._owl = self._owl_factory()
            except Exception as exc:  # missing cached weights/dependencies, OOM, ...
                self._owl_failed = True
                log.warning("[detect] OWLv2 unavailable for unseen classes (%s)", exc)
        return self._owl

    def detect(self, image, labels):
        labels = self._unique(labels)
        known = [label for label in labels if self.yolo.supports(label)]
        unknown = [label for label in labels if not self.yolo.supports(label)]

        out = self.yolo.detect(image, known) if known else []
        if unknown:
            owl = self._open_vocab()
            if owl is not None:
                out.extend(owl.detect(image, unknown))
        out.sort(key=lambda detection: -detection[1])
        return out


class OwlV2Detector:
    """Open-vocabulary detector. No training required.

    Weights are baked into the image at build time; the judging runtime has no internet
    (NOTICES.md). Runs on GPU when one is present and falls back to CPU otherwise --
    CPU works but takes seconds per image, which is fine for offline checks on a laptop
    and not fine for a scored round.
    """

    def __init__(self, model_id="google/owlv2-base-patch16-ensemble", threshold=0.05):
        import torch
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        self._torch = torch
        # A low global threshold retains weak small-object candidates.
        # Grounding filters them by spatial context; DETECT_FLOOR adds stricter
        # thresholds for prompts that also match common distractors.
        self.threshold = threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device).eval()
        log.info("[detect] OWLv2 ready on %s (threshold %.2f)", self.device, self.threshold)

    def detect(self, image, labels):
        torch = self._torch
        labels = list(labels)
        prompts = [DETECT_PROMPT.get(l, f"a {l.replace('_', ' ')}") for l in labels]
        inputs = self.processor(text=[prompts], images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        sizes = torch.tensor([image.shape[:2]], device=self.device)
        res = self.processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=sizes, threshold=self.threshold)[0]
        out = [(labels[int(i)], float(s), tuple(float(v) for v in b))
               for s, b, i in zip(res["scores"], res["boxes"], res["labels"])]
        # Shape-description prompts can match distractors; apply their class floors
        # in addition to the global threshold used during post-processing.
        out = [d for d in out if d[1] >= DETECT_FLOOR.get(d[0], self.threshold)]
        out.sort(key=lambda d: -d[1])
        return out


class StubDetector:
    """Returns nothing. Lets the pipeline run end-to-end without torch installed.

    Useful for wiring/plumbing tests on a machine that cannot host the real model.
    """

    def detect(self, image, labels):
        return []
