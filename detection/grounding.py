"""Select a camera and target instance from NLU intent and CCTV images.

Candidates are ranked using detector confidence, person pose, landmark context,
and spatial relationships. Geometry is evaluated during candidate selection:
perspective makes image overlap alone insufficient for many object relations.
When no concrete landmark is named, detected landmark types supply candidates.

Optional ROI detection recovers small objects. Person-coordinate refinement is
applied after selection; explicit fallback fields distinguish missing targets
from successful detections.
"""

import logging
import math
import os

# geometry.transform is pure numpy, so importing this module needs no ROS. CameraGeometry
# (which does need rclpy/tf2) is imported lazily in __init__ instead -- that keeps the
# selection logic below testable on a laptop with a fake geometry object.
from geometry.transform import ground_pixel, relation_from_geometry

from .images import image_bytes_to_numpy, normalize_detection_label

log = logging.getLogger("participant_app")

# Landmark classes observed in the training generator catalogue. Search these
# when NLU does not name a concrete landmark; update for new scene assets.
# Submission labels use class names without instance suffixes (bench_01 -> bench).
LANDMARK_TYPES = [
    # Street furniture and common reference objects
    "picnic_table", "bench", "hydrant", "vending_machine",
    "trash_can", "bicycle", "taxi", "police_car",
    # Vehicle classes and additional street furniture
    "postbox", "kick_scooter",
    "normal_car_1", "normal_car_2", "sports_car", "sports_car_2", "suv",
]

# Heuristic weights for camera, target-instance, and landmark-instance selection.
# These are fixed ranking terms, not calibrated probabilities or learned weights.
W_LANDMARK_VISIBLE = 1.0    # a landmark of the right type appears in the same camera
W_RELATION_MATCH = 1.0      # geometry agrees with the relation the NLU read from the text
W_POSE_MATCH = 1.0          # person detector agrees with the NLU's pose wording
W_PERSON_ON_IOU = 5.0       # rank overlapping person/support pairs within the same relation
W_PERSON_PROXIMITY = 1.0    # continuous person-landmark proximity; no brittle relation band
W_CONTEXT_VISIBLE = 1.0     # a secondary landmark named in the sentence is visible
W_REGION_VISIBLE = 0.20     # each region-characteristic landmark visible in the camera
W_REGION_VISIBLE_MAX = 0.60 # region is only a tie-breaker, never stronger than pose/context
W_TARGET_COORD_MISSING = 1.0  # detected pixels that cannot produce a coordinate are weak evidence
MAX_ANCHOR_TARGET_DIST = 8.0  # metres; prefer nearby target/anchor pairs
PRONE_BBOX_REFERENCE = (0.25, 0.75)  # measured direction-aware proxy for a prone ground origin
# Empirical world-space correction for bending people paired with trash_can.
# The condition is class-based, not a scene-ID check; its calibration may not
# generalize to other layouts. Other bending/landmark combinations are unchanged.
BENDING_TRASH_WORLD_OFFSET = (0.384, -0.167, 0.172)

# Coarse region words are not submission fields, but they are useful camera cues when the
# same person pose appears in several feeds.  Keep this semantic rather than mapping a
# region directly to a rig id: camera ids/layouts may change, while visible street/park
# furniture remains meaningful.  Only person grounding uses this bonus.
REGION_LANDMARK_TYPES = {
    "parking_lot": {
        "taxi", "police_car", "normal_car_1", "normal_car_2",
        "sports_car", "sports_car_2", "suv", "bicycle",
    },
    "road": {"taxi", "police_car", "normal_car_1", "normal_car_2", "hydrant"},
    "park": {"picnic_table", "bench", "trash_can", "bicycle"},
}

# These words describe an area/surface, not a detector class with a stable bbox.  Treating
# them as a named landmark suppresses the real landmark search and ultimately submits
# [0, 0, 0].  Keep the original NLU fields for the response boundary, but let vision infer
# a concrete nearby landmark and relation instead.
NON_VISUAL_LANDMARK_TYPES = {
    "park", "parking_lot", "parking_lot_floor", "road", "street",
    "ground", "floor", "grass",
}

# Optional crop search reuses the full-frame detector and is disabled by default.
# Enlarging crops changes object scale relative to full-frame training. The separate
# crop-trained ROI detector below handles the production second-look path.
CROP_EXPAND = 3.0        # window side = this many times the landmark's longer bbox side
CROP_MIN_PX = 320
CROP_MAX_LANDMARKS = 3   # bounds the extra inference per camera
W_RELATION_CONTRADICT = 1.0  # penalty for relations separated by more than one band

# Near/beside distance bands overlap in the practice data. Penalize only
# nonadjacent relation bands rather than treating every mismatch as contradictory.
_RELATION_ORDER = {"on": 0, "beside": 1, "near": 2}


class VisualGrounding:
    """Chooses camera + instances, and returns world coordinates for the answer."""

    def __init__(self, client, detector=None, geom=None, crop_search=False,
                 roi_detector=None, roi_classes=None, roi_crop_sizes=(384, 512, 768),
                 pose_refiner=None, missing_target_retries=1):
        self.client = client
        use_default_detectors = detector is None
        if detector is None:
            detector = self._default_detector()
        self.detector = detector
        if roi_detector is None and use_default_detectors:
            roi_detector = self._default_roi_detector()
        self.roi_detector = roi_detector
        if roi_classes is None:
            from .detector import ROI_TARGET_CLASSES
            roi_classes = ROI_TARGET_CLASSES
        self.roi_classes = {normalize_detection_label(name) for name in roi_classes}
        self.roi_crop_sizes = tuple(sorted({int(size) for size in roi_crop_sizes
                                            if int(size) > 1}))
        self.crop_search = bool(crop_search or self.roi_detector is not None)
        if pose_refiner is None and use_default_detectors:
            pose_refiner = self._default_pose_refiner()
        self.pose_refiner = pose_refiner
        if geom is None:
            from geometry.camera import CameraGeometry   # needs rclpy; import on demand
            geom = CameraGeometry(client)
        self.geom = geom
        self._geom_checked = False
        self.missing_target_retries = max(0, int(missing_target_retries))

    # -- per-camera work --

    def _image(self, camera_id):
        msg = self.client.get_cctv_image(camera_id)
        if msg is None:
            return None
        try:
            return image_bytes_to_numpy(msg.data, msg.height, msg.width, msg.encoding)
        except ValueError as e:
            log.warning("[detect] %s: %s", camera_id, e)
            return None

    def _wanted_labels(self, parsed):
        """Which types to look for: the target, plus either the named landmark or all of them."""
        target = parsed.get("target_type")
        labels = [target] if target else []
        target_pose = parsed.get("target_pose")
        if target_pose:
            labels.append(target_pose)
        landmark = parsed.get("landmark")
        if landmark:
            labels.append(landmark)
        else:
            # Sentence named no landmark -- we have to identify it visually.
            labels.extend(LANDMARK_TYPES)
        # The NLU extracts every concrete landmark noun, not only the one submitted to
        # the grader.  A phrase such as "the person on the bench near the taxi" needs
        # both boxes to disambiguate the camera.
        labels.extend(parsed.get("context_landmarks") or [])
        labels.extend(REGION_LANDMARK_TYPES.get(parsed.get("region"), ()))
        # de-duplicate, keep order, drop anything with no prompt we can phrase
        seen, out = set(), []
        for l in labels:
            if l and l not in seen:
                seen.add(l)
                out.append(l)
        return out

    @staticmethod
    def _person_camera_hint_score(dets, parsed):
        """Soft camera evidence from non-submitted sentence context.

        Context landmarks are explicit nouns and therefore receive a strong bonus.
        Region furniture is weaker and capped: seeing one car suggests a parking area,
        but must not override the requested pose or a concrete landmark relation.
        """
        if parsed.get("target_type") != "person":
            return 0.0
        by_label = {}
        for label, confidence, _ in dets:
            by_label[label] = max(by_label.get(label, 0.0), confidence)

        named = parsed.get("landmark")
        context = {normalize_detection_label(label)
                   for label in (parsed.get("context_landmarks") or []) if label}
        context.discard(named)
        score = W_CONTEXT_VISIBLE * sum(by_label.get(label, 0.0) for label in context)

        region_labels = REGION_LANDMARK_TYPES.get(parsed.get("region"), ())
        region_score = W_REGION_VISIBLE * sum(by_label.get(label, 0.0)
                                              for label in region_labels)
        return score + min(W_REGION_VISIBLE_MAX, region_score)

    @staticmethod
    def _default_detector():
        """Use YOLO for known labels and lazy OWLv2 routing for unseen labels.

        A known class missed by YOLO is not retried with OWLv2. If YOLO cannot
        initialize at all, use OWLv2 as the full fallback detector.
        """
        from .detector import (DEFAULT_YOLO_WEIGHTS, HybridDetector, OwlV2Detector,
                               YOLODetector)
        try:
            return HybridDetector(YOLODetector(DEFAULT_YOLO_WEIGHTS))
        except Exception as e:  # noqa: BLE001 -- missing weights, no ultralytics, ...
            log.warning("[detect] YOLO unavailable (%s) -- falling back to OWLv2. Stage 1 "
                        "measured 43.87 this way against 59.84 with the checkpoint, so this "
                        "is a real loss, not a neutral swap.", e)
            return OwlV2Detector()

    @staticmethod
    def _default_pose_refiner():
        if os.environ.get("MARC_POSE_ENABLED", "1").strip().lower() in {
                "0", "false", "no", "off"}:
            log.info("[detect] person pose refinement disabled by MARC_POSE_ENABLED")
            return None
        from .detector import DEFAULT_PERSON_POSE_WEIGHTS, PersonPoseRefiner
        weights = os.environ.get("MARC_PERSON_POSE_WEIGHTS", DEFAULT_PERSON_POSE_WEIGHTS)
        try:
            return PersonPoseRefiner(weights=weights)
        except Exception as exc:  # noqa: BLE001 -- optional refinement must safely fall back
            log.warning("[detect] person pose refinement unavailable (%s)", exc)
            return None

    @staticmethod
    def _default_roi_detector():
        """Load the crop-trained 10-target YOLO without disabling the full-frame path.

        ``MARC_ROI_ENABLED=0`` provides a clean live A/B switch.  A missing/corrupt ROI
        checkpoint degrades to the established full-frame detector instead of taking the
        entire real pipeline down.
        """
        if os.environ.get("MARC_ROI_ENABLED", "1").strip().lower() in {
                "0", "false", "no", "off"}:
            log.info("[detect] ROI second-look disabled by MARC_ROI_ENABLED")
            return None
        from .detector import DEFAULT_ROI_YOLO_WEIGHTS, YOLODetector
        weights = os.environ.get("MARC_ROI_YOLO_WEIGHTS", DEFAULT_ROI_YOLO_WEIGHTS)
        try:
            return YOLODetector(weights)
        except Exception as exc:  # noqa: BLE001 -- optional checkpoint/device dependency
            log.warning("[detect] ROI second-look unavailable (%s); full-frame YOLO remains active",
                        exc)
            return None

    @staticmethod
    def _iou(a, b):
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
        return inter / area if area > 0 else 0.0

    @staticmethod
    def _square_crop(cx, cy, size, width, height):
        crop_width = min(int(size), int(width))
        crop_height = min(int(size), int(height))
        left = int(round(cx - crop_width / 2))
        top = int(round(cy - crop_height / 2))
        left = max(0, min(left, width - crop_width))
        top = max(0, min(top, height - crop_height))
        return left, top, left + crop_width, top + crop_height

    def _crop_targets(self, image, target_type, landmark_dets):
        """Re-detect the target inside a window around each landmark, in full-frame coords."""
        h, w = image.shape[:2]
        out = []
        detector = self.roi_detector or self.detector
        if self.roi_detector is not None and target_type not in self.roi_classes:
            return out
        seen_bounds = set()
        for _, _, box in landmark_dets[:CROP_MAX_LANDMARKS]:
            cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
            if self.roi_detector is not None:
                bounds_list = [self._square_crop(cx, cy, size, w, h)
                               for size in self.roi_crop_sizes]
            else:
                side = max(max(box[2] - box[0], box[3] - box[1]) * CROP_EXPAND,
                           CROP_MIN_PX)
                bounds_list = [self._square_crop(cx, cy, side, w, h)]
            for bounds in bounds_list:
                if bounds in seen_bounds:
                    continue
                seen_bounds.add(bounds)
                x0, y0, x1, y1 = bounds
                if x1 - x0 >= w and y1 - y0 >= h:
                    continue                  # whole frame gives no second-look scale gain
                crop = image[y0:y1, x0:x1]
                if crop.shape[0] < 32 or crop.shape[1] < 32:
                    continue
                for label, conf, b in detector.detect(crop, [target_type]):
                    if label == target_type:
                        out.append((label, conf,
                                    (b[0] + x0, b[1] + y0,
                                     b[2] + x0, b[3] + y0)))
        return out

    def _merge_targets(self, primary, extra):
        """Merge crop detections with full-frame detections.

        Boxes with IoU above 0.5 are treated as duplicates;
        the higher-confidence box is retained. Confidence across crop scales
        is not calibrated, so this is a deduplication heuristic.
        """
        merged = list(primary)
        for cand in extra:
            hit = next((i for i, d in enumerate(merged)
                        if self._iou(cand[2], d[2]) > 0.5), None)
            if hit is None:
                merged.append(cand)
            elif cand[1] > merged[hit][1]:
                merged[hit] = cand
        merged.sort(key=lambda d: -d[1])
        return merged

    def _score_pair(self, target_det, landmark_det, target_xyz, anchor_xyz, want_relation,
                    target_pose=None, target_is_person=False):
        """How good is this (target instance, landmark instance) combination?"""
        score = target_det[1]
        if target_pose and target_det[0] == target_pose:
            score += W_POSE_MATCH
        if landmark_det is None:
            return score
        if target_is_person:
            # Person near/beside distances overlap, so an exact relation-band bonus
            # can outweigh better pose evidence. Use contradiction penalties and
            # smooth proximity instead; on-support overlap remains an instance cue.
            score += W_LANDMARK_VISIBLE
            if want_relation == "on":
                score += self._person_on_2d_score(target_det[2], landmark_det[2])
            else:
                score += min(0.0, self._relation_score(
                    anchor_xyz, target_xyz, want_relation))
                # Smooth distance evidence avoids a discontinuity at the overlapping
                # near/beside boundary and also supports commands without a relation.
                distance = math.hypot(target_xyz[0] - anchor_xyz[0],
                                      target_xyz[1] - anchor_xyz[1])
                score += W_PERSON_PROXIMITY * max(
                    0.0, 1.0 - distance / MAX_ANCHOR_TARGET_DIST)
        else:
            score += 0.5 * landmark_det[1] + W_LANDMARK_VISIBLE
            score += self._relation_score(anchor_xyz, target_xyz, want_relation)
        return score

    @staticmethod
    def _person_on_2d_score(person_box, landmark_box):
        """Score person/support overlap as an image-space instance cue.

        A sitting or lying bbox bottom can project poorly onto a support plane.
        Require overlap and horizontal-centre containment, then use IoU to rank
        supported-person candidates.
        """
        px0, py0, px1, py1 = person_box
        lx0, ly0, lx1, ly1 = landmark_box
        ix0, iy0 = max(px0, lx0), max(py0, ly0)
        ix1, iy1 = min(px1, lx1), min(py1, ly1)
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        person_cx = 0.5 * (px0 + px1)
        if iw <= 0.0 or ih <= 0.0 or not (lx0 <= person_cx <= lx1):
            return -W_RELATION_CONTRADICT
        inter = iw * ih
        person_area = max(0.0, px1 - px0) * max(0.0, py1 - py0)
        landmark_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
        union = person_area + landmark_area - inter
        iou = inter / union if union > 0.0 else 0.0
        return W_RELATION_MATCH + W_PERSON_ON_IOU * iou

    @staticmethod
    def _relation_compatible(anchor_xyz, target_xyz, want_relation):
        """Check spatial compatibility for object/landmark pairs.

        Near and beside may overlap in distance. A nonadjacent relation mismatch
        is stronger evidence that the candidate is a different object instance.
        """
        if not want_relation:
            return True
        got = relation_from_geometry(anchor_xyz, target_xyz)
        if got == want_relation:
            return True
        return {got, want_relation} == {"near", "beside"}

    @staticmethod
    def _landmark_fallback(camera_id, anchors, named_landmark, want_relation,
                           target_type):
        """Preserve a named landmark when the requested object has no valid pixels.

        A hidden object cannot be reconstructed from RGB.  Returning the strongest
        named landmark still preserves useful evidence: camera/anchor can score, and
        ``target_coord == anchor_coord`` is a defensible approximation for ``on``.
        Callers can distinguish this from a real target through ``target_detected`` and
        ``fallback_reason`` instead of silently treating an invented bbox as evidence.

        Person problems are excluded.  A person's target point is not interchangeable
        with a landmark, and the SAR selector needs to solve those instances explicitly.
        """
        if ((target_type or "").startswith("person") or
                not named_landmark or not anchors):
            return None
        landmark_det, anchor_xyz = max(anchors, key=lambda item: item[0][1])
        score = W_LANDMARK_VISIBLE + 0.5 * landmark_det[1]
        if want_relation == "on":
            score += W_RELATION_MATCH
        return (score, {
            "camera_id": camera_id,
            "target_coord": anchor_xyz,
            "anchor_coord": anchor_xyz,
            "landmark": landmark_det[0],
            "target_conf": None,
            "landmark_conf": landmark_det[1],
            "target_detected": False,
            "fallback_reason": "no_relation_compatible_target",
        })

    @staticmethod
    def _person_landmark_camera_fallback(camera_id, anchors, named_landmark,
                                         camera_hint_score):
        """Retain a landmark-only camera when person detection is missing.

        A named landmark provides camera evidence even without a detected person.
        Return an explicitly incomplete candidate without inventing target coordinates.
        """
        if not named_landmark or not anchors:
            return None
        landmark_det, anchor_xyz = max(anchors, key=lambda item: item[0][1])
        score = (W_LANDMARK_VISIBLE + landmark_det[1] + camera_hint_score
                 - W_TARGET_COORD_MISSING)
        return (score, {
            "camera_id": camera_id,
            "target_coord": None,
            "anchor_coord": anchor_xyz,
            "landmark": landmark_det[0],
            "target_conf": None,
            "landmark_conf": landmark_det[1],
            "target_detected": False,
            "fallback_reason": "person_missing_named_landmark_visible",
        })

    @staticmethod
    def _relation_score(anchor_xyz, target_xyz, want_relation):
        """Reward relation agreement and penalize nonadjacent contradictions.

        Missing relation information is distinct from contradictory geometry.
        """
        if not want_relation:
            return 0.0
        got = relation_from_geometry(anchor_xyz, target_xyz)
        if got == want_relation:
            return W_RELATION_MATCH
        want_i = _RELATION_ORDER.get(want_relation)
        got_i = _RELATION_ORDER.get(got)
        if want_i is None or got_i is None:
            # `got` is None -- the two are further apart than any named relation covers.
            return -W_RELATION_CONTRADICT
        return 0.0 if abs(want_i - got_i) <= 1 else -W_RELATION_CONTRADICT

    def _ground_one_camera(self, camera_id, parsed, labels):
        """Best (score, result) this camera can offer, or None."""
        image = self._image(camera_id)
        if image is None:
            return None
        dets = self.detector.detect(image, labels)
        if not dets:
            return None

        target_type = parsed.get("target_type")
        target_pose = parsed.get("target_pose")
        want_relation = parsed.get("relation")
        named_landmark = parsed.get("landmark")
        camera_hint_score = self._person_camera_hint_score(dets, parsed)

        target_labels = {target_type, target_pose} - {None}
        targets = [d for d in dets if d[0] in target_labels]
        # A named landmark restricts the anchor search to that class. If it is
        # absent, do not silently replace it with a different landmark type.
        landmark_pool = ([d for d in dets if d[0] == named_landmark] if named_landmark
                         else [d for d in dets if d[0] in LANDMARK_TYPES])

        # Second look, close up. Deliberately after landmark_pool and before the
        # `not targets` bail-out: on the rounds where the full frame finds nothing, the
        # landmark is still there to aim at, and that is exactly when this pays.
        if self.crop_search and landmark_pool:
            targets = self._merge_targets(
                targets, self._crop_targets(image, target_type, landmark_pool))
        # Landmarks stand on the floor, so their contact pixel unprojects to the ground.
        anchors = []
        for d in landmark_pool:
            # as_landmark corrects the visible footprint edge toward the landmark
            # centre so the anchor represents the object rather than its near edge.
            xyz = self.geom.unproject(camera_id, *ground_pixel(d[2]), as_landmark=d[0])
            if xyz is not None:
                anchors.append((d, xyz))

        # Landmark visibility alone cannot identify the correct instance: the same
        # class may appear in several cameras. Rank target/anchor pairs using their
        # spatial relationship, and prefer pairs within MAX_ANCHOR_TARGET_DIST.
        # Retain the best distant pair if no pair is inside that radius.
        best = None
        best_ungated = None
        for t in targets:
            # For on-support targets, intersect the ray with the landmark top plane
            # rather than the floor. A named landmark must be detected in this camera
            # to form an instance pair.
            candidates = anchors if named_landmark else (anchors or [(None, None)])
            for lm, anchor_xyz in candidates:
                lift = lm[0] if (lm is not None and want_relation == "on") else None
                t_xyz = self.geom.unproject(camera_id, *ground_pixel(t[2]), on_landmark=lift)
                if t_xyz is None:
                    continue
                if (not (target_type or "").startswith("person") and
                        lm is not None and want_relation and
                        not self._relation_compatible(anchor_xyz, t_xyz, want_relation)):
                    continue
                score = self._score_pair(t, lm, t_xyz, anchor_xyz, want_relation,
                                         target_pose=target_pose,
                                         target_is_person=(target_type == "person"))
                score += camera_hint_score
                near = (anchor_xyz is None or
                        math.hypot(t_xyz[0] - anchor_xyz[0],
                                   t_xyz[1] - anchor_xyz[1]) <= MAX_ANCHOR_TARGET_DIST)
                if not near and best_ungated is not None and score <= best_ungated[0]:
                    continue
                candidate = (score, {
                        "camera_id": camera_id,
                        "target_coord": t_xyz,
                        "anchor_coord": anchor_xyz if anchor_xyz is not None else t_xyz,
                        "landmark": lm[0] if lm is not None else (named_landmark or ""),
                        "target_conf": t[1],
                        "target_bbox": list(t[2]),
                        "target_plane_landmark": lift,
                        "landmark_conf": lm[1] if lm is not None else None,
                        "target_detected": True,
                        "detected_label": t[0],
                        "fallback_reason": None,
                    })
                if near:
                    if best is None or score > best[0]:
                        best = candidate
                elif best_ungated is None or score > best_ungated[0]:
                    best_ungated = candidate

        best = best or best_ungated
        if best is None:
            if target_type == "person":
                fallback = self._person_landmark_camera_fallback(
                    camera_id, anchors, named_landmark, camera_hint_score)
                if fallback is not None:
                    return fallback
            fallback = self._landmark_fallback(
                camera_id, anchors, named_landmark, want_relation, target_type)
            if fallback is not None:
                return fallback
            if not targets:
                return None
            # Geometry unavailable (tf/intrinsics missing, or the frame convention is
            # wrong). Still name a camera so the answer is not empty -- coordinates will
            # be filled with whatever the caller falls back to.
            t = targets[0]
            return (t[1] + camera_hint_score - W_TARGET_COORD_MISSING, {
                "camera_id": camera_id, "target_coord": None, "anchor_coord": None,
                "landmark": named_landmark or "", "target_conf": t[1],
                "target_bbox": list(t[2]), "detected_label": t[0],
                "landmark_conf": None, "target_detected": True,
                "fallback_reason": "geometry_unavailable",
            })
        return best

    # -- public API --

    def ground(self, parsed, camera_ids=None):
        """Return the selected camera, coordinates, landmark, and detection metadata.

        If no concrete landmark is named, the selected target/landmark pair supplies
        one. Selection combines confidence and context with geometric compatibility;
        it is not simply a nearest-landmark lookup. Return None if no candidate exists.
        """
        # Callers may bypass the VLA agent, so normalize labels at this boundary too.
        # Detector output and geometry lookup keys must use the same catalogue names.
        parsed = dict(parsed)
        parsed["target_type"] = normalize_detection_label(parsed.get("target_type"))
        parsed["detection_target_type"] = normalize_detection_label(
            parsed.get("detection_target_type"))
        parsed["landmark"] = normalize_detection_label(parsed.get("landmark"))
        target_type = parsed.get("target_type")
        target_pose = parsed.get("detection_target_type")
        if (target_type or "").startswith("person_"):
            target_pose = target_pose or target_type
            target_type = "person"
        if not target_type:
            log.warning("[detect] NLU returned no target_type -- nothing to look for")
            return None
        # Person poses are useful visual classes but are not the API submission label:
        # YOLO was trained on person_sitting/person_prone/etc., while GroundingResult
        # must submit plain "person".  Keep the pose only inside vision instead of
        # throwing it away before camera/instance selection.
        vision_parsed = dict(
            parsed,
            target_type=target_type,
            target_pose=target_pose,
            relation=(parsed.get("detection_relation")
                      if "detection_relation" in parsed else parsed.get("relation")),
        )
        if vision_parsed.get("landmark") in NON_VISUAL_LANDMARK_TYPES:
            log.info("[detect] semantic area %s is not a visual landmark; "
                     "searching for a concrete nearby anchor",
                     vision_parsed["landmark"])
            vision_parsed["landmark"] = None
            # A relation to an area surface ("on the parking lot floor") does not
            # describe the relation to the concrete anchor we are about to infer.
            vision_parsed["relation"] = None
        labels = self._wanted_labels(vision_parsed)

        # /tf_static is latched, but the first mission can still arrive before it lands.
        # Without this the first round would silently submit no coordinates.
        if not self._geom_checked:
            self._geom_checked = True
            if hasattr(self.geom, "wait_ready") and not self.geom.wait_ready():
                log.warning("[detect] geometry not ready -- this round will have "
                            "no coordinates")

        cameras = camera_ids or self.client.list_cctv()
        best = None
        # Retry incomplete outcomes using fresh CCTV frames to tolerate transient
        # misses or occlusion. Complete results keep their existing latency and selection.
        for attempt in range(self.missing_target_retries + 1):
            pass_best = None
            for cid in cameras:
                got = self._ground_one_camera(cid, vision_parsed, labels)
                if got is not None and (pass_best is None or got[0] > pass_best[0]):
                    pass_best = got
            if pass_best is not None:
                if best is None or pass_best[0] > best[0]:
                    best = pass_best
                if (pass_best[1].get("target_detected", True)
                        and pass_best[1].get("target_coord") is not None):
                    best = pass_best
                    break
            if attempt < self.missing_target_retries:
                log.info("[detect] target coordinate missing -- retrying latest CCTV frames "
                         "(%d/%d)", attempt + 1, self.missing_target_retries)

        if best is None:
            log.warning("[detect] %r not found on any of %d cameras",
                        target_pose or target_type, len(cameras))
            return None

        score, result = best
        # For a detected person on a support, align XY with the support anchor
        # while preserving the selected target plane Z. This is an empirical
        # reference-point heuristic, not a full estimate of body position.
        if (target_type == "person" and result.get("target_detected", True)
                and result.get("target_plane_landmark")
                and result.get("target_coord") and result.get("anchor_coord")):
            target_coord = list(result["target_coord"])
            target_coord[0] = result["anchor_coord"][0]
            target_coord[1] = result["anchor_coord"][1]
            result["target_coord"] = target_coord
            result["pose_refinement"] = {"policy": "support_anchor_xy"}
            log.info("[detect] person-on-support coordinate aligned to %s anchor XY",
                     result["target_plane_landmark"])
        elif (target_type == "person" and target_pose == "person_prone"
                and result.get("target_detected", True) and result.get("target_bbox")):
            x0, y0, x1, y1 = result["target_bbox"]
            nx, ny = PRONE_BBOX_REFERENCE
            pixel = (x0 + nx * (x1 - x0), y0 + ny * (y1 - y0))
            refined_xyz = self.geom.unproject(result["camera_id"], *pixel)
            if refined_xyz is not None:
                result["target_coord"] = refined_xyz
                result["pose_refinement"] = {
                    "policy": "prone_bbox_left_75", "pixel": pixel,
                }
                log.info("[detect] person_prone coordinate refined via prone bbox point")
        elif (self.pose_refiner is not None and result.get("target_detected", True)
                and target_pose in {"person_lying_down", "person_bending"}
                and result.get("target_bbox")):
            image = self._image(result["camera_id"])
            refined = self.pose_refiner.reference_pixel(
                image, result["target_bbox"], target_pose)
            if refined is not None:
                refined_xyz = self.geom.unproject(
                    result["camera_id"], *refined["pixel"],
                    on_landmark=result.get("target_plane_landmark"))
                if refined_xyz is not None:
                    if (target_pose == "person_bending"
                            and result.get("landmark") == "trash_can"):
                        refined_xyz = [value + offset for value, offset in zip(
                            refined_xyz, BENDING_TRASH_WORLD_OFFSET)]
                        refined = dict(refined)
                        refined["policy"] = "ankle_trash_world_offset"
                    result["target_coord"] = refined_xyz
                    result["pose_refinement"] = refined
                    log.info("[detect] %s pose coordinate refined via %s",
                             target_pose, refined["policy"])
        target_conf = result.get("target_conf")
        log.info("[detect] %s on %s (score %.2f, conf %s) landmark=%s target=%s",
                 target_pose or target_type, result["camera_id"], score,
                 f"{target_conf:.2f}" if target_conf is not None else "fallback",
                 result["landmark"] or "-",
                 [round(v, 2) for v in result["target_coord"]] if result["target_coord"] else None)

        # Sanity signal, not an error: text and geometry disagreeing about the relation
        # usually means we locked onto the wrong instance.
        if (result.get("target_detected", True) and parsed.get("relation") and
                result["target_coord"] and result["anchor_coord"]):
            got_rel = relation_from_geometry(result["anchor_coord"], result["target_coord"])
            if got_rel and got_rel != parsed["relation"]:
                log.warning("[detect] relation mismatch: text=%s geometry=%s",
                            parsed["relation"], got_rel)
        return result

    # Situation classification remains an NLU output. Visual pose information here
    # supports instance selection and coordinate refinement, not situation inference.
