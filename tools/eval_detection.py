#!/usr/bin/env python3
"""Score the visual grounding stage against a recorded dump (은서 담당).

Runs the **real** `detection.grounding.VisualGrounding` -- not a copy of it -- by feeding
it a client and a geometry object backed by files instead of ROS. So whatever this
reports is what the agent would have done on those frames.

    python tools/eval_detection.py ../cctv_dump_full
    python tools/eval_detection.py ../cctv_dump_full --rounds 3        # smoke test, ~5 min on CPU
    python tools/eval_detection.py ../cctv_dump_full --stub            # plumbing only, no torch
    python tools/eval_detection.py ../cctv_dump_full --threshold 0.08  # detector confidence sweep
    python tools/eval_detection.py ../cctv_dump_full --yolo-weights ../best.pt
    python tools/eval_detection.py ../cctv_dump_full --use-nlu         # end-to-end, includes 은재's parser
    python tools/eval_detection.py ../cctv_dump_full --save baseline.json

**Ground truth comes from the dump itself** -- `info["score"]["expected"]`, written by the
platform's own grader. It is NOT read from the starter kit answer table any more. That file is
the demo agent's lookup table and says so in its own header ("GT + random wrong axes"): 12
of its coordinate pairs carry a fixed +12/+12 m corruption, and several relation/landmark
labels are wrong too. Scoring against it reported ~17 m errors for answers that were
exactly right, which made a threshold sweep meaningless.

Needs no ROS, no platform, and no GPU -- OWLv2 on CPU is slow but works.
"""

import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.grounding import VisualGrounding  # noqa: E402
from geometry.transform import (LANDMARK_HALF_DEPTH, LANDMARK_TOP_Z,  # noqa: E402
                                push_from_camera, quat_to_matrix,
                                unproject_to_ground, unproject_to_plane)

# Per-component weights come from each round's own info["score"]["weights"] -- they are NOT
# the same for every problem. Object problems weight all six and sum to 1.0. Person (SAR)
# problems set **landmark and relation to 0.0**, so their weights sum to 0.8 and the platform
# normalises the total by that sum. Two consequences worth knowing before optimising anything:
# on a person problem the landmark and the relation earn nothing at all, while `situation`
# is worth 0.15/0.8 = 18.75% and the target coordinate a full 50%.
#
# The technical guide says weights are not published; the practice scenario returns them
# anyway. This table is only a fallback for a dump recorded without them.
DEFAULT_WEIGHTS = {"camera": 0.10, "object_type": 0.15, "landmark": 0.10,
                   "relation": 0.10, "anchor": 0.15, "target": 0.40}
COMPONENTS = ("camera", "object_type", "landmark", "relation", "anchor", "target")

# The distance at which the platform stops awarding a coordinate is NOT observable from the
# dump: every recorded coordinate was either exact (0.00 m) or corrupted by exactly 16.97 m,
# so the cut lies somewhere in between and nothing narrows it further. Hence the estimated
# score is always printed across a range rather than as one number.
COORD_THRESHOLDS = (0.5, 1.0, 2.0, 3.0)


# -- file-backed stand-ins for the ROS client / geometry ------------------

class DumpClient:
    """Serves one recorded round through the same API MARCClient exposes."""

    def __init__(self, dump_dir, meta):
        self.dump_dir = dump_dir
        self.meta = meta
        self.round_dir = None
        self._cache = {}

    def use_round(self, round_dir):
        self.round_dir = round_dir
        self._cache.clear()

    def list_cctv(self):
        if self.round_dir is None:
            return []
        return sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(self.round_dir) if f.endswith(".png")
        )

    def get_cctv_image(self, camera_id):
        """Returns an object shaped like sensor_msgs/Image so callers need no branching."""
        if camera_id in self._cache:
            return self._cache[camera_id]
        path = os.path.join(self.round_dir, camera_id + ".png")
        if not os.path.exists(path):
            return None
        from PIL import Image
        arr = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        msg = SimpleNamespace(data=arr.tobytes(), height=arr.shape[0],
                              width=arr.shape[1], encoding="rgb8")
        self._cache[camera_id] = msg
        return msg

    def get_cctv_info(self, camera_id):
        cam = self.meta["cameras"].get(camera_id)
        if not cam or not cam.get("K"):
            return None
        return SimpleNamespace(k=cam["K"])

    def get_cctv_ground_height(self, camera_id):
        cam = self.meta["cameras"].get(camera_id)
        return None if not cam else cam.get("ground_height")


class DumpGeometry:
    """CameraGeometry's interface, reading extrinsics from meta.json instead of /tf_static."""

    def __init__(self, meta):
        self.cams = {}
        for cid, c in meta.get("cameras", {}).items():
            if not c.get("K") or not c.get("tf") or c.get("ground_height") is None:
                continue
            qx, qy, qz, qw = c["tf"]["rotation_xyzw"]
            self.cams[cid] = (
                np.array(c["K"], dtype=float).reshape(3, 3),
                quat_to_matrix(qx, qy, qz, qw),
                np.array(c["tf"]["translation"], dtype=float),
                float(c["ground_height"]),
            )

    def unproject(self, camera_id, u, v, on_landmark=None, as_landmark=None):
        """Must stay signature-identical to CameraGeometry.unproject.

        The point of this harness is that whatever it reports is what the agent would have
        done, so it has to honour exactly the arguments the real geometry honours -- no more
        and no fewer. Mirror any change to CameraGeometry.unproject here.
        """
        got = self.cams.get(camera_id)
        if got is None:
            return None
        K, R, t, ground = got
        if on_landmark:
            lift = LANDMARK_TOP_Z.get(on_landmark)
            if lift is not None:
                return unproject_to_plane(u, v, K, R, t, ground + lift)
        p = unproject_to_ground(u, v, K, R, t, ground)
        if p is not None and as_landmark:
            p = push_from_camera(p, t, LANDMARK_HALF_DEPTH.get(as_landmark))
        return p


# -- ground truth ----------------------------------------------------------

def load_rounds(dump_dir):
    """Every recorded round that carries grader ground truth, in order.

    Rounds the mock scored 0 on are kept: the submission was empty but `expected` is
    complete, so they are perfectly good evaluation cases (5 of the 32 are like this).
    The stage2 entry is skipped -- its `score` block is the run-wide summary, not a
    per-problem answer.
    """
    rounds = []
    for entry in sorted(os.listdir(dump_dir)):
        info_path = os.path.join(dump_dir, entry, "info.json")
        if not os.path.isfile(info_path):
            continue
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        expected = (info.get("score") or {}).get("expected")
        if not expected:
            continue
        rounds.append((os.path.join(dump_dir, entry), info, expected))
    return rounds


def truth_interpretation(expected, command=""):
    """The interpretation fields as the grader expects them to be submitted.

    `target_type` is "person" for person problems, not the pose-specific asset class the
    grader stores: every recorded person round submitted "person" and scored 1.0 whenever
    the situation matched, which is also what the API reference specifies.
    """
    detection_relation = expected.get("relation")
    if expected.get("target_kind") == "person":
        # Person relation carries zero scoring weight and the grader therefore stores
        # null, even when the command explicitly says "on top of" or "near".  The NLU
        # still extracts that text relation and grounding uses it to select an instance.
        # Use the maintained NLU fixture here so oracle-NLU evaluation exercises the
        # same visual input without changing what gets submitted to the scorer.
        try:
            from nlu.test_data import KNOWN_EXAMPLES
            known = next((case for case in KNOWN_EXAMPLES
                          if case.get("sentence") == command), None)
            detection_relation = known.get("relation") if known else None
        except ImportError:
            detection_relation = None
    return {
        "target_type": ("person" if expected.get("target_kind") == "person"
                        else expected.get("target_type")),
        # The scorer wants plain "person", but vision should use the pose-specific
        # class that the NLU produces and the YOLO checkpoint was trained to detect.
        "detection_target_type": (expected.get("target_type")
                                  if expected.get("target_kind") == "person" else None),
        "detection_relation": detection_relation,
        "landmark": expected.get("landmark_prefix") or None,
        "relation": expected.get("relation"),
        "situation": expected.get("situation"),
    }


# -- grading ---------------------------------------------------------------

def dist3(a, b):
    if not a or not b:
        return None
    return float(math.dist(tuple(a)[:3], tuple(b)[:3]))


def dist_xy(a, b):
    if not a or not b:
        return None
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def grade(expected, submitted, coord_threshold):
    """Per-component result, replicating the platform's grading as far as it is observable."""
    out = {}
    cam = submitted.get("camera_id")
    out["camera"] = 1.0 if cam and cam == expected.get("camera_id") else 0.0

    # object_type is scored on `situation` for person problems: all 12 recorded person
    # rounds submitted target_type="person" and scored 1.0 exactly when the situation
    # matched, 0.0 when it did not. For object problems it is the type itself.
    if expected.get("target_kind") == "person":
        out["object_type"] = 1.0 if (submitted.get("target_type") == "person"
                                     and submitted.get("situation") == expected.get("situation")) else 0.0
    else:
        out["object_type"] = 1.0 if submitted.get("target_type") == expected.get("target_type") else 0.0

    out["landmark"] = 1.0 if ((submitted.get("landmark") or None)
                              == (expected.get("landmark_prefix") or None)) else 0.0

    # The platform awards relation 0 whenever the expected relation is null -- all 12
    # person rounds submitted null, matched it, and still scored 0. Reproduced so the
    # estimate matches what would actually be awarded rather than what seems fair.
    exp_rel = expected.get("relation")
    out["relation"] = 1.0 if (exp_rel and submitted.get("relation") == exp_rel) else 0.0

    for key in ("anchor", "target"):
        err = dist3(submitted.get(key + "_coord"), expected.get(key + "_coord"))
        out[key] = 1.0 if (err is not None and err <= coord_threshold) else 0.0
    return out


def weighted_total(comp, weights):
    """Score out of 100, normalised by the weights actually in play for this problem.

    The normalisation is what makes a person round with relation=0 still reach 100: its
    weights sum to 0.8, not 1.0.
    """
    denom = sum(weights.values()) or 1.0
    return 100.0 * sum(weights.get(c, 0.0) * comp[c] for c in COMPONENTS) / denom


def effective_weights(weights):
    """Weights as a fraction of this problem's own total -- comparable across rounds."""
    denom = sum(weights.values()) or 1.0
    return {c: weights.get(c, 0.0) / denom for c in COMPONENTS}


def total_at(rec, threshold):
    """Re-score one round at a different coordinate threshold (for the sensitivity table)."""
    comp = dict(rec["comp"])
    for key in ("anchor", "target"):
        err = rec[key + "_err"]
        comp[key] = 1.0 if (err is not None and err <= threshold) else 0.0
    return weighted_total(comp, rec["weights"])


# -- reporting -------------------------------------------------------------

def report(records, coord_threshold, elapsed):
    n = len(records) or 1
    print("\n" + "=" * 78)
    print(f"DETECTION EVAL  ({len(records)} rounds, coord threshold {coord_threshold:.2f} m, "
          f"{elapsed:.0f}s)")
    print("=" * 78)

    found = sum(1 for r in records if r["found"])
    print(f"  target found      : {found}/{len(records)} ({100 * found / n:.0f}%)")
    print("\n  target recall by expected class (found / selected correct camera):")
    by_class = {}
    for r in records:
        cls = r["expected"].get("target_type") or "unknown"
        found_hits, cam_hits, total = by_class.get(cls, (0, 0, 0))
        by_class[cls] = (found_hits + int(r["found"]),
                         cam_hits + int(r["comp"]["camera"] == 1.0), total + 1)
    for cls, (found_hits, cam_hits, total) in sorted(by_class.items()):
        print(f"    {cls:<24} {found_hits:2d}/{total:<2d} ({100 * found_hits / total:3.0f}%)"
              f"   camera {cam_hits:2d}/{total:<2d} ({100 * cam_hits / total:3.0f}%)")
    print()
    # "counts in" matters because landmark and relation carry zero weight on person
    # problems -- a hit-rate over all rounds would mix in rounds that award nothing.
    print("  component        counts in   hit-rate      avg weight   lost pts")
    for c in COMPONENTS:
        live = [r for r in records if r["weights"].get(c, 0.0) > 0]
        m = len(live) or 1
        hits = sum(r["comp"][c] for r in live)
        eff = [effective_weights(r["weights"])[c] for r in records]
        lost = sum(w * (1 - r["comp"][c]) for w, r in zip(eff, records)) * 100 / n
        print(f"  {c:<17}{len(live):3d}/{len(records):<3}  {hits:5.0f}/{m:<3} ({100 * hits / m:3.0f}%)"
              f"    {sum(eff) / n:7.3f}    -{lost:5.2f}")

    for name in ("target", "anchor"):
        errs = [r[name + "_err"] for r in records if r[name + "_err"] is not None]
        if errs:
            e = np.array(errs)
            xy = np.array([r[name + "_err_xy"] for r in records
                           if r[name + "_err_xy"] is not None])
            print(f"\n  {name}_coord error : median {np.median(e):.2f} m, "
                  f"p90 {np.percentile(e, 90):.2f} m, worst {e.max():.2f} m  "
                  f"({len(errs)} of {len(records)} produced a coordinate)")
            print(f"  {' ' * len(name)}   horizontal : median {np.median(xy):.2f} m, "
                  f"p90 {np.percentile(xy, 90):.2f} m")

    print("\n  estimated Stage 1 score, by coordinate threshold:")
    print("    (the real cut is unknown -- the dump only shows exact hits and 16.97 m misses)")
    for thr in COORD_THRESHOLDS:
        avg = sum(total_at(r, thr) for r in records) / n
        print(f"      <= {thr:4.2f} m : {avg:6.2f} / 100")

    print("\nper-round:")
    print(f"  {'round':<10}{'problem_id':<26}{'score':>6}  misses")
    for r in records:
        miss = [c for c in COMPONENTS if r["comp"][c] == 0.0]
        note = r.get("note") or ", ".join(miss) or "-"
        print(f"  {r['dir']:<10}{str(r['pid']):<26}{r['total']:6.1f}  {note}")


# -- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--stub", action="store_true",
                    help="StubDetector -- checks wiring without torch")
    ap.add_argument("--yolo-weights", metavar="PATH",
                    help="use an Ultralytics YOLO checkpoint instead of OWLv2")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="YOLO inference image size (default 1280)")
    ap.add_argument("--device", default=None,
                    help="YOLO device, e.g. 0 or cpu (default: Ultralytics auto-select)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the detector confidence threshold")
    ap.add_argument("--use-nlu", action="store_true",
                    help="parse commands with nlu/ instead of using the grader's labels")
    ap.add_argument("--blind-landmark", action="store_true",
                    help="blank the landmark before grounding, forcing the visual-search "
                         "path that has to recover it (what happens on the commands that "
                         "name no landmark). Ignored with --use-nlu.")
    ap.add_argument("--crop-search", action="store_true",
                    help="landmark-anchored second look. Off by default: it found nothing "
                         "against the 1280-trained YOLO and doubled the runtime -- see the "
                         "note in detection/grounding.py")
    ap.add_argument("--roi-yolo-weights", metavar="PATH",
                    help="evaluation-only second YOLO for landmark-centred crops")
    ap.add_argument("--roi-classes",
                    default="mug,tumbler,umbrella,sunblock,pencilcase,juice,cola_can,"
                            "cracker_box,tissue,disposable_cup",
                    help="comma-separated classes eligible for ROI retry")
    ap.add_argument("--roi-crop-sizes", default="384,512,768",
                    help="comma-separated source-pixel square crop sizes")
    ap.add_argument("--person-pose-weights", metavar="PATH",
                    help="post-selection lying/bending person pose refinement")
    ap.add_argument("--rounds", type=int, default=None,
                    help="only evaluate the first N rounds (smoke test)")
    ap.add_argument("--coord-threshold", type=float, default=1.0,
                    help="distance at which a coordinate counts as correct (default 1.0 m)")
    ap.add_argument(
        "--ground-height-override", action="append", default=[], metavar="CAMERA=METERS",
        help="override a recorded camera ground height for offline evaluation; repeatable",
    )
    ap.add_argument("--save", metavar="PATH",
                    help="write the full result to JSON so later runs can be diffed "
                         "against this baseline")
    args = ap.parse_args()

    with open(os.path.join(args.dump_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    for item in args.ground_height_override:
        try:
            camera_id, raw_height = item.split("=", 1)
            height = float(raw_height)
        except ValueError:
            ap.error("--ground-height-override must use CAMERA=METERS")
        camera = meta.get("cameras", {}).get(camera_id)
        if camera is None:
            ap.error(f"unknown camera in --ground-height-override: {camera_id}")
        camera["ground_height"] = height
    rounds = load_rounds(args.dump_dir)
    if not rounds:
        print("no rounds with grader ground truth in this dump.\n"
              "Expected info.json files carrying score.expected -- was it recorded with "
              "tools/dump_cctv.py against a self-scoring scenario (marc2026_chungmu)?")
        return
    if args.rounds:
        rounds = rounds[:args.rounds]

    if args.stub and args.yolo_weights:
        ap.error("--stub and --yolo-weights cannot be used together")
    if args.roi_yolo_weights and not args.yolo_weights:
        ap.error("--roi-yolo-weights requires --yolo-weights for the full-frame model")
    if args.stub:
        from detection.detector import StubDetector
        detector = StubDetector()
    elif args.yolo_weights:
        from detection.detector import YOLODetector
        detector = YOLODetector(
            args.yolo_weights,
            threshold=args.threshold if args.threshold is not None else 0.25,
            device=args.device,
            imgsz=args.imgsz,
        )
    else:
        from detection.detector import OwlV2Detector
        detector = (OwlV2Detector(threshold=args.threshold) if args.threshold
                    else OwlV2Detector())

    nlu = None
    if args.use_nlu:
        from nlu.parser import NLUParser
        nlu = NLUParser()

    client = DumpClient(args.dump_dir, meta)
    geom = DumpGeometry(meta)
    if not geom.cams:
        print("meta.json has no camera with K + tf + ground_height.\n"
              "Coordinates cannot be scored -- run tools/verify_geometry.py first.")
    pose_refiner = None
    if args.person_pose_weights:
        from detection.detector import PersonPoseRefiner
        pose_refiner = PersonPoseRefiner(
            weights=args.person_pose_weights, device=args.device)

    if args.roi_yolo_weights:
        from detection.detector import YOLODetector
        roi_detector = YOLODetector(
            args.roi_yolo_weights,
            threshold=args.threshold if args.threshold is not None else 0.25,
            device=args.device,
            imgsz=args.imgsz,
        )
        roi_classes = [name.strip() for name in args.roi_classes.split(",")
                       if name.strip()]
        try:
            roi_crop_sizes = [int(value) for value in args.roi_crop_sizes.split(",")]
        except ValueError:
            ap.error("--roi-crop-sizes must be comma-separated integers")
        if not roi_crop_sizes or any(size <= 1 for size in roi_crop_sizes):
            ap.error("--roi-crop-sizes must contain positive integers")
        vg = VisualGrounding(
            client, detector=detector, geom=geom,
            roi_detector=roi_detector,
            roi_classes=roi_classes,
            roi_crop_sizes=roi_crop_sizes,
            pose_refiner=pose_refiner,
        )
    else:
        vg = VisualGrounding(client, detector=detector, geom=geom,
                             crop_search=args.crop_search,
                             pose_refiner=pose_refiner)

    t0 = time.time()
    records = []
    for round_dir, info, expected in rounds:
        command = (info.get("voice_command") or info.get("task_description") or "")
        client.use_round(round_dir)

        parsed = nlu.parse(command) if nlu else truth_interpretation(expected, command)
        if args.blind_landmark and not nlu:
            parsed = dict(parsed, landmark=None)

        got = vg.ground(parsed)

        # What the agent would actually submit: the vision stage fills in camera and
        # coordinates and may correct the landmark, everything else rides on the parse.
        submitted = dict(parsed)
        note = None
        target_detected = False
        if got is None:
            note = "NOT FOUND"
            submitted.update(camera_id=None, anchor_coord=None, target_coord=None)
        else:
            target_detected = got.get("target_detected", True)
            if not target_detected:
                note = "LANDMARK FALLBACK: " + got.get("fallback_reason", "unknown")
            submitted.update(
                camera_id=got.get("camera_id"),
                anchor_coord=got.get("anchor_coord"),
                target_coord=got.get("target_coord"),
                landmark=got.get("landmark") or parsed.get("landmark"),
            )
            # Present once the vision stage starts inferring posture; until then the
            # parse's situation stands. Keeps this harness from needing a second edit.
            if got.get("situation") is not None:
                submitted["situation"] = got["situation"]

        comp = grade(expected, submitted, args.coord_threshold)
        rw = dict(DEFAULT_WEIGHTS)
        rw.update((info.get("score") or {}).get("weights") or {})
        records.append({
            "dir": os.path.basename(round_dir),
            "pid": (info.get("score") or {}).get("problem_id"),
            "command": command,
            "found": target_detected,
            "result_produced": got is not None,
            "fallback": bool(got is not None and not target_detected),
            "note": note,
            "weights": rw,
            "comp": comp,
            "total": weighted_total(comp, rw),
            "target_err": dist3(submitted.get("target_coord"), expected.get("target_coord")),
            "anchor_err": dist3(submitted.get("anchor_coord"), expected.get("anchor_coord")),
            "target_err_xy": dist_xy(submitted.get("target_coord"), expected.get("target_coord")),
            "anchor_err_xy": dist_xy(submitted.get("anchor_coord"), expected.get("anchor_coord")),
            "vision_trace": ({k: got.get(k) for k in
                              ("target_bbox", "detected_label", "target_conf",
                               "pose_refinement")}
                             if got is not None else None),
            # Coordinates are retained for offline overlays and failure analysis.  They
            # are ordinary grader/pipeline outputs, not hidden simulator annotations.
            "submitted": {k: submitted.get(k) for k in
                          ("camera_id", "target_type", "landmark", "relation", "situation",
                           "anchor_coord", "target_coord")},
            "expected": {k: expected.get(k) for k in
                         ("camera_id", "target_type", "landmark_prefix", "relation", "situation",
                          "anchor_coord", "target_coord")},
        })

    elapsed = time.time() - t0
    report(records, args.coord_threshold, elapsed)

    if args.save:
        n = len(records) or 1
        payload = {
            "dump_dir": os.path.abspath(args.dump_dir),
            "args": {k: v for k, v in vars(args).items() if k != "save"},
            "elapsed_s": round(elapsed, 1),
            "summary": {
                "rounds": len(records),
                "found": sum(1 for r in records if r["found"]),
                "target_recall_by_class": {
                    cls: {
                        "found": sum(1 for r in records
                                     if (r["expected"].get("target_type") or "unknown") == cls
                                     and r["found"]),
                        "correct_camera": sum(1 for r in records
                                              if (r["expected"].get("target_type") or "unknown") == cls
                                              and r["comp"]["camera"] == 1.0),
                        "total": sum(1 for r in records
                                     if (r["expected"].get("target_type") or "unknown") == cls),
                    }
                    for cls in sorted({r["expected"].get("target_type") or "unknown"
                                       for r in records})
                },
                # counted only over rounds where the component carries weight
                "component_hits": {
                    c: [sum(r["comp"][c] for r in records if r["weights"].get(c, 0.0) > 0),
                        sum(1 for r in records if r["weights"].get(c, 0.0) > 0)]
                    for c in COMPONENTS},
                "estimated_score": {str(t): round(sum(total_at(r, t) for r in records) / n, 2)
                                    for t in COORD_THRESHOLDS},
            },
            "rounds_detail": records,
        }
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nsaved -> {args.save}")


if __name__ == "__main__":
    main()
