#!/usr/bin/env python3
"""Measure, per class, whether the detector can find it at all (은서 담당).

`tools/eval_detection.py` answers "what would we score on these rounds". It cannot answer
"which of the 34 object classes can OWLv2 actually see", because a round only exercises the
one or two classes its command happens to mention, and a miss there is indistinguishable
from a camera-selection mistake. This tool answers the detection question directly:

    python tools/eval_class_recall.py ../trainer_output/marc2026_chungmu
    python tools/eval_class_recall.py ../trainer_output/marc2026_chungmu --limit 4
    python tools/eval_class_recall.py ../trainer_output/marc2026_chungmu --threshold 0.03
    python tools/eval_class_recall.py ../trainer_output/marc2026_chungmu --save recall.json

**Ground truth is the generator's own annotation** -- every `marc.sh dataset-gen` scene
ships a JSON listing each visible instance with its class and pixel bbox. That is a far
denser signal than the graded rounds: 36 scenes carry 917 boxes over all 34 classes and all
6 cameras, versus 32 rounds touching a handful of classes.

WHAT IT MEASURES, AND THE ONE DISTINCTION THAT MATTERS
------------------------------------------------------
For each ground-truth box we ask two separate questions:

  1. hit          -- did a detection of *the same class* overlap it (IoU >= --iou)?
  2. mislabelled  -- if not, did a detection of *any other class* overlap it?

The gap between them is the whole point. A class with 0% hit and 0% mislabelled is
invisible to the model -- the pixels produce no box at all, and only a different detector
(or a fine-tuned one) will fix it. A class with 0% hit but high mislabelled is visible and
merely misnamed, which is a prompt-wording problem and cheap to fix -- exactly the failure
that made "a tumbler" score zero while "a water bottle" found it.

Recall is also broken down by ground-truth box area, because the practice scenes put mugs
and juice cartons a few pixels across next to cars that fill a quarter of the frame, and a
single average hides that completely.

Needs no ROS, no platform, and no GPU -- OWLv2 on CPU is slow but works.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.images import DETECT_PROMPT  # noqa: E402

# Area thresholds in pixels, chosen to straddle the practice scenes' actual spread: the
# small bucket is where mugs/cans/juice land and the large one is vehicles and tables.
AREA_BUCKETS = ((0, 32 * 32, "tiny   (<32x32)"),
                (32 * 32, 96 * 96, "small  (32-96)"),
                (96 * 96, 256 * 256, "medium (96-256)"),
                (256 * 256, float("inf"), "large  (>256)"))


def load_scenes(root, limit=None):
    """[(image_path, annotation dict)] for every scene under `root`, in order."""
    scenes = []
    for entry in sorted(os.listdir(root)):
        scene_dir = os.path.join(root, entry)
        if not os.path.isdir(scene_dir):
            continue
        for name in sorted(os.listdir(scene_dir)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(scene_dir, name), encoding="utf-8") as f:
                ann = json.load(f)
            # the overlay png is a debug render with boxes burnt in -- never feed it in
            img = os.path.join(scene_dir, os.path.splitext(name)[0] + ".png")
            if os.path.exists(img):
                scenes.append((img, ann))
    return scenes[:limit] if limit else scenes


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_scene(gt_boxes, dets, iou_thr, group_of):
    """Greedy best-confidence-first matching. Returns one record per ground-truth box.

    A detection may only claim one ground-truth box, so two boxes of the same class in one
    frame need two detections -- otherwise a single lucky box would score both.

    `group_of` maps a class to the tuple of classes sharing its prompt; a detection counts
    as the right label when it falls in the same group, not only on an exact string match.
    """
    out = []
    claimed = set()
    for gi, gt in enumerate(gt_boxes):
        want = group_of[gt["class"]]
        best = None
        for di, (label, score, box) in enumerate(dets):
            if di in claimed or label not in want:
                continue
            ov = iou(gt["bbox"], box)
            if ov >= iou_thr and (best is None or score > best[1]):
                best = (di, score, ov)
        if best is not None:
            claimed.add(best[0])
            out.append({"gt": gt, "hit": True, "conf": best[1], "iou": best[2],
                        "mislabelled_as": None})
            continue
        # no same-class box -- is anything at all sitting on those pixels?
        other = None
        for label, score, box in dets:
            ov = iou(gt["bbox"], box)
            if ov >= iou_thr and (other is None or score > other[1]):
                other = (label, score)
        out.append({"gt": gt, "hit": False, "conf": None, "iou": None,
                    "mislabelled_as": other[0] if other else None})
    return out


def area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def report(records, prompts_used, elapsed, args):
    n = len(records) or 1
    hits = sum(1 for r in records if r["hit"])
    mis = sum(1 for r in records if not r["hit"] and r["mislabelled_as"])
    blind = n - hits - mis

    print("\n" + "=" * 84)
    print(f"CLASS RECALL  ({len(records)} ground-truth boxes, threshold {args.threshold:.3f}, "
          f"IoU {args.iou:.2f}, {elapsed:.0f}s)")
    print("=" * 84)
    print(f"  found (right label) : {hits}/{n} ({100 * hits / n:.0f}%)")
    print(f"  seen, wrong label   : {mis}/{n} ({100 * mis / n:.0f}%)   <- prompt problem, cheap to fix")
    print(f"  no box at all       : {blind}/{n} ({100 * blind / n:.0f}%)   <- detector problem")

    by_class = defaultdict(list)
    for r in records:
        by_class[r["gt"]["class"]].append(r)

    print(f"\n  {'class':<22}{'kind':<10}{'GT':>4}{'found':>10}{'wrong lbl':>11}"
          f"{'med conf':>10}  prompt")
    for cls in sorted(by_class, key=lambda c: (len([r for r in by_class[c] if r['hit']])
                                               / len(by_class[c]), c)):
        rs = by_class[cls]
        h = [r for r in rs if r["hit"]]
        w = [r for r in rs if not r["hit"] and r["mislabelled_as"]]
        conf = f"{np.median([r['conf'] for r in h]):.3f}" if h else "-"
        prompt = prompts_used.get(cls, "")
        star = "" if cls in DETECT_PROMPT else "  (fallback)"
        shared = [c for c in by_class if c != cls and prompts_used.get(c) == prompt]
        star += f"  [shares phrase with {len(shared)}]" if shared else ""
        print(f"  {cls:<22}{rs[0]['gt']['kind']:<10}{len(rs):>4}"
              f"{len(h):>6}/{len(rs):<3}{len(w):>8}    {conf:>8}  {prompt!r}{star}")

    print("\n  recall by ground-truth box size:")
    for lo, hi, name in AREA_BUCKETS:
        rs = [r for r in records if lo <= area(r["gt"]["bbox"]) < hi]
        if not rs:
            continue
        h = sum(1 for r in rs if r["hit"])
        w = sum(1 for r in rs if not r["hit"] and r["mislabelled_as"])
        print(f"    {name:<18} n={len(rs):<4} found {h:>3} ({100 * h / len(rs):3.0f}%)"
              f"   wrong label {w:>3}")

    conf_all = [r["conf"] for r in records if r["hit"]]
    if conf_all:
        c = np.array(conf_all)
        print(f"\n  confidence of correct detections: min {c.min():.3f}, "
              f"p10 {np.percentile(c, 10):.3f}, median {np.median(c):.3f}, max {c.max():.3f}")
        print("    (detector.py's threshold must sit under p10 or those hits are discarded)")

    worst = [(cls, rs) for cls, rs in by_class.items()
             if not any(r["hit"] for r in rs)]
    if worst:
        print(f"\n  classes never found ({len(worst)}):")
        for cls, rs in sorted(worst):
            seen = [r["mislabelled_as"] for r in rs if r["mislabelled_as"]]
            note = (f"seen as {', '.join(sorted(set(seen))[:3])}" if seen
                    else "no box at all -- invisible to the model")
            print(f"    {cls:<22} n={len(rs):<3} {note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_root", help="e.g. ../trainer_output/marc2026_chungmu")
    ap.add_argument("--threshold", type=float, default=0.03,
                    help="detector confidence floor (default 0.03 -- deliberately low, the "
                         "point is to see what is findable at all)")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="IoU at which a detection counts as covering a GT box")
    ap.add_argument("--limit", type=int, default=None, help="only the first N scenes")
    ap.add_argument("--only", metavar="A,B,C",
                    help="restrict to these classes -- use it to iterate on the wording for "
                         "one group cheaply instead of re-running all 34")
    ap.add_argument("--save", metavar="PATH", help="write the full result to JSON")
    args = ap.parse_args()

    scenes = load_scenes(args.scene_root, args.limit)
    if not scenes:
        print(f"no scenes under {args.scene_root} -- expected scene_*/<camera>.json + .png")
        return

    classes = sorted({d["class"] for _img, ann in scenes for d in ann["detections"]})
    if args.only:
        keep = {c.strip() for c in args.only.split(",") if c.strip()}
        unknown = keep - set(classes)
        if unknown:
            print(f"  --only names classes not in this dataset: {', '.join(sorted(unknown))}")
        classes = [c for c in classes if c in keep]
        if not classes:
            print("nothing left to measure")
            return
    prompts_used = {c: DETECT_PROMPT.get(c, f"a {c.replace('_', ' ')}") for c in classes}
    print(f"{len(scenes)} scenes, {len(classes)} classes, "
          f"{sum(len(a['detections']) for _i, a in scenes)} ground-truth boxes")
    missing = [c for c in classes if c not in DETECT_PROMPT]
    if missing:
        print(f"  {len(missing)} class(es) have no DETECT_PROMPT entry and fall back to a "
              f"generated phrase: {', '.join(missing)}")

    # Classes that share a phrase cannot be told apart, so they must not be scored apart.
    # OWLv2 assigns each box to exactly one text query, so querying "person_sitting" and
    # "person_walking" together -- both spelled "a person" -- lets only one of them ever
    # win a box and records the other as wrong-label by construction. The first run of this
    # tool did exactly that and reported 0/27 for person_sitting, which measured the phrasing
    # and not the model. Group them, query one representative per distinct phrase, and say
    # so in the report.
    groups = defaultdict(list)
    for c in classes:
        groups[prompts_used[c]].append(c)
    group_of = {c: tuple(sorted(g)) for g in groups.values() for c in g}
    merged = {g: p for p, g in ((p, tuple(sorted(g))) for p, g in groups.items())
              if len(g) > 1}
    if merged:
        print("\n  classes sharing a phrase -- scored as one, since nothing can separate them:")
        for g, phrase in sorted(merged.items()):
            print(f"    {phrase!r:<24} <- {', '.join(g)}")
    query_labels = [sorted(g)[0] for g in groups.values()]

    from detection.detector import OwlV2Detector
    from PIL import Image
    detector = OwlV2Detector(threshold=args.threshold)

    t0 = time.time()
    records = []
    for i, (img_path, ann) in enumerate(scenes, 1):
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        # Ask for every class in the catalog, not just the ones in this frame: a detector
        # that only ever sees the right answers would look better than the real pipeline,
        # which always queries labels that may not be present.
        dets = detector.detect(image, query_labels)
        gt = [x for x in ann["detections"] if x["class"] in group_of]
        records.extend(match_scene(gt, dets, args.iou, group_of))
        print(f"  [{i}/{len(scenes)}] {os.path.basename(os.path.dirname(img_path))}/"
              f"{os.path.basename(img_path)}: {len(ann['detections'])} GT, "
              f"{len(dets)} detections", flush=True)

    elapsed = time.time() - t0
    report(records, prompts_used, elapsed, args)

    if args.save:
        payload = {
            "scene_root": os.path.abspath(args.scene_root),
            "args": {k: v for k, v in vars(args).items() if k != "save"},
            "elapsed_s": round(elapsed, 1),
            "prompts": prompts_used,
            "boxes": [{"class": r["gt"]["class"], "kind": r["gt"]["kind"],
                       "bbox": r["gt"]["bbox"], "occlusion": r["gt"].get("occlusion"),
                       "hit": r["hit"], "conf": r["conf"], "iou": r["iou"],
                       "mislabelled_as": r["mislabelled_as"]} for r in records],
        }
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nsaved -> {args.save}")


if __name__ == "__main__":
    main()
