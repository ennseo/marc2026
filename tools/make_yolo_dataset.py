#!/usr/bin/env python3
"""trainer_output -> YOLO dataset (은서 담당).

`marc.sh dataset-gen` writes the organisers' own label format: one JSON per camera per
scene, with pixel boxes. Ultralytics wants a mirrored images/ + labels/ tree and one
normalised `.txt` per image. This converts between them. It is pure arithmetic -- the
PNGs are never decoded, because each label JSON already carries `image.width/height` --
so it costs seconds even on a few thousand scenes.

    python tools/make_yolo_dataset.py ../trainer_output ../yolo_dataset
    python tools/make_yolo_dataset.py ../trainer_output ../yolo_dataset --copy
    python tools/make_yolo_dataset.py ../trainer_output ../yolo_dataset --max-occlusion 0.8
    python tools/make_yolo_dataset.py ../trainer_output --dry-run     # just report, write nothing

Two things this deliberately does NOT leave to chance:

**Class indices come from detection/classes.txt, not from whatever this run happens to
see.** Deriving indices from the data (e.g. `sorted(set(classes))`) looks tidy and is a
trap: generate more scenes, a new class appears, every index after it shifts, and a model
trained on the old indices keeps loading and running while quietly predicting the wrong
labels. Unseen classes are appended to the end of that file instead, which cannot move an
existing index.

**The train/val split is by scene and by content hash**, not by position in the file
listing. So adding scenes later leaves the existing split exactly where it was -- a scene
that was in val stays in val, and no val image ever leaks into train.
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES_FILE = os.path.join(os.path.dirname(HERE), "detection", "classes.txt")

VAL_FRACTION = 0.2      # ~1 scene in 5 held out


# -- class table -----------------------------------------------------------

def load_classes(path=CLASSES_FILE):
    """Ordered class names. Line order is the class index; comments/blanks skipped."""
    names = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                names.append(line)
    dupes = [n for n, c in Counter(names).items() if c > 1]
    if dupes:
        sys.exit(f"detection/classes.txt has duplicate entries: {dupes}\n"
                 "Indices would be ambiguous. Remove the duplicates before continuing.")
    return names


def append_classes(new_names, path=CLASSES_FILE):
    """Add unseen classes at the end, preserving every existing index."""
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n# -- appended automatically by tools/make_yolo_dataset.py --\n")
        for n in new_names:
            f.write(n + "\n")


# -- scene discovery -------------------------------------------------------

def find_labels(src):
    """Every label JSON that has a matching image, as (json_path, image_path, scene_key)."""
    out = []
    for jp in sorted(glob.glob(os.path.join(src, "*", "scene_*", "*.json"))):
        stem = os.path.splitext(jp)[0]
        img = next((stem + e for e in (".png", ".jpg", ".jpeg")
                    if os.path.exists(stem + e)), None)
        if img is None:
            continue
        scene_dir = os.path.basename(os.path.dirname(jp))
        scenario = os.path.basename(os.path.dirname(os.path.dirname(jp)))
        out.append((jp, img, f"{scenario}/{scene_dir}"))
    return out


def is_val(scene_key):
    """Stable split: depends only on the scene name, so it never moves as data grows."""
    h = hashlib.md5(scene_key.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) % 1000) < VAL_FRACTION * 1000


# -- conversion ------------------------------------------------------------

def convert_boxes(det_list, width, height, index_of, max_occlusion, stats):
    """Pixel [x0,y0,x1,y1] -> normalised 'idx cx cy w h' lines."""
    lines = []
    for d in det_list:
        name = d.get("class")
        idx = index_of.get(name)
        if idx is None:
            stats["unknown_class"] += 1
            continue

        occ = d.get("occlusion")
        # -1.0 is the generator's "not measured", which is not the same as "fully hidden";
        # only filter on a real measurement.
        if max_occlusion is not None and occ is not None and occ >= 0 and occ > max_occlusion:
            stats["dropped_occluded"] += 1
            continue

        x0, y0, x1, y1 = (float(v) for v in d["bbox"])
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        # Boxes routinely touch or cross the frame edge; clamp rather than discard, since
        # a partially visible landmark is still the right thing to learn.
        x0, x1 = max(0.0, min(x0, width)), max(0.0, min(x1, width))
        y0, y1 = max(0.0, min(y0, height)), max(0.0, min(y1, height))
        bw, bh = x1 - x0, y1 - y0
        if bw <= 1.0 or bh <= 1.0:
            stats["dropped_degenerate"] += 1
            continue

        lines.append(f"{idx} {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} "
                     f"{bw / width:.6f} {bh / height:.6f}")
        stats["classes"][name] += 1
    return lines


def place_image(src_img, dst_img, copy):
    """Hard-link by default -- same bytes, no extra disk, and no admin rights needed
    (symlinks on Windows do). Falls back to copying across volumes."""
    if os.path.exists(dst_img):
        os.remove(dst_img)
    if not copy:
        try:
            os.link(src_img, dst_img)
            return
        except OSError:
            pass
    shutil.copy2(src_img, dst_img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="trainer_output directory")
    ap.add_argument("dst", nargs="?", help="output dataset directory (omit with --dry-run)")
    ap.add_argument("--copy", action="store_true",
                    help="copy images instead of hard-linking them")
    ap.add_argument("--max-occlusion", type=float, default=None,
                    help="drop boxes whose measured occlusion exceeds this (0-1). "
                         "Unmeasured boxes (-1.0) are always kept.")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be written, touch nothing")
    args = ap.parse_args()

    if not args.dry_run and not args.dst:
        ap.error("dst is required unless --dry-run is given")

    labels = find_labels(args.src)
    if not labels:
        sys.exit(f"no scene_*/ label JSON with a matching image under {args.src}\n"
                 "Expected trainer_output/<scenario>/scene_00000/<camera>.json + .png")

    names = load_classes()
    index_of = {n: i for i, n in enumerate(names)}

    # A class the table has never seen would otherwise be silently dropped, so find them
    # all up front and append before converting anything.
    seen = set()
    for jp, _, _ in labels:
        with open(jp, encoding="utf-8") as f:
            for d in json.load(f).get("detections", []):
                seen.add(d.get("class"))
    unseen = sorted(n for n in seen if n and n not in index_of)
    if unseen:
        print(f"[classes] {len(unseen)} class(es) not in detection/classes.txt: "
              f"{', '.join(unseen)}")
        if args.dry_run:
            print("[classes] --dry-run: not appending.")
        else:
            append_classes(unseen)
            for n in unseen:
                index_of[n] = len(names)
                names.append(n)
                print(f"[classes] appended {n!r} -> index {index_of[n]}")
            print("[classes] existing indices are unchanged; retrain from scratch because "
                  "the detector head now has more outputs.")

    stats = {"classes": Counter(), "unknown_class": 0,
             "dropped_occluded": 0, "dropped_degenerate": 0}
    counts = {"train": 0, "val": 0}
    empty = 0

    if not args.dry_run:
        for split in ("train", "val"):
            for kind in ("images", "labels"):
                os.makedirs(os.path.join(args.dst, kind, split), exist_ok=True)

    for jp, img, scene_key in labels:
        with open(jp, encoding="utf-8") as f:
            data = json.load(f)
        info = data.get("image") or {}
        w, h = info.get("width"), info.get("height")
        if not w or not h:
            sys.exit(f"{jp} has no image.width/height -- cannot normalise boxes")

        lines = convert_boxes(data.get("detections", []), w, h,
                              index_of, args.max_occlusion, stats)
        if not lines:
            # Ultralytics treats an image with an empty label file as a background sample,
            # which is useful, not broken -- keep it and just report the count.
            empty += 1
        split = "val" if is_val(scene_key) else "train"
        counts[split] += 1

        if args.dry_run:
            continue
        stem = f"{scene_key.replace('/', '_')}_{os.path.splitext(os.path.basename(jp))[0]}"
        place_image(img, os.path.join(args.dst, "images", split,
                                      stem + os.path.splitext(img)[1]), args.copy)
        with open(os.path.join(args.dst, "labels", split, stem + ".txt"),
                  "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    if not args.dry_run:
        with open(os.path.join(args.dst, "data.yaml"), "w", encoding="utf-8") as f:
            f.write("# generated by tools/make_yolo_dataset.py -- do not hand-edit;\n"
                    "# class order comes from detection/classes.txt\n")
            f.write(f"path: {os.path.abspath(args.dst)}\n")
            f.write("train: images/train\nval: images/val\n")
            f.write(f"nc: {len(names)}\nnames:\n")
            for i, n in enumerate(names):
                f.write(f"  {i}: {n}\n")

    # -- report --
    total_boxes = sum(stats["classes"].values())
    print("\n" + "=" * 66)
    print(f"{'DRY RUN -- ' if args.dry_run else ''}YOLO DATASET")
    print("=" * 66)
    print(f"  images        : {counts['train']} train / {counts['val']} val"
          f"   ({empty} with no boxes)")
    print(f"  boxes         : {total_boxes}")
    print(f"  classes       : {len(stats['classes'])} present / {len(names)} defined")
    if stats["unknown_class"]:
        print(f"  ⚠ unknown-class boxes dropped : {stats['unknown_class']}")
    if stats["dropped_occluded"]:
        print(f"  occluded boxes dropped        : {stats['dropped_occluded']}")
    if stats["dropped_degenerate"]:
        print(f"  degenerate boxes dropped      : {stats['dropped_degenerate']}")

    print("\n  per class (idx  name  boxes):")
    for i, n in enumerate(names):
        c = stats["classes"].get(n, 0)
        flag = "  <-- none" if c == 0 else ("  <-- thin" if c < 50 else "")
        print(f"    {i:3d}  {n:<22}{c:6d}{flag}")

    thin = [n for n in names if 0 < stats["classes"].get(n, 0) < 50]
    missing = [n for n in names if stats["classes"].get(n, 0) == 0]
    if thin or missing:
        print("\n  'thin' = under 50 boxes, too few to learn from. 'none' = absent here.")
        print("  Both are a signal to generate more scenes, not to edit classes.txt.")
    if not args.dry_run:
        print(f"\n  wrote -> {os.path.abspath(args.dst)}")


if __name__ == "__main__":
    main()
