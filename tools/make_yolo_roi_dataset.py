#!/usr/bin/env python3
"""Build a landmark-centred YOLO ROI dataset from MARC dataset-gen output.

The full-frame detector already finds landmarks reliably, while some target objects occupy
only a few pixels.  This converter creates the distribution needed by a second detector:
small landmark-centred crops which Ultralytics later enlarges to its training ``imgsz``.
The crop is never centred on the target because the target location is unknown at runtime.

Positive crops are built by pairing each requested target with the nearest landmark whose
centred ROI can retain the target.  Target-free landmark crops form a deterministic negative
pool.  All visible classes inside a selected crop keep their labels, so an unrequested object
is not accidentally taught as background.

Examples (``src`` must contain ``<scenario>/scene_*``):

    python tools/make_yolo_roi_dataset.py ../trainer_output ../yolo_dataset_roi
    python tools/make_yolo_roi_dataset.py ../trainer_output --dry-run
    python tools/make_yolo_roi_dataset.py ../trainer_output ../yolo_dataset_roi \
        --target-classes tumbler,umbrella,pencilcase --crop-sizes 384,512,768

R03 CCTV dumps are evaluation data, not input to this tool.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.make_yolo_dataset import (  # noqa: E402
    append_classes,
    find_labels,
    is_val,
    load_classes,
)

DEFAULT_TARGET_CLASSES = ("tumbler", "umbrella", "pencilcase")
DEFAULT_CROP_SIZES = (384, 512, 768)


@dataclass(frozen=True)
class CropSpec:
    """One source-image crop, before labels are clipped into it."""

    json_path: str
    image_path: str
    scene_key: str
    source_stem: str
    anchor_class: str
    anchor_instance: str
    bounds: tuple  # (left, top, right, bottom), integer source pixels
    requested_positive: bool
    requested_target_classes: tuple = ()

    @property
    def key(self):
        left, top, right, bottom = self.bounds
        return (self.json_path, left, top, right, bottom)

    @property
    def stable_key(self):
        left, top, right, bottom = self.bounds
        return (f"{self.scene_key}/{self.source_stem}/{self.anchor_instance}/"
                f"{left}_{top}_{right}_{bottom}")


def parse_csv(value, cast=str):
    values = []
    for item in value.split(","):
        item = item.strip()
        if item:
            values.append(cast(item))
    return values


def ordered_box(box):
    x0, y0, x1, y1 = (float(v) for v in box)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def box_center(box):
    x0, y0, x1, y1 = ordered_box(box)
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def trainable_box(box):
    """Match YOLO conversion's minimum: both pixel dimensions must exceed one."""
    try:
        x0, y0, x1, y1 = ordered_box(box)
    except (TypeError, ValueError):
        return False
    return x1 - x0 > 1.0 and y1 - y0 > 1.0


def square_crop(center_x, center_y, size, image_width, image_height):
    """A fixed-size crop clamped to the frame while preserving its size when possible."""
    width = min(int(size), int(image_width))
    height = min(int(size), int(image_height))
    left = int(round(float(center_x) - width / 2.0))
    top = int(round(float(center_y) - height / 2.0))
    left = max(0, min(left, int(image_width) - width))
    top = max(0, min(top, int(image_height) - height))
    return left, top, left + width, top + height


def clipped_box(box, crop_bounds, min_area_ratio=0.5):
    """Return crop-local xyxy if enough of the source box remains, otherwise ``None``."""
    x0, y0, x1, y1 = ordered_box(box)
    left, top, right, bottom = crop_bounds
    original_area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if original_area <= 1.0:
        return None
    ix0, iy0 = max(x0, left), max(y0, top)
    ix1, iy1 = min(x1, right), min(y1, bottom)
    kept_area = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if kept_area <= 1.0 or kept_area / original_area < min_area_ratio:
        return None
    return ix0 - left, iy0 - top, ix1 - left, iy1 - top


def measured_occlusion_too_high(det, max_occlusion):
    if max_occlusion is None:
        return False
    occ = det.get("occlusion")
    return occ is not None and occ >= 0 and occ > max_occlusion


def choose_positive_crop(target, landmarks, crop_sizes, width, height,
                         min_area_ratio):
    """Nearest landmark + smallest centred crop retaining the target."""
    tx, ty = box_center(target["bbox"])
    ranked = sorted(
        landmarks,
        key=lambda lm: ((box_center(lm["bbox"])[0] - tx) ** 2 +
                        (box_center(lm["bbox"])[1] - ty) ** 2,
                        lm.get("instance_id") or lm.get("class") or ""),
    )
    for landmark in ranked:
        lx, ly = box_center(landmark["bbox"])
        for size in crop_sizes:
            bounds = square_crop(lx, ly, size, width, height)
            if clipped_box(target["bbox"], bounds, min_area_ratio) is not None:
                return landmark, bounds
    return None, None


def choose_negative_crop(landmark, crop_sizes, width, height, context_scale):
    """One adaptive landmark crop for the negative candidate pool."""
    x0, y0, x1, y1 = ordered_box(landmark["bbox"])
    wanted = context_scale * max(x1 - x0, y1 - y0)
    size = next((value for value in crop_sizes if value >= wanted), crop_sizes[-1])
    return square_crop(*box_center(landmark["bbox"]), size, width, height)


def labels_for_crop(detections, crop_bounds, index_of, target_classes,
                    min_area_ratio, max_occlusion, stats=None):
    """Clip all source detections into an ROI and produce YOLO label lines."""
    left, top, right, bottom = crop_bounds
    crop_width, crop_height = right - left, bottom - top
    lines = []
    kept = []
    for det in detections:
        name = det.get("class")
        idx = index_of.get(name)
        if idx is None:
            if stats is not None:
                stats["unknown_class"] += 1
            continue
        if measured_occlusion_too_high(det, max_occlusion):
            if stats is not None:
                stats["dropped_occluded"] += 1
            continue
        local = clipped_box(det.get("bbox", ()), crop_bounds, min_area_ratio)
        if local is None:
            continue
        x0, y0, x1, y1 = local
        bw, bh = x1 - x0, y1 - y0
        if bw <= 1.0 or bh <= 1.0:
            continue
        lines.append(
            f"{idx} {(x0 + x1) / 2 / crop_width:.6f} "
            f"{(y0 + y1) / 2 / crop_height:.6f} "
            f"{bw / crop_width:.6f} {bh / crop_height:.6f}"
        )
        kept.append((name, local))
        if stats is not None:
            stats["classes"][name] += 1
            if name in target_classes:
                stats["target_boxes"][name] += 1
                long_side = max(bw, bh)
                scaled = long_side * 1280.0 / max(crop_width, crop_height)
                stats["target_long_side_source"].append(long_side)
                stats["target_long_side_1280"].append(scaled)
    return lines, kept


def _spec(json_path, image_path, scene_key, landmark, bounds, positive,
          target_class=None):
    return CropSpec(
        json_path=json_path,
        image_path=image_path,
        scene_key=scene_key,
        source_stem=os.path.splitext(os.path.basename(json_path))[0],
        anchor_class=landmark.get("class") or "landmark",
        anchor_instance=landmark.get("instance_id") or landmark.get("class") or "landmark",
        bounds=bounds,
        requested_positive=positive,
        requested_target_classes=((target_class,) if target_class else ()),
    )


def discover_crop_specs(label_files, target_classes, crop_sizes, min_area_ratio,
                        max_occlusion, context_scale):
    """Return deduplicated positive specs, negative candidates, and discovery stats."""
    positives = {}
    negatives = {}
    stats = Counter()
    for json_path, image_path, scene_key in label_files:
        with open(json_path, encoding="utf-8") as handle:
            data = json.load(handle)
        info = data.get("image") or {}
        width, height = info.get("width"), info.get("height")
        if not width or not height:
            raise ValueError(f"{json_path} has no image.width/height")
        detections = data.get("detections") or []
        landmarks = [det for det in detections if det.get("kind") == "landmark" and
                     trainable_box(det.get("bbox")) and
                     not measured_occlusion_too_high(det, max_occlusion)]
        raw_targets = [det for det in detections if det.get("class") in target_classes and
                       not measured_occlusion_too_high(det, max_occlusion)]
        targets = [det for det in raw_targets if trainable_box(det.get("bbox"))]
        stats["source_images"] += 1
        stats["source_targets"] += len(targets)
        stats["degenerate_source_targets"] += len(raw_targets) - len(targets)
        if not landmarks:
            stats["images_without_landmark"] += 1
            stats["uncovered_targets"] += len(targets)
            continue

        used_landmarks = set()
        for target in targets:
            landmark, bounds = choose_positive_crop(
                target, landmarks, crop_sizes, width, height, min_area_ratio)
            if landmark is None:
                stats["uncovered_targets"] += 1
                continue
            spec = _spec(
                json_path, image_path, scene_key, landmark, bounds, True,
                target.get("class"))
            previous = positives.get(spec.key)
            if previous is not None:
                merged = tuple(sorted(set(previous.requested_target_classes) |
                                      set(spec.requested_target_classes)))
                spec = replace(previous, requested_target_classes=merged)
            positives[spec.key] = spec
            used_landmarks.add(landmark.get("instance_id") or id(landmark))

        # A crop is only a negative if no selected target survives inside it.  This check
        # also handles a landmark not paired above whose ROI happens to contain a target.
        for landmark in landmarks:
            identity = landmark.get("instance_id") or id(landmark)
            if identity in used_landmarks:
                continue
            bounds = choose_negative_crop(
                landmark, crop_sizes, width, height, context_scale)
            contains_target = any(
                clipped_box(target["bbox"], bounds, min_area_ratio) is not None
                for target in targets
            )
            if contains_target:
                continue
            spec = _spec(json_path, image_path, scene_key, landmark, bounds, False)
            negatives[spec.key] = spec
    return list(positives.values()), list(negatives.values()), stats


def sample_positives(positives, max_per_class):
    """Deterministically cap requested positive crops per class and split.

    The stable train/val scene split is preserved.  Eighty percent of each class cap is
    reserved for train and the remainder for val.  A crop requested by multiple target
    classes is selected when any associated class still has capacity and then counts for
    each associated class.  This keeps multi-target crops instead of duplicating images.
    """
    if max_per_class is None:
        return positives
    split_caps = {
        "train": int(math.ceil(max_per_class * 0.8)),
        "val": max_per_class - int(math.ceil(max_per_class * 0.8)),
    }
    selected = []
    counts = Counter()
    ordered = sorted(
        positives,
        key=lambda spec: hashlib.md5(spec.stable_key.encode("utf-8")).hexdigest())
    for spec in ordered:
        split = "val" if is_val(spec.scene_key) else "train"
        classes = spec.requested_target_classes
        if not classes:
            continue
        if not any(counts[(split, name)] < split_caps[split] for name in classes):
            continue
        selected.append(spec)
        for name in classes:
            counts[(split, name)] += 1
    return selected


def sample_negatives(positives, negatives, ratio):
    """Deterministically cap negatives per split, preserving scene-level separation."""
    selected = []
    for split in ("train", "val"):
        pos_count = sum(1 for spec in positives
                        if ("val" if is_val(spec.scene_key) else "train") == split)
        candidates = [spec for spec in negatives
                      if ("val" if is_val(spec.scene_key) else "train") == split]
        limit = int(math.ceil(pos_count * ratio))
        candidates.sort(
            key=lambda spec: hashlib.md5(spec.stable_key.encode("utf-8")).hexdigest())
        selected.extend(candidates[:limit])
    return selected


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower, upper = int(math.floor(index)), int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def safe_name(spec, ordinal):
    digest = hashlib.md5(spec.stable_key.encode("utf-8")).hexdigest()[:10]
    scene = spec.scene_key.replace("/", "_").replace("\\", "_")
    anchor = spec.anchor_instance.replace("/", "_").replace("\\", "_")
    return f"{scene}_{spec.source_stem}_{anchor}_{ordinal:05d}_{digest}"


def write_data_yaml(dst, names):
    with open(os.path.join(dst, "data.yaml"), "w", encoding="utf-8") as handle:
        handle.write("# generated by tools/make_yolo_roi_dataset.py -- do not hand-edit\n")
        handle.write("# class order comes from detection/classes.txt\n")
        handle.write(f"path: {os.path.abspath(dst)}\n")
        handle.write("train: images/train\nval: images/val\n")
        handle.write(f"nc: {len(names)}\nnames:\n")
        for idx, name in enumerate(names):
            handle.write(f"  {idx}: {name}\n")


def ensure_fresh_destination(parser, dst):
    """Never mix crops from runs with different sizes, classes, or sampling options."""
    if not os.path.exists(dst):
        return
    if not os.path.isdir(dst):
        parser.error(f"destination exists and is not a directory: {dst}")
    if os.listdir(dst):
        parser.error(
            f"destination is not empty: {dst}\n"
            "Use a new directory (recommended) or remove the old generated dataset first."
        )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="trainer_output root containing <scenario>/scene_*")
    parser.add_argument("dst", nargs="?", help="output dataset (omit with --dry-run)")
    parser.add_argument("--target-classes", default=",".join(DEFAULT_TARGET_CLASSES))
    parser.add_argument("--crop-sizes", default=",".join(map(str, DEFAULT_CROP_SIZES)))
    parser.add_argument("--negative-ratio", type=float, default=1.0,
                        help="maximum target-free landmark crops per positive crop, per split")
    parser.add_argument("--max-positive-per-class", type=int,
                        help="deterministic target-request crop cap per class (80%% train, 20%% val)")
    parser.add_argument("--min-area-ratio", type=float, default=0.5,
                        help="minimum fraction of a bbox that must remain inside an ROI")
    parser.add_argument("--max-occlusion", type=float, default=0.8,
                        help="drop boxes with measured occlusion above this; -1 stays unknown")
    parser.add_argument("--context-scale", type=float, default=2.0,
                        help="negative ROI size relative to the landmark's longer side")
    parser.add_argument("--format", choices=("png", "jpg"), default="png")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.dry_run and not args.dst:
        parser.error("dst is required unless --dry-run is given")
    if not args.dry_run:
        ensure_fresh_destination(parser, args.dst)
    if args.negative_ratio < 0:
        parser.error("--negative-ratio must be >= 0")
    if args.max_positive_per_class is not None and args.max_positive_per_class <= 0:
        parser.error("--max-positive-per-class must be > 0")
    if not 0 < args.min_area_ratio <= 1:
        parser.error("--min-area-ratio must be in (0, 1]")
    if not 0 <= args.max_occlusion <= 1:
        parser.error("--max-occlusion must be in [0, 1]")
    if args.context_scale <= 0:
        parser.error("--context-scale must be > 0")

    target_classes = tuple(dict.fromkeys(parse_csv(args.target_classes)))
    crop_sizes = tuple(sorted(set(parse_csv(args.crop_sizes, int))))
    if not target_classes:
        parser.error("--target-classes cannot be empty")
    if not crop_sizes or any(size <= 1 for size in crop_sizes):
        parser.error("--crop-sizes must contain positive integers")

    label_files = find_labels(args.src)
    if not label_files:
        parser.error("no matching <scenario>/scene_* image+JSON pairs found")

    names = load_classes()
    index_of = {name: idx for idx, name in enumerate(names)}
    seen = set()
    for json_path, _, _ in label_files:
        with open(json_path, encoding="utf-8") as handle:
            seen.update(det.get("class") for det in
                        (json.load(handle).get("detections") or []))
    unseen = sorted(name for name in seen if name and name not in index_of)
    if unseen:
        print(f"[classes] not in detection/classes.txt: {', '.join(unseen)}")
        if args.dry_run:
            print("[classes] --dry-run: not appending")
        else:
            append_classes(unseen)
            for name in unseen:
                index_of[name] = len(names)
                names.append(name)
                print(f"[classes] appended {name!r} -> index {index_of[name]}")

    missing_targets = [name for name in target_classes if name not in index_of]
    if missing_targets:
        parser.error(f"target classes absent from class table: {', '.join(missing_targets)}")

    discovered_positives, negative_pool, discovery = discover_crop_specs(
        label_files, set(target_classes), crop_sizes, args.min_area_ratio,
        args.max_occlusion, args.context_scale)
    positives = sample_positives(
        discovered_positives, args.max_positive_per_class)
    negatives = sample_negatives(positives, negative_pool, args.negative_ratio)
    specs = sorted(positives + negatives,
                   key=lambda spec: (spec.scene_key, spec.source_stem, spec.stable_key))

    output_stats = {
        "classes": Counter(), "target_boxes": Counter(), "unknown_class": 0,
        "dropped_occluded": 0, "target_long_side_source": [],
        "target_long_side_1280": [],
    }
    split_counts = Counter()
    positive_counts = Counter()
    manifest = []
    image_cache = {}

    if not args.dry_run:
        from PIL import Image
        for split in ("train", "val"):
            os.makedirs(os.path.join(args.dst, "images", split), exist_ok=True)
            os.makedirs(os.path.join(args.dst, "labels", split), exist_ok=True)

    for ordinal, spec in enumerate(specs):
        with open(spec.json_path, encoding="utf-8") as handle:
            data = json.load(handle)
        lines, kept = labels_for_crop(
            data.get("detections") or [], spec.bounds, index_of, set(target_classes),
            args.min_area_ratio, args.max_occlusion, output_stats)
        present_targets = sorted({name for name, _ in kept if name in target_classes})
        is_positive = bool(present_targets)
        split = "val" if is_val(spec.scene_key) else "train"
        split_counts[split] += 1
        positive_counts[(split, is_positive)] += 1
        stem = safe_name(spec, ordinal)
        manifest.append({
            "file": f"images/{split}/{stem}.{args.format}",
            "source_image": os.path.abspath(spec.image_path),
            "source_json": os.path.abspath(spec.json_path),
            "scene_key": spec.scene_key,
            "split": split,
            "anchor_class": spec.anchor_class,
            "anchor_instance": spec.anchor_instance,
            "crop_xyxy": list(spec.bounds),
            "positive": is_positive,
            "target_classes": present_targets,
        })
        if args.dry_run:
            continue

        image = image_cache.get(spec.image_path)
        if image is None:
            image = Image.open(spec.image_path).convert("RGB")
            image_cache.clear()  # source specs are sorted, so one decoded frame is enough
            image_cache[spec.image_path] = image
        crop = image.crop(spec.bounds)
        image_path = os.path.join(args.dst, "images", split, f"{stem}.{args.format}")
        if args.format == "jpg":
            crop.save(image_path, quality=args.jpeg_quality, subsampling=0)
        else:
            crop.save(image_path)
        with open(os.path.join(args.dst, "labels", split, stem + ".txt"),
                  "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + ("\n" if lines else ""))

    if not args.dry_run:
        write_data_yaml(args.dst, names)
        with open(os.path.join(args.dst, "roi_manifest.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print(f"{'DRY RUN -- ' if args.dry_run else ''}YOLO LANDMARK ROI DATASET")
    print("=" * 72)
    print(f"  source images       : {discovery['source_images']}")
    print(f"  trainable targets   : {discovery['source_targets']}")
    print(f"  degenerate targets  : {discovery['degenerate_source_targets']}")
    print(f"  uncovered targets   : {discovery['uncovered_targets']}")
    print(f"  positive candidates : {len(discovered_positives)}")
    print(f"  selected positives  : {len(positives)}")
    print(f"  negative candidates : {len(negative_pool)}")
    print(f"  sampled negatives   : {len(negatives)}  (ratio <= {args.negative_ratio:g})")
    print(f"  output ROI          : {split_counts['train']} train / {split_counts['val']} val")
    print("  actual positives    : "
          f"{positive_counts[('train', True)]} train / "
          f"{positive_counts[('val', True)]} val")
    print("  actual negatives    : "
          f"{positive_counts[('train', False)]} train / "
          f"{positive_counts[('val', False)]} val")

    print("\n  target label instances (duplicates across ROI are possible):")
    for name in target_classes:
        print(f"    {name:<18} {output_stats['target_boxes'][name]:6d}")
    source_sizes = output_stats["target_long_side_source"]
    scaled_sizes = output_stats["target_long_side_1280"]
    if source_sizes:
        print("\n  target longer side (px, p10 / median / p90):")
        print("    source ROI : " + " / ".join(
            f"{percentile(source_sizes, q):.1f}" for q in (0.1, 0.5, 0.9)))
        print("    at imgsz=1280: " + " / ".join(
            f"{percentile(scaled_sizes, q):.1f}" for q in (0.1, 0.5, 0.9)))
    if output_stats["dropped_occluded"]:
        print(f"  measured-occlusion boxes dropped: {output_stats['dropped_occluded']}")
    if output_stats["unknown_class"]:
        print(f"  unknown-class boxes dropped: {output_stats['unknown_class']}")
    if not args.dry_run:
        print(f"\n  wrote -> {os.path.abspath(args.dst)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
