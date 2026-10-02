#!/usr/bin/env python3
"""Compare bbox reference pixels for person world-coordinate grounding.

Consumes a JSON produced by ``tools/eval_detection.py`` with ``vision_trace``
enabled.  Only person rounds whose selected camera matches the grader camera are
used, so camera/instance-selection failures do not contaminate pixel analysis.
"""

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry.transform import project_to_pixel, quat_to_matrix, unproject_to_plane


CANDIDATES = {
    "bottom_center": (0.50, 1.00),
    "bottom_90": (0.50, 0.90),
    "bottom_80": (0.50, 0.80),
    "center": (0.50, 0.50),
    "left_90": (0.10, 0.90),
    "left_75": (0.25, 0.75),
    "right_75": (0.75, 0.75),
    "right_90": (0.90, 0.90),
}


def camera_params(meta, camera_id):
    camera = meta["cameras"][camera_id]
    qx, qy, qz, qw = camera["tf"]["rotation_xyzw"]
    return (
        np.asarray(camera["K"], dtype=float).reshape(3, 3),
        quat_to_matrix(qx, qy, qz, qw),
        np.asarray(camera["tf"]["translation"], dtype=float),
    )


def bbox_pixel(bbox, normalized):
    x0, y0, x1, y1 = map(float, bbox)
    nx, ny = normalized
    return (x0 + nx * (x1 - x0), y0 + ny * (y1 - y0))


def dist3(a, b):
    return math.dist(tuple(map(float, a)), tuple(map(float, b)))


def summarize(rows, candidate_names):
    result = {}
    for name in candidate_names:
        errors = [row["candidate_errors"][name] for row in rows]
        result[name] = {
            "median_m": statistics.median(errors),
            "mean_m": statistics.fmean(errors),
            "within_1m": sum(error <= 1.0 for error in errors),
            "count": len(errors),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_json")
    parser.add_argument("--save", help="write detailed analysis JSON")
    args = parser.parse_args()

    with open(args.eval_json, encoding="utf-8") as handle:
        evaluation = json.load(handle)
    with open(os.path.join(evaluation["dump_dir"], "meta.json"), encoding="utf-8") as handle:
        meta = json.load(handle)

    rows = []
    excluded = []
    for record in evaluation["rounds_detail"]:
        expected = record["expected"]
        target_class = expected.get("target_type") or ""
        if not target_class.startswith("person"):
            continue
        trace = record.get("vision_trace") or {}
        bbox = trace.get("target_bbox")
        selected_camera = record["submitted"].get("camera_id")
        expected_camera = expected.get("camera_id")
        if not bbox or selected_camera != expected_camera:
            excluded.append({
                "round": record["dir"],
                "reason": "missing_bbox" if not bbox else "wrong_camera",
            })
            continue

        K, R, t = camera_params(meta, expected_camera)
        target_coord = expected["target_coord"]
        gt_pixel = project_to_pixel(target_coord, K, R, t)
        x0, y0, x1, y1 = map(float, bbox)
        width, height = x1 - x0, y1 - y0
        normalized_gt = (
            (gt_pixel[0] - x0) / width,
            (gt_pixel[1] - y0) / height,
        )
        # Preserve the exact plane selected by the existing pipeline (ground or a
        # landmark top). This isolates reference-pixel effects from plane policy.
        plane_z = float(record["submitted"]["target_coord"][2])
        candidate_errors = {}
        candidate_world = {}
        for name, normalized in CANDIDATES.items():
            u, v = bbox_pixel(bbox, normalized)
            world = unproject_to_plane(u, v, K, R, t, plane_z)
            candidate_world[name] = world
            candidate_errors[name] = dist3(world, target_coord) if world else None

        rows.append({
            "round": record["dir"],
            "class": target_class,
            "camera": expected_camera,
            "bbox": bbox,
            "detected_label": trace.get("detected_label"),
            "gt_pixel": gt_pixel,
            "gt_normalized": normalized_gt,
            "gt_inside_bbox": 0.0 <= normalized_gt[0] <= 1.0 and 0.0 <= normalized_gt[1] <= 1.0,
            "plane_z": plane_z,
            "candidate_errors": candidate_errors,
            "candidate_world": candidate_world,
        })

    by_class = defaultdict(list)
    for row in rows:
        by_class[row["class"]].append(row)

    fitted = {}
    for target_class, class_rows in sorted(by_class.items()):
        point = (
            statistics.median(row["gt_normalized"][0] for row in class_rows),
            statistics.median(row["gt_normalized"][1] for row in class_rows),
        )
        errors = []
        for row in class_rows:
            K, R, t = camera_params(meta, row["camera"])
            world = unproject_to_plane(
                *bbox_pixel(row["bbox"], point), K, R, t, row["plane_z"])
            expected = next(r["expected"]["target_coord"] for r in evaluation["rounds_detail"]
                            if r["dir"] == row["round"])
            errors.append(dist3(world, expected))
        fitted[target_class] = {
            "normalized_point": point,
            "median_m": statistics.median(errors),
            "mean_m": statistics.fmean(errors),
            "within_1m": sum(error <= 1.0 for error in errors),
            "count": len(errors),
            "note": "in-sample descriptive fit; not a held-out estimate",
        }

    payload = {
        "source": os.path.abspath(args.eval_json),
        "included": len(rows),
        "excluded": excluded,
        "candidate_points": CANDIDATES,
        "overall": summarize(rows, CANDIDATES),
        "by_class": {
            target_class: summarize(class_rows, CANDIDATES)
            for target_class, class_rows in sorted(by_class.items())
        },
        "fitted_class_points": fitted,
        "rounds": rows,
    }

    print(f"included={len(rows)} excluded={excluded}")
    print("\noverall")
    for name, stats in sorted(payload["overall"].items(), key=lambda item: item[1]["median_m"]):
        print(f"  {name:14s} median={stats['median_m']:.3f}m "
              f"mean={stats['mean_m']:.3f}m <=1m={stats['within_1m']}/{stats['count']}")
    print("\nGT normalized positions and in-sample class medians")
    for target_class, class_rows in sorted(by_class.items()):
        points = ", ".join(
            f"{row['round']}=({row['gt_normalized'][0]:.2f},{row['gt_normalized'][1]:.2f})"
            for row in class_rows)
        fit = fitted[target_class]
        print(f"  {target_class}: {points}")
        print(f"    median point=({fit['normalized_point'][0]:.2f},"
              f"{fit['normalized_point'][1]:.2f}) median={fit['median_m']:.3f}m "
              f"<=1m={fit['within_1m']}/{fit['count']}")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"\nsaved -> {args.save}")


if __name__ == "__main__":
    main()
