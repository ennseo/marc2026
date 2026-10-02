#!/usr/bin/env python3
"""Apply a restricted pose-coordinate policy to a saved detection evaluation."""

import argparse
import copy
import json
import math
import statistics


def dist3(a, b):
    if a is None or b is None:
        return None
    return math.dist(tuple(map(float, a)), tuple(map(float, b)))


def weighted_total(comp, weights):
    active = sum(float(weights.get(name, 0.0)) for name in comp)
    if active <= 0:
        return 0.0
    earned = sum(float(weights.get(name, 0.0)) * float(value)
                 for name, value in comp.items())
    return 100.0 * earned / active


def score_at_threshold(records, threshold):
    totals = []
    for record in records:
        comp = dict(record["comp"])
        comp["anchor"] = float(record["anchor_err"] is not None and
                               record["anchor_err"] <= threshold)
        comp["target"] = float(record["target_err"] is not None and
                               record["target_err"] <= threshold)
        totals.append(weighted_total(comp, record["weights"]))
    return statistics.fmean(totals) if totals else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_json")
    parser.add_argument("pose_json")
    parser.add_argument("--classes", default="person_lying_down")
    parser.add_argument("--point", default="foot_out_50")
    parser.add_argument(
        "--class-points",
        help="comma-separated class=point overrides, e.g. person_prone=ankle",
    )
    parser.add_argument("--min-ankle-conf", type=float, default=0.25)
    parser.add_argument("--save", required=True)
    args = parser.parse_args()

    with open(args.eval_json, encoding="utf-8") as handle:
        evaluation = json.load(handle)
    with open(args.pose_json, encoding="utf-8") as handle:
        pose = json.load(handle)

    allowed = {value.strip() for value in args.classes.split(",") if value.strip()}
    class_points = {}
    if args.class_points:
        for item in args.class_points.split(","):
            if "=" not in item:
                parser.error("--class-points entries must be class=point")
            target_class, point_name = item.split("=", 1)
            class_points[target_class.strip()] = point_name.strip()
        allowed.update(class_points)
    pose_by_round = {row["round"]: row for row in pose["rounds"]}
    output = copy.deepcopy(evaluation)
    changed = []

    for record in output["rounds_detail"]:
        pose_row = pose_by_round.get(record["dir"])
        if not pose_row or pose_row.get("status") != "ok":
            continue
        if pose_row.get("class") not in allowed:
            continue
        ankle_conf = pose_row.get("ankle_conf") or []
        if len(ankle_conf) != 2 or min(ankle_conf) < args.min_ankle_conf:
            continue
        point_name = class_points.get(pose_row.get("class"), args.point)
        point = (pose_row.get("points") or {}).get(point_name) or {}
        world = point.get("world")
        if world is None:
            continue

        before = record["target_err"]
        record["submitted"]["target_coord"] = world
        record["target_err"] = dist3(world, record["expected"]["target_coord"])
        record["target_err_xy"] = math.dist(world[:2], record["expected"]["target_coord"][:2])
        record["comp"]["target"] = float(record["target_err"] <= 1.0)
        record["total"] = weighted_total(record["comp"], record["weights"])
        record["pose_policy"] = {
            "point": point_name,
            "ankle_conf": ankle_conf,
            "before_error_m": before,
            "after_error_m": record["target_err"],
        }
        changed.append({"round": record["dir"], "before": before,
                        "after": record["target_err"]})

    records = output["rounds_detail"]
    output["pose_policy"] = {
        "classes": sorted(allowed),
        "point": args.point,
        "class_points": class_points,
        "min_ankle_conf": args.min_ankle_conf,
        "changed": changed,
        "scores": {str(value): score_at_threshold(records, value)
                   for value in (0.5, 1.0, 2.0, 3.0)},
    }
    output["summary"]["pose_policy_target_hits"] = {
        str(value): sum(record["target_err"] is not None and record["target_err"] <= value
                        for record in records)
        for value in (0.5, 1.0, 2.0, 3.0)
    }

    print("changed:")
    for item in changed:
        print("  %s %.3fm -> %.3fm" %
              (item["round"], item["before"], item["after"]))
    print("target hits:", output["summary"]["pose_policy_target_hits"])
    print("scores:", output["pose_policy"]["scores"])
    with open(args.save, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print("saved ->", args.save)


if __name__ == "__main__":
    main()
