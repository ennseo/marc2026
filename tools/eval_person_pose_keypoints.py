#!/usr/bin/env python3
"""Evaluate pretrained pose keypoints as person grounding reference pixels."""

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry.transform import project_to_pixel, quat_to_matrix, unproject_to_plane


LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16


def camera_params(meta, camera_id):
    camera = meta["cameras"][camera_id]
    qx, qy, qz, qw = camera["tf"]["rotation_xyzw"]
    return (
        np.asarray(camera["K"], dtype=float).reshape(3, 3),
        quat_to_matrix(qx, qy, qz, qw),
        np.asarray(camera["tf"]["translation"], dtype=float),
    )


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = ((a[2] - a[0]) * (a[3] - a[1]) +
             (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def expanded_crop(box, width, height, scale):
    x0, y0, x1, y1 = map(float, box)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side_w, side_h = (x1 - x0) * scale, (y1 - y0) * scale
    left = max(0, int(round(cx - side_w / 2)))
    top = max(0, int(round(cy - side_h / 2)))
    right = min(width, int(round(cx + side_w / 2)))
    bottom = min(height, int(round(cy + side_h / 2)))
    return left, top, right, bottom


def confident_mean(xy, conf, indices, threshold):
    valid = [xy[index] for index in indices if conf[index] >= threshold]
    if not valid:
        return None
    return np.mean(np.asarray(valid, dtype=float), axis=0)


def world_error(pixel, params, plane_z, expected):
    if pixel is None:
        return None, None
    world = unproject_to_plane(float(pixel[0]), float(pixel[1]), *params, plane_z)
    return world, math.dist(world, expected) if world else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_json")
    parser.add_argument("--weights", default="yolo11n-pose.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--crop-scale", type=float, default=2.0)
    parser.add_argument("--keypoint-conf", type=float, default=0.25)
    parser.add_argument("--save", help="analysis JSON path")
    args = parser.parse_args()

    with open(args.eval_json, encoding="utf-8") as handle:
        evaluation = json.load(handle)
    dump_dir = Path(evaluation["dump_dir"])
    with open(dump_dir / "meta.json", encoding="utf-8") as handle:
        meta = json.load(handle)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    rows = []

    for record in evaluation["rounds_detail"]:
        expected = record["expected"]
        target_class = expected.get("target_type") or ""
        if not target_class.startswith("person"):
            continue
        trace = record.get("vision_trace") or {}
        bbox = trace.get("target_bbox")
        selected_camera = record["submitted"].get("camera_id")
        expected_camera = expected.get("camera_id")
        row = {"round": record["dir"], "class": target_class,
               "camera_correct": selected_camera == expected_camera}
        if not bbox or selected_camera != expected_camera:
            row["status"] = "missing_bbox" if not bbox else "wrong_camera"
            rows.append(row)
            continue

        frame_path = dump_dir / record["dir"] / f"{selected_camera}.png"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            row["status"] = "missing_frame"
            rows.append(row)
            continue
        height, width = frame.shape[:2]
        left, top, right, bottom = expanded_crop(
            bbox, width, height, args.crop_scale)
        crop = frame[top:bottom, left:right]
        results = model.predict(crop, imgsz=640, conf=0.05, device="cpu", verbose=False)
        result = results[0]

        best = None
        if result.boxes is not None and result.keypoints is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            for index, local_box in enumerate(boxes):
                global_box = [local_box[0] + left, local_box[1] + top,
                              local_box[2] + left, local_box[3] + top]
                overlap = iou(global_box, bbox)
                if best is None or overlap > best[0]:
                    best = (overlap, index, global_box)

        if best is None:
            row["status"] = "pose_not_found"
            rows.append(row)
            continue

        overlap, index, pose_box = best
        xy = result.keypoints.xy[index].cpu().numpy().astype(float)
        conf = result.keypoints.conf[index].cpu().numpy().astype(float)
        xy[:, 0] += left
        xy[:, 1] += top
        ankle = confident_mean(xy, conf, [LEFT_ANKLE, RIGHT_ANKLE], args.keypoint_conf)
        hip = confident_mean(xy, conf, [LEFT_HIP, RIGHT_HIP], args.keypoint_conf)

        knee = confident_mean(xy, conf, [LEFT_KNEE, RIGHT_KNEE], args.keypoint_conf)
        points = {"hip": hip, "ankle": ankle}
        if ankle is not None and hip is not None:
            axis = ankle - hip
            points["foot_out_25"] = ankle + 0.25 * axis
            points["foot_out_50"] = ankle + 0.50 * axis
        else:
            points["foot_out_25"] = None
            points["foot_out_50"] = None

        params = camera_params(meta, selected_camera)
        plane_z = float(record["submitted"]["target_coord"][2])
        expected_coord = expected["target_coord"]
        gt_pixel = project_to_pixel(expected_coord, *params)
        point_data = {}
        for name, pixel in points.items():
            world, error = world_error(pixel, params, plane_z, expected_coord)
            point_data[name] = {
                "pixel": pixel.tolist() if pixel is not None else None,
                "world": world,
                "error_m": error,
            }

        row.update({
            "status": "ok",
            "bbox": bbox,
            "pose_bbox": [float(value) for value in pose_box],
            "bbox_iou": float(overlap),
            "gt_pixel": gt_pixel,
            "hip_conf": [float(conf[LEFT_HIP]), float(conf[RIGHT_HIP])],
            "knee_conf": [float(conf[LEFT_KNEE]), float(conf[RIGHT_KNEE])],
            "ankle_conf": [float(conf[LEFT_ANKLE]), float(conf[RIGHT_ANKLE])],
            "hip_pixel": hip.tolist() if hip is not None else None,
            "knee_pixel": knee.tolist() if knee is not None else None,
            "ankle_pixel": ankle.tolist() if ankle is not None else None,
            "points": point_data,
            "baseline_error_m": record["target_err"],
        })
        rows.append(row)

        image = Image.open(frame_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle(tuple(bbox), outline=(255, 210, 0), width=4)
        draw.rectangle(tuple(pose_box), outline=(80, 255, 80), width=3)
        colors = {"hip": (80, 255, 80), "ankle": (0, 220, 255),
                  "foot_out_25": (255, 80, 255),
                  "foot_out_50": (255, 130, 40)}
        for name, pixel in points.items():
            if pixel is None:
                continue
            x, y = map(float, pixel)
            color = colors[name]
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=color, width=4)
            draw.text((x + 9, y), name, fill=color)
        if gt_pixel is not None:
            x, y = gt_pixel
            draw.line((x - 12, y, x + 12, y), fill=(255, 30, 30), width=4)
            draw.line((x, y - 12, x, y + 12), fill=(255, 30, 30), width=4)
        image.save(output / f"{record['dir']}_{target_class}.png")

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    summary = {}
    for name in ("hip", "ankle", "foot_out_25", "foot_out_50"):
        errors = [row["points"][name]["error_m"] for row in ok_rows
                  if row["points"][name]["error_m"] is not None]
        summary[name] = {
            "available": len(errors),
            "median_m": statistics.median(errors) if errors else None,
            "mean_m": statistics.fmean(errors) if errors else None,
            "within_1m": sum(error <= 1.0 for error in errors),
        }

    payload = {"source": os.path.abspath(args.eval_json), "weights": args.weights,
               "keypoint_conf": args.keypoint_conf, "summary": summary, "rounds": rows}
    print(json.dumps(summary, indent=2))
    for row in rows:
        if row.get("status") != "ok":
            print(row["round"], row.get("status"))
            continue
        errors = {name: data["error_m"] for name, data in row["points"].items()}
        print(row["round"], row["class"], "base=%.3f" % row["baseline_error_m"], errors,
              "ankle_conf", row["ankle_conf"])
    if args.save:
        with open(args.save, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print("saved ->", args.save)


if __name__ == "__main__":
    main()
