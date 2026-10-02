#!/usr/bin/env python3
"""Overlay YOLO target boxes and the grader's projected target point on an R03 dump.

This isolates detection from camera selection and unprojection: each round is evaluated
only on its known-correct camera. Green boxes contain the grader point, orange boxes are
same-class detections elsewhere, and a frame with no requested-class box is marked as a
false negative.

    python tools/visualize_r03_target_boxes.py ../cctv_dump_r03_rt01 \
        --weights detection/weights/chungmu_yolo11s_1280_best.pt \
        --classes pencilcase,umbrella,tumbler \
        --output ../detection_debug_r03
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.detector import YOLODetector  # noqa: E402
from geometry.transform import project_to_pixel, quat_to_matrix  # noqa: E402


GREEN = (25, 210, 80)
ORANGE = (255, 145, 20)
RED = (240, 35, 45)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def _text(draw, xy, value, fill=WHITE, background=BLACK):
    """Readable label using Pillow's built-in font (no host font dependency)."""
    font = ImageFont.load_default()
    left, top, right, bottom = draw.textbbox(xy, value, font=font)
    draw.rectangle((left - 3, top - 2, right + 3, bottom + 2), fill=background)
    draw.text(xy, value, fill=fill, font=font)


def _contains(box, point):
    return (point is not None and box[0] <= point[0] <= box[2]
            and box[1] <= point[1] <= box[3])


def _camera_params(meta, camera_id):
    camera = meta["cameras"][camera_id]
    qx, qy, qz, qw = camera["tf"]["rotation_xyzw"]
    return (np.asarray(camera["K"], dtype=float).reshape(3, 3),
            quat_to_matrix(qx, qy, qz, qw),
            np.asarray(camera["tf"]["translation"], dtype=float))


def _make_contact_sheet(paths, output, thumb=(480, 270), columns=3):
    images = [Image.open(path).convert("RGB") for path in paths]
    if not images:
        return
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb[0], rows * thumb[1]), (35, 35, 35))
    for i, image in enumerate(images):
        image.thumbnail(thumb, Image.Resampling.LANCZOS)
        x = (i % columns) * thumb[0] + (thumb[0] - image.width) // 2
        y = (i // columns) * thumb[1] + (thumb[1] - image.height) // 2
        sheet.paste(image, (x, y))
    sheet.save(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--classes", default="pencilcase,umbrella,tumbler")
    parser.add_argument("--output", default="detection_debug_r03")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    wanted = {value.strip() for value in args.classes.split(",") if value.strip()}
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.dump_dir, "meta.json"), encoding="utf-8") as handle:
        meta = json.load(handle)

    detector = YOLODetector(args.weights, threshold=args.threshold,
                            device=args.device, imgsz=args.imgsz)
    records = []
    annotated = []
    round_names = sorted(name for name in os.listdir(args.dump_dir)
                         if name.startswith("round_")
                         and os.path.isdir(os.path.join(args.dump_dir, name)))
    for round_name in round_names:
        round_dir = os.path.join(args.dump_dir, round_name)
        with open(os.path.join(round_dir, "info.json"), encoding="utf-8") as handle:
            info = json.load(handle)
        expected = info["score"]["expected"]
        label = expected["target_type"]
        if label not in wanted:
            continue

        camera_id = expected["camera_id"]
        image_path = os.path.join(round_dir, f"{camera_id}.png")
        image = Image.open(image_path).convert("RGB")
        array = np.asarray(image, dtype=np.uint8)
        detections = [d for d in detector.detect(array, [label]) if d[0] == label]

        K, R, t = _camera_params(meta, camera_id)
        point = project_to_pixel(expected["target_coord"], K, R, t)
        hits = [d for d in detections if _contains(d[2], point)]
        if not detections:
            status = "FALSE_NEGATIVE"
        elif hits:
            status = "CORRECT_OBJECT"
        else:
            status = "WRONG_INSTANCE_OR_BOX"

        draw = ImageDraw.Draw(image)
        for _det_label, confidence, box in detections:
            color = GREEN if _contains(box, point) else ORANGE
            draw.rectangle(box, outline=color, width=4)
            _text(draw, (box[0] + 3, max(3, box[1] + 3)),
                  f"{label} {confidence:.3f}", background=color)
        if point is not None:
            u, v = point
            radius = 11
            draw.line((u - radius, v, u + radius, v), fill=RED, width=4)
            draw.line((u, v - radius, u, v + radius), fill=RED, width=4)
            draw.ellipse((u - 4, v - 4, u + 4, v + 4), outline=WHITE, width=2)

        problem_id = info["score"]["problem_id"]
        _text(draw, (8, 8), f"{round_name} {problem_id} | {camera_id} | {status}",
              background=GREEN if status == "CORRECT_OBJECT" else RED)
        _text(draw, (8, 29), "red cross=grader target | green=contains target | orange=other")

        output_name = f"{round_name}_{label}_{status}.png"
        output_path = os.path.join(args.output, output_name)
        image.save(output_path)
        annotated.append(output_path)
        records.append({
            "round": round_name,
            "problem_id": problem_id,
            "camera_id": camera_id,
            "target_type": label,
            "status": status,
            "grader_pixel": list(point) if point is not None else None,
            "detections": [
                {"confidence": confidence, "bbox": list(box),
                 "contains_grader_point": _contains(box, point)}
                for _det_label, confidence, box in detections
            ],
            "image": output_name,
        })

    with open(os.path.join(args.output, "report.json"), "w", encoding="utf-8") as handle:
        json.dump({"dump_dir": os.path.abspath(args.dump_dir),
                   "weights": os.path.abspath(args.weights),
                   "threshold": args.threshold, "imgsz": args.imgsz,
                   "records": records}, handle, indent=2, ensure_ascii=False)
    _make_contact_sheet(annotated, os.path.join(args.output, "contact_sheet.jpg"))

    counts = Counter(record["status"] for record in records)
    print(f"visualized {len(records)} rounds -> {os.path.abspath(args.output)}")
    for status in ("CORRECT_OBJECT", "FALSE_NEGATIVE", "WRONG_INSTANCE_OR_BOX"):
        print(f"  {status:<23} {counts[status]}")
    for record in records:
        if record["status"] != "CORRECT_OBJECT":
            print(f"  {record['round']} {record['target_type']:<12} "
                  f"{record['status']} detections={len(record['detections'])}")


if __name__ == "__main__":
    main()
