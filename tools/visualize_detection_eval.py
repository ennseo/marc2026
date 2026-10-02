#!/usr/bin/env python3
"""Overlay grader and predicted target coordinates from eval_detection JSON."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry.transform import project_to_pixel, quat_to_matrix


RED = (255, 40, 40)
CYAN = (20, 235, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def camera_params(meta, camera_id):
    camera = meta["cameras"].get(camera_id)
    if not camera:
        return None
    qx, qy, qz, qw = camera["tf"]["rotation_xyzw"]
    return (np.asarray(camera["K"], dtype=float).reshape(3, 3),
            quat_to_matrix(qx, qy, qz, qw),
            np.asarray(camera["tf"]["translation"], dtype=float))


def draw_cross(draw, point, color, label):
    if point is None or not all(math.isfinite(value) for value in point):
        return
    x, y = (int(round(point[0])), int(round(point[1])))
    radius = 12
    draw.line((x - radius, y, x + radius, y), fill=color, width=4)
    draw.line((x, y - radius, x, y + radius), fill=color, width=4)
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=WHITE, width=2)
    draw.text((x + 9, y + 7), label, fill=color, stroke_width=2, stroke_fill=BLACK)


def project(meta, camera_id, coord):
    params = camera_params(meta, camera_id)
    if params is None or coord is None:
        return None
    return project_to_pixel(coord, *params)


def load_frame(dump_dir, round_name, camera_id):
    path = dump_dir / round_name / f"{camera_id}.png"
    if not path.is_file():
        return Image.new("RGB", (1280, 720), (40, 40, 40))
    return Image.open(path).convert("RGB")


def add_header(image, lines):
    font = ImageFont.load_default()
    header_h = 22 + 16 * len(lines)
    canvas = Image.new("RGB", (image.width, image.height + header_h), (20, 20, 20))
    canvas.paste(image, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(lines):
        draw.text((10, 8 + index * 16), line, fill=WHITE, font=font)
    return canvas


def contact_sheet(paths, output, columns=3, thumb=(640, 380)):
    images = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail(thumb, Image.Resampling.LANCZOS)
        images.append(image)
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * thumb[0], rows * thumb[1]), (30, 30, 30))
    for index, image in enumerate(images):
        x = (index % columns) * thumb[0] + (thumb[0] - image.width) // 2
        y = (index // columns) * thumb[1] + (thumb[1] - image.height) // 2
        sheet.paste(image, (x, y))
    sheet.save(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    dump_dir = Path(payload["dump_dir"])
    meta = json.loads((dump_dir / "meta.json").read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    written = []
    for record in payload["rounds_detail"]:
        expected = record["expected"]
        submitted = record["submitted"]
        expected_camera = expected.get("camera_id")
        selected_camera = submitted.get("camera_id")
        cameras = [expected_camera]
        if selected_camera and selected_camera != expected_camera:
            cameras.append(selected_camera)

        panels = []
        for camera_id in cameras:
            image = load_frame(dump_dir, record["dir"], camera_id)
            draw = ImageDraw.Draw(image)
            if camera_id == expected_camera:
                draw_cross(draw, project(meta, camera_id, expected.get("target_coord")),
                           RED, "GT")
            if camera_id == selected_camera:
                draw_cross(draw, project(meta, camera_id, submitted.get("target_coord")),
                           CYAN, "PRED")
            panels.append(add_header(image, [f"camera: {camera_id}"]))

        width = sum(panel.width for panel in panels)
        height = max(panel.height for panel in panels)
        combined = Image.new("RGB", (width, height), (20, 20, 20))
        x = 0
        for panel in panels:
            combined.paste(panel, (x, 0))
            x += panel.width

        error = record.get("target_err")
        error_text = "n/a" if error is None else f"{error:.3f} m"
        status = ("DETECTED" if record.get("found") else
                  "FALLBACK" if record.get("fallback") else "NOT_FOUND")
        combined = add_header(combined, [
            f"{record['dir']}  {record.get('pid')}  {expected.get('target_type')}",
            f"{status} | camera {selected_camera} / GT {expected_camera} | target error {error_text}",
            "red = grader target (GT), cyan = submitted target (PRED)",
        ])
        path = output / f"{record['dir']}_{status.lower()}.png"
        combined.save(path)
        written.append(path)

    contact_sheet(written, output / "contact_sheet_33.png")
    print(f"wrote {len(written)} overlays -> {output}")
    print(f"contact sheet -> {output / 'contact_sheet_33.png'}")


if __name__ == "__main__":
    main()
