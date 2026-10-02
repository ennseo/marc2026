#!/usr/bin/env python3
"""Create a publication-ready Stage 1 grounding panel from an eval report.

The panel overlays the selected target bounding box, its ground-reference pixel,
and the estimated world coordinate on the original CCTV frame.  A magnified inset
is added because targets can occupy only a few pixels in a full CCTV image.

Example:
    python tools/make_paper_grounding_panel.py \
        path/to/evaluation.json \
        --round round_12 --output results/round12_stage1_grounding.png
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


TARGET = (0, 230, 118)
TARGET_DARK = (0, 105, 54)
ACCENT = (255, 213, 45)
WHITE = (255, 255, 255)
BLACK = (18, 18, 18)


def font(size, bold=False):
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
             else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def label(draw, xy, text, *, text_fill=WHITE, fill=BLACK, text_font=None,
          padding=6):
    text_font = text_font or font(18)
    x, y = xy
    left, top, right, bottom = draw.textbbox((x, y), text, font=text_font)
    draw.rounded_rectangle(
        (left - padding, top - padding, right + padding, bottom + padding),
        radius=5, fill=fill,
    )
    draw.text((x, y), text, fill=text_fill, font=text_font)


def corner_box(draw, box, color, *, width=3, offset=5, corner=14):
    """Draw only outside corner brackets so a tiny object remains visible."""
    x0, y0, x1, y1 = map(float, box)
    x0, y0, x1, y1 = x0 - offset, y0 - offset, x1 + offset, y1 + offset
    segments = [
        (x0, y0, x0 + corner, y0), (x0, y0, x0, y0 + corner),
        (x1, y0, x1 - corner, y0), (x1, y0, x1, y0 + corner),
        (x0, y1, x0 + corner, y1), (x0, y1, x0, y1 - corner),
        (x1, y1, x1 - corner, y1), (x1, y1, x1, y1 - corner),
    ]
    for segment in segments:
        draw.line(segment, fill=BLACK, width=width + 4)
        draw.line(segment, fill=color, width=width)


def find_record(payload, round_name):
    for record in payload.get("rounds_detail", []):
        if record.get("dir") == round_name:
            return record
    raise SystemExit(f"round not present in evaluation report: {round_name}")


def expanded_crop(box, image_size, scale=5.0):
    x0, y0, x1, y1 = map(float, box)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(64.0, max(x1 - x0, y1 - y0) * scale)
    width, height = image_size
    left = max(0, min(int(round(cx - side / 2)), width - int(side)))
    top = max(0, min(int(round(cy - side / 2)), height - int(side)))
    right = min(width, left + int(side))
    bottom = min(height, top + int(side))
    return left, top, right, bottom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_json", help="eval_detection JSON containing rounds_detail")
    parser.add_argument("--round", default="round_12")
    parser.add_argument("--output", required=True)
    parser.add_argument("--inset-width", type=int, default=360)
    parser.add_argument(
        "--instruction",
        help="short task text shown in the header; defaults to the recorded command",
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_json).resolve()
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    record = find_record(payload, args.round)
    trace = record.get("vision_trace") or {}
    bbox = trace.get("target_bbox")
    if not bbox:
        raise SystemExit(f"{args.round} has no target_bbox in vision_trace")

    submitted = record.get("submitted") or {}
    camera_id = submitted.get("camera_id")
    coord = submitted.get("target_coord")
    target_type = submitted.get("target_type") or trace.get("detected_label") or "target"
    landmark = submitted.get("landmark") or ""
    confidence = trace.get("target_conf")
    instruction = args.instruction or record.get("command") or ""

    dump_dir = Path(payload["dump_dir"])
    image_path = dump_dir / args.round / f"{camera_id}.png"
    if not image_path.is_file():
        raise SystemExit(f"CCTV frame not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    clean_image = image.copy()
    draw = ImageDraw.Draw(image)

    x0, y0, x1, y1 = map(float, bbox)
    corner_box(draw, bbox, TARGET, width=3, offset=5, corner=12)

    u, v = (x0 + x1) / 2.0, y1
    marker_y = v + 20
    draw.line((u, v + 5, u, marker_y - 4), fill=ACCENT, width=3)
    draw.ellipse((u - 4, marker_y - 4, u + 4, marker_y + 4),
                 fill=ACCENT, outline=BLACK, width=2)

    conf_text = "" if confidence is None else f"  {confidence:.2f}"
    coord_text = "n/a" if not coord else f"({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f}) m"
    callout_x = min(max(16, int(x0) - 210), max(16, image.width - 430))
    callout_y = min(image.height - 76, int(y1) + 20)
    label(draw, (callout_x, callout_y), f"{target_type}{conf_text}",
          fill=TARGET_DARK, text_font=font(21, bold=True))
    label(draw, (callout_x, callout_y + 34), f"world: {coord_text}",
          text_font=font(19))

    # Magnified target inset in the upper-right corner.
    crop_box = expanded_crop(bbox, image.size)
    crop = clean_image.crop(crop_box)
    inset_w = min(args.inset_width, image.width // 3)
    inset_h = int(round(crop.height * inset_w / crop.width))
    crop = crop.resize((inset_w, inset_h), Image.Resampling.LANCZOS)
    crop_draw = ImageDraw.Draw(crop)
    crop_left, crop_top, crop_right, crop_bottom = crop_box
    sx = inset_w / (crop_right - crop_left)
    sy = inset_h / (crop_bottom - crop_top)
    inset_box = (
        (x0 - crop_left) * sx,
        (y0 - crop_top) * sy,
        (x1 - crop_left) * sx,
        (y1 - crop_top) * sy,
    )
    corner_box(crop_draw, inset_box, TARGET, width=3, offset=8, corner=28)
    inset_x, inset_y = image.width - inset_w - 20, 20
    image.paste(crop, (inset_x, inset_y))
    draw = ImageDraw.Draw(image)
    draw.rectangle((inset_x - 3, inset_y - 3, inset_x + inset_w + 3, inset_y + inset_h + 3),
                   outline=WHITE, width=6)
    draw.rectangle((inset_x, inset_y, inset_x + inset_w, inset_y + inset_h),
                   outline=TARGET, width=4)
    label(draw, (inset_x + 10, inset_y + 9), "target detail",
          fill=TARGET_DARK, text_font=font(18, bold=True), padding=4)

    header_h = 86
    canvas = Image.new("RGB", (image.width, image.height + header_h), BLACK)
    canvas.paste(image, (0, header_h))
    header = ImageDraw.Draw(canvas)
    header.text((18, 9), "Stage 1: Multi-Camera Visual Grounding",
                fill=WHITE, font=font(22, bold=True))
    if instruction:
        header.text((18, 35), f'Task: "{instruction}"',
                    fill=(255, 224, 120), font=font(17, bold=True))
    context = f"camera: {camera_id}   target: {target_type}"
    if landmark:
        context += f"   landmark: {landmark}"
    header.text((18, 62), context, fill=(205, 205, 205), font=font(16))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"wrote {output.resolve()}")


if __name__ == "__main__":
    main()
