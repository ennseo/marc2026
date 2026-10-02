#!/usr/bin/env python3
"""Decide the camera frame convention offline, and re-check it on any new dump.

Needs no ROS, no GPU and no running platform -- just a dump from ``tools/dump_cctv.py``.

The idea: we know the true world coordinate of every graded object and which camera sees
it. Project those coordinates back into the image under both frame conventions; the
correct one puts them inside the frame, the wrong one throws them behind the camera or far
outside it. That settles ``geometry.transform.TF_IS_OPTICAL_FRAME`` without guessing.

    python tools/verify_geometry.py ../cctv_dump_full

**Ground truth comes from the dump itself** -- ``info["score"]["expected"]``, written by
the platform's own grader. It is NOT read from the starter kit answer table any more. That file
is the demo agent's lookup table and says so in its own header ("GT + random wrong axes,
seed 2026"): 12 of its coordinate pairs carry a fixed +12/+12 m corruption. Scoring the
frame convention against it still reached the right verdict but on far muddier evidence --
45/78 vs 1/78 in frame, because the corrupted coordinates land outside the image under
*either* convention. On the grader's key the same test reads 63/64 vs 0/64.
Dumps without grader coordinates cannot be used for this verification.

SETTLED 2026-08-24: TF_IS_OPTICAL_FRAME = True. Re-run this on a new scenario or a new
platform build -- the organizers flagged this exact difference in the 2026.R01 notice, so
it is worth re-confirming rather than assuming.

With pillow installed, each camera also gets an annotated copy showing where each answer
coordinate lands, so the result can be confirmed by eye.

Language parsing is NOT checked here -- that is ``nlu/test_parser.py``'s job.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry import transform as geom  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))




def find_images(dump_dir):
    """camera_id -> path of one recorded frame, taken from any round directory."""
    found = {}
    for entry in sorted(os.listdir(dump_dir)):
        sub = os.path.join(dump_dir, entry)
        if not os.path.isdir(sub) or entry == "annotated":
            continue
        for name in os.listdir(sub):
            if not name.endswith(".png"):
                continue
            found.setdefault(os.path.splitext(name)[0], os.path.join(sub, name))
    return found


def load_rounds(dump_dir):
    """Every recorded round's info.json, in directory order."""
    out = []
    for entry in sorted(os.listdir(dump_dir)):
        info = os.path.join(dump_dir, entry, "info.json")
        if os.path.isfile(info):
            with open(info, encoding="utf-8") as f:
                out.append(json.load(f))
    return out


def load_dump(dump_dir):
    with open(os.path.join(dump_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    images = find_images(dump_dir)
    cams = {}
    for cid, c in meta.get("cameras", {}).items():
        if not c.get("K") or not c.get("tf") or c.get("ground_height") is None:
            print(f"  [skip] {cid}: incomplete (K/tf/ground_height missing)")
            continue
        tf = c["tf"]
        qx, qy, qz, qw = tf["rotation_xyzw"]
        cams[cid] = {
            "K": np.array(c["K"], dtype=float).reshape(3, 3),
            "R": geom.quat_to_matrix(qx, qy, qz, qw),
            "t": np.array(tf["translation"], dtype=float),
            "ground": float(c["ground_height"]),
            "width": c["width"], "height": c["height"],
            "image": images.get(cid),
        }
    return cams


def collect_points(dump_dir, cams):
    """[(camera_id, label, world_xyz)] from the grader's own answer key.

    Returns an empty list when no round carries `score.expected` -- the caller then falls
    back to the mock table, with a warning.
    """
    points = []
    for entry in sorted(os.listdir(dump_dir)):
        info_path = os.path.join(dump_dir, entry, "info.json")
        if not os.path.isfile(info_path):
            continue
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        expected = (info.get("score") or {}).get("expected")
        if not expected:
            continue
        cid = expected.get("camera_id")
        if cid not in cams:
            continue
        for key in ("target_coord", "anchor_coord"):
            coord = expected.get(key)
            if coord and len(coord) >= 3:
                points.append((cid, f"{entry.replace('round_', 'r')}.{key[0]}",
                               [float(v) for v in coord]))
    return points




def score_convention(points, cams, optical):
    """How many answer coordinates land inside their own camera's frame."""
    inside = 0
    for cid, _label, xyz in points:
        c = cams[cid]
        px = geom.project_to_pixel(xyz, c["K"], c["R"], c["t"], optical_frame=optical)
        if px is None:
            continue
        u, v = px
        if 0 <= u < c["width"] and 0 <= v < c["height"]:
            inside += 1
    return inside


def check_roundtrip(points, cams, optical):
    """Project then unproject: the error shows whether the ground plane is sane."""
    errors = []
    for cid, _label, xyz in points:
        c = cams[cid]
        px = geom.project_to_pixel(xyz, c["K"], c["R"], c["t"], optical_frame=optical)
        if px is None:
            continue
        back = geom.unproject_to_ground(px[0], px[1], c["K"], c["R"], c["t"],
                                        c["ground"], optical_frame=optical)
        if back is None:
            continue
        errors.append(float(np.hypot(back[0] - xyz[0], back[1] - xyz[1])))
    return errors


def annotate(dump_dir, cams, points, optical):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("  (pillow not installed -- skipping annotated images)")
        return
    out_dir = os.path.join(dump_dir, "annotated")
    os.makedirs(out_dir, exist_ok=True)
    by_cam = {}
    for cid, label, xyz in points:
        by_cam.setdefault(cid, []).append((label, xyz))

    for cid, items in by_cam.items():
        c = cams[cid]
        src = c["image"]
        if not src or not os.path.exists(src):
            continue
        img = Image.open(src).convert("RGB")
        draw = ImageDraw.Draw(img)
        drawn = 0
        for label, xyz in items:
            px = geom.project_to_pixel(xyz, c["K"], c["R"], c["t"], optical_frame=optical)
            if px is None:
                continue
            u, v = px
            if not (0 <= u < c["width"] and 0 <= v < c["height"]):
                continue
            draw.line([(u - 12, v), (u + 12, v)], fill=(255, 0, 0), width=2)
            draw.line([(u, v - 12), (u, v + 12)], fill=(255, 0, 0), width=2)
            draw.text((u + 14, v - 6), label, fill=(255, 255, 0))
            drawn += 1
        dst = os.path.join(out_dir, f"{cid}.png")
        img.save(dst)
        print(f"  {cid}: {drawn} marker(s) -> {dst}")


def check_distortion(dump_dir):
    """Report whether the dump can say anything about lens distortion.

    It usually cannot: tools/dump_cctv.py records CameraInfo.k but not CameraInfo.d, so
    geometry/transform.py's "distortion ignored" caveat stays unverified. Print what the
    intrinsics do imply, and how to close the gap for real.
    """
    with open(os.path.join(dump_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    cams = meta.get("cameras", {})
    print("=" * 78)
    print("LENS DISTORTION")
    print("=" * 78)
    has_d = [cid for cid, c in cams.items()
             if any(k in c for k in ("d", "D", "distortion"))]
    if has_d:
        for cid in sorted(has_d):
            c = cams[cid]
            d = c.get("d") or c.get("D") or c.get("distortion")
            zero = all(abs(float(x)) < 1e-9 for x in d)
            verdict = ("zero, safe to ignore" if zero
                       else "NON-ZERO -- undistort (u, v) before unprojecting")
            print(f"  {cid}: d={list(d)}  ->  {verdict}")
        print()
        return
    print("  CameraInfo.d was not recorded, so this cannot be settled from this dump.")
    ks = {tuple(round(float(x), 4) for x in c["K"]) for c in cams.values() if c.get("K")}
    if len(ks) == 1 and cams:
        k = next(iter(ks))
        cam = next(iter(cams.values()))
        print(f"  Indirect evidence: all {len(cams)} cameras share one K, "
              f"fx={k[0]:.3f} fy={k[4]:.3f} c=({k[2]:.1f}, {k[5]:.1f}).")
        if (abs(k[0] - k[4]) < 1e-6 and abs(k[2] - cam["width"] / 2) < 1.0
                and abs(k[5] - cam["height"] / 2) < 1.0):
            print("  That is an ideal pinhole (fx == fy, principal point exactly at the")
            print("  image centre) -- what a distortion-free render looks like, not a")
            print("  calibrated real lens. Distortion is very likely zero.")
    print('  To settle it: add  "d": list(info.d)  next to the "K" line in')
    print("  tools/dump_cctv.py and re-dump.\n")


def check_geometry(dump_dir):
    print("=" * 78)
    print("FRAME CONVENTION")
    print("=" * 78)
    cams = load_dump(dump_dir)
    if not cams:
        print("No usable cameras in the dump. Was the platform on 2026.R01?")
        return
    print(f"cameras with complete data: {', '.join(sorted(cams))}\n")

    points = collect_points(dump_dir, cams)
    source = "the grader's own answer key (info.json -> score.expected)"
    if not points:
        raise ValueError("No grader answer coordinates matched a recorded camera. "
                         "Use a scored CCTV dump with info.json -> score.expected.")
    print(f"ground truth: {source}")
    print(f"coordinates testable against these cameras: {len(points)}\n")

    results = {}
    for optical in (True, False):
        inside = score_convention(points, cams, optical)
        behind = sum(1 for cid, _label, xyz in points
                     if geom.project_to_pixel(xyz, cams[cid]["K"], cams[cid]["R"],
                                              cams[cid]["t"],
                                              optical_frame=optical) is None)
        errors = check_roundtrip(points, cams, optical)
        median = float(np.median(errors)) if errors else float("inf")
        results[optical] = inside
        name = "TF_IS_OPTICAL_FRAME = True " if optical else "TF_IS_OPTICAL_FRAME = False"
        print(f"  {name}: {inside:3d}/{len(points)} inside the image, {behind:3d} behind "
              f"the camera, round-trip median {median:.3f} m")

    print()
    best = max(results, key=results.get)
    if results[best] == 0:
        print("VERDICT: neither convention works. Something else is off -- check that\n"
              "         the dump and the answer table come from the same scenario, and\n"
              "         read the coordinate-notation section of the technical guide.")
    elif results[best] == results[not best]:
        print("VERDICT: inconclusive (both scored the same). Inspect the annotated\n"
              "         images by eye and see which puts the markers on the objects.")
    else:
        print(f"VERDICT: TF_IS_OPTICAL_FRAME = {best}"
              f"   ({results[best]} vs {results[not best]} coordinates in frame)")
        if best == geom.TF_IS_OPTICAL_FRAME:
            print("         geometry/transform.py already says this -- no change needed.")
        else:
            print(f"         geometry/transform.py currently says "
                  f"{geom.TF_IS_OPTICAL_FRAME} -- CHANGE IT.")

    # The round-trip median above is NOT the frame convention's error. It is dominated by
    # targets that genuinely sit off the ground plane, which the ground-plane assumption
    # cannot recover no matter which convention is used -- see the height breakdown in
    # geometry/transform.py's unproject_to_ground.
    print("\nannotated images (verify by eye -- markers should sit on the objects):")
    annotate(dump_dir, cams, points, best)
    print()


def check_recorded(dump_dir):
    """List the commands the platform actually posed, with their live scores.

    The starter kit answer table comes from the marc2026_demo scenario and carries
    deliberately noised labels, so these recorded commands -- and especially the
    per-round scores next to them -- are the more trustworthy signal. Feed the
    sentences to nlu/test_parser.py to check parsing against them.
    """
    rounds = load_rounds(dump_dir)
    if not rounds:
        return
    print("=" * 78)
    print("RECORDED ROUNDS")
    print("=" * 78)
    scored = 0
    for rec in rounds:
        text = rec.get("voice_command") or rec.get("task_description") or ""
        score = rec.get("score")
        if isinstance(score, dict):
            scored += 1
            score_str = f"score={score.get('total')}"
        else:
            score_str = "score=-"
        print(f"  {rec['dir']:9s} {score_str:12s} {text[:52]}")
    print(f"\n{len(rounds)} rounds recorded, {scored} with a live score.")
    print("Rounds carrying a live score also carry the grader's answer key, which is what\n"
          "the frame-convention test below uses. tools/eval_detection.py grades the whole\n"
          "detection pipeline against the same key.\n")


def main():
    if len(sys.argv) <= 1:
        print("\nusage: python tools/verify_geometry.py <dump_dir>\n")
        print("Get a dump from whoever can run the platform:")
        print("    python3 tools/dump_cctv.py ../cctv_dump_full\n")
        return
    dump_dir = sys.argv[1]
    if not os.path.isdir(dump_dir):
        print(f"\nno such dump directory: {dump_dir}\n")
        return
    print()
    check_recorded(dump_dir)
    check_distortion(dump_dir)
    check_geometry(dump_dir)


if __name__ == "__main__":
    main()
