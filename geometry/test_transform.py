#!/usr/bin/env python3
"""
Offline validation tool for the camera pixel -> world coordinate transform.

Input:
    rig_2_b.json
    optional original camera image / overlay image

The JSON already contains:
    - image width / height
    - detection bounding boxes
    - camera position
    - camera Euler orientation
    - focal length
    - horizontal / vertical aperture
    - ground height

This script does NOT require Isaac Sim or a GPU.
It is a reference test harness for checking the geometry before connecting it
to the live Isaac Sim / ROS pipeline.

Important:
    Isaac/ USD cameras normally look along the local -Z axis. The Euler
    convention is kept in one function below so it can be changed easily if
    your existing tf2 transform.py uses a different frame convention.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

# Camera model

def build_intrinsic_matrix(data):
    """Build K from Isaac camera parameters stored in the dataset JSON."""
    W = float(data["image"]["width"])
    H = float(data["image"]["height"])

    cam = data["camera"]
    f_mm = float(cam["focal_length_mm"])
    h_ap_mm = float(cam["horizontal_aperture_mm"])
    v_ap_mm = float(cam["vertical_aperture_mm"])

    fx = f_mm / h_ap_mm * W
    fy = f_mm / v_ap_mm * H
    cx = W / 2.0
    cy = H / 2.0

    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def rotation_matrix_from_xyz(roll_deg, pitch_deg, yaw_deg):
    """
    Convert XYZ Euler angles to a world rotation matrix.

    This follows the common column-vector convention:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    If the project's tf2 quaternion and the dataset Euler convention differ,
    this is the first function to adjust.
    """
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    Rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr],
    ], dtype=float)

    Ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp],
    ], dtype=float)

    Rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1],
    ], dtype=float)

    return Rz @ Ry @ Rx


def camera_rotation(data):
    e = data["camera"]["orientation_euler_deg"]
    return rotation_matrix_from_xyz(
        e["roll"],
        e["pitch"],
        e["yaw"],
    )


# BBox -> pixel -> ray -> ground

def bbox_bottom_center(bbox):
    """Return the BBox bottom-center pixel (u, v)."""
    if len(bbox) != 4:
        raise ValueError(f"Invalid bbox: {bbox}")

    xmin, ymin, xmax, ymax = map(float, bbox)

    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Invalid bbox ordering: {bbox}")

    u = 0.5 * (xmin + xmax)
    v = ymax
    return u, v


def pixel_to_camera_ray(u, v, K):
    """
    Convert a pixel to a normalized camera ray.

    Isaac/ USD camera convention:
        camera forward = local -Z
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = (u - cx) / fx
    y = (v - cy) / fy

    ray = np.array([x, y, -1.0], dtype=float)
    n = np.linalg.norm(ray)

    if n < 1e-12:
        return None

    return ray / n


def intersect_ground(ray_world, camera_position, ground_z,
                     parallel_epsilon=1e-6,
                     max_distance=500.0):
    """
    Intersect a world-space ray with z = ground_z.

    Returns:
        point, status

    status is one of:
        OK
        RAY_PARALLEL
        BEHIND_CAMERA
        TOO_FAR
    """
    dz = float(ray_world[2])

    if abs(dz) < parallel_epsilon:
        return None, "RAY_PARALLEL"

    lam = (float(ground_z) - float(camera_position[2])) / dz

    if lam <= 0:
        return None, "BEHIND_CAMERA"

    if lam > max_distance:
        return None, "TOO_FAR"

    point = np.asarray(camera_position, dtype=float) + lam * ray_world
    return point, "OK"


def unproject_bbox(bbox, K, R, camera_position, ground_z,
                   max_distance=500.0):
    """Full BBox -> bottom-center -> ray -> ground intersection."""
    try:
        u, v = bbox_bottom_center(bbox)
    except ValueError as exc:
        return {
            "status": "INVALID_BBOX",
            "message": str(exc),
            "u": None,
            "v": None,
            "point": None,
        }

    ray_camera = pixel_to_camera_ray(u, v, K)
    if ray_camera is None:
        return {
            "status": "INVALID_RAY",
            "message": "Could not construct camera ray.",
            "u": u,
            "v": v,
            "point": None,
        }

    ray_world = R @ ray_camera
    point, status = intersect_ground(
        ray_world,
        camera_position,
        ground_z,
        max_distance=max_distance,
    )

    result = {
        "status": status,
        "u": u,
        "v": v,
        "ray_camera": ray_camera,
        "ray_world": ray_world,
        "point": point,
    }

    return result


# Optional reprojection consistency check

def project_world_point(point, K, R, camera_position):
    """
    Project a world point back to the image.

    This is a mathematical round-trip check:
        pixel -> 3D -> pixel

    It does not prove the camera calibration is correct; it proves that the
    chosen forward/inverse conventions are internally consistent.
    """
    p = np.asarray(point, dtype=float)
    cam = np.asarray(camera_position, dtype=float)

    # R maps camera -> world, so inverse is R.T for a proper rotation.
    p_camera = R.T @ (p - cam)

    # Isaac/ USD camera looks along -Z.
    if p_camera[2] >= 0:
        return None

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = fx * (p_camera[0] / -p_camera[2]) + cx
    v = fy * (p_camera[1] / -p_camera[2]) + cy

    return np.array([u, v], dtype=float)


# Reporting

def fmt_vec(v):
    if v is None:
        return "None"
    return "(" + ", ".join(f"{x:.3f}" for x in v) + ")"


def print_camera_summary(data, K, R):
    cam = data["camera"]
    print("=" * 72)
    print("Camera geometry")
    print("=" * 72)
    print(f"Image size       : {data['image']['width']} x {data['image']['height']}")
    print(f"Camera position  : {fmt_vec(cam['position'])}")
    print(
        "Euler (deg)      : "
        f"roll={cam['orientation_euler_deg']['roll']:.3f}, "
        f"pitch={cam['orientation_euler_deg']['pitch']:.3f}, "
        f"yaw={cam['orientation_euler_deg']['yaw']:.3f}"
    )
    print(f"Ground Z         : {cam['ground_height']:.3f}")
    print(f"Focal length     : {cam['focal_length_mm']:.3f} mm")
    print(f"Horizontal ap.   : {cam['horizontal_aperture_mm']:.3f} mm")
    print(f"Vertical ap.     : {cam['vertical_aperture_mm']:.3f} mm")
    print("\nIntrinsic K:")
    print(K)
    print("\nRotation R (camera -> world):")
    print(R)


def run(data, class_name=None, instance_id=None, limit=None, max_distance=500.0):
    K = build_intrinsic_matrix(data)
    R = camera_rotation(data)
    camera_position = np.array(data["camera"]["position"], dtype=float)
    ground_z = float(data["camera"]["ground_height"])

    print_camera_summary(data, K, R)

    detections = data.get("detections", [])

    if class_name:
        detections = [
            d for d in detections
            if d.get("class") == class_name
        ]

    if instance_id:
        detections = [
            d for d in detections
            if d.get("instance_id") == instance_id
        ]

    if limit is not None:
        detections = detections[:limit]

    print("\n" + "=" * 72)
    print(f"Detection tests: {len(detections)}")
    print("=" * 72)

    passed = 0
    failed = 0

    for d in detections:
        bbox = d["bbox"]

        result = unproject_bbox(
            bbox,
            K,
            R,
            camera_position,
            ground_z,
            max_distance=max_distance,
        )

        print(f"\n[{d.get('kind', '?')}] {d.get('class', '?')}")
        print(f"  instance       : {d.get('instance_id', '?')}")
        print(f"  bbox           : {bbox}")
        print(
            f"  bottom-center  : "
            f"({result['u']:.2f}, {result['v']:.2f})"
            if result["u"] is not None else
            "  bottom-center  : INVALID"
        )
        print(f"  status         : {result['status']}")

        if result["status"] != "OK":
            print(f"  reason         : {result.get('message', result['status'])}")
            failed += 1
            continue

        print(f"  camera ray     : {fmt_vec(result['ray_camera'])}")
        print(f"  world ray      : {fmt_vec(result['ray_world'])}")
        print(f"  world point    : {fmt_vec(result['point'])}")

        reproj = project_world_point(
            result["point"], K, R, camera_position
        )

        original_pixel = np.array([result["u"], result["v"]])
        error_px = float(np.linalg.norm(reproj - original_pixel))

        print(f"  reprojection   : ({reproj[0]:.3f}, {reproj[1]:.3f})")
        print(f"  reproj error   : {error_px:.6f} px")

        if error_px < 1e-5:
            print("  CHECK           : PASS")
            passed += 1
        else:
            print("  CHECK           : FAIL")
            failed += 1

    print("\n" + "=" * 72)
    print(f"SUMMARY: {passed} PASS / {failed} FAIL")
    print("=" * 72)

    print(
        "\n주의: reprojection PASS는 'pixel -> 3D -> pixel' 수학적 일관성을 "
        "검증하는 것이며, 실제 world 좌표의 절대 정확도를 증명하는 것은 아닙니다."
    )
    print(
        "실제 좌표 정확도는 Isaac Sim의 ground-truth object position과 "
        "world point를 비교해야 합니다."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Offline test for BBox -> world coordinate conversion."
    )
    parser.add_argument(
        "json_path",
        help="Path to rig_2_b.json",
    )
    parser.add_argument(
        "--class",
        dest="class_name",
        default=None,
        help="Only test one class, e.g. bench or tumbler",
    )
    parser.add_argument(
        "--instance",
        dest="instance_id",
        default=None,
        help="Only test one instance_id",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only test the first N detections",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=500.0,
        help="Reject ground intersections farther than this many metres.",
    )

    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        raise SystemExit(f"JSON file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    run(
        data,
        class_name=args.class_name,
        instance_id=args.instance_id,
        limit=args.limit,
        max_distance=args.max_distance,
    )


if __name__ == "__main__":
    main()
