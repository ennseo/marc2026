"""Pixel <-> world coordinate transforms (효빈 담당). See transform.py."""

from .transform import (
    LANDMARK_TOP_Z,
    TF_IS_OPTICAL_FRAME,
    ground_pixel,
    project_to_pixel,
    quat_to_matrix,
    relation_from_geometry,
    unproject_to_ground,
    unproject_to_plane,
)

# CameraGeometry is NOT re-exported here on purpose: it needs rclpy/tf2, and keeping this
# package importable without ROS is what lets the offline tools run on a laptop.
# Import it explicitly where ROS is available: `from geometry.camera import CameraGeometry`.
__all__ = [
    "LANDMARK_TOP_Z",
    "TF_IS_OPTICAL_FRAME",
    "ground_pixel",
    "project_to_pixel",
    "quat_to_matrix",
    "relation_from_geometry",
    "unproject_to_ground",
    "unproject_to_plane",
]
