import math
import numpy as np


# Keep the current configuration assuming R from /tf_static is based on the camera optical frame.
TF_IS_OPTICAL_FRAME = True

# Transformation used when the ROS body frame differs from the optical frame.
OPTICAL_TO_BODY = np.array([
    [0.0,  0.0,  1.0],
    [-1.0, 0.0,  0.0],
    [0.0, -1.0,  0.0],
], dtype=float)


# Height correction values for objects located on landmarks.
# The reference is the ground_z of the corresponding camera.
LANDMARK_TOP_Z = {
    "picnic_table": 0.85,
    "bench": 0.37,
    "vending_machine": 1.54,
    "trash_can": 0.25,
    "hydrant": 0.60,
    "bicycle": 0.90,
    "taxi": 1.20,
    "police_car": 1.20,
}


# Accounts for the issue where a landmark's BBox bottom-center is located at the object's near edge.
# Distance used to shift the point away from the camera.
LANDMARK_HALF_DEPTH = {
    "police_car": 1.94,
    "taxi": 1.79,
    "picnic_table": 1.73,
    "bench": 0.78,
    "bicycle": 0.80,
    "vending_machine": 0.75,
    "trash_can": 0.10,
    "hydrant": 0.00,
}


# Minimum z component used to determine whether a ray is nearly parallel to the ground plane.
MIN_RAY_Z = 0.05

# Maximum camera-ray distance used to prevent abnormally distant intersections.
MAX_RAY_DISTANCE = 100.0

# Minimum bounding box size used to filter invalid detections.
MIN_BBOX_SIZE = 1.0

# Small value used for numerical calculations.
EPS = 1e-9


def rotation_matrix_from_euler_zyx(roll_deg, pitch_deg, yaw_deg):
    """
    Converts roll, pitch, and yaw from a USD Camera Prim into a world rotation matrix.

    Rotation order is ZYX:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    Input angles are in degrees.
    """

    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))

    cr = math.cos(roll)
    sr = math.sin(roll)

    cp = math.cos(pitch)
    sp = math.sin(pitch)

    cy = math.cos(yaw)
    sy = math.sin(yaw)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ], dtype=float)

    Ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ], dtype=float)

    Rz = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    return Rz @ Ry @ Rx


def quat_to_matrix(x, y, z, w):
    """
    Converts a ROS /tf_static quaternion into a 3x3 rotation matrix.
    """

    q = np.array([
        float(x),
        float(y),
        float(z),
        float(w),
    ], dtype=float)

    norm = np.linalg.norm(q)

    if norm < EPS:
        return None

    x, y, z, w = q / norm

    return np.array([
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ], dtype=float)


def ground_pixel(box):
    """
    Computes the default ground-contact pixel from a bounding box.

    For typical ground objects, the BBox bottom-center approximates the
    actual contact point, so the horizontal center and bottom y coordinate are used.
    """

    if box is None or len(box) != 4:
        return None

    x0, y0, x1, y1 = map(float, box)

    if x1 <= x0 or y1 <= y0:
        return None

    if (x1 - x0) < MIN_BBOX_SIZE:
        return None

    if (y1 - y0) < MIN_BBOX_SIZE:
        return None

    u = 0.5 * (x0 + x1)
    v = y1

    return (u, v)


def pixel_to_ray(u, v, K):
    """
    Converts pixel coordinates into a viewing ray in the camera optical frame.

    Uses K^-1 [u, v, 1]^T to calculate the viewing direction corresponding to the pixel.
    """

    if not np.isfinite(float(u)) or not np.isfinite(float(v)):
        return None

    try:
        K = np.asarray(K, dtype=float).reshape(3, 3)
    except (ValueError, TypeError):
        return None

    if not np.all(np.isfinite(K)):
        return None

    if abs(np.linalg.det(K)) < EPS:
        return None

    pixel = np.array([
        float(u),
        float(v),
        1.0,
    ], dtype=float)

    try:
        ray = np.linalg.solve(K, pixel)
    except np.linalg.LinAlgError:
        return None

    norm = float(np.linalg.norm(ray))

    if norm < EPS:
        return None

    return ray / norm


def camera_ray_to_world(ray_camera, R, optical_frame=None):
    """
    Converts a camera ray into a direction vector in the world coordinate system.
    """

    if ray_camera is None:
        return None

    if optical_frame is None:
        optical_frame = TF_IS_OPTICAL_FRAME

    try:
        R = np.asarray(R, dtype=float).reshape(3, 3)
        ray = np.asarray(ray_camera, dtype=float).reshape(3)
    except (ValueError, TypeError):
        return None

    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(ray)):
        return None

    if not optical_frame:
        ray = OPTICAL_TO_BODY @ ray

    ray_world = R @ ray

    norm = float(np.linalg.norm(ray_world))

    if norm < EPS:
        return None

    return ray_world / norm


def ray_plane_intersection(
    ray_world,
    camera_position,
    plane_z,
):
    """
    Computes the intersection between a world ray and the horizontal plane z=plane_z.

    Returns None when the ray is nearly parallel to the plane, intersects
    behind the camera, or intersects at an abnormally distant location.
    """

    if ray_world is None:
        return None

    try:
        camera_position = np.asarray(
            camera_position,
            dtype=float,
        ).reshape(3)

        ray_world = np.asarray(
            ray_world,
            dtype=float,
        ).reshape(3)

        plane_z = float(plane_z)

    except (ValueError, TypeError):
        return None

    if (
        not np.all(np.isfinite(camera_position))
        or not np.all(np.isfinite(ray_world))
        or not np.isfinite(plane_z)
    ):
        return None

    # A small z component means the ray is nearly parallel to the ground plane.
    if abs(ray_world[2]) < MIN_RAY_Z:
        return None

    scale = (plane_z - camera_position[2]) / ray_world[2]

    # If scale <= 0, the intersection lies behind the camera.
    if scale <= EPS:
        return None

    # An excessively distant intersection likely indicates an invalid ray or camera parameter.
    if scale > MAX_RAY_DISTANCE:
        return None

    point = camera_position + scale * ray_world
    point[2] = plane_z

    if not np.all(np.isfinite(point)):
        return None

    return [
        float(point[0]),
        float(point[1]),
        float(point[2]),
    ]


def unproject_to_plane(
    u,
    v,
    K,
    R,
    t,
    plane_z,
    optical_frame=None,
):
    """
    pixel -> camera ray -> world ray -> horizontal plane intersection.
    """

    ray_camera = pixel_to_ray(
        u,
        v,
        K,
    )

    if ray_camera is None:
        return None

    ray_world = camera_ray_to_world(
        ray_camera,
        R,
        optical_frame,
    )

    if ray_world is None:
        return None

    return ray_plane_intersection(
        ray_world,
        t,
        plane_z,
    )


def unproject_to_ground(
    u,
    v,
    K,
    R,
    t,
    ground_z,
    optical_frame=None,
):
    """
    Converts a pixel into 3D world coordinates on the camera ground plane.

    If the actual terrain differs from the camera ground plane, the height
    difference can increase horizontal position error, so ground_z accuracy is important.
    """

    return unproject_to_plane(
        u,
        v,
        K,
        R,
        t,
        ground_z,
        optical_frame,
    )


def push_from_camera(
    point,
    cam_t,
    distance,
):
    """
    Horizontally shifts a ground point in the direction away from the camera.

    Used to correct the systematic offset caused when a landmark's BBox
    bottom-center is closer to the camera than the object's actual center.
    """

    if point is None:
        return None

    if distance is None or float(distance) <= 0.0:
        return point

    try:
        p = np.asarray(point, dtype=float).reshape(3)
        camera = np.asarray(cam_t, dtype=float).reshape(3)
        distance = float(distance)
    except (ValueError, TypeError):
        return point

    direction = p[:2] - camera[:2]
    norm = float(np.linalg.norm(direction))

    if norm < EPS:
        return [
            float(p[0]),
            float(p[1]),
            float(p[2]),
        ]

    direction /= norm

    p[:2] += direction * distance

    return [
        float(p[0]),
        float(p[1]),
        float(p[2]),
    ]


def world_to_camera(
    point,
    R,
    t,
    optical_frame=None,
):
    """
    Transforms world coordinates back into camera coordinates.
    """

    if optical_frame is None:
        optical_frame = TF_IS_OPTICAL_FRAME

    try:
        point = np.asarray(point, dtype=float).reshape(3)
        R = np.asarray(R, dtype=float).reshape(3, 3)
        t = np.asarray(t, dtype=float).reshape(3)
    except (ValueError, TypeError):
        return None

    camera = R.T @ (point - t)

    if not optical_frame:
        camera = OPTICAL_TO_BODY.T @ camera

    return camera


def project_to_pixel(
    point,
    K,
    R,
    t,
    optical_frame=None,
):
    """
    Projects world coordinates back into image pixels.

    Used for pixel -> world -> pixel validation.
    """

    camera = world_to_camera(
        point,
        R,
        t,
        optical_frame,
    )

    if camera is None:
        return None

    if camera[2] <= EPS:
        return None

    try:
        K = np.asarray(K, dtype=float).reshape(3, 3)
    except (ValueError, TypeError):
        return None

    pixel = K @ camera

    if abs(pixel[2]) < EPS:
        return None

    u = pixel[0] / pixel[2]
    v = pixel[1] / pixel[2]

    return (
        float(u),
        float(v),
    )


def reprojection_error(
    original_pixel,
    world_point,
    K,
    R,
    t,
    optical_frame=None,
):
    """
    Reprojects world coordinates to pixels and calculates the error from the original pixel.
    """

    if original_pixel is None or world_point is None:
        return None

    projected = project_to_pixel(
        world_point,
        K,
        R,
        t,
        optical_frame,
    )

    if projected is None:
        return None

    original = np.asarray(
        original_pixel,
        dtype=float,
    )

    projected = np.asarray(
        projected,
        dtype=float,
    )

    error = projected - original

    return {
        "du": float(error[0]),
        "dv": float(error[1]),
        "pixel_error": float(
            np.linalg.norm(error)
        ),
    }


def relation_from_geometry(
    anchor,
    target,
):
    """
    Estimates the spatial relationship using the horizontal distance between two 3D coordinates.

    Based on the current measured data:
        < 0.5 m  -> on
        < 2.0 m  -> beside
        < 6.0 m  -> near
    """

    if anchor is None or target is None:
        return None

    try:
        anchor = np.asarray(
            anchor,
            dtype=float,
        ).reshape(3)

        target = np.asarray(
            target,
            dtype=float,
        ).reshape(3)

    except (ValueError, TypeError):
        return None

    distance = float(
        np.linalg.norm(
            target[:2] - anchor[:2]
        )
    )

    if distance < 0.5:
        return "on"

    if distance < 2.0:
        return "beside"

    if distance < 6.0:
        return "near"

    return None
