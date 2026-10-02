"""lidar_obstacles_puregrid.py -- 원(circle) 근사 없이 raw point를 그대로 쓰는 버전.

lidar_obstacles.py(원본)는 raw point를 DBSCAN류 클러스터링으로 뭉쳐서
(cx, cy, radius) 원으로 근사하고, 그 근사가 놓치는 부분(긴 벽, 성긴 점)을
_inject_front_raw_virtual_obstacle / _inject_thin_raw_virtual_obstacles 로
땜빵했다. 이 버전은 그 근사 자체를 없애고 -- z필터+world변환+ROI필터를 거친
raw point를 그대로 "못 가는 셀" 후보로 돌려준다. 원 근사 오차, 그리고 그
오차를 메우던 보정 함수들이 전부 필요 없어진다.

occupancy grid에 실제로 굽는 건 agent_navigation.py의
OccupancyGridPlanner._block_cells()가 담당 -- 여기서는 world 좌표 (x, y)
점 리스트까지만 만든다.
"""

import math

try:
    from sensor_msgs_py import point_cloud2
except ImportError:
    point_cloud2 = None

DEFAULT_R_MIN = 0.05
DEFAULT_R_MAX = 12.0
DEFAULT_NEAR_SAFETY_M = 5.00

# 밀집도 필터 기본값을 완화.
# 기존 10개는 사람 다리/얇은 물체/원거리 성긴 점을 너무 많이 지울 수 있음.
DENSITY_RADIUS_M = 0.3
DENSITY_MIN_NEIGHBORS = 10

# NORMAL/NEAR 공통: 비정상적으로 거대한 연결 덩어리는 노이즈로 제거.
# 점 간 15cm 이내면 연결, 중심으로부터 최대 반경 3m 이상이면 drop.
GIANT_COMPONENT_RADIUS_M = 5.0
GIANT_COMPONENT_LINK_M = 0.1


def _density_filter(points, radius_m=DENSITY_RADIUS_M, min_neighbors=DENSITY_MIN_NEIGHBORS):
    """radius_m 안에 다른 점이 min_neighbors개 이상 없는 고립점 제거.

    공간 해시(격자 버킷)로 O(N) 근사 -- 셀 크기=radius_m, 3x3 이웃 버킷만 검사.
    """
    if len(points) <= 1:
        return points

    cell = radius_m
    buckets = {}
    for i, (x, y) in enumerate(points):
        key = (int(x // cell), int(y // cell))
        buckets.setdefault(key, []).append(i)

    r2 = radius_m * radius_m
    kept = []
    for i, (x, y) in enumerate(points):
        cx, cy = int(x // cell), int(y // cell)
        n = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((cx + dx, cy + dy), ()):
                    if j == i:
                        continue
                    ox, oy = points[j]
                    if (ox - x) ** 2 + (oy - y) ** 2 <= r2:
                        n += 1
                        if n >= min_neighbors:
                            break
                if n >= min_neighbors:
                    break
            if n >= min_neighbors:
                break
        if n >= min_neighbors:
            kept.append((x, y))

    return kept



def _remove_giant_components(
    points,
    link_m=GIANT_COMPONENT_LINK_M,
    max_radius_m=GIANT_COMPONENT_RADIUS_M,
):
    """Remove only abnormally huge connected point components.

    This is used in both NORMAL and NEAR modes.
    Points within link_m are considered connected.  For each connected component,
    compute its centroid and maximum radial extent.  Components with radius
    >= max_radius_m are treated as noise (often floor/scan burst) and dropped.

    Uses spatial buckets so it does not do a full O(N^2) search.
    """
    n = len(points)
    if n <= 1:
        return points, 0

    cell = max(0.05, float(link_m))
    buckets = {}

    for i, (x, y) in enumerate(points):
        key = (int(math.floor(x / cell)), int(math.floor(y / cell)))
        buckets.setdefault(key, []).append(i)

    link2 = cell * cell
    seen = [False] * n
    kept = []
    removed_components = 0

    for seed in range(n):
        if seen[seed]:
            continue

        seen[seed] = True
        queue = [seed]
        members = []

        while queue:
            i = queue.pop()
            members.append(i)

            x, y = points[i]
            bx = int(math.floor(x / cell))
            by = int(math.floor(y / cell))

            for dbx in (-1, 0, 1):
                for dby in (-1, 0, 1):
                    for j in buckets.get((bx + dbx, by + dby), ()):
                        if seen[j]:
                            continue

                        ox, oy = points[j]
                        if (ox - x) ** 2 + (oy - y) ** 2 <= link2:
                            seen[j] = True
                            queue.append(j)

        if not members:
            continue

        cx = sum(points[i][0] for i in members) / len(members)
        cy = sum(points[i][1] for i in members) / len(members)

        radius = max(
            math.hypot(points[i][0] - cx, points[i][1] - cy)
            for i in members
        )

        if radius >= max_radius_m:
            removed_components += 1
            continue

        kept.extend(points[i] for i in members)

    return kept, removed_components


LIDAR_FWD_OFFSET_M = 0.35


def read_xyz(cloud_msg, stride=1):
    if point_cloud2 is None:
        raise RuntimeError("sensor_msgs_py 없음 — ROS 2 환경에서 실행 필요")

    stride = max(1, int(stride))
    pts = []

    for i, p in enumerate(
        point_cloud2.read_points(
            cloud_msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
        )
    ):
        if i % stride != 0:
            continue

        try:
            x, y, z = float(p[0]), float(p[1]), float(p[2])
        except (TypeError, ValueError):
            continue

        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue

        pts.append((x, y, z))

    return pts


def filter_and_to_world(
    pts_local,
    robot_pose,
    z_min=-0.2,
    z_max=1.80,
    r_min=DEFAULT_R_MIN,
    r_max=DEFAULT_R_MAX,
):
    x0, y0, yaw = robot_pose
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    out = []

    for lx, ly, lz in pts_local:
        if lz < z_min or lz > z_max:
            continue

        d = math.hypot(lx, ly)
        if d < r_min or d > r_max:
            continue

        lx_base = lx + LIDAR_FWD_OFFSET_M

        wx = x0 + lx_base * cy - ly * sy
        wy = y0 + lx_base * sy + ly * cy

        out.append((wx, wy, d, lx, ly))

    return out


def _dist_point_to_segment(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    seg2 = dx * dx + dy * dy

    if seg2 <= 1e-12:
        return math.hypot(px - ax, py - ay)

    t = ((px - ax) * dx + (py - ay) * dy) / seg2
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy

    return math.hypot(px - cx, py - cy)


def _path_ahead(path, robot_pose, ahead_only=True):
    if not path:
        return []

    if len(path) < 2 or not ahead_only or robot_pose is None:
        return path

    rx, ry = robot_pose[0], robot_pose[1]

    nearest = min(
        range(len(path)),
        key=lambda i: math.hypot(path[i][0] - rx, path[i][1] - ry),
    )

    segs = path[nearest:]
    if len(segs) < 2:
        segs = path[-2:]

    return segs


def roi_filter(
    pts_world,
    path,
    corridor_w=2.0,
    ahead_only=True,
    robot_pose=None,
    near_safety_m=DEFAULT_NEAR_SAFETY_M,
):
    if not pts_world:
        return []

    if not path or len(path) < 2:
        return pts_world

    segs = _path_ahead(path, robot_pose, ahead_only=ahead_only)
    if len(segs) < 2:
        return pts_world

    kept = []

    for wx, wy, local_range, lx, ly in pts_world:
        if local_range <= near_safety_m:
            kept.append((wx, wy, local_range, lx, ly))
            continue

        for i in range(len(segs) - 1):
            ax, ay = segs[i]
            bx, by = segs[i + 1]

            if _dist_point_to_segment(wx, wy, ax, ay, bx, by) <= corridor_w:
                kept.append((wx, wy, local_range, lx, ly))
                break

    return kept


def cloud_to_blocked_points(
    cloud_msg,
    robot_pose,
    path=None,
    corridor_w=2.0,
    stride=2,
    z_min=-0.2,
    z_max=1.80,
    ahead_only=True,
    near_safety_m=DEFAULT_NEAR_SAFETY_M,
    r_min=DEFAULT_R_MIN,
    r_max=DEFAULT_R_MAX,
    density_min_neighbors=DENSITY_MIN_NEIGHBORS,
    density_radius_m=DENSITY_RADIUS_M,
    debug=False,
):
    """PointCloud2 -> world blocked-point list.

    density_min_neighbors를 호출부에서 바꿀 수 있게 해서
    목표 근거리에서는 더 민감하게 탐지할 수 있다.
    """
    if cloud_msg is None or robot_pose is None:
        return ([], {}) if debug else []

    stats = {}
    pts_local = read_xyz(cloud_msg, stride=stride)

    if debug:
        stats["n_raw"] = len(pts_local)
        if pts_local:
            xs = [p[0] for p in pts_local]
            ys = [p[1] for p in pts_local]
            zs = [p[2] for p in pts_local]
            stats["x_min_seen"] = round(min(xs), 3)
            stats["x_max_seen"] = round(max(xs), 3)
            stats["y_min_seen"] = round(min(ys), 3)
            stats["y_max_seen"] = round(max(ys), 3)
            stats["z_min_seen"] = round(min(zs), 3)
            stats["z_max_seen"] = round(max(zs), 3)
        else:
            stats["x_min_seen"] = None
            stats["x_max_seen"] = None
            stats["y_min_seen"] = None
            stats["y_max_seen"] = None
            stats["z_min_seen"] = None
            stats["z_max_seen"] = None

    pts_world = filter_and_to_world(
        pts_local,
        robot_pose,
        z_min=z_min,
        z_max=z_max,
        r_min=r_min,
        r_max=r_max,
    )

    if debug:
        stats["n_after_zfilter"] = len(pts_world)

    pts_roi = roi_filter(
        pts_world,
        path,
        corridor_w=corridor_w,
        ahead_only=ahead_only,
        robot_pose=robot_pose,
        near_safety_m=near_safety_m,
    )

    if debug:
        stats["n_after_roi"] = len(pts_roi)
        stats["nearest_raw_m"] = (
            round(min((p[2] for p in pts_world), default=float("inf")), 3)
            if pts_world else None
        )
        stats["nearest_roi_m"] = (
            round(min((p[2] for p in pts_roi), default=float("inf")), 3)
            if pts_roi else None
        )

    points = [(wx, wy) for wx, wy, _local_range, _lx, _ly in pts_roi]

    # density_min_neighbors <= 0 means NO density filtering:
    # every point that passed range/Z/ROI filters becomes a blocked-point candidate.
    # In that near raw mode only, reject abnormally huge connected components.
    giant_removed = 0

    if int(density_min_neighbors) > 0:
        # NORMAL: 기존 density 필터(예: 10개 이웃) 유지.
        points = _density_filter(
            points,
            radius_m=float(density_radius_m),
            min_neighbors=int(density_min_neighbors),
        )

    # NORMAL / NEAR 공통:
    # 15cm 이내 점들을 같은 연결 장애물로 보고,
    # 연결 덩어리의 중심 기준 반경이 3m 이상이면 비정상 노이즈로 제거.
    points, giant_removed = _remove_giant_components(
        points,
        link_m=GIANT_COMPONENT_LINK_M,
        max_radius_m=GIANT_COMPONENT_RADIUS_M,
    )

    if debug:
        stats["n_blocked_points"] = len(points)
        stats["giant_components_removed"] = giant_removed
        stats["density_min_neighbors"] = int(density_min_neighbors)
        stats["density_radius_m"] = round(float(density_radius_m), 3)
        stats["stride"] = int(stride)
        return points, stats

    return points
