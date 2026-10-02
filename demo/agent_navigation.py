"""Occupancy-grid path planning and motion control with live LiDAR obstacles."""
import heapq
import logging
import math
import os

import numpy as np
import yaml

SQRT2 = math.sqrt(2.0)
log = logging.getLogger("participant_app")


# ============================================================
# CUSTOM OCCUPANCY MAP (Isaac Sim에서 찍은 좌표)
# ============================================================
# 각 그룹의 점을 입력 순서대로 선으로 연결해 occupied(100) 경계로 사용.
CUSTOM_MAP_LINES = [
    [
        (-114.82775, 122.686),
        (-96.82238, 99.99744),
        (-52.66355, 146.73214),
               
    ],
    [
        (-49.41,144.0),
        (-36.757, 123.325),
    ],
    [
        (-66.15084, 103.55459),
        (-75.15784, 111.09892),
        (-90.3131, 94.05818),
        (-71.90983, 81.57766),
        (-43.53447, 112.88588),
        (-45.99341, 115.384493),
    ],
    [
        (-47.71429, 118.08146),
        (-57.59431, 131.39733),
        (-73.19732, 113.35656),
        (-64.48258, 105.47844),
    ],
]

# ============================================================
# CUSTOM OCCUPANCY MAP 2 (배송 구간용)
# ============================================================

CUSTOM_MAP2_LINES = [
    [
        (-114.82775, 122.686),
        (-96.82238, 99.99744),
        (-52.66355, 146.73214),
           
    ],
    [
        (-49.41,144.0),
        (-36.757, 123.325),
    ],
    [
        (-66.15084, 103.55459),
        (-75.15784, 111.09892),
        (-77.85,107.949),
    ],
    [
        (-81.75,104.52),
        (-90.3131, 94.05818),
        (-71.90983, 81.57766),
        (-56.512,95.953),
        (-66.16,97.07),
        (-66.15084, 103.55459),
    ],
    [   
        (-56.512,95.953),     
        (-43.53447, 112.88588),
        (-45.99341, 115.384493),
    ],
    [
        (-47.71429, 118.08146),
        (-57.59431, 131.39733),
        (-73.19732, 113.35656),
        (-64.48258, 105.47844),
    ],
]

CUSTOM_MAP_LINE_WIDTH_M = 0.10



def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class GoalSeekController:
    GOAL_TOLERANCE = 0.40
    HEADING_TOLERANCE = math.radians(28.0)

    MAX_LINEAR = 1.10
    MAX_ANGULAR = 1.30   # 전진하면서 도는 동안의 상한 (w_max=2.4*v/T로 추가 제한됨)
    K_LINEAR = 0.50
    K_ANGULAR = 1.10
    STEERING_TRACK = 0.38
    CURVE_LINEAR = 0.60   # 회전 중(헤딩오차 5~28도)에도 유지할 최소 선속도

    # 제자리 회전(선속도=0) 전용 최대각속도. 전진 중 회전(MAX_ANGULAR)과 달리
    # 2v<=wT Ackermann 제약이 안 걸리는 순수 제자리 스핀이라 더 빠르게 잡음.
    # 실제 섀시가 이 속도를 못 받으면(제자리서 헛돌거나 멈추면) 낮출 것.
    SPIN_ANGULAR = 1.80

    def compute(self, pose: tuple, goal: tuple, cruise: bool = False) -> tuple:
        x, y, yaw = pose
        gx, gy = goal

        dx = gx - x
        dy = gy - y
        dist = math.hypot(dx, dy)

        if dist < self.GOAL_TOLERANCE:
            return 0.0, 0.0, True

        target_yaw = math.atan2(dy, dx)
        heading_err = _wrap_pi(target_yaw - yaw)
        abs_err = abs(heading_err)

        # ------------------------------------------------------------
        # Turn-friendly speed control
        #
        # > 28 deg : 목표 방향으로 최대각속도 뱅뱅제어 -- 천천히 수렴 안 하고
        #            바로 최대속도로 확 돌림 (비례제어 아님, 방향만 보고 최대출력)
        # 5~28 deg : still curving toward the target -> keep CURVE_LINEAR (0.6)
        #            so it doesn't crawl through corners
        # < 5 deg  : full cruise speed
        # ------------------------------------------------------------
        if abs_err > self.HEADING_TOLERANCE:
            angular = math.copysign(self.SPIN_ANGULAR, heading_err)
            linear = 0.0

        else:
            angular = _clamp(
                self.K_ANGULAR * heading_err,
                -self.MAX_ANGULAR,
                self.MAX_ANGULAR,
            )

            if abs_err > math.radians(5.0):
                linear = self.CURVE_LINEAR

            elif cruise:
                linear = self.MAX_LINEAR

            else:
                linear = _clamp(
                    self.K_LINEAR * dist,
                    0.0,
                    self.MAX_LINEAR,
                )

        # When moving, keep angular command inside chassis-valid range.
        # Do NOT over-limit it: allow tighter curves than before.
        if linear > 1e-3:
            w_max = 2.4 * linear / self.STEERING_TRACK
            angular = _clamp(angular, -w_max, w_max)

        return linear, angular, False



class LocalObstacleAvoider:
    """Reactive LiDAR avoidance.

    angular_z > 0 : left
    angular_z < 0 : right
    """

    # 3-stage reactive avoidance
    # FAR   : normal path following
    # MID   : creep forward + strong steering
    # NEAR  : zero forward speed + aggressive in-place turn
    DETECT_DISTANCE = 1.80      # earlier front reaction for person/static obstacle
    NEAR_DISTANCE = 0.70        # stop/spin before collision
    SIDE_REACT_DISTANCE = 0.75   # side-only local reaction minimized; A* handles static obstacles
    FRONT_SLOW_DISTANCE = 3.00

    FRONT_HALF_ANGLE = math.radians(40.0)
    SIDE_MAX_ANGLE = math.radians(170.0)

    ROBOT_MARGIN = 0.40

    MID_LINEAR = 0.10
    MID_ANGULAR = 0.45
    NEAR_ANGULAR = 1.20
    NEAR_LINEAR = 0.00          # emergency zone: no forward motion

    # 도리도리 방지
    TURN_HYSTERESIS = 0.30      # 좌우 거리 차이가 30cm 이상일 때만 새 방향 선택
    RELEASE_DISTANCE = 2.50     # 1.8→2.5: 더 확실히 벗어난 뒤에만 로컬 해제(글로벌 복귀 지연)
    RELEASE_FRAMES = 8          # 연속 clear 프레임 후 로컬 해제

    def __init__(self):
        self.obstacles = []
        self.last_turn_sign = +1.0  # +1=left, -1=right
        self.avoid_lock = False     # 회피 시작 후 방향 유지
        self.clear_count = 0        # 연속 clear 프레임 수 (즉각 해제 방지)

    def update(self, obstacles):
        self.obstacles = []

        if not obstacles:
            return

        for obs in obstacles:
            if obs is None or len(obs) < 3:
                continue

            try:
                ox = float(obs[0])
                oy = float(obs[1])
                rad = max(0.02, float(obs[2]))
            except (TypeError, ValueError):
                continue

            if math.isfinite(ox) and math.isfinite(oy) and math.isfinite(rad):
                self.obstacles.append((ox, oy, rad))

    @staticmethod
    def _to_robot_frame(pose, obs):
        x, y, yaw = pose
        ox, oy, _ = obs

        dx = ox - x
        dy = oy - y

        c = math.cos(yaw)
        s = math.sin(yaw)

        forward = c * dx + s * dy
        left = -s * dx + c * dy

        return forward, left

    def _sector_distances(self, pose):
        front_d = float("inf")
        left_d = float("inf")
        right_d = float("inf")
        left_n = 0    # 왼쪽 섹터 장애물 개수(밀도)
        right_n = 0   # 오른쪽 섹터 장애물 개수(밀도)

        for obs in self.obstacles:
            forward, left = self._to_robot_frame(pose, obs)

            # During large-angle turns the rear body/wheels sweep sideways too.
            # Keep obstacles up to 1.20 m behind the robot center.
            if forward < -1.20:
                continue

            center_dist = math.hypot(forward, left)
            if center_dist > self.DETECT_DISTANCE + obs[2] + self.ROBOT_MARGIN:
                continue

            surface_dist = center_dist - obs[2] - self.ROBOT_MARGIN
            bearing = math.atan2(left, forward)

            if abs(bearing) <= self.FRONT_HALF_ANGLE:
                front_d = min(front_d, surface_dist)

            if 0.0 < bearing <= self.SIDE_MAX_ANGLE:
                left_d = min(left_d, surface_dist)
                left_n += 1

            if -self.SIDE_MAX_ANGLE <= bearing < 0.0:
                right_d = min(right_d, surface_dist)
                right_n += 1

        return front_d, left_d, right_d, left_n, right_n

    def compute(self, pose):
        front_d, left_d, right_d = self._sector_distances(pose)[:3]

        # ------------------------------------------------------------
        # Hybrid local avoidance
        #
        # FRONT:
        #   close -> strong local override
        #
        # SIDE:
        #   do not fully override A*
        #   return a side-bias request that PathFollower blends with A*
        # ------------------------------------------------------------

        # 1) Emergency front zone
        if front_d <= self.NEAR_DISTANCE:
            if right_d < left_d:
                self.last_turn_sign = +1.0   # turn left
            elif left_d < right_d:
                self.last_turn_sign = -1.0   # turn right

            self.avoid_lock = True
            self.clear_count = 0

            return (
                0.0,
                self.last_turn_sign * self.NEAR_ANGULAR,
                True,
                "front_emergency_spin",
            )

        # 2) Front reaction zone
        if front_d <= self.DETECT_DISTANCE:
            if not self.avoid_lock:
                if right_d + self.TURN_HYSTERESIS < left_d:
                    self.last_turn_sign = +1.0
                elif left_d + self.TURN_HYSTERESIS < right_d:
                    self.last_turn_sign = -1.0

                self.avoid_lock = True

            self.clear_count = 0

            return (
                0.12,
                self.last_turn_sign * 0.55,
                True,
                "front_slow_avoid",
            )

        # 3) Side-only obstacle
        # Do NOT use active=True because that would discard the A* angular command.
        # Instead ask PathFollower to blend a small steering correction.
        if right_d <= self.SIDE_REACT_DISTANCE and left_d > self.SIDE_REACT_DISTANCE:
            self.avoid_lock = False
            self.clear_count = 0
            return (
                None,
                +0.25,   # obstacle on right -> bias left
                False,
                "side_bias_left",
            )

        if left_d <= self.SIDE_REACT_DISTANCE and right_d > self.SIDE_REACT_DISTANCE:
            self.avoid_lock = False
            self.clear_count = 0
            return (
                None,
                -0.25,   # obstacle on left -> bias right
                False,
                "side_bias_right",
            )

        # Both sides near -> narrow corridor:
        # no side bias, trust A* and only slow down.
        if (
            left_d <= self.SIDE_REACT_DISTANCE
            and right_d <= self.SIDE_REACT_DISTANCE
        ):
            self.avoid_lock = False
            self.clear_count = 0
            return (
                0.18,
                0.0,
                False,
                "side_corridor_slow",
            )

        self.avoid_lock = False
        self.clear_count = 0
        return None, None, False, "clear"



class PathFollower:
    WAYPOINT_TOLERANCE = 0.50
    BACK_WAYPOINT_SKIP_ANGLE = math.radians(130.0)

    def __init__(
        self,
        path,
        controller: GoalSeekController = None,
        passthrough: bool = False,
    ):
        self.path = [(float(x), float(y)) for x, y in path]
        self.idx = 0
        self.ctrl = controller or GoalSeekController()
        # 순수 셀 추종 버전: 로컬 반응형 회피(LocalObstacleAvoider)를 쓰지 않음.
        # 장애물 회피는 전부 occupancy map(A*) 재계획 쪽에서 처리 -- 새로 못 가는
        # 셀이 라이다로 발견되면 participant_app 쪽에서 즉시 재계획을 트리거함.
        self._passthrough = bool(passthrough)

    def _skip_passed_or_bad_waypoints(self, pose):
        if not self.path:
            return

        x, y, yaw = pose

        while self.idx < len(self.path) - 1:
            wx, wy = self.path[self.idx]
            dist = math.hypot(wx - x, wy - y)

            # Normal reached waypoint skip.
            if dist < self.WAYPOINT_TOLERANCE:
                self.idx += 1
                continue

            # Avoid 180/360-degree spin toward a waypoint that ended up behind us.
            target_yaw = math.atan2(wy - y, wx - x)
            heading_err = _wrap_pi(target_yaw - yaw)

            if abs(heading_err) > self.BACK_WAYPOINT_SKIP_ANGLE and dist < 1.50:
                self.idx += 1
                continue

            break

    def compute(self, pose: tuple, dynamic_obstacles=None):
        # dynamic_obstacles는 인터페이스 호환용으로만 받고 쓰지 않음
        # (호출부에서 그대로 넘겨도 에러 안 나게).
        if not self.path:
            return 0.0, 0.0, True, {
                "idx": 0,
                "total": 0,
                "waypoint": None,
                "local_avoidance": False,
                "avoid_reason": "empty_path",
            }

        self._skip_passed_or_bad_waypoints(pose)

        wx, wy = self.path[self.idx]
        final = self.idx >= len(self.path) - 1
        cruise = (not final) or self._passthrough

        linear, angular, reached = self.ctrl.compute(
            pose,
            (wx, wy),
            cruise=cruise,
        )

        done = final and reached

        return linear, angular, done, {
            "idx": self.idx,
            "total": len(self.path),
            "waypoint": (wx, wy),
            "local_avoidance": False,
            "avoid_reason": "grid_only",
        }



class OccupancyGridPlanner:
    # Actual chassis footprint used for pose-aware collision checks.
    # X axis = robot forward direction, Y axis = robot lateral direction.
    ROBOT_LENGTH_M = 1.00
    ROBOT_WIDTH_M = 0.80
    ROBOT_FOOTPRINT_MARGIN_M = 0.00

    def __init__(self):
        self._grid = None
        self._original_grid = None
        self._custom_grid = None
        self._custom_grid2 = None
        self._map_mode = "ORIGINAL"
        self._res = None
        self._ox = None
        self._oy = None
        self._h = 0
        self._w = 0

    def update_from_msg(self, msg) -> bool:
        info = msg.info
        w = int(info.width)
        h = int(info.height)

        if w == 0 or h == 0:
            return False

        self._res = float(info.resolution)
        self._ox = float(info.origin.position.x)
        self._oy = float(info.origin.position.y)
        self._h = h
        self._w = w

        # 플랫폼에서 받은 원본 OccupancyGrid는 별도로 보존.
        self._original_grid = np.array(msg.data, dtype=np.int8).reshape(h, w)
        self._custom_grid = self._build_custom_grid()
        self._custom_grid2 = self._build_custom_grid2()

        # occupancy가 갱신되어도 현재 선택 모드는 유지.
        if self._map_mode == "CUSTOM":
            self._grid = self._custom_grid
        elif self._map_mode == "CUSTOM2":
            self._grid = self._custom_grid2
        else:
            self._grid = self._original_grid
        return True

    @property
    def ready(self) -> bool:
        return self._grid is not None

    @property
    def resolution(self):
        return self._res

    @property
    def map_mode(self):
        return self._map_mode

    def original_value_at(self, xy):
        """원본 OccupancyGrid에서 world XY의 셀값을 반환."""
        if self._original_grid is None or xy is None:
            return None
        r, c = self._w2c(float(xy[0]), float(xy[1]))
        if not self._in_bounds(r, c):
            return None
        return int(self._original_grid[r, c])

    def select_map_for_goal(self, target_xy):
        """원본 맵 목표 셀이 정확히 free(0)일 때만 ORIGINAL, 그 외는 CUSTOM."""
        value = self.original_value_at(target_xy)

        if value == 0:
            self._map_mode = "ORIGINAL"
            self._grid = self._original_grid
        else:
            # 100 / -1 / None(원본 맵 범위 밖) -> CUSTOM
            self._map_mode = "CUSTOM"
            self._grid = self._custom_grid

        return self._map_mode

    def _build_custom_grid(self):
        """좌표 폴리라인을 같은 origin/resolution의 별도 격자맵으로 변환."""
        if self._original_grid is None:
            return None

        custom = np.zeros_like(self._original_grid, dtype=np.int8)
        half_w = max(0, int(round((CUSTOM_MAP_LINE_WIDTH_M * 0.5) / self._res)))

        for group in CUSTOM_MAP_LINES:
            if len(group) < 2:
                continue
            for p0, p1 in zip(group, group[1:]):
                r0, c0 = self._w2c(float(p0[0]), float(p0[1]))
                r1, c1 = self._w2c(float(p1[0]), float(p1[1]))
                for r, c in self._line_cells(r0, c0, r1, c1):
                    if half_w <= 0:
                        if self._in_bounds(r, c):
                            custom[r, c] = 100
                        continue
                    for dr in range(-half_w, half_w + 1):
                        for dc in range(-half_w, half_w + 1):
                            if dr * dr + dc * dc > half_w * half_w:
                                continue
                            rr, cc = r + dr, c + dc
                            if self._in_bounds(rr, cc):
                                custom[rr, cc] = 100
        return custom

    def _build_custom_grid2(self):
        """배송 구간용 CUSTOM2 좌표 폴리라인을 별도 격자맵으로 변환."""
        if self._original_grid is None:
            return None

        custom = np.zeros_like(self._original_grid, dtype=np.int8)
        half_w = max(0, int(round((CUSTOM_MAP_LINE_WIDTH_M * 0.5) / self._res)))

        # 배송용 맵 좌표를 바꾸려면 파일 위쪽 CUSTOM_MAP2_LINES만 수정.
        for group in CUSTOM_MAP2_LINES:
            if len(group) < 2:
                continue
            for p0, p1 in zip(group, group[1:]):
                r0, c0 = self._w2c(float(p0[0]), float(p0[1]))
                r1, c1 = self._w2c(float(p1[0]), float(p1[1]))
                for r, c in self._line_cells(r0, c0, r1, c1):
                    if half_w <= 0:
                        if self._in_bounds(r, c):
                            custom[r, c] = 100
                        continue
                    for dr in range(-half_w, half_w + 1):
                        for dc in range(-half_w, half_w + 1):
                            if dr * dr + dc * dc > half_w * half_w:
                                continue
                            rr, cc = r + dr, c + dc
                            if self._in_bounds(rr, cc):
                                custom[rr, cc] = 100
        return custom

    def _w2c(self, wx, wy):
        col = int((wx - self._ox) / self._res)
        row = int((wy - self._oy) / self._res)
        return row, col

    def _c2w(self, row, col):
        wx = self._ox + (col + 0.5) * self._res
        wy = self._oy + (row + 0.5) * self._res
        return wx, wy

    def plan(
        self,
        start_xy,
        goal_xy,
        inflation_m=0.64,  # 경로 중간은 기존 inflation 유지
        margin_m=12.0,
        snap_m=1.0,
        extra_cells=None,
        debug_dump_path=None,
        debug_target_xy=None,
        final_goal_yaw=None,
        allow_goal_in_inflation=False,
        hard_noinflate_cells=None,
    ):
        """debug_dump_path: 지정하면 매 호출마다 trav 그리드를 PPM 이미지로 덮어쓰기 저장.
        "왜 검정(못감)인지" 원인별로 색을 나눠서 보여줌 -- _save_debug_grid_image 참고.
        A*가 실패해도(경로 None) 저장은 함 -- "왜 못 뚫었는지" 확인용.
        """
        if not self.ready:
            return None

        sr, sc = self._w2c(*start_xy)
        gr, gc = self._w2c(*goal_xy)

        if not (self._in_bounds(sr, sc) and self._in_bounds(gr, gc)):
            return None

        margin_c = int(math.ceil(margin_m / self._res))
        rmin = max(0, min(sr, gr) - margin_c)
        rmax = min(self._h, max(sr, gr) + margin_c + 1)
        cmin = max(0, min(sc, gc) - margin_c)
        cmax = min(self._w, max(sc, gc) + margin_c + 1)

        sub = self._grid[rmin:rmax, cmin:cmax]
        trav = sub == 0

        # 원인 구분용 마스크: 침식/라이다 반영 전, 원본 occupancy만 반영한 상태를 보존.
        trav_before_extra = trav.copy()

        # 일반 LiDAR 장애물은 기존처럼 inflation 대상.
        if extra_cells:
            self._block_cells(trav, extra_cells, rmin, cmin)

        # OccupancyGrid와 무관한 순수 LiDAR 마스크.
        # 원본 맵이 검정/unknown이어도 LiDAR가 본 셀은 디버그에서 주황색으로 표시.
        lidar_blocked = self._extra_cells_mask(
            trav.shape, extra_cells, rmin, cmin
        )

        extra_blocked = trav_before_extra & ~trav
        trav_before_erosion = trav.copy()

        infl_c = int(round(inflation_m / self._res))
        if infl_c > 0:
            trav = self._erode(trav, infl_c)

        # 분홍은 일반 occupancy/LiDAR 장애물의 inflation만 표시.
        eroded_only = trav_before_erosion & ~trav

        # 물건 주변 keepout은 inflation 이후에 hard-block.
        # 따라서 주황 keepout 자체는 막히지만, 그 때문에 추가 분홍 halo는 생기지 않는다.
        hard_blocked = np.zeros_like(trav, dtype=bool)
        if hard_noinflate_cells:
            before_hard = trav.copy()
            self._block_cells(trav, hard_noinflate_cells, rmin, cmin)
            hard_blocked = before_hard & ~trav
            eroded_only[hard_blocked] = False

        # ------------------------------------------------------------
        # FINAL GOAL ONLY inflation exception
        #
        # 경로 중간은 계속 inflation된 trav를 사용한다.
        # 단, 최종 standoff 자세가 실제 직사각형 footprint + yaw 기준으로
        # 안전하면 그 최종 footprint 영역만 다시 통과 가능하게 복원한다.
        # 실제 occupancy/LiDAR 장애물 셀은 절대 복원하지 않는다.
        # ------------------------------------------------------------
        if allow_goal_in_inflation and final_goal_yaw is not None:
            footprint_base = trav_before_erosion.copy()
            if hard_noinflate_cells:
                self._block_cells(
                    footprint_base, hard_noinflate_cells, rmin, cmin
                )

            if self._footprint_is_clear(
                footprint_base,
                goal_xy[0],
                goal_xy[1],
                final_goal_yaw,
                rmin,
                cmin,
            ):
                for grf, gcf in self._footprint_cells(
                    goal_xy[0], goal_xy[1], final_goal_yaw
                ):
                    lrf = grf - rmin
                    lcf = gcf - cmin
                    if (
                        0 <= lrf < trav.shape[0]
                        and 0 <= lcf < trav.shape[1]
                        and trav_before_erosion[lrf, lcf]
                        and not hard_blocked[lrf, lcf]
                    ):
                        trav[lrf, lcf] = True

                # 목표 중심도 실제 장애물만 아니면 반드시 복원.
                _ggr, _ggc = self._w2c(*goal_xy)
                _lgr, _lgc = _ggr - rmin, _ggc - cmin
                if (
                    0 <= _lgr < trav.shape[0]
                    and 0 <= _lgc < trav.shape[1]
                    and trav_before_erosion[_lgr, _lgc]
                    and not hard_blocked[_lgr, _lgc]
                ):
                    trav[_lgr, _lgc] = True

        raw_ls = (sr - rmin, sc - cmin)
        raw_lg = (gr - rmin, gc - cmin)
        ls = raw_ls
        lg = raw_lg

        target_cell = None
        if debug_target_xy is not None:
            _tr, _tc = self._w2c(*debug_target_xy)
            target_cell = (_tr - rmin, _tc - cmin)

        H, W = trav.shape
        goal_free = (
            0 <= lg[0] < H
            and 0 <= lg[1] < W
            and bool(trav[lg[0], lg[1]])
        )

        ls = self._snap(trav, ls, snap_m)
        lg = self._snap(trav, lg, snap_m)

        if not goal_free and lg is not None:
            snapped_wx, snapped_wy = self._c2w(lg[0] + rmin, lg[1] + cmin)
            off_m = math.hypot(snapped_wx - goal_xy[0], snapped_wy - goal_xy[1])
            log.warning(
                "[A*] 목표(%.2f, %.2f)가 안전마진/장애물에 걸려서 (%.2f, %.2f)로 스냅됨 "
                "(%.2fm 벗어남) -- 로봇이 실제 목표 대신 이 지점에서 '도착'으로 처리될 수 있음",
                goal_xy[0], goal_xy[1], snapped_wx, snapped_wy, off_m,
            )

        if ls is None or lg is None:
            if debug_dump_path:
                self._save_debug_grid_image(
                    sub, trav, extra_blocked, eroded_only,
                    ls, lg, None, debug_dump_path, target=target_cell,
                    lidar_blocked=lidar_blocked,
                )
            return None

        cells = self._astar(trav, ls, lg)

        # OccupancyGrid는 원본 그대로 지킨다.
        # 경로가 없으면 OccupancyGrid를 무시하는 2차 fallback을 사용하지 않는다.
        if not cells:
            if debug_dump_path:
                self._save_debug_grid_image(
                    sub, trav, extra_blocked, eroded_only,
                    ls, lg, None, debug_dump_path, target=target_cell,
                    lidar_blocked=lidar_blocked,
                )
            return None

        plan_grid = trav
        cells = self._simplify(plan_grid, cells)

        if debug_dump_path:
            self._save_debug_grid_image(
                sub, trav, extra_blocked, eroded_only,
                ls, lg, cells, debug_dump_path, target=target_cell,
                lidar_blocked=lidar_blocked,
            )

        path = [
            self._c2w(r + rmin, c + cmin)
            for r, c in cells
        ]

        if path:
            if goal_free:
                path[-1] = (
                    float(goal_xy[0]),
                    float(goal_xy[1]),
                )

            if len(path) > 1:
                path = path[1:]

        return path

    def dump_debug_view(
        self,
        start_xy,
        goal_xy,
        inflation_m=0.64,
        margin_m=12.0,
        extra_cells=None,
        path_world=None,
        debug_dump_path=None,
        target_xy=None,
        robot_pose=None,
        target_keepout_radius_m=0.0,
        hard_noinflate_cells=None,
    ):
        """A* 없이 "지금 이 순간의 grid 상태"만 가볍게 찍어서 저장.

        plan()은 need_replan일 때만 호출돼서(경로가 안 막히면 재계획 자체를
        안 하니까) 디버그 이미지도 그때만 갱신됨 -- 잘 가고 있는 동안엔
        이미지가 멈춰있는 것처럼 보임. 이 함수는 재계획 여부와 무관하게
        주기적으로 불러서 "지금 상태"를 계속 보여주는 용도. A* 재탐색을
        안 하니 비용도 훨씬 쌈. path_world를 주면(직전 성공한 경로) 참고용
        파란 선으로 같이 그려줌(재탐색된 경로 아님, 마지막으로 계획됐던 것).
        """
        if not self.ready or debug_dump_path is None:
            return

        sr, sc = self._w2c(*start_xy)
        gr, gc = self._w2c(*goal_xy)

        if not (self._in_bounds(sr, sc) and self._in_bounds(gr, gc)):
            return

        margin_c = int(math.ceil(margin_m / self._res))
        rmin = max(0, min(sr, gr) - margin_c)
        rmax = min(self._h, max(sr, gr) + margin_c + 1)
        cmin = max(0, min(sc, gc) - margin_c)
        cmax = min(self._w, max(sc, gc) + margin_c + 1)

        sub = self._grid[rmin:rmax, cmin:cmax]
        trav = sub == 0
        trav_before_extra = trav.copy()

        if extra_cells:
            self._block_cells(trav, extra_cells, rmin, cmin)

        lidar_blocked = self._extra_cells_mask(
            trav.shape, extra_cells, rmin, cmin
        )
        extra_blocked = trav_before_extra & ~trav
        trav_before_erosion = trav.copy()

        infl_c = int(round(inflation_m / self._res))
        if infl_c > 0:
            trav = self._erode(trav, infl_c)

        eroded_only = trav_before_erosion & ~trav

        # 물건 keepout은 inflation 이후 hard-block -> keepout 주변에 분홍 halo 없음.
        if hard_noinflate_cells:
            before_hard = trav.copy()
            self._block_cells(trav, hard_noinflate_cells, rmin, cmin)
            hard_mask = before_hard & ~trav
            eroded_only[hard_mask] = False

        ls = (sr - rmin, sc - cmin)
        lg = (gr - rmin, gc - cmin)

        cells = None
        if path_world:
            H, W = trav.shape
            cells = []
            for wx, wy in path_world:
                r, c = self._w2c(wx, wy)
                lr, lc = r - rmin, c - cmin
                if 0 <= lr < H and 0 <= lc < W:
                    cells.append((lr, lc))

        target_cell = None
        if target_xy is not None:
            _tr, _tc = self._w2c(*target_xy)
            target_cell = (_tr - rmin, _tc - cmin)

        robot_cells = None
        if robot_pose is not None:
            rx, ry, ryaw = robot_pose
            robot_cells = []
            for grb, gcb in self._footprint_cells(rx, ry, ryaw):
                lrb, lcb = grb - rmin, gcb - cmin
                if 0 <= lrb < trav.shape[0] and 0 <= lcb < trav.shape[1]:
                    robot_cells.append((lrb, lcb))

        # 물건 주변 keepout은 일반 inflation(분홍)과 별개로 디버그에서
        # 확실히 주황색으로 보이도록 전용 셀 목록을 만든다.
        target_keepout_cells = None
        if target_xy is not None and target_keepout_radius_m > 0.0:
            tgr, tgc = self._w2c(*target_xy)
            rad_c = int(math.ceil(target_keepout_radius_m / self._res))
            target_keepout_cells = []

            for dr in range(-rad_c, rad_c + 1):
                for dc in range(-rad_c, rad_c + 1):
                    grk = tgr + dr
                    gck = tgc + dc
                    if not self._in_bounds(grk, gck):
                        continue

                    wxk, wyk = self._c2w(grk, gck)
                    if math.hypot(
                        wxk - target_xy[0],
                        wyk - target_xy[1],
                    ) <= target_keepout_radius_m:
                        lrk = grk - rmin
                        lck = gck - cmin
                        if 0 <= lrk < trav.shape[0] and 0 <= lck < trav.shape[1]:
                            target_keepout_cells.append((lrk, lck))

        self._save_debug_grid_image(
            sub, trav, extra_blocked, eroded_only,
            ls, lg, cells, debug_dump_path, target=target_cell,
            robot_cells=robot_cells,
            target_keepout_cells=target_keepout_cells,
            lidar_blocked=lidar_blocked,
        )

    @staticmethod
    def _save_debug_grid_image(
        sub, trav, extra_blocked, eroded_only, ls, lg, cells, path_out,
        target=None, robot_cells=None, target_keepout_cells=None,
        lidar_blocked=None,
    ):
        """trav 그리드를 "왜 못 가는지" 원인별로 색을 나눠 PPM 이미지로 저장.

        - 밝은 회색: 갈 수 있는 셀
        - 진한 회색/검정: occupancy map 원본에서부터 장애물(값=100)
        - 보라: occupancy map이 아직 못 본 영역(unknown, 값=-1) -- 장애물 아님!
        - 주황: LiDAR가 감지한 셀(원본 occupancy가 검정/보라여도 주황으로 덮어 표시)
        - 분홍(연한 빨강): 위 어느 쪽도 아니고, 안전마진(침식) 때문에만 막힌 셀
          (근처 장애물/라이다점과의 거리가 inflation_m보다 가까움)
        - 주황(밝음): 현재 yaw를 반영한 실제 로봇 직사각형 footprint(1.0m x 0.8m)
        - 초록: 로봇 중심 / 빨강: 내비 목표(standoff) 셀
        - 진한 파랑: 실제 물건 목표 / 하늘색: A*가 계획한 경로

        "장애물 없는 곳도 검정"으로 보이면: 보라(unknown)면 지도가 그 구역을 안
        가봐서 그런 거고, 분홍(erosion)이면 안전마진이 과한 거고, 주황이 넓게
        퍼져있으면 라이다 노이즈 하나가 마진 때문에 넓게 지워진 것.
        """
        try:
            H, W = trav.shape
            img = np.zeros((H, W, 3), dtype=np.uint8)

            occ_blocked = sub == 100          # occupancy map 원본 장애물
            unknown = sub == -1               # 아직 못 본 영역 (장애물 아님)

            img[trav] = (230, 230, 230)                 # 갈 수 있음
            img[occ_blocked] = (15, 15, 15)              # 원본 장애물 -- 진짜 못 감
            img[unknown & ~occ_blocked] = (150, 60, 200)  # unknown -- 장애물 아닌데 못 감
            img[eroded_only] = (255, 170, 190)           # 안전마진 때문에만 못 감
            img[extra_blocked] = (255, 140, 0)

            # 원본 occupancy가 검정/보라여도 LiDAR가 본 셀은 주황색으로 덮어 표시.
            if lidar_blocked is not None:
                img[lidar_blocked] = (255, 140, 0)

            # 물건 주변 강제 접근금지 영역은 분홍보다 우선해서 주황색으로 표시.
            if target_keepout_cells:
                for rr, rc in target_keepout_cells:
                    if 0 <= rr < H and 0 <= rc < W:
                        img[rr, rc] = (255, 120, 0)

            if cells:
                for (r, c) in cells:
                    if 0 <= r < H and 0 <= c < W:
                        img[r, c] = (100, 180, 255)

            # Current robot body: actual yaw-rotated rectangular footprint.
            # Keep LiDAR points orange too, but use a brighter orange for the body.
            if robot_cells:
                for rr, rc in robot_cells:
                    if 0 <= rr < H and 0 <= rc < W:
                        img[rr, rc] = (255, 190, 40)

            if ls is not None:
                sr, sc = ls
                r0, r1 = max(0, sr - 1), min(H, sr + 2)
                c0, c1 = max(0, sc - 1), min(W, sc + 2)
                img[r0:r1, c0:c1] = (0, 200, 0)

            if lg is not None:
                gr, gc = lg
                r0, r1 = max(0, gr - 2), min(H, gr + 3)
                c0, c1 = max(0, gc - 2), min(W, gc + 3)
                img[r0:r1, c0:c1] = (230, 30, 30)

            if target is not None:
                tr, tc = target
                r0, r1 = max(0, tr - 2), min(H, tr + 3)
                c0, c1 = max(0, tc - 2), min(W, tc + 3)
                img[r0:r1, c0:c1] = (20, 40, 230)  # 실제 물건 목표 -- 진한 파랑

            # world 좌표계 기준 위쪽이 +y가 되도록 행 순서 뒤집기 (아니면 상하반전돼 보임)
            img = img[::-1]

            tmp_path = str(path_out) + ".tmp"
            with open(tmp_path, "wb") as f:
                f.write(("P6\n%d %d\n255\n" % (W, H)).encode("ascii"))
                f.write(img.tobytes())
            os.replace(tmp_path, path_out)  # 원자적 교체 -- 보다가 중간에 깨진 파일 안 보이게
        except Exception as e:
            log.warning("[grid-debug] 이미지 저장 실패: %s", e)

    # 이 거리(셀 개수) 이내로 가까운 두 점은 그 사이도 선으로 이어서 막음.
    # 성긴 라이다 포인트가 실제로는 연속된 벽/물체인데 틈으로 보여서 뚫고
    # 들어가는 걸 방지. 셀 단위라 실제 거리(m) = 이 값 * self._res.
    CONNECT_MAX_CELLS = 4

    def _block_cells(self, trav, cells, rmin, cmin):
        """extra_cells의 각 (row, col) 전역 셀 좌표를 not-traversable로 마킹.

        점 하나하나 막는 것에 더해, CONNECT_MAX_CELLS 이내로 가까운 점끼리는
        그 사이 셀도 선으로 이어서 막음 -- 성긴 라이다 리턴이 실제로는 이어진
        벽/물체인데 점 사이 틈으로 잘못 통과하는 걸 막기 위함.
        안전 여유(로봇 반경)는 여기서가 아니라 plan()의 inflation_m 침식
        단계에서 맵 전체에 균일하게 적용됨.
        """
        H, W = trav.shape
        cell_list = list(cells)

        for gr, gc in cell_list:
            lr, lc = gr - rmin, gc - cmin
            if 0 <= lr < H and 0 <= lc < W:
                trav[lr, lc] = False

        # 공간 버킷으로 가까운 쌍만 찾기 (O(N^2) 방지)
        bucket = self.CONNECT_MAX_CELLS
        buckets = {}
        for i, (gr, gc) in enumerate(cell_list):
            buckets.setdefault((gr // bucket, gc // bucket), []).append(i)

        max_d2 = self.CONNECT_MAX_CELLS * self.CONNECT_MAX_CELLS
        seen_pairs = set()
        for (br, bc), idxs in buckets.items():
            neighbor_idxs = []
            for dbr in (-1, 0, 1):
                for dbc in (-1, 0, 1):
                    neighbor_idxs.extend(buckets.get((br + dbr, bc + dbc), ()))
            for i in idxs:
                gr1, gc1 = cell_list[i]
                for j in neighbor_idxs:
                    if j <= i:
                        continue
                    pair = (i, j)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    gr2, gc2 = cell_list[j]
                    d2 = (gr1 - gr2) ** 2 + (gc1 - gc2) ** 2
                    if 0 < d2 <= max_d2:
                        for r, c in self._line_cells(gr1, gc1, gr2, gc2):
                            lr, lc = r - rmin, c - cmin
                            if 0 <= lr < H and 0 <= lc < W:
                                trav[lr, lc] = False


    def _extra_cells_mask(self, shape, extra_cells, rmin, cmin):
        """LiDAR blocked mask independent of the base OccupancyGrid."""
        tmp = np.ones(shape, dtype=bool)
        if extra_cells:
            self._block_cells(tmp, extra_cells, rmin, cmin)
        return ~tmp


    @staticmethod
    def _line_cells(r0, c0, r1, c1):
        """(r0,c0)-(r1,c1) 사이 셀들을 Bresenham으로 나열 (양 끝점 포함)."""
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        err = dr - dc
        r, c = r0, c0
        out = []
        while True:
            out.append((r, c))
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc
        return out

    def _footprint_cells(self, wx, wy, yaw, length_m=None, width_m=None, margin_m=None):
        """Return global grid cells covered by the yaw-rotated rectangular robot footprint.

        The rectangle is centered at (wx, wy):
          forward/back = length_m
          left/right    = width_m
        """
        if not self.ready:
            return []

        length_m = self.ROBOT_LENGTH_M if length_m is None else float(length_m)
        width_m = self.ROBOT_WIDTH_M if width_m is None else float(width_m)
        margin_m = self.ROBOT_FOOTPRINT_MARGIN_M if margin_m is None else float(margin_m)

        half_l = 0.5 * length_m + margin_m
        half_w = 0.5 * width_m + margin_m
        radius = math.hypot(half_l, half_w)
        rad_c = int(math.ceil(radius / self._res)) + 1

        cr, cc = self._w2c(wx, wy)
        cyaw = math.cos(yaw)
        syaw = math.sin(yaw)

        cells = []
        for gr in range(cr - rad_c, cr + rad_c + 1):
            for gc in range(cc - rad_c, cc + rad_c + 1):
                if not self._in_bounds(gr, gc):
                    continue

                px, py = self._c2w(gr, gc)
                dx = px - wx
                dy = py - wy

                # world -> robot frame
                forward = cyaw * dx + syaw * dy
                lateral = -syaw * dx + cyaw * dy

                # Half a cell is added so cells touched by the footprint edge
                # are conservatively included.
                cell_pad = 0.5 * self._res
                if (
                    abs(forward) <= half_l + cell_pad
                    and abs(lateral) <= half_w + cell_pad
                ):
                    cells.append((gr, gc))

        return cells

    def _footprint_is_clear(self, base_trav, wx, wy, yaw, rmin, cmin):
        """True only if the full yaw-rotated rectangular chassis is on free cells.

        base_trav is the map after LiDAR extra_cells are blocked but BEFORE circular
        inflation/erosion. This is intentional: candidate standoff feasibility should
        use the actual 1.0 x 0.8 m footprint and its yaw, not an orientation-independent
        circumscribed circle.
        """
        H, W = base_trav.shape

        footprint = self._footprint_cells(wx, wy, yaw)
        if not footprint:
            return False

        for gr, gc in footprint:
            lr = gr - rmin
            lc = gc - cmin

            if not (0 <= lr < H and 0 <= lc < W):
                return False

            if not base_trav[lr, lc]:
                return False

        return True

    def find_clear_standoff(
        self,
        goal_xy,
        standoff_m,
        extra_cells=None,
        inflation_m=0.64,
        prefer_xy=None,
        num_candidates=24,
        margin_m=3.0,
        hard_noinflate_cells=None,
        goal_keepout_clearance_m=0.0,
    ):
        """Choose a safe robot goal on the standoff circle around the object.

        Selection rules:
          1) Robot's full rectangular footprint (length/width + yaw) must fit.
          2) Candidate -> object approach line must be open.
          3) Among valid candidates, choose the one closest to the CURRENT robot.
          4) If distances are effectively tied, prefer larger obstacle clearance.

        Returns: ((x, y), approach_yaw), where approach_yaw faces the object.
        """
        if not self.ready:
            return None

        gr, gc = self._w2c(*goal_xy)
        if not self._in_bounds(gr, gc):
            return None

        margin_c = int(math.ceil(margin_m / self._res))
        rmin = max(0, gr - margin_c)
        rmax = min(self._h, gr + margin_c + 1)
        cmin = max(0, gc - margin_c)
        cmax = min(self._w, gc + margin_c + 1)

        sub = self._grid[rmin:rmax, cmin:cmax]

        # 일반 장애물은 inflation 대상.
        inflate_base = sub == 0
        if extra_cells:
            self._block_cells(inflate_base, extra_cells, rmin, cmin)

        # 실제 footprint 충돌검사에는 물건 hard keepout도 포함.
        base_trav = inflate_base.copy()
        if hard_noinflate_cells:
            self._block_cells(base_trav, hard_noinflate_cells, rmin, cmin)

        # 접근선 grid: 일반 장애물만 inflation, 물건 keepout은 그 후 hard-block.
        trav = inflate_base.copy()
        infl_c = int(round(inflation_m / self._res))
        if infl_c > 0:
            trav = self._erode(trav, infl_c)
        if hard_noinflate_cells:
            self._block_cells(trav, hard_noinflate_cells, rmin, cmin)

        lg = (gr - rmin, gc - cmin)
        H, W = trav.shape
        if not (0 <= lg[0] < H and 0 <= lg[1] < W):
            return None

        angles = [2.0 * math.pi * i / num_candidates for i in range(num_candidates)]

        # Starting with the robot-facing side makes ties deterministic and avoids
        # unnecessary wrap-around when several candidates are almost identical.
        if prefer_xy is not None:
            base_ang = math.atan2(
                prefer_xy[1] - goal_xy[1],
                prefer_xy[0] - goal_xy[0],
            )
            angles.sort(key=lambda a: abs(_wrap_pi(a - base_ang)))

        obstacle_pts = []
        if extra_cells:
            for gr2, gc2 in extra_cells:
                obstacle_pts.append(self._c2w(gr2, gc2))
        if hard_noinflate_cells:
            for gr2, gc2 in hard_noinflate_cells:
                obstacle_pts.append(self._c2w(gr2, gc2))

        best = None
        best_score = None

        for rank, ang in enumerate(angles):
            cx = goal_xy[0] + standoff_m * math.cos(ang)
            cy = goal_xy[1] + standoff_m * math.sin(ang)

            cr, cc = self._w2c(cx, cy)
            lcr, lcc = cr - rmin, cc - cmin
            if not (0 <= lcr < H and 0 <= lcc < W):
                continue

            # 빨간 목표점 중심은 "모든 주황 장애물"에서 최소 clearance 이상 떨어져야 한다.
            # - hard_noinflate_cells: 물건 주변 주황 keepout
            # - extra_cells: LiDAR로 감지된 주황 장애물 셀
            if goal_keepout_clearance_m > 0.0:
                min_center_clear = float("inf")

                if hard_noinflate_cells:
                    for hgr, hgc in hard_noinflate_cells:
                        hwx, hwy = self._c2w(hgr, hgc)
                        d = math.hypot(cx - hwx, cy - hwy)
                        if d < min_center_clear:
                            min_center_clear = d

                if extra_cells:
                    for egr, egc in extra_cells:
                        ewx, ewy = self._c2w(egr, egc)
                        d = math.hypot(cx - ewx, cy - ewy)
                        if d < min_center_clear:
                            min_center_clear = d

                if min_center_clear < goal_keepout_clearance_m:
                    continue

            # At this standoff pose the robot will face the object.
            approach_yaw = math.atan2(
                goal_xy[1] - cy,
                goal_xy[0] - cx,
            )

            # Critical change: test the REAL rectangular chassis at this yaw.
            if not self._footprint_is_clear(
                base_trav,
                cx,
                cy,
                approach_yaw,
                rmin,
                cmin,
            ):
                continue

            # The final approach toward the object must also be open.
            # The object's own inflation region is intentionally ignored near lg.
            if not self._los_to_object(
                trav,
                (lcr, lcc),
                lg,
                margin_cells=infl_c,
            ):
                continue

            if prefer_xy is not None:
                robot_dist = math.hypot(
                    cx - prefer_xy[0],
                    cy - prefer_xy[1],
                )
            else:
                robot_dist = 0.0

            if obstacle_pts:
                clearance = min(
                    math.hypot(cx - ox, cy - oy)
                    for ox, oy in obstacle_pts
                )
            else:
                clearance = float("inf")

            # Closest to the current robot FIRST.
            # Round to 1 cm so tiny floating-point differences do not beat
            # a meaningfully safer clearance when candidates are effectively tied.
            score = (
                -round(robot_dist, 2),
                clearance,
                -rank,
            )

            if best_score is None or score > best_score:
                best_score = score
                best = ((cx, cy), approach_yaw)

        return best

    def _in_bounds(self, r, c):
        return 0 <= r < self._h and 0 <= c < self._w

    @staticmethod
    def _erode(mask, iters):
        g = mask

        for _ in range(iters):
            e = np.zeros_like(g)

            e[1:-1, 1:-1] = (
                g[1:-1, 1:-1]
                & g[:-2, 1:-1]
                & g[2:, 1:-1]
                & g[1:-1, :-2]
                & g[1:-1, 2:]
                & g[:-2, :-2]
                & g[:-2, 2:]
                & g[2:, :-2]
                & g[2:, 2:]
            )

            g = e

        return g

    @staticmethod
    def _snap(trav, cell, snap_m):
        r, c = cell
        H, W = trav.shape

        if not (0 <= r < H and 0 <= c < W):
            return None

        if trav[r, c]:
            return cell

        max_rad = int(max(1, round(snap_m / 0.05)))

        for rad in range(1, max_rad + 1):
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if max(abs(dr), abs(dc)) != rad:
                        continue

                    nr = r + dr
                    nc = c + dc

                    if (
                        0 <= nr < H
                        and 0 <= nc < W
                        and trav[nr, nc]
                    ):
                        return nr, nc

        return None

    @staticmethod
    def _astar(grid, start, goal):
        H, W = grid.shape

        def h(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        openh = [(h(start, goal), 0.0, start)]
        gscore = {start: 0.0}
        came = {}

        while openh:
            _, gc, cur = heapq.heappop(openh)

            if cur == goal:
                path = [cur]

                while cur in came:
                    cur = came[cur]
                    path.append(cur)

                return path[::-1]

            if gc > gscore.get(cur, 1e18):
                continue

            r, c = cur

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue

                    nr = r + dr
                    nc = c + dc

                    if (
                        not (0 <= nr < H and 0 <= nc < W)
                        or not grid[nr, nc]
                    ):
                        continue

                    if (
                        dr != 0
                        and dc != 0
                        and not (
                            grid[r + dr, c]
                            and grid[r, c + dc]
                        )
                    ):
                        continue

                    step = SQRT2 if (dr and dc) else 1.0
                    ng = gc + step
                    nb = (nr, nc)

                    if ng < gscore.get(nb, 1e18):
                        gscore[nb] = ng
                        came[nb] = cur

                        heapq.heappush(
                            openh,
                            (ng + h(nb, goal), ng, nb),
                        )

        return None

    @classmethod
    def _simplify(cls, grid, path):
        if len(path) <= 2:
            return path

        out = [path[0]]
        i = 0
        n = len(path)

        while i < n - 1:
            j = n - 1

            while (
                j > i + 1
                and not cls._los(grid, path[i], path[j])
            ):
                j -= 1

            out.append(path[j])
            i = j

        return out

    @staticmethod
    def _los(grid, a, b):
        r0, c0 = a
        r1, c1 = b

        dr = abs(r1 - r0)
        dc = abs(c1 - c0)

        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1

        err = dr - dc
        r, c = r0, c0

        while True:
            if not grid[r, c]:
                return False

            if (r, c) == (r1, c1):
                return True

            e2 = 2 * err

            if e2 > -dc:
                err -= dc
                r += sr

            if e2 < dr:
                err += dr
                c += sc

    @staticmethod
    def _los_to_object(grid, a, b, margin_cells=0):
        """a->b 직선상의 셀들이 통과 가능한지 체크하되, b 주변 margin_cells
        반경 안(=물체의 안전마진 구간, 정의상 항상 막혀있음)은 검사 안 함.

        b가 목표 물체 위치(=주변 inflation 마진 때문에 그 반경 안이 전부
        막혀있는 셀)일 때 쓰는 용도. 그 마진 반경 안까지만 접근하면 되는
        거지 물체 자체나 그 바로 앞까지 "통과 가능"할 필요는 없음.
        """
        r0, c0 = a
        r1, c1 = b

        dr = abs(r1 - r0)
        dc = abs(c1 - c0)

        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1

        err = dr - dc
        r, c = r0, c0

        while True:
            if margin_cells > 0 and max(abs(r - r1), abs(c - c1)) <= margin_cells:
                return True

            if (r, c) == (r1, c1):
                return True

            if not grid[r, c]:
                return False

            e2 = 2 * err

            if e2 > -dc:
                err -= dc
                r += sr

            if e2 < dr:
                err += dr
                c += sc


def _default_data_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "mock_demo_data.yaml")


def load_navigation_config(path: str | None = None) -> dict:
    path = path or os.environ.get("MARC_NAVIGATION_CONFIG") or _default_data_path()

    if not os.path.exists(path):
        return {}

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def points_excluding_goal(points, goal_xy, exclude_m=0.6):
    """목표 지점 exclude_m 반경 안의 라이다 점을 제외 (목표를 못 가는 셀로 막지 않게).

    원 근사가 없어졌으므로 (x, y, r) 대신 (x, y) 점 리스트를 받는다.
    """
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    ex2 = float(exclude_m) ** 2

    out = []

    for x, y in points:
        if (x - gx) ** 2 + (y - gy) ** 2 <= ex2:
            continue

        out.append((x, y))

    return out


# 이전 원-기반 인터페이스 이름으로도 호출 가능하게 별칭 유지 (호출부 혼동 방지용).
obstacles_excluding_goal = points_excluding_goal

