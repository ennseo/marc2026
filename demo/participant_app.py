"""A*-only occupancy-grid navigation with LiDAR cell blocking and CUSTOM curb recovery."""

import logging
import math
import os, sys
import threading
import time

# Resolve the repository from this file, independently of its folder name or cwd.
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, repo_root)

from marc_sdk import MARCClient
from marc_sdk.types import GroundingResult

from agent_vla import VLAGrounding
from agent_navigation import (
    OccupancyGridPlanner, PathFollower, GoalSeekController,
    obstacles_excluding_goal, load_navigation_config,
)
                                                       
from lidar_obstacles_puregrid import cloud_to_blocked_points

                                                                                    
                     
from arm_pick import run_pick_sequence
from manipulation.grasp import GraspGenerator

log = logging.getLogger("participant_app")

STAGE2_NAV_TIMEOUT_S = 1800.0
STAGE2_NAV_RATE_HZ = 10.0
STAGE2_NAV_DT = 1.0 / STAGE2_NAV_RATE_HZ
STAGE2_INFLATION_M = 0.70                                                                        

                                                   
                                           
STAGE2_GRID_DEBUG_IMG = "/tmp/marc_trav_debug.ppm"

STAGE2_STUCK_TIME_S = 5.0                                           
STAGE2_STUCK_MOVE_M = 0.03                                    
GOAL_TOLERANCE = 0.7                             

                                
                                                    
LINEAR_ACCEL_MPS2 = 0.35
LINEAR_DECEL_MPS2 = 0.55
ANGULAR_ACCEL_RPS2 = 1.20
ANGULAR_DECEL_RPS2 = 1.80

                                                 
                                      
STAGE2_BACKUP_LINEAR = -0.15                  
STAGE2_BACKUP_TIME_S = 1.2                     
STAGE2_BACKUP_COOLDOWN_S = 3.0                                               

                                                           

                                                     
                              
STAGE2_SPIN_ANGULAR = 1.80                                                
STAGE2_SPIN_DURATION_S = (2.0 * math.pi / STAGE2_SPIN_ANGULAR) * 1.05                
STAGE2_SPIN_COOLDOWN_S = 4.0                                        

                                                 

              
EMERGENCY_STOP_DIST_M = 0.50                        
EMERGENCY_STOP_HALF_WIDTH_M = 0.40

ORIGINAL_FRONT_FORCE_DIST_M = 0.75
ORIGINAL_FRONT_FORCE_HALF_WIDTH_M = 0.55

                                            

EMERGENCY_HARD_STOP_HOLD_S = 0.3

                                               

                                                      

                                   
STATIC_CELL_STALE_S = 1.5                                  
NEAR_TARGET_LIDAR_DIST_M = 5.0

                       
                                                              
ORIGINAL_STATIC_CELL_MIN_HITS = 1
ORIGINAL_NEAR_CONFIRM_HITS = 1
ORIGINAL_LIDAR_NORMAL_STRIDE = 1
ORIGINAL_LIDAR_NEAR_STRIDE = 1
ORIGINAL_LIDAR_NORMAL_DENSITY_MIN = 2
ORIGINAL_LIDAR_NEAR_DENSITY_MIN = 1
ORIGINAL_LIDAR_NORMAL_DENSITY_RADIUS_M = 0.55
ORIGINAL_LIDAR_NEAR_DENSITY_RADIUS_M = 0.50
ORIGINAL_LIDAR_NORMAL_Z_MIN = -0.15
ORIGINAL_LIDAR_NORMAL_Z_MAX = 2.0
ORIGINAL_LIDAR_NEAR_Z_MIN = -0.32
ORIGINAL_LIDAR_NEAR_Z_MAX = 2.00

                     

CUSTOM_STATIC_CELL_MIN_HITS = 1
CUSTOM_NEAR_CONFIRM_HITS = 1
CUSTOM_LIDAR_NORMAL_STRIDE = 2
CUSTOM_LIDAR_NEAR_STRIDE = 1
CUSTOM_LIDAR_NORMAL_DENSITY_MIN = 3
CUSTOM_LIDAR_NEAR_DENSITY_MIN = 3
CUSTOM_LIDAR_NORMAL_DENSITY_RADIUS_M = 0.30
CUSTOM_LIDAR_NEAR_DENSITY_RADIUS_M = 0.45
CUSTOM_LIDAR_NORMAL_Z_MIN = 0.01
CUSTOM_LIDAR_NORMAL_Z_MAX = 1.80
CUSTOM_LIDAR_NEAR_Z_MIN = 0.1
CUSTOM_LIDAR_NEAR_Z_MAX = 2.00

                                                              

 

                                                 
CUSTOM_VERIFY_FAR_MIN_M = 2.00
CUSTOM_VERIFY_NEAR_MAX_M = 1.20
CUSTOM_VERIFY_FAR_MIN_HITS = 3
CUSTOM_VERIFY_NEAR_MIN_HITS = 2

CUSTOM_VERIFY_MATCH_RADIUS_M = 0.30

CUSTOM_VERIFY_STALE_S = 8.0

                                                
                                   
CUSTOM_CURB_STUCK_TIME_S = 1.2
CUSTOM_CURB_MIN_CMD_MPS = 0.15
CUSTOM_CURB_BOOST_LINEAR = 2.00
CUSTOM_CURB_BOOST_TIME_S = 0.90
CUSTOM_CURB_REVERSE_LINEAR = -0.30
CUSTOM_CURB_REVERSE_TIME_S = 1.00

                                       
                      
CUSTOM_CURB_ARM_ASSIST = True
CUSTOM_CURB_ARM_BACK_HOLD_S = 0.25
CUSTOM_CURB_ARM_FRONT_DELAY_S = 0.35
CUSTOM_CURB_ARM_FRONT_HOLD_S = 0.25

                     

                                                            
PICKUP_STANDOFF_M = 0.75
                                             
TARGET_KEEP_OUT_RADIUS_M = 0.22
                                              
TARGET_GOAL_CLEARANCE_FROM_ORANGE_M = 0.20
GRASP_FINAL_FWD_M = 0.75

                                            
                                                                                                 
ARM_BASE_FWD = 0.18

                                                     
GRASP_FWD_BACKOFF = -0.01

                                                                                      

                                                                                            

_DELIVERY_COORD = load_navigation_config().get("delivery_coord") or [-55.29138, 142.09763, 16.47326]
DELIVERY_XY = (_DELIVERY_COORD[0], _DELIVERY_COORD[1])

def _slew_cmd(prev_v, prev_w, target_v, target_w, dt=STAGE2_NAV_DT):
    """Limit cmd_vel change per control tick for smooth acceleration/deceleration."""

    def _axis(prev, target, accel, decel):
                                                                                 
        slowing = (
            abs(target) < abs(prev)
            or (prev != 0.0 and target != 0.0 and (prev > 0) != (target > 0))
        )
        rate = decel if slowing else accel
        step = max(0.0, rate * dt)

        delta = target - prev
        if delta > step:
            return prev + step
        if delta < -step:
            return prev - step
        return target

    v = _axis(
        float(prev_v), float(target_v),
        LINEAR_ACCEL_MPS2, LINEAR_DECEL_MPS2,
    )
    w = _axis(
        float(prev_w), float(target_w),
        ANGULAR_ACCEL_RPS2, ANGULAR_DECEL_RPS2,
    )
    return v, w

class DemoParticipant:
    """SDK-based demo participant -- connects the VLA/nav logic through handlers."""

    def __init__(self):
        self.client = MARCClient.from_env()
        self.vla = VLAGrounding(self.client)
        self._mission_lock = threading.Lock()
        self._cctv_warm = False

        self._collect_xy = None
        self._grasp_xy = None

        self._stage2_target_xy = None

        self._pickup_xy = None
        self._occ_logged = False

                                                           
                                      
        self._static_cells = {}

        self._last_nav_map_mode = "ORIGINAL"
        self._pickup_used_custom1 = False
        self._register_handlers()

    def _register_handlers(self):
        c = self.client
        c.on_mission(self._on_mission)
        c.on_stage2_mission(self._on_stage2_mission)
        c.on_stage2_reveal(self._on_stage2_reveal)
        c.on_stage2_run(self._drive)
        c.on_score(self._on_score)
        c.on_state_change(self._on_state_change)
        c.on_time_expired(self._on_time_expired)

    def _on_mission(self, mission):
        threading.Thread(target=self._run_mission, args=(mission,), daemon=True).start()

    def _wait_for_cctv(self, timeout_wall=15.0, settle_polls=3, poll_s=0.3):
        """Wait until CCTV frames are cached."""
        c = self.client
        t0 = time.monotonic()
        prev_n, settled = -1, 0
        while time.monotonic() - t0 < timeout_wall:
            ready = [cid for cid in c.list_cctv() if c.get_cctv_image(cid) is not None]
            n = len(ready)
            if n and n == prev_n:
                settled += 1
                if settled >= settle_polls:
                    return ready
            else:
                settled = 0
            prev_n = n
            c.sleep(poll_s)
        return [cid for cid in c.list_cctv() if c.get_cctv_image(cid) is not None]

    def _run_mission(self, mission):
        if not self._mission_lock.acquire(blocking=False):
            log.warning("[Stage1] round %s arrived while previous round was still running", mission.round)
            return
        try:
            log.info(
                "[Stage1] round %s/%s (limit %.0fs): \"%s\"",
                mission.round,
                mission.total_rounds,
                mission.time_limit,
                mission.voice_command,
            )
            if not self._cctv_warm:
                cams = self._wait_for_cctv()
                self._cctv_warm = bool(cams)
                log.info("[Stage1] CCTV warm-up: %d camera(s) ready %s", len(cams), sorted(cams))
                if not cams:
                    log.warning(
                        "[Stage1] round %s: no CCTV frames; grounding on language only",
                        mission.round,
                    )
            else:
                cams = [
                    cid for cid in self.client.list_cctv()
                    if self.client.get_cctv_image(cid) is not None
                ]

            try:
                result = self.vla.process(mission.voice_command, cams)
            except Exception:
                log.exception(
                    "[Stage1] round %s: vla.process() raised; submitting empty fallback",
                    mission.round,
                )
                result = GroundingResult(
                    camera_id="",
                    target_type="",
                    anchor_coord=[0.0, 0.0, 0.0],
                    target_coord=[0.0, 0.0, 0.0],
                )

            self.client.submit_grounding(result)
        finally:
            self._mission_lock.release()

    def _on_stage2_mission(self, mission):
        log.info("[Stage2] task: \"%s\" (limit %.0fs)",
                 mission.task_description, mission.time_limit)
                                                                                                                              
        op = getattr(mission, "owner_position", None)
        if op and len(op) >= 2:
            self._pickup_xy = (float(op[0]), float(op[1]))
            log.info("[Stage2] owner_position(runtime-provided)=%s", self._pickup_xy)
                                                                                       
        result = self.vla.process_stage2(mission.task_description)
        self.client.submit_stage2_grounding(result)
                                                                                         
        tc = result.target_coord
        if tc and len(tc) >= 2:
            self._stage2_target_xy = (float(tc[0]), float(tc[1]))
        log.info("[Stage2] submitted grounding target=%s, delivery point=%s",
                 self._stage2_target_xy, self._pickup_xy or DELIVERY_XY)

    def _on_stage2_reveal(self, reveal):
        """Stage 2 post-grounding reveal (msg 411) -- adopt the approximate location/type as the pick goal.

        Regardless of grounding accuracy, use the revealed approximate location
        (hint_center) as the goal of the pickup leg. There may be distractors within the
        radius, so use the type (target_type) to decide which object to pick up (the demo
        only demonstrates reaching the location).
        """
        log.info("[Stage2] reveal received -- score=%s, type=%s, center=%s, r=%.1f",
                 reveal.grounding_score, reveal.target_type,
                 reveal.hint_center, reveal.hint_radius)
        if reveal.hint_center and len(reveal.hint_center) >= 2:
            self._collect_xy = (float(reveal.hint_center[0]), float(reveal.hint_center[1]))

    def _drive(self):
        """Called once in a separate thread on entering STAGE2_RUN -- pick up the object then deliver it.

        Flow: stow the arm to lower the CoG -> approach the object standoff -> straight final
        approach into arm reach -> pick (read grip-hold) -> stow again if empty-handed ->
        deliver to the owner zone -> task_complete. All legs share one sim-time budget.
        """
        c = self.client
                                                                                                   
        deadline = c.now_s() + STAGE2_NAV_TIMEOUT_S

        collect_xy = self._wait_for_collect_goal(timeout=5.0)
                                                                                                                  
        pickup_xy = self._pickup_xy if self._pickup_xy is not None else DELIVERY_XY
        log.info("[Stage2] navigation start -- collect=%s deliver=%s (timeout %.0fs)",
                 collect_xy, pickup_xy, STAGE2_NAV_TIMEOUT_S)

        self._stow_arm()

        picked = False

        # 픽업 구간에서 CUSTOM1이 단 한 번이라도 선택되면 True로 유지.
        # 픽 완료 후 배송 구간의 CUSTOM2 전환 조건으로 사용한다.
        self._pickup_used_custom1 = False

                                                         
        log.info("[Stage2] leg1A -> hint=%s search standoff target=%.2fm",
                 tuple(round(v, 2) for v in collect_xy), PICKUP_STANDOFF_M)

        self._navigate_to(
            None,
            deadline,
            tol=0.15,                                         
            inflation_m=STAGE2_INFLATION_M,
            exclude_m=0.0,
            standoff_target=collect_xy,
            standoff_m=PICKUP_STANDOFF_M,
        )
        if self._last_nav_map_mode == "CUSTOM":
            self._pickup_used_custom1 = True
        if not c.is_running:
            return

                                                                  
                                                                            
        grasp_xy = collect_xy
        self._grasp_xy = grasp_xy

        log.info(
            "[Stage2] leg1B -> fixed object coordinate=%s precise standoff target=%.2fm",
            tuple(round(v, 2) for v in grasp_xy), PICKUP_STANDOFF_M,
        )

        self._navigate_to(
            None,
            deadline,
            tol=0.15,                                         
            inflation_m=STAGE2_INFLATION_M,
            exclude_m=0.0,
            standoff_target=grasp_xy,
            standoff_m=PICKUP_STANDOFF_M,
        )
        if self._last_nav_map_mode == "CUSTOM":
            self._pickup_used_custom1 = True
        if not c.is_running:
            return

        self._approach_straight(
            grasp_xy,
            deadline,
            target_fwd=GRASP_FINAL_FWD_M,
        )

        for _ in range(5):
            c.send_cmd_vel(0.0, 0.0)
            c.sleep(0.1)
        log.info("[Stage2] standoff stop -- ready to pick")

                                                            
        try:
            arm_grasp_xy = self._object_arm_xy(self._grasp_xy or collect_xy)
            log.info(
                "[Stage2][ARM] grasp detect start arm_xy=%s",
                tuple(round(v, 3) for v in arm_grasp_xy)
                if arm_grasp_xy else None,
            )

            held = False
            if arm_grasp_xy is not None:
                grasp = GraspGenerator(self.client)
                grasp_try = 1

                while (
                    c.is_running
                    and c.now_s() < deadline
                    and grasp_try <= grasp.p["max_try"]
                ):
                    scan_x, scan_y = arm_grasp_xy

                    target = grasp.detect(
                        [scan_x, scan_y, grasp.p["scan_z"]],
                        False,
                    )

                    if target is None:
                        log.info("[Stage2][ARM] no graspable object detected")
                        break

                    gx, gy, grasp_yaw = target
                    log.info(
                        "[Stage2][ARM] detected grasp=(%.3f, %.3f) yaw=%.3frad (%.1fdeg)",
                        gx, gy, grasp_yaw, math.degrees(grasp_yaw),
                    )

                    held = run_pick_sequence(
                        grasp_xy=(gx, gy),
                        yaw=grasp_yaw,
                        log=log,
                        client=c,
                    )

                    if held:
                        break

                    grasp_try += 1

            c.send_cmd_vel(0.0, 0.0)

            basket_ok = (
                c.is_basket_occupied()
                if hasattr(c, "is_basket_occupied")
                else None
            )
            held_ok = True if held is None else bool(held)
            picked = held_ok if basket_ok is None else bool(basket_ok)

            log.info(
                "[Stage2] pick %s (grip held=%s, basket=%s)",
                "ok" if picked else "failed",
                held,
                basket_ok,
            )

        except Exception as e:
            log.warning("[Stage2] pick failed: %s", e)

                                      
                                                            
        if picked:
            pose_after_pick = c.get_world_pose()
            if pose_after_pick is not None:
                px_pick, py_pick, _ = pose_after_pick
                before_count = len(self._static_cells)

                self._static_cells = {
                    key: value
                    for key, value in self._static_cells.items()
                    if math.hypot(
                        value[1][0] - px_pick,
                        value[1][1] - py_pick,
                    ) <= 3.0
                }

                log.info(
                    "[Stage2] post-pick LiDAR map reset: kept %d/%d cells within 3.0m",
                    len(self._static_cells),
                    before_count,
                )

                                                                                           
        if not picked:
            self._stow_arm()

                                                          

        # 픽업 구간에서 기존 CUSTOM1이 실제로 사용된 경우에만,
        # 픽앤플레이스가 끝난 뒤 배송 구간은 CUSTOM2로 전환한다.
        delivery_force_map_mode = (
            "CUSTOM2" if self._pickup_used_custom1 else None
        )
        log.info(
            "[Stage2] leg2 -> deliver to owner zone %s (force_map=%s, pickup_used_custom1=%s)",
            pickup_xy,
            delivery_force_map_mode or "AUTO",
            self._pickup_used_custom1,
        )
        reached = self._navigate_to(
            pickup_xy,
            deadline,
            force_map_mode=delivery_force_map_mode,
        )

        c.stop()
        log.info("[Stage2] driving finished (delivery_reached=%s) -> task_complete", reached)
        c.task_complete()

    def _stow_arm(self):
        """Fold the arm to the HOME pose to lower the center of gravity -- tip-over prevention.

        The arm driver re-applies the last joint_command every tick, so one publish holds the pose
        for the whole leg. (Not called after a successful pick -- leg 2 keeps the carry pose.)
        """
        try:
            from arm_pick import HOME, GRIPPER_OPEN, _make_js
            self.client.send_arm_command(_make_js(HOME, GRIPPER_OPEN))
            log.info("[Stage2] arm stow -> HOME (lower CoG, tip-over prevention)")
        except Exception as e:
            log.warning("[Stage2] arm stow failed: %s", e)

    def _curb_arm_pose(self, which):
        """CUSTOM 턱 탈출용 짧은 팔 피칭 자세.

        arm_pick.HOME을 기준으로 작은 관절 오프셋만 사용한다.
        which='back'  : 팔 질량을 뒤쪽으로 이동시켜 앞쪽 하중을 순간적으로 줄임.
        which='front' : 다시 앞쪽으로 이동시켜 차체 피칭 반작용을 줌.
        """
        try:
            from arm_pick import HOME, GRIPPER_OPEN, _make_js

            q = list(HOME)
            if which == "back":
                                               
                q[1] -= 0.22
                q[2] += 0.28
            elif which == "front":
                q[1] += 0.28
                q[2] -= 0.22
            else:
                q = list(HOME)

            self.client.send_arm_command(_make_js(q, GRIPPER_OPEN))
            log.info("[CUSTOM-CURB][ARM] pose=%s q=%s", which, [round(v, 3) for v in q])
            return True
        except Exception as e:
            log.warning("[CUSTOM-CURB][ARM] pose %s failed: %s", which, e)
            return False

    def _curb_arm_home(self):
        """턱 보조 후에는 오류 여부와 관계없이 HOME 복귀를 시도."""
        try:
            from arm_pick import HOME, GRIPPER_OPEN, _make_js
            self.client.send_arm_command(_make_js(HOME, GRIPPER_OPEN))
            log.info("[CUSTOM-CURB][ARM] restored HOME")
        except Exception as e:
            log.warning("[CUSTOM-CURB][ARM] HOME restore failed: %s", e)

    def _standoff_goal(self, dest, standoff):
        """A point ``standoff`` metres in front of ``dest`` along the robot->dest line.

        Approaching a standoff (instead of the object itself) keeps the chassis from driving
        onto/over the object.
        """
        pose = self.client.get_world_pose()
        if pose is None:
            return dest
        x, y, _ = pose
        dx, dy = dest[0] - x, dest[1] - y
        d = math.hypot(dx, dy)
        if d <= standoff:
            return (x, y)
        f = (d - standoff) / d
        return (x + dx * f, y + dy * f)

    def _object_arm_xy(self, obj_xy):
        """Object world XY -> arm_base frame (fwd, lat) using the current robot pose.

        Rotate the world offset into the car_base frame, then shift by the
        car_base->arm_base forward offset. Lets the closed-loop pick aim at the real
        object regardless of small nav misalignment.
        """
        pose = self.client.get_world_pose()
        if pose is None:
            return None
        x, y, yaw = pose
        dx, dy = obj_xy[0] - x, obj_xy[1] - y
        cc, ss = math.cos(yaw), math.sin(yaw)
        fwd = dx * cc + dy * ss                                   
        lat = -dx * ss + dy * cc                               
        return (fwd - ARM_BASE_FWD - GRASP_FWD_BACKOFF, lat)

    def _approach_straight(self, obj_xy, deadline, target_fwd=0.40, max_t=20.0):
        """Drive straight at the object until the arm_base forward distance reaches target_fwd.

        The planner cannot enter a tight spot fully (it halts ~1 m out), so fill the remaining
        empty space by driving straight into the object, with a small lateral steering term to
        center it (lat -> 0). SIM-time paced.
        """
        c = self.client
        t_end = min(deadline, c.now_s() + max_t)
        while c.is_running and c.now_s() < t_end:
            gxy = self._object_arm_xy(obj_xy)
            if gxy is None:
                break
            fwd, lat = gxy
            if fwd <= target_fwd + 0.02:
                break
            ang = max(-0.4, min(0.4, 1.5 * lat))                                       
            lin = 0.16 if abs(lat) < 0.15 else 0.08                                               
            c.send_cmd_vel(linear_x=lin, angular_z=ang)
            c.sleep(STAGE2_NAV_DT)
        for _ in range(5):
            c.send_cmd_vel(0.0, 0.0)
            c.sleep(0.05)
        gxy = self._object_arm_xy(obj_xy)
        log.info("[Stage2] straight approach done -- object arm coords %s",
                 tuple(round(v, 3) for v in gxy) if gxy else None)

    def _wait_for_collect_goal(self, timeout: float = 5.0):
        """Poll briefly until the reveal (msg 411) approximate position arrives, then return the pickup goal (x, y).

        When the reveal arrives, _on_stage2_reveal fills self._collect_xy. If not received
        within the timeout, fall back to the Stage 2 grounding answer's target (_stage2_target_xy).
        """
        c = self.client
        deadline = c.now_s() + timeout
        while c.now_s() < deadline:
            if self._collect_xy is not None:
                return self._collect_xy
            c.sleep(0.1)
        log.warning("[Stage2] reveal not received (%.0fs) -- falling back to grounding target %s",
                    timeout, self._stage2_target_xy)
        return self._stage2_target_xy

    def _align_to_target(self, target_xy, deadline, tol_deg=8.0, max_t=6.0):
        """Rotate in place so the robot finishes with target_xy straight ahead."""
        if target_xy is None:
            return True

        c = self.client
        end_t = min(deadline, c.now_s() + max_t)
        tol_rad = math.radians(tol_deg)
        align_w = 0.0

        while c.is_running and c.now_s() < end_t:
            pose = c.get_world_pose()
            if pose is None:
                c.sleep(STAGE2_NAV_DT)
                continue

            x, y, yaw = pose
            dx = target_xy[0] - x
            dy = target_xy[1] - y
            if math.hypot(dx, dy) < 0.05:
                                                                                     
                return True

            desired = math.atan2(dy, dx)
            err = (desired - yaw + math.pi) % (2.0 * math.pi) - math.pi

            if abs(err) <= tol_rad:
                for _ in range(3):
                    c.send_cmd_vel(0.0, 0.0)
                    c.sleep(0.05)
                log.info("[STAGE2] final heading aligned: err=%.1fdeg", math.degrees(err))
                return True

            w = max(-0.85, min(0.85, 1.35 * err))
            if 0.0 < abs(w) < 0.22:
                w = math.copysign(0.22, w)
            _, align_w = _slew_cmd(
                0.0, align_w,
                0.0, w,
            )
            c.send_cmd_vel(0.0, align_w)
            c.sleep(STAGE2_NAV_DT)

        c.send_cmd_vel(0.0, 0.0)
        log.warning("[STAGE2] final heading align timeout")
        return False

    def _navigate_to(self, goal, deadline, tol=GOAL_TOLERANCE,
                     inflation_m=STAGE2_INFLATION_M, exclude_m=0.0, cruise=False,
                     face_target=None, standoff_target=None, standoff_m=None,
                     force_map_mode=None):
        """Drive to a single goal (x, y) by following occupancy A*. True if reached, False on timeout.

        Args:
            goal: 고정 목표 좌표. standoff_target을 쓸 거면 None으로 둬도 됨(자동 계산).
            tol: reached-test XY distance (m). Tighter for a tight pick approach.
            inflation_m: erode the free area by this much (lower = enter tighter spots).
            exclude_m: drop scenario obstacles within this radius of the goal (keep it reachable).
            cruise: True -> pass-through waypoint, no near-goal deceleration.
            face_target: optional world XY to face after reaching the navigation goal.
            standoff_target: 지정하면 goal을 고정값으로 안 쓰고, 매 재계획마다
                planner.find_clear_standoff(standoff_target, standoff_m, ...)로
                "그 시점까지 확정된 장애물(confirmed_cells)까지 반영해서" 다시
                계산함. occupancy map만 보고 미리 한 번 계산해서 넘기면, 그 뒤
                라이다로 새로 확정된 장애물이 하필 그 지점 근처에 쌓였을 때
                goal 자체가 안전마진(침식)에 깎여도 못 알아채는 문제가 있었음.
            standoff_m: standoff_target과 함께 사용 -- 목표로부터의 거리(m).
        The deadline is a shared SIM-time budget across all Stage 2 legs.
        """
        c = self.client
        planner = OccupancyGridPlanner()
        map_mode = "ORIGINAL"
        follower = None
        path = None                                             
        self._lidar_dbg_t = 0.0                          
        self._grid_dbg_t = 0.0                             
        self._last_replan_t = -1e9               
        fallback_ctrl = GoalSeekController()

        cmd_v = 0.0
        cmd_w = 0.0

        goal_approach_yaw = None

        if standoff_target is not None and face_target is None:
            face_target = standoff_target

        if standoff_target is not None:
                                                                 
            occ0 = c.get_occupancy_map()
            if occ0 is not None:
                planner.update_from_msg(occ0)
                if force_map_mode == "CUSTOM":
                    planner._map_mode = "CUSTOM"
                    planner._grid = planner._custom_grid
                    map_mode = "CUSTOM"
                elif force_map_mode == "CUSTOM2":
                    planner._map_mode = "CUSTOM2"
                    planner._grid = planner._custom_grid2
                    map_mode = "CUSTOM2"
                elif force_map_mode == "ORIGINAL":
                    planner._map_mode = "ORIGINAL"
                    planner._grid = planner._original_grid
                    map_mode = "ORIGINAL"
                else:
                    map_mode = planner.select_map_for_goal(standoff_target)

                if map_mode == "CUSTOM" and force_map_mode != "CUSTOM2":
                    self._pickup_used_custom1 = True

                log.info(
                    "[MAP-MODE] target=(%.2f, %.2f) originalOcc=%s -> %s%s",
                    standoff_target[0], standoff_target[1],
                    str(planner.original_value_at(standoff_target)),
                    map_mode,
                    " [FORCED]" if force_map_mode else "",
                )
            pose0 = c.get_world_pose()
            prefer0 = (pose0[0], pose0[1]) if pose0 is not None else None
            result0 = (
                planner.find_clear_standoff(
                    standoff_target,
                    standoff_m,
                    inflation_m=inflation_m,
                    prefer_xy=prefer0,
                    hard_noinflate_cells=(
                        {
                            (_gr, _gc)
                            for _gr in range(
                                planner._w2c(*standoff_target)[0]
                                - int(math.ceil(TARGET_KEEP_OUT_RADIUS_M / planner._res)),
                                planner._w2c(*standoff_target)[0]
                                + int(math.ceil(TARGET_KEEP_OUT_RADIUS_M / planner._res)) + 1
                            )
                            for _gc in range(
                                planner._w2c(*standoff_target)[1]
                                - int(math.ceil(TARGET_KEEP_OUT_RADIUS_M / planner._res)),
                                planner._w2c(*standoff_target)[1]
                                + int(math.ceil(TARGET_KEEP_OUT_RADIUS_M / planner._res)) + 1
                            )
                            if math.hypot(
                                planner._c2w(_gr, _gc)[0] - standoff_target[0],
                                planner._c2w(_gr, _gc)[1] - standoff_target[1],
                            ) <= TARGET_KEEP_OUT_RADIUS_M
                        }
                    ),
                    goal_keepout_clearance_m=TARGET_GOAL_CLEARANCE_FROM_ORANGE_M,
                )
                if planner.ready else None
            )
            if result0 is not None:
                goal, goal_approach_yaw = result0
            else:
                goal = self._standoff_goal(standoff_target, standoff_m)
                goal_approach_yaw = math.atan2(
                    standoff_target[1] - goal[1],
                    standoff_target[0] - goal[0],
                )
            log.info(
                "[Stage2] standoff init: target=%s standoff=%.2fm -> goal=%s",
                tuple(round(v, 2) for v in standoff_target), standoff_m,
                tuple(round(v, 2) for v in goal),
            )

        leg_points = []

                                                                        

                                                  
        static_cells = self._static_cells

                                                             
                                                 
        custom_verify_cells = {}

        PATH_BLOCK_EXTRA_M = 0.25                                                                  

        last_xy = None
        last_progress_t = c.now_s()
        last_log_t = 0.0
        reached_goal = False
        backup_until_t = None                                    
        last_backup_end_t = -1e9                         
        estop_hold_until_t = None                         
        spin_until_t = None                                         
        last_spin_end_t = -1e9                           
        custom_curb_boost_until_t = None
        custom_curb_reverse_until_t = None
        custom_curb_arm_front_t = None
        custom_curb_arm_home_t = None
        custom_curb_arm_active = False

        def _accumulate_static_cells(cell_map, points, planner, now):
            """라이다 world 점들을 전역 occupancy 셀 좌표로 변환해 누적.

            원 병합(반경 합치기) 대신 셀 단위로 기록하되, 잡음 필터로 히트카운트를
            둠 -- cell_map[key] = [hit_count, 셀 중심 world 좌표, 마지막 감지 시각].
            같은 셀이 STATIC_CELL_STALE_S 이내에 "연속으로" 여러 프레임 반복
            감지돼야(STATIC_CELL_MIN_HITS) "확정 장애물"로 침. 정상적인 정적
            장애물은 로봇이 다가가는 동안 계속 잡히니 금방(0.1~0.3초) 확정되고,
            단발성 라이다 잡음(빈 공간에 튀는 점)은 hit_count가 안 쌓여서 걸러짐.
            static_cells가 Stage 2 전체에서 유지되는 영구 메모리라서, 사이가
            STALE_S 이상 벌어지면(연속이 아니면) 카운트를 리셋 -- 그래야 몇 분
            간격으로 어쩌다 튀는 잡음이 누적만으로 결국 확정되는 걸 막음.
            """
            if not planner.ready:
                return cell_map

            for wx, wy in points:
                gr, gc = planner._w2c(wx, wy)
                key = (gr, gc)

                if key in cell_map:
                    hits, _old_xy, last_t = cell_map[key]
                    if now - last_t > STATIC_CELL_STALE_S:
                        hits = 0                          
                    cell_map[key] = [hits + 1, planner._c2w(gr, gc), now]
                else:
                    cell_map[key] = [1, planner._c2w(gr, gc), now]

            return cell_map

        def _confirmed_cells(cell_map, min_hits):
            """min_hits 이상 반복 감지된 셀만 {cell: world_xy} 형태로 반환."""
            return {
                key: hits_xy[1]
                for key, hits_xy in cell_map.items()
                if hits_xy[0] >= min_hits
            }

        def _original_free_only(points, planner):
            """ORIGINAL 모드: 원본 OccupancyGrid에서 free(0)인 위치의 LiDAR 점만 유지.

            이미 100/-1 등으로 막힌 셀은 어차피 A*가 못 가므로 LiDAR 장애물로
            중복 저장하지 않는다.
            """
            if not planner.ready:
                return points

            out = []
            for wx, wy in points:
                try:
                    v = planner.original_value_at((wx, wy))
                except Exception:
                    v = None

                if v == 0:
                    out.append((wx, wy))
            return out

        def _custom_verify_update(verify_map, points, rx, ry, planner, now):
            """CUSTOM MAP: far 후보 -> near 재확인의 2단계 장애물 검증."""
            if not planner.ready:
                return verify_map

            match_cells = max(
                1,
                int(math.ceil(CUSTOM_VERIFY_MATCH_RADIUS_M / planner._res)),
            )
            stale_keys = [
                key
                for key, rec in verify_map.items()
                if now - rec[3] > CUSTOM_VERIFY_STALE_S
            ]
            for key in stale_keys:
                del verify_map[key]

            def _find_far_candidate(gr, gc):
                best_key = None
                best_d2 = None
                for rr in range(gr - match_cells, gr + match_cells + 1):
                    for cc in range(gc - match_cells, gc + match_cells + 1):
                        key = (rr, cc)
                        rec = verify_map.get(key)
                        if rec is None or rec[0] < CUSTOM_VERIFY_FAR_MIN_HITS:
                            continue
                        d2 = (rr - gr) ** 2 + (cc - gc) ** 2
                        if d2 > match_cells * match_cells:
                            continue
                        if best_d2 is None or d2 < best_d2:
                            best_key = key
                            best_d2 = d2
                return best_key

            for wx, wy in points:
                dist_robot = math.hypot(wx - rx, wy - ry)
                gr, gc = planner._w2c(wx, wy)
                key = (gr, gc)
                center_xy = planner._c2w(gr, gc)

                if dist_robot >= CUSTOM_VERIFY_FAR_MIN_M:
                    rec = verify_map.get(key)
                    if rec is None:
                        verify_map[key] = [1, 0, center_xy, now]
                    else:
                        rec[0] += 1
                        rec[2] = center_xy
                        rec[3] = now
                    continue

                if dist_robot > CUSTOM_VERIFY_NEAR_MAX_M:
                    continue

                far_key = _find_far_candidate(gr, gc)
                if far_key is None:
                    continue

                rec = verify_map[far_key]
                rec[1] += 1
                rec[3] = now

            return verify_map

        def _custom_verified_cells(verify_map):
            """far + near 조건을 모두 통과한 CUSTOM 장애물만 반환."""
            return {
                key: rec[2]
                for key, rec in verify_map.items()
                if rec[0] >= CUSTOM_VERIFY_FAR_MIN_HITS
                and rec[1] >= CUSTOM_VERIFY_NEAR_MIN_HITS
            }

        def _decay_unseen_cells(cell_map, seen_keys, rx, ry, ryaw, planner,
                                 goal_xy, exclude_m, radius_m=5.0, fov_half_deg=60.0,
                                 delete_at_hits=1):
            """라이다 시야(사거리 radius_m, ±fov_half_deg) 안인데 이번 프레임에
            재감지 안 된 셀은 노이즈였을 수 있음 -- 히트카운트 깎고, 0 되면 삭제.
            단, 목표 exclude_m 반경 안 셀은 애초에 leg_points에서 걸러지니(목표
            자체를 장애물로 안 잡으려고) 이 로직에서도 건드리지 않음.
            """
            if not planner.ready:
                return cell_map
            fov_half = math.radians(fov_half_deg)
            ex2 = exclude_m ** 2
            dead = []
            for key, (hits, (wx, wy), last_t) in cell_map.items():
                if key in seen_keys:
                    continue
                if (wx - goal_xy[0]) ** 2 + (wy - goal_xy[1]) ** 2 <= ex2:
                    continue
                d = math.hypot(wx - rx, wy - ry)
                if d > radius_m:
                    continue
                bearing = math.atan2(wy - ry, wx - rx) - ryaw
                bearing = (bearing + math.pi) % (2.0 * math.pi) - math.pi
                if abs(bearing) > fov_half:
                    continue
                                                   
                if hits <= delete_at_hits:
                    dead.append(key)
                else:
                    cell_map[key] = [hits - 1, (wx, wy), last_t]
            for key in dead:
                del cell_map[key]
            return cell_map

        def _prune_far_cells(cell_map, rx, ry, max_dist_m=10.0):
            """로봇에서 max_dist_m 넘게 떨어진 셀은 버림 -- static_cells가 leg
            넘어서도 무한히 쌓이면서(9만개+) 매 프레임 순회 비용이 커져 루프
            자체가 느려지는 문제 방지. 멀리 있던 장애물은 다시 가까워지면
            라이다가 재감지해서 다시 채워짐.
            """
            d2 = max_dist_m ** 2
            dead = [
                key for key, (_hits, (wx, wy), _t) in cell_map.items()
                if (wx - rx) ** 2 + (wy - ry) ** 2 > d2
            ]
            for key in dead:
                del cell_map[key]
            return cell_map

        def _point_to_segment_dist(px, py, ax, ay, bx, by):
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

        def _path_is_blocked(path_points, cell_map, inflate):
            """True if any accumulated blocked cell center intersects the path polyline."""
            if not path_points or not cell_map:
                return False

            full_path = [(x, y)] + list(path_points)

            for (ox, oy) in cell_map.values():
                for i in range(len(full_path) - 1):
                    ax, ay = full_path[i]
                    bx, by = full_path[i + 1]

                    if (
                        math.hypot(ax - x, ay - y) < 0.15
                        and math.hypot(bx - x, by - y) < 0.15
                    ):
                        continue

                    if _point_to_segment_dist(
                        ox, oy,
                        ax, ay,
                        bx, by,
                    ) <= inflate:
                        return True

            return False

        while c.is_running and c.now_s() < deadline:
            pose = c.get_world_pose()
            if pose is None:
                c.sleep(STAGE2_NAV_DT)
                continue
            x, y, yaw = pose
            dist = math.hypot(goal[0] - x, goal[1] - y)
            if dist < tol:
                reached_goal = True
                log.info("[STAGE2] reached goal at (%.2f, %.2f)", x, y)
                cmd_v, cmd_w = 0.0, 0.0
                for _ in range(3):
                    c.send_cmd_vel(0.0, 0.0)
                    c.sleep(0.05)
                if face_target is not None:
                    self._align_to_target(face_target, deadline)
                break

            occ = c.get_occupancy_map()
            if occ is not None:
                planner.update_from_msg(occ)

                map_select_target = (
                    standoff_target if standoff_target is not None else goal
                )

                if force_map_mode == "CUSTOM":
                    new_map_mode = "CUSTOM"
                    planner._map_mode = "CUSTOM"
                    planner._grid = planner._custom_grid
                elif force_map_mode == "CUSTOM2":
                    new_map_mode = "CUSTOM2"
                    planner._map_mode = "CUSTOM2"
                    planner._grid = planner._custom_grid2
                elif force_map_mode == "ORIGINAL":
                    new_map_mode = "ORIGINAL"
                    planner._map_mode = "ORIGINAL"
                    planner._grid = planner._original_grid
                else:
                    new_map_mode = planner.select_map_for_goal(map_select_target)

                if new_map_mode == "CUSTOM" and force_map_mode != "CUSTOM2":
                    self._pickup_used_custom1 = True

                if new_map_mode != map_mode:
                    log.info(
                        "[MAP-MODE] %s -> %s target=(%.2f, %.2f) originalOcc=%s%s",
                        map_mode, new_map_mode,
                        map_select_target[0], map_select_target[1],
                        str(planner.original_value_at(map_select_target)),
                        " [FORCED]" if force_map_mode else "",
                    )
                    follower = None
                    path = None
                    self._last_replan_t = -1e9

                map_mode = new_map_mode

                if not self._occ_logged:
                    self._occ_logged = True
                    log.info("[Stage2] occupancy received: %dx%d res=%.3f origin=(%.1f, %.1f)",
                             occ.info.width, occ.info.height, occ.info.resolution,
                             occ.info.origin.position.x, occ.info.origin.position.y)

                                                     

                                                   

            target_dist = (
                math.hypot(
                    standoff_target[0] - x,
                    standoff_target[1] - y,
                )
                if standoff_target is not None
                else float("inf")
            )

            near_target_mode = (
                standoff_target is not None
                and target_dist <= NEAR_TARGET_LIDAR_DIST_M
            )

            if map_mode in ("CUSTOM", "CUSTOM2"):
                confirm_hits = (
                    CUSTOM_NEAR_CONFIRM_HITS
                    if near_target_mode else CUSTOM_STATIC_CELL_MIN_HITS
                )
                lidar_stride = (
                    CUSTOM_LIDAR_NEAR_STRIDE
                    if near_target_mode else CUSTOM_LIDAR_NORMAL_STRIDE
                )
                lidar_density_min = (
                    CUSTOM_LIDAR_NEAR_DENSITY_MIN
                    if near_target_mode else CUSTOM_LIDAR_NORMAL_DENSITY_MIN
                )
                lidar_density_radius = (
                    CUSTOM_LIDAR_NEAR_DENSITY_RADIUS_M
                    if near_target_mode else CUSTOM_LIDAR_NORMAL_DENSITY_RADIUS_M
                )
                lidar_z_min = (
                    CUSTOM_LIDAR_NEAR_Z_MIN
                    if near_target_mode else CUSTOM_LIDAR_NORMAL_Z_MIN
                )
                lidar_z_max = (
                    CUSTOM_LIDAR_NEAR_Z_MAX
                    if near_target_mode else CUSTOM_LIDAR_NORMAL_Z_MAX
                )
            else:
                confirm_hits = (
                    ORIGINAL_NEAR_CONFIRM_HITS
                    if near_target_mode else ORIGINAL_STATIC_CELL_MIN_HITS
                )
                lidar_stride = (
                    ORIGINAL_LIDAR_NEAR_STRIDE
                    if near_target_mode else ORIGINAL_LIDAR_NORMAL_STRIDE
                )
                lidar_density_min = (
                    ORIGINAL_LIDAR_NEAR_DENSITY_MIN
                    if near_target_mode else ORIGINAL_LIDAR_NORMAL_DENSITY_MIN
                )
                lidar_density_radius = (
                    ORIGINAL_LIDAR_NEAR_DENSITY_RADIUS_M
                    if near_target_mode else ORIGINAL_LIDAR_NORMAL_DENSITY_RADIUS_M
                )
                lidar_z_min = (
                    ORIGINAL_LIDAR_NEAR_Z_MIN
                    if near_target_mode else ORIGINAL_LIDAR_NORMAL_Z_MIN
                )
                lidar_z_max = (
                    ORIGINAL_LIDAR_NEAR_Z_MAX
                    if near_target_mode else ORIGINAL_LIDAR_NORMAL_Z_MAX
                )

            cloud = c.get_lidar()
            if cloud is not None:
                                                  
                raw_points, _dbg = cloud_to_blocked_points(
                    cloud,
                    (x, y, yaw),
                    path=path,
                    corridor_w=3.5,
                    stride=lidar_stride,
                    z_min=lidar_z_min,
                    z_max=lidar_z_max,
                    near_safety_m=5.00,
                    r_min=0.05,
                    ahead_only=False,
                    density_min_neighbors=lidar_density_min,
                    density_radius_m=lidar_density_radius,
                    debug=True,
                )
                exclude_center = standoff_target if standoff_target is not None else goal
                leg_points = obstacles_excluding_goal(raw_points, exclude_center, exclude_m=exclude_m)
                if map_mode == "ORIGINAL":
                    leg_points = _original_free_only(leg_points, planner)
                seen_keys = (
                    {planner._w2c(wx, wy) for wx, wy in leg_points}
                    if planner.ready else set()
                )
                static_cells = _accumulate_static_cells(
                    static_cells,
                    leg_points,
                    planner,
                    c.now_s(),
                )

                                                               

                                           

                if near_target_mode:
                    self._near_decay_counter = getattr(self, "_near_decay_counter", 0) + 1

                    if self._near_decay_counter >= 3:
                        static_cells = _decay_unseen_cells(
                            static_cells, seen_keys, x, y, yaw, planner,
                            goal, exclude_m,
                            delete_at_hits=0,
                        )
                        self._near_decay_counter = 0
                else:
                    self._near_decay_counter = 0
                    static_cells = _decay_unseen_cells(
                        static_cells, seen_keys, x, y, yaw, planner,
                        goal, exclude_m,
                    )
                static_cells = _prune_far_cells(static_cells, x, y, max_dist_m=10.0)
                                                      
                if c.now_s() - self._lidar_dbg_t >= 2.0:
                    self._lidar_dbg_t = c.now_s()
                    log.info(
                        "[LIDAR-DBG] map=%s mode=%s targetDist=%.2f stride=%s densityMin=%s densityR=%s confirmHits=%s zFilter=[%.2f,%.2f] "
                        "raw=%s x=[%s,%s] y=[%s,%s] z=[%s,%s] afterZ=%s afterROI=%s "
                        "blockedPts=%s exclude=%s staticCells=%d nearestRaw=%s nearestROI=%s",
                        map_mode,
                        "NEAR" if near_target_mode else "NORMAL",
                        target_dist,
                        _dbg.get("stride"),
                        _dbg.get("density_min_neighbors"),
                        _dbg.get("density_radius_m"),
                        confirm_hits,
                        lidar_z_min,
                        lidar_z_max,
                        _dbg.get("n_raw"),
                        _dbg.get("x_min_seen"), _dbg.get("x_max_seen"),
                        _dbg.get("y_min_seen"), _dbg.get("y_max_seen"),
                        _dbg.get("z_min_seen"), _dbg.get("z_max_seen"),
                        _dbg.get("n_after_zfilter"), _dbg.get("n_after_roi"),
                        _dbg.get("n_blocked_points"), len(leg_points), len(static_cells),
                        _dbg.get("nearest_raw_m"), _dbg.get("nearest_roi_m"),
                    )
                    if map_mode == "ORIGINAL":
                        log.info(
                            "[ORIGINAL-LIDAR] free-only points=%d frontForce<=%.2fm halfW=%.2fm",
                            len(leg_points),
                            ORIGINAL_FRONT_FORCE_DIST_M,
                            ORIGINAL_FRONT_FORCE_HALF_WIDTH_M,
                        )

                    if map_mode in ("CUSTOM", "CUSTOM2"):
                        log.info(
                            "[CUSTOM-LIDAR] single-hit immediate confirm: points=%d cells=%d",
                            len(leg_points),
                            len(_confirmed_cells(static_cells, 1)),
                        )
            else:
                if c.now_s() - self._lidar_dbg_t >= 2.0:
                    self._lidar_dbg_t = c.now_s()
                    log.warning("[LIDAR-DBG] get_lidar() returned None")

                                                            

                                                                
            if map_mode in ("CUSTOM", "CUSTOM2"):
                                                                            
                confirmed_cells = _confirmed_cells(static_cells, 1)
            else:
                                      
                confirmed_cells = _confirmed_cells(static_cells, confirm_hits)

                                                                
                                                               
            target_keepout_cells = {}

            if standoff_target is not None and planner.ready:
                _tgr, _tgc = planner._w2c(*standoff_target)
                _rad_c = int(math.ceil(TARGET_KEEP_OUT_RADIUS_M / planner._res))

                for _dr in range(-_rad_c, _rad_c + 1):
                    for _dc in range(-_rad_c, _rad_c + 1):
                        _gr = _tgr + _dr
                        _gc = _tgc + _dc
                        _wx, _wy = planner._c2w(_gr, _gc)

                        if math.hypot(
                            _wx - standoff_target[0],
                            _wy - standoff_target[1],
                        ) <= TARGET_KEEP_OUT_RADIUS_M:
                            target_keepout_cells[(_gr, _gc)] = (_wx, _wy)

                                                           
                                                         
            if STAGE2_GRID_DEBUG_IMG and c.now_s() - self._grid_dbg_t >= 1.0:
                self._grid_dbg_t = c.now_s()
                planner.dump_debug_view(
                    (x, y), goal,
                    inflation_m=inflation_m,
                    extra_cells=confirmed_cells.keys(),
                    path_world=path,
                    debug_dump_path=STAGE2_GRID_DEBUG_IMG,
                    target_xy=standoff_target,
                    robot_pose=(x, y, yaw),
                    target_keepout_radius_m=(
                        TARGET_KEEP_OUT_RADIUS_M
                        if standoff_target is not None
                        else 0.0
                    ),
                    hard_noinflate_cells=target_keepout_cells.keys(),
                )

                                                               

            emergency_stop = False

                                                                
            if map_mode == "ORIGINAL":
                for px, py in leg_points:
                    dx_f = px - x
                    dy_f = py - y
                    fwd_f = dx_f * math.cos(yaw) + dy_f * math.sin(yaw)
                    lat_f = -dx_f * math.sin(yaw) + dy_f * math.cos(yaw)
                    if (
                        0.0 < fwd_f <= ORIGINAL_FRONT_FORCE_DIST_M
                        and abs(lat_f) <= ORIGINAL_FRONT_FORCE_HALF_WIDTH_M
                    ):
                        emergency_stop = True
                        break

            if not emergency_stop:
                for (px, py) in confirmed_cells.values():
                    dx_e = px - x
                    dy_e = py - y
                    fwd_e = dx_e * math.cos(yaw) + dy_e * math.sin(yaw)
                    lat_e = -dx_e * math.sin(yaw) + dy_e * math.cos(yaw)

                    if (
                        0.0 < fwd_e <= EMERGENCY_STOP_DIST_M
                        and abs(lat_e) <= EMERGENCY_STOP_HALF_WIDTH_M
                    ):
                        emergency_stop = True
                        break

            if emergency_stop:
                                                 
                self._last_replan_t = -1e9

                if backup_until_t is None:

                                                           
                                                       
                    cmd_v, cmd_w = 0.0, 0.0
                    c.send_cmd_vel(linear_x=0.0, angular_z=0.0)

                    if estop_hold_until_t is None:
                        estop_hold_until_t = c.now_s() + EMERGENCY_HARD_STOP_HOLD_S
                        log.warning(
                            "[Stage2] EMERGENCY: 정면 %.2fm 이내 장애물 -> 즉시 정지 (%.1fs 유지)",
                            EMERGENCY_STOP_DIST_M, EMERGENCY_HARD_STOP_HOLD_S,
                        )

                    if c.now_s() < estop_hold_until_t:
                        last_progress_t = c.now_s()
                        c.sleep(STAGE2_NAV_DT)
                        continue

                    estop_hold_until_t = None

                    if c.now_s() - last_backup_end_t > STAGE2_BACKUP_COOLDOWN_S:
                        log.warning(
                            "[Stage2] EMERGENCY: 정지 완료 -> %.1fs 후진",
                            STAGE2_BACKUP_TIME_S,
                        )
                        backup_until_t = c.now_s() + STAGE2_BACKUP_TIME_S
                                                                
                        last_progress_t = c.now_s()
                    else:
                                                           
                        if c.now_s() - last_log_t >= 1.0:
                            log.warning(
                                "[Stage2] EMERGENCY STOP 유지 (후진 쿨다운 중): 정면 %.2fm 이내",
                                EMERGENCY_STOP_DIST_M,
                            )
                            last_log_t = c.now_s()
                        last_progress_t = c.now_s()
                        c.sleep(STAGE2_NAV_DT)
                        continue
            else:
                                       
                estop_hold_until_t = None

                                                 

                                                     
                                                     
            path_blocked = _path_is_blocked(
                path,
                confirmed_cells,
                inflation_m + PATH_BLOCK_EXTRA_M,
            )

            if path_blocked:
                log.info(
                    "[Stage2] path blocked by LiDAR-detected cell (%d confirmed / %d total) -> replanning",
                    len(confirmed_cells), len(static_cells),
                )

                                                   
            REPLAN_COOLDOWN_S = 0.3
            blocked_replan = (
                path_blocked
                and (
                    c.now_s()
                    - getattr(self, "_last_replan_t", -1e9)
                    >= REPLAN_COOLDOWN_S
                )
            )

                                                                
                                                
            need_replan = follower is None or blocked_replan
            if last_xy is not None:
                moved = math.hypot(x - last_xy[0], y - last_xy[1])
                if moved > STAGE2_STUCK_MOVE_M:
                    last_progress_t = c.now_s()
                elif (
                    map_mode in ("CUSTOM", "CUSTOM2")
                    and custom_curb_boost_until_t is None
                    and custom_curb_reverse_until_t is None
                    and backup_until_t is None
                    and cmd_v >= CUSTOM_CURB_MIN_CMD_MPS
                    and c.now_s() - last_progress_t > CUSTOM_CURB_STUCK_TIME_S
                ):
                    custom_curb_reverse_until_t = c.now_s() + CUSTOM_CURB_REVERSE_TIME_S
                    last_progress_t = c.now_s()
                    log.warning(
                        "[CUSTOM-CURB] forward cmd %.2f but pose stalled at (%.2f, %.2f) "
                        "-> REVERSE %.2fm/s for %.2fs, then BOOST",
                        cmd_v, x, y,
                        CUSTOM_CURB_REVERSE_LINEAR,
                        CUSTOM_CURB_REVERSE_TIME_S,
                    )
                elif (
                    backup_until_t is None
                    and custom_curb_boost_until_t is None
                    and c.now_s() - last_progress_t > STAGE2_STUCK_TIME_S
                    and c.now_s() - last_backup_end_t > STAGE2_BACKUP_COOLDOWN_S
                ):
                    log.info(
                        "[Stage2] stall detected -> backing up %.1fs before replanning",
                        STAGE2_BACKUP_TIME_S,
                    )
                    backup_until_t = c.now_s() + STAGE2_BACKUP_TIME_S
                    last_progress_t = c.now_s()
            last_xy = (x, y)

                                                    
            if custom_curb_reverse_until_t is not None:
                now = c.now_s()
                if now < custom_curb_reverse_until_t:
                    cmd_v = CUSTOM_CURB_REVERSE_LINEAR
                    cmd_w = 0.0
                    c.send_cmd_vel(linear_x=cmd_v, angular_z=cmd_w)
                    if now - last_log_t >= 0.5:
                        log.warning(
                            "[CUSTOM-CURB] reversing at (%.2f, %.2f), %.2fs left",
                            x, y, custom_curb_reverse_until_t - now,
                        )
                        last_log_t = now
                    c.sleep(STAGE2_NAV_DT)
                    continue
                else:
                    custom_curb_reverse_until_t = None
                    custom_curb_boost_until_t = now + CUSTOM_CURB_BOOST_TIME_S
                    last_progress_t = now

                    if CUSTOM_CURB_ARM_ASSIST:
                        try:
                            custom_curb_arm_active = self._curb_arm_pose("back")
                            if custom_curb_arm_active:
                                custom_curb_arm_front_t = now + CUSTOM_CURB_ARM_FRONT_DELAY_S
                                custom_curb_arm_home_t = (
                                    custom_curb_arm_front_t + CUSTOM_CURB_ARM_FRONT_HOLD_S
                                )
                        except Exception as e:
                            custom_curb_arm_active = False
                            log.warning("[CUSTOM-CURB][ARM] assist start failed: %s", e)

                    log.warning(
                        "[CUSTOM-CURB] reverse finished -> BOOST %.2fm/s for %.2fs",
                        CUSTOM_CURB_BOOST_LINEAR,
                        CUSTOM_CURB_BOOST_TIME_S,
                    )

            if custom_curb_boost_until_t is not None:
                now = c.now_s()
                if now < custom_curb_boost_until_t:
                    if custom_curb_arm_active:
                        if (
                            custom_curb_arm_front_t is not None
                            and now >= custom_curb_arm_front_t
                        ):
                            self._curb_arm_pose("front")
                            custom_curb_arm_front_t = None

                        if (
                            custom_curb_arm_home_t is not None
                            and now >= custom_curb_arm_home_t
                        ):
                            self._curb_arm_home()
                            custom_curb_arm_home_t = None
                            custom_curb_arm_active = False

                    cmd_v = CUSTOM_CURB_BOOST_LINEAR
                    cmd_w = 0.0
                    c.send_cmd_vel(linear_x=cmd_v, angular_z=cmd_w)
                    if now - last_log_t >= 0.5:
                        log.warning(
                            "[CUSTOM-CURB] boosting at (%.2f, %.2f), %.2fs left",
                            x, y, custom_curb_boost_until_t - now,
                        )
                        last_log_t = now
                    c.sleep(STAGE2_NAV_DT)
                    continue
                else:
                    custom_curb_boost_until_t = None

                    if custom_curb_arm_active or custom_curb_arm_home_t is not None:
                        self._curb_arm_home()
                    custom_curb_arm_front_t = None
                    custom_curb_arm_home_t = None
                    custom_curb_arm_active = False

                    last_progress_t = now

                                                              
                    static_cells.clear()
                    try:
                        custom_verify_cells.clear()
                    except Exception:
                        pass

                    need_replan = True
                    log.info(
                        "[CUSTOM-CURB] boost finished -> LiDAR mapping reset -> replanning"
                    )

                                                                                      
                                                     
            if backup_until_t is not None:
                now = c.now_s()
                if now < backup_until_t:
                    cmd_v, cmd_w = _slew_cmd(
                        cmd_v, cmd_w,
                        STAGE2_BACKUP_LINEAR, 0.0,
                    )
                    c.send_cmd_vel(linear_x=cmd_v, angular_z=cmd_w)
                    if now - last_log_t >= 1.0:
                        log.info(
                            "[STAGE2] backing up (stall/emergency recovery) %.1fs left",
                            backup_until_t - now,
                        )
                        last_log_t = now
                    c.sleep(STAGE2_NAV_DT)
                    continue
                else:
                    backup_until_t = None
                    last_backup_end_t = now
                    need_replan = True
                    log.info("[Stage2] backup finished -> replanning")

                                                              
                                             
            if spin_until_t is not None:
                now = c.now_s()
                if now < spin_until_t:
                    cmd_v, cmd_w = _slew_cmd(
                        cmd_v, cmd_w,
                        0.0, STAGE2_SPIN_ANGULAR,
                    )
                    c.send_cmd_vel(linear_x=cmd_v, angular_z=cmd_w)
                    if now - last_log_t >= 1.0:
                        log.info(
                            "[STAGE2] spinning in place (A* recovery) %.1fs left",
                            spin_until_t - now,
                        )
                        last_log_t = now
                    c.sleep(STAGE2_NAV_DT)
                    continue
                else:
                    spin_until_t = None
                    last_spin_end_t = now
                    log.info("[Stage2] recovery spin finished -> retrying replan")

            if need_replan and spin_until_t is None:
                if not planner.ready:
                    if not self._occ_logged:
                        log.warning("[Stage2] occupancy not received -> straight-line fallback")
                else:
                    if standoff_target is not None:

                        result = planner.find_clear_standoff(
                            standoff_target,
                            standoff_m,
                            extra_cells=confirmed_cells.keys(),
                            inflation_m=inflation_m,
                            prefer_xy=(x, y),
                            hard_noinflate_cells=target_keepout_cells.keys(),
                            goal_keepout_clearance_m=TARGET_GOAL_CLEARANCE_FROM_ORANGE_M,
                        )
                        if result is not None:
                            new_goal, new_approach_yaw = result
                            if math.hypot(new_goal[0] - goal[0], new_goal[1] - goal[1]) > 0.05:
                                log.info(
                                    "[Stage2] standoff 재탐색: %s -> %s (새로 확정된 장애물 반영)",
                                    tuple(round(v, 2) for v in goal),
                                    tuple(round(v, 2) for v in new_goal),
                                )
                            goal = new_goal
                            goal_approach_yaw = new_approach_yaw

                                                                                                                   
                                                                           
                    path = planner.plan(
                        (x, y),
                        goal,
                        inflation_m=inflation_m,
                        snap_m=4.0,
                        extra_cells=confirmed_cells.keys(),
                        debug_dump_path=STAGE2_GRID_DEBUG_IMG,
                        debug_target_xy=standoff_target,
                        final_goal_yaw=goal_approach_yaw,
                        allow_goal_in_inflation=(standoff_target is not None),
                        hard_noinflate_cells=target_keepout_cells.keys(),
                    )
                    if path:
                        follower = PathFollower(path, passthrough=cruise)
                        self._last_replan_t = c.now_s()              
                        log.info("[Stage2] occupancy A* path %d waypoints", len(path))
                    elif c.now_s() - last_spin_end_t > STAGE2_SPIN_COOLDOWN_S:

                                                            
                        log.info(
                            "[Stage2] A* failed -> spinning in place once (%.1fs) then retrying",
                            STAGE2_SPIN_DURATION_S,
                        )
                        spin_until_t = c.now_s() + STAGE2_SPIN_DURATION_S
                    else:
                        sr, sc = planner._w2c(x, y)
                        gr, gc = planner._w2c(*goal)
                        sv = int(planner._grid[sr, sc]) if planner._in_bounds(sr, sc) else 99
                        gv = int(planner._grid[gr, gc]) if planner._in_bounds(gr, gc) else 99
                        log.warning("[Stage2] A* failed again after spin -> straight-line fallback "
                                    "(start cell=(%d,%d) val=%d, goal cell=(%d,%d) val=%d) "
                                    "[0=free 100=occ -1=unknown]", sr, sc, sv, gr, gc, gv)

            wp_str = ""
            if follower is not None:

                linear, angular, done, info = follower.compute(pose)

                wp_str = " wp %d/%d=(%.1f, %.1f)" % (
                    info["idx"], info["total"], info["waypoint"][0], info["waypoint"][1])
                if info.get("local_avoidance"):
                    wp_str += " AVOID=" + str(info.get("avoid_reason"))
                if done:
                    reached_goal = True
                    log.info("[STAGE2] reached goal at (%.2f, %.2f)", x, y)

                    cmd_v, cmd_w = 0.0, 0.0
                    for _ in range(3):
                        c.send_cmd_vel(0.0, 0.0)
                        c.sleep(0.05)
                    if face_target is not None:
                        self._align_to_target(face_target, deadline)
                    break
            else:
                linear, angular, reached = fallback_ctrl.compute(pose, goal, cruise=cruise)
                if reached:
                    reached_goal = True
                    log.info("[STAGE2] reached goal at (%.2f, %.2f)", x, y)

                    cmd_v, cmd_w = 0.0, 0.0
                    for _ in range(3):
                        c.send_cmd_vel(0.0, 0.0)
                        c.sleep(0.05)
                    if face_target is not None:
                        self._align_to_target(face_target, deadline)
                    break

                                     
            cmd_v, cmd_w = _slew_cmd(
                cmd_v, cmd_w,
                linear, angular,
            )
            c.send_cmd_vel(linear_x=cmd_v, angular_z=cmd_w)

            now = c.now_s()
            if now - last_log_t >= 1.0:
                log.info("[STAGE2] pose=(%.2f, %.2f, yaw=%.0fdeg) dist=%.2fm%s v=%.2f w=%+.2f",
                         x, y, math.degrees(yaw), dist, wp_str, cmd_v, cmd_w)
                last_log_t = now

            c.sleep(STAGE2_NAV_DT)

        self._last_nav_map_mode = map_mode
        log.info(
            "[MAP-MODE] navigation finished with mode=%s%s",
            map_mode,
            " [FORCED]" if force_map_mode else "",
        )
        return reached_goal

    def _on_score(self, score):
        if score.is_final:
            log.info("[FINAL] %s", score.scores)
        else:
            log.info("[Score] round %s: total=%s", score.round, score.total)

    def _on_state_change(self, old, new):
        log.info("[STATE] %s -> %s", old, new)

    def _on_time_expired(self, which):
        log.info("[TIME] expired: %s", which)

    def run(self):
        if not self.client.connect():
            raise SystemExit("Registration failed -- check the runtime and the token.")
        self.client.run()

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    DemoParticipant().run()

if __name__ == "__main__":
    main()
