import sys, os
import logging
import numpy as np

from tf2_ros import Buffer, TransformListener
from rclpy.time import Time, Duration
from scipy.spatial.transform import Rotation as R

import arm_kin
from manipulation.image import DepthImage
from arm_pick import _read_arm_q, _kin, load_params, _make_js, GRIPPER_OPEN, HOME

log = logging.getLogger("grasp_planner")

def _cam_kin():
    _URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "urdf", "gen3_6dof_vision_2f140.urdf")
    _EE_FRAME = "camera_depth_frame"
    ARM_JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    return arm_kin.PlacoArmKinematics(_URDF, _EE_FRAME, list(ARM_JOINT_NAMES))

class GraspGenerator:
    def __init__(self, client):
        self.kin = _kin()
        self.cam_kin = _cam_kin()
        self.client = client
        self.di = DepthImage(client, log=log)
        self.p = load_params()
        self.buf = Buffer()
        self.listener = TransformListener(self.buf, self.client._node)

    def _wait_q(self):
            while True:
                q = _read_arm_q(self.client)
                if q is not None:
                    return q
                self.client.sleep(0.01)

    def get_tf(self):
        tf = self.buf.lookup_transform("arm_base_link", "Gripper_Camera_Center", Time())
        t, q = tf.transform.translation, tf.transform.rotation
        T = np.eye(4)
        T[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def get_extrinsic(self):
        q = self._wait_q()
        extrinsic = self.cam_kin.fk(q)
        log.debug(f"Kin camera extrinsic: {extrinsic}")
        extrinsic = self.get_tf()
        return extrinsic

    def get_world_yaw(self, T_base_cam, pixel_yaw):
        R_base_cam = T_base_cam[:3, :3]
        d_cam = np.array([np.cos(pixel_yaw), np.sin(pixel_yaw), 0.0])
        d_base = R_base_cam @ d_cam
        world_yaw = np.arctan2(d_base[1], d_base[0])
        log.debug(f"Pixel yaw: {np.rad2deg(pixel_yaw)}, World yaw: {np.rad2deg(world_yaw)}")
        return world_yaw

    def clip_yaw(self, yaw):
        return (yaw + np.pi / 2)% np.pi

    def look_down(self, target_pos=None):
        arm_q = self._wait_q()
        if target_pos is None:
            target_pos = self.kin.fk(arm_q)[:3, 3].copy()
        target_pos = np.array(target_pos)
        tool_quat = arm_kin.R_to_quat(self.kin.fk(arm_kin.DEFAULT_ARM_Q)[:3,:3])
        target = arm_kin.ik_solve(self.kin, target_pos, tool_quat, arm_q)
        target_js = _make_js(target, GRIPPER_OPEN)
        self.client.send_arm_command(target_js)
        stable = 0
        q_prev = None
        for _ in range(int(10 * 100.0)):                  # 최대 100초
            self.client.sleep(0.03)
            q = _read_arm_q(self.client)
            if q is None:
                stable = 0
                continue
            if q_prev is not None and np.float64(np.abs(q - q_prev).max()) < 1e-4:
                stable += 1
                if stable >= 5:
                    if float(np.abs(q - target).max()) < arm_kin.SETTLE_TOL:
                        return self.client._node.get_clock().now()
            else:
                stable = 0
            q_prev = q.copy()
        log.warning("Arm did not settle at target pose within timeout.")
        return self.client._node.get_clock().now()

    def detect(self, target_pos=None, visualize=False):
        t_settle = self.look_down(target_pos=target_pos)
        t_settle += Duration(seconds=0.0)
        depth = self.di.get_fresh_depth("gripper", t_settle)
        if depth is None:
            log.warning("No fresh gripper depth frame received after arm settled.")
            return None
        T_base_cam = self.get_extrinsic()
        d2 = self.di.remove_gripper(self.p["gripper_depth"])
        d3 = self.di.delete_bottom(d2, ratio=0.1)
        grad = self.di.gradient_threshold(d3, threshold=self.p["grad_thresh"])

        res = self.di.center_rect(grad, self.p["min_area_px"], visualize)
        if res is None:
            return None
        (cx,cy), angle = res

        p_cam = self.di.get_camera_coordinates(cx, cy)
        if p_cam is None:
            return None
        p_base = T_base_cam @ p_cam
        log.debug(f"World coordinate: {p_base[:3]}")

        yaw = self.get_world_yaw(T_base_cam, angle)
        yaw = self.clip_yaw(yaw)
        yaw = yaw + self.p["yaw_offset"]
        return float(p_base[0]), float(p_base[1]), float(yaw)


        
        



