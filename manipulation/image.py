import time
import numpy as np
import cv2

from sensor_msgs.msg import CameraInfo
from marc_sdk import MARCClient

MIN_AREA_PX = 1000 #사각형 최대 크기 px단위
def normalize_image(img_array, target_min=0, target_max=255, dtype=cv2.CV_8U):
    """
    Normalizes a numpy array image using OpenCV's highly optimized backend.
    """
    # 1. Handle NaN values first (OpenCV will crash or output garbage if given NaNs)
    img_array = np.nan_to_num(img_array)
    
    # 2. Use cv2.normalize
    normalized = cv2.normalize(
        img_array, 
        None, 
        alpha=target_min,   # lower bound
        beta=target_max,    # upper bound
        norm_type=cv2.NORM_MINMAX, 
        dtype=dtype
    )
    return normalized

def image_show(img_array, window_name="Image", wait_time=0, normalize=False, target_min=0, target_max=255, dtype=cv2.CV_8U):
    """
    Displays an image using OpenCV's imshow function.
    """
    if normalize:
        img_array = normalize_image(img_array, target_min, target_max, dtype)
    cv2.imshow(window_name, img_array)
    cv2.waitKey(wait_time)  # Wait for a key press; 0 means wait indefinitely
    cv2.destroyAllWindows()  # Close the window after displaying


class DepthImage:
    def __init__(self, client, depth = None, log = None):
        self.client = client
        self.log = log
        self.depth = depth
        self.intrinsics = None  # (fx, fy, cx, cy) 저장용

        topic_name = "/marc/dongdong/robot/gripper_camera/depth/info"
        self.client.subscribe(
            topic_name, 
            CameraInfo, 
            self._camera_info_callback
        )

    def get_robot_depth(self, which):
        depth_image = self.client.get_robot_depth(which)
        while True:
            depth_image = self.client.get_robot_depth(which)
            if depth_image is not None:
                break
        img_array = np.frombuffer(depth_image.data, dtype=np.float32)
        img_array = img_array.reshape((depth_image.height, depth_image.width))

        self.depth = img_array
        return self.depth

    def get_fresh_depth(self, which, t_after, timeout=2.0):
        t_min = t_after.nanoseconds
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            msg = self.client.get_robot_depth(which)
            if msg is not None:
                stamp = msg.header.stamp
                t_msg = stamp.sec * 10**9 + stamp.nanosec
                if t_msg > t_min:
                    img_array = np.frombuffer(msg.data, dtype=np.float32)
                    self.depth = img_array.reshape((msg.height, msg.width))
                    return self.depth
            self.client.sleep(0.02)

    def gradient_threshold(self, depth, threshold=0.1):
        # Compute gradients using Sobel operator
        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=5)

        # Compute gradient magnitude
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        # Create a binary mask based on the threshold
        edge_mask = (grad_magnitude > threshold).astype(np.uint8)

        return edge_mask

    def _camera_info_callback(self, msg):
        """
        Callback function to handle incoming CameraInfo messages.
        Extracts and stores the camera intrinsics (fx, fy, cx, cy).
        """
        if self.intrinsics is not None:
            return
        self.intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])  # fx, fy, cx, cy
        if self.log:
            self.log.debug(f"Camera intrinsics set: fx={self.intrinsics[0]}, fy={self.intrinsics[1]}, cx={self.intrinsics[2]}, cy={self.intrinsics[3]}")

    def get_camera_coordinates(self, u, v):
        # 1. Pixel to Camera 3D (Deprojection)
        fx, fy, cx, cy = self.intrinsics

        u, v = int(round(u)), int(round(v))
        z_c = self.median_depth(u,v)
        if z_c is None:
            return None
        x_c = (u - cx) * z_c / fx
        y_c = (v - cy) * z_c / fy
        
        point_camera = np.array([x_c, y_c, z_c, 1.0])        
        return point_camera 

    def remove_gripper(self, threshold = 0.1):
        """
        Removes the gripper from the depth image by applying a threshold.
        Pixels with depth values below the threshold are set to zero.

        Parameters:
        - threshold: float, the depth value below which pixels are considered part of the gripper.

        Returns:
        - cleaned_depth: 2D numpy array with the gripper removed.
        """
        if self.depth is None:
            raise ValueError("Depth image not set. Call get_robot_depth() first.")
        
        depth = np.copy(self.depth)
        depth[depth < threshold] = np.nan
        return depth

    def min_rect(self, mask, min_area, visualize=False):
        m = (mask > 0).astype(np.uint8)
        
        cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnt:
            return None

        largest = max(cnt, key=cv2.contourArea)
        if cv2.contourArea(largest) < min_area:
            return None

        rect = cv2.minAreaRect(largest)
        (cx, cy), _, _ = rect
        box = cv2.boxPoints(rect)

        e1, e2 = box[1] - box[0], box[2] - box[1]
        short = e1 if np.hypot(*e1) < np.hypot(*e2) else e2
        angle = float(np.arctan2(short[1], short[0]))
        angle = (angle + np.pi / 2) % np.pi - np.pi / 2

        if visualize:
            d8 = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            self.vis = cv2.cvtColor(d8, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(self.vis, [np.intp(box)], 0, (0, 255, 0), 2)
            cv2.circle(self.vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)

            # 3. Angle 방향 선(Line) 시각화
            line_len = 30  # 그릴 선의 길이 (픽셀 단위)
            
            # angle 방향으로 벡터 계산
            dx = int(line_len * np.cos(angle))
            dy = int(line_len * np.sin(angle))
            
            # 중심을 지나는 선 (양방향으로 연장)
            p1 = (int(cx) - dx, int(cy) - dy)
            p2 = (int(cx) + dx, int(cy) + dy)
            
            # 파란색 선으로 방향 표시
            cv2.line(self.vis, p1, p2, (255, 0, 0), 2)

            cv2.imshow("Min Area Rect", self.vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return (float(cx), float(cy)), angle

    def center_rect(self, mask, min_area, visualize=False):
        m = (mask > 0).astype(np.uint8)
        image_center = (m.shape[1] / 2.0, m.shape[0] / 2.0)
        
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        valid_cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
        if not valid_cnts:
            return None
        def get_distance_to_center(c):
            (cx, cy), _, _ = cv2.minAreaRect(c)
            return np.hypot(cx - image_center[0], cy - image_center[1])

        # 이미지 중앙에서 가장 가까운 Contour 선정
        target_cnt = min(valid_cnts, key=get_distance_to_center)

        # 선택된 단일 Contour에 대해서만 minAreaRect 계산
        rect = cv2.minAreaRect(target_cnt)
        (cx, cy), _, _ = rect
        box = cv2.boxPoints(rect)

        e1, e2 = box[1] - box[0], box[2] - box[1]
        short = e1 if np.hypot(*e1) < np.hypot(*e2) else e2
        angle = float(np.arctan2(short[1], short[0]))
        angle = (angle + np.pi / 2) % np.pi - np.pi / 2

        if visualize:
            d8 = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            self.vis = cv2.cvtColor(d8, cv2.COLOR_GRAY2BGR)
            
            # 감지된 모든 컨투어 표시 (연한 회색)
            cv2.drawContours(self.vis, cnts, -1, (100, 100, 100), 1)
            
            # 선택된 가장 중앙의 사각형 표시 (녹색)
            cv2.drawContours(self.vis, [np.intp(box)], 0, (0, 255, 0), 2)
            cv2.circle(self.vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)

            # Angle 방향 선(Line) 시각화
            line_len = 30
            dx = int(line_len * np.cos(angle))
            dy = int(line_len * np.sin(angle))
            
            p1 = (int(cx) - dx, int(cy) - dy)
            p2 = (int(cx) + dx, int(cy) + dy)
            
            cv2.line(self.vis, p1, p2, (255, 0, 0), 2)

            cv2.imshow("Min Area Rect", self.vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return (float(cx), float(cy)), angle

    
    def median_depth(self, u, v, k=3):
        p = self.depth[max(0,v-k):v+k+1, max(0,u-k):u+k+1]
        p = p[np.isfinite(p) & (p > 0)]
        return float(np.median(p)) if p.size else None

    def delete_bottom(self, depth, ratio = 0.2):
        out = np.copy(depth)
        h = out.shape[0]
        cut = int(round(h * (1.0 - ratio)))
        out[cut:, :] = np.nan
        return out



class ColorImage:
    def __init__(self, client, log = None):
        self.client = client
        self.log = log
        self.color = None

    def get_robot_image(self, which):
        color_image = self.client.get_robot_image(which)
        while True:
            color_image = self.client.get_robot_image(which)
            if color_image is not None:
                break
        img_array = np.frombuffer(color_image.data, dtype=np.uint8)
        img_array = img_array.reshape((color_image.height, color_image.width, 3))

        self.color = img_array
        return self.color


if __name__ == "__main__":
    import logging
    from arm_pick import load_params

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    client = MARCClient.from_env()
    client.connect(timeout=0.0)
    p = load_params()

    depth_image = DepthImage(client, log=logging.getLogger("DepthImage"))
    depth = depth_image.get_robot_depth("gripper")
    image_show(depth, window_name="Depth Image", wait_time=0, normalize=True)
    depth = depth_image.remove_gripper(threshold=p["gripper_depth"])
    depth = depth_image.delete_bottom(depth, 0.2)
    image_show(depth, window_name="Depth Image without Gripper + crop bottom", wait_time=0, normalize=True)
    mask = depth_image.gradient_threshold(depth, threshold=p["grad_thresh"])
    image_show(mask, window_name="Gradient Threshold", wait_time=0, normalize=True)
    depth_image.min_rect(mask, 500, True)