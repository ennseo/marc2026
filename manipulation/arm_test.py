"""tester.py -- manipulation_trainer에서 픽 로직만 반복 테스트하는 하니스. 
 

사용: trainer 띄운 상태에서 
    export MARC_TEAM_ID=your_team_id MARC_TOKEN=your_token
    python3 tester.py [fwd] [lat]        # grasp_xy 생략 시 pnp_params 기본값 
GUI에서 Place near/far로 물체 놓고 Enter 치면 1회 실행 -> 결과 집계. 
""" 
import logging 
import sys 
from pathlib import Path

# Allow direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from image import DepthImage, image_show
from marc_sdk import MARCClient 
from arm_pick import run_pick_sequence, _make_js, load_params, _read_arm_q, HOME, GRIPPER_OPEN
from grasp import GraspGenerator
 

log = logging.getLogger("tester") 

p = load_params()

class PickTester: 
    def __init__(self, visualize=False): 
        self.client = MARCClient.from_env()
        self.client.connect(timeout=0.0)
        self.grasp = GraspGenerator(self.client)
        self.visualize = visualize

    def run_once(self): 
            target_pos = np.array([p["scan_x"], p["scan_y"], p["scan_z"]])
            target = self.grasp.detect(target_pos, visualize=self.visualize)
            if target is None:
                log.info("No graspable object detected.")
                return
            x, y, angle = target
            log.info(f"obj coordinate (world): {(x,y)}, grasp angle: {angle} ")
            return run_pick_sequence(grasp_xy=(x, y), yaw=angle, log=log, client=self.client)

    def run(self): 
            while True: 
                if self.run_once():
                     break
 

def main(): 
    np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})
    logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    visualize = len(sys.argv) > 1
    log.info(f"visualize: {visualize}")
    PickTester(visualize).run() 

if __name__ == "__main__": 
    main() 
