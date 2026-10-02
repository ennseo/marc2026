#!/usr/bin/env python3
"""Record a whole Stage 1 run to plain files -- the offline test set.

Everything the perception pipeline needs (CCTV frames, intrinsics, ground
heights, extrinsics, and the voice command of every round) only exists while the
simulator is running on a GPU machine. This dumps all of it to png + json so the
rest of the team can develop and evaluate without a GPU and without going back
to the lab.

This is a **passive listener**: `/marc/ops/announce` is a broadcast topic, so no
team token and no registration are needed, and nothing here interferes with
scoring. It does NOT advance the rounds by itself -- run it alongside an agent
that does (the stock demo is enough):

    # terminal 1 -- the platform (already running)
    # terminal 2 -- any agent, to drive the rounds forward
    cd demo && ./launch.sh u1
    # terminal 3 -- this recorder
    source /opt/ros/humble/setup.bash
    cd demo && python3 tools/dump_cctv.py ../cctv_dump

Pass MARC_TEAM_ID (default u1) to also capture that team's per-round scores;
listening to a response topic needs no token either.

    python3 tools/dump_cctv.py ../cctv_dump --once     # single snapshot, no rounds

Output:
    cctv_dump/meta.json          per-camera K / ground_height / tf + every tf frame
    cctv_dump/round_01/          info.json (command, score) + <camera>.png
    cctv_dump/round_02/ ...
    cctv_dump/stage2/            the Stage 2 task, same layout
"""

import json
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                       DurabilityPolicy)
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32, String
from tf2_msgs.msg import TFMessage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.images import image_bytes_to_numpy  # noqa: E402

CCTV_PREFIX = "/marc/env/cctv/"
ANNOUNCE = "/marc/ops/announce"

MSG_MISSION = 201
MSG_STAGE2_MISSION = 211
MSG_SCORE = 401

QOS_IMAGE = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                       durability=DurabilityPolicy.VOLATILE,
                       history=HistoryPolicy.KEEP_LAST, depth=1)
QOS_RELIABLE = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.VOLATILE,
                          history=HistoryPolicy.KEEP_LAST, depth=10)
QOS_LATCHED = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=100)


def save_image(array, path_noext):
    """png via pillow when available, else .npy so there is never a hard dep."""
    try:
        from PIL import Image as PILImage
        PILImage.fromarray(array).save(path_noext + ".png")
        return os.path.basename(path_noext) + ".png"
    except ImportError:
        np.save(path_noext + ".npy", array)
        return os.path.basename(path_noext) + ".npy"


class Recorder(Node):

    def __init__(self, out_dir, team_id):
        super().__init__("marc_cctv_recorder")
        self.out_dir = out_dir
        self.images = {}       # cid -> latest Image msg
        self.counts = {}       # cid -> frames received (freshness check)
        self.infos = {}
        self.grounds = {}
        self.tf = {}
        self.pending = []      # missions captured by the callback, drained by main
        self.rounds = []       # written records, for score back-fill
        # Scores whose round record did not exist yet when they arrived. capture() spends
        # up to ~2s waiting for fresh frames and then writes 6 PNGs *before* appending to
        # self.rounds, and wait_fresh() spins the executor throughout -- so a score can be
        # delivered into that window. It used to be dropped on the floor: the 2026-08-19
        # dump lost 7 of 39 rounds that way (01, 03, 08, 13, 29, 33, 39), with no pattern,
        # because whether it happens depends on how quickly that round's frames arrived.
        # Losing a score loses the answer key, which is what the offline evaluation runs
        # against, so the round becomes useless for scoring work.
        self.late_scores = {}  # round number -> scores payload
        self.stage2_seen = False

        self.create_subscription(TFMessage, "/tf_static", self._on_tf, QOS_LATCHED)
        self.create_subscription(String, ANNOUNCE, self._on_announce, QOS_RELIABLE)
        if team_id:
            self.create_subscription(
                String, f"/marc/ops/{team_id}/response", self._on_response, QOS_LATCHED)
            self.get_logger().info(f"also listening for scores of team '{team_id}'")
        self.create_timer(2.0, self._discover)

    # -- subscriptions --

    def _on_tf(self, msg):
        for tr in msg.transforms:
            t, q = tr.transform.translation, tr.transform.rotation
            self.tf[tr.child_frame_id] = {
                "parent": tr.header.frame_id,
                "translation": [t.x, t.y, t.z],
                "rotation_xyzw": [q.x, q.y, q.z, q.w],
            }

    def _on_announce(self, msg):
        try:
            data = json.loads(msg.data)
            mid = data["header"]["msg"]
            payload = data["payload"]
        except (json.JSONDecodeError, KeyError):
            return
        if mid == MSG_MISSION:
            self.pending.append(("round", payload))
            self.get_logger().info(
                f"round {payload.get('round')}/{payload.get('total_rounds')}: "
                f"{payload.get('voice_command', '')[:60]}")
        elif mid == MSG_STAGE2_MISSION and not self.stage2_seen:
            self.stage2_seen = True
            self.pending.append(("stage2", payload))
            self.get_logger().info(
                f"stage2: {payload.get('task_description', '')[:60]}")

    def _on_response(self, msg):
        """Attach msg 401 scores to the round they belong to."""
        try:
            data = json.loads(msg.data)
            if data["header"]["msg"] != MSG_SCORE:
                return
            payload = data["payload"]
        except (json.JSONDecodeError, KeyError):
            return
        rnd = payload.get("round")
        scores = payload.get("scores")
        for rec in reversed(self.rounds):
            if rec.get("round") == rnd or (rnd is None and rec is self.rounds[-1]):
                rec["score"] = scores
                self._write_info(rec)
                return
        # No record yet -- hold it for capture() to pick up (see self.late_scores).
        if rnd is not None:
            self.late_scores[rnd] = scores
        else:
            self.get_logger().warning(
                "score arrived with no round number and no record to attach it to")

    def _discover(self):
        for name, _types in self.get_topic_names_and_types():
            if not (name.startswith(CCTV_PREFIX) and name.endswith("/image")):
                continue
            cid = name[len(CCTV_PREFIX):-len("/image")]
            if cid in self.images:
                continue
            self.images[cid] = None
            self.counts[cid] = 0
            self.create_subscription(Image, name,
                                     lambda m, c=cid: self._on_image(c, m), QOS_IMAGE)
            self.create_subscription(
                CameraInfo, f"{CCTV_PREFIX}{cid}/info",
                lambda m, c=cid: self.infos.__setitem__(c, m), QOS_IMAGE)
            self.create_subscription(
                Float32, f"{CCTV_PREFIX}{cid}/ground_height",
                lambda m, c=cid: self.grounds.__setitem__(c, float(m.data)), QOS_LATCHED)
            self.get_logger().info(f"found CCTV: {cid}")

    def _on_image(self, cid, msg):
        self.images[cid] = msg
        self.counts[cid] = self.counts.get(cid, 0) + 1

    # -- capture --

    @property
    def have_images(self):
        return bool(self.images) and any(v is not None for v in self.images.values())

    def wait_fresh(self, timeout=2.0):
        """Spin until every camera delivers a frame newer than right now.

        The mission announcement and the video stream are independent, so the
        cached frame at announce time may predate the round. Waiting for a new
        frame per camera keeps the images aligned with the command.
        """
        before = dict(self.counts)
        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if all(self.counts.get(c, 0) > before.get(c, 0) for c in self.images):
                return True
        return False

    def capture(self, kind, payload):
        if kind == "stage2":
            name = "stage2"
        else:
            name = f"round_{len(self.rounds) + 1:02d}"
        sub_dir = os.path.join(self.out_dir, name)
        os.makedirs(sub_dir, exist_ok=True)

        self.wait_fresh()
        written = {}
        for cid, msg in self.images.items():
            if msg is None:
                continue
            array = image_bytes_to_numpy(
                msg.data, msg.height, msg.width, msg.encoding)
            written[cid] = save_image(array, os.path.join(sub_dir, cid))

        rec = {
            "dir": name,
            "kind": kind,
            "round": payload.get("round"),
            "total_rounds": payload.get("total_rounds"),
            "voice_command": payload.get("voice_command"),
            "task_description": payload.get("task_description"),
            "owner_position": payload.get("owner_position"),
            "time_limit": payload.get("time_limit"),
            "images": written,
            "score": None,
        }
        self.rounds.append(rec)
        # A score for this round may already have arrived while we were capturing.
        held = self.late_scores.pop(rec["round"], None)
        if held is not None:
            rec["score"] = held
        self._write_info(rec)
        print(f"[dump] {name}: {len(written)} image(s)  "
              f"{(rec['voice_command'] or rec['task_description'] or '')[:56]}")

    def _write_info(self, rec):
        path = os.path.join(self.out_dir, rec["dir"], "info.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)

    def write_meta(self):
        meta = {"cameras": {}, "all_tf_frames": self.tf}
        for cid, msg in self.images.items():
            if msg is None:
                continue
            info = self.infos.get(cid)
            meta["cameras"][cid] = {
                "width": msg.width,
                "height": msg.height,
                "encoding": msg.encoding,
                "K": list(info.k) if info is not None else None,
                # Distortion coefficients. geometry/transform.py ignores distortion and
                # has had no way to check whether that is safe, because this recorder
                # never captured `d`. Empty/all-zero here means the caveat can be closed;
                # non-zero means (u, v) must be undistorted before unprojecting.
                "d": list(info.d) if info is not None else None,
                "ground_height": self.grounds.get(cid),
                # tf frame name == camera id (2026.R01 release note)
                "tf": self.tf.get(cid),
            }
            flags = []
            if info is None:
                flags.append("K MISSING")
            if self.grounds.get(cid) is None:
                flags.append("ground_height MISSING")
            if cid not in self.tf:
                flags.append("tf MISSING")
            print(f"[dump] camera {cid}: {' / '.join(flags) if flags else 'complete'}")
        with open(os.path.join(self.out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return meta


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    once = "--once" in sys.argv
    out_dir = args[0] if args else "cctv_dump"
    idle_stop = float(args[1]) if len(args) > 1 else 600.0
    team_id = os.environ.get("MARC_TEAM_ID", "u1")
    os.makedirs(out_dir, exist_ok=True)

    rclpy.init()
    node = Recorder(out_dir, team_id)

    print("[dump] waiting for CCTV topics ...")
    deadline = time.time() + 30.0
    while time.time() < deadline and not node.have_images:
        rclpy.spin_once(node, timeout_sec=0.2)
    if not node.have_images:
        print("[dump] no CCTV frames arrived. Is the platform running, and is "
              "ROS_DOMAIN_ID the same on both machines?")
        node.destroy_node()
        rclpy.shutdown()
        return

    node.write_meta()

    if once:
        node.capture("round", {})
        print(f"\n[dump] single snapshot -> {out_dir}/")
    else:
        print(f"\n[dump] recording rounds. Start an agent to drive them "
              f"(cd demo && ./launch.sh {team_id}).")
        print(f"[dump] stops after {idle_stop:.0f}s with no new round, or Ctrl-C.\n")
        last_activity = time.time()
        try:
            while time.time() - last_activity < idle_stop:
                rclpy.spin_once(node, timeout_sec=0.2)
                while node.pending:
                    kind, payload = node.pending.pop(0)
                    node.capture(kind, payload)
                    last_activity = time.time()
        except KeyboardInterrupt:
            print("\n[dump] interrupted")

    # Re-write meta at the end: latched topics sometimes land late.
    meta = node.write_meta()
    complete = sum(1 for c in meta["cameras"].values()
                   if c["K"] and c["tf"] and c["ground_height"] is not None)

    print(f"\n[dump] {len(node.rounds)} round(s), {len(meta['cameras'])} camera(s) "
          f"({complete} with complete geometry) -> {out_dir}/")
    if not node.tf:
        print("[dump] WARNING: /tf_static was empty -- is the platform on 2026.R01+?")
    if complete == 0:
        print("[dump] WARNING: no camera has K + tf + ground_height. The offline "
              "geometry check will not work.")

    # Worth shouting about while the platform is still up and a re-run is cheap: a round
    # without a score has no answer key, so it cannot be used to evaluate anything.
    unscored = [r["dir"] for r in node.rounds if not r.get("score")]
    if unscored:
        print(f"[dump] WARNING: {len(unscored)} of {len(node.rounds)} round(s) have no "
              f"score and are therefore unusable as evaluation data:")
        print(f"[dump]          {', '.join(unscored)}")
        print("[dump]          Re-run the scenario before leaving if you can.")
    else:
        print(f"[dump] all {len(node.rounds)} round(s) carry a score.")
    if node.late_scores:
        print(f"[dump] NOTE: {len(node.late_scores)} score(s) arrived for rounds that were "
              f"never captured: {sorted(node.late_scores)}")
    print("[dump] zip this folder and share it with the team.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
