"""Camera parameter collection + the callable detection asks for coordinates with.

`transform.py` is deliberately pure math -- it takes K/R/t and returns numbers. Something
still has to gather those off the wire: intrinsics from CameraInfo, extrinsics from
/tf_static, the ground height from its own latched topic. That plumbing lives here.

This is the "geometry as a function, not a stage" contract in code (see geometry/README.md):

    geom = CameraGeometry(client)
    xyz  = geom.unproject(camera_id, u, v)     # called ~10-20x per round by detection

Detection calls it while it is still choosing between candidates, which is why it must be
cheap and stateless-per-call rather than a step that runs once after detection finishes.

Ownership note: 효빈 owns this file together with transform.py. It is implemented rather
than stubbed so detection is not blocked.

The ground plane is per-camera, not global: `client.get_cctv_ground_height(cid)` returned
six different values across the six practice cameras (16.1 to 16.8 m), so always ask the
camera the pixel came from. `params()` already does.
"""

import logging
import threading
import time

import numpy as np

from tf2_msgs.msg import TFMessage

from marc_sdk import protocol as P

from .transform import (LANDMARK_HALF_DEPTH, LANDMARK_TOP_Z, push_from_camera,
                        quat_to_matrix, unproject_to_ground,
                        unproject_to_plane)

log = logging.getLogger("participant_app")


class CameraGeometry:
    """Per-camera parameters + pixel->world conversion.

    Holds the /tf_static extrinsics (latched, so one subscription is enough for the whole
    run) and reads intrinsics/ground height from the SDK on demand.
    """

    def __init__(self, client):
        self.client = client
        self._tf = {}                       # frame_id -> (R, t)
        self._lock = threading.Lock()
        # The SDK does not subscribe to /tf_static itself, so we attach through its
        # escape hatch. TRANSIENT_LOCAL matches the latched publisher, meaning a late
        # subscriber still receives every transform.
        client.subscribe("/tf_static", TFMessage, self._on_tf_static, P.QOS_TRANSIENT)

    def _on_tf_static(self, msg):
        with self._lock:
            for tr in msg.transforms:
                q, v = tr.transform.rotation, tr.transform.translation
                self._tf[tr.child_frame_id] = (
                    quat_to_matrix(q.x, q.y, q.z, q.w),
                    np.array([v.x, v.y, v.z]),
                )
        log.info("[geom] tf_static: %d frames known", len(self._tf))

    def wait_ready(self, timeout=10.0):
        """Block until at least one /tf_static transform has arrived. True if ready.

        Bounded by WALL time, not sim time. This runs on the mission-handling path where
        the sim clock (/clock) may be paused or not yet flowing; the old sim-time loop
        could then spin forever instead of timing out.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._tf:
                    return True
            time.sleep(0.1)
        log.warning("[geom] no /tf_static after %.0fs -- coordinates unavailable this round",
                    timeout)
        return False

    def params(self, camera_id):
        """(K, R, t, ground_z) for a camera, or None if anything is missing yet."""
        info = self.client.get_cctv_info(camera_id)
        ground = self.client.get_cctv_ground_height(camera_id)
        with self._lock:
            tf = self._tf.get(camera_id)     # tf frame name == camera_id (2026.R01)
        if info is None or ground is None or tf is None:
            return None
        R, t = tf
        return (np.array(info.k, dtype=float).reshape(3, 3), R, t, float(ground))

    def unproject(self, camera_id, u, v, on_landmark=None, as_landmark=None):
        """Pixel -> world [x, y, z]. The call detection makes for every candidate.

        Args:
            camera_id: which camera the pixel came from.
            u, v: pixel coordinates (use geometry.ground_pixel(bbox) for the contact point).
            on_landmark: landmark type name when the NLU says the target rests *on* it.
                Raises the intersection plane to that landmark's resting surface instead of
                the ground, which is what keeps a mug on a table from landing metres away.
                The lift is measured from this camera's ground plane -- see the note on
                LANDMARK_TOP_Z for why it is not measured from the landmark's own anchor.
        Returns:
            [x, y, z], or None if parameters are missing or the ray misses the plane.
        """
        got = self.params(camera_id)
        if got is None:
            return None
        K, R, t, ground = got

        if on_landmark:
            lift = LANDMARK_TOP_Z.get(on_landmark)
            if lift is not None:
                return unproject_to_plane(u, v, K, R, t, ground + lift)
        p = unproject_to_ground(u, v, K, R, t, ground)
        if p is not None and as_landmark:
            # The pixel is a landmark's ground contact, i.e. the near edge of its
            # footprint, but the grader wants the landmark's centre. Slide it back along
            # the view ray. See LANDMARK_HALF_DEPTH for the measurement.
            p = push_from_camera(p, t, LANDMARK_HALF_DEPTH.get(as_landmark))
        if p is None:
            # The frame convention was settled empirically on 2026-08-24 (see
            # transform.TF_IS_OPTICAL_FRAME), so this is no longer the usual suspect.
            # A ray that misses now normally means the pixel is above the horizon --
            # a bbox whose bottom edge is not actually a ground contact point.
            log.warning("[geom] %s: ray misses the ground plane at (%.1f, %.1f) -- "
                        "pixel is probably above the horizon", camera_id, u, v)
        return p
