"""
INDOOR LOCALIZATION - LANE-AIDED HEADING CORRECTION (centre RGB camera)
=========================================================================

The robot is given NO lane map. The only assumption used here is the
structural one every warehouse floor plan satisfies: painted travel lanes
run parallel to the building walls (along world X or world Y).

    1. The centre camera's lane mask (lane_detector) is projected onto the
       floor plane through the camera's known mount (height, pitch,
       intrinsics) using a per-pixel look-up table, giving lane pixels as
       (forward, lateral) metres in the robot body frame. The same table
       feeds the live LANE LAYER of the map (occupancy_map.integrate_lane).
    2. The strip is sliced into 10 cm bands along the forward axis; every
       band where the FULL strip width is visible gives one centreline
       point. A line lateral = a + b * forward is fitted through them.
    3. While the robot is driving ALONG a straight piece of lane, the
       strip's world direction (estimated heading + atan b) must be a
       multiple of 90 deg. The difference is the heading error of the pose
       estimate and is fed back with a small gain.

Heading is what makes odometry drift sideways, so this keeps the estimate
honest between the landmark fixes (tool stands, trailer, table / shelf
depth-servo) without any prior map. Position is not corrected here.

Gates keep the correction from ever doing harm: only a single, straight,
nearly centred strip (no junctions - a crossing strip breaks the band
width / fit test), a good fit, and a small innovation.
"""

import math

import numpy as np

import camera_rig

BAND_M = 0.10
FWD_MIN_M = 0.55
FWD_MAX_M = 2.40
STRIP_WIDTH_RANGE_M = (0.42, 0.80)
MIN_BANDS = 5
MAX_FIT_RMS_M = 0.035
GATE_YAW_RAD = math.radians(5.0)
# Only trust the lane while the robot is actually DRIVING ALONG it: the
# floor projection of a strip seen far off to the side or at a steep angle
# is biased by a few degrees (measured: +4 deg with a strip 0.65 m to the
# side at 20 deg), which near junctions made the correction worse than no
# correction. Larger errors are left to the stand-leg landmark fix.
MAX_STRIP_OFFSET_M = 0.60          # fitted centreline within this of the robot axis
MAX_STRIP_ANGLE_RAD = math.radians(12.0)
GAIN_YAW = 0.12


def _floor_lut():
    """(H, W, 2) body-frame floor coordinates for every centre-camera pixel
    (nan where the ray does not hit the floor within 5 m)."""

    K = camera_rig.CENTER_INTRINSICS
    h, w = camera_rig.CENTER_HEIGHT, camera_rig.CENTER_WIDTH
    eye, forward, up = camera_rig.build_camera_frame(0.0, 0.0, 0.0, "center")
    right = np.cross(forward, up)
    rot = np.column_stack((right, -up, forward))
    us, vs = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    rays = np.stack(((us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us)), -1) @ rot.T
    t = np.where(rays[..., 2] < -1e-6, -eye[2] / np.minimum(rays[..., 2], -1e-6), np.nan)
    pts = eye[None, None, :2] + t[..., None] * rays[..., :2]
    far = np.hypot(pts[..., 0], pts[..., 1]) > 5.0
    pts[far] = np.nan
    return pts


class LaneLocalizer:
    def __init__(self):
        self.lut = _floor_lut()
        self.updates = 0
        self.rejections = 0
        self.last_update_t = -1e9
        self.last_innovation = (0.0, 0.0)
        self.last_fit = None            # (a, b): lateral = a + b*forward, metres

    def fit_centreline(self, lane_mask):
        """Returns (a, b, n_bands) for lateral = a + b*forward, or None."""

        if lane_mask is None:
            return None
        sel = lane_mask > 0
        pts = self.lut[sel]
        pts = pts[np.isfinite(pts[:, 0])]
        pts = pts[(pts[:, 0] >= FWD_MIN_M) & (pts[:, 0] <= FWD_MAX_M)]
        if pts.shape[0] < 60:
            return None
        bins = np.floor((pts[:, 0] - FWD_MIN_M) / BAND_M).astype(np.int64)
        fwd_c, lat_c = [], []
        for k in np.unique(bins):
            band = pts[bins == k]
            if band.shape[0] < 4:
                continue
            lo, hi = band[:, 1].min(), band[:, 1].max()
            if not (STRIP_WIDTH_RANGE_M[0] <= hi - lo <= STRIP_WIDTH_RANGE_M[1]):
                continue
            fwd_c.append(float(band[:, 0].mean()))
            lat_c.append(0.5 * (lo + hi))
        if len(fwd_c) < MIN_BANDS:
            return None
        f, l = np.array(fwd_c), np.array(lat_c)
        A = np.stack((np.ones_like(f), f), 1)
        (a, b), *_ = np.linalg.lstsq(A, l, rcond=None)
        rms = float(np.sqrt(np.mean((A @ np.array([a, b]) - l) ** 2)))
        if rms > MAX_FIT_RMS_M:
            return None
        return float(a), float(b), len(f)

    def correct(self, pose, lane_mask, now):
        """Returns the corrected (x, y, yaw) tuple, or None if no
        correction was applied."""

        fit = self.fit_centreline(lane_mask)
        self.last_fit = None if fit is None else (fit[0], fit[1])
        if fit is None:
            return None
        a, b, _n = fit
        if abs(a + b * 1.2) > MAX_STRIP_OFFSET_M or abs(math.atan(b)) > MAX_STRIP_ANGLE_RAD:
            return None
        lane_heading = pose.yaw + math.atan(b)
        snapped = round(lane_heading / (math.pi / 2.0)) * (math.pi / 2.0)
        d_yaw = math.atan2(math.sin(snapped - lane_heading), math.cos(snapped - lane_heading))
        if abs(d_yaw) > GATE_YAW_RAD:
            self.rejections += 1
            return None
        self.updates += 1
        self.last_update_t = now
        self.last_innovation = (d_yaw, 0.0)
        new_yaw = pose.yaw + GAIN_YAW * d_yaw
        return pose.x, pose.y, math.atan2(math.sin(new_yaw), math.cos(new_yaw))
