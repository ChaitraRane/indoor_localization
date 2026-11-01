"""
INDOOR LOCALIZATION - RGB-D VISUAL ODOMETRY + ONBOARD POSE
============================================================

Ego-motion and pose for the AMR using ONLY onboard sensing (no ground
truth, no LiDAR): the two forward RGB-D cameras provide visual odometry,
and the two wheel encoders provide wheel odometry. They are fused into a
single canonical pose the mapping and planning layers consume.

VISUAL ODOMETRY (adapted from robot_vision stage4_visual_odometry.py)
    ORB features -> BF Hamming match + Lowe ratio -> depth-backproject
    the matched previous-frame pixels to 3D -> solvePnPRansac
    (EPnP in RANSAC, iterative fallback) for the frame-to-frame motion.
    Depth is available (RGB-D), so this is depth-backed PnP rather than
    an essential-matrix method, exactly as the outdoor stack did.

FUSION (VIWO-lite, no IMU)
    Wheel odometry is the backbone (differential-drive kinematics from
    encoder deltas - robust and drift-bounded over short steps), and
    visual odometry corrects the heading and forward-scale when it has
    enough inliers to be trusted. This mirrors the robot_vision VIWO
    estimator's principle (one canonical fused pose; vision and wheels
    are measurements into it, never competing poses) in a compact form.

This file is part of the indoor_localization project only; nothing here
reads, imports, or depends on anything in the robot_vision folder.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import mission_config as cfg


# ============================================================
# CAMERA INTRINSICS
# ============================================================

def intrinsics_from_fov(width, height, fov_deg):
    """Pinhole camera matrix from a vertical FOV (PyBullet's
    computeProjectionMatrixFOV uses the vertical field of view)."""

    fov_rad = np.radians(fov_deg)
    f_y = (height / 2.0) / np.tan(fov_rad / 2.0)
    f_x = f_y   # square pixels
    c_x = width / 2.0
    c_y = height / 2.0
    return np.array(
        [[f_x, 0.0, c_x], [0.0, f_y, c_y], [0.0, 0.0, 1.0]], dtype=np.float64
    )


# ============================================================
# ORB FEATURES + MATCHING (adapted from stage4)
# ============================================================

ORB_FEATURES = 1000
MATCH_RATIO = 0.75
MIN_PNP_POINTS = 8
MIN_PNP_INLIERS = 6
PNP_ITERATIONS = 100
PNP_REPROJECTION_ERROR = 3.0
PNP_CONFIDENCE = 0.99


def detect_features(gray):
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(previous_descriptors, current_descriptors):
    if previous_descriptors is None or current_descriptors is None:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = matcher.knnMatch(previous_descriptors, current_descriptors, k=2)

    accepted = []
    for pair in raw:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < MATCH_RATIO * second.distance:
            accepted.append(best)

    accepted.sort(key=lambda m: m.distance)
    return accepted


def backproject(pixel_uv, depth_value, camera_matrix):
    """Pixel (u, v) + metric depth -> 3D point in the camera optical
    frame (x right, y down, z forward)."""

    f_x = camera_matrix[0, 0]
    f_y = camera_matrix[1, 1]
    c_x = camera_matrix[0, 2]
    c_y = camera_matrix[1, 2]
    u, v = pixel_uv
    z = depth_value
    x = (u - c_x) * z / f_x
    y = (v - c_y) * z / f_y
    return np.array([x, y, z], dtype=np.float64)


def optical_to_world_rotation(cam_forward, cam_up):
    """3x3 rotation whose columns map the optical frame axes
    (x right, y down, z forward) into world axes, given the camera's
    world forward (optical +z) and up (opposite optical +y)."""

    forward = np.asarray(cam_forward, dtype=np.float64)
    up = np.asarray(cam_up, dtype=np.float64)
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    up = up / (np.linalg.norm(up) + 1e-9)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-9)
    return np.column_stack((right, -up, forward))


def pixel_to_world(u, v, depth, camera_matrix, cam_eye, cam_forward, cam_up):
    """Single pixel+depth -> world 3D point."""

    point_optical = backproject((u, v), depth, camera_matrix)
    rot = optical_to_world_rotation(cam_forward, cam_up)
    return np.asarray(cam_eye, dtype=np.float64) + rot @ point_optical


def project_depth_to_world(depth_image, camera_matrix, cam_eye, cam_forward,
                           cam_up, stride=6, max_range_m=6.0):
    """Vectorised projection of a (strided) depth image to an Nx3 array of
    world points, with their source (u, v) pixel coordinates (Nx2)."""

    height, width = depth_image.shape[:2]
    f_x = camera_matrix[0, 0]
    f_y = camera_matrix[1, 1]
    c_x = camera_matrix[0, 2]
    c_y = camera_matrix[1, 2]

    us = np.arange(0, width, stride)
    vs = np.arange(0, height, stride)
    grid_u, grid_v = np.meshgrid(us, vs)
    z = depth_image[grid_v, grid_u].astype(np.float64)

    valid = np.isfinite(z) & (z > 0.1) & (z <= max_range_m)
    grid_u = grid_u[valid]
    grid_v = grid_v[valid]
    z = z[valid]

    x_opt = (grid_u - c_x) * z / f_x
    y_opt = (grid_v - c_y) * z / f_y
    points_optical = np.stack((x_opt, y_opt, z), axis=1)

    rot = optical_to_world_rotation(cam_forward, cam_up)
    world = np.asarray(cam_eye, dtype=np.float64) + points_optical @ rot.T
    pixels = np.stack((grid_u, grid_v), axis=1)
    return world, pixels


def estimate_motion(object_points, image_points, camera_matrix):
    """Depth-backed PnP (EPnP-in-RANSAC, iterative fallback). Returns a
    dict with the camera-frame rotation vector rvec, translation tvec
    and inlier count, or None if the solve fails."""

    if len(object_points) < MIN_PNP_POINTS:
        return None

    distortion = np.zeros((5, 1), dtype=np.float64)

    def run_ransac(flag):
        try:
            return cv2.solvePnPRansac(
                objectPoints=object_points,
                imagePoints=image_points,
                cameraMatrix=camera_matrix,
                distCoeffs=distortion,
                iterationsCount=PNP_ITERATIONS,
                reprojectionError=PNP_REPROJECTION_ERROR,
                confidence=PNP_CONFIDENCE,
                flags=flag,
            )
        except cv2.error:
            return (False, None, None, None)

    success, rvec, tvec, inliers = run_ransac(cv2.SOLVEPNP_EPNP)
    if not success or inliers is None or len(inliers) < MIN_PNP_INLIERS:
        success, rvec, tvec, inliers = run_ransac(cv2.SOLVEPNP_ITERATIVE)

    if not success or inliers is None or len(inliers) < MIN_PNP_INLIERS:
        return None

    return {
        "rvec": rvec,
        "tvec": tvec,
        "inliers": int(len(inliers)),
    }


# ============================================================
# SINGLE-CAMERA VISUAL ODOMETRY
# ============================================================

@dataclass
class VoResult:
    valid: bool
    d_forward_m: float = 0.0      # body +X translation between frames
    d_yaw_rad: float = 0.0        # body yaw change between frames
    inliers: int = 0


class CameraVO:
    """Frame-to-frame visual odometry for one RGB-D camera.

    The camera is mounted looking along the robot body +X with body +Z
    up, so the OpenCV optical frame (x right, y down, z forward) maps to
    the body frame as: forward = +z_cam, and body yaw (about +Z) is the
    rotation about the camera's -y (up) axis. We therefore read forward
    motion from tvec[2] and yaw from -rvec[1], which are the two
    planar-motion components a ground robot can actually observe.
    """

    def __init__(self, camera_matrix, max_depth_m=8.0):
        self.camera_matrix = camera_matrix
        self.max_depth_m = max_depth_m
        self._prev_gray = None
        self._prev_kp = None
        self._prev_des = None
        self._prev_depth = None

    def update(self, bgr_image, depth_image):
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = detect_features(gray)

        result = VoResult(valid=False)

        if self._prev_des is not None and descriptors is not None:
            matches = match_features(self._prev_des, descriptors)

            object_points = []
            image_points = []
            for m in matches:
                u_prev, v_prev = self._prev_kp[m.queryIdx].pt
                depth = self._prev_depth[int(v_prev), int(u_prev)]
                if not np.isfinite(depth) or depth <= 0.1 or depth > self.max_depth_m:
                    continue
                object_points.append(
                    backproject((u_prev, v_prev), depth, self.camera_matrix)
                )
                image_points.append(keypoints[m.trainIdx].pt)

            if len(object_points) >= MIN_PNP_POINTS:
                object_points = np.array(object_points, dtype=np.float64)
                image_points = np.array(image_points, dtype=np.float64)
                motion = estimate_motion(object_points, image_points, self.camera_matrix)
                if motion is not None:
                    # PnP returns the transform taking PREVIOUS-frame points
                    # into the CURRENT camera frame; the camera's own motion
                    # is its inverse. (The old code read tvec/rvec directly,
                    # which inverted the sign of both forward motion and yaw.)
                    rot, _ = cv2.Rodrigues(motion["rvec"])
                    tvec = motion["tvec"].reshape(3)
                    rot_cam = rot.T
                    t_cam = -rot.T @ tvec
                    # Yaw about the camera's up axis (= -optical y), i.e.
                    # the robot's yaw (+ = left turn); verified against
                    # ground-truth turns in simulation.
                    yaw_up = -float(np.arctan2(rot_cam[0, 2], rot_cam[2, 2]))
                    result = VoResult(
                        valid=True,
                        d_forward_m=float(t_cam[2]),
                        d_yaw_rad=yaw_up,
                        inliers=motion["inliers"],
                    )

        self._prev_gray = gray
        self._prev_kp = keypoints
        self._prev_des = descriptors
        self._prev_depth = depth_image
        return result


# ============================================================
# STEREO (TWO-CAMERA) VISUAL ODOMETRY
# ============================================================

class StereoVisualOdometry:
    """Runs CameraVO on both RGB-D cameras and fuses them by trusting
    whichever produced more inliers this frame (a simple, robust
    winner-take-higher-confidence fusion)."""

    def __init__(self, camera_matrix, max_depth_m=8.0):
        self.left = CameraVO(camera_matrix, max_depth_m)
        self.right = CameraVO(camera_matrix, max_depth_m)

    def update(self, left_bgr, left_depth, right_bgr, right_depth):
        left_result = self.left.update(left_bgr, left_depth)
        right_result = self.right.update(right_bgr, right_depth)

        candidates = [r for r in (left_result, right_result) if r.valid]
        if not candidates:
            return VoResult(valid=False)

        return max(candidates, key=lambda r: r.inliers)


# ============================================================
# FUSED ONBOARD POSE ESTIMATOR (wheel odometry + visual correction)
# ============================================================

# Minimum VO inliers before its heading/forward correction is considered.
VO_TRUST_MIN_INLIERS = 25
# Blend gains: how strongly an ACCEPTED VO estimate pulls the wheel-only
# prediction.
VO_YAW_BLEND = 0.3
VO_FORWARD_BLEND = 0.2
# Consistency gate (robot_vision VIWO principle: vision is a measurement
# into one fused pose, never a competing pose). A VO step is only blended
# in when it agrees with the wheel step to within these bounds; anything
# else is treated as an outlier. Measured in this scene, raw 128x96 VO
# forward steps scatter by +-2..40 cm per tick, so without this gate a
# single bad frame shifted the pose by tens of centimetres.
VO_GATE_YAW_RAD = np.radians(0.6)
VO_GATE_FORWARD_M = 0.008


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


class PoseEstimator:
    """Canonical onboard pose, fusing wheel odometry (backbone) with
    visual-odometry corrections (when confident). No ground truth.

    Usage per control step:
        est.update(left_wheel_angle, right_wheel_angle, vo_result)
        pose = est.get_pose()
    """

    def __init__(self, wheel_radius_m, wheel_separation_m, start_pose: Pose2D):
        self.wheel_radius_m = wheel_radius_m
        self.wheel_separation_m = wheel_separation_m
        self.pose = Pose2D(start_pose.x, start_pose.y, start_pose.yaw)
        self._last_left = None
        self._last_right = None
        self.stats = {"vo_corrections": 0, "wheel_only_steps": 0}

    def reset_wheel_reference(self, left_wheel_angle, right_wheel_angle):
        """Call once after teleporting/placing the robot so the first
        step does not integrate a bogus wheel delta."""

        self._last_left = left_wheel_angle
        self._last_right = right_wheel_angle

    def update(self, left_wheel_angle, right_wheel_angle, vo_result: Optional[VoResult] = None,
               gyro_dyaw: Optional[float] = None):
        if self._last_left is None:
            self.reset_wheel_reference(left_wheel_angle, right_wheel_angle)
            return self.pose

        d_left = (left_wheel_angle - self._last_left) * self.wheel_radius_m
        d_right = (right_wheel_angle - self._last_right) * self.wheel_radius_m
        self._last_left = left_wheel_angle
        self._last_right = right_wheel_angle

        d_center = 0.5 * (d_left + d_right)
        d_yaw = (d_right - d_left) / self.wheel_separation_m
        if gyro_dyaw is not None:
            # IMU gyro owns the heading: wheel-differential yaw is corrupted
            # by tyre/caster scrub, which changes with the payload carried.
            d_yaw = gyro_dyaw

        # Visual correction when VO is confident.
        if (
            vo_result is not None
            and vo_result.valid
            and vo_result.inliers >= VO_TRUST_MIN_INLIERS
            and abs(vo_result.d_yaw_rad - d_yaw) < VO_GATE_YAW_RAD
            and abs(vo_result.d_forward_m - d_center) < VO_GATE_FORWARD_M
        ):
            d_yaw = (1.0 - VO_YAW_BLEND) * d_yaw + VO_YAW_BLEND * vo_result.d_yaw_rad
            d_center = (
                (1.0 - VO_FORWARD_BLEND) * d_center
                + VO_FORWARD_BLEND * vo_result.d_forward_m
            )
            self.stats["vo_corrections"] += 1
        else:
            if vo_result is not None and vo_result.valid:
                self.stats["vo_rejected"] = self.stats.get("vo_rejected", 0) + 1
            self.stats["wheel_only_steps"] += 1

        # Integrate with the midpoint heading (second-order accurate).
        mid_yaw = self.pose.yaw + 0.5 * d_yaw
        self.pose.x += d_center * np.cos(mid_yaw)
        self.pose.y += d_center * np.sin(mid_yaw)
        self.pose.yaw = cfg.clamp_angle(self.pose.yaw + d_yaw)
        return self.pose

    def get_pose(self):
        return Pose2D(self.pose.x, self.pose.y, self.pose.yaw)


# ============================================================
# STANDALONE SELF-TEST
# ============================================================

def _textured_frame(width=480, height=360, shift_x=0, seed=0):
    """Random-texture grey frame, optionally horizontally shifted, so VO
    has features to match (synthetic PyBullet floors are feature-poor)."""

    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(height, width + 200, 3), dtype=np.uint8)
    return base[:, 100 + shift_x:100 + shift_x + width, :].copy()


def main():
    print("Visual odometry + pose estimator self-test")
    print("-" * 50)

    cam = intrinsics_from_fov(480, 360, 65.0)
    vo = CameraVO(cam)

    depth = np.full((360, 480), 2.0, dtype=np.float64)

    frame0 = _textured_frame(shift_x=0, seed=1)
    r0 = vo.update(frame0, depth)
    print(f"first frame      : valid={r0.valid} (expected False)")
    assert not r0.valid

    frame1 = _textured_frame(shift_x=8, seed=1)
    r1 = vo.update(frame1, depth)
    print(f"shifted frame    : valid={r1.valid} inliers={r1.inliers}")
    assert r1.valid and r1.inliers >= MIN_PNP_INLIERS

    # Pose estimator: pure wheel odometry straight line, then a turn.
    est = PoseEstimator(
        wheel_radius_m=0.10,
        wheel_separation_m=0.56,
        start_pose=Pose2D(0.0, 0.0, 0.0),
    )
    est.reset_wheel_reference(0.0, 0.0)

    # Both wheels rotate 10 rad -> forward 1.0 m, no turn.
    est.update(10.0, 10.0, None)
    pose = est.get_pose()
    print(f"straight 1 m     : x={pose.x:.3f} y={pose.y:.3f} yaw={np.degrees(pose.yaw):.1f}")
    assert abs(pose.x - 1.0) < 1e-6 and abs(pose.y) < 1e-6

    # Right wheel faster -> left turn (yaw increases).
    est.update(10.0, 20.0, None)
    pose = est.get_pose()
    print(f"after left turn  : x={pose.x:.3f} y={pose.y:.3f} yaw={np.degrees(pose.yaw):.1f}")
    assert pose.yaw > 0.0

    print("SELF-TEST PASSED")


if __name__ == "__main__":
    main()
