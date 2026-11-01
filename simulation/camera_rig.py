"""
INDOOR LOCALIZATION - CAMERA RIG GEOMETRY
===========================================

Shared camera mounting geometry for the AMR's three cameras, matching
amr_robot.urdf's sensor_mast + per-camera joint origins:

    left  RGB-D stereo : body offset ~ (0.31, +0.09, 0.31), horizontal
    right RGB-D stereo : body offset ~ (0.31, -0.09, 0.31), horizontal
    centre RGB (lane)  : body offset ~ (0.31,  0.00, 0.35), pitched down

Both the simulator harness and the mission controller use this module,
but for DIFFERENT poses, which is the crux of honest no-ground-truth
operation:

    * autonomous_mission.py renders each camera from the robot's TRUE
      physical pose (a real camera is wherever the robot really is).
    * mission_controller.py projects the returned depth into the maps
      using the robot's ESTIMATED (odometry) pose.

If the estimate is good the map is sharp; if it drifts the map smears -
exactly real-world behaviour. Ground truth never enters localisation or
mapping.

This file is part of the indoor_localization project only; nothing here
reads, imports, or depends on anything in the robot_vision folder.
"""

import math

import numpy as np

import visual_odometry as vo


# ------------------------------------------------------------
# Image sizes.
#
# PERCEPTION_SCALE sizes used by the perception/mapping pipeline itself
# (visual odometry, box detection, VFH) - kept modest so TinyRenderer
# stays fast enough to run the full mission many times over for
# development/testing. These feed numerical algorithms, not a human
# viewer, so low resolution does not hurt accuracy.
#
# DISPLAY_SCALE sizes are what the harness renders for an actual GUI/
# video viewer (this is what was looking "very distorted" - at 128x96 a
# straight rack post is only a few pixels wide and reads as jagged/warped
# once displayed or recorded). The same intrinsics' FOV is kept, just
# with far more pixels, so geometry is identical and only sharpness
# changes.
# ------------------------------------------------------------
RGBD_WIDTH = 128
RGBD_HEIGHT = 96
CENTER_WIDTH = 160
CENTER_HEIGHT = 120

DISPLAY_RGBD_WIDTH = 480
DISPLAY_RGBD_HEIGHT = 360
DISPLAY_CENTER_WIDTH = 560
DISPLAY_CENTER_HEIGHT = 420

CAMERA_FOV_DEG = 65.0
CAMERA_NEAR = 0.02
CAMERA_FAR = 12.0

# Robot base-link height off the floor when standing on its wheels (used
# to place cameras from the planar estimated pose). The wheels (r=0.10)
# sit 0.05 below base_link, so it settles at 0.15 m; the old 0.18 lifted
# every projected depth point by 3 cm.
CAMERA_BASE_Z = 0.15

# Body-frame mount offsets (x forward, y left, z up) relative to base_link,
# and per-camera downward pitch.
MOUNTS = {
    "left":   {"offset": (0.31, 0.09, 0.31), "pitch_down": 0.0},
    "right":  {"offset": (0.31, -0.09, 0.31), "pitch_down": 0.0},
    "center": {"offset": (0.31, 0.0, 0.35), "pitch_down": math.radians(28.0)},
}

RGBD_INTRINSICS = vo.intrinsics_from_fov(RGBD_WIDTH, RGBD_HEIGHT, CAMERA_FOV_DEG)
CENTER_INTRINSICS = vo.intrinsics_from_fov(CENTER_WIDTH, CENTER_HEIGHT, CAMERA_FOV_DEG)

# Same FOV, display resolution - for the GUI/video renders only.
DISPLAY_RGBD_INTRINSICS = vo.intrinsics_from_fov(
    DISPLAY_RGBD_WIDTH, DISPLAY_RGBD_HEIGHT, CAMERA_FOV_DEG
)
DISPLAY_CENTER_INTRINSICS = vo.intrinsics_from_fov(
    DISPLAY_CENTER_WIDTH, DISPLAY_CENTER_HEIGHT, CAMERA_FOV_DEG
)


def build_camera_frame(x, y, yaw, which, base_z=CAMERA_BASE_Z):
    """World-frame (eye, forward, up) for camera `which`, given a planar
    robot pose (x, y, yaw). `which` is 'left', 'right' or 'center'."""

    mount = MOUNTS[which]
    ox, oy, oz = mount["offset"]
    pitch = mount["pitch_down"]

    cos_y, sin_y = math.cos(yaw), math.sin(yaw)

    eye = np.array([
        x + ox * cos_y - oy * sin_y,
        y + ox * sin_y + oy * cos_y,
        base_z + oz,
    ], dtype=np.float64)

    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    forward = np.array([cos_y * cos_p, sin_y * cos_p, -sin_p], dtype=np.float64)
    right = np.array([sin_y, -cos_y, 0.0], dtype=np.float64)
    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-9)

    return eye, forward, up


def frame_from_true_base(base_pos, base_quat, which):
    """World (eye, forward, up) from the TRUE base pose (position +
    quaternion), used by the harness to render from physical reality.
    Extracts yaw from the quaternion and reuses build_camera_frame with
    the true base height."""

    # Yaw from quaternion (x, y, z, w).
    qx, qy, qz, qw = base_quat
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return build_camera_frame(base_pos[0], base_pos[1], yaw, which, base_z=base_pos[2])
