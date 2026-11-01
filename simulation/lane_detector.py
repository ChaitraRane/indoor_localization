"""
INDOOR LOCALIZATION - LANE DETECTION (centre RGB camera)
==========================================================

The AMR's third camera (the centre monocular RGB camera, rgb_camera_link
in amr_robot.urdf) is used purely to see the painted floor lane - the
same role the RGB camera played for pavement/lane sensing in the
robot_vision outdoor stack, repurposed here for an indoor painted lane.

Pipeline (classic, no learning needed for a solid-colour painted lane):

    BGR frame
      -> restrict to a floor region of interest (lower part of the image)
      -> HSV threshold for the bright-yellow lane paint
      -> split the ROI into a NEAR band and a FAR band
      -> lane centroid column in each band
      -> cross-track error  (near band centroid vs image centre)
         heading error      (far centroid vs near centroid direction)

The two errors are what a lane-follow controller needs: cross-track to
stay centred on the lane, heading to anticipate curves/turns. Junction
disambiguation (which branch to take at a T) is NOT done here - that is
a navigation-level concern and lives in mission_controller.py, which
biases toward the goal direction. This module only reports what the
camera sees.

This file is part of the indoor_localization project only; nothing here
reads, imports, or depends on anything in the robot_vision folder.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import mission_config as cfg


# ============================================================
# REGION OF INTEREST
# ============================================================
# The forward-facing RGB camera sees the floor in the lower part of the
# frame. The ROI keeps the bottom ROI_TOP_FRACTION..1.0 rows, where the
# lane in front of the robot appears, and ignores the upper image (racks,
# walls, horizon) which can contain other coloured objects.

ROI_TOP_FRACTION = 0.45

# NEAR band is the bottom NEAR_BAND_FRACTION of the ROI (closest floor,
# used for cross-track); the remaining upper ROI is the FAR band (used
# with the near centroid to estimate heading).
NEAR_BAND_FRACTION = 0.4

# Minimum fraction of ROI pixels that must be lane-coloured for a
# detection to be trusted.
MIN_LANE_PIXEL_FRACTION = 0.002


@dataclass
class LaneDetection:
    """Result of one lane-detection call."""

    detected: bool
    # Cross-track error, normalised to [-1, 1]. Positive => the lane
    # centre is to the RIGHT of the image centre (robot sits left of the
    # lane and should steer right to recentre).
    cross_track_norm: float = 0.0
    # Heading error (radians). Positive => the lane bends to the right
    # ahead of the robot.
    heading_error_rad: float = 0.0
    # Fraction of ROI pixels that matched the lane colour (detection
    # strength / confidence proxy).
    lane_pixel_fraction: float = 0.0
    # Binary lane mask over the FULL frame (uint8 0/255), for overlay.
    mask: Optional[np.ndarray] = None


# ============================================================
# DETECTION
# ============================================================

def _band_centroid_column(mask_roi, row_start, row_end):
    """Mean column index of lane pixels in mask_roi[row_start:row_end],
    or None if that band has no lane pixels."""

    band = mask_roi[row_start:row_end, :]
    columns = np.where(band > 0)[1]
    if columns.size == 0:
        return None
    return float(np.mean(columns))


def detect_lane(bgr_image, hsv_lower=None, hsv_upper=None):
    """
    Detects the painted lane in a BGR camera frame.

    Returns a LaneDetection. When no lane is visible, detected=False and
    the error fields are 0 (a controller should treat this as "lane
    lost" and fall back to its last command / a recovery behaviour,
    exactly as the robot_vision stack did on lane loss).
    """

    hsv_lower = hsv_lower if hsv_lower is not None else cfg.LANE_HSV_LOWER
    hsv_upper = hsv_upper if hsv_upper is not None else cfg.LANE_HSV_UPPER

    height, width = bgr_image.shape[:2]

    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    full_mask = cv2.inRange(
        hsv, np.array(hsv_lower, dtype=np.uint8), np.array(hsv_upper, dtype=np.uint8)
    )

    # Clean up speckle so stray yellow pixels do not pull the centroid.
    kernel = np.ones((5, 5), np.uint8)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)

    roi_top = int(height * ROI_TOP_FRACTION)
    mask_roi = full_mask[roi_top:, :]
    roi_height = mask_roi.shape[0]
    roi_pixels = roi_height * width

    lane_pixels = int(np.count_nonzero(mask_roi))
    lane_fraction = lane_pixels / float(roi_pixels) if roi_pixels else 0.0

    if lane_fraction < MIN_LANE_PIXEL_FRACTION:
        return LaneDetection(detected=False, mask=full_mask)

    # NEAR band = bottom of ROI, FAR band = top of ROI.
    near_start = int(roi_height * (1.0 - NEAR_BAND_FRACTION))
    near_centroid = _band_centroid_column(mask_roi, near_start, roi_height)
    far_centroid = _band_centroid_column(mask_roi, 0, near_start)

    if near_centroid is None:
        # Lane only visible far away; use the far band for cross-track too.
        near_centroid = far_centroid

    image_centre = width / 2.0
    cross_track_norm = (near_centroid - image_centre) / image_centre
    cross_track_norm = float(np.clip(cross_track_norm, -1.0, 1.0))

    heading_error = 0.0
    if far_centroid is not None and near_centroid is not None:
        # dx positive => lane is further right up ahead; dy is the
        # forward (upward in image) separation between the band centres.
        dx = far_centroid - near_centroid
        dy = max(1.0, (roi_height - near_start) / 2.0 + near_start / 2.0)
        heading_error = float(np.arctan2(dx, dy))

    return LaneDetection(
        detected=True,
        cross_track_norm=cross_track_norm,
        heading_error_rad=heading_error,
        lane_pixel_fraction=lane_fraction,
        mask=full_mask,
    )


# ============================================================
# OVERLAY (debug / preview)
# ============================================================

def draw_overlay(bgr_image, detection: LaneDetection):
    """Returns a copy of bgr_image with the lane mask tinted and the
    cross-track / heading errors printed, for the camera preview windows."""

    overlay = bgr_image.copy()

    if detection.mask is not None:
        tint = np.zeros_like(overlay)
        tint[detection.mask > 0] = (0, 255, 255)
        overlay = cv2.addWeighted(overlay, 1.0, tint, 0.4, 0.0)

    status = "LANE OK" if detection.detected else "LANE LOST"
    cv2.putText(
        overlay,
        f"{status} xtrack={detection.cross_track_norm:+.2f} "
        f"hdg={np.degrees(detection.heading_error_rad):+.1f}deg",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0) if detection.detected else (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


# ============================================================
# STANDALONE SELF-TEST (synthetic frames, no simulator needed)
# ============================================================

def _synthetic_lane_frame(width=480, height=360, lane_centre_col=240, slant=0):
    """Builds a grey floor frame with a vertical-ish yellow lane stripe
    whose centre column is lane_centre_col at the bottom and shifts by
    `slant` pixels toward the top (to fake a curve)."""

    frame = np.full((height, width, 3), 200, dtype=np.uint8)  # grey floor (BGR)
    lane_bgr = (0, 217, 255)  # bright yellow in BGR
    half_w = 30
    for row in range(height):
        # interpolate lane centre from bottom (lane_centre_col) to top
        # (lane_centre_col + slant)
        t = (height - 1 - row) / (height - 1)
        centre = int(lane_centre_col + slant * t)
        frame[row, max(0, centre - half_w):min(width, centre + half_w)] = lane_bgr
    return frame


def main():
    print("Lane detector self-test (synthetic frames)")
    print("-" * 50)

    # 1. Centred straight lane -> ~zero cross-track and heading.
    frame = _synthetic_lane_frame(lane_centre_col=240, slant=0)
    det = detect_lane(frame)
    print(f"centred straight : detected={det.detected} "
          f"xtrack={det.cross_track_norm:+.3f} hdg={np.degrees(det.heading_error_rad):+.1f}deg")
    assert det.detected
    assert abs(det.cross_track_norm) < 0.1

    # 2. Lane shifted right -> positive cross-track.
    frame = _synthetic_lane_frame(lane_centre_col=330, slant=0)
    det = detect_lane(frame)
    print(f"shifted right    : detected={det.detected} "
          f"xtrack={det.cross_track_norm:+.3f}")
    assert det.cross_track_norm > 0.2

    # 3. Lane curving right ahead -> positive heading error.
    frame = _synthetic_lane_frame(lane_centre_col=240, slant=120)
    det = detect_lane(frame)
    print(f"curving right    : detected={det.detected} "
          f"hdg={np.degrees(det.heading_error_rad):+.1f}deg")
    assert det.heading_error_rad > 0.05

    # 4. No lane -> not detected.
    frame = np.full((360, 480, 3), 200, dtype=np.uint8)
    det = detect_lane(frame)
    print(f"no lane          : detected={det.detected}")
    assert not det.detected

    print("SELF-TEST PASSED")


if __name__ == "__main__":
    main()
