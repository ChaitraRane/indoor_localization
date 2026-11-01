"""
INDOOR LOCALIZATION - BOX DETECTION & PAYLOAD-MODULE DECISION
===============================================================

At the detection pose short of point A the AMR looks at the pick table
with its forward RGB-D camera, recognises the boxes, estimates how big
and how many they are, and decides WHICH payload module it has to fetch
and carry on its back to service them.

Two passes run together:

  1. YOLO (ultralytics, yolo11s.pt) - the semantic pass. Genuinely run on
     the RGB frame and reported; COCO has no "cardboard box" class, so in
     this scene it is informational, exactly as before.

  2. RGB-D blob geometry - the authoritative pass. Cardboard-coloured
     blobs in the RGB image (same segmentation the arm's stereo camera
     uses) are lifted to 3D with the aligned depth image and the
     ESTIMATED robot pose; each blob within a gate straight ahead of the
     robot is a box candidate whose lateral extent and height give its
     size. The gate excludes the cardboard-coloured goods on the racks
     either side of the aisle.

Payload decision: every box small enough for a suction cup (footprint <=
cfg.SUCTION_MAX_BOX_FOOTPRINT_M) -> the SUCTION_ARM module (collected
from the arm-module stand); anything larger would need the conveyor
module, which this mission does not use.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

import mission_config as cfg
import visual_odometry as vo

SCRIPT_DIR = Path(__file__).resolve().parent
YOLO_MODEL_PATH = SCRIPT_DIR / "yolo11s.pt"
YOLO_CONFIDENCE = 0.25

GATE_FORWARD_MIN_M = 0.3
GATE_FORWARD_MAX_M = 3.2
GATE_LATERAL_M = 0.85
MIN_BLOB_AREA_PX = 30
SINGLE_BOX_ESTIMATE_M = 0.18


def cardboard_mask(bgr):
    """Cardboard-brown segmentation (shared with the arm stereo camera)."""

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((8, 100, 50), np.uint8), np.array((17, 215, 255), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


@dataclass
class DetectedBox:
    world_x: float
    world_y: float
    top_z: float
    bottom_z: float
    footprint_m: float
    estimated_box_count: int
    point_count: int


@dataclass
class BoxDetectionResult:
    count: int
    boxes: List[DetectedBox] = field(default_factory=list)
    yolo_labels: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def max_footprint_m(self):
        return max((b.footprint_m for b in self.boxes), default=0.0)

    @property
    def support_top_z(self):
        """Estimated surface the boxes stand on (median box bottom)."""

        if not self.boxes:
            return None
        return float(np.median([b.bottom_z for b in self.boxes]))

    @property
    def centroid(self):
        if not self.boxes:
            return None
        w = np.array([b.estimated_box_count for b in self.boxes], dtype=np.float64)
        x = np.array([b.world_x for b in self.boxes])
        y = np.array([b.world_y for b in self.boxes])
        return float(np.sum(w * x) / w.sum()), float(np.sum(w * y) / w.sum())


@dataclass
class PayloadDecision:
    module: str            # "SUCTION_ARM", "CONVEYOR" or "NONE"
    box_count: int
    max_footprint_m: float
    reason: str

    @property
    def arm_required(self):
        return self.module == "SUCTION_ARM"


class BoxDetector:
    def __init__(self, use_yolo=True):
        self._yolo = None
        if use_yolo:
            try:
                from ultralytics import YOLO
                self._yolo = YOLO(str(YOLO_MODEL_PATH))
            except Exception as exc:  # noqa: BLE001 - best effort, never fatal
                print(f"[box_detector] YOLO unavailable ({exc}); using RGB-D geometry only.")

    def _run_yolo(self, bgr):
        if self._yolo is None:
            return []
        try:
            labels = []
            for r in self._yolo.predict(bgr, conf=YOLO_CONFIDENCE, verbose=False):
                for box in r.boxes:
                    labels.append((r.names.get(int(box.cls[0]), "?"), float(box.conf[0])))
            return labels
        except Exception as exc:  # noqa: BLE001
            print(f"[box_detector] YOLO inference error: {exc}")
            return []

    def detect(self, bgr, depth, intrinsics, cam_eye, cam_forward, cam_up, robot_pose,
               run_yolo=True):
        yolo = self._run_yolo(bgr) if run_yolo else []
        mask = cardboard_mask(bgr)
        world, pixels = vo.project_depth_to_world(depth, intrinsics, cam_eye, cam_forward, cam_up,
                                                  stride=1, max_range_m=GATE_FORWARD_MAX_M + 0.5)
        if world.shape[0] == 0:
            return BoxDetectionResult(0, [], yolo)
        pix_world = np.full(depth.shape + (3,), np.nan)
        pix_world[pixels[:, 1], pixels[:, 0]] = world

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        c, s = np.cos(robot_pose.yaw), np.sin(robot_pose.yaw)
        boxes = []
        for i in range(1, n):
            if stats[i][4] < MIN_BLOB_AREA_PX:
                continue
            pts = pix_world[labels == i]
            pts = pts[np.isfinite(pts[:, 0])]
            if pts.shape[0] < 10:
                continue
            dx, dy = pts[:, 0] - robot_pose.x, pts[:, 1] - robot_pose.y
            fwd, lat = dx * c + dy * s, -dx * s + dy * c
            gate = ((fwd > GATE_FORWARD_MIN_M) & (fwd < GATE_FORWARD_MAX_M)
                    & (np.abs(lat) < GATE_LATERAL_M) & (pts[:, 2] > 0.05))
            pts, lat = pts[gate], lat[gate]
            if pts.shape[0] < 10:
                continue
            width = float(np.percentile(lat, 97) - np.percentile(lat, 3))
            top = float(np.percentile(pts[:, 2], 97))
            bottom = float(np.percentile(pts[:, 2], 3))
            height = top - bottom
            # One colour blob is one box unless it is much wider than it is
            # tall (several similar small boxes merged side by side); a
            # single big carton stays one big box.
            if width > 1.8 * height:
                count = max(1, int(round(width / SINGLE_BOX_ESTIMATE_M)))
            else:
                count = 1
            per_box = max(width / count, height)
            boxes.append(DetectedBox(
                world_x=float(np.median(pts[:, 0])), world_y=float(np.median(pts[:, 1])),
                top_z=top, bottom_z=bottom, footprint_m=per_box,
                estimated_box_count=count, point_count=int(pts.shape[0]),
            ))
        return BoxDetectionResult(sum(b.estimated_box_count for b in boxes), boxes, yolo)

    @staticmethod
    def decide_payload(result: BoxDetectionResult):
        if result.count < cfg.SUCTION_MIN_BOX_COUNT:
            return PayloadDecision("NONE", 0, 0.0, "no boxes detected at the pick table")
        biggest = result.max_footprint_m
        if biggest <= cfg.SUCTION_MAX_BOX_FOOTPRINT_M:
            return PayloadDecision(
                "SUCTION_ARM", result.count, biggest,
                f"{result.count} small box(es), largest {biggest:.2f} m <= "
                f"{cfg.SUCTION_MAX_BOX_FOOTPRINT_M:.2f} m: fetch the SUCTION-ARM module",
            )
        return PayloadDecision(
            "CONVEYOR", result.count, biggest,
            f"box footprint {biggest:.2f} m too large for suction: conveyor module needed",
        )


# ============================================================
# STANDALONE SELF-TEST (synthetic frame, no simulator/YOLO needed)
# ============================================================

def main():
    from visual_odometry import Pose2D

    print("Box detector self-test (synthetic RGB-D)")
    print("-" * 50)
    det = BoxDetector(use_yolo=False)
    w, h = 160, 120
    intr = vo.intrinsics_from_fov(w, h, 65.0)
    bgr = np.full((h, w, 3), 120, np.uint8)
    depth = np.full((h, w), 6.0)
    # Two cardboard patches ~1.5 m ahead, around camera height.
    for c0 in (55, 95):
        bgr[50:70, c0:c0 + 14] = (64, 133, 204)
        depth[50:70, c0:c0 + 14] = 1.5
    res = det.detect(bgr, depth, intr, [0.0, 0.0, 0.45], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                     Pose2D(-0.3, 0.0, 0.0), run_yolo=False)
    print(f"boxes={res.count} footprints={[round(b.footprint_m, 2) for b in res.boxes]}")
    assert res.count == 2
    d = BoxDetector.decide_payload(res)
    print(f"decision: {d.module} - {d.reason}")
    assert d.module == "SUCTION_ARM"
    print("SELF-TEST PASSED")


if __name__ == "__main__":
    main()
