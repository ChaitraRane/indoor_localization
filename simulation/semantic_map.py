"""
INDOOR LOCALIZATION - SEMANTIC MAP (map #2 of 2)
==================================================

The semantic counterpart to the metric occupancy grid: a sparse record
of what the robot has recognised and WHERE, in world coordinates. It
holds two things:

    1. Landmarks: fused object detections (class + world position +
       observation count + size estimate), clustered so repeated
       sightings of the same physical object collapse into one entry.
       Mirrors the robot_vision static-landmark fusion (distance-gated
       association + running mean), but without scipy/DBSCAN - plain
       nearest-neighbour association is enough here.

    2. Named points: specifically point A, recorded once the robot has
       sensed the box pile there, so A has a persistent map location
       "for future reference" (the B point will be added the same way).

This file is part of the indoor_localization project only; nothing here
reads, imports, or depends on anything in the robot_vision folder.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# Two detections of the same class within this distance (m) are treated
# as the same physical landmark and fused. Kept small so that distinct
# small boxes sitting ~0.3 m apart in the pile at A are not merged, while
# repeated sightings of the SAME box (with pose jitter) still collapse.
LANDMARK_ASSOCIATION_DISTANCE_M = 0.25


@dataclass
class Landmark:
    class_name: str
    x: float
    y: float
    observations: int = 1
    size_m: float = 0.0          # running-mean estimated footprint

    def fuse(self, x, y, size_m):
        """Running-mean update of position and size with a new sighting."""

        n = self.observations
        self.x = (self.x * n + x) / (n + 1)
        self.y = (self.y * n + y) / (n + 1)
        self.size_m = (self.size_m * n + size_m) / (n + 1)
        self.observations += 1


@dataclass
class NamedPoint:
    name: str
    x: float
    y: float
    note: str = ""


class SemanticMap:
    """Sparse semantic map: fused landmarks + named points (point A, B)."""

    def __init__(self):
        self.landmarks: List[Landmark] = []
        self.named_points: Dict[str, NamedPoint] = {}

    # --------------------------------------------------------- landmarks

    def add_detection(self, class_name, world_x, world_y, size_m=0.0):
        """Fuse a detection into the nearest same-class landmark within
        the association radius, else create a new landmark. Returns the
        landmark it fell into."""

        best = None
        best_dist = LANDMARK_ASSOCIATION_DISTANCE_M
        for lm in self.landmarks:
            if lm.class_name != class_name:
                continue
            dist = np.hypot(lm.x - world_x, lm.y - world_y)
            if dist < best_dist:
                best = lm
                best_dist = dist

        if best is not None:
            best.fuse(world_x, world_y, size_m)
            return best

        lm = Landmark(class_name=class_name, x=world_x, y=world_y, size_m=size_m)
        self.landmarks.append(lm)
        return lm

    def landmarks_of_class(self, class_name):
        return [lm for lm in self.landmarks if lm.class_name == class_name]

    def confirmed_landmarks(self, min_observations=2):
        return [lm for lm in self.landmarks if lm.observations >= min_observations]

    # ------------------------------------------------------ named points

    def record_point(self, name, x, y, note=""):
        """Store (or overwrite) a named world point - e.g. point A, and
        later point B - so it has a persistent map location."""

        self.named_points[name] = NamedPoint(name=name, x=x, y=y, note=note)
        return self.named_points[name]

    def record_point_a(self, x, y, note="box-pick target"):
        return self.record_point("A", x, y, note)

    def get_point(self, name) -> Optional[NamedPoint]:
        return self.named_points.get(name)

    # ------------------------------------------------------------ summary

    def summary(self):
        lines = ["Semantic map:"]
        lines.append(f"  landmarks: {len(self.landmarks)}")
        for lm in self.landmarks:
            lines.append(
                f"    - {lm.class_name:12s} at ({lm.x:+.2f},{lm.y:+.2f}) "
                f"obs={lm.observations} size={lm.size_m:.2f} m"
            )
        lines.append(f"  named points: {list(self.named_points)}")
        for name, pt in self.named_points.items():
            lines.append(f"    - {name}: ({pt.x:+.2f},{pt.y:+.2f}) {pt.note}")
        return "\n".join(lines)


# ============================================================
# STANDALONE SELF-TEST
# ============================================================

def main():
    print("Semantic map self-test")
    print("-" * 50)

    smap = SemanticMap()

    # Two nearby box detections fuse into one landmark.
    smap.add_detection("box", 4.0, 0.0, size_m=0.24)
    smap.add_detection("box", 4.05, 0.02, size_m=0.25)
    assert len(smap.landmarks_of_class("box")) == 1
    fused = smap.landmarks_of_class("box")[0]
    print(f"fused box: ({fused.x:+.3f},{fused.y:+.3f}) obs={fused.observations}")
    assert fused.observations == 2

    # A far detection creates a second landmark.
    smap.add_detection("box", 4.0, 0.30, size_m=0.24)
    assert len(smap.landmarks_of_class("box")) == 2

    # Record point A.
    smap.record_point_a(4.0, 0.0)
    a = smap.get_point("A")
    print(f"point A recorded at ({a.x:+.2f},{a.y:+.2f})")
    assert a is not None and abs(a.x - 4.0) < 1e-6

    print()
    print(smap.summary())
    print()
    print("SELF-TEST PASSED")


if __name__ == "__main__":
    main()
