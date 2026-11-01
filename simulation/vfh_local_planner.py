"""
INDOOR LOCALIZATION - LOCAL OBSTACLE AVOIDANCE (VFH+ / follow-the-gap)
=======================================================================

Depth-based local planner following robot_vision's
stage8_stabilized_navigation LocalPlanner, in this project's x/y frame.

Obstacle points come from BOTH forward RGB-D cameras (the old version
used the left camera only, so the robot was half-blind on its right),
projected into the robot body frame (fwd, lat) using the estimated pose.

Corridor test (exact swept-corridor collision condition for driving
straight along candidate heading theta with the real robot width):

    along = fwd*cos(theta) + lat*sin(theta)      in (0, LOOKAHEAD]
    |perp| = |-fwd*sin(theta) + lat*cos(theta)|  < half_clearance
    depth(theta) = min(along) over such points   (LOOKAHEAD if none)

Decisions, in priority order:
    emergency      an obstacle inside the bumper zone straight ahead
    target_blocked the corridor the robot actually WANTS (toward the
                   A* lookahead) is shallower than LOCAL_BLOCK_DEPTH_M.
                   An obstacle elsewhere in view is explicitly NOT a
                   reason to override the global route (the old version
                   blocked on anything within 2.2 m in any direction,
                   which made the AMR crawl through every aisle).
    gap selection  among contiguous open headings, minimise
                   goal-deviation + missing-depth - gap-width bonus
                   + side-switch penalty, reject headings whose short
                   rollout crosses a blocked map cell, prefer rollouts
                   that stay on the painted lane; keep the previous
                   choice unless a new one is clearly better (hysteresis
                   - no left/right flapping).
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

import mission_config as cfg
import visual_odometry as vo


# Candidate heading fan (degrees). The forward cameras' horizontal FOV is
# ~80 deg, so headings beyond ~+-36 deg cannot be verified by depth;
# sharper turns are done by rotating in place first.
LOCAL_MAX_ANGLE_DEG = 36.0
LOCAL_ANGLE_STEP_DEG = 3.0

LOCAL_LOOKAHEAD_M = 2.4
LOCAL_SAFETY_MARGIN_M = 0.07
LOCAL_REQUIRED_HALF_CLEARANCE_M = cfg.AMR_HALF_WIDTH_M + LOCAL_SAFETY_MARGIN_M
LOCAL_BLOCK_DEPTH_M = 1.20       # from robot centre (~0.85 m ahead of bumper)
LOCAL_GAP_MIN_DEPTH_M = 1.30

# Emergency: obstacle this close ahead of the front bumper, within the
# robot's width band.
EMERGENCY_BUMPER_GAP_M = 0.15

OBSTACLE_MIN_HEIGHT_M = 0.10
OBSTACLE_MAX_HEIGHT_M = 1.45

LOCAL_W_GOAL = 1.0          # per degree away from the route direction
LOCAL_W_DEPTH = 9.0         # per metre of missing corridor depth
LOCAL_W_WIDTH = 0.25        # per degree of gap width (bonus)
LOCAL_W_SIDE_SWITCH = 30.0  # changing avoidance side
LOCAL_W_OFF_LANE = 40.0     # per unit of rollout NOT on the lane (lane has priority)
LOCAL_GAP_SWITCH_MIN_IMPROVEMENT = 8.0
LOCAL_GAP_PERSIST_MATCH_DEG = 4.0

ROLLOUT_HORIZON_M = 1.4
ROLLOUT_STEP_M = 0.15


@dataclass
class LocalAnalysis:
    angles_deg: np.ndarray
    depth_per_angle: np.ndarray
    target_bearing_deg: float
    target_depth_m: float
    target_blocked: bool
    emergency: bool
    front_clearance_m: float         # free distance ahead of the bumper (robot width band)
    obstacle_count: int
    map_valid: np.ndarray = None
    lane_coverage: np.ndarray = None
    gaps: List[dict] = field(default_factory=list)


class LocalPlanner:
    def __init__(self):
        half = int(math.floor(LOCAL_MAX_ANGLE_DEG / LOCAL_ANGLE_STEP_DEG))
        self.angles_deg = np.arange(-half, half + 1, dtype=np.float64) * LOCAL_ANGLE_STEP_DEG
        self.angles_rad = np.radians(self.angles_deg)
        self._sin = np.sin(self.angles_rad)
        self._cos = np.cos(self.angles_rad)
        self.half_clearance = LOCAL_REQUIRED_HALF_CLEARANCE_M   # widened when a wide module is aboard
        self.selected_angle_deg: Optional[float] = None
        self.selected_side = 0
        self.gap_switches = 0

    # ---------------------------------------------------------- sensing

    @staticmethod
    def obstacles_body_frame(depth_frames, robot_pose):
        """depth_frames: list of (depth, intrinsics, eye, forward, up).
        Returns an (N, 2) array of (fwd, lat) obstacle points relative to
        the robot centre."""

        pts = []
        for depth, intr, eye, fwd, up in depth_frames:
            world, _ = vo.project_depth_to_world(
                depth, intr, eye, fwd, up, stride=4, max_range_m=LOCAL_LOOKAHEAD_M + 1.0,
            )
            if world.shape[0]:
                h = world[:, 2]
                pts.append(world[(h >= OBSTACLE_MIN_HEIGHT_M) & (h <= OBSTACLE_MAX_HEIGHT_M)])
        if not pts:
            return np.empty((0, 2))
        world = np.concatenate(pts, axis=0)
        if world.shape[0] == 0:
            return np.empty((0, 2))
        dx = world[:, 0] - robot_pose.x
        dy = world[:, 1] - robot_pose.y
        c, s = math.cos(robot_pose.yaw), math.sin(robot_pose.yaw)
        return np.stack((dx * c + dy * s, -dx * s + dy * c), axis=1)

    # ---------------------------------------------------------- analysis

    def analyse(self, obstacles, robot_pose, target_bearing_deg, planner_view=None,
                required_depth_m=LOCAL_BLOCK_DEPTH_M):
        """required_depth_m: how much clear corridor the route actually
        needs - less than LOCAL_BLOCK_DEPTH_M when the goal itself is
        closer (a goal parked in front of a table/stand is not 'blocked'
        by the table beyond it)."""

        n = len(self.angles_deg)
        depth = np.full(n, LOCAL_LOOKAHEAD_M, dtype=np.float64)
        if obstacles.shape[0]:
            fwd = obstacles[:, 0][None, :]
            lat = obstacles[:, 1][None, :]
            along = fwd * self._cos[:, None] + lat * self._sin[:, None]
            perp = np.abs(-fwd * self._sin[:, None] + lat * self._cos[:, None])
            inside = (along > 0.0) & (along <= LOCAL_LOOKAHEAD_M) & (perp < self.half_clearance)
            nearest = np.where(inside, along, np.inf).min(axis=1)
            depth = np.where(np.isfinite(nearest), nearest, LOCAL_LOOKAHEAD_M)

        # Bumper-zone check straight ahead (robot-width band).
        front_clear = LOCAL_LOOKAHEAD_M
        if obstacles.shape[0]:
            band = (np.abs(obstacles[:, 1]) < cfg.AMR_HALF_WIDTH_M + 0.03) & (obstacles[:, 0] > 0.0)
            if np.any(band):
                front_clear = float(obstacles[band, 0].min()) - cfg.AMR_HALF_LENGTH_M
        emergency = front_clear <= EMERGENCY_BUMPER_GAP_M

        target = float(np.clip(target_bearing_deg, -LOCAL_MAX_ANGLE_DEG, LOCAL_MAX_ANGLE_DEG))
        target_idx = int(np.argmin(np.abs(self.angles_deg - target)))
        target_depth = float(depth[target_idx])

        map_valid = np.ones(n, dtype=bool)
        lane_cov = np.zeros(n, dtype=np.float64)
        if planner_view is not None:
            for i, a in enumerate(self.angles_rad):
                heading = robot_pose.yaw + a
                horizon = min(ROLLOUT_HORIZON_M, depth[i])
                d_samples = np.arange(0.3, max(0.3, horizon) + 1e-6, ROLLOUT_STEP_M)
                on_lane = 0
                for d in d_samples:
                    x = robot_pose.x + d * math.cos(heading)
                    y = robot_pose.y + d * math.sin(heading)
                    r, c = cfg.world_to_grid(x, y)
                    if not cfg.in_grid_bounds(r, c) or planner_view.occupied[r, c]:
                        map_valid[i] = False
                        break
                    if planner_view.lane[r, c]:
                        on_lane += 1
                lane_cov[i] = on_lane / max(1, len(d_samples))

        target_blocked = target_depth < required_depth_m or not map_valid[target_idx]

        analysis = LocalAnalysis(
            angles_deg=self.angles_deg, depth_per_angle=depth,
            target_bearing_deg=target, target_depth_m=target_depth,
            target_blocked=bool(target_blocked), emergency=bool(emergency),
            front_clearance_m=front_clear, obstacle_count=int(obstacles.shape[0]),
            map_valid=map_valid, lane_coverage=lane_cov,
        )
        analysis.gaps = self._find_gaps(depth)
        return analysis

    def _find_gaps(self, depth):
        usable = depth >= LOCAL_GAP_MIN_DEPTH_M
        gaps = []
        i, n = 0, len(usable)
        while i < n:
            if not usable[i]:
                i += 1
                continue
            start = i
            while i + 1 < n and usable[i + 1]:
                i += 1
            gaps.append({
                "start": start, "end": i,
                "width_deg": float(self.angles_deg[i] - self.angles_deg[start] + LOCAL_ANGLE_STEP_DEG),
                "centre_deg": float(0.5 * (self.angles_deg[start] + self.angles_deg[i])),
            })
            i += 1
        return gaps

    # ---------------------------------------------------------- choice

    def select_heading(self, analysis: LocalAnalysis):
        """Best safe heading (deg, body frame) toward the target, or None."""

        target = analysis.target_bearing_deg
        best, best_cost = None, None
        candidates = []
        for gap in analysis.gaps:
            # Keep away from the gap edges: a heading right at the edge
            # of an opening clips the obstacle that bounds it.
            lo, hi = gap["start"], gap["end"]
            if hi - lo >= 2:
                lo, hi = lo + 1, hi - 1
            for i in range(lo, hi + 1):
                if not analysis.map_valid[i]:
                    continue
                a = float(self.angles_deg[i])
                cost = (LOCAL_W_GOAL * abs(a - target)
                        + LOCAL_W_DEPTH * (LOCAL_LOOKAHEAD_M - analysis.depth_per_angle[i])
                        - LOCAL_W_WIDTH * gap["width_deg"]
                        + LOCAL_W_OFF_LANE * (1.0 - analysis.lane_coverage[i]))
                side = 0 if abs(a) < 1e-6 else (1 if a > 0 else -1)
                if self.selected_side and side and side != self.selected_side:
                    cost += LOCAL_W_SIDE_SWITCH
                candidates.append((cost, a))
                if best_cost is None or cost < best_cost:
                    best, best_cost = a, cost

        if best is not None and self.selected_angle_deg is not None:
            near = [c for c in candidates
                    if abs(c[1] - self.selected_angle_deg) <= LOCAL_GAP_PERSIST_MATCH_DEG]
            if near:
                prev_cost, prev_a = min(near)
                if prev_cost - best_cost < LOCAL_GAP_SWITCH_MIN_IMPROVEMENT:
                    best = prev_a
                else:
                    self.gap_switches += 1

        if best is not None:
            self.selected_angle_deg = best
            if abs(best) > 1e-6:
                self.selected_side = 1 if best > 0 else -1
        return best

    def reset(self):
        self.selected_angle_deg = None
        self.selected_side = 0


# ============================================================
# STANDALONE SELF-TEST
# ============================================================

def main():
    from visual_odometry import Pose2D

    print("Local planner self-test (synthetic obstacle points)")
    print("-" * 50)
    lp = LocalPlanner()
    pose = Pose2D(0.0, 0.0, 0.0)

    a = lp.analyse(np.empty((0, 2)), pose, 0.0)
    assert not a.emergency and not a.target_blocked
    print(f"open space : blocked={a.target_blocked} heading={lp.select_heading(a)}")

    # Racks either side of a 2.2 m aisle must NOT block a straight route.
    xs = np.arange(0.5, 3.0, 0.05)
    racks = np.concatenate([np.stack((xs, np.full_like(xs, 1.1)), 1),
                            np.stack((xs, np.full_like(xs, -1.1)), 1)])
    a = lp.analyse(racks, pose, 0.0)
    print(f"aisle      : blocked={a.target_blocked} depth0={a.target_depth_m:.2f}")
    assert not a.target_blocked

    # A crate on the route, offset right: must steer LEFT (positive).
    ys = np.arange(-0.75, -0.14, 0.03)
    crate = np.stack((np.full_like(ys, 1.0), ys), 1)
    lp.reset()
    a = lp.analyse(np.concatenate([racks, crate]), pose, 0.0)
    h = lp.select_heading(a)
    print(f"crate right: blocked={a.target_blocked} -> heading {h:+.1f} deg")
    assert a.target_blocked and h is not None and h > 0

    wall = np.stack((np.full(20, 0.45), np.linspace(-0.2, 0.2, 20)), 1)
    a = lp.analyse(wall, pose, 0.0)
    print(f"bumper     : emergency={a.emergency} front_clear={a.front_clearance_m:.2f}")
    assert a.emergency
    print("SELF-TEST PASSED")


if __name__ == "__main__":
    main()
