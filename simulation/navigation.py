"""
INDOOR LOCALIZATION - NAVIGATION CONTROLLER (global route + local safety)
==========================================================================

Drives the AMR to a goal on the map that is being built live, following
robot_vision's stage8 NavigationController structure (FOLLOW_GLOBAL /
LOCAL_AVOID / RECOVER ladder) in compact form.

PRIORITY LADDER (evaluated top to bottom every control tick; the first
rule that applies owns the wheel command)

    1. terminal            ARRIVED / FAILED hold still
    2. goal check          inside tolerance -> align to the requested
                           final heading (if any) -> ARRIVED
    3. emergency safety    something inside the bumper zone straight
                           ahead -> RECOVER (stop, back off, replan).
                           Nothing below may override this.
    4. recovery in progress
    5. route validity      no route / route blocked on the current map /
                           robot far off the route / periodic refresh ->
                           replan with A* (lane-preferring, live map)
    6. large heading error rotate in place toward the route
    7. local avoidance     the corridor toward the route lookahead is
                           blocked by something the map has not (yet)
                           captured -> pick the best open gap (VFH) and
                           commit to it until the route corridor clears
    8. route following     pure-pursuit on the A* route
    9. lane trim           LOWEST priority: when the route is on the
                           painted lane and the centre camera sees it, a
                           small, clipped correction centres the robot on
                           the paint. It can never steer the robot into
                           an obstacle because it is only applied in
                           plain route following (rule 8) with a clear
                           corridor.

The old controller mixed these into one sum (VFH heading + lane
cross-track pulling back toward the obstacle it was avoiding), turned
VFH off entirely on the return legs, and planned only on the hard-coded
lane mask; those were the priority problems this module replaces.
"""

import math
from dataclasses import dataclass

import numpy as np

import mission_config as cfg
from astar_planner import GlobalPlanner, PlannerView, densify, lane_fraction
from vfh_local_planner import LocalPlanner, LOCAL_BLOCK_DEPTH_M, LOCAL_MAX_ANGLE_DEG

# States.
NAV_IDLE = "IDLE"
NAV_FOLLOW = "FOLLOW_ROUTE"
NAV_ROTATE = "ROTATE_TO_ROUTE"
NAV_AVOID = "LOCAL_AVOID"
NAV_RECOVER_STOP = "RECOVER_STOP"
NAV_RECOVER_BACK = "RECOVER_BACKOFF"
NAV_ALIGN = "ALIGN_FINAL_HEADING"
NAV_ARRIVED = "ARRIVED"
NAV_FAILED = "FAILED"

# Speeds / gains.
MAX_SPEED_MPS = 0.55
AVOID_SPEED_MPS = 0.28
MAX_ANGULAR_RPS = 1.2
ROTATE_ANGULAR_RPS = 0.9
K_HEADING = 1.8
LOOKAHEAD_M = 0.60            # short: hug the lane through its corners
ROTATE_ENTER_DEG = 40.0
ROTATE_EXIT_DEG = 12.0

LANE_TRIM_GAIN = 0.45            # rad/s per unit normalised cross-track
LANE_FOLLOW_K_ANGLE = 1.8        # rad/s per rad of lane angle (line follower)
LANE_FOLLOW_K_OFFSET = 1.5       # rad/s per metre of lane offset
LANE_TRIM_MAX_RPS = 0.20

# Route maintenance.
REPLAN_PERIOD_S = 2.0
REPLAN_MIN_INTERVAL_S = 0.5
REPLAN_IMPROVEMENT = 0.90        # accept a periodic re-plan only if >=10% cheaper
OFF_ROUTE_REPLAN_M = 0.7
ROUTE_CHECK_HORIZON_M = 4.0

# Avoidance commitment.
AVOID_COMMIT_MIN_M = 0.35
AVOID_CLEAR_FRAMES = 3
AVOID_SEARCH_MAX_S = 5.0

# Recovery.
RECOVER_STOP_S = 0.4
RECOVER_BACK_M = 0.30
RECOVER_BACK_SPEED_MPS = 0.15
RECOVER_BACK_MAX_S = 3.0
MAX_RECOVERIES = 8
STALL_WINDOW_S = 10.0
STALL_MIN_PROGRESS_M = 0.20


@dataclass
class NavCommand:
    linear: float
    angular: float
    state: str
    note: str = ""


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class Navigator:
    def __init__(self, occupancy):
        self.occ = occupancy
        self.global_planner = GlobalPlanner()
        self.local = LocalPlanner()
        self._view = None
        self.state = NAV_IDLE
        self.goal = None
        self.goal_heading = None
        self.tolerance = 0.25
        self.label = ""
        self.waypoints = []
        self.path = []
        self.path_idx = 0
        self.last_plan_t = -1e9
        self.last_analysis = None
        self.message = ""
        self.lane_trim_enabled = True
        self.hard_clearance = None
        self.verbose = True
        self.stats = {"replans": 0, "avoid_episodes": 0, "recoveries": 0,
                      "lane_trim_ticks": 0, "emergencies": 0}
        self._reset_goal_state(0.0)

    # ---------------------------------------------------------- goals

    def _reset_goal_state(self, now):
        self.recoveries = 0
        self._avoid_heading_world = None
        self._avoid_start_xy = None
        self._avoid_clear_frames = 0
        self._search_t0 = None
        self._recover_t0 = None
        self._recover_start_xy = None
        self._progress_ref = None
        self._progress_t = now
        self.local.reset()

    def set_goal(self, goal_xy, pose, now, heading=None, tolerance=0.25, label=""):
        self.goal = (float(goal_xy[0]), float(goal_xy[1]))
        self.goal_heading = heading
        self.tolerance = tolerance
        self.label = label
        self.state = NAV_FOLLOW
        self._reset_goal_state(now)
        self.waypoints, self.path = [], []
        ok = self.replan(pose, now, "new goal", force=True)
        if not ok:
            self.message = f"no route to {label or self.goal}"
        return ok

    def clear(self):
        self.goal = None
        self.state = NAV_IDLE
        self.waypoints, self.path = [], []

    @property
    def done(self):
        return self.state in (NAV_ARRIVED, NAV_FAILED)

    # ---------------------------------------------------------- map/route

    def set_half_width(self, half_width_m):
        """Robot footprint half-width (wider while the conveyor module, whose
        lift guides stand out beyond the chassis, is aboard)."""

        from astar_planner import PLANNER_SAFETY_MARGIN_M
        from vfh_local_planner import LOCAL_SAFETY_MARGIN_M
        self.hard_clearance = half_width_m + PLANNER_SAFETY_MARGIN_M
        self.local.half_clearance = half_width_m + LOCAL_SAFETY_MARGIN_M
        self._view = None

    def planner_view(self):
        if self._view is None or self._view.revision != self.occ.revision:
            self._view = PlannerView(self.occ, hard_clearance_m=self.hard_clearance)
        return self._view

    def _route_cost(self, view, pts):
        total = 0.0
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            r, c = cfg.world_to_grid(x1, y1)
            if not cfg.in_grid_bounds(r, c):
                return math.inf
            total += math.hypot(x1 - x0, y1 - y0) * float(view.cost[r, c])
        return total

    def replan(self, pose, now, reason, force=False):
        if not force and now - self.last_plan_t < REPLAN_MIN_INTERVAL_S:
            return bool(self.path)
        self.last_plan_t = now
        view = self.planner_view()
        result = self.global_planner.plan(view, (pose.x, pose.y), self.goal)
        if not result["success"]:
            self.message = f"A* failed ({result['reason']}) [{reason}]"
            return False

        new_path = densify(result["waypoints"], 0.10)
        if self.path and not force and reason == "periodic":
            remaining = self.path[self.path_idx:]
            old_cost = self._route_cost(view, [(pose.x, pose.y)] + remaining)
            new_cost = self._route_cost(view, [(pose.x, pose.y)] + new_path)
            if not new_cost < REPLAN_IMPROVEMENT * old_cost:
                return True        # keep the current route (no flip-flop)

        self.waypoints = result["waypoints"]
        self.path = new_path
        self.path_idx = 0
        self.stats["replans"] += 1
        self.message = (f"A* route ({reason}): {result['length_m']:.1f} m, "
                        f"{lane_fraction(view, result['waypoints'])*100:.0f}% on lane")
        if self.verbose and reason in ("new goal", "recovery", "route blocked on map", "off route"):
            pts = " ".join(f"({x:+.1f},{y:+.1f})" for x, y in result["waypoints"][:8])
            print(f"[nav] {self.message} -> {self.label}: {pts}")
        return True

    def _update_tracking(self, pose):
        """Advance the closest-point index along the route (forward search
        only) and return (lookahead point, lateral distance to route)."""

        if not self.path:
            return None, math.inf
        end = min(len(self.path), self.path_idx + 40)
        best_i, best_d = self.path_idx, math.inf
        for i in range(self.path_idx, end):
            d = math.hypot(self.path[i][0] - pose.x, self.path[i][1] - pose.y)
            if d < best_d:
                best_i, best_d = i, d
        self.path_idx = best_i
        look = self.path[-1]
        for i in range(best_i, len(self.path)):
            if math.hypot(self.path[i][0] - pose.x, self.path[i][1] - pose.y) >= LOOKAHEAD_M:
                look = self.path[i]
                break
        return look, best_d

    def _route_blocked_ahead(self, view):
        travelled = 0.0
        for i in range(self.path_idx, len(self.path) - 1):
            x, y = self.path[i]
            # Ignore the first 0.6 m: cells right next to the robot are
            # inflated by whatever it is passing, and replanning on them
            # made the route flip-flop beside obstacles.
            if view.is_blocked_world(x, y) and travelled > 0.6:
                return True
            nx, ny = self.path[i + 1]
            travelled += math.hypot(nx - x, ny - y)
            if travelled > ROUTE_CHECK_HORIZON_M:
                break
        return False

    def _rear_clear(self, pose, view):
        for d in (cfg.AMR_HALF_LENGTH_M + 0.10, cfg.AMR_HALF_LENGTH_M + 0.25):
            for lat in (-cfg.AMR_HALF_WIDTH_M, 0.0, cfg.AMR_HALF_WIDTH_M):
                x = pose.x - d * math.cos(pose.yaw) - lat * math.sin(pose.yaw)
                y = pose.y - d * math.sin(pose.yaw) + lat * math.cos(pose.yaw)
                r, c = cfg.world_to_grid(x, y)
                if not cfg.in_grid_bounds(r, c) or view.occupied[r, c]:
                    return False
        return True

    # ---------------------------------------------------------- recovery

    def _enter_recovery(self, pose, now, why):
        self.recoveries += 1
        self.stats["recoveries"] += 1
        if self.recoveries > MAX_RECOVERIES:
            self.state = NAV_FAILED
            self.message = f"gave up after {MAX_RECOVERIES} recoveries ({why})"
            return
        self.state = NAV_RECOVER_STOP
        self._recover_t0 = now
        self._recover_start_xy = (pose.x, pose.y)
        self.message = f"recovery #{self.recoveries}: {why}"
        self.local.reset()

    # ---------------------------------------------------------- main step

    def step(self, pose, obstacles, lane, now, lane_fit=None):
        """One control tick. `obstacles` = (N,2) body-frame obstacle points
        from both RGB-D cameras; `lane` = lane_detector.LaneDetection from
        the centre camera (or None). Returns a NavCommand."""

        # 1. terminal / idle
        if self.goal is None or self.state == NAV_IDLE:
            return NavCommand(0.0, 0.0, NAV_IDLE)
        if self.state in (NAV_ARRIVED, NAV_FAILED):
            return NavCommand(0.0, 0.0, self.state, self.message)

        view = self.planner_view()
        dist_goal = math.hypot(self.goal[0] - pose.x, self.goal[1] - pose.y)

        # 2. goal check (+ final heading alignment)
        if self.state == NAV_ALIGN or dist_goal <= self.tolerance:
            if self.goal_heading is None:
                self.state = NAV_ARRIVED
                self.message = f"arrived at {self.label}"
                return NavCommand(0.0, 0.0, NAV_ARRIVED, self.message)
            err = wrap(self.goal_heading - pose.yaw)
            if abs(err) < math.radians(2.5):
                self.state = NAV_ARRIVED
                self.message = f"arrived at {self.label} (aligned)"
                return NavCommand(0.0, 0.0, NAV_ARRIVED, self.message)
            self.state = NAV_ALIGN
            w = float(np.clip(1.5 * err, -ROTATE_ANGULAR_RPS, ROTATE_ANGULAR_RPS))
            if abs(w) < 0.12:
                w = math.copysign(0.12, err)
            return NavCommand(0.0, w, NAV_ALIGN)

        look, off_route = self._update_tracking(pose)
        bearing_deg = 0.0
        if look is not None:
            bearing_deg = math.degrees(wrap(math.atan2(look[1] - pose.y, look[0] - pose.x) - pose.yaw))

        need = min(LOCAL_BLOCK_DEPTH_M, dist_goal + cfg.AMR_HALF_LENGTH_M + 0.08)
        analysis = self.local.analyse(obstacles, pose, bearing_deg, view, need)
        self.last_analysis = analysis

        # 3. emergency safety
        if analysis.emergency and self.state in (NAV_FOLLOW, NAV_AVOID):
            self.stats["emergencies"] += 1
            self._enter_recovery(pose, now, f"obstacle {analysis.front_clearance_m:.2f} m ahead of bumper")
            return NavCommand(0.0, 0.0, "EMERGENCY_STOP", self.message)

        # 4. recovery in progress
        if self.state == NAV_RECOVER_STOP:
            if now - self._recover_t0 >= RECOVER_STOP_S:
                self.state = NAV_RECOVER_BACK if self._rear_clear(pose, view) else NAV_FOLLOW
                self._recover_t0 = now
                self._recover_start_xy = (pose.x, pose.y)
                if self.state == NAV_FOLLOW:
                    self.replan(pose, now, "recovery", force=True)
            return NavCommand(0.0, 0.0, NAV_RECOVER_STOP, self.message)
        if self.state == NAV_RECOVER_BACK:
            backed = math.hypot(pose.x - self._recover_start_xy[0], pose.y - self._recover_start_xy[1])
            if (backed >= RECOVER_BACK_M or now - self._recover_t0 > RECOVER_BACK_MAX_S
                    or not self._rear_clear(pose, view)):
                self.state = NAV_FOLLOW
                self.replan(pose, now, "recovery", force=True)
                self._progress_ref, self._progress_t = None, now
            return NavCommand(-RECOVER_BACK_SPEED_MPS, 0.0, NAV_RECOVER_BACK, self.message)

        # Stall watchdog (no progress toward the goal for a while).
        if self._progress_ref is None or dist_goal < self._progress_ref - STALL_MIN_PROGRESS_M:
            self._progress_ref, self._progress_t = dist_goal, now
        elif now - self._progress_t > STALL_WINDOW_S:
            self._progress_ref, self._progress_t = dist_goal, now
            self._enter_recovery(pose, now, "no progress toward goal")
            return NavCommand(0.0, 0.0, NAV_RECOVER_STOP, self.message)

        # 5. route validity
        if not self.path:
            if not self.replan(pose, now, "no route"):
                self._enter_recovery(pose, now, self.message)
                return NavCommand(0.0, 0.0, NAV_RECOVER_STOP, self.message)
        elif self._route_blocked_ahead(view):
            self.replan(pose, now, "route blocked on map")
        elif off_route > OFF_ROUTE_REPLAN_M:
            self.replan(pose, now, "off route")
        elif now - self.last_plan_t > REPLAN_PERIOD_S:
            self.replan(pose, now, "periodic")

        look, off_route = self._update_tracking(pose)
        if look is None:
            return NavCommand(0.0, 0.0, self.state, self.message)
        bearing = wrap(math.atan2(look[1] - pose.y, look[0] - pose.x) - pose.yaw)
        bearing_deg = math.degrees(bearing)
        if abs(bearing_deg - analysis.target_bearing_deg) > 1.0:
            analysis = self.local.analyse(obstacles, pose, bearing_deg, view, need)
            self.last_analysis = analysis

        # 6. large heading error -> rotate in place
        if self.state == NAV_ROTATE:
            if abs(bearing_deg) <= ROTATE_EXIT_DEG:
                self.state = NAV_FOLLOW
            else:
                w = math.copysign(min(ROTATE_ANGULAR_RPS, max(0.25, 1.5 * abs(bearing))), bearing)
                return NavCommand(0.0, w, NAV_ROTATE)
        elif abs(bearing_deg) > ROTATE_ENTER_DEG and self.state != NAV_AVOID:
            self.state = NAV_ROTATE
            w = math.copysign(ROTATE_ANGULAR_RPS, bearing)
            return NavCommand(0.0, w, NAV_ROTATE)

        # 7. local avoidance
        if self.state == NAV_FOLLOW and analysis.target_blocked:
            self.state = NAV_AVOID
            self.stats["avoid_episodes"] += 1
            self._avoid_heading_world = None
            self._avoid_clear_frames = 0
            self.local.reset()
            self.replan(pose, now, "corridor blocked", force=False)

        if self.state == NAV_AVOID:
            if not analysis.target_blocked:
                self._avoid_clear_frames += 1
            else:
                self._avoid_clear_frames = 0
            committed = 0.0
            if self._avoid_start_xy is not None:
                committed = math.hypot(pose.x - self._avoid_start_xy[0], pose.y - self._avoid_start_xy[1])
            if (self._avoid_clear_frames >= AVOID_CLEAR_FRAMES
                    and (committed >= AVOID_COMMIT_MIN_M or self._avoid_heading_world is None)):
                self.state = NAV_FOLLOW
                self.local.reset()
            else:
                if now - self.last_plan_t > 1.0:
                    self.replan(pose, now, "avoid refresh", force=False)
                heading = self.local.select_heading(analysis)
                if heading is None:
                    if self._search_t0 is None:
                        self._search_t0 = now
                    elif now - self._search_t0 > AVOID_SEARCH_MAX_S:
                        self._search_t0 = None
                        self._enter_recovery(pose, now, "no open gap toward the route")
                        return NavCommand(0.0, 0.0, NAV_RECOVER_STOP, self.message)
                    # No usable opening in view: turn toward the deeper
                    # side to look for one, or recover if we keep failing.
                    depth = analysis.depth_per_angle
                    left, right = depth[depth.size // 2:].mean(), depth[:depth.size // 2].mean()
                    w = 0.5 if left >= right else -0.5
                    if self._avoid_heading_world is None:
                        self._avoid_heading_world = pose.yaw
                        self._avoid_start_xy = (pose.x, pose.y)
                    return NavCommand(0.0, w, NAV_AVOID, "searching for a gap")
                self._search_t0 = None
                if self._avoid_heading_world is None:
                    self._avoid_start_xy = (pose.x, pose.y)
                self._avoid_heading_world = pose.yaw + math.radians(heading)
                err = math.radians(heading)
                w = float(np.clip(K_HEADING * err, -MAX_ANGULAR_RPS, MAX_ANGULAR_RPS))
                v = AVOID_SPEED_MPS * max(0.3, 1.0 - abs(heading) / LOCAL_MAX_ANGLE_DEG * 0.6)
                v *= float(np.clip((analysis.front_clearance_m - 0.10) / 0.6, 0.3, 1.0))
                return NavCommand(v, w, NAV_AVOID, f"gap {heading:+.0f} deg")

        # 8. route following (pure pursuit on the lookahead)
        w = K_HEADING * bearing
        v = MAX_SPEED_MPS
        v *= float(np.clip(1.0 - abs(bearing_deg) / 70.0, 0.35, 1.0))
        v *= float(np.clip((analysis.front_clearance_m - 0.10) / 1.0, 0.25, 1.0))
        v *= float(np.clip(dist_goal / 0.9, 0.35, 1.0))

        # 9. lane tracking: while the route runs straight along mapped lane,
        # steer by the PAINT itself (camera centreline fit, metres) - a line
        # follower - and let the route only decide where to turn.
        r, c = cfg.world_to_grid(pose.x, pose.y)
        lr, lc = cfg.world_to_grid(*look)
        on_lane_route = (cfg.in_grid_bounds(r, c) and cfg.in_grid_bounds(lr, lc)
                         and view.lane[r, c] and view.lane[lr, lc])
        if on_lane_route and abs(bearing_deg) < 20.0:
            if lane_fit is not None and abs(lane_fit[0]) < 0.45:
                a, b = lane_fit
                w = float(np.clip(LANE_FOLLOW_K_ANGLE * math.atan(b) + LANE_FOLLOW_K_OFFSET * a,
                                  -0.6, 0.6))
                self.stats["lane_trim_ticks"] += 1
            elif lane is not None and lane.detected and abs(lane.cross_track_norm) < 0.6:
                w += float(np.clip(-LANE_TRIM_GAIN * lane.cross_track_norm,
                                   -LANE_TRIM_MAX_RPS, LANE_TRIM_MAX_RPS))
                self.stats["lane_trim_ticks"] += 1

        w = float(np.clip(w, -MAX_ANGULAR_RPS, MAX_ANGULAR_RPS))
        return NavCommand(v, w, NAV_FOLLOW, self.message)
