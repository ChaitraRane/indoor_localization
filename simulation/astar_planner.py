"""
INDOOR LOCALIZATION - LANE-PREFERRING GLOBAL PLANNER (A* on the live map)
==========================================================================

Global planning over the occupancy grid built live from the RGB-D
cameras, following robot_vision's stage8_stabilized_navigation design
(PlannerView + GlobalPlanner), re-expressed in this project's x/y frame:

PlannerView (one per map revision)
    * hard feasibility comes from a metric CLEARANCE field
      (cv2.distanceTransform to the nearest occupied cell), not a coarse
      binary dilation: a cell is blocked only if its measured clearance
      is below the robot half-width + safety margin, so a corridor that
      genuinely fits the robot is never erased by rounding;
    * unknown cells next to obstacles are blocked (cannot squeeze into
      unseen space beside something solid), other unknown cells are
      traversable but cost more;
    * soft costs: less clearance than preferred costs more (keeps routes
      centred in aisles), unknown costs more, and - the task rule -
      cells OFF the painted lane cost LANE_OFF_COST_MULTIPLIER times more
      than lane cells. A* therefore follows the defined lane whenever it
      is clear, and leaves it only where an obstacle (or a goal that is
      not on the lane) makes that cheaper.

GlobalPlanner
    * 8-connected A* with corner-cutting prevention over the cost grid;
    * any-angle line-of-sight smoothing; a shortcut whose two ends are on
      the lane must stay on the lane, so smoothing never cuts corners of
      the lane network;
    * start/goal are snapped to the nearest traversable cell.
"""

import heapq
import math

import cv2
import numpy as np

import mission_config as cfg

SQRT2 = math.sqrt(2.0)
NEIGHBOURS_8 = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2),
)

# Hard clearance: AMR half-width + safety margin (robot_vision uses
# half-width + 0.09 m; the AMR is wider so 0.06 m keeps the 1.25 m
# detour gaps comfortably open while staying well clear of contact).
PLANNER_SAFETY_MARGIN_M = 0.06
PLANNER_HARD_CLEARANCE_M = cfg.AMR_HALF_WIDTH_M + PLANNER_SAFETY_MARGIN_M

PLANNER_BASE_COST = 1.0
PLANNER_CLEARANCE_PREFERRED_M = 0.80
PLANNER_CLEARANCE_WEIGHT = 0.8
PLANNER_UNKNOWN_COST_MULTIPLIER = 1.8
PLANNER_UNKNOWN_OBSTACLE_GUARD_CELLS = 2
# Lane priority. LANE cells (paint the robot has SEEN) cost 1; cells whose
# lane state is still unknown cost LANE_UNKNOWN_COST_MULTIPLIER; floor that
# has been seen and is NOT lane costs LANE_OFF_COST_MULTIPLIER. With these
# ratios A* follows mapped lane wherever it leads, explores along the lane
# into the unknown, and only crosses bare floor when the lane is blocked
# (an obstacle on it) or the goal itself is off the lane.
LANE_UNKNOWN_COST_MULTIPLIER = 60.0
LANE_OFF_COST_MULTIPLIER = 60.0
PLANNER_SNAP_RADIUS_CELLS = 15


class PlannerView:
    """Derived planning representation of one occupancy-map revision."""

    def __init__(self, occupancy_map, extra_blocked=None, hard_clearance_m=None):
        _prob, occupied, free, unknown = occupancy_map.masks()

        # Distance from each cell centre to the nearest occupied cell
        # centre; the obstacle surface can be up to half a cell closer.
        distance_cells = cv2.distanceTransform((~occupied).astype(np.uint8), cv2.DIST_L2, 5)
        clearance = distance_cells * cfg.GRID_RESOLUTION_M - 0.5 * cfg.GRID_RESOLUTION_M
        clearance = np.maximum(clearance, 0.0)

        hard = PLANNER_HARD_CLEARANCE_M if hard_clearance_m is None else hard_clearance_m
        blocked = clearance < hard

        guard = PLANNER_UNKNOWN_OBSTACLE_GUARD_CELLS
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * guard + 1, 2 * guard + 1))
        near_blocked = cv2.dilate(blocked.astype(np.uint8), kernel) > 0
        lane = occupancy_map.lane_mask
        non_lane = occupancy_map.non_lane_mask
        blocked |= unknown & near_blocked & ~lane

        if extra_blocked is not None:
            blocked |= extra_blocked

        blocked[0, :] = blocked[-1, :] = True
        blocked[:, 0] = blocked[:, -1] = True

        cost = np.full(occupied.shape, PLANNER_BASE_COST, dtype=np.float32)
        shortfall = np.clip(1.0 - clearance / PLANNER_CLEARANCE_PREFERRED_M, 0.0, 1.0)
        cost *= 1.0 + PLANNER_CLEARANCE_WEIGHT * shortfall
        cost[unknown] *= PLANNER_UNKNOWN_COST_MULTIPLIER
        cost[~lane & ~non_lane] *= LANE_UNKNOWN_COST_MULTIPLIER
        cost[non_lane] *= LANE_OFF_COST_MULTIPLIER
        cost[blocked] = np.inf

        self.occupied = occupied
        self.free = free
        self.unknown = unknown
        self.clearance_m = clearance
        self.blocked = blocked
        self.traversable = ~blocked
        self.cost = cost
        self.lane = lane
        self.non_lane = non_lane
        self.revision = occupancy_map.revision

    def is_blocked_world(self, x, y):
        row, col = cfg.world_to_grid(x, y)
        if not cfg.in_grid_bounds(row, col):
            return True
        return bool(self.blocked[row, col])


def bresenham(r0, c0, r1, c1):
    cells = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if (r, c) == (r1, c1):
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells


class GlobalPlanner:
    """Cost-weighted A* over a PlannerView plus line-of-sight smoothing."""

    def __init__(self):
        self.plans = 0
        self.failures = 0
        self.last_reason = ""

    @staticmethod
    def snap(view, cell, radius_cells=PLANNER_SNAP_RADIUS_CELLS):
        row, col = cell
        if cfg.in_grid_bounds(row, col) and view.traversable[row, col]:
            return (row, col)
        best = None
        best_d = None
        for radius in range(1, radius_cells + 1):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if max(abs(dr), abs(dc)) != radius:
                        continue
                    r, c = row + dr, col + dc
                    if cfg.in_grid_bounds(r, c) and view.traversable[r, c]:
                        d = dr * dr + dc * dc
                        if best is None or d < best_d:
                            best, best_d = (r, c), d
            if best is not None:
                return best
        return None

    @staticmethod
    def line_cost(view, a, b):
        """Cost of driving the straight segment a->b over the cost grid
        (inf if it crosses a blocked cell or leaves the lane while both of
        its ends are on the lane)."""

        cells = bresenham(a[0], a[1], b[0], b[1])
        both_on_lane = bool(view.lane[a] and view.lane[b])
        step = math.hypot(b[0] - a[0], b[1] - a[1]) / max(1, len(cells) - 1)
        total = 0.0
        for r, c in cells[1:]:
            if view.blocked[r, c] or (both_on_lane and not view.lane[r, c]):
                return math.inf
            total += step * float(view.cost[r, c])
        return total

    @staticmethod
    def smooth(view, path):
        """Any-angle shortcutting that is never allowed to make the route
        MORE expensive than the A* cells it replaces (within 3%). Plain
        line-of-sight smoothing let a shortcut from an off-lane start skip a
        whole lane corner across open floor - exactly the 'not following
        the lane' behaviour this planner exists to prevent."""

        if len(path) <= 2:
            return list(path)
        cum = [0.0]
        for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
            cum.append(cum[-1] + math.hypot(r1 - r0, c1 - c0) * float(view.cost[r1, c1]))
        smoothed = [path[0]]
        anchor = 0
        while anchor < len(path) - 1:
            furthest = anchor + 1
            limit = min(len(path) - 1, anchor + 60)
            for candidate in range(limit, anchor + 1, -1):
                budget = (cum[candidate] - cum[anchor]) * 1.03 + 0.5
                if GlobalPlanner.line_cost(view, path[anchor], path[candidate]) <= budget:
                    furthest = candidate
                    break
            smoothed.append(path[furthest])
            anchor = furthest
        return smoothed

    def plan(self, view, start_xy, goal_xy):
        """Returns a dict: success, waypoints (world, smoothed), cells
        (raw A* cells), length_m, reason."""

        self.plans += 1
        start = self.snap(view, cfg.world_to_grid(*start_xy))
        goal = self.snap(view, cfg.world_to_grid(*goal_xy))
        if start is None or goal is None:
            self.failures += 1
            self.last_reason = "start_not_traversable" if start is None else "goal_not_traversable"
            return {"success": False, "reason": self.last_reason}
        if start == goal:
            wp = [cfg.grid_to_world(*start)]
            return {"success": True, "waypoints": wp, "cells": [start], "length_m": 0.0,
                    "reason": "trivial"}

        cost = view.cost
        blocked = view.blocked
        rows, cols = cost.shape
        g = np.full(cost.shape, np.inf, dtype=np.float64)
        came = np.full(cost.shape + (2,), -1, dtype=np.int32)
        closed = np.zeros(cost.shape, dtype=bool)
        gr, gc = goal

        def h(r, c):
            dr, dc = abs(r - gr), abs(c - gc)
            return (dr + dc) + (SQRT2 - 2.0) * min(dr, dc)

        g[start] = 0.0
        heap = [(h(*start), 0.0, start)]
        found = False
        while heap:
            _, cg, cur = heapq.heappop(heap)
            if closed[cur]:
                continue
            closed[cur] = True
            if cur == goal:
                found = True
                break
            r, c = cur
            for dr, dc, step in NEIGHBOURS_8:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols or closed[nr, nc]:
                    continue
                cell_cost = cost[nr, nc]
                if not np.isfinite(cell_cost):
                    continue
                if dr != 0 and dc != 0 and (blocked[r, nc] or blocked[nr, c]):
                    continue
                ng = cg + step * float(cell_cost)
                if ng < g[nr, nc]:
                    g[nr, nc] = ng
                    came[nr, nc] = (r, c)
                    heapq.heappush(heap, (ng + h(nr, nc), ng, (nr, nc)))

        if not found:
            self.failures += 1
            self.last_reason = "no_route"
            return {"success": False, "reason": "no_route"}

        cells = [goal]
        node = goal
        while node != start:
            pr, pc = came[node]
            node = (int(pr), int(pc))
            cells.append(node)
        cells.reverse()

        smoothed = self.smooth(view, cells)
        waypoints = [cfg.grid_to_world(r, c) for r, c in smoothed]
        # Keep the true start/goal points at the ends when they are
        # themselves traversable (the snapped cell centre can be 5 cm off).
        if not view.is_blocked_world(*goal_xy):
            waypoints[-1] = (float(goal_xy[0]), float(goal_xy[1]))
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                     for a, b in zip(waypoints[:-1], waypoints[1:]))
        self.last_reason = "ok"
        return {"success": True, "waypoints": waypoints, "cells": cells,
                "length_m": length, "reason": "ok"}


def densify(waypoints, spacing_m=0.10):
    """Resample a polyline at roughly uniform spacing (for tracking)."""

    if not waypoints:
        return []
    dense = [tuple(waypoints[0])]
    for (x0, y0), (x1, y1) in zip(waypoints[:-1], waypoints[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.ceil(seg / spacing_m)))
        for i in range(1, n + 1):
            t = i / n
            dense.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return dense


def lane_fraction(view, waypoints):
    """Fraction of a path's (densified) length that lies on the lane."""

    dense = densify(waypoints, 0.1)
    if not dense:
        return 0.0
    on = 0
    for x, y in dense:
        r, c = cfg.world_to_grid(x, y)
        if cfg.in_grid_bounds(r, c) and view.lane[r, c]:
            on += 1
    return on / len(dense)


# ============================================================
# STANDALONE SELF-TEST
# ============================================================

def main():
    print("A* lane-preferring planner self-test")
    print("-" * 50)

    from occupancy_map import OccupancyMap

    occ = OccupancyMap()
    planner = GlobalPlanner()
    # Test fixture only: pretend the camera has already mapped the whole
    # painted network and the bare floor beside it.
    paint = np.zeros((cfg.GRID_ROWS, cfg.GRID_COLS), dtype=bool)
    hw = int(round(cfg.LANE_WIDTH_M / 2 / cfg.GRID_RESOLUTION_M))
    for poly in cfg.LANE_NETWORK:
        for (x0, y0), (x1, y1) in zip(poly[:-1], poly[1:]):
            n = int(max(abs(x1 - x0), abs(y1 - y0)) / 0.05) + 1
            for t in np.linspace(0, 1, n):
                r, c = cfg.world_to_grid(x0 + t * (x1 - x0), y0 + t * (y1 - y0))
                paint[r - hw:r + hw + 1, c - hw:c + hw + 1] = True
    occ.lane_hits[paint] = 5
    occ.floor_hits[~paint] = 5
    occ.lane_revision += 1

    view = PlannerView(occ)
    res = planner.plan(view, cfg.ROBOT_START_XY, cfg.DETECTION_POSE_XY)
    assert res["success"], res
    frac = lane_fraction(view, res["waypoints"])
    print(f"spawn -> detection pose: {len(res['waypoints'])} waypoints, "
          f"{res['length_m']:.1f} m, on-lane {frac*100:.0f}%")
    assert frac > 0.9, "with an empty map A* must follow the lane"

    # From an OFF-lane start (e.g. just after a detour) the route must go
    # back to the lane and follow it, not cut across the open floor.
    res = planner.plan(view, (-5.7, -1.25), cfg.DETECTION_POSE_XY)
    frac = lane_fraction(view, res["waypoints"])
    print(f"off-lane start -> detection pose: on-lane {frac*100:.0f}%, "
          f"first waypoints {[(round(x, 2), round(y, 2)) for x, y in res['waypoints'][:3]]}")
    assert frac > 0.75, "must rejoin and follow the lane"

    # Put a mapped crate on the aisle lane; A* must leave the lane round
    # it and come back.
    ox, oy, hx, hy = 0.8, -0.45, 0.30, 0.30
    r0, c0 = cfg.world_to_grid(ox - hx, oy - hy)
    r1, c1 = cfg.world_to_grid(ox + hx, oy + hy)
    occ.logodds[r0:r1 + 1, c0:c1 + 1] = cfg.LOGODDS_MAX
    # And the racks either side of the aisle.
    for y_lo, y_hi in ((-2.0, -1.1), (1.1, 2.0)):
        ra, ca = cfg.world_to_grid(-2.0, y_lo)
        rb, cb = cfg.world_to_grid(6.7, y_hi)
        occ.logodds[ra:rb + 1, ca:cb + 1] = cfg.LOGODDS_MAX
    view = PlannerView(occ)
    res = planner.plan(view, (-1.0, 0.0), (3.0, 0.0))
    assert res["success"], res
    ys = [y for _x, y in densify(res["waypoints"])]
    print(f"detour round crate: max |y| on path {max(abs(v) for v in ys):.2f} m, "
          f"waypoints {[(round(x, 2), round(y, 2)) for x, y in res['waypoints']]}")
    assert max(ys) > 0.2, "path must swing left (north) of the crate"
    assert min(ys) > -0.2, "path must not squeeze through the 0.35 m gap on the right"

    print("SELF-TEST PASSED")


if __name__ == "__main__":
    main()
