"""
INDOOR LOCALIZATION - OCCUPANCY MAP (map #1 of 2)
===================================================

A log-odds occupancy grid built live from the two RGB-D cameras' depth
images as the robot drives - the "map not given, build it as you go"
requirement. This is the metric map the global planner (A*) and the
local planner's map check consume.

Integration follows robot_vision's stage8_stabilized_navigation
OccupancyMap.integrate (OctoMap-style scan integration), re-expressed in
this project's world x/y frame and fully vectorised:

    * a whole RGB-D frame is ONE measurement: every cell is updated at
      most once per frame, so a dense patch of pixels can never push a
      cell many times in one go (the old per-pixel loop did, which
      over-weighted near obstacles and was ~50x slower);
    * within a frame an occupied endpoint overrides free ray evidence;
    * ray cells before an obstacle endpoint are free, a floor endpoint
      is itself free;
    * an endpoint that is neither floor nor obstacle (above the obstacle
      band) certifies the line of sight but gets no label of its own;
    * a cell needs at least MIN_OCCUPIED_VOTES_PER_FRAME endpoints in
      one frame to receive an occupied vote (rejects single-pixel noise);
    * clamped log-odds gives persistence - nothing is reset to unknown
      just because it is out of view.

LANE LAYER - also built live. Nothing about the painted lane is given to
the robot: every frame, the centre RGB camera's lane mask is projected
onto the floor (known camera mount -> per-pixel floor look-up table) and
voted into the grid, the same way robot_vision's pavement layer was built
from its camera. A cell becomes LANE once it has been seen as paint more
often than as bare floor; floor seen without paint becomes NOT-LANE;
everything else is LANE-UNKNOWN. The planner (astar_planner.py) makes
LANE cheap, LANE-UNKNOWN expensive and NOT-LANE very expensive, so the
robot keeps to the paint it has found and leaves it only when it must.
"""

import numpy as np

import mission_config as cfg
import visual_odometry as vo


# Height bands (metres above the floor, world z).
FLOOR_BAND_M = 0.05             # |z| <= this -> floor endpoint (free)
OBSTACLE_MIN_HEIGHT_M = 0.10    # lane paint / floor markers sit at ~0.01
OBSTACLE_MAX_HEIGHT_M = 1.45    # anything higher clears the AMR + arm module

# Range gating (planar distance from the camera).
MIN_RANGE_M = 0.15
MAX_RANGE_M = 6.0

DEPTH_PIXEL_STRIDE = 3
MIN_OCCUPIED_VOTES_PER_FRAME = 2


class OccupancyMap:
    """Log-odds occupancy grid over the warehouse footprint."""

    def __init__(self):
        shape = (cfg.GRID_ROWS, cfg.GRID_COLS)
        self.logodds = np.zeros(shape, dtype=np.float32)
        self.free_hits = np.zeros(shape, dtype=np.uint16)
        self.occupied_hits = np.zeros(shape, dtype=np.uint16)
        self.lane_hits = np.zeros(shape, dtype=np.uint16)
        self.floor_hits = np.zeros(shape, dtype=np.uint16)
        self._lane_cache = None
        self._lane_cache_rev = -1
        self._visited_lane = np.zeros(shape, dtype=bool)
        self.revision = 0
        self.lane_revision = 0
        self.integrations = 0
        if cfg.LANE_MAP_DEFINED:
            self.seed_defined_lanes()

    # ------------------------------------------------- live lane layer

    def seed_defined_lanes(self):
        """Load the DEFINED lane layout (cfg.LANE_NETWORK, the painted
        lane centrelines) into the lane layer: paint cells start as LANE,
        everything else as NOT-LANE. Obstacles are NOT given - the occupancy
        layer still starts empty and is built live; the camera keeps voting
        into this layer on top of the defined layout."""

        paint = np.zeros((cfg.GRID_ROWS, cfg.GRID_COLS), dtype=bool)
        hw = int(round(cfg.LANE_WIDTH_M / 2 / cfg.GRID_RESOLUTION_M))
        for poly in cfg.LANE_NETWORK:
            for (x0, y0), (x1, y1) in zip(poly[:-1], poly[1:]):
                n = int(max(abs(x1 - x0), abs(y1 - y0)) / (cfg.GRID_RESOLUTION_M / 2)) + 1
                for t in np.linspace(0.0, 1.0, n):
                    r, c = cfg.world_to_grid(x0 + t * (x1 - x0), y0 + t * (y1 - y0))
                    paint[max(r - hw, 0):r + hw + 1, max(c - hw, 0):c + hw + 1] = True
        self.lane_hits[paint] = 20
        self.floor_hits[~paint] = 20
        self.lane_revision += 1
        self.revision += 1

    def integrate_lane(self, lane_mask_img, floor_lut, pose, max_range_m=3.2, stride=2):
        """Vote one centre-camera frame into the lane layer. floor_lut is
        the (H, W, 2) body-frame floor point of every pixel (nan above the
        horizon); lane_mask_img the binary lane mask of the same frame."""

        if lane_mask_img is None:
            return
        lut = floor_lut[::stride, ::stride]
        lane = lane_mask_img[::stride, ::stride] > 0
        fwd, lat = lut[..., 0], lut[..., 1]
        ok = np.isfinite(fwd) & (fwd > 0.5) & (np.hypot(fwd, lat) < max_range_m)
        if not np.any(ok):
            return
        c, s = np.cos(pose.yaw), np.sin(pose.yaw)
        wx = pose.x + fwd[ok] * c - lat[ok] * s
        wy = pose.y + fwd[ok] * s + lat[ok] * c
        cols = ((wx - cfg.WORLD_X_MIN) / cfg.GRID_RESOLUTION_M).astype(np.int64)
        rows = ((wy - cfg.WORLD_Y_MIN) / cfg.GRID_RESOLUTION_M).astype(np.int64)
        inside = (rows >= 0) & (rows < cfg.GRID_ROWS) & (cols >= 0) & (cols < cfg.GRID_COLS)
        flat = rows[inside] * cfg.GRID_COLS + cols[inside]
        is_lane = lane[ok][inside]
        lane_flat = np.unique(flat[is_lane])
        floor_flat = np.setdiff1d(np.unique(flat[~is_lane]), lane_flat, assume_unique=True)
        lh, fh = self.lane_hits.reshape(-1), self.floor_hits.reshape(-1)
        lh[lane_flat] = np.minimum(lh[lane_flat].astype(np.int32) + 1, 65000).astype(np.uint16)
        fh[floor_flat] = np.minimum(fh[floor_flat].astype(np.int32) + 1, 65000).astype(np.uint16)
        self.lane_revision += 1
        self.revision += 1

    def _lane_layers(self):
        if self._lane_cache is None or self._lane_cache_rev != self.lane_revision:
            import cv2
            lh = self.lane_hits.astype(np.float32)
            fh = self.floor_hits.astype(np.float32)
            lane = (lh >= 1) & (lh >= fh)          # paint is unmistakable: one clear sighting
            # close 1-cell holes in the painted strip
            lane = cv2.morphologyEx(lane.astype(np.uint8), cv2.MORPH_CLOSE,
                                    np.ones((3, 3), np.uint8)) > 0
            non_lane = (fh >= 2) & (lh < 0.25 * fh) & ~lane
            self._lane_cache = (lane, non_lane)
            self._lane_cache_rev = self.lane_revision
        return self._lane_cache

    @property
    def lane_mask(self):
        """Cells mapped as painted lane (built live from the camera)."""

        return self._lane_layers()[0]

    @property
    def non_lane_mask(self):
        """Floor cells seen and found NOT to be lane."""

        return self._lane_layers()[1]

    # ------------------------------------------------------------ lane

    @staticmethod
    def build_lane_mask():
        """Boolean grid, True on cells covered by the painted lane
        network (cfg.LANE_NETWORK), rasterised at half-lane-width."""

        mask = np.zeros((cfg.GRID_ROWS, cfg.GRID_COLS), dtype=bool)
        half_w_cells = int(round((cfg.LANE_WIDTH_M / 2.0) / cfg.GRID_RESOLUTION_M))

        for polyline in cfg.LANE_NETWORK:
            for (x0, y0), (x1, y1) in zip(polyline[:-1], polyline[1:]):
                length = np.hypot(x1 - x0, y1 - y0)
                steps = max(1, int(length / (cfg.GRID_RESOLUTION_M * 0.5)))
                for s in range(steps + 1):
                    t = s / steps
                    row, col = cfg.world_to_grid(x0 + t * (x1 - x0), y0 + t * (y1 - y0))
                    r0, r1 = max(0, row - half_w_cells), min(cfg.GRID_ROWS, row + half_w_cells + 1)
                    c0, c1 = max(0, col - half_w_cells), min(cfg.GRID_COLS, col + half_w_cells + 1)
                    mask[r0:r1, c0:c1] = True
        return mask

    def mark_lane_visited(self, x, y, radius_cells=4):
        """Record that the robot has physically driven over the lane near
        world (x, y) - visualisation only, not used for planning."""

        row, col = cfg.world_to_grid(x, y)
        r0, r1 = max(0, row - radius_cells), min(cfg.GRID_ROWS, row + radius_cells + 1)
        c0, c1 = max(0, col - radius_cells), min(cfg.GRID_COLS, col + radius_cells + 1)
        self._visited_lane[r0:r1, c0:c1] |= self.lane_mask[r0:r1, c0:c1]

    def visited_lane_fraction(self):
        """Fraction of the lane mapped so far that the robot has driven on."""

        total = int(np.count_nonzero(self.lane_mask))
        if total == 0:
            return 0.0
        return float(np.count_nonzero(self._visited_lane)) / total

    # ------------------------------------------------------ edits

    def clear_region(self, world_xy, radius_m):
        """Reset the log-odds (to unknown) in a disk around a world point.
        Used when a mapped object is physically removed - e.g. the arm
        module leaving its stand on the AMR's deck."""

        cx, cy = world_xy
        row0, col0 = cfg.world_to_grid(cx, cy)
        radius_cells = int(round(radius_m / cfg.GRID_RESOLUTION_M))
        rr, cc = np.ogrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
        disk = rr * rr + cc * cc <= radius_cells * radius_cells
        for dr, dc in zip(*np.nonzero(disk)):
            r, c = row0 + dr - radius_cells, col0 + dc - radius_cells
            if cfg.in_grid_bounds(r, c):
                self.logodds[r, c] = 0.0
        self.revision += 1

    def mark_footprint_free(self, x, y, yaw, margin_m=0.0):
        """The robot is standing here, so its own footprint is free space.
        Applied once per update so the planner's start cell is never
        poisoned by smeared obstacle evidence under the robot."""

        hl = cfg.AMR_HALF_LENGTH_M + margin_m
        hw = cfg.AMR_HALF_WIDTH_M + margin_m
        c, s = np.cos(yaw), np.sin(yaw)
        steps = np.arange(-1.0, 1.0001, cfg.GRID_RESOLUTION_M / max(hl, hw) * 0.7)
        a, b = np.meshgrid(steps * hl, steps * hw)
        xs = x + a * c - b * s
        ys = y + a * s + b * c
        cols = ((xs - cfg.WORLD_X_MIN) / cfg.GRID_RESOLUTION_M).astype(np.int64).ravel()
        rows = ((ys - cfg.WORLD_Y_MIN) / cfg.GRID_RESOLUTION_M).astype(np.int64).ravel()
        ok = (rows >= 0) & (rows < cfg.GRID_ROWS) & (cols >= 0) & (cols < cfg.GRID_COLS)
        flat = np.unique(rows[ok] * cfg.GRID_COLS + cols[ok])
        lo = self.logodds.reshape(-1)
        lo[flat] = np.minimum(lo[flat], cfg.LOGODDS_FREE * 3.0)

    # ------------------------------------------------- depth integration

    def update_from_depth(self, depth_image, intrinsics, cam_eye, cam_forward, cam_up):
        """Integrate one RGB-D depth frame (see module docstring for the
        rules). cam_eye/forward/up are the camera's world frame, built
        from the ESTIMATED robot pose by the caller. Returns a small
        stats dict."""

        world, _ = vo.project_depth_to_world(
            depth_image, intrinsics, cam_eye, cam_forward, cam_up,
            stride=DEPTH_PIXEL_STRIDE, max_range_m=MAX_RANGE_M * 1.5,
        )
        empty = {"rays": 0, "free_cells": 0, "occupied_cells": 0}
        if world.shape[0] == 0:
            return empty

        eye = np.asarray(cam_eye, dtype=np.float64)
        planar = np.hypot(world[:, 0] - eye[0], world[:, 1] - eye[1])
        keep = (planar >= MIN_RANGE_M) & (planar <= MAX_RANGE_M)
        world = world[keep]
        planar = planar[keep]
        if world.shape[0] == 0:
            return empty

        height = world[:, 2]
        is_obstacle = (height >= OBSTACLE_MIN_HEIGHT_M) & (height <= OBSTACLE_MAX_HEIGHT_M)

        res = cfg.GRID_RESOLUTION_M
        origin_c = (eye[0] - cfg.WORLD_X_MIN) / res
        origin_r = (eye[1] - cfg.WORLD_Y_MIN) / res
        end_c = (world[:, 0] - cfg.WORLD_X_MIN) / res
        end_r = (world[:, 1] - cfg.WORLD_Y_MIN) / res

        # Vectorised rays: sample every ray at <= half-cell steps.
        max_steps = int(np.ceil(MAX_RANGE_M / (0.5 * res))) + 2
        t = np.linspace(0.0, 1.0, max_steps, dtype=np.float64)[:, None]
        rows = origin_r + t * (end_r[None, :] - origin_r)
        cols = origin_c + t * (end_c[None, :] - origin_c)
        remaining = (1.0 - t) * planar[None, :]
        along = t * planar[None, :]

        sample_ok = along >= MIN_RANGE_M
        # Never paint free through (or onto) an obstacle surface.
        sample_ok &= ~(is_obstacle[None, :] & (remaining < res))

        rows_i = np.floor(rows).astype(np.int64)
        cols_i = np.floor(cols).astype(np.int64)
        sample_ok &= (rows_i >= 0) & (rows_i < cfg.GRID_ROWS) & (cols_i >= 0) & (cols_i < cfg.GRID_COLS)
        free_flat = np.unique(rows_i[sample_ok] * cfg.GRID_COLS + cols_i[sample_ok])

        occupied_flat = np.empty(0, dtype=np.int64)
        if np.any(is_obstacle):
            orow = np.floor(end_r[is_obstacle]).astype(np.int64)
            ocol = np.floor(end_c[is_obstacle]).astype(np.int64)
            inside = (orow >= 0) & (orow < cfg.GRID_ROWS) & (ocol >= 0) & (ocol < cfg.GRID_COLS)
            flat, counts = np.unique(orow[inside] * cfg.GRID_COLS + ocol[inside], return_counts=True)
            occupied_flat = flat[counts >= MIN_OCCUPIED_VOTES_PER_FRAME]

        if len(occupied_flat) and len(free_flat):
            free_flat = np.setdiff1d(free_flat, occupied_flat, assume_unique=True)

        lo = self.logodds.reshape(-1)
        if len(free_flat):
            lo[free_flat] = np.maximum(lo[free_flat] + cfg.LOGODDS_FREE, cfg.LOGODDS_MIN)
            fh = self.free_hits.reshape(-1)
            fh[free_flat] = np.minimum(fh[free_flat].astype(np.int32) + 1, 65000).astype(np.uint16)
        if len(occupied_flat):
            lo[occupied_flat] = np.minimum(lo[occupied_flat] + cfg.LOGODDS_OCCUPIED, cfg.LOGODDS_MAX)
            oh = self.occupied_hits.reshape(-1)
            oh[occupied_flat] = np.minimum(oh[occupied_flat].astype(np.int32) + 1, 65000).astype(np.uint16)

        self.integrations += 1
        self.revision += 1
        return {
            "rays": int(world.shape[0]),
            "free_cells": int(len(free_flat)),
            "occupied_cells": int(len(occupied_flat)),
        }

    # ------------------------------------------------------- queries

    def probability_grid(self):
        """Occupancy probability per cell from log-odds."""

        return 1.0 - 1.0 / (1.0 + np.exp(self.logodds))

    def masks(self):
        """(probability, occupied, free, unknown) boolean grids."""

        prob = self.probability_grid()
        occupied = prob > cfg.OCCUPANCY_OBSTACLE_PROB
        free = prob < cfg.OCCUPANCY_FREE_PROB
        unknown = ~(occupied | free)
        return prob, occupied, free, unknown

    def obstacle_grid(self):
        return self.probability_grid() > cfg.OCCUPANCY_OBSTACLE_PROB

    # ------------------------------------------------------- rendering

    def render(self, robot_pose=None, point_a=None, path=None, output_size=None,
               named_points=None, goal_xy=None):
        """
        Renders the occupancy map (as built so far) to a BGR image:

            dark grey   : unknown (never observed)
            white       : confirmed free space
            red         : confirmed obstacle (mapped from RGB-D depth)
            yellow      : the painted lane network (not yet driven)
            pale yellow : lane the robot has already driven over
            cyan line   : the active A* path (world waypoints)
            green dot + arrow : the robot's current ESTIMATED pose
            magenta ring: point A;  orange rings: other recorded points
            white cross : the active navigation goal

        World +Y points UP the image; optionally resized to output_size.
        """

        import cv2

        prob, occupied, free, unknown = self.masks()

        image = np.full((cfg.GRID_ROWS, cfg.GRID_COLS, 3), (60, 60, 60), dtype=np.uint8)
        image[free] = (235, 235, 235)
        lane_free = self.lane_mask & ~occupied
        image[lane_free] = np.where(
            self._visited_lane[lane_free][:, None],
            np.array([170, 230, 250], dtype=np.uint8),
            np.array([60, 200, 240], dtype=np.uint8),
        )
        image[occupied] = (40, 40, 220)

        image = np.flipud(image).copy()

        def to_pixel(wx, wy):
            row, col = cfg.world_to_grid(wx, wy)
            return int(col), int((cfg.GRID_ROWS - 1) - row)

        if path:
            pts = [to_pixel(x, y) for x, y in path]
            for p0, p1 in zip(pts[:-1], pts[1:]):
                cv2.line(image, p0, p1, (255, 255, 0), 1, cv2.LINE_AA)

        if point_a is not None:
            cv2.circle(image, to_pixel(*point_a), 6, (220, 0, 220), 1, cv2.LINE_AA)

        for name, (px, py) in (named_points or {}).items():
            cv2.circle(image, to_pixel(px, py), 5, (0, 140, 255), 1, cv2.LINE_AA)

        if goal_xy is not None:
            gx, gy = to_pixel(*goal_xy)
            cv2.drawMarker(image, (gx, gy), (255, 255, 255), cv2.MARKER_CROSS, 8, 1)

        if robot_pose is not None:
            cx, cy = to_pixel(robot_pose.x, robot_pose.y)
            cv2.circle(image, (cx, cy), 3, (0, 220, 0), -1, cv2.LINE_AA)
            tip = (cx + int(10 * np.cos(robot_pose.yaw)), cy - int(10 * np.sin(robot_pose.yaw)))
            cv2.arrowedLine(image, (cx, cy), tip, (0, 220, 0), 1, cv2.LINE_AA, tipLength=0.4)

        if output_size is not None:
            image = cv2.resize(image, output_size, interpolation=cv2.INTER_NEAREST)

        return image


# ============================================================
# STANDALONE SELF-TEST
# ============================================================

def main():
    print("Occupancy map self-test")
    print("-" * 50)

    occ = OccupancyMap()
    assert occ.lane_mask.any() == cfg.LANE_MAP_DEFINED
    occ.lane_hits[:] = 0; occ.floor_hits[:] = 0; occ.lane_revision += 1
    # Synthetic lane frame: a 0.6 m strip straight ahead in the floor LUT.
    from visual_odometry import Pose2D
    fwd, lat = np.meshgrid(np.linspace(0.6, 3.0, 60), np.linspace(-1.0, 1.0, 80), indexing="ij")
    lut = np.stack((fwd, lat), -1)
    mask = (np.abs(lat) < 0.3).astype(np.uint8) * 255
    for _ in range(3):
        occ.integrate_lane(mask, lut, Pose2D(0.0, 2.0, 0.0), stride=1)
    r, c = cfg.world_to_grid(1.5, 2.0)
    r2, c2 = cfg.world_to_grid(1.5, 2.8)
    print(f"  live lane layer: lane at strip centre={occ.lane_mask[r, c]}, "
          f"not-lane beside it={occ.non_lane_mask[r2, c2]}")
    assert occ.lane_mask[r, c] and occ.non_lane_mask[r2, c2]

    # Synthetic depth: a wall 3 m ahead of a camera at (0, 5, 0.5).
    intr = vo.intrinsics_from_fov(64, 48, 65.0)
    depth = np.full((48, 64), 3.0, dtype=np.float64)
    stats = occ.update_from_depth(depth, intr, [0.0, 5.0, 0.5], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    print(f"  integrate stats: {stats}")
    row, col = cfg.world_to_grid(3.0, 5.0)
    peak = float(occ.probability_grid()[row - 2:row + 3, col - 2:col + 3].max())
    row_f, col_f = cfg.world_to_grid(1.5, 5.0)
    free_p = float(occ.probability_grid()[row_f, col_f])
    print(f"  wall peak occupancy prob: {peak:.2f}, free-ray prob at 1.5 m: {free_p:.2f}")
    assert peak > 0.6 and free_p < 0.5

    print("SELF-TEST PASSED")


if __name__ == "__main__":
    main()
