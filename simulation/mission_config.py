"""
INDOOR LOCALIZATION - MISSION CONFIGURATION
=============================================

Single source of truth for every constant the autonomous stack shares:
world/grid geometry, the painted lane network, the obstacles placed on
it, the pick table at point A, the arm-module stand, the truck trailer,
the AMR / arm-module geometry and the mission start pose.

Keeping these here (rather than scattered across the perception,
mapping, planning and control modules) means the environment that is
*built* (warehouse_environment.py) and the algorithms that *perceive and
reason about* it stay in exact agreement - e.g. the lane-cell grid A*
prefers is generated from the very same LANE_NETWORK segments that get
painted on the floor.

WHAT THE ROBOT IS ALLOWED TO KNOW IN ADVANCE
    * the painted lane network (it is fixed floor infrastructure)
    * the FINAL location of the arm-module stand (ARM_STATION_XY) and of
      the truck boarding point (TRUCK_BOARDING_XY) - nothing about the
      route there; routes come from A* over the map built live from the
      RGB-D cameras
    * point A only as a lane destination to explore toward; the boxes,
      their number, size and exact positions are perceived
Ground-truth constants below that the robot must NOT use at runtime
(box positions, obstacle positions, truck deck geometry) are used only
by warehouse_environment.py to build the world and by tests to score
the result.
"""

import math


# ============================================================
# WORLD BOUNDS (metres, world frame) - must match
# warehouse_environment.py's WAREHOUSE_LENGTH_M / WAREHOUSE_WIDTH_M
# ============================================================

WORLD_X_MIN = -11.0
WORLD_X_MAX = 11.0
WORLD_Y_MIN = -9.0
WORLD_Y_MAX = 9.0


# ============================================================
# OCCUPANCY / LANE GRID
# ============================================================
# 0.10 m cells over the full warehouse footprint. Rows index world Y,
# columns index world X, so GRID_ROWS spans the 18 m width and GRID_COLS
# spans the 22 m length.

GRID_RESOLUTION_M = 0.10
GRID_COLS = int(round((WORLD_X_MAX - WORLD_X_MIN) / GRID_RESOLUTION_M))  # X, 220
GRID_ROWS = int(round((WORLD_Y_MAX - WORLD_Y_MIN) / GRID_RESOLUTION_M))  # Y, 180

# Log-odds occupancy update parameters (robot_vision stage8 values).
LOGODDS_OCCUPIED = 0.85
LOGODDS_FREE = -0.35
LOGODDS_MIN = -2.0
LOGODDS_MAX = 3.5
OCCUPANCY_FREE_PROB = 0.35
OCCUPANCY_OBSTACLE_PROB = 0.65


# ============================================================
# AMR GEOMETRY (amr_robot.urdf)
# ============================================================

AMR_HALF_LENGTH_M = 0.35          # chassis 0.70 m long
AMR_HALF_WIDTH_M = 0.31           # outer drive-wheel faces (0.28 + 0.03)
AMR_CIRCUMSCRIBED_RADIUS_M = math.hypot(AMR_HALF_LENGTH_M, AMR_HALF_WIDTH_M)
AMR_WHEEL_RADIUS_M = 0.10
AMR_WHEEL_SEPARATION_M = 0.56
AMR_BASE_Z_M = 0.15               # base_link origin height when standing

# Odometry calibration (UMBmark-style: straight runs at several speeds in
# both directions + in-place turns, compared against the simulator's ground
# truth once, offline). Nominal values are r=0.10 m, b=0.56 m; the robot
# actually travels 0.9% further per wheel revolution and turns ~2.2% less
# than the nominal model predicts (tyre contact / caster scrub). The pose
# estimator uses these calibrated values; the drive command uses nominal.
ODOM_WHEEL_RADIUS_M = 0.1009
ODOM_WHEEL_SEPARATION_M = 0.578
AMR_DECK_TOP_Z_M = 0.30           # chassis top face (world)
AMR_PAD_TOP_Z_M = 0.32            # top of the chassis magnet pads (world)


# ============================================================
# PAINTED LANE NETWORK
# ============================================================
# Each entry is a polyline (list of (x, y) world points), painted as
# connected floor strips AND rasterised into the lane mask the global
# planner PREFERS (lane cells are cheap, off-lane free space costs more,
# so A* follows the lane unless an obstacle or the task forces it off).
#
# Topology (top-down, +X right, +Y up):
#
#   TRUCK (-10.2, 6.5) ------------------+
#                                        |
#   ARM STAND (-7.5, 2.2) ---------------+ spine X = -2.8
#                                        |
#                                        +---- middle aisle ----> A (pick table)
#   SPAWN (-8.0,-2.2) -------------------+
#                                        |
#                                        +---- aisle 0 ---------> B (shelf, big box)
#   CONVEYOR STAND (-7.5,-6.5) ----------+

X_CORRIDOR = -2.8          # vertical spine, just west of the rack fronts
Y_SPAWN = -2.2             # LOADING_UNLOADING bay row
Y_PICKING = 2.2            # PICKING bay row (arm-module stand)
Y_SHIPPING = 6.5           # SHIPPING bay row (truck)
Y_AISLE = 0.0              # middle storage aisle centre
Y_AISLE_0 = -5.5           # first storage aisle centre (shelf B)
Y_RECEPTION = -6.5         # RECEPTION bay row (conveyor-module stand)

LANE_WIDTH_M = 0.60
LANE_COLOR_RGBA = [1.0, 0.85, 0.0, 1.0]


# ============================================================
# ARM-MODULE STAND (PICKING dock) - only its FINAL location is stored
# ============================================================
# A gantry stand holding the swappable suction-arm module on a hoist.
# The AMR reverses in underneath it (the module overhangs the rear/centre
# of the deck, behind the sensor mast, so it has to slide on from the
# back), the hoist lowers the module onto the deck, the electromagnets
# lock it, the hoist lifts clear and the AMR drives out forward.
#
# ARM_STATION_XY / ARM_STATION_DOCK_YAW_RAD are the docked AMR pose: the
# only things about the stand the mission is given in advance.

ARM_STATION_XY = (-7.5, Y_PICKING)
ARM_STATION_DOCK_YAW_RAD = 0.0       # AMR faces +X when docked (reversed in)
ARM_STATION_PRE_DOCK_DIST_M = 1.80   # far enough that both near legs are in the camera view
ARM_STATION_NAME = "ARM_STATION"

# Stand structure (stand frame = docked AMR frame).
STAND_LEG_X_M = 0.40                 # legs at x = +-0.40
STAND_LEG_Y_M = 0.50                 # legs at y = +-0.50 (inner face 0.46)
STAND_LEG_SIZE_M = 0.08
STAND_HEIGHT_M = 1.70
STAND_HOIST_X_M = -0.11              # hoist carriage over the module plate centre
STAND_HOIST_LOWER_M = 0.05           # hoist stroke to set the module down
STAND_HOIST_RAISE_M = 0.25           # hoist lift after release

# Module plate bottom height while held by the hoist, and once docked
# (resting on the AMR's magnet pads).
MODULE_HOLD_Z_M = 0.38
MODULE_DOCKED_Z_M = AMR_PAD_TOP_Z_M

# Second tool stand (same gantry design, hoist rods spaced for the wider
# conveyor module) at the RECEPTION dock. Again only its final location
# is given to the mission.
CONVEYOR_STATION_XY = (-7.5, Y_RECEPTION)
CONVEYOR_STATION_DOCK_YAW_RAD = 0.0
CONVEYOR_STATION_NAME = "CONVEYOR_STATION"

# Module type -> its stand (docked AMR pose) and map label.
STATIONS = {
    "SUCTION_ARM": {"xy": ARM_STATION_XY, "yaw": ARM_STATION_DOCK_YAW_RAD,
                    "name": ARM_STATION_NAME},
    "CONVEYOR": {"xy": CONVEYOR_STATION_XY, "yaw": CONVEYOR_STATION_DOCK_YAW_RAD,
                 "name": CONVEYOR_STATION_NAME},
}


# ============================================================
# CONVEYOR MODULE GEOMETRY (conveyor_module.urdf)
# ============================================================
# Same module frame / plate / magnet pads as the arm module. A powered
# belt on a vertical lift; the belt's NOSE overhangs the AMR's REAR (the
# sensor mast is at the front), so the AMR reverses up to a shelf or
# truck to load / unload over the nose.
CONV_BELT_X_MIN_M = -0.62          # nose end of the belt (behind the AMR)
CONV_BELT_X_MAX_M = 0.18           # front end (end stop behind the mast)
CONV_BELT_HALF_WIDTH_M = 0.30
CONV_LIFT_BASE_Z_M = 0.10          # belt top (module frame) at lift = 0
CONV_LIFT_MAX_M = 0.50             # belt top up to module 0.60 (world 0.92)
CONV_IR_NOSE_X_M = -0.55           # IR beam across the belt near the nose
CONV_IR_END_X_M = 0.12             # IR beam near the front end stop
CONV_IR_HEIGHT_M = 0.04            # beam height above the belt
CONV_TOF_X_M = -0.63               # rear-facing ToF range sensor under the nose
CONV_TOF_DZ_M = -0.035             # ...this far below the belt top
CONV_BELT_SPEED_MPS = 0.25


# ============================================================
# SUCTION-ARM MODULE GEOMETRY (suction_arm_module.urdf)
# ============================================================
# Module frame: origin at the plate bottom, directly above the AMR
# centre when docked, x forward / y left / z up (= AMR base axes).

MODULE_PLATE_X_MIN_M = -0.36
MODULE_PLATE_X_MAX_M = 0.14
MODULE_PLATE_HALF_WIDTH_M = 0.30
MODULE_PLATE_THICKNESS_M = 0.03

# 6-DOF arm (UR-style chain) mounted on a short column.
ARM_SHOULDER_PAN_XYZ = (0.02, 0.0, 0.25)  # pan joint origin in module frame
ARM_SHOULDER_LIFT_DZ = 0.06
ARM_UPPER_ARM_M = 0.50
ARM_FOREARM_M = 0.45
ARM_WRIST1_TO_WRIST2_M = 0.08
ARM_WRIST2_TO_FLANGE_M = 0.07
SUCTION_TIP_OFFSET_M = 0.11              # flange origin -> cup lip, along tool +X

# Eye-in-hand stereo camera on the tool flange (looks along tool +X,
# baseline along the flange's local Y axis).
ARM_CAMERA_OFFSET_FLANGE = (0.03, 0.0, 0.07)   # (x along tool, y, z)
ARM_STEREO_BASELINE_M = 0.075                  # OAK-D Lite class
ARM_CAMERA_WIDTH = 320
ARM_CAMERA_HEIGHT = 240
ARM_CAMERA_VFOV_DEG = 60.0
ARM_CAMERA_NEAR = 0.03
ARM_CAMERA_FAR = 3.0

# Cargo tray on the module plate, behind the arm column: two slots per
# layer, two layers (filled bottom layer first).
CARGO_SLOT_XY = [(-0.22, 0.10), (-0.22, -0.10)]
CARGO_LAYERS = 2
CARGO_TRAY_TOP_Z_M = MODULE_PLATE_THICKNESS_M   # module frame


# ============================================================
# LANE OBSTACLES (static clutter ON the lane that forces a detour)
# ============================================================
# Each obstacle sits on the painted lane so the AMR genuinely has to
# leave it, and each leaves ONE comfortably wide detour side (>= 1.2 m
# clear for a 0.62 m wide robot, i.e. >= 0.29 m spare per side) while
# the other side is either open floor or too narrow to tempt the
# planner - so the detour choice is unambiguous and never a squeeze.
# Entry: (x, y, half_x, half_y, half_z, color, label).
LANE_OBSTACLES = [
    # Pallet straddling the spawn spur, open dock floor on both sides.
    (-5.6, Y_SPAWN + 0.05, 0.40, 0.35, 0.14, [0.55, 0.40, 0.20, 1.0], "pallet"),
    # Crate in the middle aisle on the right of the lane: 1.25 m clear
    # between it and the left rack row, 0.35 m (impassable) on its right.
    (0.8, Y_AISLE - 0.45, 0.30, 0.30, 0.30, [0.20, 0.20, 0.22, 1.0], "crate"),
    # Trolley on the spine where it runs past the third rack aisle: only
    # 0.55 m to the rack fronts on its east, open floor to the west.
    # Only reached on the way to the truck, so it is NOT in the map yet
    # when the robot gets there - tests live discovery + replanning.
    (X_CORRIDOR - 0.05, 4.2, 0.30, 0.30, 0.40, [0.25, 0.55, 0.30, 1.0], "trolley"),
    # Barrel stack on the southern spine where it crosses the open corridor
    # between rack blocks: >= 1.5 m clear on BOTH sides, and 2.5 m of
    # straight lane after it to re-centre before the aisle-0 turn.
    (X_CORRIDOR, -3.0, 0.30, 0.30, 0.35, [0.70, 0.25, 0.20, 1.0], "barrels"),
    # Cart in aisle 0, on the left of the lane: 1.25 m clear to its right.
    (1.5, Y_AISLE_0 + 0.45, 0.30, 0.30, 0.30, [0.30, 0.30, 0.55, 1.0], "cart"),
    # Waste bin beside the shipping spur in the open dock band (passable
    # on both sides, >= 1.5 m clear).
    (-6.0, Y_SHIPPING - 0.15, 0.30, 0.30, 0.30, [0.15, 0.45, 0.20, 1.0], "bin"),
]


# ============================================================
# POINT A - open pick table at the end of the middle aisle
# ============================================================
# The pickable stock sits on an open-topped pick table (top-down suction
# access), facing the AMR's approach along the aisle lane.

POINT_A_NAME = "A"

PICK_TABLE_X_MIN = 4.45
PICK_TABLE_X_MAX = 4.95
PICK_TABLE_HALF_WIDTH_M = 0.55
PICK_TABLE_TOP_Z_M = 0.40
PICK_TABLE_TOP_THICKNESS_M = 0.04
PICK_TABLE_COLOR = [0.30, 0.42, 0.50, 1.0]

BOX_HALF_EXTENT_M = 0.08          # 0.16 m cubes: small, light, suction-pickable
BOX_MASS_KG = 1.0
BOX_COLOR_RGBA = [0.80, 0.52, 0.25, 1.0]   # cardboard brown
BOX_ROW_X = PICK_TABLE_X_MIN + 0.04 + BOX_HALF_EXTENT_M
BOX_ROW_Y = [-0.30, -0.10, 0.10, 0.30]
BOXES_AT_A = [(BOX_ROW_X, y, BOX_HALF_EXTENT_M) for y in BOX_ROW_Y]

POINT_A = (BOX_ROW_X, Y_AISLE)    # centroid of the stock (nominal)

# Payload decision: boxes at most this big (footprint) are handled by
# the suction-arm module; anything bigger needs the conveyor module.
SUCTION_MAX_BOX_FOOTPRINT_M = 0.40
SUCTION_MIN_BOX_COUNT = 1

# Detection pose on the aisle lane, ~1.8 m short of the table, facing +X.
DETECTION_POSE_XY = (2.4, Y_AISLE)

# Stand-off between the AMR bumper and a pick/place surface edge.
WORK_STANDOFF_M = 0.12


# ============================================================
# POINT B - shelf with one oversized box at the end of aisle 0
# ============================================================
POINT_B_NAME = "B"
SHELF_B_X_MIN = 4.45               # front edge (facing the approaching AMR)
SHELF_B_X_MAX = 4.95
SHELF_B_HALF_WIDTH_M = 0.40
SHELF_B_TOP_Z_M = 0.62
SHELF_B_SLAB_M = 0.06
SHELF_B_COLOR = [0.35, 0.36, 0.40, 1.0]
# 0.45 (deep) x 0.50 (wide) x 0.32 (tall), overhanging the shelf front
# edge by 8 cm the way cartons do on real shelving.
BIG_BOX_HALF = (0.225, 0.25, 0.16)
BIG_BOX_OVERHANG_M = 0.08
BIG_BOX_MASS_KG = 3.0
BIG_BOX_XY = (SHELF_B_X_MIN - BIG_BOX_OVERHANG_M + BIG_BOX_HALF[0], Y_AISLE_0)
DETECTION_POSE_B_XY = (2.3, Y_AISLE_0)
SHELF_APPROACH_GAP_M = 0.60        # bumper-to-shelf gap for the forward measurement


# ============================================================
# TRUCK (SHIPPING dock) - hollow box trailer, rear door open
# ============================================================
# Backed up to the open SHIPPING dock bay; its cargo deck is at a
# low-deck delivery-trailer height so the deck-mounted arm can reach in.

TRUCK_REAR_X = -10.70             # rear sill (open door) x
TRUCK_LENGTH_M = 2.40
TRUCK_HALF_WIDTH_M = 1.00
TRUCK_DECK_Z_M = 0.40
TRUCK_INTERIOR_HEIGHT_M = 1.50
TRUCK_WALL_THICKNESS_M = 0.05
TRUCK_COLOR_RGBA = [0.20, 0.30, 0.75, 1.0]
TRUCK_Y = Y_SHIPPING
TRUCK_NAME = "TRUCK"

# Where the AMR parks to load. The trailer's right half takes the
# arm-loaded small boxes, its left half the conveyor-loaded big box.
# Only these final points are given.
TRUCK_ARM_Y = TRUCK_Y - 0.45
TRUCK_CONVEYOR_Y = TRUCK_Y + 0.45
TRUCK_BOARDING_XY = (TRUCK_REAR_X + WORK_STANDOFF_M + AMR_HALF_LENGTH_M, TRUCK_ARM_Y)
TRUCK_BOARDING_YAW_RAD = math.pi
TRUCK_APPROACH_DIST_M = 0.75      # navigation goal this far short, then creep
TRUCK_CONVEYOR_OVERLAP_M = 0.15   # belt nose this far inside the sill to unload


# ============================================================
# LANE NETWORK (built from the points above)
# ============================================================

PICK_PARK_X = PICK_TABLE_X_MIN - WORK_STANDOFF_M - AMR_HALF_LENGTH_M
SHELF_B_PARK_X = SHELF_B_X_MIN + CONV_BELT_X_MIN_M - 0.01   # reversed, nose at the shelf

LANE_NETWORK = [
    # spawn spur: LOADING bay -> spine
    [(-8.0, Y_SPAWN), (X_CORRIDOR, Y_SPAWN)],
    # vertical spine (RECEPTION row to SHIPPING row)
    [(X_CORRIDOR, Y_RECEPTION), (X_CORRIDOR, Y_SHIPPING)],
    # reception spur: spine -> through the conveyor-module stand
    [(X_CORRIDOR, Y_RECEPTION), (CONVEYOR_STATION_XY[0], Y_RECEPTION)],
    # aisle-0 branch: spine -> shelf B
    [(X_CORRIDOR, Y_AISLE_0), (SHELF_B_PARK_X, Y_AISLE_0)],
    # picking spur: spine -> through the arm-module stand
    [(X_CORRIDOR, Y_PICKING), (ARM_STATION_XY[0], Y_PICKING)],
    # shipping spur: spine -> truck boarding point
    [(X_CORRIDOR, Y_SHIPPING), (TRUCK_BOARDING_XY[0], Y_SHIPPING)],
    # aisle branch: spine -> pick table at A
    [(X_CORRIDOR, Y_AISLE), (PICK_PARK_X, Y_AISLE)],
]

# The lane layout is DEFINED infrastructure: the robot is given the painted
# lane centrelines (above) as its lane layer from the start, so A* keeps to
# the lane from the very first route. Obstacles, stations, stock and the
# truck are still found and pinned live.
LANE_MAP_DEFINED = True

# HSV thresholds for the bright yellow lane paint (centre RGB camera).
# Saturation is the separating feature: the LOADING/UNLOADING bay marker
# renders opaque at HSV ~(22, 202, 227) - the renderer ignores its alpha -
# while the lane paint is ~(25, 255, 239). The old S >= 120 bound let the
# whole bay floor through as "lane" around the spawn point.
LANE_HSV_LOWER = (20, 230, 120)
LANE_HSV_UPPER = (34, 255, 255)


# ============================================================
# MISSION START POSE
# ============================================================

ROBOT_START_XY = (-8.0, Y_SPAWN)
ROBOT_START_YAW_RAD = 0.0
ROBOT_START_Z = 0.17


# ============================================================
# GRID <-> WORLD CONVERSIONS (shared by every map/planner module)
# ============================================================

def world_to_grid(x, y):
    """World (x, y) metres -> (row, col) grid indices (row=Y, col=X)."""

    col = int((x - WORLD_X_MIN) / GRID_RESOLUTION_M)
    row = int((y - WORLD_Y_MIN) / GRID_RESOLUTION_M)
    return row, col


def grid_to_world(row, col):
    """(row, col) grid indices -> world (x, y) metres at the cell centre."""

    x = WORLD_X_MIN + (col + 0.5) * GRID_RESOLUTION_M
    y = WORLD_Y_MIN + (row + 0.5) * GRID_RESOLUTION_M
    return x, y


def in_grid_bounds(row, col):
    return 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS


def clamp_angle(angle_rad):
    """Wrap an angle to (-pi, pi]."""

    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def arm_station_pre_dock_xy():
    """Where the AMR stops (east of the stand) to turn round and reverse in."""

    return (ARM_STATION_XY[0] + ARM_STATION_PRE_DOCK_DIST_M * math.cos(ARM_STATION_DOCK_YAW_RAD),
            ARM_STATION_XY[1] + ARM_STATION_PRE_DOCK_DIST_M * math.sin(ARM_STATION_DOCK_YAW_RAD))


def truck_approach_xy(module="SUCTION_ARM"):
    """Navigation goal in front of the trailer (facing it, -X): ~0.75 m
    short of the arm boarding pose, or 0.6 m bumper gap for the conveyor
    (which then turns round and reverses its nose over the sill)."""

    if module == "CONVEYOR":
        return (TRUCK_REAR_X + AMR_HALF_LENGTH_M + SHELF_APPROACH_GAP_M + 0.1, TRUCK_CONVEYOR_Y)
    return (TRUCK_BOARDING_XY[0] - TRUCK_APPROACH_DIST_M * math.cos(TRUCK_BOARDING_YAW_RAD),
            TRUCK_BOARDING_XY[1] - TRUCK_APPROACH_DIST_M * math.sin(TRUCK_BOARDING_YAW_RAD))


def pre_dock_xy(module):
    st = STATIONS[module]
    return (st["xy"][0] + ARM_STATION_PRE_DOCK_DIST_M * math.cos(st["yaw"]),
            st["xy"][1] + ARM_STATION_PRE_DOCK_DIST_M * math.sin(st["yaw"]))


# The mission: one job per stock location, done in order, then home.
JOBS = [
    {"name": POINT_A_NAME, "detect_xy": DETECTION_POSE_XY, "heading": 0.0},
    {"name": POINT_B_NAME, "detect_xy": DETECTION_POSE_B_XY, "heading": 0.0},
]
HOME_XY = ROBOT_START_XY
HOME_YAW_RAD = ROBOT_START_YAW_RAD
