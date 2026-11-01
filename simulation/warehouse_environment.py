"""
INDOOR LOCALIZATION - LOGISTICS WAREHOUSE ENVIRONMENT
=======================================================

Builds a PyBullet warehouse world for indoor AMR localization work,
laid out to match a typical logistics-centre floor plan:

    - RECEPTION            : inbound dock, trucks arrive here
    - LOADING & UNLOADING  : dock doors where goods are moved on/off trucks
    - STORAGE               : 3 parallel pallet-rack aisles
    - PICKING               : staging area between storage and outbound dock
    - SHIPPING               : outbound dock, trucks depart here

This module only builds the static world (floor, walls, racks, dock
markers, dock trucks). Robot loading and simulation stepping live in
main.py, kept separate the same way stage1_simulation.py in the
robot_vision project separates create_environment() from load_robot().

This file is part of the indoor_localization project and does not modify,
import, or depend on anything inside the robot_vision folder.
"""

import math
from pathlib import Path

import pybullet as p
import pybullet_data

import mission_config as cfg

SCRIPT_DIR = Path(__file__).resolve().parent


# ============================================================
# FLOOR PLAN CONFIGURATION (metres, world frame)
# ============================================================
#
# World X axis -> warehouse length (reception/dock side at low X,
#                 storage racks at high X)
# World Y axis -> warehouse width
#
# Layout (top-down):
#
#   X=0                                                X=22
#   +--------------------------------------------------+
#   |  RECEPTION   LOADING&UNLOAD   PICKING   SHIPPING |  <- dock band
#   |------------------ main aisle ----------------------|
#   |   [ RACK AISLE 1 ]   [ RACK AISLE 2 ]  [ RACK AISLE 3 ] | <- storage
#   +--------------------------------------------------+
#   Y=-9                                                Y=9

WAREHOUSE_LENGTH_M = 22.0
WAREHOUSE_WIDTH_M = 18.0

WALL_HEIGHT_M = 3.0
WALL_THICKNESS_M = 0.15

FLOOR_HALF_EXTENTS = [
    WAREHOUSE_LENGTH_M / 2.0,
    WAREHOUSE_WIDTH_M / 2.0,
    0.02,
]

# ------------------------------------------------------------
# DOCK BAND (reception / loading & unloading / picking / shipping)
# ------------------------------------------------------------
# Dock band occupies the low-X end of the warehouse; storage racks occupy
# the high-X end. Each dock zone is a shallow coloured floor marker plus
# a parked "truck" box standing just outside the back wall, matching the
# reference illustration (Reception / Loading & Unloading on the left,
# Picking / Shipping on the right, trucks backed up to the building).

DOCK_BAND_X_START = -WAREHOUSE_LENGTH_M / 2.0 + 1.0
DOCK_BAND_X_END = -WAREHOUSE_LENGTH_M / 2.0 + 6.0
DOCK_BAND_DEPTH_M = DOCK_BAND_X_END - DOCK_BAND_X_START

DOCK_ZONES = [
    {
        "name": "RECEPTION",
        "color": [0.25, 0.55, 0.85, 0.55],
        "y_center": -6.5,
    },
    {
        "name": "LOADING_UNLOADING",
        "color": [0.95, 0.75, 0.20, 0.55],
        "y_center": -2.2,
    },
    {
        "name": "PICKING",
        "color": [0.35, 0.80, 0.45, 0.55],
        "y_center": 2.2,
    },
    {
        "name": "SHIPPING",
        "color": [0.85, 0.35, 0.30, 0.55],
        "y_center": 6.5,
    },
]

DOCK_ZONE_WIDTH_M = 3.6

# ------------------------------------------------------------
# STORAGE RACK AISLES
# ------------------------------------------------------------
# 3 pallet-rack aisles, each made of two facing rack rows with a travel
# aisle in between, oriented along X (matching the reference image, where
# racks run the long way with forklift/robot aisles between rows).

NUMBER_OF_RACK_AISLES = 3

RACK_AISLE_X_START = -WAREHOUSE_LENGTH_M / 2.0 + 9.0
RACK_AISLE_LENGTH_M = 10.0

RACK_ROW_HALF_DEPTH_M = 0.45        # rack shelf depth / 2
RACK_HEIGHT_M = 2.2
RACK_BAY_LENGTH_M = 1.6             # length of one shelving bay along X
RACK_BAY_GAP_M = 0.15               # small visual gap between bays

# Cupboard-style shelving: a thin steel frame (uprights + horizontal
# shelf boards) with 3 open shelf levels, each carrying a row of goods
# boxes - a "proper logistics factory" look rather than one solid block.
RACK_POST_SIZE_M = 0.06             # upright post cross-section
RACK_SHELF_THICKNESS_M = 0.03
RACK_NUM_SHELF_LEVELS = 3
RACK_LEVEL_HEIGHTS_M = [0.05, 0.80, 1.55]   # shelf-board height off the floor
RACK_GOODS_PER_LEVEL = 3            # boxes sitting on each shelf level

RACK_FRAME_COLOR = [0.08, 0.09, 0.11, 1.0]      # dark steel frame
RACK_SHELF_BOARD_COLOR = [0.65, 0.66, 0.68, 1.0]  # light grey shelf boards
# A small palette of goods-box colours so shelves read as varied stock
# rather than one uniform block.
RACK_GOODS_COLORS = [
    [0.80, 0.52, 0.25, 1.0],   # cardboard brown
    [0.75, 0.20, 0.18, 1.0],   # red tote
    [0.20, 0.45, 0.70, 1.0],   # blue tote
    [0.85, 0.78, 0.30, 1.0],   # yellow crate
]

TRAVEL_AISLE_WIDTH_M = 2.2          # clear width between the two rack rows
RACK_AISLE_PITCH_M = (
    2.0 * RACK_ROW_HALF_DEPTH_M + TRAVEL_AISLE_WIDTH_M + 2.4
)   # centre-to-centre spacing between successive rack aisles

# ------------------------------------------------------------
# WORK STATIONS (robotic arm + conveyor extension)
# ------------------------------------------------------------
# Robotic arm station sits in the PICKING zone's dock bay, at the same X
# position as the dock-zone floor markers, so the AMR drives into the
# dock bay and stops directly under the pedestal to be picked up.
#
# Conveyor station sits in the SHIPPING zone's dock bay, extending from
# the AMR's stopping point toward the parked truck (which stands further
# in -X, outside the building - see create_dock_truck/create_dock_zones)
# so goods travelling along the belt move from the AMR into the truck.
# Because the truck is on the -X side and the AMR approaches from +X,
# the conveyor URDF (whose local +X runs intake -> outfeed) is yawed
# 180 degrees so its outfeed end points toward the truck.

ARM_STATION_FILENAME = "robotic_arm_station.urdf"
CONVEYOR_STATION_FILENAME = "conveyor_station.urdf"

ARM_STATION_ZONE_NAME = "PICKING"
CONVEYOR_STATION_ZONE_NAME = "SHIPPING"

# Station base sits on the floor (z=0); the URDFs' own link origins
# already account for pedestal/frame height above that.
ARM_STATION_BASE_Z_M = 0.0
CONVEYOR_STATION_BASE_Z_M = 0.0


# ============================================================
# SIMULATION INITIALIZATION
# ============================================================

def initialize_simulation(simulation_hz=240):
    """
    Connect to PyBullet (GUI mode) and configure gravity/timestep.

    Mirrors the initialization contract used across the project family
    (connect -> search path -> gravity -> timestep) without reusing any
    code from the robot_vision folder.
    """

    physics_client = p.connect(p.GUI)

    if physics_client < 0:
        raise RuntimeError("Could not connect to PyBullet.")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.setGravity(0, 0, -9.81)

    p.setTimeStep(1.0 / simulation_hz)

    return physics_client


# ============================================================
# GENERIC STATIC GEOMETRY HELPERS
# ============================================================

def create_static_box(position, half_extents, color, yaw_degrees=0.0):
    """
    Create a fixed (mass=0) box body with the given half extents, colour
    and yaw rotation about the world Z axis.
    """

    orientation = p.getQuaternionFromEuler([0.0, 0.0, math.radians(yaw_degrees)])

    collision_shape = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=half_extents
    )

    visual_shape = p.createVisualShape(
        p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color
    )

    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        baseOrientation=orientation,
    )


def create_floor_marker(position, half_extents, color):
    """
    Thin, non-colliding coloured rectangle used to mark a floor zone
    (dock bands) without affecting navigation/collision.
    """

    visual_shape = p.createVisualShape(
        p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color
    )

    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
    )


# ============================================================
# FLOOR AND PERIMETER WALLS
# ============================================================

def create_floor():
    """
    Flat ground plane sized to the warehouse footprint instead of the
    default infinite PyBullet plane, so the dock/storage layout has a
    clearly bounded floor to render floor-zone markers on top of.
    """

    return create_static_box(
        position=[0.0, 0.0, -0.02],
        half_extents=FLOOR_HALF_EXTENTS,
        color=[0.78, 0.78, 0.80, 1.0],
    )


def create_perimeter_walls():
    """
    Four walls enclosing the warehouse footprint. The dock-band wall
    (low-X, "front" of the building) is intentionally left with a gap
    cut out per dock zone further down (create_dock_zones), approximated
    here with simple segments so the dock trucks appear to back up to
    open bays rather than a solid wall.
    """

    half_length = WAREHOUSE_LENGTH_M / 2.0
    half_width = WAREHOUSE_WIDTH_M / 2.0

    wall_ids = []

    # Back wall (storage side, high X)
    wall_ids.append(
        create_static_box(
            position=[half_length, 0.0, WALL_HEIGHT_M / 2.0],
            half_extents=[
                WALL_THICKNESS_M / 2.0,
                half_width,
                WALL_HEIGHT_M / 2.0,
            ],
            color=[0.55, 0.55, 0.58, 1.0],
        )
    )

    # Side walls (long walls running along X)
    for y_sign in (-1.0, 1.0):
        wall_ids.append(
            create_static_box(
                position=[0.0, y_sign * half_width, WALL_HEIGHT_M / 2.0],
                half_extents=[
                    half_length,
                    WALL_THICKNESS_M / 2.0,
                    WALL_HEIGHT_M / 2.0,
                ],
                color=[0.55, 0.55, 0.58, 1.0],
            )
        )

    # Front wall (dock side, low X) - built as 5 short segments so each
    # dock zone has an open bay rather than driving through solid wall.
    segment_targets = [-half_width] + [
        zone["y_center"] - DOCK_ZONE_WIDTH_M / 2.0 for zone in DOCK_ZONES
    ] + [
        zone["y_center"] + DOCK_ZONE_WIDTH_M / 2.0 for zone in DOCK_ZONES
    ] + [half_width]
    segment_targets = sorted(set(round(v, 3) for v in segment_targets))

    for y_start, y_end in zip(segment_targets[:-1], segment_targets[1:]):
        segment_width = y_end - y_start
        # Skip segments that line up with an open dock bay.
        segment_centre = (y_start + y_end) / 2.0
        is_open_bay = any(
            abs(segment_centre - zone["y_center"]) < 0.05
            for zone in DOCK_ZONES
        )
        if is_open_bay:
            continue

        wall_ids.append(
            create_static_box(
                position=[-half_length, segment_centre, WALL_HEIGHT_M / 2.0],
                half_extents=[
                    WALL_THICKNESS_M / 2.0,
                    segment_width / 2.0,
                    WALL_HEIGHT_M / 2.0,
                ],
                color=[0.55, 0.55, 0.58, 1.0],
            )
        )

    return wall_ids


# ============================================================
# DOCK ZONES (RECEPTION / LOADING & UNLOADING / PICKING / SHIPPING)
# ============================================================

def create_dock_truck(position, yaw_degrees, color):
    """
    Simplified box-truck silhouette (cab + trailer) parked outside the
    dock wall, matching the trucks shown backed up to the building in the
    reference illustration. Purely a visual/collision landmark, not a
    dynamic actor.
    """

    bodies = []

    # Trailer (long box)
    bodies.append(
        create_static_box(
            position=position,
            half_extents=[1.1, 1.0, 1.1],
            color=color,
            yaw_degrees=yaw_degrees,
        )
    )

    # Cab (smaller box offset further from the building)
    cab_offset = -2.3 if position[0] < 0 else 2.3
    cab_position = [position[0] + cab_offset, position[1], position[2] - 0.1]
    bodies.append(
        create_static_box(
            position=cab_position,
            half_extents=[0.75, 0.9, 1.0],
            color=[0.85, 0.85, 0.88, 1.0],
            yaw_degrees=yaw_degrees,
        )
    )

    return bodies


def create_dock_zones():
    """
    One floor-colour marker plus one parked truck per dock zone, in the
    order Reception / Loading & Unloading / Picking / Shipping (matching
    the reference illustration's left-to-right arrangement).
    """

    half_length = WAREHOUSE_LENGTH_M / 2.0
    zone_centre_x = -half_length + DOCK_BAND_DEPTH_M / 2.0 + 1.0

    created = {"markers": [], "trucks": []}

    for zone in DOCK_ZONES:
        marker_id = create_floor_marker(
            position=[zone_centre_x, zone["y_center"], 0.01],
            half_extents=[DOCK_BAND_DEPTH_M / 2.0, DOCK_ZONE_WIDTH_M / 2.0, 0.001],
            color=zone["color"],
        )
        created["markers"].append((zone["name"], marker_id))

        # The SHIPPING bay gets the hollow, loadable trailer instead
        # (create_truck_trailer); the others keep a solid parked truck.
        if zone["name"] == "SHIPPING":
            continue

        truck_bodies = create_dock_truck(
            position=[-half_length - 1.4, zone["y_center"], 1.1],
            yaw_degrees=0.0,
            color=[0.25, 0.35, 0.75, 1.0],
        )
        created["trucks"].extend(truck_bodies)

    return created


# ============================================================
# STORAGE RACK AISLES
# ============================================================

def _create_rack_bay(bay_x, y_centre, yaw_degrees, color_seed):
    """
    One cupboard-style shelving bay: 4 slim corner posts, a horizontal
    shelf board at each of RACK_LEVEL_HEIGHTS_M, and a row of goods boxes
    resting on each shelf level. This is an open rack you can see INTO
    (posts + boards, not a solid block), matching a real pallet-rack
    aisle rather than a painted cube.
    """

    bodies = []
    half_length = RACK_BAY_LENGTH_M / 2.0
    half_depth = RACK_ROW_HALF_DEPTH_M

    # --- 4 corner posts, full rack height ---
    for dx_sign in (-1.0, 1.0):
        for dy_sign in (-1.0, 1.0):
            post_x = bay_x + dx_sign * (half_length - RACK_POST_SIZE_M / 2.0)
            post_y = y_centre + dy_sign * (half_depth - RACK_POST_SIZE_M / 2.0)
            bodies.append(
                create_static_box(
                    position=[post_x, post_y, RACK_HEIGHT_M / 2.0],
                    half_extents=[
                        RACK_POST_SIZE_M / 2.0,
                        RACK_POST_SIZE_M / 2.0,
                        RACK_HEIGHT_M / 2.0,
                    ],
                    color=RACK_FRAME_COLOR,
                    yaw_degrees=yaw_degrees,
                )
            )

    # --- shelf boards + goods, one row per level ---
    for level_index, shelf_z in enumerate(RACK_LEVEL_HEIGHTS_M):
        bodies.append(
            create_static_box(
                position=[bay_x, y_centre, shelf_z],
                half_extents=[half_length, half_depth, RACK_SHELF_THICKNESS_M / 2.0],
                color=RACK_SHELF_BOARD_COLOR,
                yaw_degrees=yaw_degrees,
            )
        )

        # Headroom available above this shelf board (up to the next
        # level, or the rack top for the highest level), used to size
        # goods so they fit under the shelf above.
        if level_index + 1 < len(RACK_LEVEL_HEIGHTS_M):
            headroom = RACK_LEVEL_HEIGHTS_M[level_index + 1] - shelf_z
        else:
            headroom = RACK_HEIGHT_M - shelf_z
        goods_half_height = max(0.08, min(0.22, headroom * 0.35))

        goods_half_length = (2.0 * half_length - 0.10) / (2.0 * RACK_GOODS_PER_LEVEL)
        goods_half_length = min(goods_half_length, 0.26)
        goods_half_depth = half_depth * 0.70

        for goods_index in range(RACK_GOODS_PER_LEVEL):
            slot_centre_x = (
                bay_x
                - half_length
                + (goods_index + 0.5) * (2.0 * half_length / RACK_GOODS_PER_LEVEL)
            )
            color = RACK_GOODS_COLORS[
                (level_index * RACK_GOODS_PER_LEVEL + goods_index + color_seed)
                % len(RACK_GOODS_COLORS)
            ]
            bodies.append(
                create_static_box(
                    position=[
                        slot_centre_x,
                        y_centre,
                        shelf_z + RACK_SHELF_THICKNESS_M / 2.0 + goods_half_height,
                    ],
                    half_extents=[goods_half_length, goods_half_depth, goods_half_height],
                    color=color,
                    yaw_degrees=yaw_degrees,
                )
            )

    return bodies


def create_rack_row(x_start, y_centre, length_m, yaw_degrees=0.0):
    """
    One continuous rack row (one side of a storage aisle), built as a
    series of open cupboard-style shelving bays (_create_rack_bay) with a
    visible gap between bays, matching a real pallet-rack aisle with
    multiple shelf levels of goods rather than one solid coloured block.
    """

    bay_pitch = RACK_BAY_LENGTH_M + RACK_BAY_GAP_M
    number_of_bays = max(1, int(length_m // bay_pitch))

    bodies = []

    for bay_index in range(number_of_bays):
        bay_x = x_start + bay_index * bay_pitch + RACK_BAY_LENGTH_M / 2.0
        bodies.extend(_create_rack_bay(bay_x, y_centre, yaw_degrees, bay_index))

    return bodies


def create_rack_aisle(aisle_index, x_start, length_m, aisle_centre_y):
    """
    One storage aisle: two facing rack rows separated by
    TRAVEL_AISLE_WIDTH_M of clear floor space for the AMR to drive
    through.
    """

    near_row_y = aisle_centre_y - (RACK_ROW_HALF_DEPTH_M + TRAVEL_AISLE_WIDTH_M / 2.0)
    far_row_y = aisle_centre_y + (RACK_ROW_HALF_DEPTH_M + TRAVEL_AISLE_WIDTH_M / 2.0)

    bodies = []
    bodies.extend(create_rack_row(x_start, near_row_y, length_m))
    bodies.extend(create_rack_row(x_start, far_row_y, length_m))

    return bodies


def create_storage_area():
    """
    Builds NUMBER_OF_RACK_AISLES (3) parallel storage aisles, evenly
    spaced across the warehouse width, occupying the high-X half of the
    building (matching "Storage" in the reference illustration, behind
    the dock band).
    """

    total_span = (NUMBER_OF_RACK_AISLES - 1) * RACK_AISLE_PITCH_M
    first_aisle_y = -total_span / 2.0

    all_bodies = []

    for aisle_index in range(NUMBER_OF_RACK_AISLES):
        aisle_centre_y = first_aisle_y + aisle_index * RACK_AISLE_PITCH_M

        all_bodies.extend(
            create_rack_aisle(
                aisle_index=aisle_index,
                x_start=RACK_AISLE_X_START,
                length_m=RACK_AISLE_LENGTH_M,
                aisle_centre_y=aisle_centre_y,
            )
        )

    return all_bodies


# ============================================================
# WORK STATIONS (robotic arm + conveyor extension)
# ============================================================

def _dock_zone_centre_x():
    """
    X position of the dock-zone floor markers / AMR stopping point,
    matching the computation in create_dock_zones() so station
    placement always lines up with the dock bay regardless of future
    changes to DOCK_BAND_DEPTH_M.
    """

    half_length = WAREHOUSE_LENGTH_M / 2.0
    return -half_length + DOCK_BAND_DEPTH_M / 2.0 + 1.0


def _dock_zone_y_centre(zone_name):

    for zone in DOCK_ZONES:
        if zone["name"] == zone_name:
            return zone["y_center"]

    raise ValueError(f"Unknown dock zone '{zone_name}'.")


def create_robotic_arm_station():
    """
    Loads robotic_arm_station.urdf into the PICKING dock bay. The AMR
    drives in along +X and stops centred under the pedestal at this same
    (x, y); see magnetic_docking.py for the docking logic once the AMR
    is in position.

    Returns the PyBullet body id of the station.
    """

    station_x = _dock_zone_centre_x()
    station_y = _dock_zone_y_centre(ARM_STATION_ZONE_NAME)

    station_id = p.loadURDF(
        str(SCRIPT_DIR / ARM_STATION_FILENAME),
        basePosition=[station_x, station_y, ARM_STATION_BASE_Z_M],
        baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
        useFixedBase=True,
    )

    return station_id


def create_conveyor_station():
    """
    Loads conveyor_station.urdf into the SHIPPING dock bay, yawed 180
    degrees so its outfeed end points toward the parked truck (which
    stands in -X, outside the building) while its intake end faces +X,
    toward the AMR's approach direction.

    Returns the PyBullet body id of the station.
    """

    station_x = _dock_zone_centre_x()
    station_y = _dock_zone_y_centre(CONVEYOR_STATION_ZONE_NAME)

    station_id = p.loadURDF(
        str(SCRIPT_DIR / CONVEYOR_STATION_FILENAME),
        basePosition=[station_x, station_y, CONVEYOR_STATION_BASE_Z_M],
        baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, math.pi]),
        useFixedBase=True,
    )

    return station_id


# ============================================================
# PAINTED LANE NETWORK (the task path the AMR follows / A* routes on)
# ============================================================

def _paint_lane_segment(p0, p1, width_m, color):
    """
    Paints one straight lane strip between world points p0 and p1 as a
    thin, non-colliding floor marker, yawed to the segment direction and
    extended by half a width at each end so consecutive segments overlap
    cleanly at corners (no gaps at the junctions).
    """

    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return []

    yaw = math.atan2(dy, dx)
    centre = [(x0 + x1) / 2.0, (y0 + y1) / 2.0, 0.012]

    visual_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[length / 2.0 + width_m / 2.0, width_m / 2.0, 0.002],
        rgbaColor=color,
    )

    body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual_shape,
        basePosition=centre,
        baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
    )

    return [body_id]


def create_lane_network():
    """
    Paints every polyline in cfg.LANE_NETWORK as connected floor strips.
    This is the physical, camera-visible lane the AMR follows with its
    centre RGB camera, and the same geometry the planner rasterises into
    its lane-cell mask (see occupancy_map.build_lane_mask) so A* can only
    route along painted lane.

    Returns a flat list of created marker body ids.
    """

    bodies = []

    for polyline in cfg.LANE_NETWORK:
        for p0, p1 in zip(polyline[:-1], polyline[1:]):
            bodies.extend(
                _paint_lane_segment(p0, p1, cfg.LANE_WIDTH_M, cfg.LANE_COLOR_RGBA)
            )

    return bodies


# ============================================================
# PROCEDURAL TEXTURES (so stereo matching has surface detail)
# ============================================================
# Flat-colour boxes are featureless to a stereo block matcher (every
# patch on a uniform face matches every other). Real cardboard, tables
# and truck decks have print, grain and scuffs; these small generated
# textures give the same kind of matchable detail while keeping each
# surface's mean colour (so colour segmentation still works).

ASSET_DIR = SCRIPT_DIR / "assets"


def _make_texture(name, bgr_mean, seed, noise=28, stripe_bgr=None):
    """Writes (once) and loads a 128x128 noise texture around bgr_mean.
    Returns a PyBullet texture id, or -1 if it could not be created."""

    path = ASSET_DIR / f"{name}.png"
    try:
        if not path.exists():
            import cv2
            import numpy as np

            ASSET_DIR.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(seed)
            base = np.array(bgr_mean, dtype=np.float32)[None, None, :]
            speckle = rng.normal(0.0, noise, size=(128, 128)).astype(np.float32)
            speckle = cv2.GaussianBlur(speckle, (3, 3), 0)[:, :, None]
            image = base + speckle
            if stripe_bgr is not None:
                image[56:72, :, :] = (np.array(stripe_bgr, dtype=np.float32)[None, None, :]
                                      + 0.5 * speckle[56:72])
            cv2.imwrite(str(path), np.clip(image, 0, 255).astype(np.uint8))
        return p.loadTexture(str(path))
    except Exception as exc:  # noqa: BLE001 - textures are cosmetic for the AMR
        print(f"[warehouse] texture '{name}' unavailable ({exc})")
        return -1


def _apply_texture(body_id, texture_id):
    if texture_id >= 0:
        p.changeVisualShape(body_id, -1, textureUniqueId=texture_id,
                            rgbaColor=[1.0, 1.0, 1.0, 1.0])


def _rgba_to_bgr255(rgba):
    return [rgba[2] * 255.0, rgba[1] * 255.0, rgba[0] * 255.0]


# ============================================================
# PICK TABLE AT POINT A (open-topped, suction access from above)
# ============================================================

def create_pick_table():
    """
    Open pick table at the end of the middle aisle, facing the AMR's
    approach along the aisle lane. Open-topped (no roof) so a suction cup
    can come straight down onto the boxes, with a low back lip so stock
    does not slide off the far side. Returns a list of body ids.
    """

    x0, x1 = cfg.PICK_TABLE_X_MIN, cfg.PICK_TABLE_X_MAX
    hw = cfg.PICK_TABLE_HALF_WIDTH_M
    top_z = cfg.PICK_TABLE_TOP_Z_M
    thick = cfg.PICK_TABLE_TOP_THICKNESS_M
    cx = 0.5 * (x0 + x1)
    cy = cfg.Y_AISLE
    color = cfg.PICK_TABLE_COLOR

    bodies = []
    top_id = create_static_box(
        position=[cx, cy, top_z - thick / 2.0],
        half_extents=[(x1 - x0) / 2.0, hw, thick / 2.0], color=color,
    )
    _apply_texture(top_id, _make_texture("pick_table", _rgba_to_bgr255(color), seed=11))
    bodies.append(top_id)

    leg = 0.03
    for lx in (x0 + leg, x1 - leg):
        for ly in (cy - hw + leg, cy + hw - leg):
            bodies.append(create_static_box(
                position=[lx, ly, (top_z - thick) / 2.0],
                half_extents=[leg, leg, (top_z - thick) / 2.0],
                color=[0.18, 0.18, 0.20, 1.0],
            ))
    # Back lip (far +X edge).
    bodies.append(create_static_box(
        position=[x1 - 0.01, cy, top_z + 0.03],
        half_extents=[0.01, hw, 0.03], color=[0.18, 0.18, 0.20, 1.0],
    ))
    return bodies


def create_boxes_at_a():
    """
    The small boxes to be picked (cfg.BOXES_AT_A), resting on the pick
    table. Dynamic bodies with real mass: the suction arm genuinely lifts,
    carries and sets them down. Returns a list of
    (box_body_id, (x, y, half_extent)) tuples (ground truth, for tests).
    """

    texture = _make_texture(
        "cardboard", _rgba_to_bgr255(cfg.BOX_COLOR_RGBA), seed=7, noise=22,
        stripe_bgr=[45, 95, 160],   # brown packing tape
    )

    created = []
    for (box_x, box_y, half) in cfg.BOXES_AT_A:
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half] * 3)
        visual_shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[half] * 3, rgbaColor=cfg.BOX_COLOR_RGBA,
        )
        box_id = p.createMultiBody(
            baseMass=cfg.BOX_MASS_KG,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[box_x, box_y, cfg.PICK_TABLE_TOP_Z_M + half + 0.001],
        )
        p.changeDynamics(box_id, -1, lateralFriction=0.8)
        _apply_texture(box_id, texture)
        created.append((box_id, (box_x, box_y, half)))
    return created


# ============================================================
# SHELF B + OVERSIZED BOX (conveyor job)
# ============================================================

def create_shelf_b():
    """Sturdy shelf unit at the end of aisle 0 (point B), its front edge
    facing the AMR's approach along the aisle-0 lane. Returns body ids."""

    x0, x1 = cfg.SHELF_B_X_MIN, cfg.SHELF_B_X_MAX
    hw = cfg.SHELF_B_HALF_WIDTH_M
    top = cfg.SHELF_B_TOP_Z_M
    slab = cfg.SHELF_B_SLAB_M
    cx, cy = 0.5 * (x0 + x1), cfg.Y_AISLE_0
    bodies = []
    top_id = create_static_box([cx, cy, top - slab / 2.0], [(x1 - x0) / 2.0, hw, slab / 2.0],
                               cfg.SHELF_B_COLOR)
    _apply_texture(top_id, _make_texture("shelf_b", _rgba_to_bgr255(cfg.SHELF_B_COLOR), seed=31))
    bodies.append(top_id)
    post = 0.03
    for px in (x0 + post, x1 - post):
        for py in (cy - hw + post, cy + hw - post):
            bodies.append(create_static_box([px, py, (top - slab) / 2.0],
                                            [post, post, (top - slab) / 2.0],
                                            [0.15, 0.15, 0.17, 1.0]))
    # Lower shelf board (looks like a real shelf unit, keeps it rigid).
    bodies.append(create_static_box([cx, cy, 0.12], [(x1 - x0) / 2.0, hw, 0.015],
                                    cfg.SHELF_B_COLOR))
    return bodies


def create_big_box():
    """One oversized carton on shelf B, overhanging the front edge by
    cfg.BIG_BOX_OVERHANG_M. Dynamic (real mass): the conveyor genuinely
    drags it aboard and pushes it off. Returns (body_id, half_extents)."""

    hx, hy, hz = cfg.BIG_BOX_HALF
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=cfg.BOX_COLOR_RGBA)
    bx, by = cfg.BIG_BOX_XY
    body = p.createMultiBody(cfg.BIG_BOX_MASS_KG, col, vis,
                             [bx, by, cfg.SHELF_B_TOP_Z_M + hz + 0.001])
    p.changeDynamics(body, -1, lateralFriction=0.5)
    _apply_texture(body, _make_texture(
        "cardboard", _rgba_to_bgr255(cfg.BOX_COLOR_RGBA), seed=7, noise=22,
        stripe_bgr=[45, 95, 160]))
    return body, (hx, hy, hz)


# ============================================================
# LANE OBSTACLES (typical warehouse clutter the AMR must detour around)
# ============================================================

def create_lane_obstacles():
    """
    Builds the static obstacles defined in cfg.LANE_OBSTACLES. They sit ON
    the painted lane with one wide detour side each, and have real
    collision, so the AMR has to perceive them, leave the lane around
    them and rejoin it. Returns a list of created body ids.
    """

    bodies = []
    for (ox, oy, hx, hy, hz, color, _label) in cfg.LANE_OBSTACLES:
        bodies.append(create_static_box(
            position=[ox, oy, hz], half_extents=[hx, hy, hz], color=color,
        ))
    return bodies


# ============================================================
# TRUCK AT THE SHIPPING DOCK (hollow trailer, rear door open)
# ============================================================

def create_truck_trailer():
    """
    A hollow box trailer backed up to the open SHIPPING dock bay: cargo
    deck at cfg.TRUCK_DECK_Z_M, side walls, front wall and a
    semi-transparent roof (so the loaded boxes stay visible in the 3D
    view), with the REAR open toward the warehouse. Its interior is the
    target the arm loads the boxes into. A cab sits beyond the front
    wall. Returns a dict of body ids.
    """

    rear = cfg.TRUCK_REAR_X
    front = rear - cfg.TRUCK_LENGTH_M
    cx = 0.5 * (rear + front)
    cy = cfg.TRUCK_Y
    hl = cfg.TRUCK_LENGTH_M / 2.0
    hw = cfg.TRUCK_HALF_WIDTH_M
    deck = cfg.TRUCK_DECK_Z_M
    wall = cfg.TRUCK_WALL_THICKNESS_M
    inner_h = cfg.TRUCK_INTERIOR_HEIGHT_M
    color = cfg.TRUCK_COLOR_RGBA

    bodies = []
    # Chassis under the deck.
    bodies.append(create_static_box(
        position=[cx, cy, deck / 2.0],
        half_extents=[hl, hw * 0.8, deck / 2.0 - 0.031],
        color=[0.15, 0.15, 0.16, 1.0],
    ))
    # Cargo deck (textured, so the arm's stereo camera can range it).
    deck_id = create_static_box(
        position=[cx, cy, deck - 0.03], half_extents=[hl, hw, 0.03],
        color=[0.55, 0.50, 0.45, 1.0],
    )
    _apply_texture(deck_id, _make_texture("truck_deck", [110, 125, 140], seed=23))
    bodies.append(deck_id)
    # Side walls.
    for sign in (-1.0, 1.0):
        bodies.append(create_static_box(
            position=[cx, cy + sign * (hw - wall / 2.0), deck + inner_h / 2.0],
            half_extents=[hl, wall / 2.0, inner_h / 2.0], color=color,
        ))
    # Front wall (cab end).
    bodies.append(create_static_box(
        position=[front + wall / 2.0, cy, deck + inner_h / 2.0],
        half_extents=[wall / 2.0, hw, inner_h / 2.0], color=color,
    ))
    # Roof - semi-transparent so the cargo inside stays visible.
    roof_vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[hl, hw, wall / 2.0],
        rgbaColor=[color[0], color[1], color[2], 0.30],
    )
    roof_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hl, hw, wall / 2.0])
    bodies.append(p.createMultiBody(
        baseMass=0.0, baseCollisionShapeIndex=roof_col,
        baseVisualShapeIndex=roof_vis,
        basePosition=[cx, cy, deck + inner_h + wall / 2.0],
    ))
    # Rear doors swung open flat against the outside of the side walls.
    for sign in (-1.0, 1.0):
        bodies.append(create_static_box(
            position=[rear - 0.45, cy + sign * (hw + 0.03), deck + inner_h / 2.0],
            half_extents=[0.45, 0.02, inner_h / 2.0], color=[0.30, 0.40, 0.85, 1.0],
        ))
    # Cab.
    bodies.append(create_static_box(
        position=[front - 0.85, cy, 1.0], half_extents=[0.75, 0.9, 1.0],
        color=[0.85, 0.85, 0.88, 1.0],
    ))

    return {"bodies": bodies, "deck_id": deck_id}


# ============================================================
# FULL ENVIRONMENT ASSEMBLY
# ============================================================

def create_environment(include_work_stations=True):
    """
    Builds the complete warehouse world: bounded floor, perimeter walls
    with open dock bays, the four dock zones (Reception, Loading &
    Unloading, Picking, Shipping - the latter with the hollow truck
    trailer backed up to it), the storage area's 3 rack aisles, the
    painted lane network, the obstacles on it, and the pick table with
    its boxes at point A.

    When include_work_stations is True (default, used by the manual
    teleop main.py), also loads the original fixed robotic arm station
    (PICKING bay) and conveyor extension (SHIPPING bay). The autonomous
    mission passes False and loads the arm-module stand itself.

    Returns a dict of created body-id groups.
    """

    floor_id = create_floor()
    wall_ids = create_perimeter_walls()
    dock_bodies = create_dock_zones()
    storage_bodies = create_storage_area()
    lane_bodies = create_lane_network()
    lane_obstacle_bodies = create_lane_obstacles()
    pick_table_bodies = create_pick_table()
    box_bodies = create_boxes_at_a()
    shelf_b_bodies = create_shelf_b()
    big_box = create_big_box()
    truck = create_truck_trailer()

    arm_station_id = None
    conveyor_station_id = None

    if include_work_stations:
        arm_station_id = create_robotic_arm_station()
        conveyor_station_id = create_conveyor_station()

    return {
        "floor": floor_id,
        "walls": wall_ids,
        "dock_markers": dock_bodies["markers"],
        "dock_trucks": dock_bodies["trucks"],
        "storage_racks": storage_bodies,
        "lane_markers": lane_bodies,
        "lane_obstacles": lane_obstacle_bodies,
        "pick_table": pick_table_bodies,
        "boxes_at_a": box_bodies,
        "shelf_b": shelf_b_bodies,
        "big_box": big_box,
        "truck": truck["bodies"],
        "truck_deck_id": truck["deck_id"],
        "point_a": cfg.POINT_A,
        "arm_station_id": arm_station_id,
        "conveyor_station_id": conveyor_station_id,
    }


# ============================================================
# STANDALONE PREVIEW
# ============================================================

def main():
    """
    Standalone preview: builds the warehouse world and holds the GUI open
    so the layout can be inspected without also loading a robot. Run
    main.py for the full robot + environment simulation.
    """

    print("=" * 60)
    print("INDOOR LOCALIZATION - WAREHOUSE ENVIRONMENT PREVIEW")
    print("=" * 60)

    initialize_simulation()

    environment = create_environment()

    print()
    print(f"Floor body id        : {environment['floor']}")
    print(f"Perimeter wall count  : {len(environment['walls'])}")
    print(f"Dock zone markers     : {[name for name, _ in environment['dock_markers']]}")
    print(f"Dock truck bodies     : {len(environment['dock_trucks'])}")
    print(f"Storage rack bodies   : {len(environment['storage_racks'])}")
    print(f"Lane marker strips    : {len(environment['lane_markers'])}")
    print(f"Lane obstacles        : {len(environment['lane_obstacles'])}")
    print(f"Pick table at A       : {len(environment['pick_table'])} bodies")
    print(f"Boxes at point A      : {len(environment['boxes_at_a'])} at {environment['point_a']}")
    print(f"Truck trailer         : {len(environment['truck'])} bodies, rear sill at "
          f"x={cfg.TRUCK_REAR_X}")
    print(f"Robotic arm station id: {environment['arm_station_id']}")
    print(f"Conveyor station id   : {environment['conveyor_station_id']}")
    print()
    print("Close the PyBullet window or press Ctrl+C to exit.")

    p.resetDebugVisualizerCamera(
        cameraDistance=18.0,
        cameraYaw=50,
        cameraPitch=-40,
        cameraTargetPosition=[2.0, 0.0, 0.0],
    )

    try:
        while True:
            p.stepSimulation()
    except KeyboardInterrupt:
        print()
        print("Preview stopped.")
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
