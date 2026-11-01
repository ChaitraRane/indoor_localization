# Indoor Logistics AMR - Specification Reference

Source: "AMR Power, Energy and Runtime Validation Report" (simulation-based
MATLAB validation of the proposed AMR), supplied by the user. All values
below are taken directly from that report and drive the parameters used in
`amr_robot.urdf`, `warehouse_environment.py` and `main.py`.

## Mass

| Parameter          | Value  |
|---------------------|--------|
| Empty AMR mass       | 60 kg  |
| Payload              | 20 kg  |
| Design operating mass| 80 kg  |

`amr_robot.urdf` link masses sum to ~59.9 kg (empty mass). The 20 kg
payload is added at runtime in `main.py` (`attach_payload`) as a separate
rigid body fixed to the chassis, rather than baked into the URDF, so the
"empty" and "operating mass" configurations stay independently inspectable.

## Drive system

| Parameter        | Value                          |
|--------------------|---------------------------------|
| Drive motors       | 2 x 24 V geared motors          |
| Motor rated speed  | 100 RPM                         |
| Motor rated torque | 10 Nm                           |
| Motor rated current| 8 A                             |
| Wheel diameter     | 200 mm (100 mm radius)          |
| Target speed       | 1.0 m/s (3.6 km/h)              |

100 RPM at the wheel converts to ~10.47 rad/s, which at a 0.10 m wheel
radius gives a theoretical top speed of ~1.047 m/s - consistent with the
report's 1.0 m/s target. `main.py` uses this rated angular speed directly
as the commanded "FORWARD" wheel velocity, and the rated 10 Nm torque as
the PyBullet velocity-controller force limit per wheel joint.

## Sensors

| Subsystem              | Qty | Model / basis                         |
|--------------------------|-----|----------------------------------------|
| Edge computer            | 1   | Raspberry Pi 5                         |
| RGB-D / stereo camera    | 2   | Luxonis OAK-D Lite class               |
| RGB camera               | 1   | 5V camera module                       |
| IMU                      | 1   | BNO085-class                           |
| Incremental encoder      | 2   | one per drive wheel                    |
| Motor driver             | 1   | dual-channel, 90% efficiency assumed   |
| Battery                  | 1   | 25.6 V, 20 Ah LiFePO4, 512 Wh nominal  |

All sensor links in `amr_robot.urdf` (left/right stereo camera, RGB
camera, IMU) are modelled as small fixed-joint boxes mounted on a
`sensor_mast` link above the chassis deck, so the stereo pair has a clear
forward view over low shelving. Encoders are not separate URDF links -
they are implicit at the two drive wheel joints, matching how
`robot_vision`'s own stage pipeline treats wheel feedback as a joint-state
reading rather than a modelled body.

## Not modelled in this phase

The battery, motor driver, and edge computer are physical specs used only
for drive-force/speed sizing in this phase; they are not represented as
separate URDF links (their mass is folded into the chassis `base_link`
inertial block). If later work needs them as individually addressable
bodies (e.g. for thermal or wiring layout), add them as additional fixed
links off `base_link` without changing the existing link names used by
`main.py`.

## Chassis-top magnet pads (AMR side)

Added to `amr_robot.urdf` to support docking under the work stations
below: `magnet_pad_front_link` and `magnet_pad_rear_link`, two 0.05 m
radius / 0.02 m thick cylinders on the chassis top deck (local z=0.16),
offset +-0.25 m along X from the chassis centre. Two pads rather than one
central pad so a docked station is held at both ends of the AMR and
cannot rock about a single attachment point. Combined mass 0.6 kg,
negligible against the 60 kg empty-mass target.

## Work station 1: stationary 6-DOF robotic arm (`robotic_arm_station.urdf`)

Fixed-base manipulator on a gantry pedestal, placed in the PICKING dock
bay (see `warehouse_environment.py`'s `create_robotic_arm_station()`).
The AMR drives underneath the pedestal and docks via two electromagnet
pads; the arm then rides on top of the AMR while it works, with the
magnets resisting the tipping moment the arm's own motion creates.

| Property                | Value                                        |
|---------------------------|-----------------------------------------------|
| Joint count               | 6, all revolute                               |
| Joint chain                | shoulder_pan -> shoulder_lift -> elbow -> wrist_1 -> wrist_2 -> wrist_3 |
| End-effector frame        | `tool_flange_link`                            |
| Pedestal clearance height  | 1.10 m (legs), AMR passes underneath          |
| Magnet pads                | `magnet_pad_front_link`, `magnet_pad_rear_link`, +-0.25 m along X |

No gripper/tool model is attached to `tool_flange_link` - waypoints
target that frame's origin directly. Kinematics and joint commands are
in `arm_control.py` (`find_arm_joints`, `forward_kinematics`, `solve_ik`,
`move_to_waypoint`, `set_joint_angles`). Waypoint lists themselves are
left to be defined later, per the request that only the kinematics setup
be done in this phase.

## Work station 2: loading conveyor extension (`conveyor_station.urdf`)

Fixed conveyor bed in the SHIPPING dock bay (`create_conveyor_station()`
in `warehouse_environment.py`), used to move goods between the AMR deck
and a parked truck. No vertical lift mechanism is modelled, per request.
Docks to the AMR the same way as the arm station: two electromagnet pads
at the intake end lock onto the AMR's two chassis-top pads.

| Property              | Value                                      |
|-------------------------|----------------------------------------------|
| Bed length              | 2.20 m                                       |
| Bed top height           | 0.30 m (matches AMR chassis deck top)        |
| Drive joints             | `infeed_roller_joint`, `outfeed_roller_joint` (continuous) |
| Roller radius            | 0.045 m                                      |
| Default belt speed       | 0.5 m/s (not specified in the validation report; a conservative light-goods default) |
| Magnet pads              | `magnet_pad_intake_link`, `magnet_pad_outfeed_link`, +-0.25 m along X |

Belt motion is driven by spinning both end rollers at the same angular
velocity (`conveyor_control.py`'s `command_conveyor`/`set_conveyor_speed`
- FORWARD/BACKWARD/STOP); there is no separate scrolling-texture belt
body, since PyBullet has no native primitive for that. The visible belt
surface panel between the rollers is a fixed, non-driven visual link.

## Magnetic docking (`magnetic_docking.py`)

PyBullet has no magnetic-field primitive, so the electromagnet behaviour
described by the user ("once the AMR comes below it, the electromagnet
will kick in") is modelled as a rigid constraint
(`p.createConstraint`/`JOINT_FIXED`) created between a station pad link
and the AMR's matching pad link, only when commanded on AND the two pads
are within `MAGNET_ENGAGE_DISTANCE_M` (0.08 m) of each other. The front
and rear pads of a station are always engaged/disengaged together
(`DockingPair`) - this module deliberately does not expose a way to
engage only one pad, since a single-point lock is exactly the "will
topple" failure mode the user asked to avoid.

## Swappable suction-arm module (`suction_arm_module.urdf`)

Tool module the AMR collects from the arm-module stand when the boxes it
found need a suction gripper. It sits on the rear/centre of the deck,
behind the sensor mast (so the forward cameras keep a clear view while it
is carried), and locks on with two electromagnet pads that mate with the
AMR's `magnet_pad_rear_link` and the new `magnet_pad_mid_link`
(`magnetic_docking.build_arm_module_docking_pair`). The module frame is
the AMR base frame shifted up to the pad tops (z = 0.32 m in the world).

| Property                 | Value                                              |
|--------------------------|----------------------------------------------------|
| Plate                    | 0.50 x 0.60 x 0.03 m (x from -0.36 to +0.14)       |
| Arm                      | 6 revolute joints, same names as the old station   |
| Links                    | column 0.25 m, upper arm 0.50 m, forearm 0.45 m, wrist 0.08 + 0.07 m |
| Gripper                  | vacuum cup, lip 0.11 m beyond the tool flange      |
| Eye-in-hand camera       | stereo pair on the flange, 0.075 m baseline, 320 x 240, 60 deg vertical FOV |
| Cargo tray               | behind the column, 2 slots x 2 layers for 0.16 m boxes |
| Mass                     | ~10 kg + up to 4 x 1 kg boxes                      |

Kinematics, closed-form top-down IK, stereo ranging, the vacuum gripper and
the pick/place routines are in `arm_manipulator.py`; its self-test checks
the FK against PyBullet's link poses, the IK at every mission pose, and the
stereo ranging of a box.

## Arm-module stand (`arm_station_stand.urdf`, PICKING dock)

Gantry (legs at x = +-0.40 m, y = +-0.50 m, 1.70 m tall) with a prismatic
hoist that holds the module 6 cm above the AMR's pads. The AMR reverses in
between the legs (0.15 m clearance per side), the hoist lowers the module,
the magnets lock, and the hoist lifts clear. Only the stand's final
location (`mission_config.ARM_STATION_XY`) is given to the mission.

## Truck trailer (SHIPPING dock)

Hollow box trailer backed up to the open SHIPPING bay, rear door open
toward the warehouse, cargo deck at 0.40 m, semi-transparent roof. The
mission is given only the boarding point (`TRUCK_BOARDING_XY`); the arm
measures the deck height and the sill with its stereo camera before
placing the boxes.

## Swappable conveyor module (`conveyor_module.urdf`, `conveyor_hardware.py`)

Collected from its own stand (`conveyor_station_stand.urdf`, RECEPTION dock)
when the box found is too big for the suction arm (footprint > 0.40 m). Same
plate, magnet pads and docking as the arm module.

| Property            | Value                                                     |
|---------------------|-----------------------------------------------------------|
| Belt                | 0.80 x 0.60 m, nose overhangs the AMR's rear by 0.27 m     |
| Lift                | prismatic, belt top 0.42 - 0.92 m above the floor          |
| Belt drive          | traction on objects touching the belt, 0.25 m/s, <= 60 N   |
| IR beam sensors     | across the belt 7 cm in from the nose, and at the front stop |
| ToF range sensor    | under the nose, facing back (gap to a shelf edge)          |
| Footprint aboard    | 0.42 m half-width (lift guides) - the planners widen accordingly |

Load: measure the shelf height with the forward RGB-D, set the belt 12 mm
below it, reverse the nose under the box's overhanging edge (ToF stop ~1.5 cm
from the shelf), raise until the belt touches the box, run the belt in until
the front IR beam is blocked and the nose beam is clear. Unload: trailer
landmark fix, belt 3 cm above the measured deck, nose 15 cm over the deck,
belt out until both beams have been clear for 1.8 s.
