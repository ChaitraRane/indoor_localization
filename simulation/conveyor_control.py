"""
INDOOR LOCALIZATION - CONVEYOR BELT CONTROL
===============================================

Thin command interface over conveyor_station.urdf's two driven rollers:

    infeed_roller_joint    (continuous, axis Y)
    outfeed_roller_joint   (continuous, axis Y)

Driving both rollers at the same commanded angular velocity is what
"moves the belt" - see conveyor_station.urdf's header comment for why
there is no separate scrolling-belt body. This module gives that a
simple command surface:

    FORWARD   - carries goods from the AMR/intake end toward the
                outfeed end (toward the truck)
    BACKWARD  - carries goods the other way (truck toward AMR)
    STOP      - rollers halt
    set_conveyor_speed(..., speed_mps) - arbitrary signed belt speed

This file is part of the indoor_localization project only; nothing here
reads, imports, or depends on anything in the robot_vision folder.
"""

import pybullet as p


# ============================================================
# JOINT CHAIN DEFINITION
# ============================================================

CONVEYOR_ROLLER_JOINT_NAMES = [
    "infeed_roller_joint",
    "outfeed_roller_joint",
]

# Roller radius from conveyor_station.urdf's cylinder geometry
# (<cylinder radius="0.045" .../> on both roller links), used to convert
# a desired belt surface speed (m/s) into roller angular velocity
# (rad/s), the same way amr_robot.urdf's wheel radius is used in
# main.py's command_to_wheel_speeds().
ROLLER_RADIUS_M = 0.045

# Rated belt speed. Not specified in the AMR validation report (the
# conveyor is a warehouse fixture, not part of the AMR itself); 0.5 m/s
# is a conservative, typical light-goods conveyor speed used here as the
# default FORWARD/BACKWARD rate. Override via set_conveyor_speed() for
# anything else.
DEFAULT_BELT_SPEED_MPS = 0.5

MAX_ROLLER_MOTOR_FORCE_NM = 15.0


# ============================================================
# JOINT DISCOVERY
# ============================================================

def find_conveyor_roller_joints(conveyor_id):
    """
    Resolve CONVEYOR_ROLLER_JOINT_NAMES to PyBullet joint indices for a
    loaded conveyor_station.urdf body.

    Returns:
        (infeed_joint_index, outfeed_joint_index)
    """

    name_to_index = {}

    for joint_index in range(p.getNumJoints(conveyor_id)):
        joint_info = p.getJointInfo(conveyor_id, joint_index)
        joint_name = joint_info[1].decode("utf-8")
        name_to_index[joint_name] = joint_index

    missing = [
        name for name in CONVEYOR_ROLLER_JOINT_NAMES if name not in name_to_index
    ]
    if missing:
        raise RuntimeError(f"Missing conveyor roller joints: {missing}")

    return (
        name_to_index["infeed_roller_joint"],
        name_to_index["outfeed_roller_joint"],
    )


# ============================================================
# ROLLER VELOCITY CONTROL
# ============================================================

def set_roller_angular_velocity(
    conveyor_id, infeed_joint, outfeed_joint, angular_velocity_rad_s
):
    """
    Drives both rollers at the same angular velocity (rad/s) so the belt
    surface moves as one unit rather than the two ends fighting each
    other. Positive = belt surface moves from infeed toward outfeed
    (AMR/intake end toward the truck/outfeed end).
    """

    for joint_index in (infeed_joint, outfeed_joint):
        p.setJointMotorControl2(
            bodyUniqueId=conveyor_id,
            jointIndex=joint_index,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=angular_velocity_rad_s,
            force=MAX_ROLLER_MOTOR_FORCE_NM,
        )


def set_conveyor_speed(conveyor_id, infeed_joint, outfeed_joint, speed_mps):
    """
    Commands the belt to move at speed_mps (signed; positive = toward
    outfeed/truck end, negative = toward infeed/AMR end), converting the
    linear belt speed to roller angular velocity via ROLLER_RADIUS_M.
    """

    angular_velocity_rad_s = speed_mps / ROLLER_RADIUS_M

    set_roller_angular_velocity(
        conveyor_id, infeed_joint, outfeed_joint, angular_velocity_rad_s
    )


# ============================================================
# COMMAND-LEVEL API
# ============================================================

def command_conveyor(
    conveyor_id,
    infeed_joint,
    outfeed_joint,
    command,
    speed_mps=DEFAULT_BELT_SPEED_MPS,
):
    """
    High-level command surface, mirroring main.py's
    command_to_wheel_speeds() style for the AMR drivetrain:

        "FORWARD"  -> belt moves intake/AMR end -> outfeed/truck end
        "BACKWARD" -> belt moves outfeed/truck end -> intake/AMR end
        "STOP"     -> rollers halt

    speed_mps sets the magnitude for FORWARD/BACKWARD (default
    DEFAULT_BELT_SPEED_MPS); the sign is applied internally so callers
    only need to supply a positive number.
    """

    if command == "FORWARD":
        set_conveyor_speed(conveyor_id, infeed_joint, outfeed_joint, abs(speed_mps))
    elif command == "BACKWARD":
        set_conveyor_speed(conveyor_id, infeed_joint, outfeed_joint, -abs(speed_mps))
    elif command == "STOP":
        set_conveyor_speed(conveyor_id, infeed_joint, outfeed_joint, 0.0)
    else:
        raise ValueError(
            f"Unknown conveyor command '{command}'. "
            "Expected FORWARD, BACKWARD or STOP."
        )


def get_roller_angular_velocity(conveyor_id, infeed_joint):
    """Current measured angular velocity (rad/s) of the infeed roller,
    useful for confirming a command actually took effect."""

    state = p.getJointState(conveyor_id, infeed_joint)
    return state[1]


# ============================================================
# STANDALONE SELF-TEST
# ============================================================

def main():
    """
    Minimal standalone check: loads conveyor_station.urdf alone (DIRECT
    mode, no GUI), issues FORWARD/BACKWARD/STOP commands and confirms
    the measured roller angular velocity tracks each command.
    """

    import pybullet_data

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

    conveyor_id = p.loadURDF(
        "conveyor_station.urdf", basePosition=[0, 0, 0], useFixedBase=True
    )

    infeed_joint, outfeed_joint = find_conveyor_roller_joints(conveyor_id)
    print(f"infeed_joint={infeed_joint} outfeed_joint={outfeed_joint}")

    for command in ("FORWARD", "BACKWARD", "STOP"):
        command_conveyor(conveyor_id, infeed_joint, outfeed_joint, command)

        for _ in range(60):
            p.stepSimulation()

        measured = get_roller_angular_velocity(conveyor_id, infeed_joint)
        expected_sign = {"FORWARD": 1, "BACKWARD": -1, "STOP": 0}[command]
        print(
            f"{command:>8} -> measured angular velocity = "
            f"{measured:+.3f} rad/s (expected sign: {expected_sign:+d})"
        )

    p.disconnect()


if __name__ == "__main__":
    main()
