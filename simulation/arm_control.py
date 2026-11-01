"""
INDOOR LOCALIZATION - 6-DOF ARM KINEMATICS + CONTROL
========================================================

Thin control layer over robotic_arm_station.urdf's kinematic chain:

    pedestal_link
        -> shoulder_pan_joint   (revolute, Z)   -> shoulder_link
        -> shoulder_lift_joint  (revolute, Y)   -> upper_arm_link
        -> elbow_joint          (revolute, Y)   -> forearm_link
        -> wrist_1_joint        (revolute, Y)   -> wrist_1_link
        -> wrist_2_joint        (revolute, Z)   -> wrist_2_link
        -> wrist_3_joint        (revolute, X)   -> tool_flange_link

This module exposes exactly the operations needed to "play with inverse
kinematics" against waypoints defined later:

    - forward_kinematics(arm_id, joint_angles)   -> tool_flange_link pose
    - solve_ik(arm_id, target_position, target_orientation=None)
                                                  -> 6 joint angles
    - move_to_waypoint(arm_id, waypoint, ...)    -> drives the arm toward
                                                     a target end-effector
                                                     pose using PyBullet's
                                                     built-in IK solver
    - set_joint_angles(arm_id, joint_angles)     -> direct joint command
    - get_joint_angles(arm_id)                   -> current joint state

No waypoints or trajectories are defined here - this is only the
kinematics/control plumbing. Waypoint lists are expected to be supplied
by the caller later (e.g. a list of (x, y, z[, roll, pitch, yaw]) poses
in the pedestal_link's parent frame / world frame).

This file is part of the indoor_localization project only; nothing here
reads, imports, or depends on anything in the robot_vision folder.
"""

import pybullet as p


# ============================================================
# JOINT CHAIN DEFINITION
# ============================================================
# Order matches the URDF kinematic chain exactly. ARM_JOINT_NAMES is the
# single source of truth other modules should use to look joints up by
# name rather than hardcoding indices (indices depend on load order of
# fixed joints too, e.g. the magnet pads, so they are resolved at
# runtime via find_arm_joints()).

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

END_EFFECTOR_LINK_NAME = "tool_flange_link"

# Reasonable default pose: arm pointing forward and slightly down, clear
# of the pedestal legs so it does not need to be posed manually before
# first use.
DEFAULT_HOME_ANGLES_RAD = [0.0, -0.6, 1.0, -0.4, 0.0, 0.0]

# Joint motor force/velocity limits, taken directly from the <limit>
# effort/velocity values in robotic_arm_station.urdf so Python-side
# commands never exceed what the URDF declares the joint capable of.
ARM_JOINT_EFFORT_LIMIT_NM = {
    "shoulder_pan_joint": 150.0,
    "shoulder_lift_joint": 150.0,
    "elbow_joint": 100.0,
    "wrist_1_joint": 50.0,
    "wrist_2_joint": 50.0,
    "wrist_3_joint": 28.0,
}

IK_MAX_ITERATIONS = 200
IK_RESIDUAL_THRESHOLD = 1e-4

# Joint limits, matching robotic_arm_station.urdf's <limit> tags exactly.
# Passed explicitly to calculateInverseKinematics (lowerLimits/
# upperLimits/jointRanges/restPoses) because without them PyBullet's
# damped-least-squares solver is free to walk a joint across its full
# +-pi range to satisfy a small positional offset, landing on a
# physically absurd but numerically "valid" pose (e.g. spinning
# shoulder_pan_joint by ~180 degrees for a 5 cm sideways target). Rest
# poses are seeded from DEFAULT_HOME_ANGLES_RAD so the solver is biased
# toward small, local joint motion rather than a distant alternate
# solution branch.
ARM_JOINT_LOWER_LIMITS_RAD = [-3.14159, -3.14159, -2.87979, -3.14159, -3.14159, -3.14159]
ARM_JOINT_UPPER_LIMITS_RAD = [3.14159, 3.14159, 2.87979, 3.14159, 3.14159, 3.14159]
ARM_JOINT_RANGES_RAD = [
    upper - lower
    for lower, upper in zip(ARM_JOINT_LOWER_LIMITS_RAD, ARM_JOINT_UPPER_LIMITS_RAD)
]
ARM_JOINT_DAMPING = [0.1] * 6


# ============================================================
# JOINT / LINK DISCOVERY
# ============================================================

def find_arm_joints(arm_id):
    """
    Resolve ARM_JOINT_NAMES and END_EFFECTOR_LINK_NAME to PyBullet joint
    indices for a loaded robotic_arm_station.urdf body.

    Returns:
        (joint_indices, end_effector_link_index)
        joint_indices : list[int], length 6, in ARM_JOINT_NAMES order.
        end_effector_link_index : int, the link index whose joint output
            frame coincides with tool_flange_link's origin (PyBullet
            indexes links by their parent joint index).
    """

    name_to_index = {}

    for joint_index in range(p.getNumJoints(arm_id)):
        joint_info = p.getJointInfo(arm_id, joint_index)
        joint_name = joint_info[1].decode("utf-8")
        child_link_name = joint_info[12].decode("utf-8")
        name_to_index[joint_name] = joint_index
        name_to_index[f"__link__{child_link_name}"] = joint_index

    missing = [name for name in ARM_JOINT_NAMES if name not in name_to_index]
    if missing:
        raise RuntimeError(f"Missing arm joints: {missing}")

    joint_indices = [name_to_index[name] for name in ARM_JOINT_NAMES]

    end_effector_key = f"__link__{END_EFFECTOR_LINK_NAME}"
    if end_effector_key not in name_to_index:
        raise RuntimeError(
            f"Missing end-effector link: {END_EFFECTOR_LINK_NAME}"
        )
    end_effector_link_index = name_to_index[end_effector_key]

    return joint_indices, end_effector_link_index


# ============================================================
# JOINT STATE
# ============================================================

def get_joint_angles(arm_id, joint_indices):
    """Current angle (radians) of each of the 6 arm joints, in order."""

    states = p.getJointStates(arm_id, joint_indices)
    return [state[0] for state in states]


def set_joint_angles(arm_id, joint_indices, joint_angles, max_velocity=1.5):
    """
    Direct joint-space command: drives each joint toward joint_angles[i]
    using POSITION_CONTROL, respecting that joint's URDF effort limit.

    This is the lowest-level control entry point. move_to_waypoint()
    builds on top of this by first solving IK for a desired end-effector
    pose and then calling this function with the result.
    """

    if len(joint_angles) != len(joint_indices):
        raise ValueError(
            f"Expected {len(joint_indices)} joint angles, got "
            f"{len(joint_angles)}."
        )

    for joint_index, target_angle, joint_name in zip(
        joint_indices, joint_angles, ARM_JOINT_NAMES
    ):
        p.setJointMotorControl2(
            bodyUniqueId=arm_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_angle,
            maxVelocity=max_velocity,
            force=ARM_JOINT_EFFORT_LIMIT_NM[joint_name],
        )


def reset_joint_angles(arm_id, joint_indices, joint_angles=None):
    """
    Instantly teleports the arm to joint_angles (default: the home pose)
    with zero velocity, bypassing the motor controller. Useful for
    initialising the arm's pose once at startup before handing control
    over to set_joint_angles()/move_to_waypoint().
    """

    joint_angles = joint_angles or DEFAULT_HOME_ANGLES_RAD

    for joint_index, angle in zip(joint_indices, joint_angles):
        p.resetJointState(arm_id, joint_index, targetValue=angle)


# ============================================================
# FORWARD KINEMATICS
# ============================================================

def forward_kinematics(arm_id, end_effector_link_index):
    """
    Current world-frame position and orientation (quaternion) of
    tool_flange_link, computed by PyBullet from the arm's current joint
    state (i.e. whatever was last commanded via set_joint_angles /
    reset_joint_angles, after a stepSimulation or resetJointState has
    taken effect).

    Returns:
        (position_xyz, orientation_quaternion_xyzw)
    """

    link_state = p.getLinkState(
        arm_id, end_effector_link_index, computeForwardKinematics=True
    )
    world_position = link_state[4]
    world_orientation = link_state[5]

    return world_position, world_orientation


# ============================================================
# INVERSE KINEMATICS
# ============================================================

def solve_ik(
    arm_id,
    joint_indices,
    end_effector_link_index,
    target_position,
    target_orientation=None,
):
    """
    Solves for the 6 joint angles that place tool_flange_link at
    target_position (world frame, metres) and, optionally,
    target_orientation (world frame quaternion xyzw).

    Uses PyBullet's built-in damped-least-squares IK solver
    (calculateInverseKinematics), seeded with the arm's current joint
    state so repeated calls across a waypoint sequence converge smoothly
    rather than jumping between unrelated solution branches.

    Returns:
        list[float], length 6, joint angles in ARM_JOINT_NAMES order.
    """

    current_angles = get_joint_angles(arm_id, joint_indices)

    # Rest pose biases the solver toward whichever configuration is
    # "closest" to current, rather than toward DEFAULT_HOME_ANGLES_RAD
    # unconditionally - this keeps repeated waypoint calls stable (each
    # new solve prefers small motion from where the arm already is).
    rest_poses = current_angles

    ik_kwargs = dict(
        bodyUniqueId=arm_id,
        endEffectorLinkIndex=end_effector_link_index,
        targetPosition=target_position,
        lowerLimits=ARM_JOINT_LOWER_LIMITS_RAD,
        upperLimits=ARM_JOINT_UPPER_LIMITS_RAD,
        jointRanges=ARM_JOINT_RANGES_RAD,
        restPoses=rest_poses,
        jointDamping=ARM_JOINT_DAMPING,
        maxNumIterations=IK_MAX_ITERATIONS,
        residualThreshold=IK_RESIDUAL_THRESHOLD,
    )

    if target_orientation is not None:
        ik_kwargs["targetOrientation"] = target_orientation

    ik_solution = p.calculateInverseKinematics(**ik_kwargs)

    # calculateInverseKinematics returns an angle for every movable joint
    # in the body (here: 6 arm joints only, since the magnet pads and
    # mast-equivalent joints on this URDF are all `fixed`), in body joint
    # order. Re-select just the 6 arm joints defensively in case future
    # revisions of the URDF add more movable joints.
    movable_joint_indices = [
        i for i in range(p.getNumJoints(arm_id))
        if p.getJointInfo(arm_id, i)[2] != p.JOINT_FIXED
    ]
    solution_by_joint = dict(zip(movable_joint_indices, ik_solution))

    return [solution_by_joint[joint_index] for joint_index in joint_indices]


# ============================================================
# WAYPOINT-LEVEL API
# ============================================================

def move_to_waypoint(
    arm_id,
    joint_indices,
    end_effector_link_index,
    waypoint,
    max_velocity=1.5,
):
    """
    The single call a waypoint-playback loop needs: given one waypoint
    as either

        (x, y, z)                       - position only, free orientation
        (x, y, z, qx, qy, qz, qw)       - position + orientation

    solves IK for that end-effector pose and commands the arm's 6 joints
    toward the solution. Does not block or step the simulation - call
    p.stepSimulation() in the caller's own loop as usual.

    Returns the 6 joint angles that were commanded, so a caller can log
    or compare against forward_kinematics() after stepping.
    """

    if len(waypoint) == 3:
        target_position = waypoint
        target_orientation = None
    elif len(waypoint) == 7:
        target_position = waypoint[0:3]
        target_orientation = waypoint[3:7]
    else:
        raise ValueError(
            "Waypoint must be (x, y, z) or (x, y, z, qx, qy, qz, qw), "
            f"got {len(waypoint)} values."
        )

    joint_angles = solve_ik(
        arm_id,
        joint_indices,
        end_effector_link_index,
        target_position,
        target_orientation,
    )

    set_joint_angles(arm_id, joint_indices, joint_angles, max_velocity)

    return joint_angles


# ============================================================
# STANDALONE SELF-TEST
# ============================================================

def main():
    """
    Minimal standalone check: loads robotic_arm_station.urdf alone
    (DIRECT mode, no GUI), resets it to its home pose, reads back its
    forward kinematics, then solves IK for a point just in front of that
    home pose and confirms the solver drives the end effector toward it.
    Run this file directly to sanity-check the kinematics without
    needing the full warehouse + AMR scene.
    """

    import pybullet_data

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

    arm_id = p.loadURDF(
        "robotic_arm_station.urdf", basePosition=[0, 0, 0], useFixedBase=True
    )

    joint_indices, end_effector_link_index = find_arm_joints(arm_id)
    print("Arm joint indices:", joint_indices)
    print("End-effector link index:", end_effector_link_index)

    reset_joint_angles(arm_id, joint_indices)
    position, orientation = forward_kinematics(arm_id, end_effector_link_index)
    print(f"Home pose position: {position}")

    target_position = (position[0] + 0.10, position[1], position[2] - 0.05)
    print(f"Target position: {target_position}")

    for _ in range(240):
        move_to_waypoint(
            arm_id, joint_indices, end_effector_link_index, target_position
        )
        p.stepSimulation()

    final_position, _ = forward_kinematics(arm_id, end_effector_link_index)
    print(f"Final position after IK convergence: {final_position}")

    error = sum(
        (a - b) ** 2 for a, b in zip(final_position, target_position)
    ) ** 0.5
    print(f"Position error: {error:.4f} m")

    p.disconnect()


if __name__ == "__main__":
    main()
