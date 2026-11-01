"""
INDOOR LOCALIZATION - MAIN SIMULATION ENTRY POINT
====================================================

Loads the warehouse_environment world together with the amr_robot.urdf
logistics robot, drives it with simple keyboard teleop, and previews the
robot's onboard cameras (left/right stereo pair + centre RGB camera).

This is the indoor-localization counterpart to robot_vision's
stage1_simulation.py: same overall shape (initialize -> build world ->
load robot -> teleop loop -> camera preview), but written independently
for the warehouse AMR and its larger sensor payload. Nothing in the
robot_vision folder is read, imported, or modified by this script.

Run with:
    python main.py
"""

import math
import time
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import pybullet_data

import arm_control
import conveyor_control
import magnetic_docking
import warehouse_environment as warehouse


# ============================================================
# CONFIGURATION
# ============================================================

SIMULATION_HZ = 240

# Robot spawn point: inside the main travel corridor, in front of the
# LOADING_UNLOADING dock bay, facing down the warehouse toward storage.
ROBOT_START_POSITION = [-8.0, -2.2, 0.15]
ROBOT_START_ORIENTATION = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

# ------------------------------------------------------------
# DRIVE / MOTOR
# ------------------------------------------------------------
# Validation-report drive specs: 2 x 24V geared motors, 100 RPM, 10 Nm,
# 8A rated; wheel radius 0.10 m (200 mm diameter); target speed 1.0 m/s.
#
# 100 RPM at the wheel = 100 * 2*pi / 60 rad/s =~ 10.47 rad/s, which at a
# 0.10 m wheel radius gives a top ground speed of ~1.047 m/s - consistent
# with the report's 1.0 m/s target speed. That no-load top angular speed
# is used directly as the commanded wheel velocity for "FORWARD".
WHEEL_RADIUS_M = 0.10
MOTOR_RATED_RPM = 100.0
MOTOR_RATED_TORQUE_NM = 10.0

WHEEL_MAX_ANGULAR_VELOCITY_RAD_S = MOTOR_RATED_RPM * 2.0 * math.pi / 60.0
WHEEL_TURN_ANGULAR_VELOCITY_RAD_S = WHEEL_MAX_ANGULAR_VELOCITY_RAD_S * 0.6

# Motor force limit passed to PyBullet's velocity controller, derived
# from rated torque per motor (10 Nm) rather than an arbitrary constant.
MAX_MOTOR_FORCE_NM = MOTOR_RATED_TORQUE_NM

# ------------------------------------------------------------
# PAYLOAD
# ------------------------------------------------------------
# 20 kg payload box, carried on top of the chassis deck, bringing the
# robot from the 60 kg empty mass (baked into amr_robot.urdf) up to the
# validation report's 80 kg operating mass.
PAYLOAD_MASS_KG = 20.0
PAYLOAD_HALF_EXTENTS_M = [0.26, 0.20, 0.10]
PAYLOAD_LOCAL_OFFSET = [0.0, 0.0, 0.25]

# ------------------------------------------------------------
# CAMERAS
# ------------------------------------------------------------
# Three onboard cameras are previewed: left stereo, right stereo and the
# centre RGB camera, matching the sensor layout described in
# amr_robot.urdf / the validation report (2x OAK-D Lite stereo + 1x RGB).
CAMERA_WIDTH = 480
CAMERA_HEIGHT = 360

CAMERA_FOV = 65.0
CAMERA_NEAR = 0.05
CAMERA_FAR = 20.0

CAMERA_LOOK_DISTANCE = 2.0

# Camera link offsets relative to base_link, mirroring amr_robot.urdf's
# sensor_mast_joint + per-camera joint origins (mast origin + camera
# origin, summed once here so the camera pose does not need a full
# forward-kinematics walk through PyBullet link states).
SENSOR_MAST_OFFSET = np.array([0.28, 0.0, 0.24])
LEFT_STEREO_OFFSET = SENSOR_MAST_OFFSET + np.array([0.03, 0.09, 0.07])
RIGHT_STEREO_OFFSET = SENSOR_MAST_OFFSET + np.array([0.03, -0.09, 0.07])
RGB_CAMERA_OFFSET = SENSOR_MAST_OFFSET + np.array([0.03, 0.0, 0.11])


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ROBOT_URDF = SCRIPT_DIR / "amr_robot.urdf"


# ============================================================
# SIMULATION INITIALIZATION
# ============================================================

def initialize_simulation():

    physics_client = p.connect(p.GUI)

    if physics_client < 0:
        raise RuntimeError("Could not connect to PyBullet.")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.setGravity(0, 0, -9.81)

    p.setTimeStep(1.0 / SIMULATION_HZ)

    return physics_client


# ============================================================
# ROBOT
# ============================================================

def load_robot():

    robot_id = p.loadURDF(
        str(ROBOT_URDF),
        basePosition=ROBOT_START_POSITION,
        baseOrientation=ROBOT_START_ORIENTATION,
        useFixedBase=False,
    )

    return robot_id


def attach_payload(robot_id):
    """
    Rigidly attaches a 20 kg payload box to the chassis deck via a fixed
    joint, bringing total robot mass to the validation report's 80 kg
    operating mass without editing amr_robot.urdf's own empty-mass
    inertial values. The fixed joint carries the load, so payload-to-AMR
    collisions are disabled to prevent overlapping link geometry from
    fighting the constraint; payload collisions with the warehouse remain on.
    """

    collision_shape = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=PAYLOAD_HALF_EXTENTS_M
    )

    visual_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=PAYLOAD_HALF_EXTENTS_M,
        rgbaColor=[0.65, 0.45, 0.25, 1.0],
    )

    payload_id = p.createMultiBody(
        baseMass=PAYLOAD_MASS_KG,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[
            ROBOT_START_POSITION[0] + PAYLOAD_LOCAL_OFFSET[0],
            ROBOT_START_POSITION[1] + PAYLOAD_LOCAL_OFFSET[1],
            ROBOT_START_POSITION[2] + PAYLOAD_LOCAL_OFFSET[2],
        ],
    )

    p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=-1,
        childBodyUniqueId=payload_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=PAYLOAD_LOCAL_OFFSET,
        childFramePosition=[0, 0, 0],
    )

    for link_index in range(-1, p.getNumJoints(robot_id)):
        p.setCollisionFilterPair(
            robot_id, payload_id, link_index, -1, enableCollision=0
        )

    return payload_id


def teleport_robot(robot_id, payload_id, position, orientation):
    """
    Safely repositions the robot AND its rigidly-attached payload
    together. attach_payload()'s fixed constraint is anchored at the
    payload's world pose at creation time; calling
    p.resetBasePositionAndOrientation on the robot alone (without also
    moving the payload) leaves the constraint trying to snap the
    payload across a large sudden offset next physics step, which
    manifests as the robot being violently flung. Not used during
    normal teleop driving (the AMR only ever moves via wheel velocity
    there), but needed for any debugging/testing code that wants to
    place the robot directly.
    """

    p.resetBasePositionAndOrientation(robot_id, position, orientation)

    payload_offset = PAYLOAD_LOCAL_OFFSET
    rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    payload_position = np.array(position) + rotation_matrix @ np.array(payload_offset)

    p.resetBasePositionAndOrientation(payload_id, payload_position.tolist(), orientation)
    p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])
    p.resetBaseVelocity(payload_id, [0, 0, 0], [0, 0, 0])


def find_drive_wheel_joints(robot_id):

    wheel_joint_names = {
        "left_drive_wheel_joint": None,
        "right_drive_wheel_joint": None,
    }

    caster_joint_names = {
        "front_caster_wheel_joint": None,
        "rear_caster_wheel_joint": None,
    }

    number_of_joints = p.getNumJoints(robot_id)

    print()
    print("Robot joints")
    print("--------------------------------")

    for joint_index in range(number_of_joints):

        joint_info = p.getJointInfo(robot_id, joint_index)
        joint_name = joint_info[1].decode("utf-8")

        print(f"{joint_index}: {joint_name}")

        if joint_name in wheel_joint_names:
            wheel_joint_names[joint_name] = joint_index

        if joint_name in caster_joint_names:
            caster_joint_names[joint_name] = joint_index

    missing = [
        name for name, index in wheel_joint_names.items() if index is None
    ]

    if missing:
        raise RuntimeError(f"Missing drive wheel joints: {missing}")

    # Casters are left free-spinning (no motor control applied), so a
    # missing caster joint is only a warning, not a hard failure.
    for name, index in caster_joint_names.items():
        if index is None:
            print(f"WARNING: caster joint '{name}' not found.")

    return (
        wheel_joint_names["left_drive_wheel_joint"],
        wheel_joint_names["right_drive_wheel_joint"],
    )


# ============================================================
# WORK STATION SETUP (robotic arm + conveyor + magnetic docking)
# ============================================================

def setup_work_stations(environment, robot_id):
    """
    Resolves joint indices and builds the magnetic DockingPair objects
    for both work stations, so the teleop loop in main() only has to
    call high-level commands (engage/disengage/move_to_waypoint/
    command_conveyor) rather than re-discovering joints every frame.

    Returns a dict with everything the teleop loop needs:
        {
            "arm_id", "arm_joint_indices", "arm_end_effector_link",
            "arm_docking",
            "conveyor_id", "conveyor_infeed_joint",
            "conveyor_outfeed_joint", "conveyor_docking",
        }
    """

    arm_id = environment["arm_station_id"]
    conveyor_id = environment["conveyor_station_id"]

    arm_joint_indices, arm_end_effector_link = arm_control.find_arm_joints(arm_id)
    arm_control.reset_joint_angles(arm_id, arm_joint_indices)

    conveyor_infeed_joint, conveyor_outfeed_joint = (
        conveyor_control.find_conveyor_roller_joints(conveyor_id)
    )

    arm_docking = magnetic_docking.build_arm_station_docking_pair(arm_id, robot_id)
    conveyor_docking = magnetic_docking.build_conveyor_station_docking_pair(
        conveyor_id, robot_id
    )

    return {
        "arm_id": arm_id,
        "arm_joint_indices": arm_joint_indices,
        "arm_end_effector_link": arm_end_effector_link,
        "arm_docking": arm_docking,
        "conveyor_id": conveyor_id,
        "conveyor_infeed_joint": conveyor_infeed_joint,
        "conveyor_outfeed_joint": conveyor_outfeed_joint,
        "conveyor_docking": conveyor_docking,
    }


def toggle_docking(docking_pair, station_label):
    """
    Engages the magnets if currently disengaged, disengages if currently
    engaged. try_engage() already refuses (with a reason) if the AMR is
    not within tolerance of both pads, so this is safe to call from a
    single keypress regardless of where the AMR currently is.
    """

    if docking_pair.is_engaged:
        docking_pair.disengage()
        print(f"{station_label} magnets -> OFF")
        return

    engaged, reason = docking_pair.try_engage()
    if engaged:
        print(f"{station_label} magnets -> ON")
    else:
        print(f"{station_label} magnets -> refused ({reason})")


# ============================================================
# MOTOR CONTROL
# ============================================================

def set_wheel_velocity(robot_id, joint_index, angular_velocity):

    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=joint_index,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=angular_velocity,
        force=MAX_MOTOR_FORCE_NM,
    )


def drive_robot(robot_id, left_joint, right_joint, left_speed, right_speed):

    set_wheel_velocity(robot_id, left_joint, left_speed)
    set_wheel_velocity(robot_id, right_joint, right_speed)


def command_to_wheel_speeds(command):

    if command == "FORWARD":
        return (
            WHEEL_MAX_ANGULAR_VELOCITY_RAD_S,
            WHEEL_MAX_ANGULAR_VELOCITY_RAD_S,
        )

    if command == "BACKWARD":
        return (
            -WHEEL_MAX_ANGULAR_VELOCITY_RAD_S,
            -WHEEL_MAX_ANGULAR_VELOCITY_RAD_S,
        )

    if command == "LEFT":
        return (
            -WHEEL_TURN_ANGULAR_VELOCITY_RAD_S,
            WHEEL_TURN_ANGULAR_VELOCITY_RAD_S,
        )

    if command == "RIGHT":
        return (
            WHEEL_TURN_ANGULAR_VELOCITY_RAD_S,
            -WHEEL_TURN_ANGULAR_VELOCITY_RAD_S,
        )

    return (0.0, 0.0)


# ============================================================
# ROBOT POSE
# ============================================================

def get_robot_pose(robot_id):

    position, quaternion = p.getBasePositionAndOrientation(robot_id)
    roll, pitch, yaw = p.getEulerFromQuaternion(quaternion)

    return {
        "x": position[0],
        "y": position[1],
        "z": position[2],
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
    }


# ============================================================
# CAMERA POSE + CAPTURE
# ============================================================

def compute_camera_pose(robot_id, local_offset):
    """
    World-frame eye/target/up for a camera mounted at local_offset
    (robot base-link frame) looking along the robot's local +X axis.
    """

    robot_position, robot_quaternion = p.getBasePositionAndOrientation(
        robot_id
    )

    rotation_matrix = np.array(
        p.getMatrixFromQuaternion(robot_quaternion)
    ).reshape(3, 3)

    robot_position = np.array(robot_position)

    camera_position = robot_position + rotation_matrix @ local_offset

    forward_vector = rotation_matrix @ np.array([1.0, 0.0, 0.0])
    up_vector = rotation_matrix @ np.array([0.0, 0.0, 1.0])

    camera_target = camera_position + forward_vector * CAMERA_LOOK_DISTANCE

    return camera_position, camera_target, up_vector


def capture_camera(robot_id, local_offset):

    camera_position, camera_target, camera_up = compute_camera_pose(
        robot_id, local_offset
    )

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_position.tolist(),
        cameraTargetPosition=camera_target.tolist(),
        cameraUpVector=camera_up.tolist(),
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=CAMERA_FOV,
        aspect=(CAMERA_WIDTH / CAMERA_HEIGHT),
        nearVal=CAMERA_NEAR,
        farVal=CAMERA_FAR,
    )

    image = p.getCameraImage(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )

    rgba = np.array(image[2], dtype=np.uint8).reshape(
        CAMERA_HEIGHT, CAMERA_WIDTH, 4
    )
    bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)

    return bgr


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("INDOOR LOCALIZATION - WAREHOUSE AMR SIMULATION")
    print("=" * 60)

    initialize_simulation()

    environment = warehouse.create_environment()

    print()
    print("Warehouse environment built:")
    print(f"  wall segments  : {len(environment['walls'])}")
    print(f"  dock zones     : {[n for n, _ in environment['dock_markers']]}")
    print(f"  storage racks  : {len(environment['storage_racks'])}")

    print(f"  arm station id : {environment['arm_station_id']}")
    print(f"  conveyor id    : {environment['conveyor_station_id']}")

    robot_id = load_robot()
    payload_id = attach_payload(robot_id)

    left_wheel_joint, right_wheel_joint = find_drive_wheel_joints(robot_id)

    stations = setup_work_stations(environment, robot_id)

    current_command = "STOP"
    conveyor_command = "STOP"

    drive_robot(robot_id, left_wheel_joint, right_wheel_joint, 0.0, 0.0)

    p.resetDebugVisualizerCamera(
        cameraDistance=10.0,
        cameraYaw=50,
        cameraPitch=-35,
        cameraTargetPosition=[-5.0, -1.0, 0.0],
    )

    print()
    print("CONTROLS")
    print("--------------------------------")
    print("Click one of the camera preview windows.")
    print()
    print("W     -> AMR forward")
    print("S     -> AMR backward")
    print("A     -> AMR turn left")
    print("D     -> AMR turn right")
    print("SPACE -> AMR stop")
    print()
    print("M     -> toggle robotic arm station magnets (engage/disengage)")
    print("N     -> toggle conveyor station magnets (engage/disengage)")
    print("H     -> reset robotic arm to its home pose")
    print("I     -> conveyor belt forward (toward truck)")
    print("K     -> conveyor belt backward (toward AMR)")
    print("O     -> conveyor belt stop")
    print()
    print("Q     -> Quit")
    print()
    print("You DO NOT need to hold the key.")
    print("Press W once and the robot keeps moving until another command.")
    print(
        "Magnets only engage if the AMR is centred under the station's "
        "pedestal/frame - drive it into the PICKING or SHIPPING dock bay "
        "first."
    )
    print()

    last_pose_print = time.time()
    running = True

    try:

        while running and p.isConnected():

            left_speed, right_speed = command_to_wheel_speeds(current_command)

            drive_robot(
                robot_id, left_wheel_joint, right_wheel_joint,
                left_speed, right_speed,
            )

            p.stepSimulation()

            left_stereo_frame = capture_camera(robot_id, LEFT_STEREO_OFFSET)
            right_stereo_frame = capture_camera(robot_id, RIGHT_STEREO_OFFSET)
            rgb_frame = capture_camera(robot_id, RGB_CAMERA_OFFSET)

            pose = get_robot_pose(robot_id)
            yaw_degrees = math.degrees(pose["yaw"])

            overlay_text = (
                f"GT x={pose['x']:.2f} y={pose['y']:.2f} "
                f"yaw={yaw_degrees:.1f} deg | cmd={current_command}"
            )

            cv2.putText(
                rgb_frame, overlay_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(
                rgb_frame, "RGB CAMERA (centre)", (10, CAMERA_HEIGHT - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(
                left_stereo_frame, "LEFT STEREO", (10, CAMERA_HEIGHT - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(
                right_stereo_frame, "RIGHT STEREO", (10, CAMERA_HEIGHT - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

            cv2.imshow("RGB Camera (centre)", rgb_frame)
            cv2.imshow("Left Stereo Camera", left_stereo_frame)
            cv2.imshow("Right Stereo Camera", right_stereo_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("w"):
                current_command = "FORWARD"
                print("Command -> FORWARD")
            elif key == ord("s"):
                current_command = "BACKWARD"
                print("Command -> BACKWARD")
            elif key == ord("a"):
                current_command = "LEFT"
                print("Command -> LEFT")
            elif key == ord("d"):
                current_command = "RIGHT"
                print("Command -> RIGHT")
            elif key == 32:
                current_command = "STOP"
                print("Command -> STOP")
            elif key == ord("m"):
                toggle_docking(stations["arm_docking"], "Arm station")
            elif key == ord("n"):
                toggle_docking(stations["conveyor_docking"], "Conveyor station")
            elif key == ord("h"):
                arm_control.reset_joint_angles(
                    stations["arm_id"], stations["arm_joint_indices"]
                )
                print("Robotic arm -> HOME pose")
            elif key == ord("i"):
                conveyor_command = "FORWARD"
                conveyor_control.command_conveyor(
                    stations["conveyor_id"],
                    stations["conveyor_infeed_joint"],
                    stations["conveyor_outfeed_joint"],
                    conveyor_command,
                )
                print("Conveyor -> FORWARD")
            elif key == ord("k"):
                conveyor_command = "BACKWARD"
                conveyor_control.command_conveyor(
                    stations["conveyor_id"],
                    stations["conveyor_infeed_joint"],
                    stations["conveyor_outfeed_joint"],
                    conveyor_command,
                )
                print("Conveyor -> BACKWARD")
            elif key == ord("o"):
                conveyor_command = "STOP"
                conveyor_control.command_conveyor(
                    stations["conveyor_id"],
                    stations["conveyor_infeed_joint"],
                    stations["conveyor_outfeed_joint"],
                    conveyor_command,
                )
                print("Conveyor -> STOP")
            elif key == ord("q"):
                current_command = "STOP"
                running = False

            if time.time() - last_pose_print >= 1.0:
                print(
                    f"GT Pose | x={pose['x']:.3f} m | y={pose['y']:.3f} m | "
                    f"z={pose['z']:.3f} m | yaw={yaw_degrees:.2f} deg | "
                    f"command={current_command} | conveyor={conveyor_command} | "
                    f"arm_magnet={'ON' if stations['arm_docking'].is_engaged else 'OFF'} | "
                    f"conveyor_magnet={'ON' if stations['conveyor_docking'].is_engaged else 'OFF'}"
                )
                last_pose_print = time.time()

            time.sleep(1.0 / SIMULATION_HZ)

    except KeyboardInterrupt:
        print()
        print("Ctrl+C detected.")

    finally:
        if p.isConnected():
            drive_robot(robot_id, left_wheel_joint, right_wheel_joint, 0.0, 0.0)
            conveyor_control.command_conveyor(
                stations["conveyor_id"],
                stations["conveyor_infeed_joint"],
                stations["conveyor_outfeed_joint"],
                "STOP",
            )
        cv2.destroyAllWindows()
        if p.isConnected():
            p.disconnect()
        print()
        print("Simulation stopped.")


if __name__ == "__main__":
    main()
