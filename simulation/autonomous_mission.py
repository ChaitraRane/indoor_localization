"""
INDOOR LOCALIZATION - AUTONOMOUS MISSION HARNESS (main entry point)
=====================================================================

Wires the warehouse, the AMR, the arm-module stand + swappable
suction-arm module and the autonomous stack (mission_controller)
together and runs the whole job:

    follow the lane (avoiding the obstacles on it, mapping as it goes)
    -> detect the boxes at A -> decide the suction-arm module is needed
    -> backtrack to the arm-module stand on the live map -> reverse-dock,
    hoist + magnets transfer the module -> back to A -> stereo-guided IK
    picks every box onto the cargo tray -> drive to the truck -> the arm
    loads every box into the trailer.

This file owns everything simulator-side (PyBullet, robot, cameras,
wheels, the stand's hoist, the magnet lock, the arm's motors) behind a
small RobotIO object; the controller owns the algorithms. Cameras render
from the TRUE physical pose (reality); the controller only ever sees the
images, depth, wheel encoders and arm joint encoders.

Run:
    python autonomous_mission.py                        # GUI (view follows the AMR;
                                                        #  --no-follow to orbit freely)
    python autonomous_mission.py --realtime --show-camera
    python autonomous_mission.py --headless             # DIRECT (no window)
    python autonomous_mission.py --headless --record out.mp4
"""

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import pybullet_data

import camera_rig
import magnetic_docking
import mission_config as cfg
import video_overlay as vid
import warehouse_environment as warehouse
from arm_manipulator import ArmHardware, ArmStereoCamera, CargoLatches, SuctionGripper
from conveyor_hardware import ConveyorHardware
from box_detector import BoxDetector
from mission_controller import ARM_STATES, CONVEYOR_STATES, MissionController, STATE_COMPLETE

SCRIPT_DIR = Path(__file__).resolve().parent
AMR_URDF = SCRIPT_DIR / "amr_robot.urdf"
STAND_URDF = SCRIPT_DIR / "arm_station_stand.urdf"
MODULE_URDF = SCRIPT_DIR / "suction_arm_module.urdf"
CONVEYOR_STAND_URDF = SCRIPT_DIR / "conveyor_station_stand.urdf"
CONVEYOR_MODULE_URDF = SCRIPT_DIR / "conveyor_module.urdf"

SIMULATION_HZ = 240
STEPS_PER_CONTROL = 12            # control/perception at 20 Hz
MAX_MOTOR_FORCE_NM = 10.0         # rated wheel motor torque
# Gyro model (BNO085-class): white noise (rad/s), initial turn-on bias
# (rad/s) and bias random walk (rad/s per sqrt(s), ~10 deg/h instability).
GYRO_NOISE = 0.003
GYRO_INITIAL_BIAS = 0.002
GYRO_BIAS_WALK = 2.0e-5

COMMAND_LINEAR_DEADZONE_MPS = 0.03
COMMAND_ANGULAR_DEADZONE_RPS = 0.08


def classify_drive_command(linear_mps, angular_rps):
    """Human-readable label for the (linear, angular) command."""

    moving = abs(linear_mps) > COMMAND_LINEAR_DEADZONE_MPS
    turning = abs(angular_rps) > COMMAND_ANGULAR_DEADZONE_RPS
    if not moving and not turning:
        return "STOP"
    if not moving:
        return "TURN LEFT (in place)" if angular_rps > 0 else "TURN RIGHT (in place)"
    label = "FORWARD" if linear_mps >= 0 else "BACKWARD"
    if turning:
        label += " + LEFT" if angular_rps > 0 else " + RIGHT"
    return label


# ============================================================
# ROBOT IO (simulator-side interface the controller talks to)
# ============================================================

class ToolStation:
    """One tool stand + the swappable module it stores (simulator side):
    the stand's hoist, the hoist's hold on the module, and the module's
    magnet lock to the AMR."""

    def __init__(self, kind, stand_id, module_id, robot_id):
        self.kind = kind
        self.stand_id = stand_id
        self.module_id = module_id
        self.docking = magnetic_docking.build_arm_module_docking_pair(module_id, robot_id)
        self.hoist_joint = next(j for j in range(p.getNumJoints(stand_id))
                                if p.getJointInfo(stand_id, j)[1].decode() == "hoist_joint")
        self.hold = None
        self.hold_ref = None
        self.hold_orn = None
        self.aboard = False
        self.hold_module(offset_now=0.0)

    def hold_module(self, offset_now):
        """Hoist takes hold of the module where it is now; offset_now is the
        hoist's current offset, so later set_hoist() calls move the module
        with the carriage without a jump."""

        pos, orn = p.getBasePositionAndOrientation(self.module_id)
        self.hold_ref = np.array(pos) - np.array([0.0, 0.0, offset_now])
        self.hold_orn = orn
        self.hold = p.createConstraint(
            self.module_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0],
            parentFramePosition=[0, 0, 0], childFramePosition=list(pos),
            childFrameOrientation=orn)
        p.changeConstraint(self.hold, maxForce=5000.0)

    def set_hoist(self, offset_m):
        p.setJointMotorControl2(self.stand_id, self.hoist_joint, p.POSITION_CONTROL,
                                targetPosition=float(offset_m), force=3000.0, maxVelocity=0.3)
        if self.hold is not None:
            pivot = self.hold_ref + np.array([0.0, 0.0, offset_m])
            p.changeConstraint(self.hold, jointChildPivot=pivot.tolist(),
                               jointChildFrameOrientation=self.hold_orn, maxForce=5000.0)

    def engage(self):
        engaged, reason = self.docking.try_engage()
        if not engaged:
            return False, reason
        for pad in (self.docking.front_pad, self.docking.rear_pad):
            p.changeConstraint(pad.constraint_id, maxForce=4000.0)
        if self.hold is not None:
            p.removeConstraint(self.hold)
            self.hold = None
        self.aboard = True
        return True, reason

    def release(self, hoist_offset_now):
        """Hoist hooks take the module, then the magnets let go."""

        self.hold_module(offset_now=hoist_offset_now)
        self.docking.disengage()
        self.aboard = False


class RobotIO:
    control_dt = STEPS_PER_CONTROL / SIMULATION_HZ

    def __init__(self, robot_id, left_wheel, right_wheel, stations, box_detector, gui):
        self.robot_id = robot_id
        self.left_wheel_joint = left_wheel
        self.right_wheel_joint = right_wheel
        self.stations = stations                  # kind -> ToolStation
        self.box_detector = box_detector
        self.gui = gui
        self.intrinsics = camera_rig.RGBD_INTRINSICS

        arm_module = stations["SUCTION_ARM"].module_id
        self.module_id = arm_module
        self.arm_hw = ArmHardware(arm_module)
        self.arm_camera = ArmStereoCamera(self.arm_hw)
        self.gripper = SuctionGripper(self.arm_hw, ignore_bodies=(robot_id,))
        self.latches = CargoLatches(arm_module)
        self.conveyor = ConveyorHardware(stations["CONVEYOR"].module_id, ignore_bodies=(robot_id,))

        self.last_command = "STOP"
        self.last_linear_mps = 0.0
        self.last_angular_rps = 0.0
        self._rng = np.random.default_rng(42)
        self._gyro_angle = 0.0
        self._gyro_bias = self._rng.normal(0.0, GYRO_INITIAL_BIAS)
        self._debug_path_ids = []
        self._last_path_sig = None

    # ----------------------------------------------------- stand / module

    @property
    def module_aboard(self):
        for kind, st in self.stations.items():
            if st.aboard:
                return kind
        return None

    def set_hoist(self, kind, offset_m):
        self.stations[kind].set_hoist(offset_m)

    def engage_module(self, kind):
        """Electromagnets: lock the module's two pads onto the AMR's pads
        (only if both are within tolerance), then the stand lets go."""

        return self.stations[kind].engage()

    def pad_separation(self, kind):
        """Docking proximity sensors: largest pad-to-counterpart distance."""

        return self.stations[kind].docking.max_separation_m()

    def release_module(self, kind, hoist_offset_now):
        self.stations[kind].release(hoist_offset_now)

    def pre_step(self):
        """Called before every physics step (belt traction, gyro sampling)."""

        self.conveyor.apply_traction()
        # BNO085-class gyro on imu_link: body yaw rate + white noise + slowly
        # wandering bias, integrated at the physics rate (the IMU's own
        # internal integration). Simulates a sensor reading - the
        # controller only ever sees the integrated angle.
        dt = 1.0 / SIMULATION_HZ
        wz = p.getBaseVelocity(self.robot_id)[1][2]
        self._gyro_bias += self._rng.normal(0.0, GYRO_BIAS_WALK) * math.sqrt(dt)
        self._gyro_angle += (wz + self._gyro_bias + self._rng.normal(0.0, GYRO_NOISE)) * dt

    def gyro_yaw(self):
        """Integrated gyro heading (rad, arbitrary zero) - the IMU output."""

        return self._gyro_angle

    # ----------------------------------------------------- sensing

    def _render(self, which, width, height):
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot_id)
        eye, forward, up = camera_rig.frame_from_true_base(base_pos, base_quat, which)
        view = p.computeViewMatrix(eye.tolist(), (eye + forward * 2.0).tolist(), up.tolist())
        proj = p.computeProjectionMatrixFOV(fov=camera_rig.CAMERA_FOV_DEG, aspect=width / height,
                                            nearVal=camera_rig.CAMERA_NEAR, farVal=camera_rig.CAMERA_FAR)
        img = p.getCameraImage(width=width, height=height, viewMatrix=view,
                               projectionMatrix=proj, renderer=p.ER_TINY_RENDERER)
        rgba = np.reshape(img[2], (height, width, 4)).astype(np.uint8)
        bgr = rgba[:, :, :3][:, :, ::-1].copy()
        buf = np.reshape(img[3], (height, width))
        near, far = camera_rig.CAMERA_NEAR, camera_rig.CAMERA_FAR
        depth = far * near / (far - (far - near) * buf)
        return bgr, depth.astype(np.float64)

    def capture(self):
        lb, ld = self._render("left", camera_rig.RGBD_WIDTH, camera_rig.RGBD_HEIGHT)
        rb, rd = self._render("right", camera_rig.RGBD_WIDTH, camera_rig.RGBD_HEIGHT)
        cb, _ = self._render("center", camera_rig.CENTER_WIDTH, camera_rig.CENTER_HEIGHT)
        return {"left_bgr": lb, "left_depth": ld, "right_bgr": rb, "right_depth": rd,
                "center_bgr": cb}

    def render_camera(self, which, width, height):
        """Display-resolution render of one robot camera (video only)."""

        return self._render(which, width, height)[0]

    def render_front_camera_display(self):
        return self._render("left", camera_rig.DISPLAY_RGBD_WIDTH, camera_rig.DISPLAY_RGBD_HEIGHT)

    def wheel_angles(self):
        return (p.getJointState(self.robot_id, self.left_wheel_joint)[0],
                p.getJointState(self.robot_id, self.right_wheel_joint)[0])

    def true_pose(self):
        """Ground truth - for logging/scoring ONLY, never fed to the controller."""

        pos, q = p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(q)[2]
        return pos[0], pos[1], yaw

    def render_overview(self, width, height, close_up_target=None):
        base_pos, base_q = p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(base_q)[2]
        if close_up_target is not None:
            tx, ty = close_up_target
            eye = [tx - 1.5 * math.cos(yaw) + 1.2 * math.sin(yaw),
                   ty - 1.5 * math.sin(yaw) - 1.2 * math.cos(yaw), 1.7]
            target = [tx, ty, 0.45]
        else:
            eye = [base_pos[0] - 3.4 * math.cos(yaw) - 1.2, base_pos[1] - 3.4 * math.sin(yaw) - 1.2, 3.6]
            target = [base_pos[0], base_pos[1], 0.3]
        view = p.computeViewMatrix(eye, target, [0, 0, 1])
        proj = p.computeProjectionMatrixFOV(fov=60.0, aspect=width / height, nearVal=0.1, farVal=40.0)
        img = p.getCameraImage(width=width, height=height, viewMatrix=view, projectionMatrix=proj,
                               renderer=p.ER_TINY_RENDERER)
        rgba = np.reshape(img[2], (height, width, 4)).astype(np.uint8)
        return rgba[:, :, :3][:, :, ::-1].copy()

    # ----------------------------------------------------- actuation

    def drive(self, linear_mps, angular_rps):
        half = cfg.AMR_WHEEL_SEPARATION_M / 2.0
        v_left = (linear_mps - angular_rps * half) / cfg.AMR_WHEEL_RADIUS_M
        v_right = (linear_mps + angular_rps * half) / cfg.AMR_WHEEL_RADIUS_M
        for joint, vel in ((self.left_wheel_joint, v_left), (self.right_wheel_joint, v_right)):
            p.setJointMotorControl2(self.robot_id, joint, p.VELOCITY_CONTROL,
                                    targetVelocity=vel, force=MAX_MOTOR_FORCE_NM)
        self.last_command = classify_drive_command(linear_mps, angular_rps)
        self.last_linear_mps = linear_mps
        self.last_angular_rps = angular_rps

    def stop(self):
        self.drive(0.0, 0.0)

    # ----------------------------------------------------- GUI markers

    def mark_point(self, name, x, y):
        if not self.gui:
            return
        p.addUserDebugLine([x, y, 0.02], [x, y, 1.9], [1.0, 0.45, 0.0], 3.0)
        p.addUserDebugText(name, [x, y, 2.0], [1.0, 0.45, 0.0], textSize=1.4)

    def draw_path(self, waypoints):
        if not self.gui:
            return
        sig = None if not waypoints else (len(waypoints), round(waypoints[0][0], 1),
                                          round(waypoints[-1][0], 1), round(waypoints[-1][1], 1))
        if sig == self._last_path_sig:
            return
        self._last_path_sig = sig
        for i in self._debug_path_ids:
            p.removeUserDebugItem(i)
        self._debug_path_ids = []
        if waypoints:
            for (x0, y0), (x1, y1) in zip(waypoints[:-1], waypoints[1:]):
                self._debug_path_ids.append(
                    p.addUserDebugLine([x0, y0, 0.03], [x1, y1, 0.03], [0.0, 0.8, 1.0], 3.0))


# ============================================================
# WORLD / ROBOT SETUP
# ============================================================

def _find_joint(body, name):
    for j in range(p.getNumJoints(body)):
        if p.getJointInfo(body, j)[1].decode() == name:
            return j
    raise RuntimeError(f"joint {name} missing")


def _configure_amr_physics(robot_id):
    """Drive wheels grip; casters are free ball-transfers. PyBullet attaches
    a braking velocity motor to every joint by default - left on, the
    'free-spinning' casters dragged against every drive command."""

    for name in ("front_caster_wheel_joint", "rear_caster_wheel_joint"):
        j = _find_joint(robot_id, name)
        p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, targetVelocity=0.0, force=0.0)
        p.changeDynamics(robot_id, j, lateralFriction=0.0, spinningFriction=0.0,
                         rollingFriction=0.0)
    for name in ("left_drive_wheel_joint", "right_drive_wheel_joint"):
        p.changeDynamics(robot_id, _find_joint(robot_id, name), lateralFriction=1.0,
                         spinningFriction=0.001, rollingFriction=0.0)


def setup(headless):
    client = p.connect(p.DIRECT if headless else p.GUI)
    if client < 0:
        raise RuntimeError("could not connect to PyBullet")
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / SIMULATION_HZ)
    if not headless:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    env = warehouse.create_environment(include_work_stations=False)

    robot_id = p.loadURDF(
        str(AMR_URDF),
        basePosition=[cfg.ROBOT_START_XY[0], cfg.ROBOT_START_XY[1], cfg.ROBOT_START_Z],
        baseOrientation=p.getQuaternionFromEuler([0, 0, cfg.ROBOT_START_YAW_RAD]),
    )
    left = _find_joint(robot_id, "left_drive_wheel_joint")
    right = _find_joint(robot_id, "right_drive_wheel_joint")
    _configure_amr_physics(robot_id)

    stations = {}
    for kind, stand_urdf, module_urdf in (("SUCTION_ARM", STAND_URDF, MODULE_URDF),
                                          ("CONVEYOR", CONVEYOR_STAND_URDF, CONVEYOR_MODULE_URDF)):
        st = cfg.STATIONS[kind]
        dock_q = p.getQuaternionFromEuler([0, 0, st["yaw"]])
        stand_id = p.loadURDF(str(stand_urdf), [st["xy"][0], st["xy"][1], 0.0], dock_q,
                              useFixedBase=True)
        module_id = p.loadURDF(str(module_urdf), [st["xy"][0], st["xy"][1], cfg.MODULE_HOLD_Z_M],
                               dock_q, useFixedBase=False)
        # The module mates with the AMR through the magnet lock (and is
        # held by the hoist on its stand), so it never collides with
        # either; the AMR DOES collide with the stand's legs.
        for li in range(-1, p.getNumJoints(robot_id)):
            for lj in range(-1, p.getNumJoints(module_id)):
                p.setCollisionFilterPair(robot_id, module_id, li, lj, 0)
        for li in range(-1, p.getNumJoints(stand_id)):
            for lj in range(-1, p.getNumJoints(module_id)):
                p.setCollisionFilterPair(stand_id, module_id, li, lj, 0)
        stations[kind] = ToolStation(kind, stand_id, module_id, robot_id)

    if not headless:
        p.resetDebugVisualizerCamera(cameraDistance=13.0, cameraYaw=40, cameraPitch=-55,
                                     cameraTargetPosition=[-2.5, 1.5, 0.0])

    box_detector = BoxDetector(use_yolo=True)
    io = RobotIO(robot_id, left, right, stations, box_detector, gui=not headless)
    for kind in stations:
        io.set_hoist(kind, 0.0)
    io.stop()
    for _ in range(120):
        io.pre_step()
        p.stepSimulation()
    return io, env


# ============================================================
# VIDEO / LIVE DISPLAY
# ============================================================

VIDEO_WIDTH = vid.VIDEO_WIDTH
VIDEO_HEIGHT = vid.VIDEO_HEIGHT
VIDEO_FPS = 20
VIDEO_TICK_STRIDE = 3            # one frame per 3 control ticks -> 3x real time

PHASE_CAPTIONS = {
    "EXPLORE_TO_DETECTION": "exploring along the lane to the stock (mapping live)",
    "DETECT_BOXES": "detecting the boxes (RGB-D + YOLO)",
    "DECIDE_PAYLOAD": "deciding which module the boxes need",
    "GO_TO_STATION": "driving to the tool stand (A* on the live map)",
    "LOCALIZE_AT_STATION": "landmark fix on the stand legs",
    "TURN_FOR_DOCK": "turning to reverse under the stand",
    "REVERSE_DOCK": "reversing under the stand",
    "TRANSFER_MODULE": "hoist + electromagnets transferring the module",
    "LEAVE_STATION": "leaving the stand",
    "GO_TO_PICK": "driving to the pick table",
    "APPROACH_PICK": "depth-servo to the table edge",
    "ARM_LOAD": "stereo + IK suction picking, one box at a time",
    "BACK_OFF": "backing clear",
    "GO_TO_TRUCK": "driving to the truck",
    "MEASURE_TRUCK": "truck landmark fix + deck height",
    "APPROACH_TRUCK": "depth-servo to the trailer sill",
    "ARM_UNLOAD": "arm loading the boxes into the trailer",
    "GO_TO_SHELF": "driving to shelf B",
    "MEASURE_SHELF": "measuring shelf height + box size, setting the belt height",
    "TURN_AROUND": "turning to present the belt nose",
    "REVERSE_TO_SHELF": "reversing the nose under the box edge (ToF)",
    "CONVEYOR_LOAD": "belt touches the box and pulls it aboard (IR beams)",
    "PULL_AWAY": "pulling away, belt to transport height",
    "REVERSE_TO_TRUCK": "reversing the belt nose over the truck deck",
    "CONVEYOR_UNLOAD": "belt pushing the box into the trailer",
    "GO_HOME": "returning to the home station",
    "MISSION_COMPLETE": "mission complete",
}
PANEL_W = VIDEO_WIDTH // 2
PANEL_H = VIDEO_HEIGHT // 2
DEPTH_DISPLAY_CLIP_M = 8.0

LIVE_WINDOW_RGB = "AMR Front Camera - RGB"
LIVE_WINDOW_DEPTH = "AMR Front Camera - Depth"
LIVE_WINDOW_ARM = "Arm Stereo Camera"


def _label_panel(frame, text, color=(0, 255, 255)):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(frame, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


def colorize_depth(depth_m, clip=DEPTH_DISPLAY_CLIP_M):
    clipped = np.clip(depth_m, camera_rig.CAMERA_NEAR, clip)
    norm = (clipped - camera_rig.CAMERA_NEAR) / (clip - camera_rig.CAMERA_NEAR)
    return cv2.applyColorMap(((1.0 - norm) * 255.0).astype(np.uint8), cv2.COLORMAP_JET)


def arm_camera_panel(controller, width, height):
    frame = controller.arm.debug_frame
    if frame is None:
        img = np.full((height, width, 3), 25, np.uint8)
        return _label_panel(img, "ARM STEREO CAMERA (waiting)")
    left = cv2.resize(frame.left, (width, height))
    return _label_panel(left, f"ARM STEREO CAMERA | {controller.arm.status}")


class VideoState:
    """Per-recording view state (chase-camera smoothing, map zoom window)."""

    def __init__(self):
        self.chase = vid.ChaseCamera()
        self.map = vid.MapPanel()


def build_video_frame(io, controller, status, vstate):
    state = status["state"]
    in_arm = state in ARM_STATES
    mode, target = "drive", None
    x, y, yaw = io.true_pose()
    if state in ("TURN_FOR_DOCK", "REVERSE_DOCK", "TRANSFER_MODULE") and controller.station_kind:
        mode, target = "close", cfg.STATIONS[controller.station_kind]["xy"]
    if in_arm:
        mode, target = "close", (x + 0.4 * math.cos(yaw), y + 0.4 * math.sin(yaw))
    if state in CONVEYOR_STATES or (state in ("TURN_AROUND", "PULL_AWAY") and io.module_aboard == "CONVEYOR"):
        mode, target = "conveyor", (x - 0.5 * math.cos(yaw), y - 0.5 * math.sin(yaw))   # belt nose
    chase = vid.chase_panel(vstate.chase, io, controller, mode, target)

    if in_arm:
        second = arm_camera_panel(controller, PANEL_W, PANEL_H)
    else:
        second = vid.lane_camera_panel(io, controller)

    pose = controller.pose_estimator.get_pose()
    lane = controller._last_lane
    occ = vid.MapPanel.render(vstate.map, controller, pose, None if lane is None else lane.mask)
    vid.label(occ, f"LIVE OCCUPANCY MAP (built in real time) + A* ROUTE | robot = estimated pose")

    panel = np.full((PANEL_H, PANEL_W, 3), (30, 30, 30), np.uint8)
    px, py, pdeg = status["pose"]
    gx, gy, _ = io.true_pose()
    lines = [
        (f"state : {state}", (0, 255, 255), 0.62),
        (f"nav   : {status['nav']}" if status["nav"] else f"arm   : {status['arm']}", (255, 255, 255), 0.55),
        (f"drive : {io.last_command}  v={io.last_linear_mps:+.2f} m/s  w={io.last_angular_rps:+.2f} rad/s",
         (255, 255, 255), 0.5),
        (f"pose (estimated): x={px:+.2f}  y={py:+.2f}  yaw={pdeg:+.0f} deg", (255, 255, 255), 0.5),
        (f"est. error vs ground truth: {math.hypot(px - gx, py - gy)*100:.1f} cm", (170, 170, 170), 0.47),
    ]
    path = controller.active_path
    if path and len(path) > 1:
        lines.append((f"A* route: {sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(path[:-1], path[1:])):.1f} m"
                      f" | lane-first cost (off-lane x60)", (255, 255, 0), 0.47))
    lines.append((f"job {status['job']} | module aboard: {io.module_aboard or 'none'} | boxes on tray: {io.latches.count}",
                  (0, 255, 0) if io.module_aboard else (200, 200, 200), 0.5))
    if io.module_aboard == "CONVEYOR":
        c = io.conveyor.status()
        tof = "-" if c["tof"] is None else f"{c['tof']:.3f} m"
        lines.append((f"conveyor: belt top z={c['belt_top'] + cfg.MODULE_DOCKED_Z_M:.3f} m, "
                      f"belt {c['belt_speed']:+.2f} m/s", (255, 200, 0), 0.47))
        lines.append((f"  IR nose {'BLOCKED' if c['ir_nose'] else 'clear'} | IR end "
                      f"{'BLOCKED' if c['ir_end'] else 'clear'} | ToF {tof}", (255, 200, 0), 0.47))
    d = controller.payload_decision
    if d is not None:
        lines.append((f"payload decision: {d.module} ({d.box_count} box(es), {d.max_footprint_m:.2f} m)",
                      (255, 255, 0), 0.47))
    lines.append(("pinned on the map (found live):", (0, 165, 255), 0.47))
    for name, pt in controller.semantic.named_points.items():
        lines.append((f"   {name}: ({pt.x:+.2f}, {pt.y:+.2f})", (0, 165, 255), 0.45))
    yy = 50
    for text, color, scale in lines:
        if yy > PANEL_H - vid.CAPTION_H - 6:
            break
        cv2.putText(panel, text, (12, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        yy += 23
    vid.label(panel, "MISSION STATUS (fully autonomous - no operator commands)")

    frame = np.vstack((np.hstack((chase, second)), np.hstack((occ, panel))))
    caption = (f"JOB {status['job']} | {state}: "
               f"{PHASE_CAPTIONS.get(state, '')}   [t={status['t']:.0f} s, 3x speed]")
    cv2.rectangle(frame, (0, VIDEO_HEIGHT - 30), (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0), -1)
    cv2.putText(frame, caption, (10, VIDEO_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (PANEL_W, 0), (PANEL_W, VIDEO_HEIGHT - 30), (0, 0, 0), 2)
    cv2.line(frame, (0, PANEL_H), (VIDEO_WIDTH, PANEL_H), (0, 0, 0), 2)
    return frame


def show_live_windows(io, controller, status):
    rgb, depth = io.render_front_camera_display()
    depth_vis = colorize_depth(depth)
    for frame in (rgb, depth_vis):
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (0, 0, 0), -1)
        cv2.putText(frame, f"state: {status['state']} {status['nav']}{status['arm']}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"command: {io.last_command} (v={io.last_linear_mps:+.2f}, "
                           f"w={io.last_angular_rps:+.2f})", (6, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow(LIVE_WINDOW_RGB, rgb)
    cv2.imshow(LIVE_WINDOW_DEPTH, depth_vis)
    if status["state"] in ARM_STATES:
        cv2.imshow(LIVE_WINDOW_ARM, arm_camera_panel(controller, 480, 360))
    cv2.waitKey(1)


# ============================================================
# MAIN MISSION LOOP
# ============================================================

def score_cargo(env, io):
    """Ground-truth check of where the boxes ended up (reporting only):
    small boxes from A and the big box from B inside the trailer."""

    inside = 0
    bodies = [b for b, _ in env["boxes_at_a"]] + [env["big_box"][0]]
    for body in bodies:
        lo, hi = p.getAABB(body)
        # The WHOLE box must be inside the trailer, resting on its deck.
        if (lo[0] > cfg.TRUCK_REAR_X - cfg.TRUCK_LENGTH_M and hi[0] < cfg.TRUCK_REAR_X + 0.005
                and lo[1] > cfg.TRUCK_Y - cfg.TRUCK_HALF_WIDTH_M
                and hi[1] < cfg.TRUCK_Y + cfg.TRUCK_HALF_WIDTH_M
                and lo[2] > cfg.TRUCK_DECK_Z_M - 0.01):
            inside += 1
    return inside


def follow_camera(io, status):
    """GUI only: keep the AMR in view - close in for docking / arm work."""

    x, y, _ = io.true_pose()
    close = status["state"] in ARM_STATES + CONVEYOR_STATES or status["state"] in (
        "TURN_FOR_DOCK", "REVERSE_DOCK", "TRANSFER_MODULE", "APPROACH_PICK", "APPROACH_TRUCK",
        "TURN_AROUND", "PULL_AWAY")
    p.resetDebugVisualizerCamera(cameraDistance=3.2 if close else 7.5, cameraYaw=35,
                                 cameraPitch=-38 if close else -50,
                                 cameraTargetPosition=[x, y, 0.4])


def run_mission(headless=True, max_control_iters=12000, realtime=False, verbose=True,
                record_path=None, show_camera_window=False, follow=True):
    io, env = setup(headless)
    controller = MissionController(io)

    writer = None
    vstate = VideoState()
    if record_path:
        writer = cv2.VideoWriter(str(record_path), cv2.VideoWriter_fourcc(*"avc1"), VIDEO_FPS,
                                 (VIDEO_WIDTH, VIDEO_HEIGHT))
        if not writer.isOpened():       # no H.264 encoder: fall back to MPEG-4
            writer = cv2.VideoWriter(str(record_path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS,
                                 (VIDEO_WIDTH, VIDEO_HEIGHT))
        if not writer.isOpened():
            print(f"[video] WARNING: could not open writer for {record_path}")
            writer = None
        else:
            print(f"[video] recording mission to {record_path}")

    print("=" * 70)
    print("INDOOR LOCALIZATION - AUTONOMOUS PICK-AND-LOAD MISSION")
    print("=" * 70)
    print(f"start {cfg.ROBOT_START_XY} | pick table at A ~{cfg.POINT_A} | "
          f"arm-module stand {cfg.ARM_STATION_XY} | truck boarding {cfg.TRUCK_BOARDING_XY}")
    print("-" * 70)

    start_wall = time.time()
    last_state = None
    max_err = 0.0
    for _ in range(max_control_iters):
        status = controller.update()
        for _ in range(STEPS_PER_CONTROL):
            io.pre_step()
            p.stepSimulation()
            if realtime:
                time.sleep(1.0 / SIMULATION_HZ)

        px, py, _ = status["pose"]
        gx, gy, _ = io.true_pose()
        err = math.hypot(px - gx, py - gy)
        max_err = max(max_err, err)
        io.draw_path(controller.active_path)
        if follow and not headless and status["tick"] % 5 == 0:
            follow_camera(io, status)

        if writer is not None and status["tick"] % VIDEO_TICK_STRIDE == 0:
            writer.write(build_video_frame(io, controller, status, vstate))
        if show_camera_window:
            show_live_windows(io, controller, status)

        if verbose and (status["state"] != last_state or status["tick"] % 100 == 0):
            print(f"  t={status['t']:7.1f}s {status['state']:22s} {status['nav']:20s}"
                  f"{status['arm']:18s} est={status['pose']} gt=({gx:+.2f},{gy:+.2f}) "
                  f"err={err*100:4.1f}cm")
            last_state = status["state"]
        if controller.finished:
            break

    loaded = score_cargo(env, io)
    if writer is not None:
        final = build_video_frame(io, controller, controller.update(), vstate)
        ok = controller.state == STATE_COMPLETE
        for _ in range(int(VIDEO_FPS * 3)):
            f = final.copy()
            cv2.rectangle(f, (0, 0), (f.shape[1], 46), (0, 0, 0), -1)
            cv2.putText(f, f"{'MISSION COMPLETE' if ok else controller.state}: {loaded} box(es) in the truck",
                        (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if ok else (0, 0, 255), 2,
                        cv2.LINE_AA)
            writer.write(f)
        writer.release()
        print(f"[video] saved {record_path}")

    elapsed = time.time() - start_wall
    nav = controller.nav.stats
    print("-" * 70)
    print(f"FINAL STATE   : {controller.state}")
    print(f"RESULT        : {controller.result_message}")
    if controller.payload_decision:
        print(f"PAYLOAD       : {controller.payload_decision.module} - {controller.payload_decision.reason}")
    if controller.station_fix:
        print(f"STATION FIX   : {controller.station_fix[0]*100:.1f} cm / {controller.station_fix[1]:+.1f} deg")
    if controller.load_report:
        print(f"ARM LOAD      : {controller.load_report.message} "
              f"({controller.load_report.observations} stereo looks)")
    if controller.unload_report:
        print(f"ARM UNLOAD    : {controller.unload_report.message}")
    print(f"NAVIGATION    : {nav}")
    print(f"POSE ERROR    : max {max_err*100:.1f} cm vs ground truth (logging only)")
    total = len(env["boxes_at_a"]) + 1
    print(f"BOXES IN TRUCK: {loaded}/{total} (ground-truth check: 4 small + 1 big)")
    home_err = math.hypot(io.true_pose()[0] - cfg.HOME_XY[0], io.true_pose()[1] - cfg.HOME_XY[1])
    print(f"MODULES       : {'all returned to their stands' if io.module_aboard is None else io.module_aboard + ' still aboard'}"
          f" | robot {home_err*100:.0f} cm from home (ground truth)")
    print()
    print(controller.semantic.summary())
    print("-" * 70)
    print(f"sim time {controller.t:.1f} s, control ticks {controller.tick}, wall {elapsed:.1f} s")
    success = controller.state == STATE_COMPLETE and loaded == total and io.module_aboard is None
    print(f"MISSION {'SUCCESS' if success else 'INCOMPLETE'}")

    if show_camera_window:
        cv2.destroyAllWindows()
    p.disconnect()
    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="DIRECT mode, no GUI")
    parser.add_argument("--max-iters", type=int, default=40000)
    parser.add_argument("--realtime", action="store_true", help="throttle to realtime (GUI)")
    parser.add_argument("--record", type=str, default=None, help="record the mission to this MP4")
    parser.add_argument("--show-camera", action="store_true",
                        help="live AMR front RGB/depth (+ arm stereo) windows")
    parser.add_argument("--no-follow", action="store_true",
                        help="GUI: do not move the 3D view camera with the robot")
    parser.add_argument("--jobs", type=str, default=None,
                        help="comma-separated stock locations to run, e.g. B (conveyor only)")
    args = parser.parse_args()
    if args.jobs:
        wanted = [j.strip().upper() for j in args.jobs.split(",") if j.strip()]
        cfg.JOBS = [j for j in cfg.JOBS if j["name"] in wanted]
        if not cfg.JOBS:
            parser.error(f"--jobs {args.jobs}: no such stock location")
        print(f"[mission] running only job(s): {', '.join(j['name'] for j in cfg.JOBS)}")

    success = run_mission(
        headless=args.headless, max_control_iters=args.max_iters,
        realtime=args.realtime and not args.headless, record_path=args.record,
        show_camera_window=args.show_camera and not args.headless,
        follow=not args.no_follow,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(0 if success else 1)


if __name__ == "__main__":
    main()
