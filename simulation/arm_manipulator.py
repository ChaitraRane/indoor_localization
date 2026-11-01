"""
INDOOR LOCALIZATION - SUCTION-ARM MODULE: IK, EYE-IN-HAND STEREO, PICK/PLACE
============================================================================

Everything the deck-mounted suction arm (suction_arm_module.urdf) needs to
load the boxes at point A onto the AMR's cargo tray and unload them into
the truck:

    ArmKinematics      forward kinematics of the 6-DOF chain in the MODULE
                       frame (= AMR base axes, origin at the plate
                       underside) and closed-form INVERSE KINEMATICS for
                       top-down tool poses (suction cup pointing straight
                       down): pan toward the target, a 2-link elbow-up
                       solution for shoulder/elbow, wrist_1 keeps the tool
                       vertical, wrist_3 sets the camera/roll. Verified
                       against PyBullet's own link poses in the self-test.

    ArmStereoCamera    the eye-in-hand stereo pair on the tool flange.
                       Renders a rectified left/right pair, computes a
                       disparity map (OpenCV StereoSGBM), converts it to
                       metric depth Z = f*B/d, back-projects to 3D and
                       transforms into the module frame using the arm's
                       OWN joint encoders + FK (never the simulator's
                       ground truth). From that 3D it finds boxes (colour
                       segmentation of the left image + the top-face
                       points) and measures support surfaces (the truck
                       deck height and its sill).

    SuctionGripper     vacuum cup: grips when the cup lip is in contact
                       with a movable object's surface (the "vacuum
                       sensor"), releases on command.

    ArmTasks           generator-based pick-and-place routines driven one
                       control tick at a time by the mission controller:
                       load_from_table() and unload_to_truck().

Coordinates: unless stated otherwise every position is in the MODULE
frame. world = AMR pose (+ deck height) o module.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import pybullet as p

import mission_config as cfg

ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
JOINT_LOWER = np.array([-math.pi, -math.pi, -2.95, -math.pi, -math.pi, -math.pi])
JOINT_UPPER = -JOINT_LOWER

PAN_XYZ = np.array(cfg.ARM_SHOULDER_PAN_XYZ, dtype=np.float64)
LIFT_Z = PAN_XYZ[2] + cfg.ARM_SHOULDER_LIFT_DZ
L1 = cfg.ARM_UPPER_ARM_M
L2 = cfg.ARM_FOREARM_M
WRIST_TO_TIP = cfg.ARM_WRIST1_TO_WRIST2_M + cfg.ARM_WRIST2_TO_FLANGE_M + cfg.SUCTION_TIP_OFFSET_M

# Folded transport pose: upper arm vertical, forearm folded down in front
# of the column, cup hanging behind the sensor mast - inside the AMR
# footprint and out of the forward cameras' view.
STOW_Q = np.array([0.0, -math.pi / 2.0, 2.90, 0.0, 0.0, 0.0])
STOW_Q[3] = math.pi / 2.0 - (STOW_Q[1] + STOW_Q[2])

TRANSIT_TIP_Z = 0.72            # module-frame tip height for carrying moves
CARRY_VIA_TIP = (0.12, 0.0, 0.78)

JOINT_SPEED_RPS = 1.1
LINE_SPEED_MPS = 0.12
DESCEND_SPEED_MPS = 0.06
SETTLE_TOL_RAD = 0.02


# ============================================================
# SMALL TRANSFORM HELPERS
# ============================================================

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def homog(R=None, t=(0.0, 0.0, 0.0)):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = t
    return T


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def quat_to_R(q):
    return np.array(p.getMatrixFromQuaternion(q), dtype=np.float64).reshape(3, 3)


# ============================================================
# KINEMATICS
# ============================================================

class ArmKinematics:
    """FK / closed-form top-down IK for the module arm (module frame)."""

    @staticmethod
    def frames(q):
        """4x4 frames after each joint: [pan, lift, elbow, wrist1, wrist2, flange]."""

        out = []
        T = homog(rot_z(q[0]), PAN_XYZ)
        out.append(T)
        T = T @ homog(rot_y(q[1]), (0.0, 0.0, cfg.ARM_SHOULDER_LIFT_DZ))
        out.append(T)
        T = T @ homog(rot_y(q[2]), (L1, 0.0, 0.0))
        out.append(T)
        T = T @ homog(rot_y(q[3]), (L2, 0.0, 0.0))
        out.append(T)
        T = T @ homog(rot_z(q[4]), (cfg.ARM_WRIST1_TO_WRIST2_M, 0.0, 0.0))
        out.append(T)
        T = T @ homog(rot_x(q[5]), (cfg.ARM_WRIST2_TO_FLANGE_M, 0.0, 0.0))
        out.append(T)
        return out

    @classmethod
    def flange(cls, q):
        return cls.frames(q)[-1]

    @classmethod
    def tip(cls, q):
        return cls.flange(q) @ homog(None, (cfg.SUCTION_TIP_OFFSET_M, 0.0, 0.0))

    @classmethod
    def camera(cls, q):
        return cls.flange(q) @ homog(None, cfg.ARM_CAMERA_OFFSET_FLANGE)

    @staticmethod
    def camera_z_axis(psi):
        """Flange +Z (horizontal) for a top-down pose whose flange +Y points
        at angle psi in the module x-y plane."""

        return np.array([math.sin(psi), -math.cos(psi), 0.0])

    @classmethod
    def tip_for_camera_at(cls, camera_xyz, psi=math.pi / 2.0):
        """Tip position that puts the (top-down) stereo camera centre at
        camera_xyz."""

        ox, _oy, oz = cfg.ARM_CAMERA_OFFSET_FLANGE
        c = np.asarray(camera_xyz, dtype=np.float64)
        flange = c + np.array([0.0, 0.0, ox]) - oz * cls.camera_z_axis(psi)
        return flange - np.array([0.0, 0.0, cfg.SUCTION_TIP_OFFSET_M])

    @staticmethod
    def ik_top_down(tip_xyz, psi=math.pi / 2.0):
        """Closed-form IK: suction cup lip at tip_xyz, tool pointing straight
        down, flange +Y at heading psi. Returns 6 joint angles (elbow-up
        branch) or None if out of reach / outside joint limits."""

        tip = np.asarray(tip_xyz, dtype=np.float64)
        wrist = tip + np.array([0.0, 0.0, WRIST_TO_TIP])
        dx, dy = wrist[0] - PAN_XYZ[0], wrist[1] - PAN_XYZ[1]
        r = math.hypot(dx, dy)
        z = wrist[2] - LIFT_Z
        q1 = math.atan2(dy, dx)
        d = (r * r + z * z - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
        if d > 0.999 or d < -0.999:
            return None
        beta = math.acos(d)
        e1 = math.atan2(z, r) + math.atan2(L2 * math.sin(beta), L1 + L2 * math.cos(beta))
        q2 = -e1
        q3 = beta
        q4 = wrap(math.pi / 2.0 - (q2 + q3))
        q5 = 0.0
        q6 = wrap(q1 + math.pi / 2.0 - psi)
        q = np.array([q1, q2, q3, q4, q5, q6])
        if np.any(q < JOINT_LOWER) or np.any(q > JOINT_UPPER):
            return None
        return q


# ============================================================
# PYBULLET-SIDE ARM INTERFACE
# ============================================================

def _link_index(body, name):
    for j in range(p.getNumJoints(body)):
        if p.getJointInfo(body, j)[12].decode() == name:
            return j
    raise RuntimeError(f"link {name} not found")


class ArmHardware:
    """Joint encoders/motors of the module arm in PyBullet."""

    def __init__(self, module_id):
        self.module_id = module_id
        names = {p.getJointInfo(module_id, j)[1].decode(): j for j in range(p.getNumJoints(module_id))}
        self.joints = [names[n] for n in ARM_JOINT_NAMES]
        self.cup_link = _link_index(module_id, "suction_cup_link")
        self.camera_link = _link_index(module_id, "arm_stereo_camera_link")
        self.q_cmd = STOW_Q.copy()
        for j, a in zip(self.joints, STOW_Q):
            p.resetJointState(module_id, j, a)
        self.command(STOW_Q)

    def q(self):
        return np.array([s[0] for s in p.getJointStates(self.module_id, self.joints)])

    def command(self, q, max_velocity=2.0):
        self.q_cmd = np.asarray(q, dtype=np.float64).copy()
        efforts = (80.0, 80.0, 60.0, 30.0, 30.0, 20.0)
        for j, a, f in zip(self.joints, self.q_cmd, efforts):
            p.setJointMotorControl2(self.module_id, j, p.POSITION_CONTROL, targetPosition=float(a),
                                    maxVelocity=max_velocity, force=f, positionGain=0.3)

    def module_world_pose(self):
        """Physical module base pose (COM frame) - used only by the
        simulator side (rendering / constraints), never by perception."""

        pos, orn = p.getBasePositionAndOrientation(self.module_id)
        return np.array(pos), orn


# ============================================================
# EYE-IN-HAND STEREO CAMERA
# ============================================================

@dataclass
class DetectedBlock:
    x: float          # top-face centre, module frame
    y: float
    top_z: float
    size_m: float     # measured top-face footprint (max side)
    points: int
    partial: bool     # blob touches the image border -> re-centre first
    pixel: tuple = (0, 0)


@dataclass
class StereoFrame:
    left: np.ndarray
    right: np.ndarray
    disparity: np.ndarray
    points_module: np.ndarray        # (H, W, 3) module-frame points (nan where invalid)
    valid: np.ndarray                # (H, W) bool
    blocks: List[DetectedBlock] = field(default_factory=list)
    box_mask: Optional[np.ndarray] = None


class ArmStereoCamera:
    def __init__(self, hardware: ArmHardware):
        self.hw = hardware
        self.w, self.h = cfg.ARM_CAMERA_WIDTH, cfg.ARM_CAMERA_HEIGHT
        self.f = (self.h / 2.0) / math.tan(math.radians(cfg.ARM_CAMERA_VFOV_DEG) / 2.0)
        self.cx, self.cy = self.w / 2.0, self.h / 2.0
        self.baseline = cfg.ARM_STEREO_BASELINE_M
        self.proj = p.computeProjectionMatrixFOV(
            fov=cfg.ARM_CAMERA_VFOV_DEG, aspect=self.w / self.h,
            nearVal=cfg.ARM_CAMERA_NEAR, farVal=cfg.ARM_CAMERA_FAR,
        )
        block = 7
        # 64 px of disparity covers ranges down to f*B/64 ~ 0.24 m (the
        # closest look is 0.30 m above a box top).
        self.num_disp = 64
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=self.num_disp, blockSize=block,
            P1=8 * block * block, P2=32 * block * block,
            disp12MaxDiff=1, uniquenessRatio=8,
            speckleWindowSize=60, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self.last_frame: Optional[StereoFrame] = None

    # ---------------------------------------------------- rendering (sim)

    def _render_pair(self):
        """Physical camera: render from where the camera link really is."""

        state = p.getLinkState(self.hw.module_id, self.hw.camera_link, computeForwardKinematics=True)
        pos, orn = np.array(state[4]), state[5]
        R = quat_to_R(orn)
        forward, right = R[:, 0], R[:, 1]
        up = np.cross(right, forward)
        images = []
        for sign in (-1.0, 1.0):
            eye = pos + sign * 0.5 * self.baseline * right
            view = p.computeViewMatrix(eye.tolist(), (eye + forward).tolist(), up.tolist())
            img = p.getCameraImage(self.w, self.h, view, self.proj, renderer=p.ER_TINY_RENDERER)
            rgba = np.reshape(img[2], (self.h, self.w, 4)).astype(np.uint8)
            images.append(rgba[:, :, :3][:, :, ::-1].copy())
        return images[0], images[1]

    # ---------------------------------------------------- stereo -> 3D

    def capture(self):
        left, right = self._render_pair()
        gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        # SGBM leaves the first numDisparities columns of the left image
        # empty (it never searches past the image edge); padding both
        # images first recovers them, so a box near the left edge of the
        # view is still ranged as long as the right camera sees it.
        pad = self.num_disp
        gl = cv2.copyMakeBorder(gl, 0, 0, pad, 0, cv2.BORDER_REPLICATE)
        gr = cv2.copyMakeBorder(gr, 0, 0, pad, 0, cv2.BORDER_REPLICATE)
        disp = (self.matcher.compute(gl, gr).astype(np.float32) / 16.0)[:, pad:]
        valid = disp > 2.0
        Z = np.where(valid, self.f * self.baseline / np.maximum(disp, 1e-3), np.nan)
        valid &= Z < cfg.ARM_CAMERA_FAR

        us, vs = np.meshgrid(np.arange(self.w, dtype=np.float64), np.arange(self.h, dtype=np.float64))
        X = (us - self.cx) * Z / self.f
        Y = (vs - self.cy) * Z / self.f
        # Optical (x right, y down, z fwd) -> camera link (x fwd, y right,
        # z down); left eye sits -B/2 along the link's y.
        p_link = np.stack((Z, X - 0.5 * self.baseline, Y), axis=-1)

        # Camera pose from the arm's OWN encoders + FK (module frame).
        T = ArmKinematics.camera(self.hw.q())
        pts = p_link @ T[:3, :3].T + T[:3, 3]
        pts[~valid] = np.nan
        frame = StereoFrame(left=left, right=right, disparity=disp, points_module=pts, valid=valid)
        self.last_frame = frame
        return frame

    # ---------------------------------------------------- box finding

    @staticmethod
    def cardboard_mask(bgr):
        """Cardboard-brown pixels. Hue 8-17 / saturation <= 215 keeps the
        boxes (H~15, S 165-190 measured) and rejects the AMR's own yellow
        chassis (H~19, S~229) and the lane paint (H~25, S 255)."""

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((8, 100, 50), np.uint8), np.array((17, 215, 255), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return mask

    def find_blocks(self, frame: StereoFrame, region=None, min_area=150):
        """Boxes in view. Colour blobs in the left image are lifted to 3D by
        the stereo depth; inside each blob only the TOP-FACE pixels (within
        2 cm of the blob's highest level) are kept and split into connected
        components - neighbouring boxes merge into one colour blob through
        their visible side faces, but their top faces stay separate. Each
        top face gives one box: centre, height and size in the module frame.
        region: (x_min, x_max, y_min, y_max) module-frame filter."""

        mask = self.cardboard_mask(frame.left)
        frame.box_mask = mask
        z_img = frame.points_module[..., 2]
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        blocks = []
        for i in range(1, n):
            if stats[i][4] < min_area:
                continue
            blob_valid = (labels == i) & frame.valid
            if np.count_nonzero(blob_valid) < 40:
                continue
            top_level = float(np.percentile(z_img[blob_valid], 90))
            top_mask = (blob_valid & (z_img > top_level - 0.02)).astype(np.uint8)
            top_mask = cv2.erode(top_mask, np.ones((3, 3), np.uint8))
            m, tl, tst, _ = cv2.connectedComponentsWithStats(top_mask, 8)
            for k in range(1, m):
                x0, y0, bw, bh, area = tst[k]
                if area < min_area * 0.6:
                    continue
                pts = frame.points_module[tl == k]
                cx, cy = float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
                if region is not None and not (region[0] <= cx <= region[1] and region[2] <= cy <= region[3]):
                    continue
                ext_x = np.percentile(pts[:, 0], 97) - np.percentile(pts[:, 0], 3)
                ext_y = np.percentile(pts[:, 1], 97) - np.percentile(pts[:, 1], 3)
                partial = (x0 <= 3 or y0 <= 3 or x0 + bw >= self.w - 3 or y0 + bh >= self.h - 3)
                blocks.append(DetectedBlock(
                    x=cx, y=cy, top_z=float(np.median(pts[:, 2])),
                    size_m=float(max(ext_x, ext_y)), points=int(pts.shape[0]),
                    partial=bool(partial), pixel=(int(x0 + bw / 2), int(y0 + bh / 2)),
                ))
        frame.blocks = blocks
        return blocks

    def support_height(self, frame: StereoFrame, block: "DetectedBlock", radius=0.30):
        """Height of the surface a block stands on: the dominant level of
        the NON-box stereo points around it, below its top face. With the
        block's measured top this gives the box height - measured, not
        assumed."""

        pts = frame.points_module
        sel = frame.valid & (frame.box_mask == 0) if frame.box_mask is not None else frame.valid.copy()
        near = sel & (np.abs(pts[..., 0] - block.x) < radius) & (np.abs(pts[..., 1] - block.y) < radius)
        z = pts[..., 2][near]
        z = z[(z < block.top_z - 0.03) & (z > block.top_z - 0.60)]
        if z.size < 100:
            return None
        hist, edges = np.histogram(z, bins=60)
        k = int(np.argmax(hist))
        return float(np.median(z[(z >= edges[k] - 0.01) & (z <= edges[k + 1] + 0.01)]))

    def measure_surface(self, frame: StereoFrame, expected_z, band=0.06, lateral=0.45):
        """Dominant horizontal surface near expected_z (e.g. the truck
        deck): returns (height, near_edge_x, far_x) in the module frame,
        or None. near_edge_x is the closest surface point ahead (the sill)."""

        pts = frame.points_module[frame.valid]
        if pts.shape[0] < 200:
            return None
        sel = (np.abs(pts[:, 2] - expected_z) < band) & (np.abs(pts[:, 1]) < lateral) & (pts[:, 0] > 0.2)
        pts = pts[sel]
        if pts.shape[0] < 200:
            return None
        hist, edges = np.histogram(pts[:, 2], bins=24, range=(expected_z - band, expected_z + band))
        k = int(np.argmax(hist))
        z0 = 0.5 * (edges[k] + edges[k + 1])
        plane = pts[np.abs(pts[:, 2] - z0) < 0.012]
        if plane.shape[0] < 150:
            return None
        height = float(np.median(plane[:, 2]))
        near = float(np.percentile(plane[:, 0], 1.5))
        far = float(np.percentile(plane[:, 0], 98.5))
        return height, near, far


# ============================================================
# SUCTION GRIPPER
# ============================================================

class SuctionGripper:
    CONTACT_M = 0.012

    def __init__(self, hardware: ArmHardware, ignore_bodies=()):
        self.hw = hardware
        self.ignore = set(ignore_bodies) | {hardware.module_id}
        self.held = None
        self._constraint = None

    def _cup_tip_world(self):
        state = p.getLinkState(self.hw.module_id, self.hw.cup_link, computeForwardKinematics=True)
        R = quat_to_R(state[5])
        return np.array(state[4]) + R[:, 0] * cfg.SUCTION_TIP_OFFSET_M

    def sense_contact(self):
        """Vacuum-seal sensor: the movable object the cup lip is touching, or None."""

        tip = self._cup_tip_world()
        lo, hi = (tip - 0.04).tolist(), (tip + 0.04).tolist()
        for body, _link in (p.getOverlappingObjects(lo, hi) or []):
            if body in self.ignore or p.getDynamicsInfo(body, -1)[0] <= 0.0:
                continue
            close = p.getClosestPoints(self.hw.module_id, body, self.CONTACT_M,
                                       linkIndexA=self.hw.cup_link)
            if close:
                return body
        return None

    def grip(self, body):
        cup = p.getLinkState(self.hw.module_id, self.hw.cup_link, computeForwardKinematics=True)
        cup_com_pos, cup_com_orn = cup[0], cup[1]
        box_pos, box_orn = p.getBasePositionAndOrientation(body)
        inv_pos, inv_orn = p.invertTransform(cup_com_pos, cup_com_orn)
        rel_pos, rel_orn = p.multiplyTransforms(inv_pos, inv_orn, box_pos, box_orn)
        self._constraint = p.createConstraint(
            self.hw.module_id, self.hw.cup_link, body, -1, p.JOINT_FIXED, [0, 0, 0],
            parentFramePosition=rel_pos, childFramePosition=[0, 0, 0],
            parentFrameOrientation=rel_orn, childFrameOrientation=[0, 0, 0, 1],
        )
        p.changeConstraint(self._constraint, maxForce=300.0)
        p.setCollisionFilterPair(self.hw.module_id, body, self.hw.cup_link, -1, 0)
        self.held = body

    def release(self):
        if self._constraint is not None:
            p.removeConstraint(self._constraint)
        body = self.held
        self._constraint = None
        self.held = None
        return body

    def held_in_contact(self):
        """True when the held box touches anything except the cup itself."""

        if self.held is None:
            return False
        for c in p.getContactPoints(bodyA=self.held):
            if c[2] == self.hw.module_id and c[4] == self.hw.cup_link:
                continue
            if c[8] < 0.002:
                return True
        return False


# ============================================================
# CARGO LATCHES (tray clamps holding loaded boxes for transport)
# ============================================================

class CargoLatches:
    def __init__(self, module_id):
        self.module_id = module_id
        self.latched = {}           # body -> constraint id

    def latch(self, body):
        mpos, morn = p.getBasePositionAndOrientation(self.module_id)
        bpos, born = p.getBasePositionAndOrientation(body)
        ipos, iorn = p.invertTransform(mpos, morn)
        rpos, rorn = p.multiplyTransforms(ipos, iorn, bpos, born)
        c = p.createConstraint(self.module_id, -1, body, -1, p.JOINT_FIXED, [0, 0, 0],
                               parentFramePosition=rpos, childFramePosition=[0, 0, 0],
                               parentFrameOrientation=rorn, childFrameOrientation=[0, 0, 0, 1])
        p.changeConstraint(c, maxForce=400.0)
        self.latched[body] = c

    def unlatch(self, body):
        c = self.latched.pop(body, None)
        if c is not None:
            p.removeConstraint(c)

    @property
    def count(self):
        return len(self.latched)


# ============================================================
# PICK / PLACE TASKS (generators: one yield = one control tick)
# ============================================================

@dataclass
class ArmReport:
    picked: int = 0
    placed: int = 0
    observations: int = 0
    measured: list = field(default_factory=list)   # (x, y, top_z, size) per pick
    message: str = ""
    success: bool = False


class ArmTasks:
    def __init__(self, hardware: ArmHardware, camera: ArmStereoCamera,
                 gripper: SuctionGripper, latches: CargoLatches, dt):
        self.hw = hardware
        self.cam = camera
        self.grip = gripper
        self.latches = latches
        self.dt = dt
        self.status = "IDLE"
        self.report = ArmReport()
        self.tray_fill = 0
        self.truck_slots_used = 0
        self.box_height = None          # measured by stereo during loading
        self.box_footprint = None
        self.debug_frame = None
        self.verbose = True

    def log(self, msg):
        if self.verbose:
            print(f"[arm] {msg}")

    # ---------------------------------------------------- motion primitives

    def _settled(self):
        return float(np.max(np.abs(self.hw.q() - self.hw.q_cmd))) < SETTLE_TOL_RAD

    def move_joints(self, q_target, speed=JOINT_SPEED_RPS, timeout_s=8.0):
        q0 = self.hw.q_cmd.copy()
        q_target = np.asarray(q_target, dtype=np.float64)
        delta = q_target - q0
        duration = max(0.3, float(np.max(np.abs(delta))) / speed)
        t = 0.0
        while t < duration:
            t += self.dt
            s = min(1.0, t / duration)
            s = s * s * (3.0 - 2.0 * s)                  # smoothstep
            self.hw.command(q0 + s * delta)
            yield
        waited = 0.0
        while not self._settled() and waited < 1.5:
            waited += self.dt
            yield

    def move_tip_top_down(self, tip_xyz, psi=math.pi / 2.0, speed=JOINT_SPEED_RPS):
        q = ArmKinematics.ik_top_down(tip_xyz, psi)
        if q is None:
            raise RuntimeError(f"IK: tip {np.round(tip_xyz, 3)} unreachable")
        yield from self.move_joints(q, speed)

    def move_line(self, tip_to, psi=math.pi / 2.0, speed=LINE_SPEED_MPS, until=None):
        """Straight-line Cartesian move of the cup (IK every tick). Stops
        early when until() becomes true. Returns True if stopped by until."""

        start = ArmKinematics.tip(self.hw.q_cmd)[:3, 3]
        tip_to = np.asarray(tip_to, dtype=np.float64)
        dist = float(np.linalg.norm(tip_to - start))
        steps = max(1, int(math.ceil(dist / (speed * self.dt))))
        for k in range(1, steps + 1):
            target = start + (tip_to - start) * (k / steps)
            q = ArmKinematics.ik_top_down(target, psi)
            if q is None:
                raise RuntimeError(f"IK: line target {np.round(target, 3)} unreachable")
            self.hw.command(q)
            yield
            if until is not None and until():
                return True
        waited = 0.0
        while not self._settled() and waited < 1.0:
            waited += self.dt
            yield
            if until is not None and until():
                return True
        return False

    def wait(self, seconds):
        t = 0.0
        while t < seconds:
            t += self.dt
            yield

    def stow(self):
        tip = ArmKinematics.tip(self.hw.q_cmd)[:3, 3]
        if tip[2] < TRANSIT_TIP_Z - 0.05:
            yield from self.move_line((tip[0], tip[1], TRANSIT_TIP_Z), speed=0.2)
        yield from self.move_tip_top_down(CARRY_VIA_TIP)
        yield from self.move_joints(STOW_Q)

    @staticmethod
    def transit_height(x, y, z_min, psi=math.pi / 2.0):
        """Highest tip height <= TRANSIT_TIP_Z the arm can reach above
        (x, y) top-down (far targets cannot be reached at full transit
        height), but never below z_min."""

        z = TRANSIT_TIP_Z
        while z >= z_min - 1e-9:
            if ArmKinematics.ik_top_down((x, y, z), psi) is not None:
                return z
            z -= 0.02
        return None

    def _transit_to(self, tip_xyz, psi=math.pi / 2.0):
        """Lift to a safe transit height, swing via the carry pose above
        the column when crossing between the front and the tray, and come
        down above tip_xyz at transit height."""

        cur = ArmKinematics.tip(self.hw.q_cmd)[:3, 3]
        clear_z = max(tip_xyz[2], 0.45 if self.grip.held is not None else 0.30)
        z_goal = self.transit_height(tip_xyz[0], tip_xyz[1], clear_z, psi)
        if z_goal is None:
            raise RuntimeError(f"no reachable transit height above "
                               f"({tip_xyz[0]:+.3f},{tip_xyz[1]:+.3f})")
        z_here = self.transit_height(cur[0], cur[1], cur[2], psi) or cur[2]
        if cur[2] < z_here - 0.02:
            yield from self.move_line((cur[0], cur[1], z_here), psi, speed=0.15)
        cur_front = cur[0] > PAN_XYZ[0]
        goal_front = tip_xyz[0] > PAN_XYZ[0]
        if cur_front != goal_front:
            yield from self.move_tip_top_down(CARRY_VIA_TIP, psi)
        yield from self.move_tip_top_down((tip_xyz[0], tip_xyz[1], z_goal), psi)

    # ---------------------------------------------------- perception steps

    def observe(self, camera_xyz, region=None, settle_s=0.35):
        """Move the camera to camera_xyz (looking down), capture stereo and
        detect blocks. Result left in self.last_blocks / self.last_frame."""

        tip = ArmKinematics.tip_for_camera_at(camera_xyz)
        if ArmKinematics.ik_top_down(tip) is None:
            # Out of reach: slide the viewpoint toward the arm base, then
            # lower it, until the arm can get there.
            cx, cy, cz = camera_xyz
            for k in range(1, 16):
                cand = (PAN_XYZ[0] + (cx - PAN_XYZ[0]) * (1.0 - 0.04 * k), cy, cz - 0.01 * k)
                if ArmKinematics.ik_top_down(ArmKinematics.tip_for_camera_at(cand)) is not None:
                    self.log(f"viewpoint {np.round(camera_xyz, 2)} out of reach - using {np.round(cand, 2)}")
                    tip = ArmKinematics.tip_for_camera_at(cand)
                    break
            else:
                raise RuntimeError(f"no reachable viewpoint near {np.round(camera_xyz, 2)}")
        yield from self._transit_to(tip) if tip[2] < TRANSIT_TIP_Z - 0.02 else self.move_tip_top_down(tip)
        if np.linalg.norm(ArmKinematics.tip(self.hw.q_cmd)[:3, 3] - tip) > 0.01:
            yield from self.move_tip_top_down(tip)
        yield from self.wait(settle_s)
        frame = self.cam.capture()
        self.cam.find_blocks(frame, region=region)
        self.debug_frame = frame
        self.report.observations += 1
        self.last_frame = frame
        self.last_blocks = frame.blocks

    # ---------------------------------------------------- pick / place

    def pick_block(self, block: DetectedBlock):
        """Approach above, descend until the vacuum sensor reports a seal,
        grip, lift. Returns the gripped body id (or None)."""

        above = (block.x, block.y, block.top_z + 0.10)
        yield from self._transit_to(above)
        yield from self.move_tip_top_down(above, speed=0.8)
        self.status = "DESCEND_TO_BLOCK"
        target = (block.x, block.y, block.top_z - 0.025)
        yield from self.move_line(target, speed=DESCEND_SPEED_MPS,
                                  until=lambda: self.grip.sense_contact() is not None)
        body = self.grip.sense_contact()
        tip_now = ArmKinematics.tip(self.hw.q())[:3, 3]
        if body is None:
            self.log(f"no vacuum seal above ({block.x:+.3f},{block.y:+.3f}) "
                     f"- cup reached z {tip_now[2]:+.3f}; retrying from a fresh look")
            yield from self.move_line(above, speed=0.15)
            return None
        self.log(f"vacuum seal at cup ({tip_now[0]:+.3f},{tip_now[1]:+.3f},{tip_now[2]:+.3f})")
        self.grip.grip(body)
        self.latches.unlatch(body)          # no-op for boxes on the table
        self.status = "LIFT"
        yield from self.wait(0.15)
        lift_z = self.transit_height(block.x, block.y, block.top_z + 0.12) or (block.top_z + 0.12)
        yield from self.move_line((block.x, block.y, lift_z), speed=0.15)
        return body

    def place_held(self, place_xyz, surface_label):
        """Carry the held box over place_xyz (box-top centre target),
        descend until it touches the surface, release."""

        x, y, top = place_xyz
        self.log(f"placing on {surface_label} at ({x:+.3f},{y:+.3f}), box top {top:+.3f}")
        yield from self._transit_to((x, y, top + 0.10))
        yield from self.move_tip_top_down((x, y, top + 0.08), speed=0.8)
        self.status = f"PLACE_ON_{surface_label}"
        touched = yield from self.move_line((x, y, top - 0.02), speed=DESCEND_SPEED_MPS,
                                            until=self.grip.held_in_contact)
        if not touched:
            # Never release in the air: keep creeping down until the box
            # actually rests on something (up to 12 cm below the estimate).
            touched = yield from self.move_line((x, y, top - 0.14), speed=DESCEND_SPEED_MPS / 2,
                                                until=self.grip.held_in_contact)
            if not touched:
                self.log('place: no contact felt - releasing at the lowest safe height')
        yield from self.wait(0.15)
        body = self.grip.release()
        yield from self.wait(0.20)
        return body

    # ---------------------------------------------------- full routines

    def tray_slot_top(self, index, box_height=None):
        """Target box-top for cargo slot `index` (tray geometry is the
        robot's own body; the stacking height is the stereo-measured box
        height). The place move stops on contact anyway, so this is only
        the deepest the box is allowed to go."""

        h = box_height if box_height is not None else self.box_height
        layer = index // len(cfg.CARGO_SLOT_XY)
        sx, sy = cfg.CARGO_SLOT_XY[index % len(cfg.CARGO_SLOT_XY)]
        return (sx, sy, cfg.CARGO_TRAY_TOP_Z_M + (layer + 1) * h)

    def load_from_table(self, stock_hint_xy, top_hint_z, max_boxes=None, expected_size=0.18,
                        expected_height=None):
        """Pick every box in view at the pick table, one by one, onto the
        cargo tray. stock_hint_xy/top_hint_z: where the forward RGB-D
        detection put the stock (module frame), used only to aim the
        first look. expected_size: box footprint measured by the forward
        RGB-D detection (used to spot two top faces merged into one)."""

        capacity = len(cfg.CARGO_SLOT_XY) * cfg.CARGO_LAYERS
        max_boxes = capacity if max_boxes is None else min(max_boxes, capacity)
        hx, hy = stock_hint_xy
        view_xyz = (hx, hy, top_hint_z + 0.56)
        # Ahead of the AMR's bumper only (never its own chassis/mast).
        region = (cfg.AMR_HALF_LENGTH_M + 0.05, 1.1, -0.6, 0.6)
        empty_looks = 0
        try:
            while self.report.picked < max_boxes:
                self.status = "OBSERVE_STOCK"
                yield from self.observe(view_xyz, region=region)
                blocks = [b for b in self.last_blocks if 0.08 < b.size_m < cfg.SUCTION_MAX_BOX_FOOTPRINT_M]
                if not blocks:
                    empty_looks += 1
                    if empty_looks >= 2:
                        break
                    continue
                empty_looks = 0
                # Prefer complete, single-box-sized top faces: an oversized
                # 'block' is two tops merged at the view edge - re-look first.
                single = [b for b in blocks if b.size_m < 1.4 * expected_size]
                complete = [b for b in single if not b.partial] or single or blocks
                target = min(complete, key=lambda b: math.hypot(b.x - PAN_XYZ[0], b.y - PAN_XYZ[1]))
                overview_support = self.cam.support_height(self.last_frame, target, radius=0.45)
                # Close-range re-measure straight above the chosen box.
                self.status = "REFINE_BLOCK"
                yield from self.observe((target.x, target.y, target.top_z + 0.30),
                                        region=(target.x - 0.12, target.x + 0.12,
                                                target.y - 0.12, target.y + 0.12))
                refined = [b for b in self.last_blocks if not b.partial]
                if refined:
                    target = min(refined, key=lambda b: math.hypot(b.x - target.x, b.y - target.y))
                    support = self.cam.support_height(self.last_frame, target)
                    if support is None:
                        support = overview_support
                    if support is not None:
                        h = target.top_z - support
                        self.box_height = h if self.box_height is None else 0.5 * (self.box_height + h)
                        self.log(f"stereo: box stands on a surface at z {support:+.3f} "
                                 f"-> box height {h:.3f} m")
                    self.box_footprint = max(self.box_footprint or 0.0, target.size_m)
                self.report.measured.append((target.x, target.y, target.top_z, target.size_m))
                self.log(f"stereo: {len(blocks)} block(s) in view; target top-face centre "
                         f"({target.x:+.3f},{target.y:+.3f}) top z {target.top_z:+.3f} "
                         f"size {target.size_m:.3f} m")
                self.status = "PICK_BLOCK"
                body = yield from self.pick_block(target)
                if body is None:
                    continue
                self.report.picked += 1
                if self.box_height is None and expected_height is not None:
                    self.box_height = expected_height
                    self.log(f'stereo could not see the table beside the box - using the '
                             f'forward-camera box height {expected_height:.3f} m')
                if self.box_height is None:
                    raise RuntimeError("box height could not be measured")
                slot_top = self.tray_slot_top(self.tray_fill)
                self.status = "PLACE_ON_TRAY"
                placed = yield from self.place_held(slot_top, "TRAY")
                self.latches.latch(placed)
                self.tray_fill += 1
                self.report.placed += 1
                view_xyz = (hx, hy, top_hint_z + 0.56)
            self.status = "STOW"
            yield from self.stow()
            self.report.success = self.report.picked > 0
            self.report.message = f"loaded {self.report.picked} box(es) onto the AMR"
        except RuntimeError as exc:
            self.report.message = f"arm error: {exc}"
            self.report.success = False
            if self.grip.held is not None:
                self.grip.release()
        self.status = "DONE"
        return self.report

    def unload_to_truck(self, deck_hint_z):
        """Measure the truck deck with stereo, then move every box from the
        tray into a 2x2 grid just inside the sill."""

        self.report = ArmReport()
        try:
            self.status = "OBSERVE_TRUCK_DECK"
            yield from self.observe((0.75, 0.0, deck_hint_z + 0.62), region=(10, 11, 10, 11))
            surface = self.cam.measure_surface(self.last_frame, deck_hint_z)
            if surface is None:
                raise RuntimeError("could not range the truck deck with stereo")
            deck_z, sill_x, _far = surface
            self.deck_measurement = surface
            slots = None
            while True:
                self.status = "OBSERVE_TRAY"
                yield from self.observe((-0.20, 0.0, cfg.CARGO_TRAY_TOP_Z_M + 0.62),
                                        region=(-0.40, -0.05, -0.25, 0.25))
                blocks = [b for b in self.last_blocks if 0.08 < b.size_m < cfg.SUCTION_MAX_BOX_FOOTPRINT_M]
                if not blocks:
                    break
                target = max(blocks, key=lambda b: (round(b.top_z, 2), -abs(b.y)))
                self.report.measured.append((target.x, target.y, target.top_z, target.size_m))
                if slots is None:
                    # Box size from stereo: footprint from the loading looks
                    # (or this look), height from loading or from the
                    # stack on the robot's own tray.
                    foot = max(self.box_footprint or 0.0, target.size_m)
                    # (No loading history: assume a cube - the support seen
                    # under a stacked box would be the tray, not the box below.)
                    h = self.box_height if self.box_height is not None else foot
                    half = 0.5 * foot + 0.01
                    # First row: four boxes across the trailer just inside
                    # the sill (centre slots first); a second row behind.
                    slots = []
                    for row in range(2):
                        for k, sy in enumerate((0.5, -0.5, 1.5, -1.5)):
                            slots.append((sill_x + 0.05 + half + row * (2 * half + 0.03),
                                          sy * (2 * half + 0.03), deck_z + h))
                    self.log(f"truck slots from measured box {foot:.3f} m x {h:.3f} m high")
                self.status = "PICK_FROM_TRAY"
                body = yield from self.pick_block(target)
                if body is None:
                    continue
                self.report.picked += 1
                if self.truck_slots_used >= len(slots):
                    raise RuntimeError("truck slots full")
                self.status = "PLACE_IN_TRUCK"
                yield from self.place_held(slots[self.truck_slots_used], "TRUCK_DECK")
                self.truck_slots_used += 1
                self.report.placed += 1
                self.tray_fill = max(0, self.tray_fill - 1)
            self.status = "STOW"
            yield from self.stow()
            self.report.success = self.report.placed > 0
            self.report.message = (f"unloaded {self.report.placed} box(es) into the truck "
                                   f"(deck measured at z={deck_z:+.3f}, sill x={sill_x:+.3f} module)")
        except RuntimeError as exc:
            self.report.message = f"arm error: {exc}"
            self.report.success = False
            if self.grip.held is not None:
                self.grip.release()
        self.status = "DONE"
        return self.report


# ============================================================
# STANDALONE SELF-TEST
# ============================================================

def main():
    import pybullet_data
    from pathlib import Path

    here = Path(__file__).resolve().parent
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    module = p.loadURDF(str(here / "suction_arm_module.urdf"), [0, 0, 0.32], useFixedBase=True)
    hw = ArmHardware(module)

    # 1. FK agrees with PyBullet's link frames.
    rng = np.random.default_rng(1)
    flange_link = _link_index(module, "tool_flange_link")
    worst = 0.0
    for _ in range(20):
        q = rng.uniform(-2.0, 2.0, 6)
        for j, a in zip(hw.joints, q):
            p.resetJointState(module, j, a)
        st = p.getLinkState(module, flange_link, computeForwardKinematics=True)
        mine = ArmKinematics.flange(q)
        err = np.linalg.norm(np.array(st[4]) - (mine[:3, 3] + [0, 0, 0.32]))
        rot_err = np.linalg.norm(quat_to_R(st[5]) - mine[:3, :3])
        worst = max(worst, err, rot_err)
    print(f"FK vs PyBullet worst error: {worst:.2e}")
    assert worst < 1e-6

    # 2. IK round trip for the poses the mission uses.
    for name, tip in [("table far", (0.59, 0.30, 0.24)), ("tray low", (-0.22, 0.10, 0.19)),
                      ("tray high", (-0.22, -0.10, 0.35)), ("truck far", (0.78, -0.10, 0.24)),
                      ("observe", tuple(ArmKinematics.tip_for_camera_at((0.6, 0.0, 0.80)))),
                      ("carry", CARRY_VIA_TIP)]:
        q = ArmKinematics.ik_top_down(tip)
        assert q is not None, name
        T = ArmKinematics.tip(q)
        print(f"  IK {name:10s}: pos err {np.linalg.norm(T[:3, 3] - tip):.1e} "
              f"tool axis z {T[2, 0]:+.3f}")
        assert np.linalg.norm(T[:3, 3] - tip) < 1e-9 and T[2, 0] < -0.9999
    stow_tip = ArmKinematics.tip(STOW_Q)[:3, 3]
    print(f"  stow tip (module): {np.round(stow_tip, 3)}")

    # 3. Stereo ranging of a textured box placed under the camera.
    from warehouse_environment import _make_texture, _apply_texture
    box_xyz = np.array([0.62, 0.05, 0.08 + 0.08])          # module frame centre
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.08] * 3)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.08] * 3, rgbaColor=cfg.BOX_COLOR_RGBA)
    box = p.createMultiBody(0, col, vis, (box_xyz + [0, 0, 0.32]).tolist())
    _apply_texture(box, _make_texture("cardboard", [64, 133, 204], seed=7, noise=22,
                                      stripe_bgr=[45, 95, 160]))
    table_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.5, 0.02], rgbaColor=cfg.PICK_TABLE_COLOR)
    table = p.createMultiBody(0, -1, table_vis, [0.65, 0.0, 0.32 + 0.06])
    _apply_texture(table, _make_texture("pick_table", [127, 107, 76], seed=11))
    q = ArmKinematics.ik_top_down(ArmKinematics.tip_for_camera_at((0.6, 0.0, 0.80)))
    for j, a in zip(hw.joints, q):
        p.resetJointState(module, j, a)
    cam = ArmStereoCamera(hw)
    frame = cam.capture()
    blocks = cam.find_blocks(frame)
    print(f"  stereo valid px: {np.count_nonzero(frame.valid)}  blocks: "
          f"{[(round(b.x, 3), round(b.y, 3), round(b.top_z, 3), round(b.size_m, 3)) for b in blocks]}")
    assert len(blocks) == 1
    b = blocks[0]
    err = math.hypot(b.x - box_xyz[0], b.y - box_xyz[1])
    print(f"  block centre error {err*100:.1f} cm, top error {(b.top_z - 0.24)*100:+.1f} cm")
    assert err < 0.02 and abs(b.top_z - 0.24) < 0.02
    p.disconnect()
    print("SELF-TEST PASSED")


if __name__ == "__main__":
    main()
