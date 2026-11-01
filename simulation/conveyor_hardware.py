"""
INDOOR LOCALIZATION - CONVEYOR MODULE HARDWARE (lift, belt, IR, ToF)
=====================================================================

Simulator-side driver for conveyor_module.urdf, the swappable conveyor
the AMR carries for boxes too big for the suction arm.

    lift      conveyor_lift_joint (prismatic): belt-top height, set from the
              measured shelf / truck-deck height. Read back from the joint
              encoder.
    belt      PyBullet has no moving-surface primitive, so belt motion is
              applied as TRACTION: every physics step, each object touching
              the belt surface gets a force along the belt that drives its
              along-belt velocity toward the belt speed, limited to what
              belt friction could deliver (BELT_MAX_FORCE_N). An object
              that is not touching the belt is not moved.
    IR beams  two through-beam sensors across the belt (near the nose and
              near the front end stop): blocked when anything other than
              the robot itself is in the beam.
    ToF       rear-facing range sensor under the nose: distance to the
              first surface behind the nose (a shelf edge), or None.
    contact   belt-surface contact sensing (the "touch" when the lifting
              belt meets the underside of the box).

All sensor readings come from ray casts / contacts against the physical
scene - the same information real IR, ToF and load sensing would give.
"""

import numpy as np
import pybullet as p

import mission_config as cfg

BELT_TRACTION_GAIN = 80.0         # 1/s: velocity error -> acceleration (stiff belt grip)
BELT_MAX_FORCE_N = 60.0           # what belt friction on a ~3 kg carton can deliver
TOF_MAX_RANGE_M = 1.5


def _link(body, name):
    for j in range(p.getNumJoints(body)):
        if p.getJointInfo(body, j)[12].decode() == name:
            return j
    raise RuntimeError(f"link {name} missing")


def _joint(body, name):
    for j in range(p.getNumJoints(body)):
        if p.getJointInfo(body, j)[1].decode() == name:
            return j
    raise RuntimeError(f"joint {name} missing")


class ConveyorHardware:
    def __init__(self, module_id, ignore_bodies=()):
        self.module_id = module_id
        self.ignore = set(ignore_bodies) | {module_id}
        self.lift_joint = _joint(module_id, "conveyor_lift_joint")
        self.frame_link = _link(module_id, "conveyor_frame_link")
        self.belt_link = _link(module_id, "belt_link")
        self.belt_speed = 0.0          # + = toward the front end, - = out over the nose
        self.lift_target = 0.0
        self.set_lift(cfg.CONV_LIFT_BASE_Z_M)

    # ------------------------------------------------------------ lift

    def set_lift(self, belt_top_module_z, speed=0.12):
        self.lift_target = float(np.clip(belt_top_module_z - cfg.CONV_LIFT_BASE_Z_M,
                                         0.0, cfg.CONV_LIFT_MAX_M))
        p.setJointMotorControl2(self.module_id, self.lift_joint, p.POSITION_CONTROL,
                                targetPosition=self.lift_target, force=1500.0,
                                maxVelocity=speed, positionGain=0.4)

    def belt_top(self):
        """Belt-top height in the module frame, from the lift encoder."""

        return cfg.CONV_LIFT_BASE_Z_M + p.getJointState(self.module_id, self.lift_joint)[0]

    def lift_settled(self, tol=0.004):
        return abs(p.getJointState(self.module_id, self.lift_joint)[0] - self.lift_target) < tol

    # ------------------------------------------------------------ belt

    def run_belt(self, speed_mps):
        self.belt_speed = float(speed_mps)

    def _frame_pose(self):
        st = p.getLinkState(self.module_id, self.frame_link, computeForwardKinematics=True)
        R = np.array(p.getMatrixFromQuaternion(st[5])).reshape(3, 3)
        return np.array(st[4]), R

    def _to_world(self, local):
        pos, R = self._frame_pose()
        return pos + R @ np.asarray(local, dtype=np.float64)

    def apply_traction(self):
        """Call before EVERY physics step (external forces last one step)."""

        if abs(self.belt_speed) < 1e-6:
            return
        _pos, R = self._frame_pose()
        direction = R[:, 0]                       # module +x in world
        bodies = {c[2] for c in p.getContactPoints(bodyA=self.module_id, linkIndexA=self.belt_link)
                  if c[2] not in self.ignore}
        for body in bodies:
            mass = p.getDynamicsInfo(body, -1)[0]
            if mass <= 0.0:
                continue
            vel = np.array(p.getBaseVelocity(body)[0])
            err = self.belt_speed - float(np.dot(vel, direction))
            force = float(np.clip(BELT_TRACTION_GAIN * err * mass, -BELT_MAX_FORCE_N, BELT_MAX_FORCE_N))
            com = p.getBasePositionAndOrientation(body)[0]
            p.applyExternalForce(body, -1, (direction * force).tolist(), com, p.WORLD_FRAME)

    # ------------------------------------------------------------ sensors

    def _ray_hit(self, a_local, b_local):
        a, b = self._to_world(a_local), self._to_world(b_local)
        hit = p.rayTest(a.tolist(), b.tolist())[0]
        if hit[0] < 0 or hit[0] in self.ignore:
            return None
        return hit[2] * float(np.linalg.norm(b - a))

    def ir_blocked(self, which):
        x = cfg.CONV_IR_NOSE_X_M if which == "nose" else cfg.CONV_IR_END_X_M
        z = cfg.CONV_IR_HEIGHT_M
        return self._ray_hit((x, 0.29, z), (x, -0.29, z)) is not None

    def tof_range(self):
        x, dz = cfg.CONV_TOF_X_M, cfg.CONV_TOF_DZ_M
        return self._ray_hit((x - 0.005, 0.0, dz), (x - 0.005 - TOF_MAX_RANGE_M, 0.0, dz))

    def _load_contacts(self):
        """Belt load cell: contacts that press DOWN on the belt surface
        (contact normal near vertical). A side brush against a shelf face
        or a rail is not a load and is ignored."""

        _pos, R = self._frame_pose()
        up = R[:, 2]
        return [c for c in p.getContactPoints(bodyA=self.module_id, linkIndexA=self.belt_link)
                if c[2] not in self.ignore and abs(float(np.dot(c[7], up))) > 0.7]

    def belt_touching(self):
        return len(self._load_contacts()) > 0

    def belt_load_bodies(self):
        return sorted({c[2] for c in self._load_contacts()})

    def status(self):
        return {"belt_top": self.belt_top(), "belt_speed": self.belt_speed,
                "ir_nose": self.ir_blocked("nose"), "ir_end": self.ir_blocked("end"),
                "tof": self.tof_range()}
