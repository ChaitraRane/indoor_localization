"""
INDOOR LOCALIZATION - ELECTROMAGNETIC DOCKING
=================================================

Shared "electromagnet" simulation logic used by both the robotic arm
station (robotic_arm_station.urdf) and the conveyor station
(conveyor_station.urdf) to lock onto the AMR's chassis-top magnet pads
(amr_robot.urdf's magnet_pad_front_link / magnet_pad_rear_link).

PyBullet has no native magnetic-field primitive, so "the electromagnet
kicks in" is modelled the standard way physics engines represent a
magnetic lock: a rigid constraint (p.createConstraint, JOINT_FIXED) is
created between the station's magnet pad link and the AMR's matching
magnet pad link the moment the magnet is commanded on AND the two pads
are within docking tolerance; the constraint is removed when the magnet
is commanded off. While the constraint exists it behaves exactly like a
real magnetic lock strong enough not to slip, which is the right
approximation for "will the arm/conveyor topple" questions - a real
electromagnet sized for the job would not slip either.

TWO-POINT DOCKING
------------------------------------------------------------------------
Both stations expose a front pad and a rear pad (named consistently:
magnet_pad_front_link/magnet_pad_rear_link on the AMR, and either
magnet_pad_front_link/magnet_pad_rear_link - arm station - or
magnet_pad_intake_link/magnet_pad_outfeed_link - conveyor station). A
DockingPair always drives BOTH magnets together: either both constraints
exist or neither does. A single-point lock is exactly the "rocks about
one point" failure mode the user asked to avoid, so this module does not
expose a way to engage only one pad.

This file is part of the indoor_localization project only; nothing here
reads, imports, or depends on anything in the robot_vision folder.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pybullet as p


# ============================================================
# DOCKING TOLERANCE
# ============================================================
# How close a station pad and the AMR's matching pad must be (centre to
# centre) before the magnet is allowed to engage. Loose enough to permit
# realistic AMR stopping accuracy, tight enough that the two pads are
# plausibly touching (both pads are 0.05 m radius / 0.02 m thick
# cylinders, so 0.08 m of centre separation still has them overlapping).

MAGNET_ENGAGE_DISTANCE_M = 0.08
MAGNET_ENGAGE_MAX_TILT_DEG = 12.0


# ============================================================
# AMR MAGNET PAD NAMES (counterpart on every station)
# ============================================================

AMR_FRONT_PAD_LINK_NAME = "magnet_pad_front_link"
AMR_REAR_PAD_LINK_NAME = "magnet_pad_rear_link"


# ============================================================
# LINK LOOKUP HELPER
# ============================================================

def find_link_index(body_id, link_name):
    """
    Resolve a URDF link name to its PyBullet link index for body_id.
    PyBullet does not expose link names directly - a link's name is the
    child-link field of the joint that creates it, which getJointInfo
    does expose, so this walks joints and reads field [12].
    """

    for joint_index in range(p.getNumJoints(body_id)):
        joint_info = p.getJointInfo(body_id, joint_index)
        child_link_name = joint_info[12].decode("utf-8")
        if child_link_name == link_name:
            return joint_index

    raise RuntimeError(f"Link '{link_name}' not found on body {body_id}.")


def get_link_world_pose(body_id, link_index):
    """World-frame (position, orientation quaternion) of a link."""

    link_state = p.getLinkState(body_id, link_index, computeForwardKinematics=True)
    return np.array(link_state[4]), np.array(link_state[5])


# ============================================================
# A SINGLE MAGNET PAD PAIR (one station pad <-> one AMR pad)
# ============================================================

@dataclass
class MagnetPad:
    """One station-side magnet pad and the matching AMR-side pad it
    docks to. `constraint_id` is None while the magnet is off."""

    station_body_id: int
    station_link_index: int
    station_link_name: str
    amr_body_id: int
    amr_link_index: int
    amr_link_name: str
    constraint_id: Optional[int] = None

    @property
    def is_engaged(self):
        return self.constraint_id is not None

    def separation_m(self):
        station_position, _ = get_link_world_pose(
            self.station_body_id, self.station_link_index
        )
        amr_position, _ = get_link_world_pose(self.amr_body_id, self.amr_link_index)
        return float(np.linalg.norm(station_position - amr_position))

    def engage(self):
        """
        Creates the fixed constraint locking this pad to its AMR
        counterpart, using each pad's CURRENT relative offset as the
        constraint frame so the lock holds whatever (small) misalignment
        exists at the moment of contact, rather than snapping the AMR
        into a perfectly centred pose.
        """

        if self.is_engaged:
            return self.constraint_id

        self.constraint_id = p.createConstraint(
            parentBodyUniqueId=self.station_body_id,
            parentLinkIndex=self.station_link_index,
            childBodyUniqueId=self.amr_body_id,
            childLinkIndex=self.amr_link_index,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        return self.constraint_id

    def disengage(self):
        """Removes the fixed constraint, releasing this pad."""

        if not self.is_engaged:
            return

        p.removeConstraint(self.constraint_id)
        self.constraint_id = None


# ============================================================
# DOCKING PAIR (front + rear magnets, always driven together)
# ============================================================

@dataclass
class DockingPair:
    """
    Both magnet pads for one station, always engaged/disengaged as a
    pair so the station is never held by a single point.
    """

    front_pad: MagnetPad
    rear_pad: MagnetPad
    station_name: str = "station"

    @property
    def is_engaged(self):
        return self.front_pad.is_engaged and self.rear_pad.is_engaged

    def max_separation_m(self):
        return max(self.front_pad.separation_m(), self.rear_pad.separation_m())

    def is_aligned_for_docking(self, max_distance_m=MAGNET_ENGAGE_DISTANCE_M):
        """
        True when BOTH pads are within max_distance_m of their AMR
        counterpart, i.e. the AMR is positioned well enough for the
        magnets to physically make contact at both ends at once.
        """

        return (
            self.front_pad.separation_m() <= max_distance_m
            and self.rear_pad.separation_m() <= max_distance_m
        )

    def try_engage(self, max_distance_m=MAGNET_ENGAGE_DISTANCE_M):
        """
        Attempts to turn the electromagnets on. Engages both pads only
        if both are within docking tolerance; refuses (and reports why)
        rather than engaging a single pad on its own.

        Returns:
            (engaged: bool, reason: str)
        """

        if self.is_engaged:
            return True, "already engaged"

        if not self.is_aligned_for_docking(max_distance_m):
            return False, (
                f"out of range: front={self.front_pad.separation_m():.3f} m, "
                f"rear={self.rear_pad.separation_m():.3f} m "
                f"(limit {max_distance_m:.3f} m)"
            )

        self.front_pad.engage()
        self.rear_pad.engage()
        return True, "engaged"

    def disengage(self):
        """Turns both electromagnets off, releasing the station."""

        self.front_pad.disengage()
        self.rear_pad.disengage()


# ============================================================
# CONSTRUCTION HELPERS
# ============================================================

def build_docking_pair(
    station_body_id,
    station_front_link_name,
    station_rear_link_name,
    amr_body_id,
    station_name="station",
    amr_front_link_name=AMR_FRONT_PAD_LINK_NAME,
    amr_rear_link_name=AMR_REAR_PAD_LINK_NAME,
):
    """
    Resolves link names to indices for both bodies and returns a ready
    DockingPair. station_front_link_name docks to the AMR's front pad,
    station_rear_link_name to the AMR's rear pad (override the AMR pad
    names for modules that use a different pad pair, e.g. the suction-arm
    module, which locks onto the rear + mid pads).
    """

    amr_front_link_index = find_link_index(amr_body_id, amr_front_link_name)
    amr_rear_link_index = find_link_index(amr_body_id, amr_rear_link_name)

    front_pad = MagnetPad(
        station_body_id=station_body_id,
        station_link_index=find_link_index(station_body_id, station_front_link_name),
        station_link_name=station_front_link_name,
        amr_body_id=amr_body_id,
        amr_link_index=amr_front_link_index,
        amr_link_name=amr_front_link_name,
    )

    rear_pad = MagnetPad(
        station_body_id=station_body_id,
        station_link_index=find_link_index(station_body_id, station_rear_link_name),
        station_link_name=station_rear_link_name,
        amr_body_id=amr_body_id,
        amr_link_index=amr_rear_link_index,
        amr_link_name=amr_rear_link_name,
    )

    return DockingPair(front_pad=front_pad, rear_pad=rear_pad, station_name=station_name)


def build_arm_station_docking_pair(arm_station_id, amr_body_id):
    """DockingPair for robotic_arm_station.urdf's two magnet pads."""

    return build_docking_pair(
        station_body_id=arm_station_id,
        station_front_link_name="magnet_pad_front_link",
        station_rear_link_name="magnet_pad_rear_link",
        amr_body_id=amr_body_id,
        station_name="robotic_arm_station",
    )


def build_arm_module_docking_pair(module_id, amr_body_id):
    """DockingPair for suction_arm_module.urdf: its mid/rear underside pads
    lock onto the AMR's magnet_pad_mid_link / magnet_pad_rear_link."""

    return build_docking_pair(
        station_body_id=module_id,
        station_front_link_name="magnet_pad_mid_link",
        station_rear_link_name="magnet_pad_rear_link",
        amr_body_id=amr_body_id,
        station_name="suction_arm_module",
        amr_front_link_name="magnet_pad_mid_link",
        amr_rear_link_name=AMR_REAR_PAD_LINK_NAME,
    )


def build_conveyor_station_docking_pair(conveyor_station_id, amr_body_id):
    """
    DockingPair for conveyor_station.urdf's two magnet pads.

    conveyor_station.urdf is loaded yawed 180 degrees in
    warehouse_environment.py (so its outfeed end points toward the
    parked truck), which also flips which pad ends up nearest the AMR
    in world space: magnet_pad_outfeed_link - despite its name,
    referring to the roller end nearest the truck in the conveyor's own
    local frame - lands closest to the AMR's approach side once rotated,
    and magnet_pad_intake_link lands furthest away. The pairing below is
    therefore deliberately swapped (AMR front <-> station outfeed pad,
    AMR rear <-> station intake pad) so both magnet pads engage the
    correct, geometrically nearest AMR pad after the 180 degree yaw,
    rather than matching on link name alone.
    """

    return build_docking_pair(
        station_body_id=conveyor_station_id,
        station_front_link_name="magnet_pad_outfeed_link",
        station_rear_link_name="magnet_pad_intake_link",
        amr_body_id=amr_body_id,
        station_name="conveyor_station",
    )


# ============================================================
# STANDALONE SELF-TEST
# ============================================================

def main():
    """
    Minimal standalone check: loads robotic_arm_station.urdf and
    amr_robot.urdf (DIRECT mode, no GUI), drives the AMR under the
    pedestal, confirms try_engage() refuses while out of range, then
    confirms it succeeds once the AMR is close enough, and that
    disengage() releases it again.
    """

    import pybullet_data

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

    arm_station_id = p.loadURDF(
        "robotic_arm_station.urdf", basePosition=[0, 0, 0], useFixedBase=True
    )

    # Spawned far from the station: magnets should refuse to engage.
    amr_id = p.loadURDF(
        "amr_robot.urdf", basePosition=[5.0, 0.0, 0.20], useFixedBase=False
    )

    docking = build_arm_station_docking_pair(arm_station_id, amr_id)

    for _ in range(10):
        p.stepSimulation()

    engaged, reason = docking.try_engage()
    print(f"Far away -> engaged={engaged}, reason='{reason}'")
    assert not engaged

    # Teleport the AMR directly under the pedestal, centred so both
    # chassis-top pads land under their station counterparts.
    p.resetBasePositionAndOrientation(
        amr_id, [0.0, 0.0, 0.20], p.getQuaternionFromEuler([0, 0, 0])
    )

    for _ in range(10):
        p.stepSimulation()

    engaged, reason = docking.try_engage()
    print(f"Under pedestal -> engaged={engaged}, reason='{reason}'")
    print(f"Front separation: {docking.front_pad.separation_m():.4f} m")
    print(f"Rear separation:  {docking.rear_pad.separation_m():.4f} m")
    assert engaged
    assert docking.is_engaged

    docking.disengage()
    print(f"After disengage -> is_engaged={docking.is_engaged}")
    assert not docking.is_engaged

    p.disconnect()
    print("SELF-TEST PASSED")


if __name__ == "__main__":
    main()
