"""
INDOOR LOCALIZATION - AUTONOMOUS MISSION STATE MACHINE
========================================================

The brain. It owns the perception / mapping / planning / manipulation
stack and drives the AMR + its swappable modules through every job in
cfg.JOBS, then home. It never uses ground truth: pose comes from fused
wheel + visual odometry corrected by the painted lane and by landmark
fixes on the tool stands; the occupancy and semantic maps are built live
from the RGB-D cameras; the arm sees boxes only through its own stereo
camera + joint encoders; the conveyor knows its load only through its
IR beams, ToF sensor and belt contact.

PER JOB (stock location A, B, ...)
    EXPLORE_TO_DETECTION   follow the lane (lane-preferring A* on the live
                           map + VFH) to the job's detection pose, detouring
                           round obstacles, mapping as it goes
    DETECT_BOXES           forward RGB-D (+YOLO) look at the stock
    DECIDE_PAYLOAD         box size/count -> which module is needed; the
                           stock location is pinned on the map; a plan of
                           steps is built for the rest of the job:

    fetch module           GO_TO_STATION -> LOCALIZE_AT_STATION (stand-leg
                           landmark fix) -> TURN_FOR_DOCK -> REVERSE_DOCK ->
                           TRANSFER_MODULE (hoist + magnets) -> LEAVE_STATION
    SUCTION_ARM job        GO_TO_PICK -> APPROACH_PICK -> ARM_LOAD ->
                           BACK_OFF -> GO_TO_TRUCK -> APPROACH_TRUCK ->
                           ARM_UNLOAD -> BACK_OFF
    CONVEYOR job           GO_TO_SHELF -> MEASURE_SHELF (shelf height, edge;
                           lift set) -> TURN_AROUND -> REVERSE_TO_SHELF (nose
                           ToF) -> CONVEYOR_LOAD (lift until the belt touches
                           the box, belt pulls it aboard until the IR beams
                           say it is fully on) -> PULL_AWAY -> GO_TO_TRUCK ->
                           MEASURE_TRUCK (deck height, sill) -> TURN_AROUND
                           -> REVERSE_TO_TRUCK -> CONVEYOR_UNLOAD (belt out
                           until both beams clear) -> PULL_AWAY
    return module          the same docking steps, but the hoist takes the
                           module back off the deck

AFTER THE LAST JOB
    GO_HOME -> MISSION_COMPLETE
"""

import math
from collections import deque

import numpy as np

import box_detector as boxmod
import camera_rig
import lane_detector
from lane_localizer import LaneLocalizer
import mission_config as cfg
from arm_manipulator import ArmTasks
from navigation import Navigator, NAV_ARRIVED, NAV_FAILED
from occupancy_map import OccupancyMap
from semantic_map import SemanticMap
import visual_odometry as vo
from visual_odometry import Pose2D, PoseEstimator, StereoVisualOdometry


STATE_EXPLORE = "EXPLORE_TO_DETECTION"
STATE_DETECT = "DETECT_BOXES"
STATE_DECIDE = "DECIDE_PAYLOAD"
STATE_GO_STATION = "GO_TO_STATION"
STATE_LOCALIZE = "LOCALIZE_AT_STATION"
STATE_TURN_DOCK = "TURN_FOR_DOCK"
STATE_REVERSE_DOCK = "REVERSE_DOCK"
STATE_TRANSFER = "TRANSFER_MODULE"
STATE_LEAVE = "LEAVE_STATION"
STATE_GO_PICK = "GO_TO_PICK"
STATE_APPROACH_PICK = "APPROACH_PICK"
STATE_ARM_LOAD = "ARM_LOAD"
STATE_BACK_OFF = "BACK_OFF"
STATE_GO_TRUCK = "GO_TO_TRUCK"
STATE_APPROACH_TRUCK = "APPROACH_TRUCK"
STATE_ARM_UNLOAD = "ARM_UNLOAD"
STATE_GO_SHELF = "GO_TO_SHELF"
STATE_MEASURE_SHELF = "MEASURE_SHELF"
STATE_TURN_AROUND = "TURN_AROUND"
STATE_REVERSE_SHELF = "REVERSE_TO_SHELF"
STATE_CONV_LOAD = "CONVEYOR_LOAD"
STATE_PULL_AWAY = "PULL_AWAY"
STATE_MEASURE_TRUCK = "MEASURE_TRUCK"
STATE_REVERSE_TRUCK = "REVERSE_TO_TRUCK"
STATE_CONV_UNLOAD = "CONVEYOR_UNLOAD"
STATE_GO_HOME = "GO_HOME"
STATE_COMPLETE = "MISSION_COMPLETE"
STATE_FAILED = "FAILED"

NAV_STATES = (STATE_EXPLORE, STATE_GO_STATION, STATE_GO_PICK, STATE_GO_TRUCK,
              STATE_GO_SHELF, STATE_GO_HOME)
ARM_STATES = (STATE_ARM_LOAD, STATE_ARM_UNLOAD)
CONVEYOR_STATES = (STATE_MEASURE_SHELF, STATE_REVERSE_SHELF, STATE_CONV_LOAD,
                   STATE_MEASURE_TRUCK, STATE_REVERSE_TRUCK, STATE_CONV_UNLOAD)
# States in which the forward cameras are rendered + mapped every tick.
# REVERSE_DOCK included: reversing under a stand the front camera looks
# straight down the stand's lane spur, so the lane fix keeps the heading
# and lateral offset honest all the way in.
PERCEPTION_STATES = NAV_STATES + (STATE_DETECT, STATE_LOCALIZE, STATE_TURN_DOCK, STATE_LEAVE,
                                  STATE_REVERSE_DOCK,
                                  STATE_APPROACH_PICK, STATE_BACK_OFF, STATE_APPROACH_TRUCK,
                                  STATE_MEASURE_SHELF, STATE_TURN_AROUND, STATE_REVERSE_SHELF,
                                  STATE_PULL_AWAY, STATE_MEASURE_TRUCK, STATE_REVERSE_TRUCK)

DETECT_FRAMES = 4
CREEP_SPEED_MPS = 0.12
DOCK_REVERSE_SPEED_MPS = 0.12
EDGE_BAND_Z = (0.30, 0.47)        # world heights of a table / deck front edge
STATE_TIMEOUT_S = 240.0
CONVEYOR_HALF_WIDTH_M = 0.42      # footprint while the conveyor (lift guides) is aboard


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class MissionController:
    def __init__(self, robot_io):
        self.io = robot_io
        self.dt = robot_io.control_dt
        start = Pose2D(cfg.ROBOT_START_XY[0], cfg.ROBOT_START_XY[1], cfg.ROBOT_START_YAW_RAD)
        self.pose_estimator = PoseEstimator(cfg.ODOM_WHEEL_RADIUS_M, cfg.ODOM_WHEEL_SEPARATION_M, start)
        self.intrinsics = camera_rig.RGBD_INTRINSICS
        self.vo = StereoVisualOdometry(self.intrinsics)
        self.occupancy = OccupancyMap()
        self.semantic = SemanticMap()
        self.nav = Navigator(self.occupancy)
        self.nav.set_half_width(cfg.AMR_HALF_WIDTH_M)
        self.lane_localizer = LaneLocalizer()
        self.local = self.nav.local
        self.box_detector = robot_io.box_detector
        self.arm = ArmTasks(robot_io.arm_hw, robot_io.arm_camera, robot_io.gripper,
                            robot_io.latches, self.dt)
        self.conveyor = robot_io.conveyor

        self.jobs = list(cfg.JOBS)
        self.job_idx = 0
        self.job_reports = []
        self.plan = deque()
        self.step_args = {}
        self.station_kind = None
        self.dock_purpose = None

        self.state = STATE_EXPLORE
        self.state_t0 = 0.0
        self.t = 0.0
        self.tick = 0
        self.result_message = ""
        self.payload_decision = None
        self.detections = []
        self.stock_hint = None
        self.load_report = None
        self.unload_report = None
        self.station_fix = None
        self.events = []
        self.conv = {}                 # conveyor job measurements
        self._first_wheels = True
        self._phase = None
        self._phase_t0 = 0.0
        self._task = None
        self._ref_xy = None
        self._edge_dist = None
        self._edge_ref_xy = None
        self._last_obstacles = np.empty((0, 2))
        self._last_lane = None
        self._dock_attempts = 0
        self._station_attempts = 0
        self._lookbacks = 0
        print(f"[mission] job 1/{len(self.jobs)}: stock location {self.job['name']}")

    # ------------------------------------------------------- helpers

    @property
    def job(self):
        return self.jobs[min(self.job_idx, len(self.jobs) - 1)]

    @property
    def active_path(self):
        return self.nav.waypoints if self.state in NAV_STATES else None

    @property
    def finished(self):
        return self.state in (STATE_COMPLETE, STATE_FAILED)

    def _set_state(self, state, note=""):
        self.events.append((round(self.t, 1), state, note))
        print(f"[mission t={self.t:6.1f}s] {self.state} -> {state}" + (f"  ({note})" if note else ""))
        self.state = state
        self.state_t0 = self.t
        self._phase = None
        self._task = None
        self._ref_xy = None

    def _advance(self, note=""):
        """Move on to the next planned step."""

        if not self.plan:
            self._fail("plan exhausted")
            return
        state, args = self.plan.popleft()
        self.step_args = args
        if "station" in args and (args["station"], args["purpose"]) != (self.station_kind, self.dock_purpose):
            # A new docking task (re-planned steps for the same one keep
            # their attempt counters).
            self.station_kind = args["station"]
            self.dock_purpose = args["purpose"]
            self._station_attempts = 0
            self._dock_attempts = 0
            self._lookbacks = 0
            self.station_fix = None
        self._set_state(state, note)

    def _fail(self, why):
        self.result_message = why
        self.io.conveyor.run_belt(0.0)
        self._set_state(STATE_FAILED, why)

    def _record(self, name, x, y, note):
        self.semantic.record_point(name, x, y, note)
        self.io.mark_point(name, x, y)
        print(f"[mission] pinned '{name}' on the map at ({x:+.2f}, {y:+.2f}) - {note}")

    def world_to_module(self, pose, x, y, z):
        dx, dy = x - pose.x, y - pose.y
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        return dx * c + dy * s, -dx * s + dy * c, z - cfg.MODULE_DOCKED_Z_M

    def _station_steps(self, kind, purpose):
        a = {"station": kind, "purpose": purpose}
        return [(STATE_GO_STATION, a), (STATE_LOCALIZE, a), (STATE_TURN_DOCK, a),
                (STATE_REVERSE_DOCK, a), (STATE_TRANSFER, a), (STATE_LEAVE, a)]

    def _build_plan(self, module):
        plan = []
        aboard = self.io.module_aboard
        if aboard is not None and aboard != module:
            plan += self._station_steps(aboard, "drop")
        if aboard != module:
            plan += self._station_steps(module, "pick")
        if module == "SUCTION_ARM":
            plan += [(STATE_GO_PICK, {}), (STATE_APPROACH_PICK, {}), (STATE_ARM_LOAD, {}),
                     (STATE_BACK_OFF, {}), (STATE_GO_TRUCK, {"module": module}),
                     (STATE_MEASURE_TRUCK, {"module": module}),
                     (STATE_APPROACH_TRUCK, {}), (STATE_ARM_UNLOAD, {}), (STATE_BACK_OFF, {})]
        else:
            plan += [(STATE_GO_SHELF, {}), (STATE_MEASURE_SHELF, {}),
                     (STATE_TURN_AROUND, {}), (STATE_REVERSE_SHELF, {}), (STATE_CONV_LOAD, {}),
                     (STATE_PULL_AWAY, {}), (STATE_GO_TRUCK, {"module": module}),
                     (STATE_MEASURE_TRUCK, {"module": module}), (STATE_TURN_AROUND, {}),
                     (STATE_REVERSE_TRUCK, {}), (STATE_CONV_UNLOAD, {}), (STATE_PULL_AWAY, {})]
        # Every job ends by putting its module back on its stand.
        plan += self._station_steps(module, "drop")
        plan += [("NEXT_JOB", {})]
        self.plan = deque(plan)

    def _next_job(self):
        self.job_reports.append({"job": self.job["name"],
                                 "module": self.payload_decision.module if self.payload_decision else None})
        self.job_idx += 1
        self.detections = []
        self.payload_decision = None
        if self.job_idx < len(self.jobs):
            print(f"[mission] job {self.job_idx + 1}/{len(self.jobs)}: stock location {self.job['name']}")
            self._set_state(STATE_EXPLORE, f"job {self.job['name']}")
        else:
            self._set_state(STATE_GO_HOME, "all jobs done - returning to the home station")

    # ------------------------------------------------------- perception

    def _perceive(self, full):
        wheels = self.io.wheel_angles()
        gyro = self.io.gyro_yaw()
        if self._first_wheels:
            self.pose_estimator.reset_wheel_reference(*wheels)
            self._last_gyro = gyro
            self._last_wheels = wheels
            self._gyro_bias_est = 0.0
            self._first_wheels = False
        gyro_dyaw = gyro - self._last_gyro
        self._last_gyro = gyro
        # Zero-velocity update: when neither wheel turned this tick the robot
        # is not rotating, so whatever the gyro reports is bias - learn it
        # and do not integrate it (otherwise the heading creeps while the AMR
        # stands still for minutes during arm / conveyor work).
        still = max(abs(wheels[0] - self._last_wheels[0]), abs(wheels[1] - self._last_wheels[1])) < 2e-3
        self._last_wheels = wheels
        if still:
            self._gyro_bias_est += 0.05 * (gyro_dyaw / self.dt - self._gyro_bias_est)
            gyro_dyaw = 0.0
        else:
            gyro_dyaw -= self._gyro_bias_est * self.dt
        if not full:
            self.pose_estimator.update(wheels[0], wheels[1], None, gyro_dyaw=gyro_dyaw)
            return self.pose_estimator.get_pose(), None

        sensors = self.io.capture()
        vo_result = self.vo.update(sensors["left_bgr"], sensors["left_depth"],
                                   sensors["right_bgr"], sensors["right_depth"])
        self.pose_estimator.update(wheels[0], wheels[1], vo_result, gyro_dyaw=gyro_dyaw)
        pose = self.pose_estimator.get_pose()

        # Lane-aided localisation: the painted lane is surveyed
        # infrastructure, so seeing it corrects heading + lateral drift.
        self._last_lane = lane_detector.detect_lane(sensors["center_bgr"])
        # Lateral (cross-lane) fixes only while travelling between work
        # poses; docking / shelf / truck manoeuvres use their own landmark
        # sensing and must not be nudged mid-manoeuvre.
        fixed = self.lane_localizer.correct(pose, self._last_lane.mask, self.t,
                                            lateral=self.state in NAV_STATES)
        if fixed is not None:
            self.pose_estimator.pose = Pose2D(*fixed)
            pose = self.pose_estimator.get_pose()
        # Live lane layer: what the camera sees of the paint, voted into the map.
        self.occupancy.integrate_lane(self._last_lane.mask, self.lane_localizer.lut, pose)
        self.nav.lane_trim_enabled = self.t - self.lane_localizer.last_update_t > 1.0

        frames = []
        for side in ("left", "right"):
            eye, fwd, up = camera_rig.build_camera_frame(pose.x, pose.y, pose.yaw, side)
            self.occupancy.update_from_depth(sensors[f"{side}_depth"], self.intrinsics, eye, fwd, up)
            frames.append((sensors[f"{side}_depth"], self.intrinsics, eye, fwd, up))
        self.occupancy.mark_footprint_free(pose.x, pose.y, pose.yaw)
        self.occupancy.mark_lane_visited(pose.x, pose.y)
        self._last_obstacles = self.local.obstacles_body_frame(frames, pose)
        return pose, sensors

    def _forward_points(self, sensors, pose, stride=2, max_range=3.0):
        """Both forward depth images -> (world xyz, fwd, lat) arrays."""

        pts = []
        for side in ("left", "right"):
            eye, fwd, up = camera_rig.build_camera_frame(pose.x, pose.y, pose.yaw, side)
            world, _ = vo.project_depth_to_world(sensors[f"{side}_depth"], self.intrinsics,
                                                 eye, fwd, up, stride=stride, max_range_m=max_range)
            if world.shape[0]:
                pts.append(world)
        if not pts:
            return None
        world = np.concatenate(pts)
        dx, dy = world[:, 0] - pose.x, world[:, 1] - pose.y
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        return world, dx * c + dy * s, -dx * s + dy * c

    def _edge_distance(self, sensors, pose, band=EDGE_BAND_Z, lateral=0.30):
        """Distance from the bumper to the nearest surface edge in a height
        band straight ahead (table top / deck / shelf slab), or None."""

        res = self._forward_points(sensors, pose)
        if res is None:
            return None
        world, f, l = res
        z = world[:, 2]
        sel = (z > band[0]) & (z < band[1]) & (np.abs(l) < lateral) & (f > 0.3)
        if np.count_nonzero(sel) < 12:
            return None
        return float(np.percentile(f[sel], 3)) - cfg.AMR_HALF_LENGTH_M

    # ------------------------------------------------------- main update

    def update(self):
        self.tick += 1
        self.t += self.dt
        pose, sensors = self._perceive(self.state in PERCEPTION_STATES)

        if self.t - self.state_t0 > STATE_TIMEOUT_S and not self.finished:
            self._fail(f"timeout in {self.state}")

        if self.state == "NEXT_JOB":
            self._next_job()

        handler = {
            STATE_EXPLORE: self._do_explore,
            STATE_DETECT: self._do_detect,
            STATE_DECIDE: self._do_decide,
            STATE_GO_STATION: self._do_go_station,
            STATE_LOCALIZE: self._do_localize,
            STATE_TURN_DOCK: self._do_turn_dock,
            STATE_REVERSE_DOCK: self._do_reverse_dock,
            STATE_TRANSFER: self._do_transfer,
            STATE_LEAVE: self._do_leave,
            STATE_GO_PICK: self._do_go_pick,
            STATE_APPROACH_PICK: self._do_approach_pick,
            STATE_ARM_LOAD: self._do_arm_load,
            STATE_BACK_OFF: self._do_back_off,
            STATE_GO_TRUCK: self._do_go_truck,
            STATE_APPROACH_TRUCK: self._do_approach_truck,
            STATE_ARM_UNLOAD: self._do_arm_unload,
            STATE_GO_SHELF: self._do_go_shelf,
            STATE_MEASURE_SHELF: self._do_measure_shelf,
            STATE_TURN_AROUND: self._do_turn_around,
            STATE_REVERSE_SHELF: self._do_reverse_shelf,
            STATE_CONV_LOAD: self._do_conveyor_load,
            STATE_PULL_AWAY: self._do_pull_away,
            STATE_MEASURE_TRUCK: self._do_measure_truck,
            STATE_REVERSE_TRUCK: self._do_reverse_truck,
            STATE_CONV_UNLOAD: self._do_conveyor_unload,
            STATE_GO_HOME: self._do_go_home,
            STATE_COMPLETE: self._do_terminal,
            STATE_FAILED: self._do_terminal,
        }[self.state]
        handler(pose, sensors)

        return {
            "tick": self.tick, "t": round(self.t, 2), "state": self.state,
            "nav": self.nav.state if self.state in NAV_STATES else "",
            "arm": self.arm.status if self.state in ARM_STATES else "",
            "job": self.job["name"],
            "pose": (round(pose.x, 2), round(pose.y, 2), round(math.degrees(pose.yaw), 1)),
            "lane_visited_frac": round(self.occupancy.visited_lane_fraction(), 3),
        }

    # ------------------------------------------------------- navigation legs

    def _navigate(self, pose, goal, heading, tol, label, on_arrive=None):
        if self._phase is None:
            self._phase = "NAV"
            if not self.nav.set_goal(goal, pose, self.t, heading=heading, tolerance=tol, label=label):
                print(f"[mission] initial route failed: {self.nav.message} - will retry")
        cmd = self.nav.step(pose, self._last_obstacles, self._last_lane, self.t,
                            lane_fit=self.lane_localizer.last_fit)
        self.io.drive(cmd.linear, cmd.angular)
        if cmd.state == NAV_ARRIVED:
            self.io.stop()
            self.nav.clear()
            if on_arrive is None:
                self._advance(f"arrived at {label}")
            else:
                on_arrive()
        elif cmd.state == NAV_FAILED:
            self.io.stop()
            self._fail(f"navigation to {label} failed: {self.nav.message}")

    def _do_explore(self, pose, sensors):
        job = self.job
        self._navigate(pose, job["detect_xy"], job["heading"], 0.15,
                       f"detection pose for {job['name']}",
                       on_arrive=lambda: self._set_state(STATE_DETECT, "arrived at detection pose"))

    def _do_go_station(self, pose, sensors):
        st = cfg.STATIONS[self.station_kind]
        face_stand = wrap(st["yaw"] + math.pi)
        tol = 0.12 if self._station_attempts == 0 else 0.05
        self._navigate(pose, cfg.pre_dock_xy(self.station_kind), face_stand, tol,
                       f"{st['name']} pre-dock pose")

    def _do_go_pick(self, pose, sensors):
        a = self.semantic.get_point(cfg.POINT_A_NAME)
        self._navigate(pose, (a.x - 1.15, a.y), 0.0, 0.12, "pick approach")

    def _do_go_truck(self, pose, sensors):
        module = self.step_args.get("module", "SUCTION_ARM")
        tol = 0.10 if not self.conv.get("truck_attempts") else 0.04
        self._navigate(pose, cfg.truck_approach_xy(module), cfg.TRUCK_BOARDING_YAW_RAD, tol,
                       "truck approach")

    def _do_go_shelf(self, pose, sensors):
        b = self.semantic.get_point(self.job["name"])
        goal = (b.x - cfg.AMR_HALF_LENGTH_M - cfg.SHELF_APPROACH_GAP_M - 0.30, b.y)
        tol = 0.03 if self.step_args.get("tight") else 0.10
        self._navigate(pose, goal, 0.0, tol, "shelf approach")

    def _do_go_home(self, pose, sensors):
        self._navigate(pose, cfg.HOME_XY, cfg.HOME_YAW_RAD, 0.15, "home station",
                       on_arrive=self._arrived_home)

    def _arrived_home(self):
        self._record("HOME", self.pose_estimator.pose.x, self.pose_estimator.pose.y,
                     "back at the home station")
        self.result_message = (f"completed {len(self.job_reports)} job(s): "
                               + ", ".join(f"{r['job']} with {r['module']}" for r in self.job_reports)
                               + "; modules returned; robot home")
        self._set_state(STATE_COMPLETE, self.result_message)

    # ------------------------------------------------------- detection

    def _do_detect(self, pose, sensors):
        self.io.stop()
        bgr, depth = self.io.render_front_camera_display()
        eye, fwd, up = camera_rig.build_camera_frame(pose.x, pose.y, pose.yaw, "left")
        result = self.box_detector.detect(bgr, depth, camera_rig.DISPLAY_RGBD_INTRINSICS, eye, fwd, up,
                                          pose, run_yolo=(len(self.detections) == 0))
        self.detections.append(result)
        if len(self.detections) >= DETECT_FRAMES:
            self._set_state(STATE_DECIDE)

    def _do_decide(self, pose, sensors):
        self.io.stop()
        best = max(self.detections, key=lambda r: r.count)
        yolo = next((r.yolo_labels for r in self.detections if r.yolo_labels), [])
        decision = boxmod.BoxDetector.decide_payload(best)
        self.payload_decision = decision
        for b in best.boxes:
            self.semantic.add_detection("box", b.world_x, b.world_y, b.footprint_m)
        print(f"[mission] YOLO semantic pass: {yolo if yolo else 'no COCO classes on the stock'}")
        dims = ", ".join(f"{b.footprint_m:.2f} m wide x {b.top_z - b.bottom_z:.2f} m tall"
                         for b in best.boxes)
        print(f"[mission] RGB-D: {best.count} box(es) [{dims}] -> {decision.module}: {decision.reason}")
        if decision.module == "NONE":
            self._fail(decision.reason)
            return
        ax, ay = best.centroid
        top = max(b.top_z for b in best.boxes)
        self.stock_hint = (ax, ay, top)
        self._record(self.job["name"], ax, ay, f"{best.count} box(es), top z~{top:.2f} m")
        self._build_plan(decision.module)
        self._advance(f"{decision.module} module required")

    # ------------------------------------------------------- tool stands

    def _station(self):
        return cfg.STATIONS[self.station_kind]

    def _dock_axis_offsets(self, pose):
        st = self._station()
        sx, sy = st["xy"]
        c, s = math.cos(st["yaw"]), math.sin(st["yaw"])
        along = (pose.x - sx) * c + (pose.y - sy) * s
        lateral = -(pose.x - sx) * s + (pose.y - sy) * c
        return along, lateral

    def _measure_posts(self, pose):
        """Post-sized obstacle clusters (stand legs) in a HIGH-resolution
        RGB-D frame (the robot is stationary here; at the 128x96 perception
        resolution an 8 cm leg is only ~2 depth samples wide and cannot be
        told apart from a 2 cm hoist rod). Robot body frame. The hoist rods
        are too thin to qualify; module clusters are too big."""

        _bgr, depth = self.io.render_front_camera_display()
        eye, fwd, up = camera_rig.build_camera_frame(pose.x, pose.y, pose.yaw, "left")
        obs = self.local.obstacles_body_frame(
            [(depth, camera_rig.DISPLAY_RGBD_INTRINSICS, eye, fwd, up)], pose)
        self._post_obstacles = obs
        if obs.shape[0] == 0:
            return []
        sel = (obs[:, 0] > 0.3) & (obs[:, 0] < 2.8) & (np.abs(obs[:, 1]) < 1.8)
        pts = obs[sel]
        cell = 0.04
        keys = {}
        for f, l in pts:
            keys.setdefault((int(math.floor(f / cell)), int(math.floor(l / cell))), []).append((f, l))
        seen, posts = set(), []
        for seed in keys:
            if seed in seen:
                continue
            stack, comp = [seed], []
            seen.add(seed)
            while stack:
                cx, cy = stack.pop()
                comp.extend(keys[(cx, cy)])
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nb = (cx + dx, cy + dy)
                        if nb in keys and nb not in seen:
                            seen.add(nb)
                            stack.append(nb)
            c = np.array(comp)
            if c.shape[0] < 6:
                continue
            ext = c.max(axis=0) - c.min(axis=0)
            if ext[0] <= 0.16 and 0.045 <= ext[1] <= 0.16:
                posts.append((float(c[:, 0].min()) + cfg.STAND_LEG_SIZE_M / 2.0, float(np.median(c[:, 1]))))
        return posts

    def _stand_fix(self, pose, posts):
        """Pose fix from the stand's two NEAR legs (the pair facing the
        robot): two posts the known leg spacing apart at about the same
        depth. Which world legs they are follows from geometry alone - the
        robot approaches from the stand's open (+x) side and the post on
        the robot's left is the leg on its left - so no prior pose is
        needed for the association. 2-D rigid transform from the pair."""

        spacing = 2.0 * cfg.STAND_LEG_Y_M
        pair = None
        for i in range(len(posts)):
            for j in range(i + 1, len(posts)):
                a, b = posts[i], posts[j]
                if abs(math.hypot(a[0] - b[0], a[1] - b[1]) - spacing) < 0.06 and abs(a[0] - b[0]) < 0.25:
                    if pair is None or a[0] + b[0] < pair[0][0] + pair[1][0]:
                        pair = (a, b)
        if pair is None:
            return None
        right_post, left_post = sorted(pair, key=lambda q: q[1])
        # The far pair has the module between it and the robot (or, after
        # a drop-off, nothing - but then the near pair is nearer anyway).
        depth = min(right_post[0], left_post[0])
        obs = self._post_obstacles
        lo, hi = right_post[1] + 0.15, left_post[1] - 0.15
        between = obs[(obs[:, 0] > 0.3) & (obs[:, 0] < depth - 0.12)
                      & (obs[:, 1] > lo) & (obs[:, 1] < hi)]
        if between.shape[0] > 15:
            return None
        st = self._station()
        c, s_ = math.cos(st["yaw"]), math.sin(st["yaw"])
        sx, sy = st["xy"]

        def leg(ly):
            return (sx + cfg.STAND_LEG_X_M * c - ly * s_, sy + cfg.STAND_LEG_X_M * s_ + ly * c)

        B = np.array([left_post, right_post])
        W = np.array([leg(-cfg.STAND_LEG_Y_M), leg(cfg.STAND_LEG_Y_M)])
        bc, wc = B.mean(0), W.mean(0)
        H = (B - bc).T @ (W - wc)
        yaw = math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        t = wc - R @ bc
        return Pose2D(float(t[0]), float(t[1]), yaw), 2

    def _do_localize(self, pose, sensors):
        if self._phase == "BACK_OFF":
            if math.hypot(pose.x - self._ref_xy[0], pose.y - self._ref_xy[1]) < 0.7:
                self.io.drive(-0.15, 0.0)
                return
            self.io.stop()
            self._phase = None
            self.state_t0 = self.t
            return
        self.io.stop()
        if self._phase is None:
            self._phase = "MEASURE"
            self._fixes = []
        res = self._stand_fix(pose, self._measure_posts(pose))
        if res is not None:
            self._fixes.append(res)
        if self.t - self.state_t0 < 0.6:
            return
        name = self._station()["name"]
        if len(self._fixes) < 3:
            if self._lookbacks < 2:
                self._lookbacks += 1
                self._phase = "BACK_OFF"
                self._ref_xy = (pose.x, pose.y)
                print("[mission] stand legs not both in view - backing off to look again")
                return
            self._fail(f"could not localise against the {name}")
            return
        xs = np.array([[f.x, f.y, math.cos(f.yaw), math.sin(f.yaw)] for f, _n in self._fixes])
        m = np.median(xs, axis=0)
        fix = Pose2D(float(m[0]), float(m[1]), math.atan2(m[3], m[2]))
        dxy = math.hypot(fix.x - pose.x, fix.y - pose.y)
        dyaw = math.degrees(wrap(fix.yaw - pose.yaw))
        if dxy > 1.0 or abs(dyaw) > 20.0:
            self._fail(f"stand-leg fix implausible ({dxy:.2f} m, {dyaw:+.1f} deg)")
            return
        self.pose_estimator.pose = fix
        self.station_fix = (dxy, dyaw)
        print(f"[mission] {name} leg landmark fix: pose corrected by {dxy*100:.1f} cm, {dyaw:+.1f} deg")
        if dxy > 0.15 or abs(dyaw) > 3.0:
            self.occupancy.clear_region(self._station()["xy"], 2.6)
            print("[mission] large correction - cleared the local map around the stand to re-map it")
        self._lookbacks = 0
        pose = self.pose_estimator.get_pose()
        along, lateral = self._dock_axis_offsets(pose)
        if (abs(lateral) > 0.07 or abs(along - cfg.ARM_STATION_PRE_DOCK_DIST_M) > 0.3) \
                and self._station_attempts < 3:
            self._station_attempts += 1
            self.plan.appendleft((STATE_LOCALIZE, self.step_args))
            self.plan.appendleft((STATE_GO_STATION, self.step_args))
            self._advance(f"{lateral*100:+.0f} cm off the dock axis - re-aligning")
            return
        self._advance(f"on dock axis ({lateral*100:+.1f} cm)")

    def _do_turn_dock(self, pose, sensors):
        err = wrap(self._station()["yaw"] - pose.yaw)
        if abs(err) < math.radians(1.0):
            self.io.stop()
            self._advance()
            return
        w = float(np.clip(1.6 * err, -0.8, 0.8))
        if abs(w) < 0.08:
            w = math.copysign(0.08, err)
        self.io.drive(0.0, w)

    def _do_reverse_dock(self, pose, sensors):
        along, lateral = self._dock_axis_offsets(pose)
        if along <= 0.003:
            self.io.stop()
            self._advance(f"docked (lateral {lateral*100:+.1f} cm)")
            return
        if self._phase is None:
            self._phase = "REVERSE"
            self._stall_ref = (pose.x, pose.y, self.t)
        if math.hypot(pose.x - self._stall_ref[0], pose.y - self._stall_ref[1]) > 0.04:
            self._stall_ref = (pose.x, pose.y, self.t)
        elif self.t - self._stall_ref[2] > 2.5 or self.t - self.state_t0 > 30.0:
            self.io.stop()
            self._dock_attempts += 1
            if self._dock_attempts >= 3:
                self._fail("reverse docking blocked repeatedly")
                return
            args = self.step_args
            for st in reversed([STATE_GO_STATION, STATE_LOCALIZE, STATE_TURN_DOCK, STATE_REVERSE_DOCK]):
                self.plan.appendleft((st, args))
            self._station_attempts = 0
            self._advance("reverse docking stalled - pulling out to re-align")
            return
        heading_err = wrap(pose.yaw - self._station()["yaw"])
        desired = float(np.clip(2.0 * lateral, -math.radians(8), math.radians(8)))
        w = float(np.clip(-2.5 * (heading_err - desired), -0.4, 0.4))
        v = -DOCK_REVERSE_SPEED_MPS * float(np.clip(along / 0.25, 0.3, 1.0))
        self.io.drive(v, w)

    def _do_transfer(self, pose, sensors):
        self.io.stop()
        kind = self.station_kind
        name = self._station()["name"]
        el = self.t - self._phase_t0
        lower = cfg.STAND_HOIST_LOWER_M
        if self._phase is None:
            self._phase, self._phase_t0 = "LOWER", self.t
            what = "the module onto the deck" if self.dock_purpose == "pick" else "its hooks onto the module"
            print(f"[mission] {name}: hoist lowering {what}")
            return
        if self.dock_purpose == "drop":
            if self._phase == "LOWER":
                self.io.set_hoist(kind, -lower * min(1.0, el / 1.0))
                if el > 1.3:
                    self.io.release_module(kind, -lower)
                    print(f"[mission] {name}: hoist holds the module, electromagnets released")
                    self._phase, self._phase_t0 = "LIFT_OFF", self.t
                return
            if self._phase == "LIFT_OFF":
                self.io.set_hoist(kind, -lower + lower * min(1.0, el / 1.2))
                if el > 1.5:
                    if kind == "CONVEYOR":
                        self.nav.set_half_width(cfg.AMR_HALF_WIDTH_M)
                    self._record(name, pose.x, pose.y, "module returned to its stand")
                    self._advance(f"{kind} module returned to the {name}")
                return
            return
        # pick-up
        if self._phase == "LOWER":
            self.io.set_hoist(kind, -lower * min(1.0, el / 1.2))
            if el > 1.5:
                sep = self.io.pad_separation(kind)
                ok, reason = self.io.engage_module(kind)
                if ok:
                    print(f"[mission] electromagnets engaged ({reason}, pad gap {sep*100:.1f} cm)")
                    if sep < 0.03:
                        # Pads (1 cm apart vertically) within 3 cm: the AMR
                        # sits at the stand's docked pose to ~2.8 cm - a
                        # landmark fix as good as the leg fix.
                        st = self._station()
                        fix = Pose2D(st["xy"][0], st["xy"][1], st["yaw"])
                        print(f"[mission] docked-pose fix: estimate corrected by "
                              f"{100*math.hypot(fix.x - pose.x, fix.y - pose.y):.1f} cm")
                        self.pose_estimator.pose = fix
                    self._phase, self._phase_t0 = "RAISE", self.t
                else:
                    print(f"[mission] magnets refused: {reason}")
                    self._dock_attempts += 1
                    self._phase, self._phase_t0 = "ABORT_RAISE", self.t
            return
        if self._phase == "ABORT_RAISE":
            self.io.set_hoist(kind, -lower * max(0.0, 1.0 - el / 1.0))
            if el > 1.2:
                if self._dock_attempts >= 3:
                    self._fail(f"could not align under the {name}")
                else:
                    # Fine docking: the pads' proximity sensors report the
                    # distance to their counterparts. Sweep a few cm each way
                    # along the dock axis and settle where it is smallest.
                    along, _lat = self._dock_axis_offsets(pose)
                    self._seek = {"stage": 0, "start": along, "samples": []}
                    self._phase, self._phase_t0 = "SEEK", self.t
                    print("[mission] fine docking: sweeping along the dock axis on the pad proximity reading")
            return
        if self._phase == "SEEK":
            along, _lat = self._dock_axis_offsets(pose)
            sk = self._seek
            sk["samples"].append((along, self.io.pad_separation(kind)))
            heading_err = wrap(pose.yaw - self._station()["yaw"])
            w = float(np.clip(-2.0 * heading_err, -0.2, 0.2))
            if sk["stage"] == 0:                       # reverse 15 cm
                if along > sk["start"] - 0.15:
                    self.io.drive(-0.03, w)
                    return
                sk["stage"] = 1
            if sk["stage"] == 1:                       # forward to 15 cm past the start
                if along < sk["start"] + 0.15:
                    self.io.drive(0.03, w)
                    return
                sk["stage"] = 2
                sk["target"] = min(sk["samples"], key=lambda a_s: a_s[1])[0]
                best = min(x for _a, x in sk["samples"])
                print(f"[mission] fine docking: closest pad alignment at {100*(sk['target']-sk['start']):+.1f} cm "
                      f"(separation {best*100:.1f} cm)")
                # Pads ~6 cm above their counterparts while the hoist is up:
                # a minimum much above that means a SIDEWAYS offset, which
                # moving along the axis cannot fix - pull out and re-reverse
                # (the lane fix corrects the lateral error on the way in).
                if best > 0.085:
                    print("[mission] fine docking: offset is lateral - pulling out to re-reverse")
                    self._phase, self._phase_t0 = "PULL_OUT", self.t
                    self._ref_xy = (pose.x, pose.y)
                    return
            err = sk["target"] - along
            if abs(err) > 0.004:
                self.io.drive(float(np.clip(1.0 * err, -0.03, 0.03)), w)
                return
            self.io.stop()
            self._phase, self._phase_t0 = "LOWER", self.t
            return
        if self._phase == "RAISE":
            frac = min(1.0, el / 1.5)
            self.io.set_hoist(kind, -lower + (lower + cfg.STAND_HOIST_RAISE_M) * frac)
            if frac >= 1.0:
                if kind == "CONVEYOR":
                    self.nav.set_half_width(CONVEYOR_HALF_WIDTH_M)
                    self.conveyor.set_lift(cfg.CONV_LIFT_BASE_Z_M)
                self._record(name, pose.x, pose.y, "tool stand (docked pose, measured on arrival)")
                self._advance(f"{kind} module aboard")

    def _do_leave(self, pose, sensors):
        if self._phase is None:
            self._phase = "OUT"
            self._ref_xy = (pose.x, pose.y)
        if math.hypot(pose.x - self._ref_xy[0], pose.y - self._ref_xy[1]) >= 1.0:
            self.io.stop()
            # The module left with us (or was left behind): forget the stale
            # imprint at the stand centre; the legs stay mapped.
            self.occupancy.clear_region(self._station()["xy"], 0.32)
            self.station_kind, self.dock_purpose = None, None
            self._advance()
            return
        err = wrap(self._station()["yaw"] - pose.yaw)
        self.io.drive(0.25, float(np.clip(2.0 * err, -0.3, 0.3)))

    # ------------------------------------------------------- arm job

    def _creep_to_edge(self, pose, sensors, target_heading, label, point_name,
                       standoff=cfg.WORK_STANDOFF_M, band=EDGE_BAND_Z, on_done=None):
        """Depth-servoed final approach: measure the distance to the surface
        edge ahead while it is visible, dead-reckon the last few cm, stop
        `standoff` short of it, then pin the reached pose on the map."""

        if self._phase is None:
            self._phase = "CREEP"
            self._edge_dist, self._edge_ref_xy = None, (pose.x, pose.y)
        d = self._edge_distance(sensors, pose, band) if sensors is not None else None
        if d is not None and d > 0.18:
            self._edge_dist, self._edge_ref_xy = d, (pose.x, pose.y)
        if self._edge_dist is None:
            if self.t - self.state_t0 > 8.0:
                self._fail(f"{label}: no surface edge seen ahead")
                return
            self.io.drive(0.06, 0.0)
            return
        moved = math.hypot(pose.x - self._edge_ref_xy[0], pose.y - self._edge_ref_xy[1])
        remaining = self._edge_dist - moved - standoff
        if remaining <= 0.004:
            self.io.stop()
            if point_name:
                self._record(point_name, pose.x, pose.y, f"{label}: bumper {standoff:.2f} m from the edge")
            if on_done is not None:
                on_done(pose)
            else:
                self._advance(f"{label} reached")
            return
        err = wrap(target_heading - pose.yaw)
        v = float(np.clip(remaining * 1.2, 0.03, CREEP_SPEED_MPS))
        self.io.drive(v, float(np.clip(2.0 * err, -0.3, 0.3)))

    def _do_approach_pick(self, pose, sensors):
        a = self.semantic.get_point(cfg.POINT_A_NAME)
        heading = math.atan2(a.y - pose.y, a.x - pose.x) if a.x - pose.x > 0.5 else 0.0
        self._creep_to_edge(pose, sensors, heading, "pick table", "PICK_POSE")

    def _do_approach_truck(self, pose, sensors):
        self._creep_to_edge(pose, sensors, cfg.TRUCK_BOARDING_YAW_RAD, "truck sill", "TRUCK_BOARDING")

    def _do_back_off(self, pose, sensors):
        if self._phase is None:
            self._phase = "BACK"
            self._ref_xy = (pose.x, pose.y)
        if math.hypot(pose.x - self._ref_xy[0], pose.y - self._ref_xy[1]) >= 0.55:
            self.io.stop()
            self._advance()
            return
        self.io.drive(-0.15, 0.0)

    def _run_task(self, factory, on_done):
        self.io.stop()
        if self._task is None:
            self._task = factory()
        try:
            next(self._task)
        except StopIteration as stop:
            report = stop.value
            self._task = None
            on_done(report)

    def _do_arm_load(self, pose, sensors):
        def factory():
            ax, ay, top = self.stock_hint
            mx, my, mz = self.world_to_module(pose, ax, ay, top)
            # The stock hint was pinned from far away during exploration;
            # the robot has just depth-servoed to the table edge, so the
            # stock must lie just beyond that edge: aim the first look there.
            edge_x = cfg.AMR_HALF_LENGTH_M + cfg.WORK_STANDOFF_M
            mx = float(np.clip(mx, edge_x + 0.06, edge_x + 0.30))
            my = float(np.clip(my, -0.25, 0.25))
            print(f"[mission] arm: first look aimed at module ({mx:+.2f}, {my:+.2f}), top z {mz:+.2f}")
            size = self.payload_decision.max_footprint_m if self.payload_decision else 0.18
            best = max(self.detections, key=lambda r: r.count) if self.detections else None
            height = (float(np.median([b.top_z - b.bottom_z for b in best.boxes]))
                      if best is not None and best.boxes else None)
            return self.arm.load_from_table((mx, my), mz, expected_size=max(0.10, size),
                                            expected_height=height)

        def done(report):
            self.load_report = report
            for (x, y, z, size) in report.measured:
                print(f"[mission]   stereo-measured block at module ({x:+.3f}, {y:+.3f}), "
                      f"top {z:+.3f}, size {size:.3f} m")
            if report.measured:
                c, s_ = math.cos(pose.yaw), math.sin(pose.yaw)
                xs = [pose.x + x * c - y * s_ for x, y, _z, _s in report.measured]
                ys = [pose.y + x * s_ + y * c for x, y, _z, _s in report.measured]
                self._record(cfg.POINT_A_NAME, float(np.mean(xs)), float(np.mean(ys)),
                             f"stock centroid from {len(xs)} arm-stereo measurements")
            if report.success:
                self._advance(report.message)
            else:
                self._fail(report.message or "no boxes loaded")

        self._run_task(factory, done)

    def _do_arm_unload(self, pose, sensors):
        def factory():
            return self.arm.unload_to_truck(cfg.TRUCK_DECK_Z_M - cfg.MODULE_DOCKED_Z_M)

        def done(report):
            self.unload_report = report
            if report.success:
                self._advance(report.message)
            else:
                self._fail(report.message or "unload failed")

        self._run_task(factory, done)

    # ------------------------------------------------------- conveyor job

    def _heading_hold(self, pose, heading, v):
        err = wrap(heading - pose.yaw)
        self.io.drive(v, float(np.clip(2.0 * err, -0.3, 0.3)))

    def _shelf_measure(self, sensors, pose):
        """From the forward RGB-D: the box's bottom (= shelf top) height and
        width from its cardboard-coloured pixels, the shelf slab's front edge
        distance, and the box's lateral offset. None if not resolved."""

        bgr, depth = self.io.render_front_camera_display()
        eye, fwd, up = camera_rig.build_camera_frame(pose.x, pose.y, pose.yaw, "left")
        world, pix = vo.project_depth_to_world(depth, camera_rig.DISPLAY_RGBD_INTRINSICS, eye, fwd, up,
                                               stride=2, max_range_m=3.0)
        if world.shape[0] == 0:
            return None
        mask = boxmod.cardboard_mask(bgr)
        is_box = mask[pix[:, 1], pix[:, 0]] > 0
        dx, dy = world[:, 0] - pose.x, world[:, 1] - pose.y
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        f, l = dx * c + dy * s, -dx * s + dy * c
        box = is_box & (np.abs(l) < 0.6) & (f > 0.3)
        if np.count_nonzero(box) < 80:
            return None
        bz, bl, bf = world[box, 2], l[box], f[box]
        bottom = float(np.percentile(bz, 2))
        width = float(np.percentile(bl, 98) - np.percentile(bl, 2))
        centre_l = float(0.5 * (np.percentile(bl, 98) + np.percentile(bl, 2)))
        face_f = float(np.percentile(bf, 3))
        slab = (~is_box) & (world[:, 2] > bottom - 0.05) & (world[:, 2] < bottom - 0.008) \
            & (np.abs(l - centre_l) < 0.3) & (f > 0.3)
        if np.count_nonzero(slab) < 20:
            return None
        edge_f = float(np.percentile(f[slab], 3))
        return {"shelf_top": bottom, "width": width, "lateral": centre_l,
                "box_face": face_f - cfg.AMR_HALF_LENGTH_M, "edge": edge_f - cfg.AMR_HALF_LENGTH_M}

    def _do_measure_shelf(self, pose, sensors):
        """Creep to ~0.6 m from the shelf, then measure it (several frames)
        and set the belt just below the shelf top."""

        if self._phase in (None, "CREEP"):
            def at_gap(_pose):
                self._phase, self._samples = "MEASURE", []
            self._creep_to_edge(pose, sensors, 0.0 if abs(pose.yaw) < 1.0 else pose.yaw,
                                "shelf", None, standoff=cfg.SHELF_APPROACH_GAP_M,
                                band=(0.50, 0.66), on_done=at_gap)
            return
        self.io.stop()
        m = self._shelf_measure(sensors, pose)
        if m is not None:
            self._samples.append(m)
        if len(self._samples) < 3:
            if self.t - self.state_t0 > 20.0:
                self._fail("could not measure the shelf / box with the RGB-D camera")
            return
        keys = self._samples[0].keys()
        meas = {k: float(np.median([sm[k] for sm in self._samples])) for k in keys}
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        edge_d = meas["edge"] + cfg.AMR_HALF_LENGTH_M
        attempts = self.conv.get("shelf_attempts", 0)
        self.conv = {
            "shelf_attempts": attempts,
            "shelf_top": meas["shelf_top"], "width": meas["width"],
            # Shelf front edge point on the robot's approach line, and the
            # line (through the box centre) to reverse along.
            "edge_xy": (pose.x + edge_d * c - meas["lateral"] * s, pose.y + edge_d * s + meas["lateral"] * c),
            "line_heading": pose.yaw,
            "line_point": (pose.x - meas["lateral"] * s, pose.y + meas["lateral"] * c),
        }
        print(f"[mission] shelf measured: top z={meas['shelf_top']:.3f} m, box {meas['width']:.2f} m wide, "
              f"box face {meas['box_face']:.2f} m / shelf edge {meas['edge']:.2f} m ahead, "
              f"box centre {meas['lateral']*100:+.1f} cm lateral")
        # Pin B at the measured box centre (real-time location of the stock).
        bx = pose.x + (meas["box_face"] + cfg.AMR_HALF_LENGTH_M) * c - meas["lateral"] * s
        by = pose.y + (meas["box_face"] + cfg.AMR_HALF_LENGTH_M) * s + meas["lateral"] * c
        self._record(self.job["name"], bx, by, f"big box front face, {meas['width']:.2f} m wide (measured)")
        if abs(meas["lateral"]) > 0.03 and attempts < 2:
            # The belt is only 10 cm wider than the box: get onto the box's
            # centreline before turning round (a 0.4 m reverse cannot
            # correct a sideways offset).
            self.conv["shelf_attempts"] = attempts + 1
            self.plan.appendleft((STATE_MEASURE_SHELF, {}))
            self.plan.appendleft((STATE_GO_SHELF, {"tight": True}))
            self._advance(f"{meas['lateral']*100:+.0f} cm off the box centreline - re-aligning")
            return
        self.conv["shelf_attempts"] = 0
        if meas["width"] > 2 * cfg.CONV_BELT_HALF_WIDTH_M - 0.04:
            self._fail(f"box {meas['width']:.2f} m wide does not fit the {2*cfg.CONV_BELT_HALF_WIDTH_M:.2f} m belt")
            return
        # Belt just BELOW the box bottom so the nose slides under the
        # overhanging carton edge without touching it.
        belt_top = meas["shelf_top"] - 0.012 - cfg.MODULE_DOCKED_Z_M
        self.conveyor.set_lift(belt_top)
        print(f"[mission] conveyor lift -> belt top z={belt_top + cfg.MODULE_DOCKED_Z_M:.3f} m")
        self._record(self.job["name"] + "_SHELF", *self.conv["edge_xy"],
                     f"shelf edge, top at z={meas['shelf_top']:.2f} m")
        self._advance("shelf measured")

    def _do_turn_around(self, pose, sensors):
        if self._phase is None:
            self._phase = "TURN"
            self._turn_target = wrap(self.conv["line_heading"] + math.pi)
        err = wrap(self._turn_target - pose.yaw)
        if abs(err) < math.radians(1.0):
            self.io.stop()
            self._advance()
            return
        w = float(np.clip(1.6 * err, -0.8, 0.8))
        if abs(w) < 0.08:
            w = math.copysign(0.08, err)
        self.io.drive(0.0, w)

    def _reverse_along_line(self, pose, speed):
        """Reverse along the measured line (heading = line + pi) with
        lateral correction toward it."""

        lh = self.conv["line_heading"]
        px, py = self.conv["line_point"]
        lateral = -(pose.x - px) * math.sin(lh) + (pose.y - py) * math.cos(lh)
        heading = wrap(lh + math.pi)
        heading_err = wrap(pose.yaw - heading)
        # Reversing toward +line direction: lateral error shrinks when the
        # robot's yaw (facing away) is tilted toward the line.
        desired = float(np.clip(-2.0 * lateral, -math.radians(6), math.radians(6)))
        w = float(np.clip(-2.5 * (heading_err - desired), -0.3, 0.3))
        self.io.drive(-speed, w)

    def _nose_to_edge(self, pose):
        """Odometry estimate of the gap between the belt nose and the
        measured edge (along the line)."""

        lh = self.conv["line_heading"]
        ex, ey = self.conv["edge_xy"]
        nose_x = pose.x + cfg.CONV_BELT_X_MIN_M * math.cos(pose.yaw)
        nose_y = pose.y + cfg.CONV_BELT_X_MIN_M * math.sin(pose.yaw)
        return (ex - nose_x) * math.cos(lh) + (ey - nose_y) * math.sin(lh)

    def _do_reverse_shelf(self, pose, sensors):
        gap_odo = self._nose_to_edge(pose)
        tof = self.conveyor.tof_range()
        lift_ready = self.conveyor.lift_settled()
        if self._phase is None:
            self._phase = "REVERSE"
            print(f"[mission] reversing the belt nose to the shelf (odometry gap {gap_odo:.2f} m)")
        gap = tof if (tof is not None and tof < 0.4) else gap_odo
        if gap <= 0.015 or gap_odo < -0.03:
            self.io.stop()
            self.conv["tof_at_stop"] = tof
            self._advance(f"belt nose at the shelf (ToF {'%.3f m' % tof if tof is not None else 'n/a'}, "
                          f"under the box overhang)")
            return
        if gap < 0.25 and not lift_ready:
            self.io.stop()            # wait for the lift before the nose goes under the box
            return
        self._reverse_along_line(pose, float(np.clip(gap * 0.6, 0.02, 0.10)))

    def _do_conveyor_load(self, pose, sensors):
        conv = self.conveyor
        if self._phase in ("RETRY_OUT", "RETRY_BACK"):
            self._conveyor_load_retry(pose)
            return
        self.io.stop()
        if self._phase is None:
            self._phase, self._phase_t0 = "RAISE", self.t
            self._raise_start = conv.belt_top()
            self._load_retries = 0
            self._box_seen = False
            print("[mission] lift raising the belt until it touches the box underside")
            return
        if self._phase == "RAISE":
            if conv.belt_touching():
                self._touch_z = conv.belt_top()
                conv.set_lift(self._touch_z + 0.004, speed=0.02)
                self._phase, self._phase_t0 = "PULL", self.t
                print(f"[mission] belt load cell: box resting on the belt at belt top "
                      f"z={self._touch_z + cfg.MODULE_DOCKED_Z_M:.3f} m - running the belt inward")
                conv.run_belt(cfg.CONV_BELT_SPEED_MPS)
                return
            target = self._raise_start + 0.002 * ((self.t - self._phase_t0) / self.dt)
            if target > self._raise_start + 0.04:
                self._fail("belt never touched the box (lift +4 cm)")
                return
            conv.set_lift(target, speed=0.05)
            return
        if self._phase == "PULL":
            if conv.ir_blocked("end"):
                conv.run_belt(0.0)
                self._phase, self._phase_t0 = "CHECK", self.t
                return
            self._box_seen = self._box_seen or conv.ir_blocked("nose")
            if not self._box_seen and self.t - self._phase_t0 > 3.0:
                # Belt running but nothing moved past the nose beam: the box
                # is not riding on the belt. Lower, ease out, re-seat, retry.
                conv.run_belt(0.0)
                if self._load_retries >= 2:
                    self._fail("belt could not draw the box aboard (no IR nose beam after 3 retries)")
                    return
                self._load_retries += 1
                conv.set_lift(self._raise_start)
                self._retry_ref = (pose.x, pose.y)
                self._phase, self._phase_t0 = "RETRY_OUT", self.t
                print(f"[mission] IR nose beam never blocked - lowering and re-seating the nose (retry {self._load_retries})")
                return
            if self.t - self._phase_t0 > 20.0:
                conv.run_belt(0.0)
                self._fail("box did not reach the end IR beam")
            return
        if self._phase == "CHECK":
            if self.t - self._phase_t0 < 0.5:
                return
            nose, end = conv.ir_blocked("nose"), conv.ir_blocked("end")
            if end and not nose:
                print("[mission] IR: end beam blocked, nose beam clear -> box fully on the AMR")
                self._advance("big box aboard")
            else:
                conv.run_belt(cfg.CONV_BELT_SPEED_MPS * 0.5)
                self._phase, self._phase_t0 = "PULL", self.t

    def _conveyor_load_retry(self, pose):
        conv = self.conveyor
        if self._phase == "RETRY_OUT":
            if not conv.lift_settled():
                self.io.stop()
                return
            if math.hypot(pose.x - self._retry_ref[0], pose.y - self._retry_ref[1]) < 0.05:
                self._heading_hold(pose, wrap(self.conv["line_heading"] + math.pi), 0.03)
                return
            self.io.stop()
            self._phase, self._phase_t0 = "RETRY_BACK", self.t
            return
        tof = conv.tof_range()
        if (tof is not None and tof <= 0.010) or self.t - self._phase_t0 > 8.0:
            self.io.stop()
            self._phase, self._phase_t0 = "RAISE", self.t
            self._box_seen = False
            print(f"[mission] nose re-seated (ToF {tof if tof is None else round(tof, 3)}) - raising again")
            return
        self._reverse_along_line(pose, 0.02)

    def _do_pull_away(self, pose, sensors):
        if self._phase is None:
            self._phase = "OUT"
            self._ref_xy = (pose.x, pose.y)
            self._hold_heading = pose.yaw
        if math.hypot(pose.x - self._ref_xy[0], pose.y - self._ref_xy[1]) >= 0.6:
            self.io.stop()
            self.conveyor.set_lift(cfg.CONV_LIFT_BASE_Z_M)
            self._advance("clear - belt lowered to transport height")
            return
        self._heading_hold(pose, self._hold_heading, 0.15)

    def _truck_measure(self, sensors, pose):
        """Forward RGB-D look into the trailer's open rear: the sill
        distance (deck front face), the deck-top height (points ON the deck
        just inside the sill) and the lateral positions of the two inner
        side walls (above deck height). Robot body frame; None if not
        resolved."""

        res = self._forward_points(sensors, pose, stride=1, max_range=4.0)
        if res is None:
            return None
        world, f, l = res
        z = world[:, 2]
        face = (z > EDGE_BAND_Z[0]) & (z < EDGE_BAND_Z[1]) & (np.abs(l) < 0.30) & (f > 0.3)
        if np.count_nonzero(face) < 12:
            return None
        f0 = float(np.percentile(f[face], 3))
        deck_pts = (f > f0 + 0.03) & (f < f0 + 0.8) & (np.abs(l) < 0.4) & (z > 0.25) & (z < 0.55)
        # Side walls only: stop well short of the trailer's front wall
        # (TRUCK_LENGTH_M behind the sill), whose face spans every lateral.
        walls = (z > 0.75) & (z < 1.35) & (f > f0 + 0.05) & (f < f0 + cfg.TRUCK_LENGTH_M - 0.5)
        left = l[walls & (l > 0.15)]
        right = l[walls & (l < -0.15)]
        if np.count_nonzero(deck_pts) < 20 or left.size < 20 or right.size < 20:
            return None
        deck = float(np.percentile(z[deck_pts], 85))   # the deck top is the highest surface there

        def wall_line(sel, inner):
            """Fit lateral = a + b*fwd to the wall's inner face (per 10 cm
            depth band, the point nearest the truck centreline)."""

            ff, ll = f[sel], l[sel]
            fs, ls = [], []
            for k in np.unique(np.floor(ff / 0.10)):
                band = ll[np.floor(ff / 0.10) == k]
                if band.size >= 3:
                    fs.append((k + 0.5) * 0.10)
                    ls.append(inner(band))
            if len(fs) < 4:
                return None
            b, a = np.polyfit(np.array(fs), np.array(ls), 1)
            return a, b

        lw = wall_line(walls & (l > 0.15), lambda v: np.percentile(v, 10))
        rw = wall_line(walls & (l < -0.15), lambda v: np.percentile(v, 90))
        if lw is None and rw is None:
            return None
        inner = cfg.TRUCK_HALF_WIDTH_M - cfg.TRUCK_WALL_THICKNESS_M
        # One wall plus the trailer's known inner width is enough when the
        # other wall is out of view (e.g. from a boarding line near a wall).
        left_face = lw[0] + lw[1] * f0 if lw is not None else rw[0] + rw[1] * f0 + 2 * inner
        right_face = rw[0] + rw[1] * f0 if rw is not None else left_face - 2 * inner
        slopes = [w[1] for w in (lw, rw) if w is not None]
        slope = float(np.mean(slopes))         # walls' direction in the body frame
        return f0, deck, left_face, right_face, slope

    def _do_measure_truck(self, pose, sensors):
        """Creep to ~0.6 m from the sill, then use the trailer itself as a
        landmark: sill distance + both inner side walls -> pose fix against
        the trailer's defined location, lateral re-alignment onto the
        boarding line if needed, and (conveyor) the belt set to the
        measured deck height."""

        module = self.step_args.get("module", "SUCTION_ARM")
        if self._phase in (None, "CREEP"):
            def at_gap(_pose):
                self._phase, self._samples = "MEASURE", []
            self._creep_to_edge(pose, sensors, cfg.TRUCK_BOARDING_YAW_RAD, "truck sill", None,
                                standoff=cfg.SHELF_APPROACH_GAP_M, on_done=at_gap)
            return
        self.io.stop()
        m = self._truck_measure(sensors, pose)
        if m is not None:
            self._samples.append(m)
        if len(self._samples) < 3:
            if self.t - self.state_t0 > 20.0:
                self._fail("could not measure the truck opening")
            return
        f0, deck, left_face, right_face, slope = (float(np.median([sm[i] for sm in self._samples]))
                                                 for i in range(5))
        width = left_face - right_face
        centre_l = 0.5 * (left_face + right_face)
        if abs(width - 2 * (cfg.TRUCK_HALF_WIDTH_M - cfg.TRUCK_WALL_THICKNESS_M)) < 0.15:
            # Truck landmark fix: the side walls run along the trailer axis
            # (world -X), so their direction in the body frame gives the
            # robot's true heading; the sill and the walls' midline at the
            # sill give its position, against the trailer's defined location.
            yaw = wrap(cfg.TRUCK_BOARDING_YAW_RAD - math.atan(slope))
            c, s = math.cos(yaw), math.sin(yaw)
            sill_x = pose.x + f0 * c - centre_l * s
            centre_y = pose.y + f0 * s + centre_l * c
            dx, dy = cfg.TRUCK_REAR_X - sill_x, cfg.TRUCK_Y - centre_y
            dyaw = math.degrees(wrap(yaw - pose.yaw))
            self.pose_estimator.pose = Pose2D(pose.x + dx, pose.y + dy, yaw)
            pose = self.pose_estimator.get_pose()
            print(f"[mission] truck landmark fix (opening {width:.2f} m wide): pose corrected by "
                  f"{dx*100:+.1f} cm x, {dy*100:+.1f} cm y, {dyaw:+.1f} deg")
        else:
            print(f"[mission] truck opening not resolved ({width:.2f} m) - using odometry")
        board_y = cfg.TRUCK_CONVEYOR_Y if module == "CONVEYOR" else cfg.TRUCK_ARM_Y
        off = pose.y - board_y
        attempts = self.conv.get("truck_attempts", 0)
        if abs(off) > 0.06 and attempts < 2:
            self.conv["truck_attempts"] = attempts + 1
            self.plan.appendleft((STATE_MEASURE_TRUCK, self.step_args))
            self.plan.appendleft((STATE_GO_TRUCK, self.step_args))
            self._advance(f"{off*100:+.0f} cm off the boarding line - re-aligning")
            return
        self.conv["truck_attempts"] = 0
        self.conv.update({
            "edge_xy": (cfg.TRUCK_REAR_X, board_y),
            "line_heading": cfg.TRUCK_BOARDING_YAW_RAD,
            "line_point": (pose.x, board_y),
            "deck": deck,
        })
        self._record("TRUCK", cfg.TRUCK_REAR_X, cfg.TRUCK_Y,
                     f"trailer sill, deck at z={deck:.2f} m (measured)")
        if module == "CONVEYOR":
            belt_top = deck + 0.03 - cfg.MODULE_DOCKED_Z_M
            self.conveyor.set_lift(belt_top)
            print(f"[mission] truck deck z={deck:.3f} m -> conveyor belt top z="
                  f"{belt_top + cfg.MODULE_DOCKED_Z_M:.3f} m")
        self._advance("truck measured")

    def _do_reverse_truck(self, pose, sensors):
        # Nose TRUCK_CONVEYOR_OVERLAP_M past the sill, over the deck.
        gap = self._nose_to_edge(pose) + cfg.TRUCK_CONVEYOR_OVERLAP_M
        if self._phase is None:
            self._phase = "REVERSE"
        if gap <= 0.005:
            self.io.stop()
            self._record("TRUCK_CONVEYOR", pose.x, pose.y,
                         f"belt nose {cfg.TRUCK_CONVEYOR_OVERLAP_M:.2f} m inside the trailer")
            self._advance("belt nose over the truck deck")
            return
        if gap < 0.3 and not self.conveyor.lift_settled():
            self.io.stop()
            return
        self._reverse_along_line(pose, float(np.clip(gap * 0.6, 0.02, 0.10)))

    def _do_conveyor_unload(self, pose, sensors):
        self.io.stop()
        conv = self.conveyor
        if self._phase is None:
            self._phase, self._phase_t0 = "PUSH", self.t
            self._seen_nose = False
            self._clear_t = None
            conv.run_belt(-cfg.CONV_BELT_SPEED_MPS)
            print("[mission] belt running out over the nose into the truck")
            return
        nose, end = conv.ir_blocked("nose"), conv.ir_blocked("end")
        self._seen_nose |= nose
        if self._seen_nose and not nose and not end:
            # The nose beam sits 7 cm in from the nose tip: keep the belt
            # running so the carton's trailing edge clears the tip too.
            self._clear_t = self._clear_t or self.t
            if self.t - self._clear_t > 1.8:
                conv.run_belt(0.0)
                print("[mission] IR: both beams clear -> box is off the belt, inside the truck")
                self._advance("big box unloaded into the truck")
                return
        else:
            self._clear_t = None
        if self.t - self._phase_t0 > 25.0:
            conv.run_belt(0.0)
            self._fail("box did not leave the belt")

    def _do_terminal(self, pose, sensors):
        self.io.stop()
