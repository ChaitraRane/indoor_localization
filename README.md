# Indoor Localization: Autonomous Warehouse AMR Simulation

A PyBullet simulation of an indoor logistics autonomous mobile robot (AMR) that **localizes, maps and plans using only onboard sensing**. There is no ground-truth pose, no LiDAR and no pre-built map. The robot follows painted floor lanes, builds its own maps from RGB-D cameras, fetches a swappable payload module, and loads boxes onto a truck.

> Full write-up: [AMR_Simulation_Report.pdf](AMR_Simulation_Report.pdf)

## What it does

A complete mission runs end to end:

1. **Explore:** follow the painted lane to stock location A, avoiding obstacles and mapping as it goes.
2. **Detect:** look at the pick table, find the boxes (YOLO plus depth-based sizing) and decide which payload module is needed.
3. **Fetch module:** drive to the arm-module stand, reverse-dock, and transfer the suction-arm module onto the AMR with a hoist and electromagnets.
4. **Pick:** use stereo-guided inverse kinematics to pick every box onto the cargo tray.
5. **Deliver:** drive to the truck and load the boxes into the trailer.
6. **Alternative job:** a conveyor-module job at location B (shelf measurement, reverse approach, belt load and unload).
7. **Return:** go home and finish.

## Key features

| Area | Approach |
|---|---|
| **Localization** | Wheel odometry fused with ORB + depth-backed PnP visual odometry. Heading is corrected by painted lanes (lane-aided heading) and landmark fixes at the tool stands. |
| **Mapping** | Two live maps. A log-odds occupancy grid with a lane layer, built from the two RGB-D cameras. A semantic map of fused object landmarks. |
| **Global planning** | A* over the live map with a clearance field. It prefers painted lanes by applying a cost multiplier off-lane. |
| **Local planning** | VFH+ / follow-the-gap obstacle avoidance using both forward depth cameras. |
| **Perception** | Lane detection from the RGB camera, and box detection with YOLO11s plus depth. |
| **Manipulation** | Swappable suction-arm module with stereo-guided IK, and a conveyor module with IR-beam and ToF sensing. |
| **Docking** | PyBullet constraint-based electromagnet model that locks onto the AMR's chassis magnet pads. |
| **Mission logic** | State machine in `mission_controller.py`. The controller sees only images, depth, wheel encoders and arm joint encoders. Cameras render from the true pose, and the controller never reads it. |

## Robot specification

Based on a simulation-based power and runtime validation. Full details are in [simulation/config/robot_specs.md](simulation/config/robot_specs.md).

- 60 kg empty and 20 kg payload (80 kg operating mass)
- 2 x 24 V geared drive motors, 100 RPM, 10 Nm, 200 mm wheels, 1.0 m/s target speed
- Sensors: 2 x OAK-D Lite-class stereo RGB-D cameras, 1 RGB camera, BNO085-class IMU, 2 wheel encoders
- Compute: Raspberry Pi 5 class edge computer
- Battery: 25.6 V, 20 Ah LiFePO4 (512 Wh)

## Repository layout

```
simulation/
├── autonomous_mission.py   # Main entry point: wires sim + robot + controller, runs the mission
├── main.py                 # Keyboard teleop + camera preview of the world and robot
├── mission_controller.py   # Mission state machine (the "brain")
├── mission_config.py       # Shared constants: geometry, lane network, jobs
├── warehouse_environment.py# Builds the warehouse world in PyBullet
├── visual_odometry.py      # RGB-D VO + wheel odometry pose fusion
├── lane_detector.py        # Painted-lane segmentation
├── lane_localizer.py       # Lane-aided heading correction
├── occupancy_map.py        # Live occupancy grid + lane layer
├── semantic_map.py         # Object landmark map
├── astar_planner.py        # Lane-preferring global planner
├── vfh_local_planner.py    # Local obstacle avoidance
├── navigation.py           # Path following / navigation layer
├── box_detector.py         # YOLO + depth box detection, module decision
├── arm_manipulator.py      # Suction arm, stereo camera, IK tasks
├── arm_control.py          # Arm module control
├── conveyor_control.py     # Conveyor module control
├── conveyor_hardware.py    # Conveyor sensors (IR beams, ToF)
├── magnetic_docking.py     # Electromagnet docking model
├── camera_rig.py           # Onboard camera setup
├── video_overlay.py        # Mission recording overlay
├── *.urdf                  # AMR, arm module, conveyor module, stations
├── assets/                 # Floor and prop textures
└── config/robot_specs.md   # Robot specification reference
AMR_Simulation_Report.pdf   # Project report
```

## Getting started

### Requirements

- Python 3.9+
- `pybullet`, `numpy`, `opencv-python`, `ultralytics`

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
pip install pybullet numpy opencv-python ultralytics
```

The YOLO11s weights (`simulation/yolo11s.pt`) are included in the repo.

### Run the full autonomous mission

```bash
cd simulation
python autonomous_mission.py                      # GUI; the view follows the AMR
python autonomous_mission.py --realtime --show-camera
python autonomous_mission.py --headless           # no window
python autonomous_mission.py --headless --record out.mp4
python autonomous_mission.py --jobs B             # run only the conveyor job
```

| Flag | Meaning |
|---|---|
| `--headless` | PyBullet DIRECT mode, no GUI |
| `--realtime` | Throttle to real time (GUI) |
| `--show-camera` | Show the onboard camera preview |
| `--record FILE` | Record the mission to an MP4 |
| `--no-follow` | Don't move the 3D view with the robot |
| `--jobs A,B` | Run only the listed stock locations |
| `--max-iters N` | Iteration cap (default 40000) |

### Drive the robot manually

```bash
cd simulation
python main.py
```

Loads the warehouse and AMR with keyboard teleop and a live preview of the stereo and RGB cameras.

## Notes

- Recorded mission videos (`*.mp4`, `*.avi`) are git-ignored because they exceed GitHub's file size limit.
- Ground-truth constants in `mission_config.py` (box, obstacle and truck positions) are used only to build the world and score results, never by the robot at runtime.
