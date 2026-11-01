"""
INDOOR LOCALIZATION - MISSION VIDEO PANELS (recording / display only)
=====================================================================

Builds the 1280x720 mission video frame:

    +---------------------------+---------------------------+
    | CHASE CAMERA (3D)         | LANE CAMERA (centre RGB)  |
    |  bot always in view,      |  lane mask tinted, fitted |
    |  camera FOVs + A* route   |  centreline, depth inset  |
    |  drawn on the floor       |  (arm stereo when picking)|
    +---------------------------+---------------------------+
    | LIVE OCCUPANCY MAP (zoom) | MISSION STATUS            |
    |  lane layer, obstacles,   |                           |
    |  route, footprint, FOVs,  |                           |
    |  live lane detections     |                           |
    |  + full-map inset         |                           |
    +---------------------------+---------------------------+

Nothing here feeds the controller: it only draws what the controller
already has (estimated pose, map, route, lane detections) plus a
ground-truth 3D render of the scene.
"""

import math

import cv2
import numpy as np
import pybullet as p

import camera_rig
import lane_detector
import mission_config as cfg
from lane_localizer import _floor_lut

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
PANEL_W = VIDEO_WIDTH // 2
PANEL_H = VIDEO_HEIGHT // 2
CAPTION_H = 30                              # caption bar over the bottom of the lower panels

MAP_VIEW_W_M = 8.0                          # zoomed map window (16:9)
MAP_VIEW_H_M = MAP_VIEW_W_M * PANEL_H / PANEL_W
DEPTH_FOV_RANGE_M = 4.0                     # drawn range of the RGB-D wedge
LANE_FOV_RANGE_M = 3.2                      # lane voting range (occupancy_map.integrate_lane)

COL_DEPTH_FOV = (255, 200, 0)               # BGR: cyan-blue
COL_LANE_FOV = (0, 220, 255)                # BGR: yellow
COL_ROUTE = (255, 255, 0)
COL_ROBOT = (0, 230, 0)


def label(img, text, color=(0, 255, 255), scale=0.55):
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return img


# ------------------------------------------------------------------ FOV geometry (body frame)

def _depth_fov_polygon():
    """Floor wedge of the two forward RGB-D cameras (union), body frame."""

    K = camera_rig.RGBD_INTRINSICS
    half_h = math.atan((camera_rig.RGBD_WIDTH / 2.0) / K[0, 0])
    lx, ly, _ = camera_rig.MOUNTS["left"]["offset"]
    rx, ry, _ = camera_rig.MOUNTS["right"]["offset"]
    poly = [(rx, ry)]
    for a in np.linspace(-half_h, half_h, 17):
        ox, oy = (rx, ry) if a < 0 else (lx, ly)
        poly.append((ox + DEPTH_FOV_RANGE_M * math.cos(a), oy + DEPTH_FOV_RANGE_M * math.sin(a)))
    poly.append((lx, ly))
    return np.array(poly)


def _lane_fov_polygon(lut):
    """Floor footprint of the centre (lane) camera inside the voting range."""

    pts = lut.reshape(-1, 2)
    pts = pts[np.isfinite(pts[:, 0])]
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < LANE_FOV_RANGE_M]
    return cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2).astype(np.float64)


LANE_LUT = _floor_lut()
DEPTH_FOV_BODY = _depth_fov_polygon()
LANE_FOV_BODY = _lane_fov_polygon(LANE_LUT)


def body_to_world(poly, x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack((x + poly[:, 0] * c - poly[:, 1] * s, y + poly[:, 0] * s + poly[:, 1] * c), 1)


def footprint_world(x, y, yaw):
    hl, hw = cfg.AMR_HALF_LENGTH_M, cfg.AMR_HALF_WIDTH_M
    return body_to_world(np.array([(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]), x, y, yaw)


def densify(path, step):
    out = [path[0]]
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
        for k in range(1, n + 1):
            out.append((x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n))
    return out


# ------------------------------------------------------------------ chase camera

class ChaseCamera:
    """Third-person camera that keeps the AMR in view. The view direction
    follows the robot heading through a low-pass filter (no whip-pan when
    it spins on the spot); close-ups for docking, arm and conveyor work."""

    def __init__(self):
        self.yaw = None
        self.view = None
        self.proj = None

    def render(self, robot_id, mode, close_target=None):
        pos, q = p.getBasePositionAndOrientation(robot_id)
        ryaw = p.getEulerFromQuaternion(q)[2]
        if self.yaw is None:
            self.yaw = ryaw
        self.yaw += 0.12 * math.atan2(math.sin(ryaw - self.yaw), math.cos(ryaw - self.yaw))
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        if mode == "conveyor" and close_target is not None:
            # Look back along the aisle over the robot at the belt nose:
            # aisles are clear end to end, the racks beside them are not.
            tx, ty = close_target
            eye = [pos[0] + 1.9 * math.cos(ryaw) + 0.35 * math.sin(ryaw),
                   pos[1] + 1.9 * math.sin(ryaw) - 0.35 * math.cos(ryaw), 2.1]
            target = [tx, ty, 0.5]
        elif mode == "close" and close_target is not None:
            tx, ty = close_target
            eye = [tx - 1.6 * cy + 1.1 * sy, ty - 1.6 * sy - 1.1 * cy, 2.2]
            target = [tx, ty, 0.45]
        else:
            eye = [pos[0] - 2.7 * cy + 0.5 * sy, pos[1] - 2.7 * sy - 0.5 * cy, 2.5]
            target = [pos[0] + 1.0 * cy, pos[1] + 1.0 * sy, 0.2]
        self.view = p.computeViewMatrix(eye, target, [0, 0, 1])
        self.proj = p.computeProjectionMatrixFOV(fov=62.0, aspect=PANEL_W / PANEL_H,
                                                 nearVal=0.1, farVal=40.0)
        img = p.getCameraImage(width=PANEL_W, height=PANEL_H, viewMatrix=self.view,
                               projectionMatrix=self.proj, renderer=p.ER_TINY_RENDERER)
        rgba = np.reshape(img[2], (PANEL_H, PANEL_W, 4)).astype(np.uint8)
        return rgba[:, :, :3][:, :, ::-1].copy()

    def project(self, world_xyz):
        """World points (N,3) -> pixels (N,2) and a mask of points in front."""

        V = np.array(self.view).reshape(4, 4).T
        P = np.array(self.proj).reshape(4, 4).T
        h = np.hstack((world_xyz, np.ones((world_xyz.shape[0], 1))))
        clip = (P @ V @ h.T).T
        front = clip[:, 3] > 1e-3
        ndc = clip[:, :3] / np.where(front, clip[:, 3], 1.0)[:, None]
        u = (ndc[:, 0] + 1.0) * 0.5 * PANEL_W
        v = (1.0 - ndc[:, 1]) * 0.5 * PANEL_H
        return np.stack((u, v), 1), front

    def draw_floor_poly(self, img, poly_xy, color, alpha, z=0.015):
        pts3 = np.hstack((poly_xy, np.full((poly_xy.shape[0], 1), z)))
        uv, front = self.project(pts3)
        if not front.all():
            return
        pts = np.round(uv).astype(np.int32)
        over = img.copy()
        cv2.fillPoly(over, [pts], color, cv2.LINE_AA)
        cv2.addWeighted(over, alpha, img, 1.0 - alpha, 0, dst=img)
        cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)

    def draw_polyline(self, img, xy, color, z=0.03, thickness=2):
        if xy is None or len(xy) < 2:
            return
        pts3 = np.hstack((np.asarray(xy, dtype=np.float64), np.full((len(xy), 1), z)))
        uv, front = self.project(pts3)
        for i in range(len(uv) - 1):
            if front[i] and front[i + 1]:
                cv2.line(img, tuple(int(c) for c in np.round(uv[i])),
                         tuple(int(c) for c in np.round(uv[i + 1])), color, thickness, cv2.LINE_AA)


def chase_panel(chase, io, controller, mode, close_target):
    img = chase.render(io.robot_id, mode, close_target)
    gx, gy, gyaw = io.true_pose()
    if mode == "drive":
        chase.draw_floor_poly(img, body_to_world(DEPTH_FOV_BODY, gx, gy, gyaw), COL_DEPTH_FOV, 0.18)
        chase.draw_floor_poly(img, body_to_world(LANE_FOV_BODY, gx, gy, gyaw), COL_LANE_FOV, 0.25)
    path = controller.active_path
    if path:
        chase.draw_polyline(img, densify(path, 0.25), COL_ROUTE)
    if mode == "drive":
        return label(img, "CHASE CAMERA | blue = RGB-D depth FOV | yellow = lane-camera FOV | cyan = A* route",
                     scale=0.45)
    return label(img, "CLOSE-UP CAMERA (belt nose)" if mode == "conveyor" else "CLOSE-UP CAMERA")


# ------------------------------------------------------------------ lane camera panel

_LANE_GEOM = {}


def _lane_geom(w, h):
    if (w, h) not in _LANE_GEOM:
        eye, fwd, up = camera_rig.build_camera_frame(0.0, 0.0, 0.0, "center")
        right = np.cross(fwd, up)
        R = np.column_stack((right, -up, fwd))
        f = (h / 2.0) / math.tan(math.radians(camera_rig.CAMERA_FOV_DEG) / 2.0)
        _LANE_GEOM[(w, h)] = (eye, R, f)
    return _LANE_GEOM[(w, h)]


def lane_camera_panel(io, controller, depth_vis):
    """Centre (lane) camera at display resolution with the lane detection
    overlaid, the fitted centreline the controller is tracking, and a
    depth inset from the forward RGB-D camera."""

    w, h = 480, PANEL_H
    rgb = io.render_camera("center", w, h)
    det = lane_detector.detect_lane(rgb)
    out = rgb.copy()
    if det.mask is not None:
        tint = np.zeros_like(out)
        tint[det.mask > 0] = (0, 255, 0)
        out = cv2.addWeighted(out, 1.0, tint, 0.45, 0)
        cnts, _ = cv2.findContours(det.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 255, 0), 1, cv2.LINE_AA)

    eye, R, f = _lane_geom(w, h)

    def to_px(fw, lat):
        pc = R.T @ (np.array([fw, lat, 0.0]) - eye)
        if pc[2] <= 0.05:
            return None
        return int(f * pc[0] / pc[2] + w / 2.0), int(f * pc[1] / pc[2] + h / 2.0)

    a0, a1 = to_px(0.6, 0.0), to_px(3.0, 0.0)          # straight-ahead heading line
    if a0 and a1:
        cv2.line(out, a0, a1, (255, 255, 255), 1, cv2.LINE_AA)
    fit = controller.lane_localizer.last_fit
    if fit is not None:
        a, b = fit
        pts = [to_px(fw, a + b * fw) for fw in np.linspace(0.6, 3.0, 12)]
        pts = [q for q in pts if q is not None]
        for q0, q1 in zip(pts[:-1], pts[1:]):
            cv2.line(out, q0, q1, (255, 0, 255), 3, cv2.LINE_AA)
        txt = f"tracked lane centreline: offset {a*100:+.0f} cm, angle {math.degrees(math.atan(b)):+.1f} deg"
        col = (255, 0, 255)
    else:
        txt = "lane seen (junction / curve - no straight strip to fit)" if det.detected else "no lane in view"
        col = (200, 200, 200)
    cv2.rectangle(out, (0, h - 26), (w, h), (0, 0, 0), -1)
    cv2.putText(out, txt, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    sw = PANEL_W - w
    side = np.full((PANEL_H, sw, 3), 20, np.uint8)
    dh = int(sw * 0.75)
    cv2.putText(side, "RGB-D depth", (6, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    side[48:48 + dh] = cv2.resize(depth_vis, (sw, dh))
    y0 = 48 + dh + 24
    cv2.putText(side, "lane mask (HSV)", (6, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    if det.mask is not None:
        m = cv2.resize(det.mask, (sw, dh), interpolation=cv2.INTER_NEAREST) > 0
        side[y0:y0 + dh][m] = (0, 255, 0)
    y1 = y0 + dh + 20
    for t, c in (("green   lane pixels", (0, 255, 0)), ("magenta tracked line", (255, 0, 255)),
                 ("white   robot heading", (255, 255, 255))):
        if y1 < PANEL_H - 4:
            cv2.putText(side, t, (6, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1, cv2.LINE_AA)
        y1 += 16
    panel = np.hstack((out, side))
    return label(panel, f"LANE CAMERA - live lane detection: {'LANE DETECTED' if det.detected else 'no lane'}"
                        f" | drive cmd {io.last_command}", (0, 255, 0) if det.detected else (0, 200, 255), 0.5)


# ------------------------------------------------------------------ occupancy map panel

def _map_base(occ):
    _prob, occupied, free, _unknown = occ.masks()
    img = np.full((cfg.GRID_ROWS, cfg.GRID_COLS, 3), (55, 55, 55), np.uint8)
    img[free] = (232, 232, 232)
    lane = occ.lane_mask & ~occupied
    img[lane] = np.where(occ._visited_lane[lane][:, None], np.array([150, 225, 250], np.uint8),
                         np.array([40, 195, 240], np.uint8))
    img[occupied] = (40, 40, 210)
    return np.flipud(img).copy()


def _world_to_inset(x, y, iw, ih):
    return (int((x - cfg.WORLD_X_MIN) / (cfg.WORLD_X_MAX - cfg.WORLD_X_MIN) * iw),
            int((cfg.WORLD_Y_MAX - y) / (cfg.WORLD_Y_MAX - cfg.WORLD_Y_MIN) * ih))


class MapPanel:
    """Robot-centred zoom of the live occupancy map (estimated pose)."""

    def __init__(self):
        self.scale = PANEL_W / MAP_VIEW_W_M          # px per metre
        self.cx = self.cy = None

    def render(self, controller, pose, lane_mask_small):
        occ = controller.occupancy
        base = _map_base(occ)
        cell_px = self.scale * cfg.GRID_RESOLUTION_M
        big = cv2.resize(base, (int(round(cfg.GRID_COLS * cell_px)), int(round(cfg.GRID_ROWS * cell_px))),
                         interpolation=cv2.INTER_NEAREST)
        # Window centre leads the robot a little so the camera FOV is in view.
        ax, ay = pose.x + 1.2 * math.cos(pose.yaw), pose.y + 0.8 * math.sin(pose.yaw)
        if self.cx is None:
            self.cx, self.cy = ax, ay
        self.cx += 0.2 * (ax - self.cx)
        self.cy += 0.2 * (ay - self.cy)
        hx, hy = MAP_VIEW_W_M / 2, MAP_VIEW_H_M / 2
        cx = float(np.clip(self.cx, cfg.WORLD_X_MIN + hx, cfg.WORLD_X_MAX - hx))
        cy = float(np.clip(self.cy, cfg.WORLD_Y_MIN + hy, cfg.WORLD_Y_MAX - hy))
        x0, y1 = cx - hx, cy + hy

        def px(x, y):
            return int(round((x - x0) * self.scale)), int(round((y1 - y) * self.scale))

        c0 = int(round((x0 - cfg.WORLD_X_MIN) * self.scale))
        r0 = int(round((cfg.WORLD_Y_MAX - y1) * self.scale))
        img = big[r0:r0 + PANEL_H, c0:c0 + PANEL_W].copy()
        if img.shape[:2] != (PANEL_H, PANEL_W):
            img = cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_NEAREST)

        for gx in range(int(math.floor(x0)), int(math.ceil(x0 + MAP_VIEW_W_M)) + 1):     # 1 m grid
            u = px(gx, 0)[0]
            cv2.line(img, (u, 0), (u, PANEL_H), (110, 110, 110), 1)
        for gy in range(int(math.floor(y1 - MAP_VIEW_H_M)), int(math.ceil(y1)) + 1):
            v = px(0, gy)[1]
            cv2.line(img, (0, v), (PANEL_W, v), (110, 110, 110), 1)

        def poly(world, color, alpha):
            pts = np.array([px(x, y) for x, y in world], np.int32)
            over = img.copy()
            cv2.fillPoly(over, [pts], color, cv2.LINE_AA)
            cv2.addWeighted(over, alpha, img, 1 - alpha, 0, dst=img)
            cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)

        poly(body_to_world(DEPTH_FOV_BODY, pose.x, pose.y, pose.yaw), COL_DEPTH_FOV, 0.15)
        poly(body_to_world(LANE_FOV_BODY, pose.x, pose.y, pose.yaw), COL_LANE_FOV, 0.20)

        # This frame's lane detections, projected onto the floor.
        if lane_mask_small is not None:
            sel = lane_mask_small[::2, ::2] > 0
            pts = LANE_LUT[::2, ::2][sel]
            pts = pts[np.isfinite(pts[:, 0])]
            pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < LANE_FOV_RANGE_M]
            if pts.size:
                for wx, wy in body_to_world(pts, pose.x, pose.y, pose.yaw):
                    cv2.circle(img, px(wx, wy), 2, (0, 190, 0), -1)

        path = controller.active_path
        if path and len(path) > 1:
            pts = np.array([px(x, y) for x, y in path], np.int32)
            cv2.polylines(img, [pts], False, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.polylines(img, [pts], False, COL_ROUTE, 3, cv2.LINE_AA)
        goal = controller.nav.goal
        if goal is not None:
            cv2.drawMarker(img, px(*goal), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 3)
        for name, pt in controller.semantic.named_points.items():
            u, v = px(pt.x, pt.y)
            if 0 <= u < PANEL_W and 0 <= v < PANEL_H:
                cv2.circle(img, (u, v), 9, (0, 140, 255), 2, cv2.LINE_AA)
                cv2.putText(img, name, (u + 11, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, name, (u + 11, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA)

        fp = np.array([px(x, y) for x, y in footprint_world(pose.x, pose.y, pose.yaw)], np.int32)
        cv2.fillPoly(img, [fp], (0, 110, 0), cv2.LINE_AA)
        cv2.polylines(img, [fp], True, COL_ROBOT, 2, cv2.LINE_AA)
        tip = px(pose.x + 0.55 * math.cos(pose.yaw), pose.y + 0.55 * math.sin(pose.yaw))
        cv2.arrowedLine(img, px(pose.x, pose.y), tip, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)

        # Full-map inset (top-right) with the zoom window marked.
        iw = 176
        ih = int(iw * cfg.GRID_ROWS / cfg.GRID_COLS)
        inset = cv2.resize(base, (iw, ih), interpolation=cv2.INTER_AREA)
        if path and len(path) > 1:
            ip = np.array([_world_to_inset(x, y, iw, ih) for x, y in path], np.int32)
            cv2.polylines(inset, [ip], False, COL_ROUTE, 1, cv2.LINE_AA)
        for name, pt in controller.semantic.named_points.items():
            cv2.circle(inset, _world_to_inset(pt.x, pt.y, iw, ih), 3, (0, 140, 255), 1, cv2.LINE_AA)
        cv2.circle(inset, _world_to_inset(pose.x, pose.y, iw, ih), 3, COL_ROBOT, -1)
        cv2.rectangle(inset, _world_to_inset(x0, y1, iw, ih),
                      _world_to_inset(x0 + MAP_VIEW_W_M, y1 - MAP_VIEW_H_M, iw, ih), (255, 255, 255), 1)
        cv2.rectangle(inset, (0, 0), (iw - 1, ih - 1), (255, 255, 255), 1)
        iy = PANEL_H - CAPTION_H - ih - 6
        img[iy:iy + ih, PANEL_W - iw - 6:PANEL_W - 6] = inset

        leg = [("lane layer", (40, 195, 240)), ("lane driven", (150, 225, 250)), ("obstacle", (40, 40, 210)),
               ("free (mapped)", (232, 232, 232)), ("unknown", (55, 55, 55)), ("lane seen now", (0, 190, 0)),
               ("A* route", COL_ROUTE), ("RGB-D FOV", COL_DEPTH_FOV), ("lane-cam FOV", COL_LANE_FOV)]
        y = PANEL_H - CAPTION_H - 8 - 15 * (len(leg) - 1)
        cv2.rectangle(img, (4, y - 14), (126, PANEL_H - CAPTION_H - 2), (25, 25, 25), -1)
        for t, col in leg:
            cv2.rectangle(img, (10, y - 9), (20, y + 1), col, -1)
            cv2.putText(img, t, (26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (255, 255, 255), 1, cv2.LINE_AA)
            y += 15
        return img
