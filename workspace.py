"""Robot / workspace coordinate frame.

The camera computes positions in its own optical frame, which moves whenever the
camera is bumped. A robot arm needs positions in ITS frame instead. This module
defines that frame and the transform from the camera to it.

Robot (world) frame, right-handed, millimeters:
    origin  = a point you choose on the robot base, at table height
    +X      = forward from the robot, along the table
    +Y      = to the robot's left, along the table
    +Z      = up, perpendicular to the table (table surface is Z = table_z)

Calibration: print an ArUco marker, lay it flat on the table at a known spot
relative to the robot origin, and press the calibrate key. The marker's pose in
the camera (from solvePnP) plus where you said it lies in the robot frame gives
the camera -> robot transform, saved to workspace.json:

    p_robot = T_robot_cam @ p_cam

Marker axes as drawn on screen: red = marker +X (its printed right side),
green = marker +Y (its printed top), blue = +Z (up, out of the table).
"""
import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ball_detector import Ball, Intrinsics, cone_axis

DEFAULT_WORKSPACE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "workspace.json")
ARUCO_DICT = cv2.aruco.DICT_4X4_50


def camera_matrix(k: Intrinsics) -> np.ndarray:
    return np.array([[k.fx, 0, k.cx], [0, k.fy, k.cy], [0, 0, 1]], dtype=np.float64)


def make_transform(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make_transform(R.T, -R.T @ t)


def marker_in_robot(x_mm: float, y_mm: float, z_mm: float = 0.0,
                    yaw_deg: float = 0.0) -> np.ndarray:
    """Pose of a marker lying flat on the table: center at (x, y, z) in the robot
    frame, its +X axis turned `yaw_deg` counter-clockwise from robot +X."""
    a = math.radians(yaw_deg)
    Rz = np.array([[math.cos(a), -math.sin(a), 0],
                   [math.sin(a), math.cos(a), 0],
                   [0, 0, 1]])
    return make_transform(Rz, (x_mm, y_mm, z_mm))


@dataclass
class WorkspaceFrame:
    T_robot_cam: np.ndarray = field(default_factory=lambda: np.eye(4))
    table_z: float = 0.0   # height of the table surface in the robot frame (mm)

    def to_robot(self, p_cam) -> Tuple[float, float, float]:
        p = self.T_robot_cam[:3, :3] @ np.asarray(p_cam, dtype=np.float64) + self.T_robot_cam[:3, 3]
        return tuple(float(c) for c in p)

    @property
    def camera_position(self) -> np.ndarray:
        """Camera optical center expressed in the robot frame (mm)."""
        return self.T_robot_cam[:3, 3].copy()

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"T_robot_cam": self.T_robot_cam.tolist(), "table_z": self.table_z,
                       "units": "mm"}, f, indent=2)

    @staticmethod
    def load(path: str) -> "WorkspaceFrame":
        with open(path) as f:
            d = json.load(f)
        return WorkspaceFrame(np.array(d["T_robot_cam"], dtype=np.float64),
                              float(d.get("table_z", 0.0)))


# ---------- ball position in the robot frame ----------

def locate_ball_in_robot(ball: Ball, frame: WorkspaceFrame) -> bool:
    """Convert a ball already located in camera coordinates (locate_ball or
    locate_ball_by_size) into the robot frame."""
    if ball.center_xyz is None:
        return False
    ball.robot_xyz = frame.to_robot(ball.center_xyz)
    return True


def locate_ball_on_table(ball: Ball, diameter_mm: float, k: Intrinsics,
                         frame: WorkspaceFrame) -> bool:
    """3D position of a ball RESTING ON THE TABLE, without depth.

    The ball's center lies on the ray from the camera through the middle of its
    image (the cone axis), and at height table_z + R above the table. Intersecting
    that ray with the plane Z = table_z + R gives the position. Much more accurate
    than judging distance from apparent size, since it doesn't depend on the
    ball's pixel radius.
    """
    axis_cam, _ = cone_axis(ball, k)
    if axis_cam is None:
        return False
    R = diameter_mm / 2
    R_rc = frame.T_robot_cam[:3, :3]
    origin = frame.camera_position
    direction = R_rc @ axis_cam
    if abs(direction[2]) < 1e-6:   # ray parallel to the table
        return False
    s = (frame.table_z + R - origin[2]) / direction[2]
    if s <= R:                     # plane is behind the camera / inside the ball
        return False
    ball.center_xyz = tuple(float(c) for c in axis_cam * s)
    ball.surface_xyz = tuple(float(c) for c in axis_cam * (s - R))
    ball.distance_mm = float(s)
    ball.diameter_mm = float(diameter_mm)
    ball.robot_xyz = tuple(float(c) for c in origin + direction * s)
    return True


# ---------- ArUco calibration ----------

def detect_marker(bgr: np.ndarray, k: Intrinsics, marker_size_mm: float,
                  marker_id: int, dist: Optional[np.ndarray] = None):
    """Find the marker and return (T_cam_marker 4x4, corners) or None."""
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
                                       cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return None
    for c, i in zip(corners, ids.flatten()):
        if i != marker_id:
            continue
        s = marker_size_mm / 2
        # ArUco corner order: top-left, top-right, bottom-right, bottom-left.
        obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, c.reshape(4, 2).astype(np.float64),
                                      camera_matrix(k),
                                      np.zeros(5) if dist is None else dist,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None
        return make_transform(cv2.Rodrigues(rvec)[0], tvec), c
    return None


def average_transforms(Ts: List[np.ndarray]) -> np.ndarray:
    """Mean translation + mean rotation (projected back onto a valid rotation)."""
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    U, _, Vt = np.linalg.svd(np.sum([T[:3, :3] for T in Ts], axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return make_transform(R, t)


class WorkspaceManager:
    """Holds the workspace frame and runs marker calibration from the camera loop.

    Key:  k = calibrate (collects `samples` frames of the marker, then saves)
    """

    def __init__(self, path: str = DEFAULT_WORKSPACE_PATH, marker_size_mm: float = 100.0,
                 marker_id: int = 0, marker_pose: Optional[np.ndarray] = None,
                 samples: int = 30):
        self.path = path
        self.marker_size_mm = marker_size_mm
        self.marker_id = marker_id
        self.T_robot_marker = marker_in_robot(0, 0) if marker_pose is None else marker_pose
        self.samples = samples
        self.frame: Optional[WorkspaceFrame] = None
        self._collected: Optional[List[np.ndarray]] = None   # not None while calibrating
        self._last_corners = None
        if os.path.exists(path):
            self.frame = WorkspaceFrame.load(path)
            cam = self.frame.camera_position
            print(f"Loaded robot frame from {path}; camera at "
                  f"({cam[0]:.0f}, {cam[1]:.0f}, {cam[2]:.0f}) mm in robot coordinates")
        else:
            print(f"No robot frame yet: place ArUco marker id {marker_id} "
                  f"({marker_size_mm:.0f} mm) on the table and press k. "
                  f"Positions are in camera coordinates until then.")

    @property
    def calibrating(self) -> bool:
        return self._collected is not None

    def handle_key(self, key: int) -> None:
        if key == ord("k"):
            self._collected = []
            print(f"Calibrating: keep marker id {self.marker_id} still and in view...")

    def update(self, bgr: np.ndarray, k: Intrinsics) -> None:
        """Call once per frame. Advances calibration when it is running."""
        if self._collected is None:
            return
        found = detect_marker(bgr, k, self.marker_size_mm, self.marker_id)
        self._last_corners = None if found is None else found[1]
        if found is None:
            return
        T_cam_marker = found[0]
        # Marker +Z must face the camera; otherwise IPPE picked the mirrored solution.
        if T_cam_marker[:3, 2] @ T_cam_marker[:3, 3] >= 0:
            return
        self._collected.append(T_cam_marker)
        if len(self._collected) < self.samples:
            return

        T_cam_marker = average_transforms(self._collected)
        spread = np.std([T[:3, 3] for T in self._collected], axis=0)
        T_robot_cam = self.T_robot_marker @ invert_transform(T_cam_marker)
        self.frame = WorkspaceFrame(T_robot_cam, table_z=float(self.T_robot_marker[2, 3]))
        self.frame.save(self.path)
        self._collected = None
        self._last_corners = None
        cam = self.frame.camera_position
        print(f"Robot frame calibrated and saved to {self.path}\n"
              f"  camera at ({cam[0]:.0f}, {cam[1]:.0f}, {cam[2]:.0f}) mm in robot coordinates, "
              f"{np.linalg.norm(T_cam_marker[:3, 3]):.0f} mm from the marker\n"
              f"  marker jitter (std): {spread[0]:.1f}, {spread[1]:.1f}, {spread[2]:.1f} mm")

    def draw(self, view: np.ndarray, k: Intrinsics) -> None:
        """Overlay: robot axes at the origin, or calibration progress."""
        if self._collected is not None:
            if self._last_corners is not None:
                cv2.aruco.drawDetectedMarkers(view, [self._last_corners])
            text = f"calibrating {len(self._collected)}/{self.samples}"
            cv2.putText(view, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            return
        if self.frame is None:
            return
        T_cam_robot = invert_transform(self.frame.T_robot_cam)
        if T_cam_robot[2, 3] <= 0:   # robot origin behind the camera
            return
        rvec = cv2.Rodrigues(T_cam_robot[:3, :3])[0]
        cv2.drawFrameAxes(view, camera_matrix(k), np.zeros(5), rvec, T_cam_robot[:3, 3], 100, 2)


def add_workspace_args(parser) -> None:
    g = parser.add_argument_group("robot frame (ArUco calibration, press k)")
    g.add_argument("--workspace", default=DEFAULT_WORKSPACE_PATH,
                   help="camera -> robot transform file")
    g.add_argument("--marker-size", type=float, default=100.0,
                   help="side of the printed marker's black square, mm")
    g.add_argument("--marker-id", type=int, default=0)
    g.add_argument("--marker-x", type=float, default=0.0,
                   help="marker center X in the robot frame, mm (forward from the robot origin)")
    g.add_argument("--marker-y", type=float, default=0.0,
                   help="marker center Y in the robot frame, mm (to the robot's left)")
    g.add_argument("--marker-z", type=float, default=0.0,
                   help="table surface height in the robot frame, mm (0 if the origin is at table height)")
    g.add_argument("--marker-yaw", type=float, default=0.0,
                   help="angle from robot +X to the marker's +X (red axis), degrees CCW")
    g.add_argument("--udp", default=None, metavar="HOST:PORT",
                   help="send ball positions as JSON to the arm controller, e.g. 127.0.0.1:5005")


def manager_from_args(args) -> WorkspaceManager:
    return WorkspaceManager(args.workspace, args.marker_size, args.marker_id,
                            marker_in_robot(args.marker_x, args.marker_y,
                                            args.marker_z, args.marker_yaw))


def save_marker_image(path: str, marker_id: int = 0, pixels: int = 800) -> None:
    """Write a printable marker PNG with a white border."""
    img = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
                                        marker_id, pixels)
    border = pixels // 8
    cv2.imwrite(path, cv2.copyMakeBorder(img, border, border, border, border,
                                         cv2.BORDER_CONSTANT, value=255))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate a printable ArUco marker for calibration.")
    p.add_argument("--id", type=int, default=0)
    p.add_argument("--out", default=None, help="default: marker_<id>.png")
    a = p.parse_args()
    a.out = a.out or f"marker_{a.id}.png"
    save_marker_image(a.out, a.id)
    print(f"Saved {a.out}. Print it at 100% scale and measure the black square's side "
          f"(pass it as --marker-size).")
