"""Record every ball detection to a CSV file (one row per ball per frame).

Columns:
    timestamp        Unix time in seconds
    time_iso         same time, human readable (local time)
    frame            frame number since the program started
    color            name of the color spec that matched
    color_name       closest common color name of the measured RGB
    r, g, b          measured median color of the ball
    u_px, v_px       ball center in the image (pixels)
    radius_px        ball radius in the image (pixels)
    cam_x/y/z_mm     ball center in the camera frame (+X right, +Y down, +Z out of the lens)
    distance_mm      camera -> ball center
    diameter_mm      ball diameter (measured with depth, or the given --ball-diameter)
    robot_x/y/z_mm   ball center in the robot frame (empty until the robot frame is calibrated)
    method           how the 3D position was found: depth / table_plane / size / none
    kind             object mode only: pen / marker / object (diameter_mm is then the width)
    length_mm        object mode only: measured length
    robot_yaw_deg    object mode only: axis direction on the table (robot frame, -90..90)

Open it with Excel / Numbers / pandas:  pd.read_csv("balls_....csv")
"""
import csv
import os
import time
from datetime import datetime
from typing import List, Optional

from ball_detector import Ball

COLUMNS = ["timestamp", "time_iso", "frame", "color", "color_name", "r", "g", "b",
           "u_px", "v_px", "radius_px", "cam_x_mm", "cam_y_mm", "cam_z_mm",
           "distance_mm", "diameter_mm", "robot_x_mm", "robot_y_mm", "robot_z_mm", "method",
           "kind", "length_mm", "robot_yaw_deg"]


def default_csv_path() -> str:
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"balls_{datetime.now():%Y%m%d_%H%M%S}.csv")


def _num(x: Optional[float], digits: int = 1):
    return "" if x is None else round(float(x), digits)


class CsvLogger:
    def __init__(self, path: str):
        self.path = path
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self.file = open(path, "a", newline="")
        self.writer = csv.writer(self.file)
        if new:
            self.writer.writerow(COLUMNS)
        self.frame = 0
        self.rows = 0
        self._last_flush = time.time()
        print(f"Recording detections to {path}")

    def log(self, balls: List[Ball], methods: Optional[List[str]] = None) -> None:
        """Call once per frame with that frame's balls (`methods` parallel to `balls`)."""
        now = time.time()
        iso = datetime.fromtimestamp(now).isoformat(timespec="milliseconds")
        for i, b in enumerate(balls):
            cam = b.center_xyz or (None, None, None)
            rob = b.robot_xyz or (None, None, None)
            method = methods[i] if methods else ("none" if b.center_xyz is None else "")
            self.writer.writerow([
                round(now, 3), iso, self.frame, b.color, b.color_name, *b.rgb,
                _num(b.u), _num(b.v), _num(b.radius_px),
                *(_num(c) for c in cam), _num(b.distance_mm), _num(b.diameter_mm),
                *(_num(c) for c in rob), method, getattr(b, "kind", ""),
                _num(getattr(b, "length_mm", None)), _num(getattr(b, "robot_yaw_deg", None))])
            self.rows += 1
        self.frame += 1
        if now - self._last_flush > 1.0:   # don't lose much if the program is killed
            self.file.flush()
            self._last_flush = now

    def close(self) -> None:
        self.file.close()
        print(f"Saved {self.rows} detections from {self.frame} frames to {self.path}")
