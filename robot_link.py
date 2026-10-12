"""Send ball positions to the robot arm controller as JSON over UDP.

One datagram per frame:
    {"t": 1760000000.123, "frame": "robot",
     "balls": [{"color": "red", "x": 312.4, "y": -45.0, "z": 33.1, "diameter": 65.0}, ...]}

"frame" is "robot" once the workspace is calibrated (x, y, z in the robot frame),
otherwise "camera" (camera optical frame; don't move the arm with those).
Balls without a valid position are left out. Units: mm.

In object mode (main.py --objects) each entry also has "kind" ("pen", "marker", "object"),
"length" (mm), and "yaw" (degrees from robot +X toward +Y, -90..90; null in camera frame);
"diameter" is the object's width.

Receiver example (on the arm controller):
    import json, socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(("0.0.0.0", 5005))
    while True: print(json.loads(s.recv(65536)))
"""
import json
import socket
import time
from typing import List

from ball_detector import Ball


def balls_to_message(balls: List[Ball]) -> dict:
    use_robot = any(b.robot_xyz is not None for b in balls)
    out = []
    for b in balls:
        p = b.robot_xyz if use_robot else b.center_xyz
        if p is None:
            continue
        msg = {"color": b.color, "x": round(p[0], 1), "y": round(p[1], 1), "z": round(p[2], 1),
               "diameter": None if b.diameter_mm is None else round(b.diameter_mm, 1)}
        if hasattr(b, "kind"):   # object mode (object_detector.DetectedObject)
            yaw = b.robot_yaw_deg if use_robot else None
            msg.update(kind=b.kind,
                       length=None if b.length_mm is None else round(b.length_mm, 1),
                       yaw=None if yaw is None else round(yaw, 1))
        out.append(msg)
    return {"t": time.time(), "frame": "robot" if use_robot else "camera", "balls": out}


class UdpSender:
    def __init__(self, target: str):
        host, port = target.rsplit(":", 1)
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"Sending ball positions to udp://{host}:{port}")

    def send(self, balls: List[Ball]) -> None:
        try:
            self.sock.sendto(json.dumps(balls_to_message(balls)).encode(), self.addr)
        except OSError as e:   # receiver not up yet, network down, ...: keep running
            print("UDP send failed:", e)
