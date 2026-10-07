"""Camera-independent ball detection + 3D localization.

Coordinate frame (camera optical frame, origin O(0,0,0) = color camera center):
    +X -> right in the image
    +Y -> down in the image
    +Z -> forward, out of the lens
All 3D values are in millimeters.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# HSV ranges in OpenCV units (H: 0-179, S: 0-255, V: 0-255).
# Tune these for your lighting / balls. Red wraps around hue 0, so it has two ranges.
COLOR_RANGES: Dict[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = {
    "red":    [((0, 120, 70), (8, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "orange": [((9, 120, 70), (21, 255, 255))],
    "yellow": [((22, 100, 100), (35, 255, 255))],
    "green":  [((36, 80, 50), (85, 255, 255))],
    "blue":   [((90, 100, 50), (130, 255, 255))],
    "purple": [((131, 60, 50), (169, 255, 255))],
}

# BGR colors used to draw each detection.
DRAW_COLORS = {
    "red": (0, 0, 255), "orange": (0, 140, 255), "yellow": (0, 255, 255),
    "green": (0, 200, 0), "blue": (255, 80, 0), "purple": (200, 0, 200),
}


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class Ball:
    color: str
    u: float                 # pixel column of ball center
    v: float                 # pixel row of ball center
    radius_px: float
    # Filled in by locate_ball() when depth is available:
    surface_xyz: Optional[Tuple[float, float, float]] = None  # front surface point (mm)
    center_xyz: Optional[Tuple[float, float, float]] = None   # estimated ball center (mm)
    distance_mm: Optional[float] = None                        # |O -> ball center|
    diameter_mm: Optional[float] = None


def color_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def detect_balls(bgr: np.ndarray,
                 min_radius_px: float = 8,
                 min_circularity: float = 0.70,
                 min_fill_ratio: float = 0.60) -> Tuple[List[Ball], Dict[str, np.ndarray]]:
    """Find round, uniformly colored blobs. Returns (balls, masks-by-color)."""
    blurred = cv2.GaussianBlur(bgr, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    balls: List[Ball] = []
    masks: Dict[str, np.ndarray] = {}
    for color in COLOR_RANGES:
        mask = color_mask(hsv, color)
        masks[color] = mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if area <= 0 or perimeter <= 0:
                continue
            (u, v), r = cv2.minEnclosingCircle(cnt)
            if r < min_radius_px:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)  # 1.0 = perfect circle
            fill_ratio = area / (np.pi * r * r)                       # blob fills its circle?
            if circularity < min_circularity or fill_ratio < min_fill_ratio:
                continue
            balls.append(Ball(color=color, u=u, v=v, radius_px=r))
    return balls, masks


def deproject(u: float, v: float, z_mm: float, k: Intrinsics) -> Tuple[float, float, float]:
    """Pinhole back-projection of pixel (u, v) at depth Z into camera coordinates."""
    x = (u - k.cx) * z_mm / k.fx
    y = (v - k.cy) * z_mm / k.fy
    return x, y, z_mm


def locate_ball(ball: Ball, depth_mm: np.ndarray, k: Intrinsics,
                color_mask_img: Optional[np.ndarray] = None,
                min_depth_mm: float = 100, max_depth_mm: float = 10000) -> bool:
    """Compute 3D position of a ball. `depth_mm` must be aligned to the color image.

    Uses the median depth of the inner 60% of the ball disk (robust to edge
    pixels that hit the background) as the depth of the ball's front surface,
    then pushes back by the ball radius to estimate the ball's center.
    """
    h, w = depth_mm.shape[:2]
    sample = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(sample, (int(round(ball.u)), int(round(ball.v))),
               max(1, int(ball.radius_px * 0.6)), 255, -1)
    if color_mask_img is not None:
        sample &= color_mask_img

    values = depth_mm[sample > 0]
    values = values[(values > min_depth_mm) & (values < max_depth_mm)]
    if values.size < 5:
        return False

    z_surface = float(np.median(values))
    ball.surface_xyz = deproject(ball.u, ball.v, z_surface, k)

    # Real radius from apparent size: R = r_px * Z / f
    f = (k.fx + k.fy) / 2
    radius_mm = ball.radius_px * z_surface / f
    ball.diameter_mm = 2 * radius_mm

    # The ball center lies on the ray through (u, v), one radius beyond the surface.
    sx, sy, sz = ball.surface_xyz
    surface_dist = float(np.sqrt(sx * sx + sy * sy + sz * sz))
    scale = (surface_dist + radius_mm) / surface_dist
    ball.center_xyz = (sx * scale, sy * scale, sz * scale)
    ball.distance_mm = surface_dist + radius_mm
    return True


def draw(bgr: np.ndarray, balls: List[Ball]) -> np.ndarray:
    out = bgr.copy()
    for b in balls:
        c = DRAW_COLORS.get(b.color, (255, 255, 255))
        center = (int(b.u), int(b.v))
        cv2.circle(out, center, int(b.radius_px), c, 2)
        cv2.circle(out, center, 3, c, -1)
        lines = [b.color]
        if b.center_xyz is not None:
            x, y, z = b.center_xyz
            lines.append(f"X={x:.0f} Y={y:.0f} Z={z:.0f} mm")
            lines.append(f"dist={b.distance_mm / 1000:.3f} m  d={b.diameter_mm:.0f} mm")
        else:
            lines.append("no depth")
        tx, ty = int(b.u + b.radius_px + 6), int(b.v - b.radius_px)
        for i, text in enumerate(lines):
            pos = (tx, ty + 18 * i)
            cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    return out
