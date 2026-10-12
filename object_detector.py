"""Detection + 3D localization of long, thin colored objects (pens, markers, ...).

Uses the same color specs as the ball detector (colors.json, click-to-learn), but
instead of round blobs it keeps elongated ones and fits a rotated rectangle to each.
With aligned depth it then measures the object's real length and width (mm), which
is what tells a pen from a marker (see OBJECT_SIZES), and its two end points, which
give the orientation the robot needs to grasp it.

Frames and units are the same as ball_detector.py: camera optical frame, mm.
"""
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ball_detector import (DEFAULT_COLORS, Background, Ball, ColorSpec, Intrinsics, Rejected,
                           deproject, name_color, resample_outline)

Xyz = Tuple[float, float, float]

# Real sizes in mm: width = barrel diameter, length = tip to end (cap on).
# An object gets the first kind whose ranges contain BOTH measurements, otherwise "object".
# Measure your own pens/markers and adjust, or pass a JSON file with --object-sizes:
#   {"pen": {"width": [6, 13], "length": [90, 180]}, "highlighter": {"width": [15, 25], ...}}
OBJECT_SIZES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "pen":    {"width": (6, 13), "length": (90, 180)},
    "marker": {"width": (13, 30), "length": (80, 170)},
}


def load_object_sizes(path: str) -> Dict[str, Dict[str, Tuple[float, float]]]:
    with open(path) as f:
        data = json.load(f)
    return {name: {"width": tuple(e["width"]), "length": tuple(e["length"])}
            for name, e in data.items()}


@dataclass
class DetectedObject(Ball):
    """A Ball with a long axis. radius_px is half the object's length in the image and
    diameter_mm its measured width, so code written for balls (tracking, UDP, CSV,
    robot frame) keeps working; center_xyz is the middle of the object's axis."""
    kind: str = "object"                 # "pen", "marker", ... or "object" if no size matched
    length_px: float = 0.0
    width_px: float = 0.0
    angle_deg: float = 0.0               # long axis in the image: 0 = horizontal, + = clockwise
    ends_px: Optional[np.ndarray] = field(default=None, repr=False)        # 2 x 2 (u, v)
    raw_contour: Optional[np.ndarray] = field(default=None, repr=False)    # for the pixel mask
    # Filled in by locate_object():
    ends_xyz: Optional[Tuple[Xyz, Xyz]] = None    # axis end points, camera frame (mm)
    length_mm: Optional[float] = None
    # Filled in by locate_object_in_robot():
    robot_ends: Optional[Tuple[Xyz, Xyz]] = None
    robot_yaw_deg: Optional[float] = None         # axis direction on the table, -90..90 from +X


def _clean_thin(mask: np.ndarray) -> np.ndarray:
    # Much gentler than the ball cleaning (5x5, 2 iterations), which erases anything
    # under ~8 px wide, i.e. a pen half a meter away.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def object_mask(bgr: np.ndarray, hsv: np.ndarray, spec: ColorSpec) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in spec.hsv_ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    for lo, hi in spec.rgb_ranges:  # convert RGB bounds to OpenCV's BGR order
        mask |= cv2.inRange(bgr, np.array(lo[::-1]), np.array(hi[::-1]))
    return _clean_thin(mask)


def label(view: np.ndarray, text: str, pos, color=(255, 255, 255), scale: float = 0.5) -> None:
    """Text on a dark box. (Drawing it twice, thick black then thin white, misaligns in
    OpenCV 5, where thicker text is also wider.)"""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(pos[0]), int(pos[1])
    cv2.rectangle(view, (x - 2, y - h - 3), (x + w + 2, y + base), (0, 0, 0), -1)
    cv2.putText(view, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def fit_table(depth: np.ndarray, k: Intrinsics, iters: int = 300,
              rng: Optional[np.random.Generator] = None) -> Optional[np.ndarray]:
    """Pixels that see the table: the plane fitted (RANSAC) to the empty-table depth.

    Candidate points come from the bottom third of the image, where the table is (a
    wall can be the bigger plane overall). Returns a uint8 mask, or None if no plane."""
    h, w = depth.shape
    vs, us = np.mgrid[0:h, 0:w]
    z = depth
    pts = np.dstack([(us - k.cx) * z / k.fx, (vs - k.cy) * z / k.fy, z])   # NaN where no depth
    ok = np.isfinite(z)
    cand = np.argwhere(ok & (vs >= 2 * h // 3))
    if len(cand) < 500:
        cand = np.argwhere(ok)
    if len(cand) < 500:
        return None
    rng = rng or np.random.default_rng(0)
    sample = pts[tuple(cand[rng.choice(len(cand), min(5000, len(cand)), replace=False)].T)]
    tol = np.maximum(8.0, 0.015 * sample[:, 2])   # depth error grows with distance
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-6:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((sample - a) @ n) < tol
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    if best is None or best_n < 0.3 * len(sample):
        return None
    # Refine on the inliers (least squares plane through their centroid).
    p = sample[best]
    centroid = p.mean(axis=0)
    n = np.linalg.svd(p - centroid, full_matrices=False)[2][2]
    with np.errstate(invalid="ignore"):
        dist = np.abs((pts - centroid) @ n)
        on = ok & (dist < np.maximum(8.0, 0.015 * z) * 2)
    mask = cv2.morphologyEx(np.uint8(on) * 255, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    # Keep the big connected table area(s), drop small coplanar bits elsewhere.
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_lab <= 1:
        return None
    keep = [i for i in range(1, n_lab) if stats[i, cv2.CC_STAT_AREA] >= 0.2 * stats[1:, cv2.CC_STAT_AREA].max()]
    mask = np.uint8(np.isin(lab, keep)) * 255
    # Fill the holes left by shiny / dark spots so an object there still counts, and
    # stay a few px inside the edge, where depth mixes table and floor.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, -1)
    return cv2.erode(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))


class TableBackground:
    """The EMPTY table, in color and in depth. An object is anything that looks different
    from it OR sticks up above it. Depth is what finds a white marker on a light table in
    poor light, where its color differs from the table's by less than the table's own glare.

    Only the table itself is searched (fit_table): walls, windows and doors far behind it
    have noisy depth and changing light, which otherwise show up as specks.

    Color uses ball_detector.Background with a light blur / cleanup (its defaults erase
    anything under ~8 px wide, i.e. a pen half a meter away)."""

    def __init__(self, frames: int = 15, min_height_mm: float = 6.0, table_only: bool = True):
        self.color = Background(frames, blur=3, clean=_clean_thin)
        self.min_height_mm = min_height_mm
        self.table_only = table_only
        self.table: Optional[np.ndarray] = None    # uint8 mask of table pixels (None = everywhere)
        self.depth: Optional[np.ndarray] = None    # median empty-table depth (NaN = no data)
        self.noise: Optional[np.ndarray] = None    # per-pixel depth jitter while capturing
        self._depths: Optional[List[np.ndarray]] = None

    @property
    def lab(self):
        return self.color.lab

    @property
    def capturing(self) -> bool:
        return self.color.capturing

    @property
    def changed_fraction(self) -> float:
        return self.color.changed_fraction

    def start(self) -> None:
        self.color.start()
        self._depths = []
        self.depth = self.noise = self.table = None

    def clear(self) -> None:
        self.color.clear()
        self._depths = None
        self.depth = self.noise = self.table = None

    def update(self, bgr: np.ndarray, depth_mm: Optional[np.ndarray] = None,
               k: Optional[Intrinsics] = None) -> bool:
        """Call every frame. Returns True on the frame the empty table is captured."""
        if self._depths is not None and depth_mm is not None:
            self._depths.append(depth_mm.astype(np.float32, copy=True))
        if not self.color.update(bgr):
            return False
        if self._depths:
            stack = np.stack(self._depths)
            stack[stack <= 0] = np.nan   # 0 = no depth
            enough = np.sum(~np.isnan(stack), axis=0) >= len(self._depths) / 2
            with np.errstate(all="ignore"):   # all-NaN pixels warn
                self.depth = np.where(enough, np.nanmedian(stack, axis=0), np.nan)
                self.noise = np.where(enough, np.nanstd(stack, axis=0), np.nan)
            if self.table_only and k is not None:
                self.table = fit_table(self.depth, k)
                if self.table is None:
                    print("Could not find the table plane in the depth image: searching the "
                          "whole picture.")
                else:
                    print(f"Table found: {np.count_nonzero(self.table) / self.table.size:.0%} "
                          "of the image; only objects on it are detected.")
        self._depths = None
        return True

    def mask(self, bgr: np.ndarray, threshold: float = 25.0, brighter_only: bool = False,
             depth_mm: Optional[np.ndarray] = None) -> np.ndarray:
        mask = self.color.mask(bgr, threshold, brighter_only)
        if depth_mm is not None and self.depth is not None and depth_mm.shape == self.depth.shape:
            # Closer than the empty table by more than min_height (or 3x that pixel's noise).
            with np.errstate(invalid="ignore"):
                limit = np.maximum(self.min_height_mm, 3 * self.noise)
                above = (depth_mm > 0) & (self.depth - depth_mm > limit)
            mask |= _clean_thin(np.uint8(above) * 255)
        if self.table is not None and self.table.shape == mask.shape:
            mask &= self.table
        return mask


def object_color(blurred: np.ndarray, hsv: np.ndarray, blob: np.ndarray,
                 colors: Dict[str, ColorSpec], min_share: float = 0.05
                 ) -> Tuple[str, Tuple[int, int, int]]:
    """Color of an object found without color (background mode): the color spec that
    covers most of its pixels, so a white marker with a blue cap is "blue". Needs only
    `min_share` of the object, since the colored part (cap, tip, label) is often small.
    Returns (name, median RGB of those pixels); else the closest named color of the whole."""
    inside = blob > 0
    total = np.count_nonzero(inside)
    best, best_px = None, None
    for name, spec in colors.items():
        hit = np.zeros(blob.shape, bool)
        for lo, hi in spec.hsv_ranges:
            hit |= cv2.inRange(hsv, np.array(lo), np.array(hi)) > 0
        for lo, hi in spec.rgb_ranges:
            hit |= cv2.inRange(blurred, np.array(lo[::-1]), np.array(hi[::-1])) > 0
        hit &= inside
        if np.count_nonzero(hit) >= max(5, min_share * total) and \
                (best_px is None or np.count_nonzero(hit) > np.count_nonzero(best_px)):
            best, best_px = name, hit
    px = blurred[best_px if best is not None else inside]
    b, g, r = (int(c) for c in np.median(px, axis=0))
    return (best or name_color((r, g, b))), (r, g, b)


def detect_objects(bgr: np.ndarray,
                   colors: Optional[Dict[str, ColorSpec]] = None,
                   min_length_px: float = 25,
                   max_length_px: Optional[float] = None,
                   min_width_px: float = 3,
                   min_aspect: float = 2.5,
                   min_fill_ratio: float = 0.55,
                   rejected: Optional[List[Rejected]] = None,
                   background: Optional[TableBackground] = None,
                   bg_threshold: float = 25.0,
                   bg_brighter_only: bool = False,
                   depth_mm: Optional[np.ndarray] = None
                   ) -> Tuple[List[DetectedObject], Dict[str, np.ndarray]]:
    """Find elongated, uniformly colored blobs. Returns (objects, masks-by-color-name).

    min_aspect: length / width in the image (a pen seen from above is ~10-15, a ball 1).
    min_fill_ratio: how much of its rotated rectangle the blob fills (rejects L / curved shapes).
    With a ready `background`, objects are whatever differs from the empty table, or (with
    `depth_mm`) sticks up above it, white or gray bodies included; each one's color then
    comes from object_color().
    """
    colors = DEFAULT_COLORS if colors is None else colors
    blurred = cv2.GaussianBlur(bgr, (3, 3), 0)   # light blur: a thin pen must keep its color
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    use_bg = background is not None and background.lab is not None
    if use_bg:
        masks = {"object": background.mask(bgr, bg_threshold, bg_brighter_only, depth_mm)}
    else:
        masks = {name: object_mask(blurred, hsv, spec) for name, spec in colors.items()}

    objects: List[DetectedObject] = []
    for name, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_width_px * min_length_px / 2:
                continue   # specks: not worth a debug label
            rect = cv2.minAreaRect(cnt)
            (u, v), (w, h), _ = rect
            length, width = max(w, h), min(w, h)

            def reject(reason: str) -> None:
                if rejected is not None:
                    rejected.append(Rejected(name, u, v, length / 2, reason))

            if max_length_px is not None and length > max_length_px:
                reject(f"too long {length:.0f}px")
                continue
            if length < min_length_px:
                reject(f"short {length:.0f}px")
                continue
            if width < min_width_px:
                reject(f"thin {width:.1f}px")
                continue
            aspect = length / width
            fill = area / (w * h)
            if aspect < min_aspect:
                reject(f"not long {aspect:.1f}:1")
                continue
            if fill < min_fill_ratio:
                reject(f"fill={fill:.2f}")
                continue

            # Long-axis direction from the box corners (minAreaRect's angle convention
            # differs between OpenCV versions; the corners don't).
            box = cv2.boxPoints(rect)
            e0, e1 = box[1] - box[0], box[2] - box[1]
            axis = e0 if np.linalg.norm(e0) >= np.linalg.norm(e1) else e1
            axis = axis / np.linalg.norm(axis)
            if axis[0] < 0 or (axis[0] == 0 and axis[1] < 0):
                axis = -axis   # pens have no front: always point it rightwards
            ends = np.array([(u, v) - axis * length / 2, (u, v) + axis * length / 2])

            blob = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(blob, [cnt], -1, 255, -1)
            if use_bg:
                color, (rr, g, b) = object_color(blurred, hsv, blob, colors)
            else:
                color = name
                b, g, rr = (int(c) for c in np.median(blurred[blob > 0], axis=0))
            objects.append(DetectedObject(
                color=color, u=u, v=v, radius_px=length / 2, quality=fill,
                rgb=(rr, g, b), color_name=name_color((rr, g, b)),
                contour=resample_outline(cnt.reshape(-1, 2), 128),
                length_px=length, width_px=width,
                angle_deg=float(np.degrees(np.arctan2(axis[1], axis[0]))),
                ends_px=ends, raw_contour=cnt))
    return objects, masks


def locate_object(obj: DetectedObject, depth_mm: np.ndarray, k: Intrinsics,
                  min_depth_mm: float = 100, max_depth_mm: float = 10000) -> bool:
    """3D position, length and width of an object. `depth_mm` must be aligned to color.

    The depth camera sees the object's top surface. Its median depth gives the scale
    (width_mm = width_px * Z / f); the median depth of each end quarter gives the end
    points, so a pen tilted toward the camera still gets its true length. Each point is
    then pushed back by half the width to lie on the object's axis.
    """
    h, w = depth_mm.shape[:2]
    blob = np.zeros((h, w), np.uint8)
    cv2.drawContours(blob, [obj.raw_contour], -1, 255, -1)
    # Edge pixels mix the object with the table behind it: use the inside when possible.
    inner = cv2.erode(blob, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    region = inner if np.count_nonzero(inner) >= 20 else blob

    ys, xs = np.nonzero(region)
    z = depth_mm[ys, xs]
    ok = (z > min_depth_mm) & (z < max_depth_mm)
    xs, ys, z = xs[ok], ys[ok], z[ok]
    if z.size < 5:
        return False
    z_mid = float(np.median(z))

    f = (k.fx + k.fy) / 2
    width_mm = obj.width_px * z_mid / f
    half_w = width_mm / 2

    # Depth of each end: median of the outer quarter of the object (centered 3/8 L from
    # the middle), extrapolated linearly to the end (1/2 L).
    (u0, v0), (u1, v1) = obj.ends_px
    dx, dy = (u1 - u0) / obj.length_px, (v1 - v0) / obj.length_px
    t = (xs - obj.u) * dx + (ys - obj.v) * dy   # pixels along the axis, 0 = middle
    quarter = obj.length_px / 4

    def end_depth(sel: np.ndarray) -> float:
        if np.count_nonzero(sel) < 3:
            return z_mid
        return z_mid + (float(np.median(z[sel])) - z_mid) / 0.75

    surface = [deproject(u0, v0, end_depth(t < -quarter), k),
               deproject(u1, v1, end_depth(t > quarter), k)]

    def push(p) -> np.ndarray:   # surface point -> object axis, along the viewing ray
        p = np.asarray(p, dtype=np.float64)
        return p * (np.linalg.norm(p) + half_w) / np.linalg.norm(p)

    e0, e1 = push(surface[0]), push(surface[1])
    center = (e0 + e1) / 2
    obj.surface_xyz = deproject(obj.u, obj.v, z_mid, k)
    obj.ends_xyz = (tuple(float(c) for c in e0), tuple(float(c) for c in e1))
    obj.center_xyz = tuple(float(c) for c in center)
    obj.distance_mm = float(np.linalg.norm(center))
    obj.length_mm = float(np.linalg.norm(e1 - e0))
    obj.diameter_mm = float(width_mm)
    return True


def classify_object(obj: DetectedObject,
                    sizes: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None) -> str:
    """Name the object by its measured size (needs locate_object first)."""
    sizes = OBJECT_SIZES if sizes is None else sizes
    obj.kind = "object"
    if obj.length_mm is not None and obj.diameter_mm is not None:
        for name, r in sizes.items():
            if (r["width"][0] <= obj.diameter_mm <= r["width"][1]
                    and r["length"][0] <= obj.length_mm <= r["length"][1]):
                obj.kind = name
                break
    return obj.kind


def locate_object_in_robot(obj: DetectedObject, frame) -> bool:
    """Center, end points and yaw on the table in the robot frame (workspace.WorkspaceFrame)."""
    if obj.center_xyz is None or obj.ends_xyz is None:
        return False
    obj.robot_xyz = frame.to_robot(obj.center_xyz)
    obj.robot_ends = (frame.to_robot(obj.ends_xyz[0]), frame.to_robot(obj.ends_xyz[1]))
    (x0, y0, _), (x1, y1, _) = obj.robot_ends
    yaw = np.degrees(np.arctan2(y1 - y0, x1 - x0))
    obj.robot_yaw_deg = float((yaw + 90) % 180 - 90)   # a pen at 100 deg is the same as -80
    return True


def draw_objects(bgr: np.ndarray, objects: List[DetectedObject],
                 colors: Optional[Dict[str, ColorSpec]] = None) -> np.ndarray:
    colors = DEFAULT_COLORS if colors is None else colors
    out = bgr.copy()
    for o in objects:
        spec = colors.get(o.color)
        c = spec.draw_bgr if spec else (255, 255, 255)
        (u0, v0), (u1, v1) = o.ends_px
        n = np.array([-(v1 - v0), u1 - u0]) / o.length_px * (o.width_px / 2 + 2)
        box = np.array([o.ends_px[0] + n, o.ends_px[1] + n, o.ends_px[1] - n, o.ends_px[0] - n])
        cv2.polylines(out, [np.int32(np.round(box))], True, c, 2)
        cv2.line(out, (int(u0), int(v0)), (int(u1), int(v1)), c, 1)
        cv2.circle(out, (int(o.u), int(o.v)), 3, c, -1)

        color_text = o.color if o.color == o.color_name else f"{o.color} ({o.color_name})"
        lines = [f"{o.kind}: {color_text}"]
        if o.robot_xyz is not None:
            x, y, z = o.robot_xyz
            lines.append(f"robot X={x:.0f} Y={y:.0f} Z={z:.0f} mm yaw={o.robot_yaw_deg:.0f}")
        elif o.center_xyz is not None:
            x, y, z = o.center_xyz
            lines.append(f"camera X={x:.0f} Y={y:.0f} Z={z:.0f} mm")
        if o.center_xyz is not None:
            lines.append(f"dist={o.distance_mm / 1000:.3f} m  L={o.length_mm:.0f} W={o.diameter_mm:.0f} mm")
        else:
            lines.append("no depth")
        tx, ty = int(max(u0, u1) + 8), int(min(v0, v1))
        for i, text in enumerate(lines):
            label(out, text, (tx, ty + 18 * i))
    return out
