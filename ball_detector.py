"""Camera-independent ball detection + 3D localization.

Positions are first computed in the camera optical frame (center_xyz):
    origin = color camera center, +X right in the image, +Y down, +Z out of the lens
workspace.py then converts them to the robot frame (robot_xyz), which is what
the arm should use. All 3D values are in millimeters.
"""
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

Range = Tuple[Tuple[int, int, int], Tuple[int, int, int]]  # (low, high), inclusive


@dataclass
class ColorSpec:
    """A color to detect. A pixel matches if it falls in ANY of the ranges.

    hsv_ranges use OpenCV units (H: 0-179, S: 0-255, V: 0-255).
    rgb_ranges are in normal R, G, B order (0-255).
    """
    hsv_ranges: List[Range] = field(default_factory=list)
    rgb_ranges: List[Range] = field(default_factory=list)
    draw_bgr: Tuple[int, int, int] = (255, 255, 255)


# Default colors. Red wraps around hue 0, so it has two ranges.
DEFAULT_COLORS: Dict[str, ColorSpec] = {
    "red":    ColorSpec([((0, 120, 70), (8, 255, 255)), ((170, 120, 70), (179, 255, 255))],
                        draw_bgr=(0, 0, 255)),
    "orange": ColorSpec([((9, 120, 70), (21, 255, 255))], draw_bgr=(0, 140, 255)),
    "yellow": ColorSpec([((22, 100, 100), (35, 255, 255))], draw_bgr=(0, 255, 255)),
    "green":  ColorSpec([((36, 80, 50), (85, 255, 255))], draw_bgr=(0, 200, 0)),
    "blue":   ColorSpec([((90, 100, 50), (130, 255, 255))], draw_bgr=(255, 80, 0)),
    "purple": ColorSpec([((131, 60, 50), (169, 255, 255))], draw_bgr=(200, 0, 200)),
}

# Reference colors (RGB) used to give any color a human-readable name.
NAMED_RGB = {
    "red": (220, 30, 30), "dark red": (130, 0, 0), "orange": (255, 140, 0),
    "yellow": (250, 220, 20), "lime": (160, 230, 30), "green": (30, 160, 50),
    "dark green": (0, 90, 30), "teal": (0, 140, 140), "cyan": (0, 220, 230),
    "light blue": (120, 180, 240), "blue": (20, 60, 220), "navy": (10, 20, 100),
    "purple": (120, 40, 170), "violet": (190, 120, 230), "magenta": (230, 0, 190),
    "pink": (255, 150, 190), "light pink": (255, 195, 205), "brown": (130, 75, 30), "beige": (220, 200, 160),
    "white": (245, 245, 245), "gray": (128, 128, 128), "black": (20, 20, 20),
}


def _rgb_to_lab(rgb) -> np.ndarray:
    px = np.uint8([[list(rgb)[::-1]]])  # RGB -> BGR pixel
    return cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


_NAMED_LAB = {name: _rgb_to_lab(rgb) for name, rgb in NAMED_RGB.items()}


def name_color(rgb) -> str:
    """Closest named color in CIELAB space, which roughly matches human perception."""
    lab = _rgb_to_lab(rgb)
    return min(_NAMED_LAB, key=lambda n: float(np.linalg.norm(_NAMED_LAB[n] - lab)))


# ---------- loading / saving color specs (JSON) ----------
#
# colors.json example. Each entry may use any combination of these keys:
# {
#   "red":    {"hsv": [[[0,120,70],[8,255,255]], [[170,120,70],[179,255,255]]]},
#   "teal":   {"rgb_min": [0,100,100], "rgb_max": [80,200,200]},
#   "pink":   {"rgb": [255,150,190], "tolerance": 40},
#   "myball": {"hsv": [[[100,80,40],[115,255,255]]], "draw_rgb": [30,60,200]}
# }

def spec_from_json(entry: dict) -> ColorSpec:
    spec = ColorSpec()
    for lo, hi in entry.get("hsv", []):
        spec.hsv_ranges.append((tuple(lo), tuple(hi)))
    if "rgb_min" in entry and "rgb_max" in entry:
        spec.rgb_ranges.append((tuple(entry["rgb_min"]), tuple(entry["rgb_max"])))
    if "rgb" in entry:
        tol = int(entry.get("tolerance", 40))
        r = entry["rgb"]
        spec.rgb_ranges.append((tuple(max(0, c - tol) for c in r),
                                tuple(min(255, c + tol) for c in r)))
    if "draw_rgb" in entry:
        spec.draw_bgr = tuple(entry["draw_rgb"][::-1])
    elif "rgb" in entry:
        spec.draw_bgr = tuple(entry["rgb"][::-1])
    elif spec.rgb_ranges:
        lo, hi = spec.rgb_ranges[0]
        spec.draw_bgr = tuple((l + h) // 2 for l, h in zip(lo, hi))[::-1]
    elif spec.hsv_ranges:
        (h0, _, _), (h1, _, _) = spec.hsv_ranges[0]
        px = np.uint8([[[(h0 + h1) // 2, 255, 255]]])
        spec.draw_bgr = tuple(int(c) for c in cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0])
    if not spec.hsv_ranges and not spec.rgb_ranges:
        raise ValueError(f"color entry has no hsv / rgb / rgb_min+rgb_max: {entry}")
    return spec


def spec_to_json(spec: ColorSpec) -> dict:
    out: dict = {}
    if spec.hsv_ranges:
        out["hsv"] = [[list(lo), list(hi)] for lo, hi in spec.hsv_ranges]
    if spec.rgb_ranges:
        lo, hi = spec.rgb_ranges[0]
        out["rgb_min"], out["rgb_max"] = list(lo), list(hi)
    out["draw_rgb"] = [int(c) for c in spec.draw_bgr[::-1]]
    return out


def load_colors(path: str) -> Dict[str, ColorSpec]:
    with open(path) as f:
        data = json.load(f)
    return {name: spec_from_json(entry) for name, entry in data.items()}


def save_colors(path: str, colors: Dict[str, ColorSpec]) -> None:
    with open(path, "w") as f:
        json.dump({n: spec_to_json(s) for n, s in colors.items()}, f, indent=2)


# ---------- learning a color from pixels ----------

def learn_color(bgr_pixels: np.ndarray, hue_margin: int = 6) -> ColorSpec:
    """Build an HSV range from sample pixels of a ball (N x 3 BGR uint8).

    HSV is used rather than RGB because hue stays fairly stable when the ball
    is in light or shadow, while R, G and B all change together.
    """
    px = bgr_pixels.reshape(-1, 1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    mean_bgr = tuple(int(c) for c in np.median(bgr_pixels.reshape(-1, 3), axis=0))

    # Allow the shadowed side to be about half as bright as the clicked spot, but not
    # near-black: dark hair, clothes and shadows have unreliable hue and match anything.
    v_lo = int(max(25, min(np.percentile(v, 2) - 50, np.percentile(v, 2) * 0.5)))
    v_hi = int(min(255, np.percentile(v, 98) + 50))

    if np.median(s) < 45:
        # White / gray / black ball: hue is meaningless, match on low saturation + brightness.
        s_hi = int(min(255, np.percentile(s, 98) + 30))
        return ColorSpec([((0, 0, v_lo), (179, s_hi, v_hi))], draw_bgr=mean_bgr)

    # Hue is circular (0 and 179 are both red), so measure spread around a circular mean.
    ang = h * (2 * np.pi / 180)
    mean_h = (np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2) % 180
    d = (h - mean_h + 90) % 180 - 90
    h_lo = int(np.floor(mean_h + np.percentile(d, 2) - hue_margin))
    h_hi = int(np.ceil(mean_h + np.percentile(d, 98) + hue_margin))
    s_lo = int(max(30, np.percentile(s, 2) - 40))

    ranges: List[Range] = []
    if h_lo < 0:
        ranges += [((h_lo + 180, s_lo, v_lo), (179, 255, v_hi)), ((0, s_lo, v_lo), (h_hi, 255, v_hi))]
    elif h_hi > 179:
        ranges += [((h_lo, s_lo, v_lo), (179, 255, v_hi)), ((0, s_lo, v_lo), (h_hi - 180, 255, v_hi))]
    else:
        ranges.append(((h_lo, s_lo, v_lo), (h_hi, 255, v_hi)))
    return ColorSpec(ranges, draw_bgr=mean_bgr)


def sample_disk(bgr: np.ndarray, u: int, v: int, radius: int = 8) -> np.ndarray:
    mask = np.zeros(bgr.shape[:2], np.uint8)
    cv2.circle(mask, (u, v), radius, 255, -1)
    return bgr[mask > 0]


# ---------- detection ----------

@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class Ball:
    color: str               # name of the ColorSpec that matched
    u: float                 # pixel column of ball center
    v: float                 # pixel row of ball center
    radius_px: float
    rgb: Tuple[int, int, int] = (0, 0, 0)   # median measured color of the ball
    color_name: str = ""                     # closest named color to `rgb`
    contour: Optional[np.ndarray] = field(default=None, repr=False)  # outline pixels (N x 2)
    quality: float = 0.0     # circularity x fill ratio: 1.0 = perfect filled circle
    bulge_mm: Optional[float] = None   # depth only: how much the middle is closer than the rim
    # Filled in by locate_ball() when depth is available:
    surface_xyz: Optional[Tuple[float, float, float]] = None  # front surface point (mm)
    center_xyz: Optional[Tuple[float, float, float]] = None   # estimated ball center (mm)
    distance_mm: Optional[float] = None                        # |camera -> ball center|
    diameter_mm: Optional[float] = None
    # Filled in by workspace.py once the camera -> robot transform is calibrated:
    robot_xyz: Optional[Tuple[float, float, float]] = None    # ball center, robot frame (mm)


def _clean(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def color_mask(bgr: np.ndarray, hsv: np.ndarray, spec: ColorSpec) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in spec.hsv_ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    for lo, hi in spec.rgb_ranges:  # convert RGB bounds to OpenCV's BGR order
        mask |= cv2.inRange(bgr, np.array(lo[::-1]), np.array(hi[::-1]))
    return _clean(mask)


# ---------- background subtraction (fixed camera over a table) ----------

def _lab(bgr: np.ndarray, blur: int = 7) -> np.ndarray:
    return cv2.cvtColor(cv2.GaussianBlur(bgr, (blur, blur), 0), cv2.COLOR_BGR2LAB).astype(np.float32)


class Background:
    """Picture of the empty scene. Anything that differs from it is foreground, so
    balls of ANY color are found, including white/beige balls that color thresholds
    can't tell apart from a gray table or wall. Needs a fixed camera."""

    def __init__(self, frames: int = 15, blur: int = 7, clean=None):
        self.frames = frames
        self.blur = blur                               # smaller keeps thin objects (pens)
        self.clean = clean or _clean                   # mask cleanup, same idea
        self._acc: Optional[List[np.ndarray]] = None   # not None while capturing
        self.lab: Optional[np.ndarray] = None          # mean Lab image of the empty scene
        self.changed_fraction = 0.0                    # share of the image that differs (last mask)

    @property
    def capturing(self) -> bool:
        return self._acc is not None

    def start(self) -> None:
        self._acc = []
        self.lab = None

    def clear(self) -> None:
        self._acc = None
        self.lab = None

    def update(self, bgr: np.ndarray) -> bool:
        """Call every frame. Returns True on the frame the background becomes ready."""
        if self._acc is None:
            return False
        self._acc.append(_lab(bgr, self.blur))
        if len(self._acc) < self.frames:
            return False
        self.lab = np.mean(self._acc, axis=0)
        self._acc = None
        return True

    def mask(self, bgr: np.ndarray, threshold: float = 25.0,
             brighter_only: bool = False) -> np.ndarray:
        """Pixels that differ from the empty scene (soft shadows mostly ignored).

        brighter_only: count only pixels that got BRIGHTER. For a white / cream ball on a
        darker table this ignores hands, arms and hair, but also BLACK / dark balls."""
        d = _lab(bgr, self.blur) - self.lab
        # Webcams keep re-adjusting exposure and white balance, which shifts the WHOLE
        # image. The ball covers a small part of it, so the median change is that global
        # shift: remove it, leaving only what really changed.
        d -= np.median(d.reshape(-1, 3), axis=0)
        dist = np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
        # Shadow: the table got somewhat darker (kept 40-95% of its brightness) but kept
        # its color. A black ball is much darker than that, so it still counts.
        ratio = (self.lab[..., 0] + d[..., 0]) / np.maximum(self.lab[..., 0], 1.0)
        chroma_change = np.hypot(d[..., 1], d[..., 2])
        shadow = (d[..., 0] < 0) & (ratio > 0.4) & (ratio < 0.95) & (chroma_change < 12)
        dist[shadow] = 0
        if brighter_only:
            dist[d[..., 0] <= 0] = 0
        mask = self.clean(np.uint8(dist > threshold) * 255)
        self.changed_fraction = float(np.count_nonzero(mask)) / mask.size
        return mask


@dataclass
class Rejected:
    """A colored blob that was NOT accepted as a ball, and why (for the debug view)."""
    color: str
    u: float
    v: float
    radius_px: float
    reason: str


def resample_outline(pts: np.ndarray, n: int = 64) -> np.ndarray:
    """n points evenly spaced along a closed polygon, so every part of the outline
    counts equally when averaging (raw contours bunch points on some edges)."""
    closed = np.vstack([pts, pts[:1]]).astype(np.float64)
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    dist = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, dist[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, dist, closed[:, 0]), np.interp(t, dist, closed[:, 1])])


def detect_balls(bgr: np.ndarray,
                 colors: Optional[Dict[str, ColorSpec]] = None,
                 min_radius_px: float = 8,
                 max_radius_px: Optional[float] = None,
                 min_circularity: float = 0.70,
                 min_fill_ratio: float = 0.60,
                 rejected: Optional[List[Rejected]] = None,
                 background: Optional[Background] = None,
                 bg_threshold: float = 25.0,
                 bg_brighter_only: bool = False) -> Tuple[List[Ball], Dict[str, np.ndarray]]:
    """Find round, uniformly colored blobs. Returns (balls, masks-by-color-name).

    With a ready `background`, colors are ignored: round blobs that differ from the
    empty scene are reported as color "ball".

    Shape is judged on the blob's convex hull: on a real ball, shiny highlights and
    shadows cut notches into the color mask that would otherwise make it look
    non-round. Pass a list as `rejected` to collect blobs that failed, with the reason.
    """
    colors = DEFAULT_COLORS if colors is None else colors
    blurred = cv2.GaussianBlur(bgr, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    balls: List[Ball] = []
    if background is not None and background.lab is not None:
        masks = {"ball": background.mask(bgr, bg_threshold, bg_brighter_only)}
    else:
        masks = {name: color_mask(blurred, hsv, spec) for name, spec in colors.items()}
    for name, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            area = cv2.contourArea(hull)
            perimeter = cv2.arcLength(hull, True)
            if area <= 0 or perimeter <= 0:
                continue
            (u, v), r = cv2.minEnclosingCircle(hull)
            if max_radius_px is not None and r > max_radius_px:
                if rejected is not None:
                    rejected.append(Rejected(name, u, v, r, f"too big r={r:.0f}px"))
                continue
            if r < min_radius_px:
                if rejected is not None and r >= min_radius_px / 2:
                    rejected.append(Rejected(name, u, v, r, f"small r={r:.0f}px"))
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)  # 1.0 = perfect circle
            fill_ratio = area / (np.pi * r * r)                       # blob fills its circle?
            if circularity < min_circularity or fill_ratio < min_fill_ratio:
                if rejected is not None:
                    rejected.append(Rejected(name, u, v, r,
                                             f"circ={circularity:.2f} fill={fill_ratio:.2f}"))
                continue
            blob = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(blob, [cnt], -1, 255, -1)
            b, g, rr = (int(c) for c in np.median(blurred[blob > 0], axis=0))
            balls.append(Ball(color=name, u=u, v=v, radius_px=r, quality=circularity * fill_ratio,
                              rgb=(rr, g, b), color_name=name_color((rr, g, b)),
                              contour=resample_outline(cnt.reshape(-1, 2), 128)))
    return balls, masks


class StableFilter:
    """Only pass balls seen at about the same place for `min_frames` frames in a row.
    Noise blobs flicker for a frame or two; a real ball stays put."""

    def __init__(self, min_frames: int = 3):
        self.min_frames = min_frames
        self._tracks: List[Tuple[float, float, float, int]] = []   # (u, v, r, frames seen)

    def update(self, balls: List[Ball]) -> List[Ball]:
        if self.min_frames <= 1:
            return balls
        tracks, stable = [], []
        for b in balls:
            seen = 0
            for u, v, r, n in self._tracks:
                if (np.hypot(b.u - u, b.v - v) < max(8.0, 0.5 * r)
                        and abs(b.radius_px - r) < max(5.0, 0.3 * r)):
                    seen = max(seen, n)
            tracks.append((b.u, b.v, b.radius_px, seen + 1))
            if seen + 1 >= self.min_frames:
                stable.append(b)
        self._tracks = tracks
        return stable


def best_balls(balls: List[Ball], n: int) -> List[Ball]:
    """The n most ball-like detections (roundest and best filled, not biggest)."""
    return sorted(balls, key=lambda b: -b.quality)[:n]


def draw_rejected(view: np.ndarray, rejected: List[Rejected],
                  min_circularity: float = 0.70, min_fill_ratio: float = 0.60,
                  footer: Optional[str] = None) -> None:
    """Debug overlay: gray circles + reason for blobs that were not accepted."""
    for rj in rejected:
        c = (int(rj.u), int(rj.v))
        cv2.circle(view, c, int(rj.radius_px), (160, 160, 160), 1)
        text = f"{rj.color}: {rj.reason}"
        pos = (int(rj.u - rj.radius_px), int(rj.v + rj.radius_px + 14))
        # dark box behind the text (a thick black outline misaligns in OpenCV 5)
        (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(view, (pos[0] - 2, pos[1] - h - 3), (pos[0] + w + 2, pos[1] + base), (0, 0, 0), -1)
        cv2.putText(view, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    footer = footer or f"debug: need circ>={min_circularity:.2f} fill>={min_fill_ratio:.2f}"
    cv2.putText(view, footer,
                (10, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


def deproject(u: float, v: float, z_mm: float, k: Intrinsics) -> Tuple[float, float, float]:
    """Pinhole back-projection of pixel (u, v) at depth Z into camera coordinates."""
    x = (u - k.cx) * z_mm / k.fx
    y = (v - k.cy) * z_mm / k.fy
    return x, y, z_mm


def _depth_in(depth_mm, ball, r0, r1, mask, lo, hi) -> np.ndarray:
    """Valid depth values in the ring r0..r1 (fractions of the ball's pixel radius)."""
    ring = np.zeros(depth_mm.shape[:2], np.uint8)
    c = (int(round(ball.u)), int(round(ball.v)))
    cv2.circle(ring, c, max(1, int(round(ball.radius_px * r1))), 255, -1)
    if r0 > 0:
        cv2.circle(ring, c, int(round(ball.radius_px * r0)), 0, -1)
    if mask is not None:
        ring &= mask
    v = depth_mm[ring > 0]
    return v[(v > lo) & (v < hi)]


def check_ball_3d(ball: Ball, expected_diameter_mm: Optional[float] = None,
                  size_tolerance: float = 0.35, min_bulge_ratio: float = 0.08) -> Optional[str]:
    """Reject depth detections that are not ball-shaped or not ball-sized.
    Returns the reason, or None if it passes. Needs locate_ball() first."""
    if ball.diameter_mm is None:
        return None
    R = ball.diameter_mm / 2
    if expected_diameter_mm and abs(ball.diameter_mm - expected_diameter_mm) > \
            size_tolerance * expected_diameter_mm:
        return f"size {ball.diameter_mm:.0f}mm, expected {expected_diameter_mm:.0f}"
    if min_bulge_ratio > 0 and ball.bulge_mm is not None and ball.bulge_mm < min_bulge_ratio * R:
        return f"flat in depth (bulge {ball.bulge_mm:.1f}mm, sphere ~{0.2 * R:.0f})"
    return None


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

    # 3D shape: a sphere's middle is closer to the camera than the area near its rim
    # (by ~0.2 R between the inner 30% and the 55-80% ring); a flat mark is not.
    center_vals = _depth_in(depth_mm, ball, 0.0, 0.3, color_mask_img, min_depth_mm, max_depth_mm)
    ring_vals = _depth_in(depth_mm, ball, 0.55, 0.8, color_mask_img, min_depth_mm, max_depth_mm)
    if center_vals.size >= 3 and ring_vals.size >= 5:
        ball.bulge_mm = float(np.median(ring_vals) - np.median(center_vals))

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


def cone_axis(ball: Ball, k: Intrinsics) -> Tuple[Optional[np.ndarray], float]:
    """Unit direction (camera frame) from the camera to the ball's center, and the
    half-angle of the cone the ball fills. The sphere's center lies exactly on this
    axis. Uses the outline pixels when available, which stays correct for balls
    far from the image center (they appear stretched into ellipses)."""
    if ball.radius_px <= 0:
        return None, 0.0
    if ball.contour is not None and len(ball.contour) >= 8:
        pts = ball.contour
        rays = np.column_stack([(pts[:, 0] - k.cx) / k.fx, (pts[:, 1] - k.cy) / k.fy,
                                np.ones(len(pts))])
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        axis = rays.mean(axis=0)
        axis /= np.linalg.norm(axis)
        half_angle = float(np.mean(np.arccos(np.clip(rays @ axis, -1, 1))))
    else:  # no outline: treat the enclosing circle as centered, small-angle approx
        f = (k.fx + k.fy) / 2
        axis = np.array([(ball.u - k.cx) / k.fx, (ball.v - k.cy) / k.fy, 1.0])
        axis /= np.linalg.norm(axis)
        half_angle = float(np.arctan(ball.radius_px / f))
    return axis, half_angle


def locate_ball_by_size(ball: Ball, diameter_mm: float, k: Intrinsics) -> bool:
    """3D position from a single (non-depth) camera, using the ball's known real size.

    Seen from the camera, a sphere of radius R at distance d fills a cone of
    half-angle a with sin(a) = R / d, so d = R / sin(a). The ball's outline pixels
    are the edge of that cone: turn them into 3D rays, the cone axis is their mean
    direction and `a` is their average angle from it (see cone_axis).
    """
    axis, half_angle = cone_axis(ball, k)
    if axis is None or half_angle <= 0:
        return False
    R = diameter_mm / 2

    dist = R / np.sin(half_angle)
    ball.center_xyz = tuple(float(c) for c in axis * dist)
    ball.surface_xyz = tuple(float(c) for c in axis * (dist - R))
    ball.distance_mm = float(dist)
    ball.diameter_mm = float(diameter_mm)
    return True


def focal_from_known_distance(ball: Ball, diameter_mm: float, distance_mm: float,
                              k: Intrinsics) -> float:
    """Calibration: the focal length (px) that makes locate_ball_by_size() report
    `distance_mm` for this ball. Found by bisection since the outline method has no
    closed-form inverse. `k` supplies the principal point; its focal is ignored."""
    lo, hi = 50.0, 20000.0
    for _ in range(60):
        f = (lo + hi) / 2
        locate_ball_by_size(ball, diameter_mm, Intrinsics(f, f, k.cx, k.cy))
        # Longer focal -> smaller angles -> farther estimate.
        if ball.distance_mm < distance_mm:
            lo = f
        else:
            hi = f
    return (lo + hi) / 2


def draw(bgr: np.ndarray, balls: List[Ball],
         colors: Optional[Dict[str, ColorSpec]] = None) -> np.ndarray:
    colors = DEFAULT_COLORS if colors is None else colors
    out = bgr.copy()
    for b in balls:
        spec = colors.get(b.color)
        c = spec.draw_bgr if spec else (255, 255, 255)
        center = (int(b.u), int(b.v))
        cv2.circle(out, center, int(b.radius_px), c, 2)
        cv2.circle(out, center, 3, c, -1)
        label = b.color if b.color == b.color_name else f"{b.color} ({b.color_name})"
        lines = [f"{label} RGB{b.rgb}"]
        if b.robot_xyz is not None:
            x, y, z = b.robot_xyz
            lines.append(f"robot X={x:.0f} Y={y:.0f} Z={z:.0f} mm")
        elif b.center_xyz is not None:
            x, y, z = b.center_xyz
            lines.append(f"camera X={x:.0f} Y={y:.0f} Z={z:.0f} mm")
        if b.center_xyz is not None:
            lines.append(f"dist={b.distance_mm / 1000:.3f} m  d={b.diameter_mm:.0f} mm")
        else:
            lines.append("no depth")
        tx, ty = int(b.u + b.radius_px + 6), int(b.v - b.radius_px)
        for i, text in enumerate(lines):
            pos = (tx, ty + 18 * i)
            cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out
