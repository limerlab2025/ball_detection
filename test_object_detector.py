"""Synthetic test (no camera needed): render pens / markers lying on a table with known
3D positions, then check that detection + depth recovers kind, size, distance and yaw."""
import numpy as np
import cv2

from ball_detector import Intrinsics, Rejected
from object_detector import (classify_object, detect_objects, draw_objects, locate_object,
                             locate_object_in_robot, TableBackground)
from robot_link import balls_to_message

K = Intrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
W, H = 640, 480
TABLE_Z = 600.0   # camera looks straight down at a table 600 mm away
BLUE, RED, GREEN = (200, 60, 20), (30, 30, 210), (40, 170, 40)   # BGR


def project(p):
    x, y, z = p
    return K.fx * x / z + K.cx, K.fy * y / z + K.cy


def render(objects, table_bgr=(90, 90, 90)):
    """objects: list of (bgr, end0_xyz, end1_xyz, width_mm) with end points ON THE AXIS.
    The top surface (axis - width/2 toward the camera) gets the depth."""
    bgr = np.full((H, W, 3), table_bgr, np.uint8)
    depth = np.full((H, W), TABLE_Z, np.float32)
    vs, us = np.mgrid[0:H, 0:W].astype(np.float32)
    for color, e0, e1, width in objects:
        e0, e1 = np.array(e0, float), np.array(e1, float)
        p0, p1 = np.array(project(e0)), np.array(project(e1))
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        n = np.array([-d[1], d[0]])
        half_px = width / 2 * K.fx / ((e0[2] + e1[2]) / 2)
        # pixel centers within half the width of the axis segment (exact, unlike fillPoly
        # which paints its boundary too and makes everything ~1 px wider)
        along = (us - p0[0]) * d[0] + (vs - p0[1]) * d[1]
        across = (us - p0[0]) * n[0] + (vs - p0[1]) * n[1]
        hit = (along >= 0) & (along <= np.linalg.norm(p1 - p0)) & (np.abs(across) <= half_px)
        # depth along the axis: interpolate in the image (fine for a test)
        t = np.clip(along / np.linalg.norm(p1 - p0), 0, 1)
        z = e0[2] + (e1[2] - e0[2]) * t - width / 2
        bgr[hit] = color
        depth[hit] = z[hit]
    return bgr, depth


def on_table(cx, cy, length, width, yaw_deg, tilt_mm=0.0):
    """Axis end points of an object lying on the table, centered at (cx, cy) camera mm."""
    a = np.radians(yaw_deg)
    dx, dy = np.cos(a) * length / 2, np.sin(a) * length / 2
    z = TABLE_Z - width / 2
    return (cx - dx, cy - dy, z + tilt_mm / 2), (cx + dx, cy + dy, z - tilt_mm / 2)


def run(objects, **kw):
    bgr, depth = render(objects)
    rejected = []
    found, _ = detect_objects(bgr, rejected=rejected, **kw)
    for o in found:
        assert locate_object(o, depth, K)
        classify_object(o)
    return found, rejected


def test_pen_and_marker():
    pen = on_table(-60, 0, 140, 10, 30)
    marker = on_table(80, 20, 130, 19, -70)
    found, _ = run([(BLUE, *pen, 10), (RED, *marker, 19)])
    by_color = {o.color: o for o in found}
    assert set(by_color) == {"blue", "red"}, [o.color for o in found]
    p, m = by_color["blue"], by_color["red"]
    assert p.kind == "pen", (p.kind, p.length_mm, p.diameter_mm)
    assert m.kind == "marker", (m.kind, m.length_mm, m.diameter_mm)
    for o, (e0, e1), L in ((p, pen, 140), (m, marker, 130)):
        true_center = (np.array(e0) + np.array(e1)) / 2
        assert abs(o.length_mm - L) < 0.06 * L, o.length_mm
        assert np.linalg.norm(np.array(o.center_xyz) - true_center) < 6, (o.center_xyz, true_center)
        assert abs(o.distance_mm - np.linalg.norm(true_center)) < 6
    print(f"pen: L={p.length_mm:.1f} W={p.diameter_mm:.1f} dist={p.distance_mm:.1f}  "
          f"marker: L={m.length_mm:.1f} W={m.diameter_mm:.1f} dist={m.distance_mm:.1f}")


def test_tilted_pen_true_length():
    # One end propped up 50 mm: seen from above it looks shorter, depth gives it back.
    pen = on_table(0, 0, 140, 10, 0, tilt_mm=50)
    found, _ = run([(GREEN, *pen, 10)])
    assert len(found) == 1
    assert abs(found[0].length_mm - 140) < 10, found[0].length_mm


def test_round_blob_rejected():
    bgr, depth = render([])
    cv2.circle(bgr, (320, 240), 30, BLUE, -1)
    rejected = []
    found, _ = detect_objects(bgr, rejected=rejected)
    assert not found
    assert any("not long" in r.reason for r in rejected), [r.reason for r in rejected]


def test_robot_yaw_and_message():
    class Frame:   # robot frame = camera frame turned 180 deg about Z: same line on the table
        def to_robot(self, p):
            x, y, z = p
            return (-x, -y, TABLE_Z - z)

    pen = on_table(0, 0, 140, 10, 30)
    found, _ = run([(BLUE, *pen, 10)])
    o = found[0]
    assert locate_object_in_robot(o, Frame())
    assert abs(o.robot_yaw_deg - 30) < 3, o.robot_yaw_deg   # 180 deg rotation: same line
    msg = balls_to_message([o])
    assert msg["frame"] == "robot" and msg["balls"][0]["kind"] == "pen"
    assert abs(msg["balls"][0]["yaw"] - 30) < 3


def test_background_mode_white_markers():
    # Like a real desk: tan table, white-bodied markers whose only color is a cap / tip.
    # Color mode can't separate white from the table; background mode can.
    table = (80, 98, 108)   # BGR of a tan table
    white = (225, 225, 220)
    expo = on_table(-40, 10, 130, 19, 10)
    hl = on_table(60, -40, 140, 11, -15)

    def part(ends, start, stop):   # sub-segment of an object's axis, fractions 0..1
        e0, e1 = np.array(ends[0]), np.array(ends[1])
        return tuple(e0 + (e1 - e0) * start), tuple(e0 + (e1 - e0) * stop)

    empty, _ = render([], table)
    scene, depth = render([(white, *expo, 19), (BLUE, *part(expo, 0.75, 1.0), 19),
                           (white, *hl, 11), ((40, 220, 200), *part(hl, 0.9, 1.0), 11)], table)
    # a soft shadow next to the marker must not become part of it
    scene[int(240 + 35):int(240 + 45), 250:330] = (np.array(table) * 0.7).astype(np.uint8)

    assert not detect_objects(scene)[0], "color mode should not find white markers"
    bg = TableBackground(frames=3)
    bg.start()
    for _ in range(3):
        bg.update(empty)
    found, _ = detect_objects(scene, background=bg)
    for o in found:
        assert locate_object(o, depth, K)
        classify_object(o)
    got = sorted((o.kind, o.color) for o in found)
    assert got == [("marker", "blue"), ("pen", "yellow")], \
        [(o.kind, o.color, round(o.length_mm), round(o.diameter_mm)) for o in found]


def test_background_depth_finds_table_colored_marker():
    # Worst case from a real desk: the marker body is the SAME color as the table, so only
    # depth (19 mm above it) can find it; its color still comes from the blue cap.
    table = (80, 98, 108)
    expo = on_table(-40, 10, 130, 19, 10)
    cap = tuple(np.array(expo[0]) + (np.array(expo[1]) - np.array(expo[0])) * 0.75), expo[1]
    empty, empty_depth = render([], table)
    scene, depth = render([(table, *expo, 19), (BLUE, *cap, 19)], table)
    rng = np.random.default_rng(0)
    noisy = lambda d: d + rng.normal(0, 1.0, d.shape).astype(np.float32)   # ~1 mm sensor noise

    bg = TableBackground(frames=5)
    bg.start()
    for _ in range(5):
        bg.update(empty, noisy(empty_depth))
    found, _ = detect_objects(scene, background=bg, depth_mm=noisy(depth))
    assert len(found) == 1, [(o.color, o.length_px) for o in found]
    o = found[0]
    assert locate_object(o, depth, K) and classify_object(o) == "marker", (o.length_mm, o.diameter_mm)
    assert o.color == "blue" and abs(o.length_mm - 130) < 10, (o.color, o.length_mm)


def test_background_ignores_far_wall():
    # Top 40% of the picture is a wall 3 m away with noisy depth and flickering light,
    # like the door / window behind a desk. Only the table should be searched.
    table = (80, 98, 108)
    expo = on_table(-40, 60, 130, 19, 10)
    cap = tuple(np.array(expo[0]) + (np.array(expo[1]) - np.array(expo[0])) * 0.75), expo[1]
    rng = np.random.default_rng(1)
    wall = slice(0, int(0.4 * H))

    def frame(objects):
        bgr, depth = render(objects, table)
        bgr[wall] = (190, 195, 190)
        depth[wall] = 3000.0
        depth += rng.normal(0, 1.0, depth.shape).astype(np.float32)
        return bgr, depth

    bg = TableBackground(frames=5)
    bg.start()
    for _ in range(5):
        bg.update(*frame([]), K)
    assert bg.table is not None and not bg.table[wall].any(), "wall must not be table"
    scene, depth = frame([(table, *expo, 19), (BLUE, *cap, 19)])
    # speckles on the wall: depth glitches 1 m closer and light changes
    for _ in range(25):
        y, x = rng.integers(5, int(0.4 * H) - 15), rng.integers(5, W - 40)
        depth[y:y + 6, x:x + 30] -= 1000
        scene[y:y + 6, x:x + 30] = (60, 60, 60)
    found, _ = detect_objects(scene, background=bg, depth_mm=depth)
    assert len(found) == 1 and found[0].color == "blue", [(o.color, o.u, o.v) for o in found]


def test_draw_objects():
    # Drawing runs only with a window, so exercise it here (with and without robot frame).
    pen = on_table(-60, 0, 140, 10, 30)
    bgr, _ = render([(BLUE, *pen, 10)])
    found, _ = run([(BLUE, *pen, 10)])
    out = draw_objects(bgr, found)
    assert out.shape == bgr.shape and not np.array_equal(out, bgr)
    found[0].robot_xyz, found[0].robot_yaw_deg = (300.0, 20.0, 5.0), 30.0
    draw_objects(bgr, found)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all object tests passed")
