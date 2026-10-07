"""Synthetic test (no camera needed): render balls with known 3D positions,
then check that detection + deprojection recovers them."""
import cv2
import numpy as np

from ball_detector import Intrinsics, detect_balls, locate_ball

K = Intrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
W, H = 640, 480


def render(balls):
    """balls: list of (bgr, center_xyz_mm, radius_mm). Returns color + depth images."""
    bgr = np.full((H, W, 3), 90, np.uint8)
    depth = np.full((H, W), 3000.0, np.float32)  # flat wall 3 m away
    vs, us = np.mgrid[0:H, 0:W]
    for color, (X, Y, Z), R in balls:
        # Ray-sphere intersection for each pixel: ray dir d = ((u-cx)/fx, (v-cy)/fy, 1)
        dx = (us - K.cx) / K.fx
        dy = (vs - K.cy) / K.fy
        a = dx * dx + dy * dy + 1
        b = -2 * (dx * X + dy * Y + Z)
        c = X * X + Y * Y + Z * Z - R * R
        disc = b * b - 4 * a * c
        hit = disc >= 0
        t = (-b - np.sqrt(np.where(hit, disc, 0))) / (2 * a)  # t == Z of the hit point
        hit &= t < depth
        depth[hit] = t[hit]
        bgr[hit] = color
    return bgr, depth


def test_recovers_positions():
    truth = {
        "red":    ((0, 0, 255),   (0.0, 0.0, 1000.0), 33.0),
        "blue":   ((255, 60, 0),  (-300.0, 100.0, 1500.0), 50.0),
        "yellow": ((0, 230, 255), (250.0, -150.0, 800.0), 20.0),
    }
    bgr, depth = render(truth.values())
    balls, masks = detect_balls(bgr)
    found = {b.color: b for b in balls}
    assert set(found) == set(truth), found.keys()

    for name, (_, (X, Y, Z), R) in truth.items():
        b = found[name]
        assert locate_ball(b, depth, K, masks[name])
        cx, cy, cz = b.center_xyz
        err = np.linalg.norm(np.subtract((cx, cy, cz), (X, Y, Z)))
        true_dist = np.linalg.norm((X, Y, Z))
        print(f"{name:>6}: est=({cx:7.1f},{cy:7.1f},{cz:7.1f}) true=({X},{Y},{Z}) "
              f"err={err:.1f}mm dist={b.distance_mm:.1f}/{true_dist:.1f} "
              f"diam={b.diameter_mm:.1f}/{2 * R}")
        assert err < 0.03 * true_dist, err
        assert abs(b.diameter_mm - 2 * R) < 0.15 * 2 * R


def test_rejects_non_round():
    bgr = np.full((H, W, 3), 90, np.uint8)
    cv2.rectangle(bgr, (100, 100), (400, 140), (0, 0, 255), -1)  # red bar
    balls, _ = detect_balls(bgr)
    assert balls == [], balls


if __name__ == "__main__":
    test_recovers_positions()
    test_rejects_non_round()
    print("all tests passed")
