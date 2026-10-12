"""Synthetic test (no camera needed): render balls with known 3D positions,
then check that detection + deprojection recovers them."""
import os
import tempfile

import cv2
import numpy as np

from ball_detector import (Intrinsics, detect_balls, focal_from_known_distance, learn_color,
                           load_colors, locate_ball, locate_ball_by_size,
                           name_color, sample_disk, save_colors, spec_from_json)
from ball_detector import Background, Ball, StableFilter, best_balls, check_ball_3d
from color_ui import ColorManager
from robot_link import balls_to_message
from workspace import (ARUCO_DICT, WorkspaceFrame, WorkspaceManager, invert_transform,
                       locate_ball_in_robot, locate_ball_on_table, make_transform,
                       marker_in_robot)

K = Intrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
W, H = 640, 480


def render(balls, bgr=None):
    """balls: list of (bgr, center_xyz_mm, radius_mm). Returns color + depth images."""
    bgr = np.full((H, W, 3), 90, np.uint8) if bgr is None else bgr.copy()
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


PASTEL_PINK = (203, 192, 255)  # BGR; too unsaturated for the default "red" range


def test_ball_with_highlight_and_shadow():
    """Real balls: a glare spot on the rim and a dark lower edge cut notches into the
    color mask. The ball must still be detected."""
    bgr, _ = render([((0, 0, 220), (0.0, 0.0, 600.0), 33.0)])   # r ~ 33 px
    cv2.circle(bgr, (305, 212), 14, (255, 255, 255), -1)          # glare touching the rim
    vs, us = np.mgrid[0:H, 0:W]
    shadow = ((us - 320) ** 2 + (vs - 240) ** 2 < 34 ** 2) & ((us - 320) ** 2 + (vs - 222) ** 2 > 34 ** 2)
    bgr[shadow] = (15, 15, 40)                                      # dark crescent at the bottom
    rejected = []
    balls, _ = detect_balls(bgr, rejected=rejected)
    assert [b.color for b in balls] == ["red"], (balls, rejected)
    assert abs(balls[0].u - 320) < 6 and abs(balls[0].v - 240) < 8, balls[0]


def test_background_mode_finds_colorless_ball():
    """A beige / off-white ball against a gray scene has almost no color, so color
    thresholds fail. Background subtraction still finds it."""
    rng = np.random.default_rng(0)
    scene = np.clip(rng.normal(115, 6, (H, W, 3)), 0, 255).astype(np.uint8)   # gray, noisy
    bg = Background(frames=3)
    bg.start()
    for _ in range(3):
        bg.update(np.clip(scene + rng.normal(0, 3, scene.shape), 0, 255).astype(np.uint8))
    assert bg.lab is not None and not bg.capturing

    beige = (150, 170, 178)   # BGR: low saturation, a bit brighter than the background
    frame, _ = render([(beige, (0.0, 0.0, 800.0), 20.0)], bgr=scene)
    assert detect_balls(frame)[0] == [], "colors should not find a near-gray ball"
    balls, _ = detect_balls(frame, background=bg)
    assert len(balls) == 1 and balls[0].color == "ball", balls
    assert abs(balls[0].u - 320) < 3 and abs(balls[0].v - 240) < 3
    assert detect_balls(scene, background=bg)[0] == [], "empty scene must give nothing"

    # Webcam auto-exposure + white balance: the whole picture gets brighter and warmer
    # when the ball appears (without compensation, 100% of the image counts as changed).
    shifted = np.clip(frame.astype(np.float32) * [1.25, 1.5, 1.6], 0, 255).astype(np.uint8)
    balls, _ = detect_balls(shifted, background=bg)
    assert len(balls) == 1 and abs(balls[0].u - 320) < 3, balls
    assert bg.changed_fraction < 0.05, bg.changed_fraction

    # Something huge right in front of the lens is not a ball.
    big, _ = render([(beige, (0.0, 0.0, 150.0), 40.0)], bgr=scene)   # r ~ 160 px
    rejected = []
    assert detect_balls(big, background=bg, max_radius_px=H / 4, rejected=rejected)[0] == []
    assert any("too big" in r.reason for r in rejected), rejected


def test_background_cream_ball_with_shadow():
    """Measured from a real frame: cream ball (BGR ~100,115,120) on a gray floor
    (BGR ~80,80,78), with a dark shadow beside it. The ball differs from the floor mostly
    in brightness; the shadow must not be glued onto it."""
    rng = np.random.default_rng(2)
    floor = np.clip(rng.normal(0, 5, (H, W, 3)) + (80, 80, 78), 0, 255).astype(np.uint8)
    floor[50:190, 450:590] = (140, 140, 136)               # a lighter patch of floor
    bg = Background(frames=3)
    bg.start()
    for _ in range(3):
        bg.update(np.clip(floor + rng.normal(0, 3, floor.shape), 0, 255).astype(np.uint8))
    frame = floor.copy()
    cv2.ellipse(frame, (290, 290), (75, 40), 30, 0, 360, (44, 48, 43), -1)   # cast shadow
    frame, _ = render([((100, 115, 120), (0.0, 0.0, 400.0), 40.0)], bgr=frame)   # r ~ 60 px
    cv2.circle(frame, (520, 120), 45, (20, 22, 22), -1)     # dark round thing (fist / hair) on it
    balls, _ = detect_balls(frame, background=bg, max_radius_px=H / 4)
    assert len(balls) == 2, balls                          # both differ from the empty floor
    balls, _ = detect_balls(frame, background=bg, max_radius_px=H / 4, bg_brighter_only=True)
    assert len(balls) == 1, balls                          # only the (brighter) ball is left
    b = balls[0]
    print(f"cream ball: center=({b.u:.0f},{b.v:.0f}) r={b.radius_px:.0f}  (true 320,240 r~60)")
    assert abs(b.u - 320) < 5 and abs(b.v - 240) < 5 and abs(b.radius_px - 60) < 6


def test_background_black_ball_and_shadow():
    """A black ball on a mid-gray floor must be found (it is far darker than the floor),
    while a soft shadow of the same size (partly darker, same color) must not."""
    rng = np.random.default_rng(3)
    floor = np.clip(rng.normal(0, 5, (H, W, 3)) + (80, 80, 78), 0, 255).astype(np.uint8)
    bg = Background(frames=3)
    bg.start()
    for _ in range(3):
        bg.update(np.clip(floor + rng.normal(0, 3, floor.shape), 0, 255).astype(np.uint8))
    frame = floor.copy()
    cv2.circle(frame, (480, 240), 50, (46, 48, 45), -1)                   # round soft shadow
    frame, _ = render([((18, 22, 21), (-120.0, 0.0, 600.0), 40.0)], bgr=frame)   # black ball
    balls, _ = detect_balls(frame, background=bg, max_radius_px=H / 4)
    assert len(balls) == 1, balls
    print(f"black ball: center=({balls[0].u:.0f},{balls[0].v:.0f}) r={balls[0].radius_px:.0f} "
          f"(true ~200,240 r~37)")
    assert abs(balls[0].u - 200) < 5 and abs(balls[0].v - 240) < 5
    assert detect_balls(frame, background=bg, bg_brighter_only=True)[0] == []


def test_depth_rejects_flat_and_wrong_size():
    """Depth camera: a real sphere bulges toward the camera; a flat painted circle of the
    same color and size (sticker, mark on the table) does not."""
    for R, Z in [(20.0, 500.0), (33.0, 1000.0), (12.5, 500.0)]:   # 40 mm, 66 mm, 25 mm balls
        bgr, depth = render([((0, 0, 255), (-150.0, 0.0, Z), R)])   # real red ball, left
        r_px = K.fx * R / Z
        cv2.circle(bgr, (int(K.cx + K.fx * 150 / Z), 240), int(r_px), (0, 0, 255), -1)  # flat disk
        cv2.circle(depth, (int(K.cx + K.fx * 150 / Z), 240), int(r_px), Z, -1)          # on a wall
        balls, masks = detect_balls(bgr, min_radius_px=5)
        assert len(balls) == 2, balls
        verdict = {}
        for b in balls:
            assert locate_ball(b, depth, K, masks[b.color])
            verdict["sphere" if b.u < 320 else "flat"] = (check_ball_3d(b), b.bulge_mm)
        print(f"R={R}mm Z={Z}mm: sphere bulge={verdict['sphere'][1]:.1f}mm -> {verdict['sphere'][0]}, "
              f"flat bulge={verdict['flat'][1]:.1f}mm -> {verdict['flat'][0]}")
        assert verdict["sphere"][0] is None
        assert verdict["flat"][0] and "flat" in verdict["flat"][0]

        sphere = next(b for b in balls if b.u < 320)
        assert check_ball_3d(sphere, expected_diameter_mm=2 * R) is None
        assert "size" in check_ball_3d(sphere, expected_diameter_mm=4 * R)


def test_stable_filter_and_best_ball():
    sf = StableFilter(min_frames=3)
    ball = lambda u, v, r=30, q=0.8: Ball("ball", u, v, r, quality=q)
    out = []
    for i in range(5):
        noise = ball(100 + 80 * i, 400, 20)          # pops up somewhere new every frame
        out.append(sf.update([ball(320 + i, 240), noise]))   # real ball drifts 1 px/frame
    assert out[0] == [] and out[1] == [], "needs 3 frames before reporting"
    assert all(len(o) == 1 and abs(o[0].u - 320) < 6 for o in out[2:]), out

    big_ragged, small_round = ball(100, 100, 80, q=0.45), ball(300, 300, 25, q=0.85)
    assert best_balls([big_ragged, small_round], 1) == [small_round]


def test_click_learning_refines_and_rejects_dark():
    path = os.path.join(tempfile.mkdtemp(), "colors.json")
    cm = ColorManager(path)
    bgr, _ = render([(PASTEL_PINK, (0.0, 0.0, 1000.0), 40.0)])
    bgr[:, :100] = (8, 12, 7)                         # black shirt on the left
    cm.latest_bgr = bgr
    n0 = len(cm.colors)
    cm._on_mouse(cv2.EVENT_LBUTTONDOWN, 50, 240, 0, None)    # click black: refused
    assert len(cm.colors) == n0
    for x, y in [(320, 240), (330, 230), (310, 250)]:        # click the ball 3 times
        cm._on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
    assert len(cm.colors) == n0 + 1, list(cm.colors)          # one color, refined


def test_learn_unusual_color():
    bgr, _ = render([(PASTEL_PINK, (0.0, 0.0, 1000.0), 40.0)])
    balls, _ = detect_balls(bgr)
    assert balls == [], "defaults should not catch pastel pink"

    spec = learn_color(sample_disk(bgr, 320, 240))   # simulate clicking the ball
    balls, _ = detect_balls(bgr, {"mine": spec})
    assert len(balls) == 1 and balls[0].color == "mine", balls
    assert "pink" in balls[0].color_name, balls[0].color_name
    print("learned pink:", spec.hsv_ranges)


def test_learn_hue_wraparound_red():
    bgr, _ = render([((10, 10, 200), (0.0, 0.0, 1000.0), 40.0)])
    bgr[240:, :][np.all(bgr[240:, :] == (10, 10, 200), axis=-1)] = (40, 0, 200)  # hue ~ 175
    spec = learn_color(sample_disk(bgr, 320, 240, radius=15))
    assert len(spec.hsv_ranges) == 2, spec.hsv_ranges  # split across 179 -> 0
    assert len(detect_balls(bgr, {"r": spec})[0]) == 1


def test_rgb_range_and_json_roundtrip():
    bgr, _ = render([(PASTEL_PINK, (0.0, 0.0, 1000.0), 40.0)])
    colors = {"pink": spec_from_json({"rgb": [255, 192, 203], "tolerance": 30})}
    assert len(detect_balls(bgr, colors)[0]) == 1

    path = os.path.join(tempfile.mkdtemp(), "colors.json")
    save_colors(path, colors)
    assert len(detect_balls(bgr, load_colors(path))[0]) == 1


def test_locate_by_size_without_depth():
    truth = {"red": ((0, 0, 255), (0.0, 0.0, 1000.0), 33.0),
             "blue": ((255, 60, 0), (-300.0, 100.0, 1500.0), 50.0),
             "yellow": ((0, 230, 255), (250.0, -150.0, 800.0), 20.0)}
    bgr, _ = render(truth.values())
    balls, _ = detect_balls(bgr)
    for b in balls:
        _, (X, Y, Z), R = truth[b.color]
        locate_ball_by_size(b, 2 * R, K)
        err = np.linalg.norm(np.subtract(b.center_xyz, (X, Y, Z)))
        true_dist = np.linalg.norm((X, Y, Z))
        print(f"size-only {b.color:>6}: est={tuple(round(c, 1) for c in b.center_xyz)} "
              f"true=({X},{Y},{Z}) err={err:.1f}mm")
        assert err < 0.06 * true_dist, err

    # Calibration round trip: recover the focal length from a ball at a known distance.
    red = next(b for b in balls if b.color == "red")
    f = focal_from_known_distance(red, 66.0, 1000.0, K)
    assert abs(f - K.fx) < 0.05 * K.fx, f


def look_at(eye, target):
    """T_robot_cam for a camera at `eye` looking at `target` (robot frame, Z up)."""
    z = np.subtract(target, eye).astype(float)
    z /= np.linalg.norm(z)
    x = np.cross(z, (0, 0, 1.0))       # image right, parallel to the table
    x /= np.linalg.norm(x)
    y = np.cross(z, x)                 # image down
    return make_transform(np.column_stack([x, y, z]), eye)


def project(p_robot, T_cam_robot):
    p = T_cam_robot[:3, :3] @ np.asarray(p_robot, float) + T_cam_robot[:3, 3]
    return (K.fx * p[0] / p[2] + K.cx, K.fy * p[1] / p[2] + K.cy)


def render_marker(T_cam_robot, T_robot_marker, size_mm, marker_id=0):
    """Gray image with an ArUco marker (plus white quiet zone) lying on the table."""
    bgr = np.full((H, W, 3), 90, np.uint8)
    def quad(half):
        pts = [(-half, half), (half, half), (half, -half), (-half, -half)]  # TL, TR, BR, BL
        return np.float32([project((T_robot_marker @ [x, y, 0, 1])[:3], T_cam_robot)
                           for x, y in pts])
    cv2.fillConvexPoly(bgr, quad(size_mm * 0.75).astype(np.int32), (255, 255, 255))
    img = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
                                        marker_id, 400)
    src = np.float32([[0, 0], [399, 0], [399, 399], [0, 399]])
    warped = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, quad(size_mm / 2)),
                                 (W, H), flags=cv2.INTER_LINEAR, borderValue=255)
    inside = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(inside, quad(size_mm / 2).astype(np.int32), 255)
    bgr[inside > 0] = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)[inside > 0]
    return bgr


def test_robot_frame():
    # Camera 700 mm above the table, behind and above the robot, looking down at the workspace.
    T_robot_cam = look_at((-100.0, 50.0, 700.0), (350.0, 0.0, 0.0))
    T_cam_robot = invert_transform(T_robot_cam)
    T_robot_marker = marker_in_robot(300.0, 80.0, 0.0, yaw_deg=30.0)

    ws_path = os.path.join(tempfile.mkdtemp(), "workspace.json")
    ws = WorkspaceManager(ws_path, marker_size_mm=120.0, marker_pose=T_robot_marker, samples=3)
    marker_img = render_marker(T_cam_robot, T_robot_marker, 120.0)
    ws.handle_key(ord("k"))
    for _ in range(3):
        ws.update(marker_img, K)
    assert ws.frame is not None, "marker calibration did not finish"
    cam_err = np.linalg.norm(ws.frame.camera_position - T_robot_cam[:3, 3])
    rot_err = np.degrees(np.arccos(np.clip(
        (np.trace(ws.frame.T_robot_cam[:3, :3].T @ T_robot_cam[:3, :3]) - 1) / 2, -1, 1)))
    print(f"robot frame: camera position err={cam_err:.1f}mm, rotation err={rot_err:.2f}deg")
    assert cam_err < 15 and rot_err < 1.0
    frame = WorkspaceFrame.load(ws_path)

    # Balls resting on the table, positions in the robot frame.
    truth = {"red": ((0, 0, 255), (350.0, -60.0), 33.0),
             "blue": ((255, 60, 0), (250.0, 120.0), 33.0),
             "yellow": ((0, 230, 255), (480.0, 30.0), 33.0)}
    scene = [(c, tuple((T_cam_robot @ [x, y, R, 1])[:3]), R) for c, (x, y), R in truth.values()]
    bgr, depth = render(scene)
    balls, masks = detect_balls(bgr)
    assert {b.color for b in balls} == set(truth), [b.color for b in balls]

    for b in balls:
        _, (x, y), R = truth[b.color]
        # Webcam path: ray / table-plane intersection.
        assert locate_ball_on_table(b, 2 * R, K, frame)
        err_plane = np.linalg.norm(np.subtract(b.robot_xyz, (x, y, R)))
        # Depth-camera path: measured 3D point, transformed to the robot frame.
        assert locate_ball(b, depth, K, masks[b.color]) and locate_ball_in_robot(b, frame)
        err_depth = np.linalg.norm(np.subtract(b.robot_xyz, (x, y, R)))
        print(f"robot {b.color:>6}: true=({x},{y},{R}) est={tuple(round(c, 1) for c in b.robot_xyz)} "
              f"err plane={err_plane:.1f}mm depth={err_depth:.1f}mm")
        assert err_plane < 15 and err_depth < 25

    msg = balls_to_message(balls)
    assert msg["frame"] == "robot" and len(msg["balls"]) == 3, msg


if __name__ == "__main__":
    test_recovers_positions()
    test_rejects_non_round()
    test_ball_with_highlight_and_shadow()
    test_background_mode_finds_colorless_ball()
    test_background_cream_ball_with_shadow()
    test_background_black_ball_and_shadow()
    test_depth_rejects_flat_and_wrong_size()
    test_stable_filter_and_best_ball()
    test_click_learning_refines_and_rejects_dark()
    test_learn_unusual_color()
    test_learn_hue_wraparound_red()
    test_rgb_range_and_json_roundtrip()
    test_locate_by_size_without_depth()
    test_robot_frame()
    print("all tests passed")
