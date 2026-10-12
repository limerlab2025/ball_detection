# Pen / marker detection + 3D distance with Orbbec Gemini

> `main.py` now detects **pens, markers and other long colored objects** (see
> "Pens, markers and other long objects" below). The ball sections further down describe the
> earlier ball version of `main.py`; `ball_detector.py` is still here and tested, but no
> script runs it any more.

Detects colored balls (red, orange, yellow, green, blue, purple) in the color stream,
reads the aligned depth, and reports each ball's 3D position relative to the camera.

## Coordinate frame (robot arm)

Ball positions are reported in the **robot arm's frame**, not the camera's, so the
camera can be mounted anywhere (and re-mounted) without changing the numbers the arm gets.
Units: millimeters, right-handed.

| Axis | Direction |
|------|-----------|
| Origin | a point you pick on the robot base, at table height (e.g. the base's center axis) |
| +X | forward from the robot, along the table |
| +Y | to the robot's left, along the table |
| +Z | up from the table (table surface = Z 0) |

A ball of radius R lying on the table has Z ≈ R. The reported point is the **ball's center**.

### Calibrating the camera → robot transform (once per camera placement)

1. Make a marker: `python workspace.py` writes `marker_0.png`. Print it at 100% scale and
   measure the **black square's side** in mm (bigger is more accurate; 100–150 mm is good).
2. Lay it flat on the table. Measure where its **center** is in the robot frame, e.g.
   300 mm forward, 80 mm to the left. Simplest: point the marker's printed right side
   (+X, drawn red on screen) straight away from the robot so `--marker-yaw` stays 0.
3. Run with those numbers and press **`k`** with the marker in view:
   ```bash
   python main.py --marker-size 120 --marker-x 300 --marker-y 80
   python main_usb.py --ball-diameter 65 --marker-size 120 --marker-x 300 --marker-y 80
   ```
   It averages 30 frames, saves `workspace.json`, and draws the robot axes at the origin
   (red X, green Y, blue Z). Check they point where you expect, then remove the marker.

If the robot origin is not at table height, give the table height in the robot frame
with `--marker-z`. Redo `k` whenever the camera moves. With the USB camera, redo `k` after
recalibrating the focal length (`c`), since the transform depends on it.

Before calibration the scripts fall back to the **camera frame** (origin at the lens,
+X right, +Y down, +Z out of the lens) and label positions "camera".

### How the position is computed

- **Orbbec (depth):** 3D point from depth in the camera frame, then `p_robot = T_robot_cam · p_cam`.
- **USB webcam:** the ball's center lies on the viewing ray through the middle of its image
  and at height `table + R`. Intersecting the ray with that plane gives the position, far more
  accurate than estimating distance from apparent size. Requires the ball to be on the table.

### Sending positions to the arm

`--udp HOST:PORT` sends one JSON datagram per frame (see `robot_link.py` for a receiver):

```json
{"t": 1760000000.12, "frame": "robot",
 "balls": [{"color": "red", "x": 347.4, "y": -60.1, "z": 33.0, "diameter": 65.0}]}
```

Only move the arm when `"frame"` is `"robot"`.

## How it works

1. **Color segmentation**: HSV threshold per color → morphology → contours.
2. **Shape filter**: keep blobs with circularity `4πA/P² ≥ 0.7` that fill ≥ 60% of their enclosing circle.
3. **Depth**: depth is aligned to color; take the median depth of the inner 60% of the ball disk (ignores edge/background pixels and holes).
4. **Back-projection** (pinhole model, color intrinsics):
   `X = (u − cx)·Z / fx`, `Y = (v − cy)·Z / fy`
5. **Ball center**: real radius `R = r_px·Z / f`; push the surface point back by `R` along its ray.
6. **Robot frame**: transform the center with the calibrated camera → robot transform.

## Install

```bash
pip install -r requirements.txt
```

Then install the Orbbec Python SDK (v2, needed by Gemini 2 / 330; imports as `pyorbbecsdk`):

```bash
python -m pip install pyorbbecsdk2
```

If there is no wheel for your Python/OS, see https://github.com/orbbec/pyorbbecsdk.
On macOS the Orbbec usually needs `sudo` (otherwise: `uvc_open failed ... Return Code: -3`).
Use the full path of your Python so sudo doesn't pick the system one:

```bash
sudo $(which python) main.py
```

## Run

```bash
python main.py            # window with overlays; prints positions twice a second
python main.py --no-gui   # print only
python test_ball_detector.py   # synthetic test, no camera needed
```

Mouse: **left-click a ball** to learn its color.
Keys: `q`/`Esc` quit · `m` show color masks · `s` save snapshot · `k` calibrate robot frame ·
`w` save colors to `colors.json` · `u` undo last learned color · `r` reset to defaults.

## Without the Orbbec: regular USB webcam (`main_usb.py`)

A normal webcam has no depth sensor, so distance comes from the ball's **known real size**:
a ball looks smaller the farther away it is. Once the robot frame is calibrated, balls on the
table are located by ray–table intersection instead (see above), which is much more accurate.

```bash
python main_usb.py --ball-diameter 65         # set this to YOUR ball's diameter in mm
```

- The camera is found by name (`--camera-name "USB Camera"`, via ffmpeg on macOS); force one with `--camera 0`.
- On first run macOS asks for camera permission for your terminal app (System Settings › Privacy & Security › Camera).
- **Calibrate once for accurate distance:** hold one ball so its **center** is exactly 50 cm from the lens
  (or pass `--calib-distance <mm>`) and press `c`. The focal length is saved to `camera_calib.json`.
  Until then it guesses from a 60° field of view (`--hfov`), which can be off by 20% or more.
- Accuracy is worse than with depth, especially for far/small balls: at 640 px wide a 1-pixel error
  in a 10 px radius is a 10% distance error. Use `--width 1280 --height 720` for better range.
- All balls are assumed to be the same size (`--ball-diameter`).

## Detecting any ball color

**1. Click to learn (easiest).** Put the ball in view, click its center, check it is detected,
then press `w`. The color is saved to `colors.json` and loaded automatically next time.
It is named after the closest common color ("light pink", "teal", "brown"…); rename it in the file if you like.
For best results click the ball where it is lit normally, not on a shiny highlight.

**2. Edit `colors.json` by hand.** When this file exists it replaces the built-in defaults.
Each entry can use any of these forms:

```json
{
  "red":    {"hsv": [[[0,120,70],[8,255,255]], [[170,120,70],[179,255,255]]]},
  "teal":   {"rgb_min": [0,100,100], "rgb_max": [80,200,200]},
  "pink":   {"rgb": [255,150,190], "tolerance": 40}
}
```

- `hsv`: list of `[low, high]` ranges, OpenCV units (H 0–179, S 0–255, V 0–255). Recommended.
- `rgb_min` / `rgb_max`: a pixel matches if each of R, G, B lies between them.
- `rgb` + `tolerance`: shortcut for `rgb ± tolerance` on each channel.

HSV handles changing light better than RGB: in a shadow R, G and B all drop, but the hue (H) stays almost the same.

Every detection also reports the ball's measured `rgb` and the closest `color_name`.

## Background mode: any ball color (recommended for the robot table)

Color thresholds can't separate a white / cream / gray ball from a gray table or wall.
With the camera fixed over the table, use the empty table instead:

1. Start with `--background` (captures the empty scene 3 s after start, keep the ball out of
   view), or remove all balls from view and press **`b`**. It averages 15 frames.
2. Put balls down. Any round object that wasn't there before is detected (color "ball").

Overall brightness / white-balance changes from the webcam's auto-exposure are compensated.
If more than 30% of the image differs (camera moved, room light switched), detection pauses
and asks you to press `b` again. Blobs bigger than 1/4 of the image height (a hand right in
front of the lens) are ignored; change with `--max-radius`. `n` returns to color mode.
Hands and arms are detected as changes too, but are rejected by the round-shape test
(add `--max-balls 1` to keep only the most ball-like detection: roundest and best filled).
A ball is reported only after staying in place for 3 frames (`--min-frames`), which removes
flickering noise. If balls are missed, lower
`--bg-threshold` (default 25); if shadows show up, raise it.

Shadows are recognized as "somewhat darker, same color" and ignored; black balls are much
darker than a shadow, so they are still found. (A mid-gray ball on a light table can look
like a shadow and be missed.)

**Only white / cream balls?** `--bright-ball` counts only pixels that got brighter, which
also ignores hands and arms, but it ignores black / dark balls too.

## Pens, markers and other long objects (`main.py`)

```bash
sudo $(which python) main.py                      # pens, markers, any long colored thing
sudo $(which python) main.py --kinds pen,marker   # only report pens and markers
sudo $(which python) main.py --no-gui --udp 192.168.1.50:5005   # headless, send to the arm
python test_object_detector.py                              # synthetic test, no camera needed
```

The color detection is the same as for balls (built-in colors, `colors.json`, **left-click the
object** to learn its color, `w` to save). Instead of round blobs it keeps **long, thin** ones
(length/width >= `--min-aspect`, default 2.5) and fits a rotated box to each. With depth it
then measures:

- **distance** and **center** (on the object's axis, not its top surface), camera or robot frame
- **length and width in mm**, which decide the kind (`OBJECT_SIZES` in `object_detector.py`):
  pen = 6-13 mm wide, 90-180 mm long; marker = 13-30 mm wide, 80-170 mm long; else `object`
- **yaw**: the object's direction on the table in the robot frame (-90..90 deg from +X),
  once the robot frame is calibrated (`k`). This is the angle to turn the gripper to.

Measure your own pens and markers (the window shows `L=... W=...`) and adjust the ranges,
or put them in a file and pass `--object-sizes my_sizes.json`:

```json
{"pen": {"width": [6, 13], "length": [90, 180]}, "highlighter": {"width": [15, 25], "length": [100, 140]}}
```

UDP and CSV output include `kind`, `length`, `yaw` (`diameter` is the width).

Tips: a pen is only ~10 px wide at 50 cm, so click its colored body exactly (the click samples
a 3 px circle). Pens with a differently colored cap or label show up as separate
pieces, so learn the body color. Black, white and clear pens have no color it can use.
`d` shows rejected blobs and why
(`not long`, `fill=`, `short`, `no depth`).

## Live view in a browser

Add `--stream` to also publish the annotated window at http://localhost:8090 (only this
computer can open it). Useful for watching from another window or letting Claude see it.
Use `--stream 8091` for a different port. Works with `main.py --no-gui` too.

## Recording to CSV

Add `--csv` to save every detection (one row per ball per frame) to
`recordings/balls_<date>_<time>.csv`, or `--csv myrun.csv` to choose the file
(new rows are appended if it exists). "REC" shows in the window while recording.

```bash
python main_usb.py --ball-diameter 65 --csv
```

Columns: time, frame number, color, measured RGB, pixel position and radius, camera-frame
X/Y/Z, distance, diameter, robot-frame X/Y/Z (empty until calibrated), and `method`
(`depth`, `table_plane`, `size` or `none`). See `data_log.py` for details.

## Ball not detected?

Press **`d`** (or start with `--debug`). Every colored blob that was *rejected* gets a
gray circle with its scores, e.g. `yellow: circ=0.62 fill=0.55`. After ~2 s with no ball
the terminal also prints the closest candidates.

- **No gray circle on the ball:** its color isn't in any range. Click the ball to learn
  its color (then `w` to save). Press `m` to see which pixels match.
- **Gray circle, low `circ`/`fill`:** the color only covers part of the ball (glare,
  shadow, or a ball between two colors, like a yellow-green tennis ball split into
  "yellow" and "green"). Click the ball to learn one color covering all of it, or lower
  `--min-circularity` / `--min-fill` (e.g. 0.6 / 0.5).
- **White / cream / beige / gray balls:** use background mode (`b`, see above). Clicking
  can't help when the ball has almost no color.
- **Clicking:** click the ball's evenly lit middle (not the glare or shadow). Clicking the same
  ball again refines that color instead of adding a new one; clicks on near-black pixels are
  refused. Too many learned colors cause false detections everywhere: press `r` then `w` to reset.
- **`small r=..`:** ball too far / too small; move closer or lower `--min-radius`.
- **Wrong camera:** pass `--camera 1`, `--camera 2`, ... (index order can differ from ffmpeg's list).

## Tuning

- If a ball isn't picked up, press `m` to see which pixels match and widen its range.
- Raise `--min-radius` to ignore small/far false positives.
- Depth sensors have a minimum range (~0.2–0.4 m depending on model); closer balls report "no depth".
