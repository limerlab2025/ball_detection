# Ball detection + 3D distance with Orbbec Gemini

Detects colored balls (red, orange, yellow, green, blue, purple) in the color stream,
reads the aligned depth, and reports each ball's 3D position relative to the camera.

## Coordinate frame

Origin **O(0,0,0)** = color camera optical center. Units: millimeters.

| Axis | Direction |
|------|-----------|
| +X | right in the image |
| +Y | down in the image |
| +Z | straight out of the lens |

`distance` = √(X² + Y² + Z²) to the **ball's center** (front-surface depth + ball radius).

## How it works

1. **Color segmentation**: HSV threshold per color → morphology → contours.
2. **Shape filter**: keep blobs with circularity `4πA/P² ≥ 0.7` that fill ≥ 60% of their enclosing circle.
3. **Depth**: depth is aligned to color; take the median depth of the inner 60% of the ball disk (ignores edge/background pixels and holes).
4. **Back-projection** (pinhole model, color intrinsics):
   `X = (u − cx)·Z / fx`, `Y = (v − cy)·Z / fy`
5. **Ball center**: real radius `R = r_px·Z / f`; push the surface point back by `R` along its ray.

## Install

```bash
pip install -r requirements.txt
```

Then install the Orbbec Python SDK from https://github.com/orbbec/pyorbbecsdk
(prebuilt wheels on its Releases page, or build from source). Gemini 2 / 330 series
need the v2 SDK. On macOS you may need to run with `sudo` for USB access.

## Run

```bash
python main.py            # window with overlays; prints positions twice a second
python main.py --no-gui   # print only
python test_ball_detector.py   # synthetic test, no camera needed
```

Keys: `q`/`Esc` quit · `m` show color masks · `s` save snapshot.

## Tuning

- Edit `COLOR_RANGES` in `ball_detector.py` if a ball isn't picked up (press `m` to see masks).
- Raise `--min-radius` to ignore small/far false positives.
- Depth sensors have a minimum range (~0.2–0.4 m depending on model); closer balls report "no depth".
