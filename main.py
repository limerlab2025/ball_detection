"""Detect colored balls with an Orbbec Gemini camera and report their 3D position.

Origin O(0,0,0) is the color camera's optical center (depth is aligned to color).
    +X right, +Y down, +Z forward (mm)

Keys:  q / ESC = quit,  m = toggle color-mask view,  s = save snapshot
"""
import argparse
import sys
import time

import cv2
import numpy as np

from ball_detector import Intrinsics, detect_balls, draw, locate_ball

try:
    from pyorbbecsdk import Config, OBFormat, OBSensorType, Pipeline
except ImportError:
    sys.exit("pyorbbecsdk not found. Install it first (see README.md).")

# SDK v2 aligns depth->color with a filter; SDK v1 uses Config.set_align_mode.
try:
    from pyorbbecsdk import AlignFilter, OBStreamType
    SDK_V2 = True
except ImportError:
    from pyorbbecsdk import OBAlignMode
    SDK_V2 = False


def frame_to_bgr(frame) -> np.ndarray | None:
    w, h = frame.get_width(), frame.get_height()
    fmt = frame.get_format()
    data = np.asanyarray(frame.get_data())
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return data.reshape(h, w, 3)
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == OBFormat.YUYV:
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUY2)
    if fmt == OBFormat.UYVY:
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
    if fmt == OBFormat.NV12:
        return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV12)
    if fmt == OBFormat.NV21:
        return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV21)
    if fmt == OBFormat.I420:
        return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_I420)
    print(f"Unsupported color format: {fmt}")
    return None


def depth_to_mm(depth_frame) -> np.ndarray:
    w, h = depth_frame.get_width(), depth_frame.get_height()
    raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(h, w)
    return raw.astype(np.float32) * depth_frame.get_depth_scale()


def start_pipeline():
    pipeline = Pipeline()
    config = Config()

    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    try:  # prefer RGB 640x480@30 so no decoding is needed; fall back to default
        color_profile = color_profiles.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
    except Exception:
        color_profile = color_profiles.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    config.enable_stream(depth_profiles.get_default_video_stream_profile())

    align = None
    if SDK_V2:
        align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    else:
        config.set_align_mode(OBAlignMode.SW_MODE)

    try:
        pipeline.enable_frame_sync()
    except Exception:
        pass
    pipeline.start(config)

    # After alignment the depth image lives in the color camera's frame,
    # so the color intrinsics are the ones to deproject with.
    rgb = pipeline.get_camera_param().rgb_intrinsic
    print(f"Color: {color_profile.get_width()}x{color_profile.get_height()} "
          f"{color_profile.get_format()} | fx={rgb.fx:.1f} fy={rgb.fy:.1f} "
          f"cx={rgb.cx:.1f} cy={rgb.cy:.1f}")
    return pipeline, align, rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-radius", type=float, default=8, help="min ball radius in pixels")
    parser.add_argument("--no-gui", action="store_true", help="print only, no window")
    args = parser.parse_args()

    pipeline, align, rgb = start_pipeline()
    show_masks = False
    last_print = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            if align is not None:
                frames = align.process(frames)
                if frames is None:
                    continue
                frames = frames.as_frame_set()

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue

            bgr = frame_to_bgr(color_frame)
            if bgr is None:
                continue
            depth_mm = depth_to_mm(depth_frame)

            # Intrinsics are for the profile's native size; rescale if the
            # aligned images differ (e.g. MJPG decoded at another resolution).
            sx = bgr.shape[1] / color_frame.get_width()
            sy = bgr.shape[0] / color_frame.get_height()
            k = Intrinsics(rgb.fx * sx, rgb.fy * sy, rgb.cx * sx, rgb.cy * sy)
            if depth_mm.shape != bgr.shape[:2]:
                depth_mm = cv2.resize(depth_mm, (bgr.shape[1], bgr.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)

            balls, masks = detect_balls(bgr, min_radius_px=args.min_radius)
            for b in balls:
                locate_ball(b, depth_mm, k, masks[b.color])

            now = time.time()
            if now - last_print > 0.5:
                last_print = now
                for b in balls:
                    if b.center_xyz:
                        x, y, z = b.center_xyz
                        print(f"{b.color:>6}: center=({x:7.1f}, {y:7.1f}, {z:7.1f}) mm  "
                              f"distance={b.distance_mm / 1000:.3f} m  "
                              f"diameter~{b.diameter_mm:.0f} mm")
                    else:
                        print(f"{b.color:>6}: pixel=({b.u:.0f},{b.v:.0f}) no valid depth")

            if args.no_gui:
                continue
            view = draw(bgr, balls)
            cv2.drawMarker(view, (int(k.cx), int(k.cy)), (255, 255, 255),
                           cv2.MARKER_CROSS, 12, 1)  # optical axis (X=Y=0)
            if show_masks:
                combined = np.zeros_like(masks["red"])
                for m in masks.values():
                    combined |= m
                view = np.hstack([view, cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)])
            cv2.imshow("Orbbec ball detection", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                show_masks = not show_masks
            if key == ord("s"):
                name = f"snapshot_{int(now)}.png"
                cv2.imwrite(name, view)
                print("saved", name)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
