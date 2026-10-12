"""Detect pens, markers and other long colored objects with an Orbbec Gemini camera,
and report each one's color, kind, distance, size and position.

For every object it finds:
    color     which color spec matched (built-in, colors.json, or learned by clicking)
    kind      "pen" / "marker" by its measured size (object_detector.OBJECT_SIZES), else "object"
    distance  camera -> object center, from the aligned depth image
    length, width in mm, and the 3D center (camera frame, or robot frame after pressing k)

Two ways to find objects:
    color mode (default)  blobs of a known color. Fine for pens/markers colored all over.
    background mode (b)   first capture the EMPTY table (color and depth), then anything new
                          or sticking up above it is an object, whatever its color (white-bodied
                          markers, gray pens, ...). Its color is read from its colored part
                          (cap, tip, label).

Mouse: left-click an object's colored body to learn that color
Keys:  q / ESC = quit     d = debug view (rejected blobs and why)   m = show masks
       b = capture the empty table -> background mode    n = back to color mode
       s = save snapshot  w = write colors.json  u = undo learned color  r = reset colors
       k = calibrate robot frame from the ArUco marker on the table
"""
import argparse
import sys
import time

import cv2
import numpy as np

from ball_detector import Intrinsics, Rejected, StableFilter, draw_rejected
from color_ui import DEFAULT_COLORS_PATH, ColorManager
from data_log import CsvLogger, default_csv_path
from object_detector import (OBJECT_SIZES, classify_object, detect_objects, draw_objects,
                             load_object_sizes, locate_object, locate_object_in_robot,
                             TableBackground, label)
from robot_link import UdpSender
from workspace import add_workspace_args, manager_from_args

WINDOW = "Orbbec object detection"

try:
    from pyorbbecsdk import Config, OBFormat, OBSensorType, Pipeline
except ImportError:
    sys.exit("pyorbbecsdk not found: python -m pip install pyorbbecsdk2 (see README.md)")

# SDK v2 aligns depth to color with a filter; SDK v1 uses Config.set_align_mode.
try:
    from pyorbbecsdk import AlignFilter, OBStreamType
    SDK_V2 = True
except ImportError:
    from pyorbbecsdk import OBAlignMode
    SDK_V2 = False


# ---------- camera ----------

def color_to_bgr(frame) -> np.ndarray | None:
    w, h = frame.get_width(), frame.get_height()
    fmt = frame.get_format()
    data = np.asanyarray(frame.get_data())
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return data.reshape(h, w, 3)
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    yuv = {OBFormat.YUYV: (2, cv2.COLOR_YUV2BGR_YUY2), OBFormat.UYVY: (2, cv2.COLOR_YUV2BGR_UYVY)}
    if fmt in yuv:
        ch, code = yuv[fmt]
        return cv2.cvtColor(data.reshape(h, w, ch), code)
    planar = {OBFormat.NV12: cv2.COLOR_YUV2BGR_NV12, OBFormat.NV21: cv2.COLOR_YUV2BGR_NV21,
              OBFormat.I420: cv2.COLOR_YUV2BGR_I420}
    if fmt in planar:
        return cv2.cvtColor(data.reshape(h * 3 // 2, w), planar[fmt])
    print(f"Unsupported color format: {fmt}")
    return None


def depth_to_mm(frame) -> np.ndarray:
    raw = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(frame.get_height(), frame.get_width())
    return raw.astype(np.float32) * frame.get_depth_scale()


class GeminiCamera:
    """Color + depth aligned to the color image, and the color intrinsics."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        try:
            self.pipeline = Pipeline()
        except RuntimeError as e:
            if "uvc_open" in str(e):
                sys.exit(f"{e}\nThe camera was found but could not be opened. On macOS run with sudo:\n"
                         f"    sudo {sys.executable} {' '.join(sys.argv)}\n"
                         "and close other apps using the camera.")
            if "No device" in str(e):
                sys.exit(f"{e}\nNo Orbbec camera found: check the USB cable (use a USB 3 port).")
            raise
        config = Config()
        colors = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:   # raw RGB needs no decoding; fall back to the camera's default
            profile = colors.get_video_stream_profile(width, height, OBFormat.RGB, fps)
        except Exception:
            profile = colors.get_default_video_stream_profile()
        config.enable_stream(profile)
        depths = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        config.enable_stream(depths.get_default_video_stream_profile())

        self.align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM) if SDK_V2 else None
        if not SDK_V2:
            config.set_align_mode(OBAlignMode.SW_MODE)
        try:
            self.pipeline.enable_frame_sync()
        except Exception:
            pass
        self.pipeline.start(config)
        # Depth is aligned to color, so the color intrinsics apply to both images.
        self.rgb = self.pipeline.get_camera_param().rgb_intrinsic
        print(f"Color {profile.get_width()}x{profile.get_height()} {profile.get_format()}  "
              f"fx={self.rgb.fx:.1f} fy={self.rgb.fy:.1f} cx={self.rgb.cx:.1f} cy={self.rgb.cy:.1f}")

    def read(self):
        """(bgr, depth_mm, intrinsics), or None if no complete frame arrived."""
        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            return None
        if self.align is not None:
            frames = self.align.process(frames)
            if frames is None:
                return None
            frames = frames.as_frame_set()
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if color is None or depth is None:
            return None
        bgr = color_to_bgr(color)
        if bgr is None:
            return None
        depth_mm = depth_to_mm(depth)
        # Intrinsics belong to the profile's size; rescale if decoding changed it.
        sx, sy = bgr.shape[1] / color.get_width(), bgr.shape[0] / color.get_height()
        k = Intrinsics(self.rgb.fx * sx, self.rgb.fy * sy, self.rgb.cx * sx, self.rgb.cy * sy)
        if depth_mm.shape != bgr.shape[:2]:
            depth_mm = cv2.resize(depth_mm, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        return bgr, depth_mm, k

    def stop(self) -> None:
        self.pipeline.stop()


# ---------- main loop ----------

def put_text(view, text, pos, color=(255, 255, 255), scale=0.5):
    label(view, text, pos, color, scale)


def describe(o) -> str:
    frame, (x, y, z) = ("robot", o.robot_xyz) if o.robot_xyz else ("camera", o.center_xyz)
    yaw = f"  yaw={o.robot_yaw_deg:.0f} deg" if o.robot_yaw_deg is not None else ""
    return (f"{o.kind:>7} {o.color:>10} RGB{o.rgb}  distance={o.distance_mm / 1000:.3f} m  "
            f"{frame}=({x:.0f}, {y:.0f}, {z:.0f}) mm  L={o.length_mm:.0f} W={o.diameter_mm:.0f} mm{yaw}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kinds", default=None,
                   help="comma-separated kinds to report, e.g. 'pen,marker' (default: all)")
    p.add_argument("--object-sizes", default=None, metavar="FILE",
                   help="JSON with real size ranges per kind (default: OBJECT_SIZES in object_detector.py)")
    p.add_argument("--colors", default=DEFAULT_COLORS_PATH,
                   help="color definitions (used instead of the defaults if the file exists)")
    p.add_argument("--resolution", default="640x480", metavar="WxH",
                   help="color resolution, e.g. 1280x720: more pixels on far objects, so they are "
                        "found farther away (slower; falls back to the camera default if unsupported)")
    p.add_argument("--min-length", type=float, default=15,
                   help="min object length in the image, px (lower = finds farther / smaller objects, "
                        "but also more noise)")
    p.add_argument("--min-width", type=float, default=2, help="min object width in the image, px")
    p.add_argument("--min-aspect", type=float, default=2.5,
                   help="min length/width in the image (pen ~10+, ball 1)")
    p.add_argument("--min-fill", type=float, default=0.55,
                   help="how much of its rotated box a blob must fill (0-1)")
    p.add_argument("--min-frames", type=int, default=3,
                   help="report an object only after it stayed in place this many frames (1 = off)")
    p.add_argument("--background", action="store_true",
                   help="start in background mode: the EMPTY table is captured 3 s after start")
    p.add_argument("--bg-threshold", type=float, default=25.0,
                   help="background mode: how different from the empty table a pixel must be "
                        "(lower = more sensitive)")
    p.add_argument("--min-height", type=float, default=6.0,
                   help="background mode: anything this many mm above the empty table is an "
                        "object, whatever its color (raise it if the table shows depth noise)")
    p.add_argument("--whole-image", action="store_true",
                   help="background mode: search the whole picture, not just the table plane")
    p.add_argument("--csv", nargs="?", const="", default=None, metavar="FILE",
                   help="record detections to CSV (no FILE: recordings/<date>_<time>.csv)")
    p.add_argument("--no-gui", action="store_true", help="print only, no window")
    add_workspace_args(p)
    args = p.parse_args()

    sizes = load_object_sizes(args.object_sizes) if args.object_sizes else OBJECT_SIZES
    kinds = set(args.kinds.split(",")) if args.kinds else None
    print("Kinds by size (width, length mm): " +
          ", ".join(f"{n} {r['width']}, {r['length']}" for n, r in sizes.items()))

    colors = ColorManager(args.colors)
    colors.sample_radius = 3   # a pen is ~10 px wide: sample its body, not the table
    if not args.no_gui:
        colors.attach(WINDOW)
    ws = manager_from_args(args)
    if args.no_gui and ws.frame is None:
        ws.handle_key(ord("k"))   # no keyboard: calibrate right away if the marker is visible
    udp = UdpSender(args.udp) if args.udp else None
    log = CsvLogger(args.csv or default_csv_path()) if args.csv is not None else None
    stable = StableFilter(args.min_frames)
    debug = show_masks = False
    last_print = last_warning = 0.0
    bg = TableBackground(min_height_mm=args.min_height, table_only=not args.whole_image)
    bg_countdown = time.time() + 3 if args.background else None
    if args.background:
        print("Background mode: clear the table and keep your hands out of view, "
              "capturing the empty table in 3 s...")

    width, height = (int(n) for n in args.resolution.lower().split("x"))
    cam = GeminiCamera(width, height)
    try:
        while True:
            got = cam.read()
            if got is None:
                continue
            bgr, depth_mm, k = got
            colors.latest_bgr = bgr
            ws.update(bgr, k)
            if bg_countdown is not None and time.time() >= bg_countdown:
                bg_countdown = None
                bg.start()
            if bg.update(bgr, depth_mm, k):
                print("Empty table captured: now anything new is an object. "
                      "Press b to recapture (e.g. after the light changes), n for color mode.")

            # 1. color + shape: long, thin blobs of a known color
            rejected = []
            found, masks = detect_objects(bgr, colors.colors, min_length_px=args.min_length,
                                          min_width_px=args.min_width,
                                          min_aspect=args.min_aspect, min_fill_ratio=args.min_fill,
                                          rejected=rejected, background=bg,
                                          bg_threshold=args.bg_threshold, depth_mm=depth_mm)
            if bg.lab is not None and bg.changed_fraction > 0.3:
                found = []   # most of the picture changed: the scene, not an object
                if time.time() - last_warning > 3:
                    last_warning = time.time()
                    print(f"{bg.changed_fraction:.0%} of the image differs from the empty table: "
                          "the camera moved or the light changed. Press b to recapture.")
            # 2. depth: distance, real size -> kind
            located = []
            for o in found:
                if not locate_object(o, depth_mm, k):
                    rejected.append(Rejected(o.color, o.u, o.v, o.radius_px, "no depth"))
                    continue
                classify_object(o, sizes)
                if kinds and o.kind not in kinds:
                    rejected.append(Rejected(o.color, o.u, o.v, o.radius_px,
                                             f"{o.kind} L={o.length_mm:.0f} W={o.diameter_mm:.0f}mm"))
                    continue
                located.append(o)
            # 3. keep only objects that stay put, then move them to the robot frame
            objects = stable.update(located)
            for o in objects:
                if ws.frame is not None:
                    locate_object_in_robot(o, ws.frame)

            if udp:
                udp.send(objects)
            if log:
                log.log(objects, ["depth"] * len(objects))
            now = time.time()
            if objects and now - last_print > 0.5:
                last_print = now
                for o in objects:
                    print(describe(o))

            if args.no_gui:
                continue
            view = draw_objects(bgr, objects, colors.colors)
            ws.draw(view, k)
            colors.draw(view)
            if debug:
                draw_rejected(view, rejected, footer=f"debug: need length/width>={args.min_aspect:.1f} "
                                                     f"fill>={args.min_fill:.2f}")
            if bg_countdown is not None:
                status = f"CLEAR THE TABLE - capturing it in {bg_countdown - time.time():.0f} s"
            elif bg.capturing:
                status = "capturing empty table..."
            elif bg.lab is not None:
                status = f"background mode: {len(objects)} object(s)  (n = color mode)"
            else:
                status = (f"color mode: {len(objects)} object(s)  "
                          "(white/gray objects? clear the table, press b)")
            put_text(view, status, (10, 50), (0, 255, 255) if bg_countdown else (255, 255, 255))
            if log:
                put_text(view, "REC", (view.shape[1] - 55, 25), (0, 0, 255), 0.7)
            if show_masks:
                combined = np.zeros(bgr.shape[:2], np.uint8)
                for m in masks.values():
                    combined |= m
                view = np.hstack([view, cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)])
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                debug = not debug
            if key == ord("m"):
                show_masks = not show_masks
            if key == ord("b"):
                bg_countdown = None
                bg.start()
                print("Capturing the empty table: keep objects and hands OUT of view...")
            if key == ord("n") and bg.lab is not None:
                bg.clear()
                print("Back to color mode")
            if key == ord("s"):
                name = f"snapshot_{int(now)}.png"
                cv2.imwrite(name, view)
                print("saved", name)
            colors.handle_key(key)
            ws.handle_key(key)
    except KeyboardInterrupt:
        pass
    finally:
        if log:
            log.close()
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
