"""Check which OpenCV camera index actually delivers frames, and save a picture from each.

    python probe_cameras.py

For each index it tries the camera's own resolution and then 640x480 (some cameras stop
streaming when asked for a size they don't support), and saves probe_<index>.jpg so you
can see which camera it is. Use the right one with:  python main_usb.py --camera <index>
"""
import sys
import time

import cv2

backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY


def grab(index, size=None, warmup_s=1.5, timeout_s=4.0):
    """Open, wait for auto-exposure to settle, return the last frame (or None)."""
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return "does not open"
    if size:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    t0, first, frame = time.time(), None, None
    while time.time() - t0 < timeout_s:
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
            first = first or time.time()
            if time.time() - first > warmup_s:
                break
        else:
            time.sleep(0.05)
    cap.release()
    return frame if frame is not None else f"opens, but NO frames in {timeout_s:.0f} s"


for index in range(5):
    native = grab(index)
    if isinstance(native, str):
        print(f"index {index}: {native}")
        if native == "does not open":
            continue
    else:
        name = f"probe_{index}.jpg"
        cv2.imwrite(name, native)
        print(f"index {index}: OK at {native.shape[1]}x{native.shape[0]}, "
              f"brightness {native.mean():.0f}/255 -> saved {name}")
    small = grab(index, (640, 480), warmup_s=0.3)
    print(f"          asking for 640x480: "
          + (small if isinstance(small, str) else f"OK, got {small.shape[1]}x{small.shape[0]}"))
