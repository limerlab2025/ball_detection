"""Shared color management for the camera scripts: load/save colors.json,
click-to-learn a ball color, and the keyboard shortcuts.

Mouse: left-click a ball to learn its color
Keys:  w = write colors.json   u = undo last learned color   r = reset to default colors
"""
import copy
import os
import time

import cv2
import numpy as np

from ball_detector import (DEFAULT_COLORS, learn_color, load_colors, name_color,
                           sample_disk, save_colors)

MIN_LEARN_BRIGHTNESS = 40   # median V (0-255) below this has no reliable color

DEFAULT_COLORS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colors.json")


class ColorManager:
    def __init__(self, path: str = DEFAULT_COLORS_PATH):
        self.path = path
        if os.path.exists(path):
            self.colors = load_colors(path)
            print(f"Loaded {len(self.colors)} colors from {path}: {', '.join(self.colors)}")
        else:
            self.colors = copy.deepcopy(DEFAULT_COLORS)
            print(f"Using default colors ({', '.join(self.colors)}). Click an object to add its color.")
        self.learned = []        # names added by clicking, for undo
        self.samples = {}        # learned name -> clicked pixels, so repeat clicks refine it
        self.last_click = None   # (x, y, time, ok) shown on screen so you can see where it landed
        self.latest_bgr = None   # set every frame by the camera loop
        self.sample_radius = 8   # px around the click to learn from (smaller for thin objects)

    def attach(self, window: str) -> None:
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        bgr = self.latest_bgr
        if event != cv2.EVENT_LBUTTONDOWN or bgr is None or x >= bgr.shape[1]:
            return
        px = sample_disk(bgr, x, y, radius=self.sample_radius)
        b, g, r = (int(c) for c in np.median(px, axis=0))
        brightness = max(r, g, b)
        print(f"Clicked ({x},{y}): RGB({r},{g},{b})")
        self.last_click = (x, y, time.time(), brightness >= MIN_LEARN_BRIGHTNESS)
        if brightness < MIN_LEARN_BRIGHTNESS:
            print("  Too dark to learn a color from (shadow/black). Click a well-lit part of "
                  "the ball, or add light.")
            return

        # Clicking the same ball again refines its color instead of adding a duplicate.
        hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
        for name in reversed(self.learned):
            spec = self.colors.get(name)
            if spec is None:
                continue
            hit = np.zeros(len(px), bool)
            for lo, hi in spec.hsv_ranges:
                hit |= cv2.inRange(hsv, np.array(lo), np.array(hi)).ravel() > 0
            match = hit.mean()
            if match >= 0.5:
                self.samples[name] = np.vstack([self.samples[name], px])
                self.colors[name] = learn_color(self.samples[name])
                print(f"  Refined '{name}' ({len(self.samples[name])} samples): "
                      f"HSV ranges {self.colors[name].hsv_ranges}  (press w to save)")
                return

        spec = learn_color(px)
        if spec.hsv_ranges and spec.hsv_ranges[0][0][0] == 0 and spec.hsv_ranges[0][1][0] == 179:
            print("  Warning: this is white/gray/black, which also matches walls, the table and "
                  "shadows. For such objects use background mode (b) instead; u = undo.")
        base = name_color((r, g, b))
        name, i = base, 2
        while name in self.colors:
            name, i = f"{base} {i}", i + 1
        self.colors[name] = spec
        self.samples[name] = px
        self.learned.append(name)
        print(f"  Learned '{name}': HSV ranges {spec.hsv_ranges}  (press w to save)")

    def draw(self, view) -> None:
        """Mark the last click for 2 s: the sampled circle, green if learned, red if refused.
        If this mark is not where you clicked, the window is mapping clicks wrongly."""
        if self.last_click is None or time.time() - self.last_click[2] > 2.0:
            return
        x, y, _, ok = self.last_click
        c = (0, 255, 0) if ok else (0, 0, 255)
        cv2.circle(view, (x, y), self.sample_radius, c, 2)
        cv2.drawMarker(view, (x, y), c, cv2.MARKER_CROSS, 30, 1)

    def handle_key(self, key: int) -> None:
        if key == ord("w"):
            save_colors(self.path, self.colors)
            print(f"Saved {len(self.colors)} colors to {self.path}")
        elif key == ord("u") and self.learned:
            name = self.learned.pop()
            self.colors.pop(name, None)
            self.samples.pop(name, None)
            print("Removed", name)
        elif key == ord("r"):
            self.colors = copy.deepcopy(DEFAULT_COLORS)
            self.learned.clear()
            self.samples.clear()
            print("Reset to default colors (press w to save)")
