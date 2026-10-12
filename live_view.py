"""Publish the annotated camera window as a live video in the browser.

    python main_usb.py ... --stream          # then open http://localhost:8090
    sudo python main.py ... --stream 8091    # another port

Pages (local machine only, bound to 127.0.0.1):
    /              live view
    /stream        MJPEG video (works in any browser <img>)
    /snapshot.jpg  the latest frame
"""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>Ball detection live</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{margin:0;background:#111;color:#ccc;font:14px system-ui;text-align:center}
img{max-width:100%;height:auto;display:block;margin:0 auto}</style></head>
<body><img src="/stream" alt="live camera view"><p>Ball detection live view</p></body></html>"""


class LiveView:
    def __init__(self, port: int = 8090, max_fps: float = 10.0):
        self._jpeg = None
        self._cond = threading.Condition()
        self._min_dt = 1.0 / max_fps
        self._last = 0.0
        live = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):   # keep the terminal clean
                pass

            def do_GET(self):
                if self.path == "/":
                    self._send(200, "text/html; charset=utf-8", PAGE)
                elif self.path.startswith("/snapshot"):
                    jpeg = live._jpeg
                    if jpeg is None:
                        self._send(503, "text/plain", b"no frame yet")
                    else:
                        self._send(200, "image/jpeg", jpeg)
                elif self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    try:
                        while True:
                            with live._cond:
                                live._cond.wait(timeout=2.0)
                                jpeg = live._jpeg
                            if jpeg is None:
                                continue
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                             b"Content-Length: " + str(len(jpeg)).encode() +
                                             b"\r\n\r\n" + jpeg + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        pass   # browser closed the page
                else:
                    self._send(404, "text/plain", b"not found")

            def _send(self, code, ctype, body):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        print(f"Live view: http://localhost:{port}")

    def update(self, view: np.ndarray) -> None:
        """Call with each annotated frame; encodes at most max_fps frames per second."""
        now = time.time()
        if now - self._last < self._min_dt:
            return
        self._last = now
        ok, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._cond:
                self._jpeg = buf.tobytes()
                self._cond.notify_all()

    def close(self) -> None:
        self.server.shutdown()
