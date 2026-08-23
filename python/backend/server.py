"""
FreshIQ local server — runs ON the Arduino UNO Q, serves the dashboard.

    python3 backend/server.py --port 8000

Then open http://<board-ip>:8000 from any device on the same network.

WHY STDLIB ONLY
---------------
No Flask, no FastAPI. Every dependency is one more thing to install on an ARM
board and one more thing that can fail at 2am. http.server is in Python itself.
It is not a production web server, and for one rig on a local network it does not
need to be.

We also deliberately avoid the `cgi` module, which the original service.py used:
it was REMOVED in Python 3.13, so anything importing it is a time bomb.

WORKS WITHOUT A CAMERA
----------------------
If no camera is found the server starts in DEMO mode and inspects the bundled
sample images instead. That means you can develop the dashboard on your laptop
and only plug in hardware at the end.

ENDPOINTS
---------
    GET  /                     the dashboard
    GET  /api/health           what is loaded, is the camera alive
    GET  /api/snapshot         current camera frame as JPEG
    POST /api/inspect          capture N views -> classify -> save -> return
    GET  /api/history?limit=N  past inspections, newest first
    GET  /api/stats            totals and spoiled rate
    GET  /api/export.csv       full history as CSV
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from ai import inspect, get_config, __version__ as ai_version   # noqa: E402
from ai import model as model_module                            # noqa: E402

# The scan/hardware layer is optional at import time. If config/freshiq.json or
# hardware/ is missing the server must still start and serve the dashboard in
# single-shot mode - losing tray control should not cost you the whole UI.
try:
    from config import load_settings                            # noqa: E402
    from hardware import build_lighting, build_motor, MotorError  # noqa: E402
    from backend import scan as scan_module                     # noqa: E402
    SCAN_AVAILABLE = True
    SCAN_IMPORT_ERROR = None
except Exception as _exc:                                       # noqa: BLE001
    SCAN_AVAILABLE = False
    SCAN_IMPORT_ERROR = "{}: {}".format(type(_exc).__name__, _exc)

DASHBOARD = os.path.join(os.path.dirname(__file__), "dashboard.html")
DATA_DIR = os.path.join(REPO, "data")
HISTORY_FILE = os.path.join(DATA_DIR, "inspections.jsonl")

DISCLAIMER = ("Visual estimate of appearance only. NOT a food-safety "
              "measurement. Do not use to decide whether food is safe to eat.")


# --------------------------------------------------------------------------- #
# History — append-only JSON Lines
# --------------------------------------------------------------------------- #
# One JSON object per line. Chosen over SQLite because it survives a power cut
# mid-write (you lose at most the last line, not the file), it is readable with
# `tail`, and it needs no schema migration when the result shape changes.
class History:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def append(self, record: dict) -> None:
        with self.lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")

    def read(self, limit: int = 50) -> list:
        if not os.path.exists(self.path):
            return []
        with self.lock:
            with open(self.path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
        out = []
        for line in reversed(lines):            # newest first
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue                        # skip a torn final line
            if len(out) >= limit:
                break
        return out

    def all(self) -> list:
        return self.read(limit=10 ** 9)


# --------------------------------------------------------------------------- #
# Camera — one shared instance, guarded by a lock
# --------------------------------------------------------------------------- #
class CameraSource:
    """
    Wraps the USB camera. Falls back to bundled sample images when there is no
    camera, so the dashboard is fully usable before any hardware exists.

    A lock is essential: the dashboard polls /api/snapshot on a timer while the
    user may click Inspect. Two threads calling cap.read() at once returns torn
    frames or crashes the driver.
    """

    def __init__(self, index: int = 0, rotate_cmd: str = None, rotate_serial: str = None):
        self.index = index
        self.rotate_cmd = rotate_cmd
        self.rotate_serial = rotate_serial
        self.lock = threading.Lock()
        self.cam = None
        self.demo = False
        self.error = None
        self._serial = None
        self._open()

    def _open(self):
        try:
            from camera.capture import Camera, CameraError
            self.cam = Camera(index=self.index).open()
            self.demo = False
            print(f"[camera] opened index {self.index}")
            self._start_reader()
        except Exception as exc:
            self.cam = None
            self.demo = True
            self.error = f"{exc.__class__.__name__}: {exc}"
            print(f"[camera] not available ({self.error}) -> DEMO MODE, using sample_data/")

        if self.rotate_serial:
            # This is a SECOND, legacy rotation path that predates the scan
            # controller. It bypasses hardware/motor.py entirely, so if the real
            # driver is also active the two fight over the same tray and the
            # software position stops matching reality. Warn rather than
            # silently double-rotate.
            print("[tray] WARNING: --rotate-serial is the legacy external-Arduino "
                  "path and bypasses the configured motor driver. On an UNO Q, "
                  "drop this flag and let config/freshiq.json drive the tray.")
            try:
                import serial  # type: ignore
                self._serial = serial.Serial(self.rotate_serial, 115200, timeout=1)
                print(f"[tray] serial trigger on {self.rotate_serial}")
            except Exception as exc:
                print(f"[tray] serial unavailable ({exc}); rotation will be skipped")
                self._serial = None

    def _demo_frames(self, n: int):
        import cv2
        names = ["tomato_front.jpg", "tomato_back.jpg", "tomato_fresh.jpg", "tomato_spoiled.jpg"]
        frames = []
        for i in range(n):
            p = os.path.join(REPO, "sample_data", names[i % len(names)])
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
        return frames

    def rotate(self, i: int) -> None:
        """Advance the tray one position. Both mechanisms are optional."""
        if self._serial is not None:
            try:
                self._serial.write(b"R\n")      # the MCU sketch listens for this
                self._serial.flush()
            except Exception as exc:
                print(f"[tray] serial write failed: {exc}")
        elif self.rotate_cmd:
            os.system(self.rotate_cmd)

    # ---- continuous reader -------------------------------------------------
    def _start_reader(self):
        """
        Drain the camera continuously in the background, keeping only the newest
        frame.

        THIS IS THE FIX FOR "LAGGY" PREVIEW.

        OpenCV's VideoCapture keeps a driver-side buffer of frames. If you call
        read() once every 1.2 seconds, you do NOT get the current view - you get
        the OLDEST frame still sitting in that buffer. The preview then trails
        reality by however many frames are queued, which looks exactly like
        network latency even though everything is local.

        Reading flat out in a background thread keeps the buffer empty, so the
        stored frame is always the newest one. Serving the preview then costs
        nothing: no camera access, no lock contention with an inspection.
        """
        self._latest = None
        self._latest_at = 0.0
        self._frame_lock = threading.Lock()
        self._stop = threading.Event()
        self.fps = 0.0

        def loop():
            n, t0 = 0, time.perf_counter()
            failures = 0
            while not self._stop.is_set():
                try:
                    frame = self.cam.capture()
                except Exception as exc:
                    self.error = str(exc)
                    failures += 1
                    # Say this ONCE, loudly. A camera that opens but never
                    # delivers a frame is the worst case to diagnose: the log
                    # says "opened index 0", /api/stream returns 200 and then
                    # streams nothing, and the preview just sits blank. Without
                    # this line there is nothing anywhere that names the fault.
                    if failures == 1:
                        print("[camera] opened, but read() is failing: {}".format(exc),
                              flush=True)
                    time.sleep(0.5)
                    continue
                if failures:
                    print("[camera] reading again after {} failed read(s)".format(failures),
                          flush=True)
                    failures = 0
                with self._frame_lock:
                    self._latest = frame
                    self._latest_at = time.time()
                n += 1
                if n % 30 == 0:
                    now = time.perf_counter()
                    self.fps = 30.0 / max(now - t0, 1e-6)
                    t0 = now

        self._reader = threading.Thread(target=loop, daemon=True, name="camera-reader")
        self._reader.start()

    def stop(self):
        if getattr(self, "_stop", None):
            self._stop.set()

    def snapshot(self):
        """The newest frame. Instant - no camera access, no waiting."""
        if self.demo:
            frames = self._demo_frames(1)
            return frames[0] if frames else None
        with self._frame_lock:
            return None if self._latest is None else self._latest.copy()

    def views(self, n: int, delay: float = 0.8):
        """
        Capture n tray positions, rotating between them.

        Samples the reader's newest frame after each rotation + settle, rather
        than calling read() directly. That avoids two threads fighting over the
        device, and guarantees the frame is post-rotation rather than a stale
        buffered one from before the tray moved.
        """
        if self.demo:
            return self._demo_frames(n)

        frames = []
        for i in range(n):
            self.rotate(i)
            time.sleep(delay)          # let the tray settle AND the buffer refresh
            f = self.snapshot()
            if f is not None:
                frames.append(f)
        return frames


# --------------------------------------------------------------------------- #
# Request handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "FreshIQ"
    camera: CameraSource = None      # injected below
    history: History = None
    scanner = None                   # ScanController, or None if unavailable
    settings = None                  # device settings, or None
    scan_error: str = None           # why the scanner could not be built
    device_id = "freshiq-unit-01"
    preview_width = 640              # downscale the preview, not the model input
    preview_quality = 70
    preview_fps = 15

    # ---- small helpers ----------------------------------------------------
    def _json(self, code: int, payload) -> None:
        body = json.dumps(payload, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, code: int, body: bytes, ctype: str, extra=None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode() or "{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    # ---- routes -----------------------------------------------------------
    def do_GET(self):
        route = urlparse(self.path)
        path, query = route.path, parse_qs(route.query)

        if path in ("/", "/index.html"):
            try:
                with open(DASHBOARD, "rb") as fh:
                    self._bytes(200, fh.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._bytes(500, b"dashboard.html is missing", "text/plain")

        elif path == "/api/health":
            backend = model_module.build_backend(get_config())
            self._json(200, {
                "status": "ok",
                "ai_version": ai_version,
                "backend": backend.name,
                "trained_model": backend.name != "heuristic",
                "camera": "demo" if self.camera.demo else f"index {self.camera.index}",
                "camera_error": self.camera.error,
                "conditions": get_config().conditions,
                "foods": [f for f in get_config().foods if f != "unknown"],
                # Sent so the dashboard can say "focus 23, needs 55" instead of
                # just "blurred", which tells you nothing about how to fix it.
                "quality_thresholds": {
                    "min_focus_variance": get_config().quality.min_focus_variance,
                    "min_brightness": get_config().quality.min_brightness,
                    "max_brightness": get_config().quality.max_brightness,
                    "min_food_area_fraction": get_config().quality.min_food_area_fraction,
                },
                "device_id": self.device_id,
                "history_count": len(self.history.all()),
            })

        elif path == "/api/snapshot":
            frame = self.camera.snapshot()
            if frame is None:
                self._json(503, {"error": "no frame", "detail": self.camera.error})
                return
            jpg = self._encode_preview(frame)
            if jpg is None:
                self._json(500, {"error": "encode failed"})
                return
            self._bytes(200, jpg, "image/jpeg", {"Cache-Control": "no-store"})

        elif path == "/api/stream":
            self._mjpeg_stream()

        elif path == "/api/history":
            limit = int((query.get("limit") or ["50"])[0])
            self._json(200, {"items": self.history.read(limit)})

        elif path == "/api/stats":
            self._json(200, self._stats())

        elif path == "/api/export.csv":
            self._bytes(200, self._csv().encode(), "text/csv",
                        {"Content-Disposition": 'attachment; filename="freshiq_history.csv"'})

        # ---- scan / hardware -------------------------------------------
        elif path == "/api/scan/status":
            if not self._need_scanner():
                return
            self._json(200, self.scanner.status())

        elif path == "/api/scan/results":
            if not self._need_scanner():
                return
            st = self.scanner.status()
            self._json(200, {
                "scan_id": st["scan_id"],
                "state": st["state"],
                "compartments": st["compartments"],
                "results": st["results"],
            })

        elif path == "/api/motor/status":
            if not self._need_scanner():
                return
            self._json(200, self.scanner.motor.status())

        elif path == "/api/system/status":
            self._json(200, self._system_status())

        else:
            self._json(404, {"error": "not_found", "path": path})

    # ---- scan helpers ------------------------------------------------------
    def _need_scanner(self) -> bool:
        """Reply 503 with a useful reason if tray control is not available."""
        if self.scanner is not None:
            return True
        self._json(503, {
            "error": "scan_unavailable",
            "detail": self.scan_error or SCAN_IMPORT_ERROR
                      or "scan controller was not started",
            "hint": "Check config/freshiq.json and the motor port. Single-shot "
                    "/api/inspect still works without the tray.",
        })
        return False

    def _system_status(self) -> dict:
        cfg = get_config()
        backend = model_module.build_backend(cfg)
        out = {
            "device_id": self.device_id,
            "ai_version": ai_version,
            "ai": {
                "backend": backend.name,
                "trained_model": backend.name != "heuristic",
                "conditions": cfg.conditions,
                "ready": True,
            },
            "camera": {
                "ready": not self.camera.demo,
                "mode": "demo" if self.camera.demo else "live",
                "index": self.camera.index,
                "error": self.camera.error,
            },
            "scan_available": self.scanner is not None,
            "scan_error": self.scan_error or SCAN_IMPORT_ERROR,
            "history_count": len(self.history.all()),
        }
        if self.scanner is not None:
            st = self.scanner.status()
            out["mode"] = st["mode"]
            out["tray"] = {
                "compartments": st["compartments"],
                "position": st["motor"]["position"],
                "state": st["state"],
                "moving": st["motor"]["moving"],
            }
            out["motor"] = st["motor"]
            out["lighting"] = st["lighting"]
        return out

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/classify":
            self._classify_upload()
            return
        if path == "/api/capture":
            self._capture_training_example()
            return

        # ---- scan control ----------------------------------------------
        if path == "/api/scan/start":
            if not self._need_scanner():
                return
            body = self._read_json_body()
            comps = body.get("compartments")
            hints = body.get("food_hints") or {}
            try:
                info = self.scanner.start(compartments=comps, food_hints=hints)
            except scan_module.ScanError as exc:
                self._json(409, {"error": "scan_rejected", "detail": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": "scan_failed", "detail": str(exc)})
                return
            self._json(202, {"started": True, **info})
            return

        if path == "/api/scan/stop":
            if not self._need_scanner():
                return
            self.scanner.cancel()
            self._json(200, {"cancelled": True, "state": self.scanner.status()["state"]})
            return

        if path == "/api/scan/compartment":
            if not self._need_scanner():
                return
            body = self._read_json_body()
            try:
                target = int(body.get("compartment", 0))
            except (TypeError, ValueError):
                self._json(400, {"error": "bad_request",
                                 "detail": "compartment must be an integer"})
                return
            food = body.get("food") or None
            if food in ("", "auto"):
                food = None
            try:
                record = self.scanner.scan_one(target, food_hint=food)
            except (scan_module.ScanError, MotorError) as exc:
                self._json(409, {"error": "scan_rejected", "detail": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": "scan_failed", "detail": str(exc)})
                return
            self._json(200, record)
            return

        if path == "/api/motor/position":
            # Tell the software where the tray physically is. Needed because the
            # rig has no home switch, so after manual alignment the logical index
            # has to be corrected by hand.
            if not self._need_scanner():
                return
            body = self._read_json_body()
            try:
                self.scanner.motor.set_position(int(body.get("position", 1)))
            except (MotorError, TypeError, ValueError) as exc:
                self._json(400, {"error": "bad_request", "detail": str(exc)})
                return
            self._json(200, self.scanner.motor.status())
            return

        if path != "/api/inspect":
            self._json(404, {"error": "not_found"})
            return

        body = self._read_json_body()
        views = max(1, min(int(body.get("views", 4)), 8))
        food = body.get("food") or None
        if food in ("", "auto", None):
            food = None
        # Live mode polls this endpoint every couple of seconds. Writing all of
        # that to history would bury the inspections you actually care about
        # under hundreds of throwaway frames, so live mode sends save=false.
        save = bool(body.get("save", True))

        started = time.perf_counter()
        try:
            frames = self.camera.views(views)
        except Exception as exc:
            self._json(503, {"error": "capture_failed", "detail": str(exc)})
            return
        if not frames:
            self._json(503, {"error": "no_frames"})
            return

        result = inspect(frames, food_hint=food)
        for v in result.get("views", []):
            v.pop("confidence_raw", None)

        record = {
            "inspection_id": f"insp_{uuid.uuid4().hex[:12]}",
            "device_id": self.device_id,
            "ai_version": ai_version,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "demo": self.camera.demo,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "result": result,
            "disclaimer": DISCLAIMER,
        }
        if save:
            self.history.append(record)
        record["saved"] = save
        self._json(200, record)

    # ---- preview encoding + MJPEG stream -----------------------------------
    def _encode_preview(self, frame):
        """
        Downscale before encoding.

        The camera runs at 1280x720, but the preview is displayed in a card a few
        hundred pixels wide. Encoding and shipping full 720p JPEGs many times a
        second is most of the bandwidth and most of the CPU, for detail nobody
        can see. The MODEL never sees this image - it gets the full-resolution
        frame - so shrinking the preview costs no accuracy.
        """
        import cv2
        w = self.preview_width
        if w and frame.shape[1] > w:
            h = int(frame.shape[0] * (w / frame.shape[1]))
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, self.preview_quality])
        return buf.tobytes() if ok else None

    def _mjpeg_stream(self):
        """
        multipart/x-mixed-replace: one connection, frames pushed as they come.

        Beats polling for a live view - no per-frame request overhead, and the
        browser swaps the image the moment each part arrives. Costs one thread
        for as long as the tab is open, which is why ThreadingHTTPServer matters.
        """
        boundary = "freshiqframe"
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type",
                         f"multipart/x-mixed-replace; boundary={boundary}")
        self.end_headers()

        interval = 1.0 / max(self.preview_fps, 1)
        starved = False
        try:
            while True:
                start = time.perf_counter()
                frame = self.camera.snapshot()
                if frame is None:
                    # Headers are already sent, so the browser holds an open
                    # connection that never receives a part: a blank preview and
                    # no error, on either side. Name it once so the log says why.
                    if not starved:
                        starved = True
                        print("[stream] no frame to send yet ({}) - the preview "
                              "will stay blank until the camera produces one"
                              .format(self.camera.error or "camera has not "
                                      "delivered a frame since startup"),
                              flush=True)
                else:
                    if starved:
                        starved = False
                        print("[stream] frames flowing again", flush=True)
                    jpg = self._encode_preview(frame)
                    if jpg:
                        self.wfile.write(f"--{boundary}\r\n".encode())
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                time.sleep(max(0.0, interval - (time.perf_counter() - start)))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass        # the tab closed or navigated away - entirely normal

    # ---- classify an image sent BY THE BROWSER -----------------------------
    def _classify_upload(self):
        """
        Classify a frame captured on the viewing device (laptop webcam, phone
        camera) rather than from a camera wired to the server.

        The image arrives as a base64 data URL in a JSON body. Base64 in JSON is
        ~33% bigger than raw multipart, but it needs no multipart parser - and
        Python's `cgi` module, which is what you would normally reach for, was
        REMOVED in 3.13. A few extra kilobytes over the LAN is a good trade for
        code that will not break on a future Python.
        """
        import base64
        import cv2
        import numpy as np

        body = self._read_json_body()
        raw = body.get("image") or ""
        if "," in raw[:64]:                       # strip "data:image/jpeg;base64,"
            raw = raw.split(",", 1)[1]
        if not raw:
            self._json(400, {"error": "no_image", "detail": "send {'image': '<base64 or data URL>'}"})
            return

        try:
            data = base64.b64decode(raw, validate=False)
        except Exception as exc:
            self._json(400, {"error": "bad_base64", "detail": str(exc)})
            return

        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self._json(400, {"error": "undecodable_image",
                             "detail": f"{len(data)} bytes did not decode as an image"})
            return

        food = body.get("food") or None
        if food in ("", "auto"):
            food = None
        save = bool(body.get("save", False))      # live preview defaults to NOT saving

        started = time.perf_counter()
        result = inspect([frame], food_hint=food)
        for v in result.get("views", []):
            v.pop("confidence_raw", None)

        record = {
            "inspection_id": f"insp_{uuid.uuid4().hex[:12]}",
            "device_id": self.device_id,
            "ai_version": ai_version,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "device_camera",
            "demo": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "result": result,
            "disclaimer": DISCLAIMER,
        }
        if save:
            self.history.append(record)
        record["saved"] = save
        self._json(200, record)

    # ---- capture a LABELLED training example -------------------------------
    def _capture_training_example(self):
        """
        Save a frame straight into dataset/train/<condition>/<food>/ .

        THIS IS THE ONLY THING THAT FIXES REAL-WORLD ACCURACY.

        A model trained on web photos of supermarket produce scores 99% on web
        photos of supermarket produce. Your rig has different light, a different
        lens, different backgrounds and different distances. That gap does not
        close with more web photos - it closes with a few hundred photos taken
        THROUGH THIS EXACT SETUP, used to fine-tune.

        The layout written here is what training/dataset.py detects as "nested",
        which gives the two-head model. So captures drop straight into:

            python training/train.py --data-dir dataset --fine-tune --lr 1e-5
        """
        import base64
        import cv2
        import numpy as np

        body = self._read_json_body()
        condition = (body.get("condition") or "").strip().lower()
        food = (body.get("food") or "").strip().lower().replace(" ", "_")
        split = (body.get("split") or "train").strip().lower()

        cfg = get_config()
        if condition not in cfg.conditions:
            self._json(400, {"error": "bad_condition",
                             "detail": f"must be one of {cfg.conditions}"})
            return
        if not food or food == "unknown":
            self._json(400, {"error": "no_food",
                             "detail": "pick which food this is - the folder name depends on it"})
            return
        if split not in ("train", "validation", "test"):
            self._json(400, {"error": "bad_split"})
            return

        raw = body.get("image") or ""
        if "," in raw[:64]:
            raw = raw.split(",", 1)[1]
        if not raw:
            self._json(400, {"error": "no_image"})
            return
        try:
            frame = cv2.imdecode(np.frombuffer(base64.b64decode(raw), np.uint8),
                                 cv2.IMREAD_COLOR)
        except Exception as exc:
            self._json(400, {"error": "bad_image", "detail": str(exc)})
            return
        if frame is None:
            self._json(400, {"error": "undecodable_image"})
            return

        out_dir = os.path.join(REPO, "dataset", split, condition, food)
        os.makedirs(out_dir, exist_ok=True)
        name = f"rig_{uuid.uuid4().hex[:10]}.jpg"
        cv2.imwrite(os.path.join(out_dir, name), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Count what has been collected so far, per class, so the UI can show
        # progress toward a useful fine-tuning set.
        counts = {}
        base = os.path.join(REPO, "dataset", split)
        for cond in cfg.conditions:
            n = 0
            cdir = os.path.join(base, cond)
            for root, _dirs, files in os.walk(cdir):
                n += sum(1 for f in files if f.lower().endswith((".jpg", ".jpeg", ".png")))
            counts[cond] = n

        self._json(200, {
            "saved": os.path.relpath(os.path.join(out_dir, name), REPO),
            "counts": counts,
            "total": sum(counts.values()),
        })

    # ---- derived data -----------------------------------------------------
    def _stats(self) -> dict:
        items = self.history.all()
        counts, foods = {}, {}
        today = datetime.now(timezone.utc).date().isoformat()
        today_n = 0
        for r in items:
            res = r.get("result", {})
            c = res.get("condition") or "unknown"
            counts[c] = counts.get(c, 0) + 1
            f = res.get("food") or "unknown"
            foods[f] = foods.get(f, 0) + 1
            if str(r.get("timestamp", "")).startswith(today):
                today_n += 1
        total = len(items)
        return {
            "total": total,
            "today": today_n,
            "by_condition": counts,
            "by_food": foods,
            "spoiled_rate": round(counts.get("spoiled", 0) / total, 4) if total else 0.0,
        }

    def _csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["timestamp", "inspection_id", "compartment", "food", "condition",
                    "fresh_pct", "spoiled_pct", "image_quality", "views_used",
                    "elapsed_ms", "demo", "evidence"])
        for r in reversed(self.history.all()):      # oldest first in the file
            res = r.get("result", {})
            conf = res.get("confidence", {}) or {}
            w.writerow([
                r.get("timestamp", ""), r.get("inspection_id", ""),
                # Blank for pre-tray single-shot records, which had no compartment.
                res.get("compartment", r.get("compartment", "")),
                res.get("food", ""), res.get("condition", ""),
                conf.get("fresh", ""), conf.get("spoiled", ""),
                res.get("image_quality", ""), res.get("views_used", ""),
                r.get("elapsed_ms", ""), r.get("demo", ""),
                "; ".join(res.get("evidence", []) or []),
            ])
        return buf.getvalue()

    def log_message(self, fmt, *args):
        # Default logging prints every snapshot poll - far too noisy, hence the
        # filter. But args[0] is only the request line for NORMAL requests:
        # BaseHTTPRequestHandler.log_error() routes through here too, and it is
        # called as log_error("code %d, message %s", code, message) - so args[0]
        # is an HTTPStatus. Testing `"..." not in <HTTPStatus>` raised
        # TypeError, which escaped the handler and tore down the connection
        # mid-response. Every error path - a HEAD request, a 404, a malformed
        # line - therefore reached the client as "Empty reply from server"
        # instead of a real status code, and hid whatever the actual fault was.
        first = args[0] if args else ""
        if not isinstance(first, str):
            first = str(first)
        if "/api/snapshot" not in first:
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


def main():
    ap = argparse.ArgumentParser(description="FreshIQ local server + dashboard.")
    # These default to None so config/freshiq.json supplies the value and an
    # explicit flag still wins. Config is the source of truth; CLI is the override.
    ap.add_argument("--config", default=None,
                    help="path to the device config (default: config/freshiq.json)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default=None,
                    help="0.0.0.0 makes it reachable from other devices on the LAN")
    ap.add_argument("--camera", type=int, default=None, help="USB camera index")
    ap.add_argument("--device-id", default=None)
    ap.add_argument("--no-tray", action="store_true",
                    help="skip motor init; single-shot inspection only")
    ap.add_argument("--rotate-serial", default=None,
                    help="LEGACY, external Arduino only: serial port that gets a bare "
                         "'R' between preview views. NOT for the UNO Q - its onboard "
                         "MCU is driven by the tray driver in config/freshiq.json.")
    ap.add_argument("--rotate-cmd", default=None,
                    help="shell command to run between views instead of serial")
    ap.add_argument("--min-focus", type=float, default=None,
                    help="lower the blur threshold (default 55). Laptop and phone "
                         "cameras are softer than a tray-mounted USB camera - try 20.")
    ap.add_argument("--min-brightness", type=float, default=None,
                    help="lower the too-dark threshold (default 40)")
    ap.add_argument("--preview-width", type=int, default=640,
                    help="preview downscale width; lower = smoother on weak hardware")
    ap.add_argument("--preview-quality", type=int, default=70, help="preview JPEG quality 1-100")
    ap.add_argument("--preview-fps", type=int, default=15, help="target preview frames per second")
    args = ap.parse_args()

    # Patch the shared default config BEFORE anything calls get_config(), which
    # hands out deep copies of it. Camera-specific thresholds are deployment
    # settings, not facts about the model, so overriding them here is right.
    from ai import config as ai_config
    if args.min_focus is not None:
        ai_config.DEFAULT_CONFIG.quality.min_focus_variance = args.min_focus
        print(f"[quality] blur threshold lowered to {args.min_focus}")
    if args.min_brightness is not None:
        ai_config.DEFAULT_CONFIG.quality.min_brightness = args.min_brightness
        print(f"[quality] darkness threshold lowered to {args.min_brightness}")

    # ---- device settings ---------------------------------------------------
    settings = None
    scan_error = None
    if SCAN_AVAILABLE:
        try:
            settings = load_settings(args.config)
            print("[config] {}  mode={}  compartments={}  steps/compartment={}".format(
                settings.source_path, settings.mode,
                settings.tray.compartments, settings.steps_per_compartment()))
        except Exception as exc:  # noqa: BLE001
            scan_error = "{}: {}".format(type(exc).__name__, exc)
            print("[config] NOT loaded ({}) -> tray control disabled".format(scan_error))
    else:
        scan_error = SCAN_IMPORT_ERROR
        print("[config] scan layer unavailable ({})".format(scan_error))

    def pick(cli, path, fallback):
        if cli is not None:
            return cli
        if settings is not None:
            obj = settings
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    return fallback
            return obj
        return fallback

    host = pick(args.host, "server.host", "0.0.0.0")
    port = int(pick(args.port, "server.port", 8000))
    # camera.index may be an int OR a device path like "/dev/video0". Forcing
    # int() here would reject the path form that camera/capture.py now accepts.
    cam_index = pick(args.camera, "camera.index", 0)
    if not isinstance(cam_index, str):
        cam_index = int(cam_index)
    device_id = pick(args.device_id, "device_id", "freshiq-unit-01")

    Handler.camera = CameraSource(cam_index, args.rotate_cmd, args.rotate_serial)
    Handler.history = History(HISTORY_FILE)
    Handler.device_id = device_id
    Handler.settings = settings
    Handler.scan_error = scan_error
    Handler.preview_width = args.preview_width
    Handler.preview_quality = args.preview_quality
    Handler.preview_fps = args.preview_fps

    backend = model_module.build_backend(get_config())
    print(f"[ai]     backend: {backend.name}"
          + ("   <- no trained model found, using rules" if backend.name == "heuristic" else ""))
    print(f"[data]   history: {HISTORY_FILE}")

    # ---- tray -------------------------------------------------------------
    # A motor that will not open must NOT take the dashboard down with it. The
    # server degrades to single-shot inspection and says why on /api/system/status.
    if settings is not None and not args.no_tray:
        try:
            motor = build_motor(settings)
            lighting = build_lighting(settings, motor=motor)
            Handler.scanner = scan_module.ScanController(
                settings=settings,
                camera=Handler.camera,
                motor=motor,
                lighting=lighting,
                history=Handler.history,
                inspect_fn=inspect,
                device_id=device_id,
            )
            print("[tray]   {} on {}, {} compartments, {} steps each".format(
                motor.__class__.__name__,
                getattr(motor, "port", "n/a"), settings.tray.compartments,
                settings.steps_per_compartment()))
            print("[light]  {}".format(lighting.status()["mode"]))
        except Exception as exc:  # noqa: BLE001
            Handler.scan_error = "{}: {}".format(type(exc).__name__, exc)
            print("[tray]   UNAVAILABLE ({})".format(Handler.scan_error))
            print("         Diagnose it with:  python3 tools/check_tray.py")
            print("         Single-shot /api/inspect still works.")
    elif args.no_tray:
        Handler.scan_error = "--no-tray was passed"
        print("[tray]   disabled by --no-tray")

    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"\n  FreshIQ dashboard -> http://localhost:{port}")
    if host == "0.0.0.0":
        print(f"  from other devices -> http://<this-board-ip>:{port}")
    print("\n  Ctrl+C to stop\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
        if Handler.scanner is not None:
            Handler.scanner.cancel()
            try:
                Handler.scanner.motor.close()
            except Exception:  # noqa: BLE001
                pass
        Handler.camera.stop()
        srv.server_close()


if __name__ == "__main__":
    main()
