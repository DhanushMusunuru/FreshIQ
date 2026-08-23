"""
Scan controller: the state machine that turns a tray of food into a set of
per-compartment results.

WHY THIS EXISTS
---------------
ai.inspect([...]) aggregates N images into ONE verdict using "worst view wins".
That is exactly right for four photos of one tomato, and exactly wrong for four
compartments holding four different foods - it would blend a fresh apple and a
rotten banana into a single answer.

So the loop is inverted here. This module owns the compartment index and calls
ai.inspect() once PER COMPARTMENT, with only that compartment's views. The AI
package is not modified at all.

    for each compartment:
        MOVING       motor.move_to(n)        <- blocks until the MCU says OK
        STABILIZING  sleep(stabilize)        <- the platter is still wobbling
        CAPTURING    camera.snapshot() x N   <- tray is stationary
        ANALYZING    ai.inspect(views)       <- one item, N views
        SAVING       history + images

THE COMPARTMENT NUMBER NEVER COMES FROM THE AI. It comes from this controller,
which is the only thing that knows where the tray was pointed. The AI only ever
answers "what is in this picture".

SAFETY
------
Every exit path from the scan thread - success, failure, or cancellation -
passes through _safe_stop(). A Python exception must never leave a stepper
energised and turning.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

# States -------------------------------------------------------------------
IDLE = "IDLE"
INITIALIZING = "INITIALIZING"
MOVING = "MOVING"
STABILIZING = "STABILIZING"
CAPTURING = "CAPTURING"
ANALYZING = "ANALYZING"
SAVING_RESULT = "SAVING_RESULT"
NEXT_COMPARTMENT = "NEXT_COMPARTMENT"
COMPLETE = "COMPLETE"
CANCELLED = "CANCELLED"
ERROR = "ERROR"
SAFE_STOP = "SAFE_STOP"

TERMINAL = (COMPLETE, CANCELLED, ERROR, IDLE)


class ScanError(RuntimeError):
    pass


class ScanCancelled(ScanError):
    """
    Distinct from ScanError on purpose.

    A cancellation that arrives mid-capture used to unwind as a plain
    ScanError, so a user pressing Stop got a scan reported as ERROR - which
    looks like the rig broke. Stopping on request is a normal outcome and must
    read as one.
    """


def normalise_result(agg: Dict, compartment: int, timestamp: str) -> Dict:
    """
    Flatten one aggregated AI result into the single result contract shared by
    the AI, the backend and the dashboard.

    Normalising ONCE, here, is deliberate: the alternative is three components
    each inventing their own shape for the same data and drifting apart.

    `condition_confidence` is the winning condition's own percentage, expressed
    0..1. `food_confidence` is not present on the aggregate, so it is taken from
    the best view that actually identified this food.
    """
    conf = agg.get("confidence") or {}
    condition = agg.get("condition")
    condition_conf = float(conf.get(condition, 0)) / 100.0 if condition else 0.0

    food = agg.get("food")
    food_conf = 0.0
    for v in agg.get("views", []):
        if v.get("status") == "ok" and v.get("food") == food:
            food_conf = max(food_conf, float(v.get("food_confidence", 0.0)))

    return {
        "status": agg.get("status", "ok"),
        "compartment": compartment,
        "food": food,
        "food_confidence": round(food_conf, 3),
        "condition": condition,
        "condition_confidence": round(condition_conf, 3),
        "confidence": conf,
        "image_quality": agg.get("image_quality"),
        "evidence": agg.get("evidence", []),
        "reason": agg.get("reason"),
        "views_used": agg.get("views_used"),
        "views_total": agg.get("views_total"),
        "timestamp": timestamp,
    }


class ImageStore:
    """
    Saves inspection frames under data/inspections/<date>/compartment_NN/.

    Retention is enforced on every scan rather than by a cron job, because a
    device that is only ever powered on to do a scan would never run the cron.
    """

    def __init__(self, root: str, enabled: bool = True,
                 retention_days: int = 14, max_images: int = 2000):
        self.root = root
        self.enabled = enabled
        self.retention_days = int(retention_days)
        self.max_images = int(max_images)

    def save(self, frame, compartment: int, scan_id: str, view: int) -> Optional[str]:
        if not self.enabled or frame is None:
            return None
        try:
            import cv2
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            folder = os.path.join(self.root, day,
                                  "compartment_{:02d}".format(compartment))
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, "{}_v{}.jpg".format(scan_id, view))
            cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            return os.path.relpath(path, os.path.dirname(self.root))
        except Exception:  # noqa: BLE001
            # Losing a debug image must never abort a scan.
            return None

    def enforce_retention(self) -> int:
        if not self.enabled or not os.path.isdir(self.root):
            return 0
        removed = 0
        try:
            files = []
            for dirpath, _dirs, names in os.walk(self.root):
                for n in names:
                    if n.lower().endswith(".jpg"):
                        p = os.path.join(dirpath, n)
                        files.append((os.path.getmtime(p), p))
            files.sort()

            cutoff = time.time() - self.retention_days * 86400
            keep = []
            for mtime, p in files:
                if mtime < cutoff:
                    try:
                        os.remove(p)
                        removed += 1
                    except OSError:
                        pass
                else:
                    keep.append(p)

            excess = len(keep) - self.max_images
            for p in keep[:max(0, excess)]:
                try:
                    os.remove(p)
                    removed += 1
                except OSError:
                    pass
        except Exception:  # noqa: BLE001
            pass
        return removed


class ScanController:
    """
    Owns the tray, the scan thread and the results.

    One scan may run at a time. start() returns immediately; poll status().
    """

    def __init__(self, settings, camera, motor, lighting, history,
                 inspect_fn: Callable, device_id: str = "freshiq-unit-01",
                 log: Optional[Callable[[str], None]] = None):
        self.settings = settings
        self.camera = camera
        self.motor = motor
        self.lighting = lighting
        self.history = history
        self.inspect_fn = inspect_fn
        self.device_id = device_id
        self.log = log or (lambda m: print(m, flush=True))

        self.compartments = int(settings.tray.compartments)
        self.stabilize = float(settings.tray.stabilize_seconds)
        self.views_per = max(1, int(settings.camera.views_per_compartment))
        self.view_delay = float(getattr(settings.camera, "inter_view_delay_seconds", 0.3))

        self.images = ImageStore(
            settings.path("image_dir"),
            enabled=bool(settings.storage.save_images),
            retention_days=int(settings.storage.retention_days),
            max_images=int(settings.storage.max_images),
        )

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()

        self.state = IDLE
        self.scan_id: Optional[str] = None
        self.current: Optional[int] = None
        self.queue: List[int] = []
        self.done: int = 0
        self.results: Dict[int, Dict] = {}
        self.error: Optional[str] = None
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None

    # ---- public API --------------------------------------------------------
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> Dict:
        with self._lock:
            return {
                "state": self.state,
                "running": self.is_running(),
                "scan_id": self.scan_id,
                "current_compartment": self.current,
                "completed": self.done,
                "total": len(self.queue),
                "compartments": self.compartments,
                "results": [self.results[k] for k in sorted(self.results)],
                "error": self.error,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "motor": self.motor.status(),
                "lighting": self.lighting.status(),
                "mode": self.settings.mode,
            }

    def start(self, compartments: Optional[List[int]] = None,
              food_hints: Optional[Dict[int, str]] = None) -> Dict:
        if self.is_running():
            raise ScanError("a scan is already running")

        targets = compartments or list(range(1, self.compartments + 1))
        for c in targets:
            if not 1 <= int(c) <= self.compartments:
                raise ScanError(
                    "invalid compartment {} (valid: 1..{})".format(c, self.compartments))

        with self._lock:
            self._cancel.clear()
            self.scan_id = "scan_{}".format(uuid.uuid4().hex[:10])
            self.queue = [int(c) for c in targets]
            self.results = {}
            self.done = 0
            self.current = None
            self.error = None
            self.finished_at = None
            self.started_at = _now()
            self.state = INITIALIZING

        self._thread = threading.Thread(
            target=self._run, args=(dict(food_hints or {}),),
            name="freshiq-scan", daemon=True)
        self._thread.start()
        return {"scan_id": self.scan_id, "compartments": self.queue}

    def cancel(self) -> None:
        """Ask the scan to stop. The motor is stopped immediately."""
        self._cancel.set()
        try:
            self.motor.stop()
        except Exception as exc:  # noqa: BLE001
            self.log("[motor] stop during cancel failed: {}".format(exc))

    def scan_one(self, compartment: int, food_hint: Optional[str] = None) -> Dict:
        """Synchronously inspect a single compartment. Used by /api/scan/compartment."""
        if self.is_running():
            raise ScanError("a scan is already running")
        # The cancel flag is sticky until the next scan starts. Without this,
        # the first single-compartment scan after any cancelled run aborts
        # instantly with "cancelled during capture".
        self._cancel.clear()
        self.scan_id = "scan_{}".format(uuid.uuid4().hex[:10])
        try:
            return self._one(compartment, food_hint)
        finally:
            self._safe_stop()
            self.state = IDLE

    # ---- internals ---------------------------------------------------------
    def _set(self, state: str, compartment: Optional[int] = None) -> None:
        with self._lock:
            self.state = state
            if compartment is not None:
                self.current = compartment

    def _safe_stop(self) -> None:
        try:
            self.motor.stop()
        except Exception as exc:  # noqa: BLE001
            self.log("[motor] safe stop failed: {}".format(exc))

    def _run(self, food_hints: Dict) -> None:
        self.log("[scan]   {} starting, {} compartment(s)".format(
            self.scan_id, len(self.queue)))
        # Decided here, applied in `finally`. The cleanup path passes through
        # SAFE_STOP, so it cannot be allowed to infer the outcome afterwards -
        # doing that turned a cancellation into COMPLETE.
        final_state = COMPLETE
        try:
            self.lighting.on()
            for target in self.queue:
                if self._cancel.is_set():
                    final_state = CANCELLED
                    self.log("[scan]   cancelled by user")
                    break

                hint = food_hints.get(target) or food_hints.get(str(target))
                record = self._one(target, hint)

                with self._lock:
                    self.results[target] = record["result"]
                    self.done += 1
                self._set(NEXT_COMPARTMENT)
            else:
                self.log("[scan]   complete")

        except ScanCancelled:
            final_state = CANCELLED
            self.log("[scan]   cancelled mid-compartment")
        except Exception as exc:  # noqa: BLE001
            final_state = ERROR
            with self._lock:
                self.error = "{}: {}".format(type(exc).__name__, exc)
            self.log("[error]  scan failed: {}".format(self.error))
        finally:
            # Every path lands here: success, error, cancel.
            self._set(SAFE_STOP)
            self._safe_stop()
            removed = self.images.enforce_retention()
            if removed:
                self.log("[data]   retention removed {} old image(s)".format(removed))
            with self._lock:
                self.finished_at = _now()
                self.current = None
                self.state = final_state

    def _one(self, target: int, food_hint: Optional[str]) -> Dict:
        """Move to one compartment, capture, analyse, persist. Returns the record."""
        # --- MOVE ----------------------------------------------------------
        self._set(MOVING, target)
        self.log("[scan]   moving to compartment {}".format(target))
        moved = self.motor.move_to(target)

        # --- STABILIZE -----------------------------------------------------
        # Only wait if we actually moved. The tray is a 200 mm platter on a
        # lazy-susan bearing and keeps oscillating after the motor stops; the
        # motor reporting OK is NOT the same as the tray being still.
        if moved:
            self._set(STABILIZING, target)
            time.sleep(self.stabilize)

        # --- CAPTURE -------------------------------------------------------
        self._set(CAPTURING, target)
        frames = []
        for v in range(self.views_per):
            if self._cancel.is_set():
                raise ScanCancelled("cancelled during capture")

            # Retry briefly. USB cameras drop the occasional frame - especially
            # just after the tray shakes the rig - and losing a whole scan to
            # one missed grab is not acceptable. Persistent failure is a real
            # fault and still raises.
            frame = None
            for attempt in range(3):
                frame = self.camera.snapshot()
                if frame is not None:
                    break
                if self._cancel.is_set():
                    raise ScanCancelled("cancelled during capture")
                self.log("[camera] no frame (attempt {}/3), retrying".format(attempt + 1))
                time.sleep(0.35)

            if frame is None:
                raise ScanError(
                    "camera returned no frame for compartment {} after 3 attempts "
                    "(camera disconnected, or another process is holding it)"
                    .format(target))
            frames.append(frame)
            if v + 1 < self.views_per:
                time.sleep(self.view_delay)

        # --- ANALYSE -------------------------------------------------------
        self._set(ANALYZING, target)
        started = time.perf_counter()
        agg = self.inspect_fn(frames, food_hint=food_hint)
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        for v in agg.get("views", []):
            v.pop("confidence_raw", None)

        # --- SAVE ----------------------------------------------------------
        self._set(SAVING_RESULT, target)
        ts = _now()
        result = normalise_result(agg, target, ts)

        paths = [self.images.save(f, target, self.scan_id or "scan", i)
                 for i, f in enumerate(frames)]
        paths = [p for p in paths if p]
        if paths:
            result["images"] = paths

        record = {
            "inspection_id": "insp_{}".format(uuid.uuid4().hex[:12]),
            "scan_id": self.scan_id,
            "device_id": self.device_id,
            "timestamp": ts,
            "compartment": target,
            "elapsed_ms": elapsed,
            "mode": self.settings.mode,
            "result": result,
            "views": agg.get("views", []),
        }
        try:
            self.history.append(record)
        except Exception as exc:  # noqa: BLE001
            self.log("[data]   history write failed: {}".format(exc))

        self.log("[scan]   compartment {}: {} / {} ({}% {})".format(
            target, result.get("food"), result.get("condition"),
            int(result.get("condition_confidence", 0) * 100), result.get("condition")))
        return record


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
