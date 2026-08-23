"""
Device settings loader.

Two different "configs" exist in FreshIQ and they are deliberately separate:

    ai/config.py       what the MODEL is and how to judge an image
                       (classes, thresholds, aggregation weights)

    config/freshiq.json   what this RIG is
                       (how many compartments, which serial port, which camera)

Keeping them apart means you can swap trays or cameras without touching
anything the model depends on, and retrain the model without touching hardware.

Usage:

    from config import load_settings
    s = load_settings()
    s.tray.compartments        -> 4
    s.motor.driver             -> "bridge"
    s.steps_per_compartment()  -> 50
    s.step_delay_us()          -> 70000   (50 steps x 70 ms = 3.5 s)

Search order:
    1. explicit path argument
    2. FRESHIQ_DEVICE_CONFIG environment variable
    3. config/freshiq.json next to this file
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_PATH = os.path.join(HERE, "freshiq.json")


class Settings(SimpleNamespace):
    """Dot-accessible settings with a few derived helpers."""

    # ---- derived values ---------------------------------------------------
    def steps_per_compartment(self) -> int:
        """
        Motor steps for exactly one compartment.

        1:1 direct drive (coupling hub straight onto the motor shaft, no
        gearing - see cad/), so one tray revolution == one motor revolution.
        """
        t = self.tray
        total = int(t.steps_per_revolution) * int(t.microsteps)
        if total % int(t.compartments) != 0:
            # Not fatal, but the tray will drift a fraction of a step each
            # revolution, so say so loudly rather than silently rounding.
            raise ValueError(
                "steps_per_revolution * microsteps ({}) is not divisible by "
                "compartments ({}). The tray would drift every revolution. "
                "Pick a microstep setting that divides evenly."
                .format(total, t.compartments))
        return total // int(t.compartments)

    def seconds_per_compartment(self) -> float:
        """How long one compartment of travel should take, in seconds."""
        value = float(getattr(self.tray, "seconds_per_compartment", 0) or 0)
        return value if value > 0 else 3.5

    def step_delay_us(self) -> int:
        """
        Per-step pause handed to the MCU.

        The sketch has no idea how big the tray is or how fast you want it; it
        just waits this long between steps. Deriving the value here keeps
        config/freshiq.json the single source of truth for geometry AND speed,
        so tuning the tray never means re-flashing the microcontroller.
        """
        steps = self.steps_per_compartment()
        if steps < 1:
            return 0

        # The sketch stretches the first and last `ramp` steps to ease the
        # platter in and out, which adds time. Dividing the target by `steps`
        # alone therefore overshoots - a tray asked for 3.5 s took 4.1 s. Budget
        # for the ramp so the configured number is the number you observe.
        # Mirrors delayForStep() in sketch/sketch.ino: the ramp starts 60%
        # slower than cruise and eases down, averaging ~1.3x cruise.
        ramp = min(10, steps // 5)
        effective_steps = steps + 0.6 * ramp
        return int(self.seconds_per_compartment() * 1_000_000 / effective_steps)

    def is_development(self) -> bool:
        return str(self.mode).lower() == "development"

    def path(self, key: str) -> str:
        """Resolve a storage path from the config into an absolute path."""
        raw = getattr(self.storage, key)
        return raw if os.path.isabs(raw) else os.path.join(REPO, raw)


def _strip_comments(obj: Any) -> Any:
    """Drop the _underscore documentation keys so they never reach the code."""
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items()
                if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Settings(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_settings(path: Optional[str] = None) -> Settings:
    candidate = path or os.environ.get("FRESHIQ_DEVICE_CONFIG") or DEFAULT_PATH
    if not os.path.exists(candidate):
        raise FileNotFoundError(
            "device config not found: {}\n"
            "Expected config/freshiq.json. Copy it from the repo or set "
            "FRESHIQ_DEVICE_CONFIG.".format(candidate))

    with open(candidate, "r", encoding="utf-8") as fh:
        data = _strip_comments(json.load(fh))

    s = _to_namespace(data)
    s.source_path = candidate
    _validate(s)
    return s


def _validate(s: Settings) -> None:
    """Fail at startup with a clear message rather than mid-scan."""
    problems = []

    if int(getattr(s.tray, "compartments", 0)) < 1:
        problems.append("tray.compartments must be >= 1")
    if str(s.mode).lower() not in ("production", "development"):
        problems.append("mode must be 'production' or 'development'")
    if str(s.motor.driver).lower() not in ("bridge", "serial", "simulated", "auto"):
        problems.append(
            "motor.driver must be 'bridge' (UNO Q onboard MCU), 'serial' "
            "(external Arduino), 'simulated' or 'auto'")
    if float(s.tray.stabilize_seconds) < 0:
        problems.append("tray.stabilize_seconds must be >= 0")

    # The single most damaging misconfiguration on an UNO Q. /dev/ttyHS1 is the
    # internal MPU-to-MCU UART and arduino-router owns it exclusively; opening
    # it with the serial driver fights the router for the device instead of
    # failing cleanly, which is very hard to diagnose from the symptom.
    port = str(getattr(s.motor, "port", "") or "")
    if "ttyHS" in port:
        problems.append(
            "motor.port is {!r}, but /dev/ttyHS* is the UNO Q's internal "
            "MPU-to-MCU link and belongs to arduino-router. Use "
            "\"driver\": \"bridge\" instead of pointing the serial driver at it."
            .format(port))

    try:
        s.steps_per_compartment()
    except ValueError as exc:
        problems.append(str(exc))

    if problems:
        raise ValueError("Invalid config/freshiq.json:\n  - "
                         + "\n  - ".join(problems))


__all__ = ["Settings", "load_settings", "DEFAULT_PATH", "REPO"]
