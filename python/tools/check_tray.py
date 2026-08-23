#!/usr/bin/env python3
"""
Verify the Linux -> MCU tray link BEFORE starting FreshIQ.

Run this whenever the dashboard says "[tray] UNAVAILABLE". It checks one thing
at a time and stops at the first failure, so the output names the actual fault
instead of leaving you guessing which of five things is wrong.

    python3 tools/check_tray.py            # checks only
    python3 tools/check_tray.py --rotate   # also turns the tray one full lap,
                                           # compartment by compartment

The --rotate pass is the real end-to-end proof: it exercises exactly the path
a scan uses (move one compartment -> stop -> move to the next -> repeat).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

OK, BAD, INFO = "  ok  ", " FAIL ", "      "


def say(tag: str, msg: str) -> None:
    print("[{}] {}".format(tag, msg), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Check the FreshIQ tray link.")
    ap.add_argument("--config", default=None, help="path to config/freshiq.json")
    ap.add_argument("--rotate", action="store_true",
                    help="physically turn the tray through every compartment")
    args = ap.parse_args()

    # ---- 1. config -------------------------------------------------------
    from config import load_settings
    settings = load_settings(args.config)
    steps = settings.steps_per_compartment()
    say(OK, "config {}".format(settings.source_path))
    say(INFO, "mode={}  driver={}  compartments={}  steps/compartment={}".format(
        settings.mode, settings.motor.driver,
        settings.tray.compartments, steps))
    say(INFO, "speed: {:.2f} s per compartment ({} us per step)".format(
        settings.seconds_per_compartment(), settings.step_delay_us()))
    say(INFO, "each move should turn the tray {:.1f} degrees".format(
        360.0 / int(settings.tray.compartments)))

    if settings.is_development() or str(settings.motor.driver).lower() == "simulated":
        say(INFO, "this config SIMULATES the motor - nothing physical will move.")

    # ---- 2. router socket -------------------------------------------------
    sock = str(getattr(settings.motor, "rpc_socket", "")
               or "/var/run/arduino-router.sock")
    if str(settings.motor.driver).lower() in ("bridge", "auto"):
        if os.path.exists(sock):
            say(OK, "arduino-router socket present: {}".format(sock))
        else:
            say(BAD, "no router socket at {}".format(sock))
            say(INFO, "  systemctl status arduino-router")
            return 1

    # ---- 3. build the motor (this is what the server does) ----------------
    from hardware import MotorError, build_motor
    try:
        motor = build_motor(settings)
    except MotorError as exc:
        say(BAD, "could not open the tray motor:")
        for line in str(exc).splitlines():
            say(INFO, "  " + line)
        return 1
    say(OK, "motor driver: {}".format(motor.__class__.__name__))

    try:
        # ---- 4. does the sketch answer? -----------------------------------
        ping = getattr(motor, "ping", None)
        if ping is not None:
            try:
                say(OK, "MCU answered tray_status: {}".format(ping()))
            except MotorError as exc:
                say(BAD, "the MCU did not answer: {}".format(exc))
                return 1
        else:
            say(INFO, "this driver has no ping (simulated)")

        # ---- 5. move it ---------------------------------------------------
        if not args.rotate:
            say(INFO, "link is good. Re-run with --rotate to turn the tray.")
            return 0

        n = int(settings.tray.compartments)
        say(INFO, "rotating one full lap: {} compartments x {} steps".format(n, steps))
        for i in range(1, n + 1):
            target = (i % n) + 1
            started = time.perf_counter()
            try:
                motor.move_to(target)
            except MotorError as exc:
                say(BAD, "move to compartment {} failed:".format(target))
                for line in str(exc).splitlines():
                    say(INFO, "  " + line)
                return 1
            elapsed = time.perf_counter() - started
            say(OK, "moved to compartment {} in {:.2f} s".format(target, elapsed))
            time.sleep(float(settings.tray.stabilize_seconds))
        say(OK, "full lap complete - the tray is back where it started.")
        say(INFO, "If a move took about {:.1f} s but the tray barely turned, the".format(
            settings.seconds_per_compartment()))
        say(INFO, "  distance is wrong, not the speed: your A4988 MS1/MS2/MS3 jumpers")
        say(INFO, "  are set to microstep while tray.microsteps says {}. Match them."
            .format(settings.tray.microsteps))
        return 0
    finally:
        try:
            motor.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    sys.exit(main())
