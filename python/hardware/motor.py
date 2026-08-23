"""
Tray motor control.

The UNO Q is the HIGH-LEVEL controller: it says "go to compartment 3". The MCU
does the real-time stepping. That split is deliberate - Linux is not a real-time
OS, and a scheduler hiccup mid-step on a Python-driven GPIO loop produces
missed steps and a tray that silently drifts out of alignment.

THREE DRIVERS
-------------
BridgeTrayMotor     the Arduino UNO Q's ONBOARD MCU, reached through the
                    arduino-router RPC service. This is the correct driver for
                    this board and the default in config/freshiq.json.

SerialTrayMotor     a SEPARATE Arduino wired over USB, on /dev/ttyACM* or
                    /dev/ttyUSB*. Kept because it still works, but it is not
                    how the UNO Q reaches its own microcontroller: that MCU is
                    on the same PCB and never enumerates as a USB serial
                    device. Expecting /dev/ttyACM0 on an UNO Q is the single
                    most common way to end up with "[tray] UNAVAILABLE".

SimulatedTrayMotor  no hardware, movement faked with sleeps.

BRIDGE PROTOCOL (see firmware/freshiq_tray/freshiq_tray.ino)

    tray_rotate(steps) -> int   turn `steps` steps, RETURN ONLY WHEN STOPPED,
                                answering with the steps actually taken
    tray_stop()        -> int   de-energise the A4988 now
    tray_status()      -> int   0 = READY, 1 = BUSY

SERIAL PROTOCOL (external Arduino only)

    ->  R<steps>\n   advance this many motor steps, then reply
    ->  S\n          stop now and disable the driver
    ->  ?\n          status request
    <-  OK / READY / BUSY / ERR <text>

Both protocols keep the same guarantee, which backend/scan.py depends on: the
call does not return until the tray is physically stationary. Otherwise
STABILIZING starts while the platter is still turning and the camera
photographs a blur.

NO POSITION FEEDBACK EXISTS. There is no home switch, encoder or index sensor
on this rig, so absolute tray position is not physically knowable. We track a
LOGICAL index instead: on startup the tray is assumed to be at compartment 1,
and the operator aligns it once. Every move is relative from there. If the
motor skips steps, software cannot detect it - that is a hardware limitation,
documented rather than papered over.
"""

from __future__ import annotations

import threading
import time
from typing import Optional


class MotorError(RuntimeError):
    """Raised when the tray cannot be moved or the MCU stops answering."""


class TrayMotor:
    """Common interface. `position` is 1-based: compartments are 1..N."""

    def __init__(self, compartments: int, steps_per_compartment: int):
        if compartments < 1:
            raise ValueError("compartments must be >= 1")
        self.compartments = int(compartments)
        self.steps_per_compartment = int(steps_per_compartment)
        self.position = 1
        self.moving = False
        self.last_error: Optional[str] = None
        self._lock = threading.Lock()

    # ---- to implement in subclasses ---------------------------------------
    def _advance_one(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    # ---- shared logic ------------------------------------------------------
    def move_to(self, target: int) -> int:
        """
        Rotate forward to `target` (1-based). Always turns the same direction,
        which keeps any mechanical backlash consistent between compartments.
        Returns the number of compartment steps taken.
        """
        if not 1 <= target <= self.compartments:
            raise MotorError(
                "invalid compartment {} (valid: 1..{})".format(target, self.compartments))

        with self._lock:
            delta = (target - self.position) % self.compartments
            if delta == 0:
                return 0
            self.moving = True
            try:
                for _ in range(delta):
                    self._advance_one()
                    self.position = (self.position % self.compartments) + 1
            except Exception as exc:
                self.last_error = str(exc)
                # Never leave the driver energised on the way out.
                try:
                    self.stop()
                except Exception:  # noqa: BLE001
                    pass
                raise
            finally:
                self.moving = False
            return delta

    def status(self) -> dict:
        return {
            "driver": self.__class__.__name__,
            "position": self.position,
            "compartments": self.compartments,
            "steps_per_compartment": self.steps_per_compartment,
            "step_delay_us": getattr(self, "step_delay_us", None),
            "moving": self.moving,
            "connected": self.connected(),
            "last_error": self.last_error,
        }

    def connected(self) -> bool:
        return True

    def set_position(self, position: int) -> None:
        """Tell the software where the tray physically is, after manual alignment."""
        if not 1 <= position <= self.compartments:
            raise MotorError("invalid compartment {}".format(position))
        self.position = int(position)


class SimulatedTrayMotor(TrayMotor):
    """
    Development driver. Same timing shape as the real thing so the scan state
    machine, the dashboard and the delays all behave realistically with no
    hardware attached.
    """

    def __init__(self, compartments: int, steps_per_compartment: int,
                 move_seconds: float = 3.5):
        super().__init__(compartments, steps_per_compartment)
        self.move_seconds = move_seconds
        self._stopped = False

    def _advance_one(self) -> None:
        self._stopped = False
        time.sleep(self.move_seconds)

    def stop(self) -> None:
        self._stopped = True


class SerialTrayMotor(TrayMotor):
    """Real driver: talks to the MCU sketch over a serial port."""

    def __init__(self, compartments: int, steps_per_compartment: int,
                 port: str, baud: int = 115200,
                 ack_timeout: float = 15.0, boot_wait: float = 2.5):
        super().__init__(compartments, steps_per_compartment)
        self.port = port
        self.baud = baud
        self.ack_timeout = ack_timeout
        self._serial = None

        try:
            import serial  # type: ignore
        except ImportError as exc:
            raise MotorError(
                "pyserial is not installed, so the tray cannot be driven.\n"
                "  pip3 install --break-system-packages pyserial\n"
                "Or set motor.driver to 'simulated' in config/freshiq.json."
            ) from exc

        try:
            self._serial = serial.Serial(port, baud, timeout=ack_timeout)
        except Exception as exc:  # noqa: BLE001
            raise MotorError(
                "could not open motor serial port {!r}: {}\n"
                "Check: ls /dev/ttyACM* /dev/ttyUSB*   and that the user is in "
                "the 'dialout' group.".format(port, exc)) from exc

        # Most Arduino-compatible boards reset when the port is opened. Talking
        # to a booting MCU gets you silence and a confusing timeout, so wait.
        time.sleep(boot_wait)
        self._serial.reset_input_buffer()

    def connected(self) -> bool:
        return self._serial is not None and getattr(self._serial, "is_open", False)

    def _command(self, line: str, expect: str = "OK") -> str:
        if not self.connected():
            raise MotorError("motor serial port is not open")
        self._serial.reset_input_buffer()
        self._serial.write((line + "\n").encode("ascii"))
        self._serial.flush()

        deadline = time.time() + self.ack_timeout
        while time.time() < deadline:
            raw = self._serial.readline()
            if not raw:
                continue
            reply = raw.decode("ascii", errors="replace").strip()
            if not reply:
                continue
            if reply.startswith("ERR"):
                raise MotorError("MCU reported: {}".format(reply))
            if reply == expect or reply in ("READY", "BUSY"):
                return reply
            # Ignore anything else (boot banners, debug prints).
        raise MotorError(
            "no '{}' from the MCU within {:.0f}s on {}. The tray may still be "
            "moving - stopping it.".format(expect, self.ack_timeout, self.port))

    def _advance_one(self) -> None:
        # WAITS for OK. The old fire-and-forget 'write R then sleep 0.8s'
        # could photograph a still-moving tray whenever a move ran long.
        self._command("R{}".format(self.steps_per_compartment), expect="OK")

    def stop(self) -> None:
        try:
            if self.connected():
                self._serial.write(b"S\n")
                self._serial.flush()
        except Exception as exc:  # noqa: BLE001
            self.last_error = "stop failed: {}".format(exc)

    def ping(self) -> str:
        return self._command("?", expect="READY")

    def close(self) -> None:
        try:
            self.stop()
        finally:
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:  # noqa: BLE001
                    pass
                self._serial = None


class BridgeTrayMotor(TrayMotor):
    """
    Real driver for the Arduino UNO Q's ONBOARD MCU.

    There is no /dev/ttyACM0 on this board and there never will be: the MCU is
    not a USB peripheral, it is the other half of the same PCB. It is reached
    through the arduino-router service, which owns the internal UART
    (/dev/ttyHS1) and forwards MessagePack-RPC calls to the sketch. See
    hardware/bridge_rpc.py for the transport and firmware/freshiq_tray/ for the
    sketch that registers these three methods.

    RPC CONTRACT (must match firmware/freshiq_tray/freshiq_tray.ino)

        tray_rotate(steps) -> int   turn `steps` steps, RETURN ONLY WHEN STOPPED,
                                    answering with the number of steps actually
                                    taken (less than asked = aborted by a stop)
        tray_stop()        -> int   de-energise the A4988 now, return 0
        tray_status()      -> int   0 = READY, 1 = BUSY

    The blocking return is the same guarantee the old serial "OK" gave, and
    backend/scan.py still depends on it: move_to() must not come back until the
    tray is physically stationary, or STABILIZING starts while it is still
    turning and the camera photographs a blur.
    """

    def __init__(self, compartments: int, steps_per_compartment: int,
                 socket_path: str = "/var/run/arduino-router.sock",
                 ack_timeout: float = 15.0, step_delay_us: int = 70000):
        super().__init__(compartments, steps_per_compartment)
        from .bridge_rpc import BridgeError, build_rpc

        self.socket_path = socket_path
        self.step_delay_us = int(step_delay_us)
        # A move that takes longer than the ack timeout would be reported as a
        # dead MCU while the tray is still turning perfectly well, so the
        # timeout has to follow the configured speed rather than fight it.
        move_seconds = self.step_delay_us * self.steps_per_compartment / 1_000_000
        self.ack_timeout = max(float(ack_timeout), move_seconds * 2 + 5.0)
        # Named `port` so the existing status/logging code, which prints
        # getattr(motor, "port", ...), keeps working unchanged.
        self.port = socket_path
        self._rpc = None
        self._bridge_error = BridgeError

        # Callers (and tools/check_tray.py) contract on MotorError, so a
        # transport failure must not escape as a BridgeError.
        try:
            self._rpc = build_rpc(socket_path=socket_path, timeout=self.ack_timeout)
        except BridgeError as exc:
            raise MotorError(str(exc)) from exc

        # Fail loudly HERE rather than halfway through the first scan. If the
        # sketch is not flashed, the router answers "no such method" and the
        # message says exactly that.
        try:
            self.ping()
        except (MotorError, BridgeError) as exc:
            try:
                self._rpc.close()
            except Exception:  # noqa: BLE001
                pass
            self._rpc = None
            raise MotorError(
                "reached the Arduino router but the tray sketch did not answer: {}\n"
                "sketch/sketch.ino is not running on the onboard MCU. Deploy both\n"
                "halves of the app together:\n"
                "  arduino-app-cli app start ~/ArduinoApps/FreshIQ\n"
                "(or press Run in Arduino App Lab)."
                .format(exc)) from exc

    def connected(self) -> bool:
        return self._rpc is not None and self._rpc.connected()

    def _call(self, method: str, *args):
        if self._rpc is None:
            raise MotorError("the bridge to the MCU is closed")
        try:
            return self._rpc.call(method, *args, timeout=self.ack_timeout)
        except self._bridge_error as exc:
            raise MotorError(str(exc)) from exc

    def _advance_one(self) -> None:
        steps = int(self.steps_per_compartment)
        done = self._call("tray_rotate", steps, self.step_delay_us)
        # A short move means the sketch aborted on a stop request. Silently
        # accepting it would leave self.position claiming an alignment the tray
        # does not actually have.
        if isinstance(done, int) and done < steps:
            raise MotorError(
                "the MCU stopped after {} of {} steps - the tray is no longer "
                "aligned. Re-align it and POST /api/motor/position.".format(done, steps))

    def stop(self) -> None:
        try:
            self._call("tray_stop")
        except MotorError as exc:
            self.last_error = "stop failed: {}".format(exc)

    def ping(self) -> str:
        """Returns 'READY' or 'BUSY'. Raises if the sketch is not answering."""
        return "BUSY" if int(self._call("tray_status") or 0) else "READY"

    def close(self) -> None:
        try:
            if self.connected():
                self.stop()
        finally:
            if self._rpc is not None:
                try:
                    self._rpc.close()
                except Exception:  # noqa: BLE001
                    pass
                self._rpc = None


def find_serial_port() -> Optional[str]:
    """
    Best guess at an EXTERNAL MCU's port.

    Only relevant when motor.driver is "serial", i.e. a separate Arduino wired
    over USB. The UNO Q's own MCU never appears here - it is not a USB device -
    which is exactly why driver "bridge" exists.
    """
    import glob
    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        found = sorted(glob.glob(pattern))
        if found:
            return found[0]
    return None


def _on_uno_q(socket_path: str) -> bool:
    """True when the Arduino router is present, i.e. we are on an UNO Q."""
    import os
    return os.path.exists(socket_path)


def build_motor(settings) -> TrayMotor:
    """
    Construct the motor named by config/freshiq.json.

    driver values:
        "bridge"     UNO Q onboard MCU via arduino-router RPC   <- this board
        "serial"     a separate Arduino on /dev/ttyACM* or ttyUSB*
        "simulated"  no hardware
        "auto"       bridge if the router socket exists, else serial
    """
    compartments = int(settings.tray.compartments)
    steps = settings.steps_per_compartment()
    driver = str(settings.motor.driver).lower()

    move_seconds = float(settings.seconds_per_compartment())
    step_delay_us = int(settings.step_delay_us())

    if settings.is_development() or driver == "simulated":
        return SimulatedTrayMotor(compartments, steps, move_seconds=move_seconds)

    socket_path = str(getattr(settings.motor, "rpc_socket", "")
                      or "/var/run/arduino-router.sock")
    ack_timeout = float(settings.motor.ack_timeout_seconds)

    if driver == "auto":
        driver = "bridge" if _on_uno_q(socket_path) else "serial"

    if driver == "bridge":
        return BridgeTrayMotor(
            compartments, steps,
            socket_path=socket_path,
            ack_timeout=ack_timeout,
            step_delay_us=step_delay_us,
        )

    port = str(settings.motor.port or "").strip()
    if port.lower() in ("", "auto"):
        port = find_serial_port()
        if not port:
            hint = ""
            if _on_uno_q(socket_path):
                hint = ("\nThis board IS an Arduino UNO Q: its MCU is on the same "
                        "PCB and never enumerates as /dev/ttyACM*. Set "
                        "motor.driver to \"bridge\" in config/freshiq.json.")
            raise MotorError(
                "no serial port found (looked for /dev/ttyACM* and /dev/ttyUSB*)."
                + hint)

    return SerialTrayMotor(
        compartments, steps,
        port=port,
        baud=int(settings.motor.baud),
        ack_timeout=ack_timeout,
    )
