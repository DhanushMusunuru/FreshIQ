"""
MPU -> MCU transport for the Arduino UNO Q.

WHY THIS FILE EXISTS
--------------------
The UNO Q is not a Linux box with an Arduino plugged into it. It is one board
with two processors:

    Qualcomm QRB2210 (MPU)  Debian, Python, OpenCV, TFLite  <- FreshIQ runs here
    STM32U585 (MCU)         Zephyr + Arduino sketch         <- the stepper lives here

They are wired together internally by a UART. On the Linux side that UART is
/dev/ttyHS1, and it is ALREADY OPEN: the `arduino-router` system service holds
it and multiplexes it. Arduino's own documentation is explicit that ttyHS1 is
reserved and must not be opened by application code. Two readers on one tty
steal bytes from each other, so "just point pyserial at /dev/ttyHS1" produces
a tray that works until the router happens to consume the 'OK' first.

The supported path is the router itself. It listens on a Unix socket and speaks
MessagePack-RPC. The sketch registers named functions with Bridge.provide();
the router forwards a call for that name to the MCU and returns the sketch's
return value. That is the mechanism this file implements.

    FreshIQ (Python)  ->  /var/run/arduino-router.sock  (MessagePack-RPC)
                      ->  arduino-router               (owns /dev/ttyHS1)
                      ->  /dev/ttyHS1                  (internal UART)
                      ->  Arduino_RouterBridge         (in the sketch)
                      ->  tray_rotate() / tray_stop()  (A4988 -> NEMA 17)

TWO TRANSPORTS, ONE INTERFACE
-----------------------------
RouterRpc   speaks the wire protocol directly over the Unix socket. This is the
            default because it works from ANY interpreter - including a plain
            .venv - and because it lets us set our own per-call timeout, which
            a blocking tray move needs.

AppUtilsRpc uses Arduino's own `arduino.app_utils.Bridge`. That package is
            installed system-wide on the UNO Q, so a .venv created without
            --system-site-packages cannot see it. Used as a fallback when
            msgpack is missing.

WIRE PROTOCOL (MessagePack-RPC, as implemented by arduino-router)
    request   [0, msgid, "method", [params...]]
    response  [1, msgid, error, result]
    notify    [2, "method", [params...]]
"""

from __future__ import annotations

import itertools
import os
import socket
import threading
import time
from typing import Any, Optional

DEFAULT_SOCKET = "/var/run/arduino-router.sock"

_TYPE_REQUEST = 0
_TYPE_RESPONSE = 1


class BridgeError(RuntimeError):
    """Raised when the MPU cannot reach the MCU through the Arduino router."""


class RouterRpc:
    """MessagePack-RPC client for arduino-router's Unix socket."""

    name = "router-socket"

    def __init__(self, socket_path: str = DEFAULT_SOCKET, timeout: float = 15.0):
        try:
            import msgpack  # type: ignore
        except ImportError as exc:
            raise BridgeError(
                "the 'msgpack' package is required to talk to the Arduino router.\n"
                "  pip install msgpack\n"
                "(it is in requirements.txt; if you use a .venv, install it there)"
            ) from exc

        self._msgpack = msgpack
        self.socket_path = socket_path
        self.timeout = float(timeout)
        self._sock: Optional[socket.socket] = None
        self._unpacker = None
        self._ids = itertools.count(1)
        # One in-flight call at a time. The scan thread and the dashboard's
        # /api/motor/status can both land here; interleaving two request/reply
        # exchanges on one socket mixes up the replies.
        self._lock = threading.Lock()
        self.connect()

    # ---- connection --------------------------------------------------------
    def connect(self) -> None:
        if not os.path.exists(self.socket_path):
            raise BridgeError(
                "the Arduino router socket {} does not exist.\n"
                "The router service is what talks to the onboard MCU. Check it:\n"
                "  systemctl status arduino-router".format(self.socket_path))

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect(self.socket_path)
        except OSError as exc:
            sock.close()
            raise BridgeError(
                "could not connect to the Arduino router at {}: {}\n"
                "  systemctl status arduino-router\n"
                "If it is running but refuses the connection, your user may not "
                "have permission on the socket - try 'sudo' or add your user to "
                "the socket's group.".format(self.socket_path, exc)) from exc

        self._sock = sock
        self._unpacker = self._msgpack.Unpacker(raw=False, strict_map_key=False)

    def connected(self) -> bool:
        return self._sock is not None

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None
                self._unpacker = None

    def _drop(self) -> None:
        """Tear the socket down after a protocol error so the next call reconnects."""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None
        self._unpacker = None

    # ---- rpc ---------------------------------------------------------------
    def call(self, method: str, *args: Any, timeout: Optional[float] = None) -> Any:
        """
        Invoke `method` on the MCU and return its value.

        Blocks until the sketch's handler RETURNS, which is the whole point: a
        tray move is finished when this returns, so the camera can never
        photograph a moving tray.
        """
        with self._lock:
            if self._sock is None:
                self.connect()
            deadline = time.time() + (self.timeout if timeout is None else float(timeout))
            msgid = next(self._ids) & 0x7FFFFFFF

            packed = self._msgpack.packb(
                [_TYPE_REQUEST, msgid, method, list(args)], use_bin_type=True)
            try:
                self._sock.sendall(packed)
            except OSError as exc:
                self._drop()
                raise BridgeError(
                    "lost the connection to arduino-router while sending {!r}: {}"
                    .format(method, exc)) from exc

            return self._await_reply(method, msgid, deadline)

    def _await_reply(self, method: str, msgid: int, deadline: float) -> Any:
        while True:
            # Anything already buffered may complete the reply without a read.
            for msg in self._unpacker:
                value = self._match(method, msgid, msg)
                if value is not _NO_MATCH:
                    return value

            remaining = deadline - time.time()
            if remaining <= 0:
                self._drop()
                raise BridgeError(
                    "the MCU did not answer {!r} within the timeout. The sketch "
                    "may not be running, or it does not provide that method. "
                    "Check that firmware/freshiq_tray/ is flashed to the onboard "
                    "MCU.".format(method))

            self._sock.settimeout(remaining)
            try:
                chunk = self._sock.recv(4096)
            except socket.timeout:
                continue
            except OSError as exc:
                self._drop()
                raise BridgeError(
                    "lost the connection to arduino-router while waiting for {!r}: {}"
                    .format(method, exc)) from exc

            if not chunk:
                self._drop()
                raise BridgeError(
                    "arduino-router closed the connection while waiting for {!r}. "
                    "Check: systemctl status arduino-router".format(method))
            self._unpacker.feed(chunk)

    def _match(self, method: str, msgid: int, msg: Any) -> Any:
        """Return the result for our msgid, or _NO_MATCH for unrelated traffic."""
        if not isinstance(msg, (list, tuple)) or len(msg) < 4:
            return _NO_MATCH                      # notification or noise
        if msg[0] != _TYPE_RESPONSE or msg[1] != msgid:
            return _NO_MATCH                      # someone else's reply
        error, result = msg[2], msg[3]
        if error is not None:
            raise BridgeError("the MCU reported an error for {!r}: {}"
                              .format(method, error))
        return result


class _NoMatch:
    """Sentinel: `None` is a legitimate RPC result, so it cannot mean 'no reply'."""


_NO_MATCH = _NoMatch()


class AppUtilsRpc:
    """Fallback transport using Arduino's own preinstalled Python bridge."""

    name = "arduino.app_utils"

    def __init__(self, timeout: float = 15.0):
        try:
            from arduino.app_utils import Bridge  # type: ignore
        except ImportError as exc:
            raise BridgeError(
                "neither 'msgpack' nor 'arduino.app_utils' is importable, so "
                "there is no way to reach the onboard MCU.\n"
                "Easiest fix:  pip install msgpack\n"
                "(arduino.app_utils is installed system-wide on the UNO Q and is "
                "invisible to a .venv created without --system-site-packages.)"
            ) from exc
        self._bridge = Bridge
        self.timeout = float(timeout)

    def connected(self) -> bool:
        return True

    def call(self, method: str, *args: Any, timeout: Optional[float] = None) -> Any:
        try:
            return self._bridge.call(method, *args)
        except Exception as exc:  # noqa: BLE001
            raise BridgeError("Bridge.call({!r}) failed: {}".format(method, exc)) from exc

    def close(self) -> None:
        pass


def build_rpc(socket_path: str = DEFAULT_SOCKET, timeout: float = 15.0):
    """
    Pick a transport. The raw socket is preferred because it works inside a
    .venv and honours our timeout; app_utils is the fallback.
    """
    try:
        return RouterRpc(socket_path=socket_path, timeout=timeout)
    except BridgeError as socket_exc:
        try:
            return AppUtilsRpc(timeout=timeout)
        except BridgeError:
            # Report the socket failure: it is the one the user can act on.
            raise socket_exc


__all__ = ["BridgeError", "RouterRpc", "AppUtilsRpc", "build_rpc", "DEFAULT_SOCKET"]
