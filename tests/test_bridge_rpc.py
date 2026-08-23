"""
Tests for the UNO Q MPU -> MCU transport (hardware/bridge_rpc.py) and the
BridgeTrayMotor that sits on top of it.

They run against a FAKE arduino-router: a socketpair speaking the same
MessagePack-RPC the real service does. That exercises the framing, msgid
matching, the error path and the motor's use of them with no board attached,
and on any OS - AF_UNIX is not available everywhere.

What they CANNOT verify is the far end of the wire: that the sketch really
registers tray_rotate/tray_stop/tray_status, and that the A4988 turns. Only the
hardware proves that, which is what python/tools/check_tray.py is for.
"""

import itertools
import os
import socket
import sys
import threading

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

msgpack = pytest.importorskip("msgpack")

from hardware.bridge_rpc import BridgeError, RouterRpc  # noqa: E402
from hardware.motor import BridgeTrayMotor, MotorError  # noqa: E402


class FakeRouter:
    """MessagePack-RPC over a socketpair, shaped like arduino-router."""

    def __init__(self, handlers):
        self.handlers = handlers
        self.calls = []
        self.moved = {"steps": 0}
        self.client, self._server = socket.socketpair()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._serve, daemon=True)
        self._t.start()

    def _serve(self):
        unpacker = msgpack.Unpacker(raw=False)
        self._server.settimeout(0.2)
        while not self._stop.is_set():
            try:
                chunk = self._server.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                return
            if not chunk:
                return
            unpacker.feed(chunk)
            for msg in unpacker:
                if not isinstance(msg, (list, tuple)) or len(msg) < 3:
                    continue
                if msg[0] == 2:                       # notify
                    continue
                _, msgid, method, params = msg
                self.calls.append((method, list(params)))
                fn = self.handlers.get(method)
                if fn is None:
                    reply = [1, msgid, "no such method: %s" % method, None]
                else:
                    try:
                        reply = [1, msgid, None, fn(*params)]
                    except Exception as exc:          # noqa: BLE001
                        reply = [1, msgid, str(exc), None]
                try:
                    self._server.sendall(msgpack.packb(reply, use_bin_type=True))
                except OSError:
                    return

    def rpc(self):
        """A RouterRpc bound to this fake, bypassing the AF_UNIX connect."""
        # Mirror RouterRpc.__init__ exactly, minus the AF_UNIX connect.
        r = RouterRpc.__new__(RouterRpc)
        r._msgpack = msgpack
        r.socket_path = "<fake>"
        r.timeout = 5.0
        r._sock = self.client
        r._unpacker = msgpack.Unpacker(raw=False, strict_map_key=False)
        r._ids = itertools.count(1)
        r._lock = threading.Lock()
        self.client.settimeout(5.0)
        return r

    def close(self):
        self._stop.set()
        for s in (self.client, self._server):
            try:
                s.close()
            except OSError:
                pass


@pytest.fixture
def router():
    r = None

    def tray_rotate(steps, delay_us=None):
        r.moved["steps"] += steps
        return steps                      # full move completed

    r = FakeRouter({
        "tray_rotate": tray_rotate,
        "tray_stop": lambda: 0,
        "tray_status": lambda: 0,
        "boom": lambda: (_ for _ in ()).throw(RuntimeError("driver fault")),
    })
    yield r
    r.close()


def motor(router, compartments=4, steps=50):
    """BridgeTrayMotor wired to the fake, skipping the real constructor."""
    m = BridgeTrayMotor.__new__(BridgeTrayMotor)
    from hardware.motor import TrayMotor
    TrayMotor.__init__(m, compartments, steps)
    m.socket_path = "<fake>"
    m.port = "<fake>"
    m.step_delay_us = 70000
    m.ack_timeout = 5.0
    m._rpc = router.rpc()
    m._bridge_error = BridgeError
    return m


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #
def test_call_returns_the_mcu_result(router):
    assert router.rpc().call("tray_rotate", 50, 70000) == 50
    assert router.calls[-1] == ("tray_rotate", [50, 70000])


def test_msgids_match_across_repeated_calls(router):
    r = router.rpc()
    for _ in range(5):
        assert r.call("tray_rotate", 50, 70000) == 50
    assert router.moved["steps"] == 250


def test_mcu_error_surfaces_as_bridge_error(router):
    with pytest.raises(BridgeError, match="driver fault"):
        router.rpc().call("boom")


def test_unknown_method_is_reported(router):
    with pytest.raises(BridgeError, match="no such method"):
        router.rpc().call("not_provided")


# --------------------------------------------------------------------------- #
# Motor
# --------------------------------------------------------------------------- #
def test_move_sends_exactly_steps_per_compartment(router):
    m = motor(router)
    m.move_to(2)
    assert router.calls[-1][0] == "tray_rotate"
    assert router.calls[-1][1][0] == 50


def test_move_wraps_forward_only(router):
    m = motor(router)
    assert m.move_to(3) == 2
    assert m.position == 3
    assert m.move_to(2) == 3          # the long way round, never backwards
    assert m.position == 2


def test_full_four_compartment_cycle_is_one_revolution(router):
    m = motor(router)
    for target in (2, 3, 4, 1):
        m.move_to(target)
    assert m.position == 1
    assert router.moved["steps"] == 200        # 4 x 50


def test_short_move_is_an_error_not_a_silent_misalignment():
    """
    If the sketch aborts mid-move it returns fewer steps than asked. Accepting
    that quietly would leave the software believing an alignment the tray does
    not have, and every later result would carry the wrong compartment number.
    """
    r = FakeRouter({"tray_rotate": lambda steps, d=None: steps // 2,
                    "tray_stop": lambda: 0})
    try:
        m = motor(r)
        with pytest.raises(MotorError, match="no longer"):
            m.move_to(2)
    finally:
        r.close()


def test_mcu_failure_stops_the_motor(router):
    """A transport fault must still leave the driver de-energised."""
    r = FakeRouter({"tray_stop": lambda: 0})       # tray_rotate NOT provided
    try:
        m = motor(r)
        with pytest.raises(MotorError):
            m.move_to(2)
        assert m.moving is False
        assert any(c[0] == "tray_stop" for c in r.calls), \
            "stop() must be called when a move fails"
    finally:
        r.close()


def test_ping_maps_status_to_ready_busy(router):
    assert motor(router).ping() == "READY"
    r = FakeRouter({"tray_status": lambda: 1, "tray_stop": lambda: 0})
    try:
        assert motor(r).ping() == "BUSY"
    finally:
        r.close()
