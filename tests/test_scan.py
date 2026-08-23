"""
Tests for the tray/scan layer.

None of these need hardware: the simulated motor and a stub camera stand in for
the rig, which is the whole point of having a hardware abstraction.
"""

import json
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from backend import scan as scan_module            # noqa: E402
from backend.scan import ScanController, normalise_result  # noqa: E402
from config import load_settings                   # noqa: E402
from hardware.motor import (                       # noqa: E402
    MotorError, SimulatedTrayMotor, TrayMotor,
)
from hardware.lighting import Lighting, NoLighting  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYDIR = os.path.join(REPO, "python")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_shipped_config_loads_and_is_valid():
    s = load_settings()
    assert s.tray.compartments >= 1
    assert s.mode in ("production", "development")


def test_steps_per_compartment_is_derived_not_hardcoded():
    s = load_settings()
    expected = (s.tray.steps_per_revolution * s.tray.microsteps) // s.tray.compartments
    assert s.steps_per_compartment() == expected


def test_shipped_config_matches_the_printed_tray():
    """config compartments must match the CAD, or the tray stops at the walls."""
    s = load_settings()
    params = os.path.join(REPO, "cad", "freshiq_cad", "params.py")
    if not os.path.exists(params):
        pytest.skip("CAD not present")
    ns = {}
    exec(compile(open(params, encoding="utf-8").read(), params, "exec"), ns)
    assert s.tray.compartments == ns["BUILD"]["n_compartments"]


def test_indivisible_microstepping_is_rejected(tmp_path):
    """5 compartments at 200 steps is 40.0 - fine. 3 is 66.67 and would drift."""
    cfg = json.load(open(os.path.join(PYDIR, "config", "freshiq.json"), encoding="utf-8"))
    cfg["tray"]["compartments"] = 3          # 200/3 is not an integer
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="drift"):
        load_settings(str(p))


def test_ttyHS_port_is_rejected_with_an_explanation(tmp_path):
    """
    /dev/ttyHS1 is the UNO Q's internal MPU-MCU link, owned by arduino-router.
    Pointing the serial driver at it is the mistake the old error message
    actively encouraged, so config now refuses it by name.
    """
    cfg = json.load(open(os.path.join(PYDIR, "config", "freshiq.json"), encoding="utf-8"))
    cfg["motor"]["driver"] = "serial"
    cfg["motor"]["port"] = "/dev/ttyHS1"
    p = tmp_path / "hs.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="arduino-router"):
        load_settings(str(p))


def test_driver_names_are_validated(tmp_path):
    cfg = json.load(open(os.path.join(PYDIR, "config", "freshiq.json"), encoding="utf-8"))
    cfg["motor"]["driver"] = "carrier-pigeon"
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="motor.driver"):
        load_settings(str(p))


def test_underscore_comment_keys_never_reach_the_code():
    s = load_settings()
    assert not hasattr(s.tray, "_compartments")
    assert not hasattr(s, "_mode")


# --------------------------------------------------------------------------- #
# Motor
# --------------------------------------------------------------------------- #
def test_move_to_wraps_forward_around_the_tray():
    m = SimulatedTrayMotor(4, 50, move_seconds=0)
    assert m.position == 1
    assert m.move_to(3) == 2
    assert m.position == 3
    # 3 -> 2 must go the long way round (3 steps), never backwards.
    assert m.move_to(2) == 3
    assert m.position == 2


def test_move_to_same_compartment_does_nothing():
    m = SimulatedTrayMotor(4, 50, move_seconds=0)
    assert m.move_to(1) == 0


def test_invalid_compartment_is_rejected():
    m = SimulatedTrayMotor(4, 50, move_seconds=0)
    for bad in (0, 5, -1):
        with pytest.raises(MotorError):
            m.move_to(bad)


def test_motor_is_stopped_when_a_move_fails():
    """A stepper must never be left energised by an exception."""
    class Exploding(TrayMotor):
        def __init__(self):
            super().__init__(4, 50)
            self.stopped = False

        def _advance_one(self):
            raise RuntimeError("driver fault")

        def stop(self):
            self.stopped = True

    m = Exploding()
    with pytest.raises(RuntimeError):
        m.move_to(3)
    assert m.stopped, "stop() must be called when a move raises"
    assert m.moving is False


def test_position_can_be_corrected_after_manual_alignment():
    m = SimulatedTrayMotor(4, 50, move_seconds=0)
    m.set_position(3)
    assert m.position == 3
    with pytest.raises(MotorError):
        m.set_position(99)


# --------------------------------------------------------------------------- #
# Lighting
# --------------------------------------------------------------------------- #
def test_hardwired_lighting_reports_uncontrollable_and_refuses_dimming():
    l = Lighting()
    assert l.status()["controllable"] is False
    l.on(); l.off()                     # no-ops, must not raise
    with pytest.raises(NotImplementedError):
        l.set_brightness(0.5)


def test_missing_lighting_warns_about_colour():
    assert "warning" in NoLighting().status()


# --------------------------------------------------------------------------- #
# Result contract
# --------------------------------------------------------------------------- #
def test_normalise_result_matches_the_shared_contract():
    agg = {
        "status": "ok", "food": "tomato", "condition": "fresh",
        "confidence": {"fresh": 96, "spoiled": 4},
        "evidence": [], "image_quality": "good",
        "views_used": 1, "views_total": 1,
        "views": [{"status": "ok", "food": "tomato", "food_confidence": 0.97}],
    }
    r = normalise_result(agg, compartment=2, timestamp="T")
    assert r["compartment"] == 2
    assert r["food"] == "tomato"
    assert r["food_confidence"] == 0.97
    assert r["condition"] == "fresh"
    assert r["condition_confidence"] == 0.96      # 96% -> 0.96
    assert r["timestamp"] == "T"


def test_normalise_result_survives_a_rejected_image():
    agg = {"status": "invalid_image", "reason": "blurred", "food": "unknown",
           "condition": None, "confidence": {}, "views": []}
    r = normalise_result(agg, 1, "T")
    assert r["status"] == "invalid_image"
    assert r["reason"] == "blurred"
    assert r["condition_confidence"] == 0.0


# --------------------------------------------------------------------------- #
# Scan controller
# --------------------------------------------------------------------------- #
class StubCamera:
    def __init__(self, frame=None, fail_after=None):
        self.frame = frame if frame is not None else np.zeros((64, 64, 3), np.uint8)
        self.calls = 0
        self.fail_after = fail_after

    def snapshot(self):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            return None
        return self.frame


class StubHistory:
    def __init__(self):
        self.records = []

    def append(self, r):
        self.records.append(r)

    def all(self):
        return list(self.records)


def fake_inspect(frames, food_hint=None):
    return {
        "status": "ok", "food": food_hint or "tomato", "condition": "fresh",
        "confidence": {"fresh": 90, "spoiled": 10},
        "evidence": [], "image_quality": "good",
        "views_used": len(frames), "views_total": len(frames),
        "views": [{"status": "ok", "food": food_hint or "tomato",
                   "food_confidence": 0.9}],
    }


def build_controller(tmp_path, compartments=4, camera=None, motor=None,
                     inspect_fn=fake_inspect, save_images=False):
    s = load_settings()
    s.tray.compartments = compartments
    s.tray.stabilize_seconds = 0.0
    s.camera.views_per_compartment = 1
    s.camera.inter_view_delay_seconds = 0.0
    s.storage.save_images = save_images
    s.storage.image_dir = str(tmp_path / "img")
    return ScanController(
        settings=s,
        camera=camera or StubCamera(),
        motor=motor or SimulatedTrayMotor(compartments, 50, move_seconds=0),
        lighting=Lighting(),
        history=StubHistory(),
        inspect_fn=inspect_fn,
        log=lambda m: None,
    )


def wait_done(c, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if not c.is_running():
            return c.status()
        time.sleep(0.02)
    raise AssertionError("scan did not finish within {}s".format(timeout))


def test_end_to_end_scan_visits_every_compartment_once(tmp_path):
    """START -> MOVE -> STOP -> CAPTURE -> AI -> RESULT -> NEXT -> COMPLETE."""
    c = build_controller(tmp_path, compartments=4)
    c.start()
    st = wait_done(c)

    assert st["state"] == scan_module.COMPLETE
    assert st["completed"] == 4
    assert [r["compartment"] for r in st["results"]] == [1, 2, 3, 4]
    assert all(r["status"] == "ok" for r in st["results"])
    assert len(c.history.records) == 4


def test_compartment_number_comes_from_the_controller_not_the_ai(tmp_path):
    """The AI is never asked where the tray is - it only sees a picture."""
    def lying_inspect(frames, food_hint=None):
        out = fake_inspect(frames, food_hint)
        out["compartment"] = 999          # must be ignored
        return out

    c = build_controller(tmp_path, compartments=3, inspect_fn=lying_inspect)
    c.start()
    st = wait_done(c)
    assert [r["compartment"] for r in st["results"]] == [1, 2, 3]


def test_tray_is_stationary_before_every_capture(tmp_path):
    """Capturing mid-rotation is the one thing that must never happen."""
    events = []

    class RecordingMotor(SimulatedTrayMotor):
        def _advance_one(self):
            events.append("move")
            super()._advance_one()

    class RecordingCamera(StubCamera):
        def snapshot(self):
            events.append("capture")
            return super().snapshot()

    motor = RecordingMotor(3, 50, move_seconds=0)
    c = build_controller(tmp_path, compartments=3,
                         camera=RecordingCamera(), motor=motor)
    c.start()
    wait_done(c)

    # Every capture must be preceded by a completed move, never interleaved.
    assert "capture" in events
    for i, e in enumerate(events):
        if e == "capture":
            assert events[i - 1] in ("move", "capture"), \
                "capture happened without a completed move before it"
    assert motor.moving is False


def test_scan_subset_of_compartments(tmp_path):
    c = build_controller(tmp_path, compartments=4)
    c.start(compartments=[2, 4])
    st = wait_done(c)
    assert [r["compartment"] for r in st["results"]] == [2, 4]


def test_invalid_compartment_rejected_before_the_motor_moves(tmp_path):
    c = build_controller(tmp_path, compartments=4)
    with pytest.raises(scan_module.ScanError):
        c.start(compartments=[9])
    assert c.motor.position == 1


def test_two_scans_cannot_run_at_once(tmp_path):
    slow = SimulatedTrayMotor(4, 50, move_seconds=0.15)
    c = build_controller(tmp_path, compartments=4, motor=slow)
    c.start()
    try:
        with pytest.raises(scan_module.ScanError):
            c.start()
    finally:
        c.cancel()
        wait_done(c)


def test_camera_failure_stops_the_motor_and_reports_error(tmp_path):
    cam = StubCamera(fail_after=1)          # second compartment returns nothing
    c = build_controller(tmp_path, compartments=4, camera=cam)
    c.start()
    st = wait_done(c)

    assert st["state"] == scan_module.ERROR
    assert "camera" in (st["error"] or "").lower()
    assert st["motor"]["moving"] is False    # failed safe


def test_cancel_stops_a_running_scan(tmp_path):
    slow = SimulatedTrayMotor(4, 50, move_seconds=0.2)
    c = build_controller(tmp_path, compartments=4, motor=slow)
    c.start()
    time.sleep(0.05)
    c.cancel()
    st = wait_done(c)
    assert st["completed"] < 4
    assert st["motor"]["moving"] is False
    # Stopping on request is a normal outcome, NOT a fault. This asserted only
    # "motor stopped" before, and missed that the state came back as ERROR.
    assert st["state"] == scan_module.CANCELLED
    assert st["error"] is None


def test_cancel_flag_does_not_leak_into_the_next_scan(tmp_path):
    """
    The cancel Event is sticky. If it is not cleared, the very next scan after
    a cancelled one aborts immediately - which looked like a broken rig.
    """
    slow = SimulatedTrayMotor(4, 50, move_seconds=0.2)
    c = build_controller(tmp_path, compartments=4, motor=slow)
    c.start()
    time.sleep(0.05)
    c.cancel()
    wait_done(c)

    # single-compartment path
    rec = c.scan_one(2)
    assert rec["result"]["compartment"] == 2

    # and a full scan
    c.motor.move_seconds = 0
    c.start()
    st = wait_done(c)
    assert st["state"] == scan_module.COMPLETE
    assert st["completed"] == 4


def test_transient_dropped_frame_does_not_kill_the_scan(tmp_path):
    """USB cameras drop frames; one miss must not lose a whole tray."""
    class FlakyCamera(StubCamera):
        def __init__(self):
            super().__init__()
            self.n = 0

        def snapshot(self):
            self.n += 1
            if self.n == 1:          # first grab fails, retry succeeds
                return None
            return self.frame

    c = build_controller(tmp_path, compartments=2, camera=FlakyCamera())
    c.start()
    st = wait_done(c)
    assert st["state"] == scan_module.COMPLETE
    assert st["completed"] == 2


def test_images_are_saved_per_compartment_by_date(tmp_path):
    c = build_controller(tmp_path, compartments=2, save_images=True)
    c.start()
    st = wait_done(c)
    assert st["state"] == scan_module.COMPLETE

    found = []
    for dirpath, _d, names in os.walk(str(tmp_path / "img")):
        found += [os.path.join(dirpath, n) for n in names if n.endswith(".jpg")]
    assert len(found) == 2
    assert any(os.sep + "compartment_01" + os.sep in p for p in found)


def test_scan_one_is_synchronous_and_records_history(tmp_path):
    c = build_controller(tmp_path, compartments=4)
    rec = c.scan_one(3)
    assert rec["compartment"] == 3
    assert rec["result"]["compartment"] == 3
    assert len(c.history.records) == 1
    assert c.state == scan_module.IDLE


# --------------------------------------------------------------------------- #
# End-to-end with the REAL model
# --------------------------------------------------------------------------- #
def test_end_to_end_with_the_real_ai_and_a_real_photo(tmp_path):
    """
    Full path with nothing faked but the hardware: a real sample photo through
    the real trained model, into a real per-compartment result.
    """
    import cv2
    from ai import inspect as real_inspect

    photo = os.path.join(PYDIR, "sample_data", "tomato_fresh.jpg")
    if not os.path.exists(photo):
        pytest.skip("sample_data missing")
    frame = cv2.imread(photo, cv2.IMREAD_COLOR)
    assert frame is not None

    c = build_controller(tmp_path, compartments=2,
                         camera=StubCamera(frame=frame),
                         inspect_fn=real_inspect)
    c.start()
    st = wait_done(c, timeout=120)

    assert st["state"] == scan_module.COMPLETE
    assert len(st["results"]) == 2
    for r in st["results"]:
        assert r["compartment"] in (1, 2)
        assert r["status"] in ("ok", "invalid_image", "not_food")
        if r["status"] == "ok":
            assert 0.0 <= r["condition_confidence"] <= 1.0
            assert 0.0 <= r["food_confidence"] <= 1.0
