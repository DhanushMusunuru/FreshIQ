"""
Automated tests for the FreshIQ AI pipeline.

These run against the heuristic backend (no training required) using synthetic
images, so they work anywhere OpenCV + NumPy are installed. Run with:

    pytest -q
"""

import os
import sys

import numpy as np
import pytest
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from ai import classify_image, classify_frame, inspect, get_config
from ai.utils import to_int_percentages
from ai.quality_check import check_quality


# --------------------------------------------------------------------------- #
# Synthetic image helpers
# --------------------------------------------------------------------------- #
def _tray(bg=180):
    img = np.full((480, 640, 3), bg, dtype=np.uint8)
    noise = np.random.default_rng(0).normal(0, 5, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def fresh_tomato():
    img = _tray()
    cv2.circle(img, (320, 240), 150, (40, 40, 210), -1)
    return img


def mouldy_tomato():
    img = _tray()
    cv2.circle(img, (320, 240), 150, (40, 45, 150), -1)
    rng = np.random.default_rng(1)
    for _ in range(30):
        x = int(320 + rng.uniform(-120, 120))
        y = int(240 + rng.uniform(-120, 120))
        cv2.circle(img, (x, y), int(rng.integers(6, 18)), (20, 20, 25), -1)
    for _ in range(400):
        x = int(360 + rng.uniform(-60, 60))
        y = int(240 + rng.uniform(-60, 60))
        g = int(rng.integers(130, 180))
        cv2.circle(img, (x, y), int(rng.integers(1, 4)), (g, g, g - 20), -1)
    return img


# --------------------------------------------------------------------------- #
# utils
# --------------------------------------------------------------------------- #
def test_percentages_sum_to_100():
    for probs in [{"a": 0.3333, "b": 0.3333, "c": 0.3334},
                  {"a": 1, "b": 1, "c": 1},
                  {"a": 0, "b": 0, "c": 0},
                  {"a": 0.9, "b": 0.05, "c": 0.05}]:
        pct = to_int_percentages(probs)
        assert sum(pct.values()) == 100
        assert all(v >= 0 for v in pct.values())


# --------------------------------------------------------------------------- #
# single-view classification
# --------------------------------------------------------------------------- #
def test_fresh_classification_shape():
    res = classify_frame(fresh_tomato(), food_hint="tomato")
    assert res["status"] == "ok"
    assert res["food"] == "tomato"
    conditions = get_config().conditions
    assert res["condition"] in conditions
    # Flat percentage keys present, one per configured condition, summing to 100.
    # Driven by the config rather than hard-coded names, so adding or removing a
    # condition class cannot silently break the API contract.
    assert all(c in res for c in conditions)
    assert sum(res[c] for c in conditions) == 100
    assert res["fresh"] >= res["spoiled"]        # fresh item should not read spoiled
    assert isinstance(res["evidence"], list)
    assert res["image_quality"] in ("good", "acceptable", "poor")


def test_mould_reads_spoiled():
    res = classify_frame(mouldy_tomato(), food_hint="tomato")
    assert res["status"] == "ok"
    assert res["spoiled"] > res["fresh"]
    assert any("mould" in e.lower() or "discolor" in e.lower() for e in res["evidence"])


# --------------------------------------------------------------------------- #
# quality gate
# --------------------------------------------------------------------------- #
def test_too_dark_rejected():
    dark = (fresh_tomato() * 0.05).astype(np.uint8)
    res = classify_frame(dark)
    assert res["status"] == "invalid_image"
    assert res["reason"] == "too_dark"


def test_blurred_rejected():
    blurred = cv2.GaussianBlur(fresh_tomato(), (51, 51), 0)
    res = classify_frame(blurred)
    assert res["status"] == "invalid_image"
    assert res["reason"] == "blurred"


def test_quality_report_metrics():
    rep = check_quality(fresh_tomato(), get_config())
    assert rep.usable
    assert rep.brightness > 0
    assert rep.focus_variance > 0


# --------------------------------------------------------------------------- #
# multi-view aggregation
# --------------------------------------------------------------------------- #
def test_worst_view_drives_result():
    # Fresh front, mouldy back -> aggregate must reflect spoilage, not average it away.
    final = inspect([fresh_tomato(), mouldy_tomato()], food_hint="tomato")
    assert final["status"] == "ok"
    assert sum(final["confidence"].values()) == 100
    assert final["condition"] == "spoiled"
    assert final["views_used"] == 2


def test_all_invalid_views():
    dark = (fresh_tomato() * 0.05).astype(np.uint8)
    final = inspect([dark, dark])
    assert final["status"] == "invalid_image"
    assert final["views_used"] == 0


def test_invalid_view_is_skipped_not_fatal():
    dark = (fresh_tomato() * 0.05).astype(np.uint8)
    final = inspect([fresh_tomato(), dark], food_hint="tomato")
    assert final["status"] == "ok"
    assert final["views_used"] == 1
    assert final["views_total"] == 2


# --------------------------------------------------------------------------- #
# file path + error handling
# --------------------------------------------------------------------------- #
def test_unreadable_file():
    res = classify_image("/nonexistent/path/to/image.jpg")
    assert res["status"] == "invalid_image"
    assert res["reason"] == "unreadable_file"


def test_food_hint_overrides_identity():
    res = classify_frame(fresh_tomato(), food_hint="apple")
    assert res["food"] == "apple"
    assert res["food_confidence"] >= 0.9
