"""
Tests for the "that isn't produce" path.

WHY THIS MATTERS
----------------
Without a not_food class the model must pick one of its foods for any image you
show it. Point it at a hand and it says "fresh cucumber, 94%" with total
confidence. A reject class is what lets the answer be "that is not fruit".

The freshness verdict must be SUPPRESSED, not merely relabelled: the condition
head was never trained on non-food (those rows had their condition loss masked
out), so its numbers for such a frame are meaningless.

Runs without TensorFlow via a stub backend, like the rest of the suite.
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from ai import classifier, get_config
from ai import multihead
from ai.aggregation import aggregate

CLASSES = ["fresh", "spoiled"]
FOODS = ["apple", "banana", "tomato", "not_food"]
SAMPLE = os.path.join(os.path.dirname(__file__), "..", "python", "sample_data", "tomato_fresh.jpg")


class StubNet(multihead._MultiHeadBackend):
    name = "stub"

    def __init__(self, cond, food, not_food_class="not_food"):
        super().__init__(CLASSES, FOODS, 128, not_food_class)
        self._c = np.array(cond, dtype=np.float64)
        self._f = np.array(food, dtype=np.float64)

    def _raw_predict(self, tensor):
        return self._c, self._f


@pytest.fixture
def cfg():
    c = get_config()
    c.conditions = list(CLASSES)
    c.foods = FOODS + ["unknown"]
    return c


@pytest.fixture
def frame():
    img = cv2.imread(SAMPLE, cv2.IMREAD_COLOR)
    assert img is not None
    return img


# --------------------------------------------------------------------------- #
# backend level
# --------------------------------------------------------------------------- #
def test_not_food_wins_and_suppresses_condition(cfg, frame):
    # food head strongly says not_food; condition head says "fresh" loudly.
    r = StubNet([0.95, 0.05], [0.01, 0.01, 0.03, 0.95]).predict(frame, cfg)
    assert r["not_food"] is True
    assert r["food"] == "not_food"
    # The confident "fresh" must NOT survive.
    assert all(v == 0.0 for v in r["probs"].values())
    assert r["evidence"] == []


def test_real_food_is_unaffected(cfg, frame):
    r = StubNet([0.1, 0.9], [0.02, 0.02, 0.94, 0.02]).predict(frame, cfg)
    assert r["not_food"] is False
    assert r["food"] == "tomato"
    assert r["probs"]["spoiled"] > r["probs"]["fresh"]


def test_model_without_negatives_never_rejects(cfg, frame):
    """An older model has no not_food class; behaviour must be unchanged."""
    net = StubNet([0.1, 0.9], [0.02, 0.02, 0.94, 0.02], not_food_class=None)
    assert net.not_food_class is None
    r = net.predict(frame, cfg)
    assert r["not_food"] is False


def test_food_hint_overrides_a_not_food_prediction(cfg, frame):
    """The operator saying 'this is a tomato' beats the model's rejection."""
    r = StubNet([0.1, 0.9], [0.0, 0.0, 0.0, 1.0]).predict(frame, cfg, food_hint="tomato")
    assert r["food"] == "tomato"
    assert r["not_food"] is False


# --------------------------------------------------------------------------- #
# classifier level
# --------------------------------------------------------------------------- #
def test_classifier_returns_not_food_status(cfg, frame, monkeypatch):
    monkeypatch.setattr(classifier, "_get_backend",
                        lambda config: StubNet([0.95, 0.05], [0, 0, 0, 1.0]))
    res = classifier.classify_frame(frame, config=cfg)
    assert res["status"] == "not_food"
    assert res["condition"] is None
    assert res["fresh"] == 0 and res["spoiled"] == 0


# --------------------------------------------------------------------------- #
# aggregation level
# --------------------------------------------------------------------------- #
def test_aggregate_reports_not_food_rather_than_camera_failure(cfg):
    views = [{"status": "not_food", "food": "not_food", "image_quality": "good"}
             for _ in range(3)]
    out = aggregate(views, cfg)
    assert out["status"] == "not_food"
    assert out["condition"] is None
    assert out["reason"] == "not_produce"
    assert out["views_used"] == 3


def test_one_good_view_still_wins_over_a_rejection(cfg):
    """A single usable produce view should still produce a verdict."""
    views = [
        {"status": "not_food", "food": "not_food", "image_quality": "good"},
        {"status": "ok", "food": "tomato", "condition": "spoiled",
         "confidence_raw": {"fresh": 0.1, "spoiled": 0.9},
         "evidence": ["dark spots"], "image_quality": "good",
         "fresh": 10, "spoiled": 90},
    ]
    out = aggregate(views, cfg)
    assert out["status"] == "ok"
    assert out["condition"] == "spoiled"


def test_blurred_views_still_report_as_invalid_not_not_food(cfg):
    views = [{"status": "invalid_image", "reason": "blurred"} for _ in range(2)]
    out = aggregate(views, cfg)
    assert out["status"] == "invalid_image"
    assert out["reason"] == "blurred"
