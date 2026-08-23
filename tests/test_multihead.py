"""
Tests for the two-head (condition + food) backends in ai/multihead.py.

These run without TensorFlow: a stub subclass supplies the head outputs that a
real TFLite/Keras model would produce, so all the surrounding logic (softmax
detection, food-hint precedence, evidence gating, head resolution) is exercised
on every machine, GPU or not.
"""

import os

import cv2
import numpy as np
import pytest

from ai import get_config
from ai import model as model_module
from ai import multihead

CLASSES = ["fresh", "spoiled"]
FOODS = ["apple", "banana", "tomato"]
SAMPLE = os.path.join(os.path.dirname(__file__), "..", "python", "sample_data", "tomato_spoiled.jpg")


class StubNet(multihead._MultiHeadBackend):
    """Returns fixed head outputs so we can test everything around the net."""

    name = "stub"

    def __init__(self, cond, food):
        super().__init__(CLASSES, FOODS, 128)
        self._c = np.array(cond, dtype=np.float64)
        self._f = np.array(food, dtype=np.float64)

    def _raw_predict(self, tensor):
        # The preprocessing contract: RGB float32 in [0,1], shape (1,S,S,3).
        assert tensor.shape == (1, 128, 128, 3)
        assert tensor.dtype == np.float32
        assert tensor.min() >= 0.0 and tensor.max() <= 1.0
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
    assert img is not None, f"missing sample image: {SAMPLE}"
    return img


def test_probabilities_pass_through(cfg, frame):
    r = StubNet([0.1, 0.9], [0.05, 0.05, 0.90]).predict(frame, cfg)
    assert r["probs"]["spoiled"] == pytest.approx(0.9)
    assert sum(r["probs"].values()) == pytest.approx(1.0)
    assert r["food"] == "tomato"
    assert r["food_confidence"] == pytest.approx(0.90, abs=1e-3)


def test_logits_are_softmaxed(cfg, frame):
    r = StubNet([-2.0, 3.0], [1.0, 0.0, 5.0]).predict(frame, cfg)
    assert sum(r["probs"].values()) == pytest.approx(1.0)
    assert r["probs"]["spoiled"] > r["probs"]["fresh"]
    assert r["food"] == "tomato"


def test_food_hint_overrides_the_net(cfg, frame):
    # Net says apple; the rig says banana. The rig wins - it knows what it loaded.
    r = StubNet([0.1, 0.9], [0.9, 0.05, 0.05]).predict(frame, cfg, food_hint="banana")
    assert r["food"] == "banana"
    assert r["food_confidence"] == pytest.approx(0.99)


def test_fresh_prediction_suppresses_spoilage_evidence(cfg, frame):
    # Never attach "visible fungal growth" to something the net called fresh.
    r = StubNet([0.95, 0.05], [0.9, 0.05, 0.05]).predict(frame, cfg)
    assert r["evidence"] == []


def test_spoiled_prediction_may_carry_evidence(cfg, frame):
    r = StubNet([0.05, 0.95], [0.05, 0.05, 0.9]).predict(frame, cfg)
    assert isinstance(r["evidence"], list)


@pytest.mark.parametrize("cond,food", [([0.1, 0.2, 0.7], [0.9, 0.05, 0.05]),
                                       ([0.1, 0.9], [0.5, 0.5])])
def test_head_size_mismatch_raises(cfg, frame, cond, food):
    with pytest.raises(ValueError):
        StubNet(cond, food).predict(frame, cfg)


@pytest.mark.parametrize("labels,expected", [
    (None, False),
    ({}, False),
    ({"type": "condition", "classes": CLASSES}, False),
    ({"type": "condition+food", "classes": CLASSES}, False),          # no foods
    ({"type": "condition+food", "classes": CLASSES, "foods": FOODS}, True),
])
def test_is_multihead(labels, expected):
    assert multihead.is_multihead(labels) is expected


def _cfg_without_models(base):
    """
    Point the config at paths that cannot exist.

    These tests are about the NO-MODEL code path, so they must not depend on
    whether someone has installed a real model into models/. Before this, they
    passed on a fresh checkout and failed the moment a trained model arrived -
    which is exactly backwards.
    """
    base.model.tflite_model_path = "models/__absent__.tflite"
    base.model.keras_model_path = "models/__absent__.keras"
    base.model.labels_path = "models/__absent__.json"
    return base


def test_try_build_returns_none_without_model_file(cfg):
    labels = {"type": "condition+food", "classes": CLASSES, "foods": FOODS}
    assert multihead.try_build(labels, _cfg_without_models(cfg)) is None


def test_build_backend_falls_back_to_heuristic_when_no_model():
    # With no trained model on disk the pipeline must keep working, unchanged.
    assert model_module.build_backend(_cfg_without_models(get_config())).name == "heuristic"


def test_mismatched_labels_are_rejected_not_guessed(cfg):
    """
    If labels.json describes different classes than the .tflite actually has,
    refuse to build rather than silently misinterpreting the outputs. A model
    that reads food probabilities as condition probabilities produces confident
    nonsense, which is far worse than a crash.
    """
    import os
    if not os.path.exists(get_config().model.tflite_model_path):
        pytest.skip("no trained model installed")
    wrong = {"type": "condition+food", "classes": CLASSES,
             "foods": ["only", "three", "foods"]}
    with pytest.raises(ValueError):
        multihead.try_build(wrong, get_config())


def test_head_resolution_by_width_is_order_independent():
    # TFLite does not guarantee output order matches how the heads were declared.
    b = multihead.MultiHeadTFLiteBackend.__new__(multihead.MultiHeadTFLiteBackend)
    b.classes, b.foods = CLASSES, FOODS
    b.output_details = [{"shape": [1, 3]}, {"shape": [1, 2]}]   # food first
    assert b._resolve_heads(None) == (1, 0)
    assert b._resolve_heads({"condition": 0, "food": 1}) == (0, 1)


def test_head_resolution_raises_on_ambiguous_widths():
    b = multihead.MultiHeadTFLiteBackend.__new__(multihead.MultiHeadTFLiteBackend)
    b.classes, b.foods = CLASSES, FOODS
    b.output_details = [{"shape": [1, 7]}, {"shape": [1, 9]}]   # matches neither head
    with pytest.raises(ValueError):
        b._resolve_heads(None)
