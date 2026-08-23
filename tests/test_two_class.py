"""
Regression tests for the two-class (fresh/spoiled) configuration.

THE BUG THESE GUARD AGAINST
---------------------------
ai/heuristic.py used to pick its middle "ripe" bucket like this:

    ripe_key = "ripe" if "ripe" in scores else config.conditions[min(1, len - 1)]

With conditions = ["fresh", "spoiled"] that resolves to "spoiled". So every mild
ageing signal -- a banana's brown flecks, a slightly dull tomato, a tomato that
has ripened from green to red -- was scored as SPOILAGE at full strength. A
perfectly good ripening banana would be reported as rotten.

The fix routes those signals through add_ageing(), which damps them when there is
no middle bucket to put them in. These tests pin that behaviour down.
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from ai import get_config
from ai import heuristic
from ai.config import Config


def _tray(bg=180):
    img = np.full((480, 640, 3), bg, dtype=np.uint8)
    noise = np.random.default_rng(0).normal(0, 5, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def ripening_banana():
    """
    A yellow banana with a scattering of brown flecks: ripe, NOT rotten.

    Tuned to land in the "ageing" band of ai/config.HeuristicThresholds, which is
    what makes this a meaningful regression test:

        dark_spot_ripe_fraction    = 0.04   <-- this image measures ~0.074
        dark_spot_spoiled_fraction = 0.12

    Below 0.04 nothing fires at all; above 0.12 it is scored as real spoilage.
    Only between them does the damping in heuristic.add_ageing() matter.
    """
    img = _tray()
    cv2.ellipse(img, (320, 240), (190, 90), 0, 0, 360, (60, 200, 230), -1)  # BGR yellow
    rng = np.random.default_rng(7)
    for _ in range(26):
        x = int(320 + rng.uniform(-150, 150))
        y = int(240 + rng.uniform(-55, 55))
        cv2.circle(img, (x, y), int(rng.integers(6, 10)), (35, 60, 90), -1)
    return img


def test_fixture_lands_in_the_ageing_band():
    """If this fails the other tests are vacuous -- they'd be testing nothing."""
    ht = get_config().heuristic
    res = heuristic.classify(ripening_banana(), two_class_config(), food_hint="banana")
    dark = res["features"]["dark_spot_fraction"]
    assert ht.dark_spot_ripe_fraction < dark < ht.dark_spot_spoiled_fraction, (
        f"fixture measures {dark:.4f}, outside the ageing band "
        f"({ht.dark_spot_ripe_fraction}, {ht.dark_spot_spoiled_fraction})")
    assert "dark spots / bruising" in res["evidence"]


def two_class_config() -> Config:
    cfg = get_config()
    cfg.conditions = ["fresh", "spoiled"]
    return cfg


def three_class_config() -> Config:
    cfg = get_config()
    cfg.conditions = ["fresh", "ripe", "spoiled"]
    return cfg


def test_default_config_is_two_class():
    # The dataset has no 'ripe' data, so a 'ripe' class could never be predicted.
    assert get_config().conditions == ["fresh", "spoiled"]


def test_ripening_banana_is_not_called_spoiled():
    """
    The core regression: mild ripening must not read as rot.

    Measured numbers for this fixture:
        with damping (correct):  fresh 0.625 / spoiled 0.375  -> fresh wins
        without damping (bug):   fresh 0.368 / spoiled 0.632  -> spoiled wins
    """
    res = heuristic.classify(ripening_banana(), two_class_config(), food_hint="banana")
    assert res["probs"]["fresh"] > res["probs"]["spoiled"], (
        f"ripening banana scored as spoiled: {res['probs']} / {res['evidence']}")
    # ...but the signal must still be present, not silently discarded.
    assert res["probs"]["spoiled"] > 0.05, (
        "ageing evidence vanished entirely; damping should attenuate, not erase")


def test_ageing_still_reduces_freshness():
    """Damped is not the same as ignored -- the signal must still move the needle."""
    cfg = two_class_config()
    clean = _tray()
    cv2.ellipse(clean, (320, 240), (190, 90), 0, 0, 360, (60, 200, 230), -1)

    pristine = heuristic.classify(clean, cfg, food_hint="banana")
    aged = heuristic.classify(ripening_banana(), cfg, food_hint="banana")
    assert aged["probs"]["fresh"] < pristine["probs"]["fresh"], (
        "brown flecks should lower the fresh score even in a two-class config")


def test_three_class_config_routes_ageing_to_ripe():
    """With a middle bucket available, ageing goes there at full strength."""
    res = heuristic.classify(ripening_banana(), three_class_config(), food_hint="banana")
    assert "ripe" in res["probs"]
    assert res["probs"]["ripe"] > res["probs"]["spoiled"], (
        f"ageing signal did not land in 'ripe': {res['probs']}")


def test_mould_still_reads_spoiled_in_two_class():
    """Damping ageing must not weaken genuine spoilage detection."""
    img = _tray()
    cv2.circle(img, (320, 240), 150, (40, 45, 150), -1)
    rng = np.random.default_rng(1)
    for _ in range(30):
        cv2.circle(img, (int(320 + rng.uniform(-120, 120)),
                         int(240 + rng.uniform(-120, 120))),
                   int(rng.integers(6, 18)), (20, 20, 25), -1)
    for _ in range(400):
        g = int(rng.integers(130, 180))
        cv2.circle(img, (int(360 + rng.uniform(-60, 60)),
                         int(240 + rng.uniform(-60, 60))),
                   int(rng.integers(1, 4)), (g, g, g - 20), -1)

    res = heuristic.classify(img, two_class_config(), food_hint="tomato")
    assert res["probs"]["spoiled"] > res["probs"]["fresh"], res["probs"]


@pytest.mark.parametrize("conditions", [
    ["fresh", "spoiled"],
    ["fresh", "ripe", "spoiled"],
    ["fresh", "ripe", "overripe", "spoiled"],
])
def test_probs_always_cover_configured_conditions(conditions):
    cfg = get_config()
    cfg.conditions = conditions
    res = heuristic.classify(ripening_banana(), cfg, food_hint="banana")
    assert set(res["probs"]) == set(conditions)
    assert sum(res["probs"].values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# The food list must cover every class the trained food head can emit.
# --------------------------------------------------------------------------- #
DATASET_FOODS = ["apple", "banana", "bell_pepper", "bitter_gourd", "capsicum",
                 "carrot", "cucumber", "mango", "okra", "orange", "potato",
                 "strawberry", "tomato"]


def test_config_covers_every_dataset_food():
    foods = get_config().foods
    missing = [f for f in DATASET_FOODS if f not in foods]
    assert not missing, f"trained food head can emit these but config rejects them: {missing}"


def test_every_dataset_food_has_a_colour_rule():
    """Without a rule, guess_food_by_hue can never name the food -> always 'unknown'."""
    from ai import food_rules
    missing = [f for f in DATASET_FOODS if food_rules.get_rule(f) is None]
    assert not missing, f"foods with no FoodRule entry: {missing}"


def test_input_size_matches_dataset_resolution():
    # Training images are natively 128x128; upscaling to 224 costs ~3x the
    # arithmetic on the Pi for zero extra information.
    assert get_config().model.input_size == 128
