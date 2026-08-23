"""
Heuristic (classical computer-vision) backend.

This is the zero-training fallback that makes the whole FreshIQ pipeline runnable
on day one and on any machine with just OpenCV + NumPy — including the Arduino
UNO Q's Linux side. It analyses colour, dark spots, mould-like texture and
surface uniformity to produce a fresh/ripe/spoiled estimate plus human-readable
evidence.

Honesty note: this is a *baseline*. It encodes reasonable visual rules but has
NOT been validated for accuracy on real produce. Treat its numbers as rough
estimates. The deep-learning backend (once you train it on your own dataset) is
expected to do materially better; use this to get the pipeline working and to
have a graceful fallback when no trained model is present.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import food_rules, preprocessing
from .config import Config
from .utils import normalise


def _food_pixels(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return an (N, 3) array of HSV values for the food pixels only."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sel = mask > 0
    return hsv[sel]


def _dominant_hue(hsv_pixels: np.ndarray) -> Tuple[float, float]:
    """Dominant hue (peak of the histogram) and mean saturation of food pixels."""
    if hsv_pixels.size == 0:
        return 0.0, 0.0
    # Consider only reasonably saturated pixels when finding the colour, so grey
    # shadows do not dominate the hue estimate.
    sat = hsv_pixels[:, 1]
    hue = hsv_pixels[:, 0]
    vivid = hue[sat > 40]
    src = vivid if vivid.size > 50 else hue
    hist = np.bincount(src.astype(np.int64), minlength=180)[:180]
    dominant = float(np.argmax(hist)) if hist.sum() > 0 else 0.0
    return dominant, float(np.mean(sat))


def _dark_spot_fraction(frame: np.ndarray, mask: np.ndarray, ratio: float) -> float:
    """Fraction of food pixels that are much darker than the food's median value."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    sel = mask > 0
    food_v = v[sel]
    if food_v.size == 0:
        return 0.0
    median_v = float(np.median(food_v))
    if median_v <= 1:
        return 0.0
    dark = food_v < (ratio * median_v)
    return float(np.count_nonzero(dark)) / float(food_v.size)


def _mold_fraction(frame: np.ndarray, mask: np.ndarray, sat_max: float) -> float:
    """
    Fraction of food area that looks mould-like: low saturation (grey/white/fuzzy)
    AND locally textured (fuzz has high local variance), sitting on the food.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    s = hsv[:, :, 1]
    # Local texture energy.
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    tex = cv2.GaussianBlur(np.abs(lap), (7, 7), 0)
    sel = mask > 0
    if np.count_nonzero(sel) == 0:
        return 0.0
    tex_thresh = float(np.percentile(tex[sel], 80)) if np.count_nonzero(sel) else 0.0
    mold_like = (s < sat_max) & (tex > tex_thresh) & sel
    return float(np.count_nonzero(mold_like)) / float(np.count_nonzero(sel))


def _texture_std(frame: np.ndarray, mask: np.ndarray) -> float:
    """Std-dev of value channel on the food — a proxy for shrivel/surface damage."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    sel = mask > 0
    return float(np.std(v[sel])) if np.count_nonzero(sel) else 0.0


def _hue_std(hsv_pixels: np.ndarray) -> float:
    """Circular-ish spread of hue on the food — high spread suggests uneven colour."""
    if hsv_pixels.size == 0:
        return 0.0
    hue = hsv_pixels[:, 1] if False else hsv_pixels[:, 0]
    # Simple linear std is fine here as a relative signal (documented heuristic).
    return float(np.std(hue.astype(np.float32)))


def classify(frame: np.ndarray, config: Config,
             food_hint: Optional[str] = None) -> Dict:
    """
    Produce a single-view heuristic classification.

    Returns a dict with keys: food, food_confidence, probs (dict of condition->
    float 0..1 that sums to 1), evidence (list[str]).
    """
    frame = preprocessing.resize_max(frame, 640)
    mask, _bbox = preprocessing.segment_food(frame)
    hsv_pixels = _food_pixels(frame, mask)

    dominant_hue, mean_sat = _dominant_hue(hsv_pixels)
    ht = config.heuristic
    dark_frac = _dark_spot_fraction(frame, mask, ht.dark_spot_value_ratio)
    mold_frac = _mold_fraction(frame, mask, ht.mold_saturation_max)
    tex_std = _texture_std(frame, mask)
    hue_std = _hue_std(hsv_pixels)

    # --- Identity -----------------------------------------------------------
    if food_hint and food_hint in config.foods:
        food, food_conf = food_hint, 0.99
    else:
        food, food_conf = food_rules.guess_food_by_hue(
            dominant_hue, mean_sat, allowed=config.foods)
    rule = food_rules.get_rule(food)

    # --- Condition scoring --------------------------------------------------
    # Start from a "fresh" prior and accumulate ageing/spoilage evidence.
    scores = {c: 0.0 for c in config.conditions}
    fresh_key = "fresh" if "fresh" in scores else config.conditions[0]
    spoiled_key = "spoiled" if "spoiled" in scores else config.conditions[-1]

    # "Ageing" signals are things like a banana's brown flecks or a slightly dull
    # tomato: real evidence the item is past peak, but NOT evidence it is rotten.
    #
    # When the config has a middle bucket ("ripe"), those signals go there at full
    # strength. When it does not — our two-class fresh/spoiled setup — they must
    # NOT be dumped into "spoiled" at full strength, or every ripe banana gets
    # condemned. We damp them instead: they still pull away from fresh, but only
    # weakly commit to spoiled. AGEING_DAMPING is that trade-off, made explicit.
    ripe_key = "ripe" if "ripe" in scores else spoiled_key
    AGEING_DAMPING = 1.0 if ripe_key != spoiled_key else 0.35

    def add_ageing(amount: float) -> None:
        scores[ripe_key] += amount * AGEING_DAMPING

    scores[fresh_key] = 1.0
    evidence: List[str] = []

    # Mould is the strongest spoilage signal.
    if mold_frac >= ht.mold_min_fraction:
        strength = min(1.0, mold_frac / (ht.mold_min_fraction * 6))
        scores[spoiled_key] += 2.5 + 3.0 * strength
        scores[fresh_key] *= 0.2
        evidence.append("visible mould / fungal growth")

    # Dark spots: a lot -> spoilage; a little -> ripening/ageing.
    if dark_frac >= ht.dark_spot_spoiled_fraction:
        strength = min(1.0, dark_frac / (ht.dark_spot_spoiled_fraction * 3))
        scores[spoiled_key] += 1.5 + 2.5 * strength
        scores[fresh_key] *= 0.4
        evidence.append("extensive dark discoloration")
    elif dark_frac >= ht.dark_spot_ripe_fraction:
        add_ageing(1.2)
        scores[fresh_key] *= 0.7
        evidence.append("dark spots / bruising")

    # Surface texture: shrivel / damage.
    if tex_std >= ht.texture_shrivel_std:
        add_ageing(0.6)
        scores[spoiled_key] += 0.6
        evidence.append("uneven surface texture (possible shrivel/damage)")

    # Colour analysis, food-specific.
    off_colour = False
    if rule is not None:
        if not rule.hue_matches(dominant_hue):
            off_colour = True
        dull = mean_sat < rule.healthy_min_saturation
    else:
        dull = mean_sat < 30

    if off_colour:
        if rule is not None and rule.colour_change_is_ripening:
            # e.g. banana/tomato: colour drift is mostly ripening, unless it is dark.
            # This is the signal that most needs damping in a two-class config —
            # ripening is emphatically not rotting.
            add_ageing(1.6)
            scores[fresh_key] *= 0.7
            evidence.append("colour indicates ripening")
        else:
            add_ageing(0.8)
            scores[spoiled_key] += 0.6
            note = rule.discoloration_note if rule else "unusual discoloration"
            evidence.append(note)

    if hue_std >= ht.discoloration_hue_std:
        add_ageing(0.4)
        scores[spoiled_key] += 0.4
        evidence.append("uneven colouring across surface")

    if dull and not off_colour and not evidence:
        add_ageing(0.5)
        evidence.append("slightly dull colour")

    probs = normalise(scores)

    # Keep evidence concise and de-duplicated.
    from .utils import dedupe_preserve_order
    evidence = dedupe_preserve_order(evidence)[:5]

    return {
        "food": food,
        "food_confidence": round(float(food_conf), 3),
        "probs": probs,
        "evidence": evidence,
        "features": {
            "dominant_hue": round(dominant_hue, 1),
            "mean_saturation": round(mean_sat, 1),
            "dark_spot_fraction": round(dark_frac, 4),
            "mold_fraction": round(mold_frac, 4),
            "texture_std": round(tex_std, 2),
            "hue_std": round(hue_std, 2),
        },
    }
