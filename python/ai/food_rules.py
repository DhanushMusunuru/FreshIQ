"""
Food-specific visual logic (CORE AI TASK #10).

Different produce ripens and spoils differently, so we should not apply one set
of colour rules to everything. Each ``FoodRule`` captures:

  * the OpenCV HSV hue band(s) typical of a *healthy* example of that food, used
    both for a rough colour-based identity guess and to detect discoloration;
  * whether colour change is a *ripening* signal (banana yellow->brown, tomato
    green->red) versus a *spoilage* signal;
  * a ``ripeness_hue`` mapping that turns dominant hue into a fresh/ripe lean.

OpenCV hue is 0-179 (not 0-359). Red wraps around 0/179.

These rules are consumed by the heuristic backend. The deep-learning backend
learns food-specific behaviour directly from the training folders, so it does
not need this table — but the table still drives the identity guess and the
human-readable evidence strings, which is why it lives in its own module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class FoodRule:
    name: str
    # Healthy-colour hue bands (inclusive) in OpenCV's 0-179 space. A list so we
    # can express red (wraps) or multi-tone produce.
    healthy_hue_bands: List[Tuple[int, int]]
    # If True, a shift AWAY from the healthy hue is (mostly) natural ripening
    # rather than spoilage (e.g. banana going yellow then brown-flecked).
    colour_change_is_ripening: bool = False
    # Optional note surfaced to humans in evidence when discoloration is seen.
    discoloration_note: str = "unusual discoloration"
    # Minimum saturation for a "vivid healthy" example — dull produce reads as older.
    healthy_min_saturation: float = 60.0

    def hue_matches(self, hue: float) -> bool:
        for lo, hi in self.healthy_hue_bands:
            if lo <= hi:
                if lo <= hue <= hi:
                    return True
            else:  # wrapping band, e.g. (170, 10) for red
                if hue >= lo or hue <= hi:
                    return True
        return False


# NOTE: hue bands are approximate and lighting-dependent. They are a heuristic
# prior, not a calibrated measurement. Retune per your camera/lighting.
FOOD_RULES: Dict[str, FoodRule] = {
    "tomato": FoodRule(
        name="tomato",
        healthy_hue_bands=[(0, 10), (170, 179)],   # red
        colour_change_is_ripening=True,            # green->red is ripening
        discoloration_note="dark or sunken discoloration",
    ),
    "apple": FoodRule(
        name="apple",
        healthy_hue_bands=[(0, 12), (35, 85), (170, 179)],  # red / green apples
        discoloration_note="browning or bruising",
    ),
    "banana": FoodRule(
        name="banana",
        healthy_hue_bands=[(20, 38)],              # yellow
        colour_change_is_ripening=True,            # yellow->brown flecks = ripe
        discoloration_note="brown/black patches",
    ),
    "orange": FoodRule(
        name="orange",
        healthy_hue_bands=[(8, 25)],               # orange
        discoloration_note="soft or greenish mould-prone spots",
    ),
    "lemon": FoodRule(
        name="lemon",
        healthy_hue_bands=[(22, 40)],              # yellow
        discoloration_note="mould or softening",
    ),
    "carrot": FoodRule(
        name="carrot",
        healthy_hue_bands=[(8, 22)],               # orange
        discoloration_note="dark rot or white 'blush' dehydration",
    ),
    "potato": FoodRule(
        name="potato",
        healthy_hue_bands=[(15, 35)],              # tan/brown
        healthy_min_saturation=25.0,               # potatoes are dull
        discoloration_note="green patches, sprouting or dark rot",
    ),
    "onion": FoodRule(
        name="onion",
        healthy_hue_bands=[(15, 40), (120, 160)],  # tan / red onion
        healthy_min_saturation=25.0,
        discoloration_note="soft brown spots or mould at the neck",
    ),
    "cucumber": FoodRule(
        name="cucumber",
        healthy_hue_bands=[(35, 85)],              # green
        discoloration_note="yellowing or soft mushy ends",
    ),
    "capsicum": FoodRule(
        name="capsicum",
        healthy_hue_bands=[(0, 12), (35, 85), (170, 179)],  # green/red/yellow
        discoloration_note="wrinkling or soft dark spots",
    ),
    # --- the remaining foods in the training dataset ------------------------
    # Without an entry here, guess_food_by_hue can never name these and the
    # heuristic falls back to "unknown" for them — a mismatch with the trained
    # food head, which predicts all 13.
    "bell_pepper": FoodRule(
        name="bell_pepper",
        healthy_hue_bands=[(0, 12), (20, 35), (35, 85), (170, 179)],  # red/yellow/green
        discoloration_note="wrinkling or soft dark spots",
    ),
    "mango": FoodRule(
        name="mango",
        healthy_hue_bands=[(15, 40), (40, 70)],    # yellow-orange, or green when unripe
        colour_change_is_ripening=True,            # green->yellow/red is ripening
        discoloration_note="black spots or sap-burn patches",
    ),
    "strawberry": FoodRule(
        name="strawberry",
        healthy_hue_bands=[(0, 10), (168, 179)],   # deep red
        colour_change_is_ripening=True,            # white/green->red is ripening
        discoloration_note="white mould bloom or darkened mushy areas",
        healthy_min_saturation=70.0,               # healthy berries are vivid
    ),
    "okra": FoodRule(
        name="okra",
        healthy_hue_bands=[(35, 85)],              # green
        discoloration_note="blackened ridges or slimy brown patches",
    ),
    "bitter_gourd": FoodRule(
        name="bitter_gourd",
        healthy_hue_bands=[(35, 90)],              # green
        # A bitter gourd turning yellow/orange is over-mature, and for cooking
        # purposes that is closer to spoiled than to a desirable ripe state.
        discoloration_note="yellowing (over-mature) or soft brown rot",
    ),
}


def get_rule(food: str) -> Optional[FoodRule]:
    return FOOD_RULES.get(food)


def guess_food_by_hue(dominant_hue: float, mean_saturation: float,
                      allowed: Optional[List[str]] = None) -> Tuple[str, float]:
    """
    Very rough identity guess from the dominant hue. Returns (food, confidence
    0..1). This is deliberately low-confidence: colour alone cannot reliably tell
    a tomato from a red apple. In production the trained model — or an operator
    'food_hint' — should provide identity. See docs/LIMITATIONS.md.
    """
    candidates = allowed if allowed else list(FOOD_RULES.keys())
    matches = [name for name in candidates
               if name in FOOD_RULES and FOOD_RULES[name].hue_matches(dominant_hue)]
    if not matches:
        return "unknown", 0.0
    # Confidence is intentionally capped low; colour is weak evidence of identity.
    conf = 0.35 if len(matches) == 1 else 0.2
    if mean_saturation < 20:
        conf *= 0.7
    return matches[0], conf
