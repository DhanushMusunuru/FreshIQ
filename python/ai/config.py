"""
FreshIQ AI configuration.

Central place for tunable parameters so the rest of the system does not hard-code
values. A JSON file (default: ``ai/config.json`` or the path in the ``FRESHIQ_CONFIG``
environment variable) can override any of these defaults at runtime, which lets you
add food categories or retune thresholds without touching the code.

IMPORTANT: The condition percentages produced by this system are the model's
*visual classification confidence*, NOT a scientific measurement of physical
freshness. See docs/LIMITATIONS.md.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
# Condition categories
# --------------------------------------------------------------------------- #
# The ordered list of condition classes the system reports.
#
# WHY TWO CLASSES AND NOT THREE: the training data (Kaggle food-freshness-dataset)
# is labelled Fresh/Rotten only. There is no "ripe" data anywhere, so a "ripe"
# class could never be predicted by the trained model — it would sit at 0% forever
# and quietly mislead anyone reading the output. We report what we can actually
# measure. See docs/TRAINING.md.
#
# TO ADD "ripe" BACK: put it in this list, create dataset/<split>/ripe/ folders,
# collect real ripe photos, and retrain. The heuristic backend already knows how
# to score an intermediate "ageing" bucket and will start using it automatically
# (see ai/heuristic.py). Nothing else needs changing.
DEFAULT_CONDITIONS: List[str] = ["fresh", "spoiled"]


# --------------------------------------------------------------------------- #
# Food categories
# --------------------------------------------------------------------------- #
# Designed to be extended. Add a name here and (optionally) a FoodRule entry in
# ai/food_rules.py to give it food-specific colour/ripeness logic.
#
# WHICH OF THESE THE MODEL ACTUALLY KNOWS: only the first eight. The deployed
# food head has EXACTLY 8 outputs - verified against models/freshiq_model.tflite,
# whose output tensor is (1, 8), and matching models/labels.json "foods".
# An earlier version of this comment claimed 13 trained classes; that was wrong
# and would have had you trusting predictions for produce the model has never
# seen.
#
# The remainder are heuristic-only: the rule-based fallback can take a guess at
# them, and they are accepted as a manual food_hint, but the neural food head
# can never output them. labels.json is the authority for the head; this list
# only governs which hints are accepted and what the heuristic may name.
DEFAULT_FOODS: List[str] = [
    # --- the 8 classes the trained food head predicts, in labels.json order -
    "apple",
    "banana",
    "cucumber",
    "mango",
    "okra",
    "orange",
    "potato",
    "tomato",
    # --- heuristic-only guesses (no training data, no food-head class) -----
    "bell_pepper",
    "bitter_gourd",
    "capsicum",
    "carrot",
    "strawberry",
    "lemon",
    "onion",
    "unknown",
]


# --------------------------------------------------------------------------- #
# Thresholds used by quality_check.py and the heuristic backend
# --------------------------------------------------------------------------- #
@dataclass
class QualityThresholds:
    # Mean brightness (0-255) below / above which the frame is unusable.
    min_brightness: float = 40.0
    max_brightness: float = 225.0
    # Fraction of pixels that may be blown-out (value == 255) before "too bright".
    max_overexposed_fraction: float = 0.35
    # Variance of the Laplacian; below this the frame is considered blurred.
    min_focus_variance: float = 55.0
    # Foreground (food) area as a fraction of the frame. Too small -> "too far" /
    # "no food"; too large -> possible obstruction / food pressed on the lens.
    min_food_area_fraction: float = 0.03
    max_food_area_fraction: float = 0.97
    # If more than one large blob is present we may be seeing multiple objects.
    multi_object_area_fraction: float = 0.08  # a blob must be at least this big to count
    max_expected_objects: int = 1


@dataclass
class HeuristicThresholds:
    # Dark-spot detection: pixels much darker than the food's median value.
    dark_spot_value_ratio: float = 0.45   # pixel value < ratio * median food value
    dark_spot_spoiled_fraction: float = 0.12   # >=this fraction of dark spots -> strong spoilage
    dark_spot_ripe_fraction: float = 0.04      # moderate spotting -> ripening
    # Mold / fungal-fuzz detection: low-saturation, textured, off-colour patches.
    mold_saturation_max: float = 55.0     # mold tends to be desaturated (grey/white/green fuzz)
    mold_min_fraction: float = 0.015      # fraction of food area that looks mold-like
    # Shrivel / surface-damage proxy: high local texture variance on the food.
    texture_shrivel_std: float = 62.0
    # Colour uniformity: how much hue spread suggests uneven discoloration.
    discoloration_hue_std: float = 34.0


@dataclass
class AggregationConfig:
    # A view is considered "significant evidence of spoilage" if its spoiled
    # probability exceeds this, even if other views look fine. The final result
    # then reflects the worst significant view rather than a blind average.
    spoiled_significant_threshold: float = 45.0
    ripe_significant_threshold: float = 55.0
    # How strongly the worst view dominates the averaged view (0..1). 1.0 = only
    # the worst view matters; 0.0 = pure average. 0.65 leans toward the worst.
    worst_view_weight: float = 0.65
    # Views whose image_quality is "poor" are down-weighted, not discarded.
    poor_quality_weight: float = 0.35


@dataclass
class ModelConfig:
    # Backend selection: "auto" picks a trained model if one is found, else the
    # heuristic OpenCV backend. Force with "heuristic", "tflite", or "keras".
    backend: str = "auto"
    tflite_model_path: str = "models/freshiq_model.tflite"
    keras_model_path: str = "models/freshiq_model.keras"
    # JSON file written by the training pipeline describing the trained model's
    # class order (foods x conditions). Required to interpret a trained model.
    labels_path: str = "models/labels.json"
    # Input size the model expects (square).
    #
    # 128, not 224: the training images are natively 128x128, so upscaling to 224
    # adds literally no information while costing ~3x the arithmetic. On a Pi Zero
    # that is the difference between usable and not. labels.json carries the real
    # value and overrides this at load time, so the two can never drift.
    input_size: int = 128


@dataclass
class Config:
    conditions: List[str] = field(default_factory=lambda: list(DEFAULT_CONDITIONS))
    foods: List[str] = field(default_factory=lambda: list(DEFAULT_FOODS))
    quality: QualityThresholds = field(default_factory=QualityThresholds)
    heuristic: HeuristicThresholds = field(default_factory=HeuristicThresholds)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Loading / overriding
# --------------------------------------------------------------------------- #
def _apply_overrides(cfg: Config, data: dict) -> Config:
    """Shallow-merge a dict of overrides onto a Config (one level into sub-configs)."""
    for key, value in data.items():
        if not hasattr(cfg, key):
            continue
        current = getattr(cfg, key)
        # Nested dataclasses (quality, heuristic, ...) -> update their fields.
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if hasattr(current, sub_key):
                    setattr(current, sub_key, sub_val)
        else:
            setattr(cfg, key, value)
    return cfg


def load_config(path: Optional[str] = None) -> Config:
    """
    Build a Config, optionally overriding defaults from a JSON file.

    Search order for the JSON file:
      1. explicit ``path`` argument
      2. ``FRESHIQ_CONFIG`` environment variable
      3. ``ai/config.json`` next to this module (if it exists)
    """
    cfg = Config()

    candidate = path or os.environ.get("FRESHIQ_CONFIG")
    if candidate is None:
        local = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(local):
            candidate = local

    if candidate and os.path.exists(candidate):
        with open(candidate, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        cfg = _apply_overrides(cfg, data)

    _resolve_model_paths(cfg)
    return cfg


def _resolve_model_paths(cfg: Config) -> None:
    """
    Anchor relative model paths to the project root, not the current directory.

    "models/freshiq_model.tflite" is relative, so whether the trained model is
    found used to depend on where the process happened to be started from. Get
    that wrong and there is no error - the backend quietly falls back to the
    heuristic and every result is subtly worse, reported only as
    "backend: heuristic" in one startup line. Anchoring to the package's parent
    makes the model load from any working directory.

    An ABSOLUTE path set by the user is left exactly as given.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for attr in ("tflite_model_path", "keras_model_path", "labels_path"):
        value = getattr(cfg.model, attr, None)
        if value and not os.path.isabs(value):
            setattr(cfg.model, attr, os.path.join(root, value))


# A module-level default instance for convenience. Callers that need isolation
# (e.g. tests) should call load_config() to get their own copy.
DEFAULT_CONFIG = load_config()


def get_config() -> Config:
    """Return a deep copy of the default config so callers can mutate safely."""
    return deepcopy(DEFAULT_CONFIG)
