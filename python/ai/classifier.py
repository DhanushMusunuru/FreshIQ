"""
Public classification API (CORE AI TASK #11).

The backend should be able to call these functions without knowing anything about
how the model works internally:

    from ai import classify_image, classify_frame, inspect

    result = classify_image("photo.jpg")
    result = classify_frame(frame_from_opencv)          # numpy BGR array
    final  = inspect(["front.jpg", "left.jpg", "back.jpg"])   # rotating-tray views

All functions return plain Python dicts (JSON-serialisable).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Union

import numpy as np

from . import model as model_module
from . import preprocessing, quality_check
from .aggregation import aggregate
from .config import Config, get_config
from .utils import to_int_percentages

# Lazily-created, reused backend so we do not reload a trained model on every call.
_BACKEND = None
_BACKEND_CONFIG_ID = None


def _get_backend(config: Config):
    global _BACKEND, _BACKEND_CONFIG_ID
    key = (config.model.backend, config.model.tflite_model_path,
           config.model.keras_model_path, config.model.labels_path)
    if _BACKEND is None or _BACKEND_CONFIG_ID != key:
        _BACKEND = model_module.build_backend(config)
        _BACKEND_CONFIG_ID = key
    return _BACKEND


def reset_backend() -> None:
    """Force the backend to be rebuilt on the next call (e.g. after training)."""
    global _BACKEND, _BACKEND_CONFIG_ID
    _BACKEND = None
    _BACKEND_CONFIG_ID = None


def _classify_bgr(frame: np.ndarray, config: Config,
                  food_hint: Optional[str]) -> Dict:
    """Core single-frame path shared by classify_image and classify_frame."""
    start = time.perf_counter()
    frame = preprocessing.ensure_bgr(frame)

    # 1) Quality gate.
    qc = quality_check.check_quality(frame, config)
    if not qc.usable:
        return {
            "status": "invalid_image",
            "reason": qc.reason,
            "image_quality": qc.quality,
            "quality_metrics": qc.to_dict()["metrics"],
            "inference_ms": round((time.perf_counter() - start) * 1000, 1),
        }

    # 2) Backend prediction.
    backend = _get_backend(config)
    pred = backend.predict(frame, config, food_hint=food_hint)

    # The model can report "this is not produce at all". Freshness is undefined
    # for such a frame - the condition head was never trained on non-food - so we
    # return a rejection rather than a meaningless percentage.
    if pred.get("not_food"):
        return {
            "status": "not_food",
            "food": pred.get("food", "not_food"),
            "condition": None,
            "food_confidence": round(float(pred.get("food_confidence", 0.0)), 3),
            "confidence": {c: 0 for c in config.conditions},
            **{c: 0 for c in config.conditions},
            "evidence": [],
            "image_quality": qc.quality,
            "backend": backend.name,
            "inference_ms": round((time.perf_counter() - start) * 1000, 1),
        }

    probs = pred["probs"]  # dict condition -> float 0..1
    percentages = to_int_percentages(probs)
    condition = max(percentages, key=percentages.get)

    result = {
        "status": "ok",
        "food": pred["food"],
        "condition": condition,
        # Flat percentage keys (spec example #4): e.g. "fresh": 82, ...
        **percentages,
        # Also grouped, for the richer single-view schema.
        "confidence": percentages,
        "food_confidence": round(float(pred.get("food_confidence", 0.0)), 3),
        "evidence": pred.get("evidence", []),
        "image_quality": qc.quality,
        "backend": backend.name,
        "inference_ms": round((time.perf_counter() - start) * 1000, 1),
    }
    # Keep raw float probs around for the aggregator (not part of the public contract).
    result["confidence_raw"] = {k: float(probs.get(k, 0.0)) for k in config.conditions}
    return result


def classify_image(image_path: str, food_hint: Optional[str] = None,
                   config: Optional[Config] = None) -> Dict:
    """
    Classify a single image file.

    Args:
        image_path: path to a JPG/PNG/... readable by OpenCV.
        food_hint:  optional known food name (e.g. the operator selected "tomato").
                    When given, it overrides the automatic identity guess.
        config:     optional Config; defaults to the loaded default config.

    Returns a single-view result dict (see module docstring / docs/API.md).
    """
    config = config or get_config()
    try:
        frame = preprocessing.load_image(image_path)
    except ValueError as exc:
        return {"status": "invalid_image", "reason": "unreadable_file", "detail": str(exc)}
    return _classify_bgr(frame, config, food_hint)


def classify_frame(frame: np.ndarray, food_hint: Optional[str] = None,
                   config: Optional[Config] = None) -> Dict:
    """
    Classify a single camera frame (a BGR numpy array from cv2.VideoCapture).
    """
    config = config or get_config()
    return _classify_bgr(frame, config, food_hint)


def inspect(images: List[Union[str, np.ndarray]], food_hint: Optional[str] = None,
            config: Optional[Config] = None) -> Dict:
    """
    Run a full multi-view inspection over several images/frames of the SAME item
    (CORE AI TASK #7). Accepts a mix of file paths and BGR numpy frames.

    Returns the final aggregated inspection result. Also includes a "views" list
    with each per-view result for debugging/traceability.
    """
    config = config or get_config()
    view_results: List[Dict] = []
    for item in images:
        if isinstance(item, str):
            view_results.append(classify_image(item, food_hint=food_hint, config=config))
        else:
            view_results.append(classify_frame(item, food_hint=food_hint, config=config))

    final = aggregate(view_results, config)
    # Attach cleaned per-view results (drop internal raw floats from the copy).
    cleaned_views = []
    for v in view_results:
        vc = dict(v)
        vc.pop("confidence_raw", None)
        cleaned_views.append(vc)
    final["views"] = cleaned_views
    return final
