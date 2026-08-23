"""
Image-quality gate.

Runs BEFORE classification. If the frame is unusable we return a structured
"invalid_image" result instead of forcing a (meaningless) food prediction from a
bad picture. This directly implements CORE AI TASK #5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from . import preprocessing
from .config import Config


@dataclass
class QualityReport:
    usable: bool
    quality: str                 # "good" | "acceptable" | "poor"
    reason: Optional[str]        # machine-readable reason when not usable
    # Raw metrics, useful for debugging / tuning / tests.
    brightness: float
    overexposed_fraction: float
    focus_variance: float
    food_area_fraction: float
    object_count: int

    def to_dict(self) -> dict:
        return {
            "usable": self.usable,
            "quality": self.quality,
            "reason": self.reason,
            "metrics": {
                "brightness": round(self.brightness, 2),
                "overexposed_fraction": round(self.overexposed_fraction, 4),
                "focus_variance": round(self.focus_variance, 2),
                "food_area_fraction": round(self.food_area_fraction, 4),
                "object_count": self.object_count,
            },
        }


def check_quality(frame: np.ndarray, config: Config) -> QualityReport:
    """
    Inspect a BGR frame and decide whether it is good enough to classify.

    Reasons returned when unusable (matches the spec's vocabulary):
      "too_dark", "too_bright", "blurred", "no_food_detected", "food_too_far",
      "camera_obstructed", "multiple_objects".
    """
    q = config.quality
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    overexposed_fraction = float(np.mean(gray >= 250))
    focus_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    mask, bbox = preprocessing.segment_food(frame)
    food_area_fraction = float(np.count_nonzero(mask)) / float(mask.size)
    objects = preprocessing.list_food_contours(frame, q.multi_object_area_fraction)
    object_count = len(objects)

    def report(usable: bool, quality: str, reason: Optional[str]) -> QualityReport:
        return QualityReport(
            usable=usable, quality=quality, reason=reason,
            brightness=brightness, overexposed_fraction=overexposed_fraction,
            focus_variance=focus_variance, food_area_fraction=food_area_fraction,
            object_count=object_count,
        )

    # --- Hard failures (return invalid_image) --------------------------------
    if brightness < q.min_brightness:
        return report(False, "poor", "too_dark")
    if brightness > q.max_brightness or overexposed_fraction > q.max_overexposed_fraction:
        return report(False, "poor", "too_bright")
    if focus_variance < q.min_focus_variance:
        return report(False, "poor", "blurred")

    # Full-frame foreground with very low focus detail suggests the lens is
    # covered / the item is pressed against the camera.
    if food_area_fraction > q.max_food_area_fraction and focus_variance < q.min_focus_variance * 2:
        return report(False, "poor", "camera_obstructed")

    if food_area_fraction < q.min_food_area_fraction:
        # Distinguish "nothing there" from "something tiny / far away".
        reason = "no_food_detected" if food_area_fraction < q.min_food_area_fraction * 0.4 else "food_too_far"
        return report(False, "poor", reason)

    if object_count > q.max_expected_objects:
        return report(False, "poor", "multiple_objects")

    # --- Usable, but grade the quality ---------------------------------------
    quality = "good"
    # Soft signals nudge the grade to "acceptable" without rejecting the frame.
    if (focus_variance < q.min_focus_variance * 2.0
            or brightness < q.min_brightness * 1.4
            or brightness > q.max_brightness * 0.92
            or overexposed_fraction > q.max_overexposed_fraction * 0.5):
        quality = "acceptable"

    return report(True, quality, None)
