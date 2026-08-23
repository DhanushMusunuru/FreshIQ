"""
Multi-view aggregation (CORE AI TASKS #6 and #7).

The rotating tray photographs the same item from several angles. A tomato can
look perfect from the front and be mouldy on the back, so we must NOT blindly
average the per-view confidences. Instead we:

  1. Drop views that failed the quality gate (but remember how many).
  2. Down-weight (not discard) views graded "poor"/"acceptable".
  3. Combine per-view probabilities with a blend that leans toward the WORST
     significant view — if any view shows convincing spoilage, the final result
     reflects it rather than being diluted by the good-looking angles.
  4. Merge evidence from all views, keeping the reasons that drove the decision.

The output matches the FINAL INSPECTION RESULT schema in the spec.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .config import Config
from .utils import to_int_percentages, dedupe_preserve_order, normalise


def _weight_for_quality(quality: str, poor_weight: float) -> float:
    if quality == "good":
        return 1.0
    if quality == "acceptable":
        return 0.7
    return poor_weight  # "poor"


def _majority_food(views: List[Dict]) -> str:
    """Pick the food by confidence-weighted vote across valid views."""
    tally: Dict[str, float] = {}
    for v in views:
        food = v.get("food", "unknown")
        conf = float(v.get("food_confidence", 0.0)) + 0.01
        tally[food] = tally.get(food, 0.0) + conf
    # Prefer a confident non-"unknown" identity over "unknown".
    non_unknown = {k: val for k, val in tally.items() if k != "unknown"}
    pool = non_unknown if non_unknown else tally
    return max(pool, key=pool.get) if pool else "unknown"


def aggregate(view_results: List[Dict], config: Config) -> Dict:
    """
    Combine per-view single-image results into one inspection result.

    Each element of ``view_results`` is the dict produced by
    ``classifier.classify_*`` (it has ``status`` and, when ok, ``probs`` /
    ``evidence`` / ``food`` / ``image_quality``).

    Returns the final inspection dict.
    """
    ag = config.aggregation
    conditions = config.conditions
    spoiled_key = "spoiled" if "spoiled" in conditions else conditions[-1]

    valid = [v for v in view_results if v.get("status") == "ok"]
    invalid = [v for v in view_results if v.get("status") != "ok"]

    # If the usable views agree this is not produce, say so instead of falling
    # through to "no valid views", which would blame the camera for a correct
    # rejection. Freshness is undefined here, so no percentages are reported.
    not_food = [v for v in view_results if v.get("status") == "not_food"]
    if not valid and not_food and len(not_food) >= len(invalid):
        return {
            "status": "not_food",
            "food": not_food[0].get("food", "not_food"),
            "condition": None,
            "confidence": {c: 0 for c in conditions},
            "evidence": [],
            "image_quality": not_food[0].get("image_quality", "good"),
            "reason": "not_produce",
            "views_used": len(not_food),
            "views_total": len(view_results),
        }

    if not valid:
        # Nothing usable — report that honestly instead of guessing.
        reasons = dedupe_preserve_order(
            [v.get("reason", "invalid_image") for v in invalid]) or ["no_valid_views"]
        return {
            "status": "invalid_image",
            "food": "unknown",
            "condition": None,
            "confidence": {c: 0 for c in conditions},
            "evidence": [],
            "image_quality": "poor",
            "reason": reasons[0],
            "views_used": 0,
            "views_total": len(view_results),
        }

    # --- Weighted average across views --------------------------------------
    weighted: Dict[str, float] = {c: 0.0 for c in conditions}
    total_w = 0.0
    # --- Worst view (highest spoiled probability) ---------------------------
    worst_view = None
    worst_spoiled = -1.0

    for v in valid:
        probs = v.get("confidence_raw") or v.get("probs_normalised")
        if probs is None:
            # classify_* stores normalised probs under "confidence" as ints; rebuild floats
            probs = {c: v.get(c, 0) / 100.0 for c in conditions}
        w = _weight_for_quality(v.get("image_quality", "good"), ag.poor_quality_weight)
        for c in conditions:
            weighted[c] += probs.get(c, 0.0) * w
        total_w += w
        sp = probs.get(spoiled_key, 0.0)
        if sp > worst_spoiled:
            worst_spoiled = sp
            worst_view = probs

    avg = {c: (weighted[c] / total_w if total_w else 0.0) for c in conditions}

    # --- Blend average with the worst significant view ----------------------
    # Only let the worst view dominate if it is *significant* (real spoilage
    # evidence), so a single noisy frame does not condemn a fresh item.
    final = dict(avg)
    if worst_view is not None and worst_spoiled * 100.0 >= ag.spoiled_significant_threshold:
        wv = ag.worst_view_weight
        final = {c: (1 - wv) * avg.get(c, 0.0) + wv * worst_view.get(c, 0.0)
                 for c in conditions}

    final = normalise(final)
    percentages = to_int_percentages(final)
    condition = max(percentages, key=percentages.get)

    # --- Merge evidence -----------------------------------------------------
    # Collect evidence from all valid views, prioritising the worst view's.
    evidence: List[str] = []
    ordered_views = sorted(
        valid,
        key=lambda v: (v.get(spoiled_key, 0)),
        reverse=True,
    )
    for v in ordered_views:
        evidence.extend(v.get("evidence", []) or [])
    evidence = dedupe_preserve_order(evidence)[:6]

    # Image-quality summary: worst grade among the views actually used.
    grade_rank = {"good": 0, "acceptable": 1, "poor": 2}
    worst_grade = max((v.get("image_quality", "good") for v in valid),
                      key=lambda g: grade_rank.get(g, 0))

    result = {
        "status": "ok",
        "food": _majority_food(valid),
        "condition": condition,
        "confidence": percentages,
        "evidence": evidence,
        "image_quality": worst_grade,
        "views_used": len(valid),
        "views_total": len(view_results),
    }
    if invalid:
        result["skipped_views"] = [
            {"index": view_results.index(v), "reason": v.get("reason")} for v in invalid
        ]
    return result
