"""Small shared helpers with no heavy dependencies."""

from __future__ import annotations

from typing import Dict, List


def to_int_percentages(probs: Dict[str, float]) -> Dict[str, int]:
    """
    Convert a dict of non-negative scores into integer percentages that sum to
    exactly 100, using the largest-remainder method (avoids the rounding drift
    you get from naive round()).

    If all scores are zero the mass is spread as evenly as possible.
    """
    keys = list(probs.keys())
    if not keys:
        return {}
    values = [max(0.0, float(probs[k])) for k in keys]
    total = sum(values)
    if total <= 0:
        # Even split.
        base = 100 // len(keys)
        out = {k: base for k in keys}
        for k in keys[: 100 - base * len(keys)]:
            out[k] += 1
        return out

    scaled = [v / total * 100.0 for v in values]
    floors = [int(x) for x in scaled]
    remainder = 100 - sum(floors)
    # Distribute the remaining points to the largest fractional parts.
    fracs = sorted(range(len(keys)), key=lambda i: scaled[i] - floors[i], reverse=True)
    for i in fracs[:remainder]:
        floors[i] += 1
    return {keys[i]: floors[i] for i in range(len(keys))}


def normalise(probs: Dict[str, float]) -> Dict[str, float]:
    """Return a copy of ``probs`` scaled so the values sum to 1.0."""
    total = sum(max(0.0, v) for v in probs.values())
    if total <= 0:
        n = len(probs)
        return {k: 1.0 / n for k in probs} if n else {}
    return {k: max(0.0, v) / total for k, v in probs.items()}


def dedupe_preserve_order(items: List[str]) -> List[str]:
    """Remove duplicates from a list while keeping first-seen order."""
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def load_interpreter_class():
    """
    Import a TFLite Interpreter, trying the packages newest-first.

    Returns (InterpreterClass, package_name).

    WHY THREE OPTIONS, in this order:

      1. ai-edge-litert   Google's CURRENT name for the standalone runtime.
                          This is what you should install on an Arduino UNO Q or
                          a Raspberry Pi today:  pip install ai-edge-litert
      2. tflite-runtime   The older name. Still works, but its ARM64 wheels stop
                          at Python 3.11 and Debian's own package is pinned to
                          TensorFlow 2.5 - so on a modern board it often will not
                          install at all.
      3. tensorflow       Works, but is ~400 MB installed. Fine on a dev laptop
                          or the UNO Q's 2-4 GB, hopeless on a 512 MB board.

    Raises ImportError naming all three if none is available, because "No module
    named tflite_runtime" on its own sends people down the wrong path.
    """
    attempts = []
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore
        return Interpreter, "ai-edge-litert"
    except Exception as exc:  # pragma: no cover - depends on environment
        attempts.append(f"ai_edge_litert ({exc.__class__.__name__})")
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
        return Interpreter, "tflite-runtime"
    except Exception as exc:  # pragma: no cover
        attempts.append(f"tflite_runtime ({exc.__class__.__name__})")
    try:
        from tensorflow.lite import Interpreter  # type: ignore
        return Interpreter, "tensorflow"
    except Exception as exc:  # pragma: no cover
        attempts.append(f"tensorflow ({exc.__class__.__name__})")

    raise ImportError(
        "No TFLite interpreter available. Tried: " + ", ".join(attempts) +
        ".\nInstall one with:  pip install ai-edge-litert"
    )
