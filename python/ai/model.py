"""
Model backends and backend selection.

FreshIQ supports three interchangeable backends behind one interface:

  * HeuristicBackend  - classical OpenCV rules (ai/heuristic.py). No training,
                        no extra dependencies. Always available.
  * TFLiteBackend     - a trained TensorFlow Lite model. Lightweight, ideal for
                        the Arduino UNO Q / edge devices. Preferred when present.
  * KerasBackend      - a trained Keras/TensorFlow model (dev machines).

The trained backends predict the CONDITION (fresh/ripe/spoiled or whatever
classes the model was trained on — read from models/labels.json). Food identity
comes from an explicit ``food_hint`` if the operator provides one, otherwise from
the heuristic colour guess. This mirrors the dataset layout in the spec
(dataset/train/<condition>/...), which is condition-labelled. A separate food
model can be added later; see docs/MODEL.md.

All heavy imports (tensorflow / tflite_runtime) are done lazily so that importing
FreshIQ, and running the heuristic backend, never requires them.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

from . import heuristic, preprocessing
from .config import Config
from .utils import load_interpreter_class, normalise


class BackendResult(dict):
    """Same shape the heuristic backend returns (food, food_confidence, probs, evidence, features)."""


class ModelBackend:
    name = "base"

    def predict(self, frame: np.ndarray, config: Config,
                food_hint: Optional[str] = None) -> Dict:
        raise NotImplementedError


class HeuristicBackend(ModelBackend):
    name = "heuristic"

    def predict(self, frame, config, food_hint=None):
        return heuristic.classify(frame, config, food_hint=food_hint)


def _load_labels(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class _TrainedBackend(ModelBackend):
    """Shared logic for NN backends: run the net for condition, borrow identity/evidence."""

    def __init__(self, classes: List[str], input_size: int):
        self.classes = classes
        self.input_size = input_size

    def _raw_predict(self, tensor: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, frame, config, food_hint=None):
        tensor = preprocessing.prepare_for_model(frame, self.input_size)
        raw = self._raw_predict(tensor).astype(np.float64).reshape(-1)
        if raw.shape[0] != len(self.classes):
            raise ValueError(
                f"Model output size {raw.shape[0]} != number of labels {len(self.classes)}")
        # Softmax if the outputs are logits; if they already look like probs, keep them.
        if raw.min() < 0 or raw.max() > 1.0 or abs(raw.sum() - 1.0) > 0.05:
            ex = np.exp(raw - raw.max())
            raw = ex / ex.sum()
        probs = {self.classes[i]: float(raw[i]) for i in range(len(self.classes))}

        # Map the model's classes onto the configured condition names where they
        # match; unknown classes are still reported as-is.
        probs = normalise(probs)

        # Identity + human-readable evidence come from the heuristic pass (cheap),
        # unless the operator supplied a hint.
        h = heuristic.classify(frame, config, food_hint=food_hint)
        food = food_hint if (food_hint and food_hint in config.foods) else h["food"]
        food_conf = 0.99 if (food_hint and food_hint in config.foods) else h["food_confidence"]

        # Build evidence: keep heuristic evidence but only if the NN also leans
        # away from "fresh", so we do not attach spoilage words to a fresh call.
        top = max(probs, key=probs.get)
        evidence = h["evidence"] if top != "fresh" else []

        return {
            "food": food,
            "food_confidence": food_conf,
            "probs": probs,
            "evidence": evidence,
            "features": h.get("features", {}),
        }


class TFLiteBackend(_TrainedBackend):
    name = "tflite"

    def __init__(self, model_path: str, classes: List[str], input_size: int):
        super().__init__(classes, input_size)
        Interpreter, self.runtime = load_interpreter_class()
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def _raw_predict(self, tensor):
        inp = self.input_details[0]
        data = tensor
        # Handle quantised (uint8/int8) models.
        if inp["dtype"] in (np.uint8, np.int8):
            scale, zero = inp["quantization"]
            if scale == 0:
                scale = 1.0 / 255
            data = (tensor / scale + zero).astype(inp["dtype"])
        self.interpreter.set_tensor(inp["index"], data.astype(inp["dtype"]))
        self.interpreter.invoke()
        out = self.output_details[0]
        result = self.interpreter.get_tensor(out["index"])
        if out["dtype"] in (np.uint8, np.int8):
            scale, zero = out["quantization"]
            result = (result.astype(np.float32) - zero) * (scale if scale else 1.0)
        return result


class KerasBackend(_TrainedBackend):
    name = "keras"

    def __init__(self, model_path: str, classes: List[str], input_size: int):
        super().__init__(classes, input_size)
        from tensorflow import keras  # type: ignore
        self.model = keras.models.load_model(model_path)

    def _raw_predict(self, tensor):
        return self.model.predict(tensor, verbose=0)


def build_backend(config: Config) -> ModelBackend:
    """
    Select a backend according to config.model.backend:
      "auto"      -> two-head model if present, else tflite, else keras, else heuristic
      "multihead" -> force the two-head (condition + food) backend
      "tflite"    -> force tflite (raises if unavailable)
      "keras"     -> force keras
      "heuristic" -> force heuristic
    """
    mc = config.model
    want = (mc.backend or "auto").lower()

    def try_multihead() -> Optional[ModelBackend]:
        # Two-head (condition + food) models need their own reader; see ai/multihead.py.
        from . import multihead
        return multihead.try_build(_load_labels(mc.labels_path), config)

    def try_tflite() -> Optional[ModelBackend]:
        labels = _load_labels(mc.labels_path)
        if labels is None or not os.path.exists(mc.tflite_model_path):
            return None
        classes = labels["classes"]
        size = labels.get("input_size", mc.input_size)
        return TFLiteBackend(mc.tflite_model_path, classes, size)

    def try_keras() -> Optional[ModelBackend]:
        labels = _load_labels(mc.labels_path)
        if labels is None or not os.path.exists(mc.keras_model_path):
            return None
        classes = labels["classes"]
        size = labels.get("input_size", mc.input_size)
        return KerasBackend(mc.keras_model_path, classes, size)

    if want == "heuristic":
        return HeuristicBackend()
    if want == "multihead":
        b = try_multihead()
        if b is None:
            raise FileNotFoundError(
                "multihead backend requested but no two-head model/labels found "
                f"({mc.labels_path} must have type 'condition+food')")
        return b
    if want == "tflite":
        # A two-head .tflite must be read by the multi-head backend, not this one.
        b = try_multihead() or try_tflite()
        if b is None:
            raise FileNotFoundError(
                f"tflite backend requested but model/labels not found "
                f"({mc.tflite_model_path} / {mc.labels_path})")
        return b
    if want == "keras":
        b = try_keras()
        if b is None:
            raise FileNotFoundError("keras backend requested but model/labels not found")
        return b

    # auto
    for factory in (try_multihead, try_tflite, try_keras):
        try:
            b = factory()
            if b is not None:
                return b
        except Exception:
            # A broken trained model should not take down the pipeline; fall back.
            continue
    return HeuristicBackend()
