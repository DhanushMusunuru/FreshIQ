"""
Two-head (condition + food) trained backends.

The stock TFLiteBackend/KerasBackend in ai/model.py assume a single output tensor
(the condition). A model trained with notebooks/freshiq_kaggle_training.ipynb has
TWO output heads:

    condition -> fresh / spoiled          (the product)
    food      -> apple / banana / ...     (fixes the heuristic's weakest link)

This module adds backends that read both. It is purely additive: build_backend()
falls through to the single-head backends and then the heuristic exactly as
before if no multi-head labels.json is present.

A multi-head models/labels.json looks like:

    {
      "type": "condition+food",
      "classes": ["fresh", "spoiled"],
      "foods": ["apple", "banana", ...],
      "input_size": 128,
      "tflite_output_index": {"condition": 0, "food": 1}
    }

The ``tflite_output_index`` map matters: TFLite does NOT guarantee that output
tensor order matches the order the heads were declared in Keras. We resolve the
heads by declared index when available, and fall back to resolving them by tensor
width (2 vs. 13), which is unambiguous here.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from . import heuristic, preprocessing
from .config import Config
from .utils import load_interpreter_class, normalise


class _MultiHeadBackend:
    """Shared logic: run the net for BOTH condition and food identity."""

    name = "multihead"

    def __init__(self, classes: List[str], foods: List[str], input_size: int,
                 not_food_class: Optional[str] = None):
        self.classes = classes
        self.foods = foods
        self.input_size = input_size
        # Set only when the model was trained with non-produce negatives.
        self.not_food_class = not_food_class if (
            not_food_class and not_food_class in foods) else None

    # Subclasses return (condition_vector, food_vector) as 1-D float arrays.
    def _raw_predict(self, tensor: np.ndarray):
        raise NotImplementedError

    @staticmethod
    def _to_probs(raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float64).reshape(-1)
        # Softmax only if these look like logits rather than probabilities.
        if raw.min() < 0 or raw.max() > 1.0 or abs(raw.sum() - 1.0) > 0.05:
            ex = np.exp(raw - raw.max())
            raw = ex / ex.sum()
        return raw

    def predict(self, frame, config: Config, food_hint: Optional[str] = None) -> Dict:
        tensor = preprocessing.prepare_for_model(frame, self.input_size)
        cond_raw, food_raw = self._raw_predict(tensor)

        cond_vec = self._to_probs(cond_raw)
        if cond_vec.shape[0] != len(self.classes):
            raise ValueError(
                f"condition head size {cond_vec.shape[0]} != {len(self.classes)} labels")
        probs = normalise({self.classes[i]: float(cond_vec[i])
                           for i in range(len(self.classes))})

        # --- Food identity -------------------------------------------------
        # An explicit hint from the rig always wins: the operator knows what they
        # loaded, and that is more reliable than any classifier.
        if food_hint and food_hint in config.foods:
            food, food_conf = food_hint, 0.99
        else:
            food_vec = self._to_probs(food_raw)
            if food_vec.shape[0] != len(self.foods):
                raise ValueError(
                    f"food head size {food_vec.shape[0]} != {len(self.foods)} labels")
            i = int(np.argmax(food_vec))
            food, food_conf = self.foods[i], float(food_vec[i])

        # --- "That isn't produce" ------------------------------------------
        # A model trained with a not_food class can reject non-produce. When it
        # does, the freshness numbers are meaningless - the condition head was
        # never trained on these photos (their loss was masked out). Reporting
        # "82% fresh" for a photo of someone's hand would be worse than useless,
        # so we surface the rejection instead.
        if self.not_food_class and food == self.not_food_class:
            return {
                "food": self.not_food_class,
                "food_confidence": round(food_conf, 3),
                "probs": {c: 0.0 for c in self.classes},
                "evidence": [],
                "features": {},
                "not_food": True,
            }

        # --- Evidence ------------------------------------------------------
        # The net gives no human-readable reasons, so we borrow the heuristic's
        # (cheap, pure OpenCV) -- but only attach spoilage language when the net
        # itself leans away from "fresh".
        h = heuristic.classify(frame, config, food_hint=food)
        top = max(probs, key=probs.get)
        evidence = h["evidence"] if top != "fresh" else []

        return {
            "food": food,
            "food_confidence": round(food_conf, 3),
            "probs": probs,
            "evidence": evidence,
            "features": h.get("features", {}),
            "not_food": False,
        }


class MultiHeadTFLiteBackend(_MultiHeadBackend):
    name = "tflite_multihead"

    def __init__(self, model_path: str, classes, foods, input_size: int,
                 output_index: Optional[Dict[str, int]] = None,
                 not_food_class: Optional[str] = None):
        super().__init__(classes, foods, input_size, not_food_class)
        Interpreter, self.runtime = load_interpreter_class()
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        if len(self.output_details) < 2:
            raise ValueError(
                f"{model_path} has {len(self.output_details)} output(s); a multi-head "
                "model needs 2. Use the single-head TFLiteBackend instead.")

        # Resolve which output tensor is which head. Prefer the index recorded at
        # export time; otherwise fall back to matching by tensor width.
        self.i_cond, self.i_food = self._resolve_heads(output_index)

    def _resolve_heads(self, output_index):
        if output_index and "condition" in output_index and "food" in output_index:
            return int(output_index["condition"]), int(output_index["food"])
        widths = [int(o["shape"][-1]) for o in self.output_details]
        try:
            return widths.index(len(self.classes)), widths.index(len(self.foods))
        except ValueError as exc:
            raise ValueError(
                f"Cannot match output widths {widths} to condition "
                f"({len(self.classes)}) and food ({len(self.foods)}) heads.") from exc

    def _dequantise(self, detail, arr):
        if detail["dtype"] in (np.uint8, np.int8):
            scale, zero = detail["quantization"]
            return (arr.astype(np.float32) - zero) * (scale if scale else 1.0)
        return arr

    def _raw_predict(self, tensor):
        inp = self.input_details[0]
        data = tensor
        if inp["dtype"] in (np.uint8, np.int8):
            scale, zero = inp["quantization"]
            if scale == 0:
                scale = 1.0 / 255
            data = (tensor / scale + zero)
        self.interpreter.set_tensor(inp["index"], data.astype(inp["dtype"]))
        self.interpreter.invoke()

        d_cond = self.output_details[self.i_cond]
        d_food = self.output_details[self.i_food]
        return (self._dequantise(d_cond, self.interpreter.get_tensor(d_cond["index"])),
                self._dequantise(d_food, self.interpreter.get_tensor(d_food["index"])))


class MultiHeadKerasBackend(_MultiHeadBackend):
    name = "keras_multihead"

    def __init__(self, model_path: str, classes, foods, input_size: int,
                 not_food_class: Optional[str] = None):
        super().__init__(classes, foods, input_size, not_food_class)
        from tensorflow import keras  # type: ignore
        self.model = keras.models.load_model(model_path)

    def _raw_predict(self, tensor):
        out = self.model.predict(tensor, verbose=0)
        if isinstance(out, dict):
            return out["condition"], out["food"]
        # A list/tuple: match by width, since order is not guaranteed.
        arrs = [np.asarray(o).reshape(-1) for o in out]
        cond = next(a for a in arrs if a.shape[0] == len(self.classes))
        food = next(a for a in arrs if a.shape[0] == len(self.foods))
        return cond, food


def is_multihead(labels: Optional[dict]) -> bool:
    """True if a labels.json describes a two-head (condition + food) model."""
    if not labels:
        return False
    return bool(labels.get("foods")) and labels.get("type") == "condition+food"


def try_build(labels: Optional[dict], config: Config):
    """
    Return a multi-head backend if labels.json describes one and the model file
    exists, else None so the caller can fall through to the single-head path.
    """
    if not is_multihead(labels):
        return None

    mc = config.model
    classes = labels["classes"]
    foods = labels["foods"]
    size = labels.get("input_size", mc.input_size)
    out_idx = labels.get("tflite_output_index")

    not_food = labels.get("not_food_class")
    if os.path.exists(mc.tflite_model_path):
        return MultiHeadTFLiteBackend(mc.tflite_model_path, classes, foods, size,
                                      out_idx, not_food)
    if os.path.exists(mc.keras_model_path):
        return MultiHeadKerasBackend(mc.keras_model_path, classes, foods, size, not_food)
    return None
