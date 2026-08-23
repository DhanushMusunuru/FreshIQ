# Architecture

## Data flow

```
USB Camera (camera/capture.py)
      │  BGR frame(s)
      ▼
Quality check (ai/quality_check.py) ──► invalid_image? ──► return reason, stop
      │  usable frame
      ▼
Backend prediction (ai/model.py -> heuristic.py | tflite | keras)
      │  food + condition probabilities + evidence
      ▼
Single-view result (ai/classifier.py: classify_image / classify_frame)
      │  one dict per view
      ▼
Aggregation (ai/aggregation.py)  ◄── multiple rotating-tray views
      │  worst-significant-view logic
      ▼
Final inspection result  ──►  backend/service.py  ──►  (your Firebase / dashboard)
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `ai/config.py` | Central tunables: condition list, food list, thresholds, model paths. JSON-overridable so you can retune without code changes. |
| `ai/preprocessing.py` | Load images, normalise to BGR, segment the food from the tray, resize/normalise tensors for the NN. |
| `ai/quality_check.py` | Gate keeper. Rejects too-dark/too-bright/blurred/no-food/too-far/obstructed/multi-object frames **before** classification. |
| `ai/heuristic.py` | Classical-CV classifier: dark-spot ratio, mould-like texture, colour/discoloration, shrivel. Works with zero training. |
| `ai/food_rules.py` | Per-food colour bands and whether colour change means ripening vs. spoilage. Drives identity guessing and evidence text. |
| `ai/model.py` | Loads trained TFLite/Keras models, normalises their output, and auto-selects the best available backend. |
| `ai/aggregation.py` | Combines several views into one result, leaning toward the worst *significant* view rather than averaging. |
| `ai/classifier.py` | The public API: `classify_image`, `classify_frame`, `inspect`. Orchestrates quality → backend → result. |
| `camera/capture.py` | USB camera wrapper; can capture a sequence with a rotate-tray callback hook. |
| `backend/service.py` | Wraps an inspection into a record + a stdlib HTTP test server. The seam to the rest of FreshIQ. |
| `training/` | `train.py`, `export.py` (→ TFLite), `evaluate.py` (honest metrics). |

## Why a heuristic backend *and* a neural backend?

- **Get the pipeline working immediately** (Phase-1 priority in the brief) without
  waiting for a dataset or training.
- **Graceful degradation**: if a trained model is missing or fails to load, the
  system still returns a sane, structured answer instead of crashing.
- **Honest baseline**: you can measure the neural model against the heuristic with
  `scripts/compare_ground_truth.py` and only ship the NN if it actually wins.

The interface (`ai/model.py::ModelBackend`) is identical for all three, so
swapping backends never changes calling code.

## Extending the system

- **Add a food:** add its name to `config.foods` (or `ai/config.json`) and,
  optionally, a `FoodRule` in `ai/food_rules.py` for food-specific colour logic.
- **Add a condition class** (e.g. `overripe`): add it to `config.conditions` and
  create the matching `dataset/<split>/overripe/` folders, then retrain. The
  heuristic backend only understands the three core buckets, so a new class is
  only populated by the trained model.
- **Change thresholds:** copy `ai/config.example.json` to `ai/config.json`.
