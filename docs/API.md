# API contract

```python
from ai import classify_image, classify_frame, inspect
```

All functions return plain, JSON-serialisable Python dicts. The backend does not
need to know anything about the model internals.

## `classify_image(image_path, food_hint=None, config=None)`
## `classify_frame(frame, food_hint=None, config=None)`

Single view. `frame` is a BGR `numpy` array (OpenCV's native format, e.g. from
`cv2.VideoCapture.read()`). `food_hint` is an optional known food name that
overrides automatic identification.

**Success:**

```json
{
  "status": "ok",
  "food": "tomato",
  "condition": "fresh",
  "fresh": 82,
  "ripe": 15,
  "spoiled": 3,
  "confidence": { "fresh": 82, "ripe": 15, "spoiled": 3 },
  "food_confidence": 0.99,
  "evidence": [],
  "image_quality": "good",
  "backend": "heuristic",
  "inference_ms": 17.2
}
```

- `condition` — the top class (`fresh` | `ripe` | `spoiled`, or your configured set).
- Flat keys (`fresh`/`ripe`/`spoiled`) **and** the grouped `confidence` object are
  both provided; they are identical integers summing to **100**.
- `food_confidence` — 0..1; **low** (~0.2–0.35) when identity was guessed from
  colour, ~0.99 when a `food_hint` was supplied.
- `evidence` — short human-readable reasons (empty when it looks fresh).
- `image_quality` — `good` | `acceptable` | `poor`.
- `backend` — which backend answered (`heuristic` | `tflite` | `keras`).

**Rejected (bad image):**

```json
{ "status": "invalid_image", "reason": "blurred", "image_quality": "poor" }
```

`reason` ∈ `too_dark`, `too_bright`, `blurred`, `no_food_detected`,
`food_too_far`, `camera_obstructed`, `multiple_objects`, `unreadable_file`.

## `inspect(images, food_hint=None, config=None)`

Multiple views of the **same** item (rotating tray). `images` is a list mixing
file paths and/or BGR frames.

```json
{
  "status": "ok",
  "food": "tomato",
  "condition": "spoiled",
  "confidence": { "fresh": 8, "ripe": 12, "spoiled": 80 },
  "evidence": ["visible fungal growth", "dark discoloration"],
  "image_quality": "good",
  "views_used": 3,
  "views_total": 4,
  "views": [ { ...per-view result... } ],
  "skipped_views": [ { "index": 2, "reason": "blurred" } ]
}
```

- **Aggregation is not a blind average.** If any view shows *significant* spoilage,
  the result leans to that worst view (config: `aggregation.worst_view_weight`,
  `spoiled_significant_threshold`) — a tomato fresh at the front but mouldy at the
  back is reported spoiled.
- Invalid views are **skipped, not fatal** (`views_used` < `views_total`); they're
  listed in `skipped_views`.
- If **every** view is invalid: `{"status": "invalid_image", "reason": ...,
  "views_used": 0}`.
- `views` contains each single-view result for traceability.

## Backend adapter — `backend/service.py`

```python
from backend.service import build_inspection_record
record = build_inspection_record(["front.jpg", "back.jpg"], food_hint="tomato")
```

Wraps the `inspect(...)` result with `inspection_id`, `device_id`, `ai_version`,
and a `disclaimer`, ready to write to Firebase/DB. A stdlib HTTP test server is
included (`python backend/service.py --serve`).

## Notes

- The percentages are **classification confidence**, not a physical measurement.
- `config` lets a caller pass a custom `ai.Config` (e.g. different thresholds or
  food/condition lists) without touching global state.
- After training/exporting a model, call `ai.reset_backend()` (or restart) so the
  new model is picked up.
