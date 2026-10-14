# models/

Trained-model artifacts live here. They are gitignored (regenerate via `training/`):

- `freshiq_model.keras` — trained Keras model (from `training/train.py`)
- `freshiq_model.tflite` — exported edge model (from `training/export.py`)
- `labels.json` — class order + input size; **required** to interpret a model
- `training_log.csv`, `evaluation.json` — training/eval reports

With none of these present, FreshIQ automatically uses the heuristic backend.
