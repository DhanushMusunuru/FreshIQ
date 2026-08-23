# Training

Train/fine-tune the deep-learning backend on your own produce photos. Everything
here is optional — the heuristic backend works without it — but a trained model is
expected to be materially more accurate.

```bash
pip install -r requirements-train.txt   # adds TensorFlow
```

## 1. Dataset layout

```
dataset/
  train/       fresh/  ripe/  spoiled/
  validation/  fresh/  ripe/  spoiled/
  test/        fresh/  ripe/  spoiled/
```

The **sub-folder names become the class names** (order captured in
`models/labels.json`). Add a folder (e.g. `overripe/`) in every split to add a
class. Keep the same classes across all three splits.

## 2. Image requirements

- **Format:** JPG/PNG, RGB. Any resolution ≥ 224×224 (they're resized to 224).
- **Framing:** one item, roughly centred, filling a good part of the frame — the
  same view the rig produces. Include the tray background you'll use in production
  so the model sees realistic context.
- **Lighting:** shoot under the **actual rig lighting**. A model trained on bright
  daylight will do poorly under LED ring light, and vice-versa.
- **Coverage:** capture each item from **multiple angles** (the tray rotates), and
  include a realistic mix of good and bad examples, borderline cases included.
- **Labelling honesty:** if you can't tell fresh from ripe in a photo, the model
  won't either. Be consistent; ambiguous images hurt more than they help.

## 3. Recommended dataset size

This is a prototype guideline, not a guarantee of accuracy:

| Tier | Images per class (train) | Expectation |
|---|---|---|
| Bare minimum (demo) | ~100 | Works in a demo; fragile, overfits easily |
| Reasonable | 300–500 | Usable prototype accuracy |
| Good | 1,000+ | Noticeably more robust |

Because we use **transfer learning**, you can get a demo working with fewer
images than training from scratch would need. Balance the classes — similar counts
per class — or the model will just predict the majority class.

## 4. Train / validation / test split

- **train** (~70%): the model learns from these.
- **validation** (~15%): used *during* training for early stopping / LR scheduling.
  The model never trains on these, but they influence when training stops.
- **test** (~15%): touched **only once**, at the very end, for the number you
  report. Never tune against the test set.

Split by **item**, not by photo: all angles of the same physical tomato must go to
the *same* split, or near-duplicate views leak between train and test and inflate
your score.

## 5. Augmentation

Applied to the training set only (`training/dataset.py`), deliberately
conservative for tray-mounted produce:

- horizontal flip, small rotation (±~8°), small zoom (±10%);
- small brightness (±0.12) and contrast (±0.12) jitter.

We avoid heavy colour distortion and cutout — those can *fabricate*
spoilage-looking artefacts and teach the model the wrong thing.

## 6. Train

```bash
# Feature-extraction pass (MobileNet frozen)
python training/train.py --data-dir dataset --epochs 25

# Optional gentle fine-tuning (unfreeze top layers; use a lower LR)
python training/train.py --data-dir dataset --fine-tune --lr 1e-4 --epochs 15
```

## 6b. Improving a model you ALREADY have

The commands above start from ImageNet. To make an existing model better —
for example on photos captured through your own rig — you must add
`--init-from`, or the run silently discards everything that model learned and
starts over:

```bash
python training/train.py --data-dir dataset --fine-tune --lr 1e-5 --epochs 10 \
    --init-from models/freshiq_model.keras
```

Input size and width are read back from the checkpoint's `labels.json`, so they
cannot drift. If your class list has changed, matching layers transfer and any
head whose width changed is reported and left freshly initialised:

```
[init] transferred 4 layer(s) from the existing model
[init] 'food' left randomly initialised: [(1280, 17), (17,)] -> [(1280, 8), (8,)]
```

Key parameters: `--image-size` (224), `--batch-size` (32), `--lr` (1e-3, or 1e-4
for `--fine-tune`), `--epochs`. Outputs go to `models/`:
`freshiq_model.keras`, `labels.json`, `training_log.csv`.

## 7. Overfitting detection & protection

Built in to `training/train.py`:

- **EarlyStopping** on `val_loss` (patience 5, restores best weights).
- **ReduceLROnPlateau** on `val_loss`.
- **Dropout (0.3)** in the head.

Signs of overfitting to watch for in `training_log.csv`: training accuracy climbs
while validation accuracy stalls or falls, or `val_loss` rises for several epochs.
Remedies: more/varied data, more augmentation, fewer epochs, or a lower LR.

## 8. Export to the edge device

```bash
python training/export.py --keras models/freshiq_model.keras            # float TFLite
python training/export.py --keras models/freshiq_model.keras \
    --quantize --rep-dir dataset/train                                  # int8, smaller/faster
```

Produces `models/freshiq_model.tflite`. Inference then **auto-selects** the TFLite
backend on the next run.

## 9. Evaluate — honestly

```bash
python training/evaluate.py --data-dir dataset --split test
```

Reports **accuracy, per-class precision/recall/F1, and a confusion matrix**, and
saves `models/evaluation.json`. Report these numbers as-is.

You can also measure **any** backend (including the zero-training heuristic) on a
labelled folder:

```bash
python scripts/compare_ground_truth.py --dir labelled --save report.json
```

> **Do not claim accuracy the held-out test set doesn't support.** A high training
> accuracy means nothing on its own. The confusion matrix on real, unseen photos
> is the truth. See `docs/LIMITATIONS.md`.
