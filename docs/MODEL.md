# Model

## Backends

### 1. Heuristic (classical CV) — `ai/heuristic.py`
No training, no ML dependencies. It segments the food from the tray and computes
interpretable features on the food pixels:

- **dark-spot fraction** — pixels much darker than the food's median value →
  bruising / rot;
- **mould-like fraction** — low-saturation *and* locally textured (fuzzy) patches →
  fungal growth;
- **surface texture std** — proxy for shrivel / surface damage;
- **colour vs. food rule** — is the dominant hue where a healthy example of this
  food should be? For bananas/tomatoes a colour shift is treated as *ripening*;
  for others it leans toward spoilage;
- **hue spread** — uneven colouring across the surface.

These accumulate into fresh/ripe/spoiled scores and human-readable `evidence`.
It is a **transparent baseline**, not a validated classifier — see
`docs/LIMITATIONS.md`.

### 2. Trained neural network — `ai/model.py`
A **MobileNetV2** transfer-learning classifier (ImageNet features + a small custom
head) fine-tuned on your dataset, exported to **TensorFlow Lite** (optionally int8
quantised) for the edge device.

**Why MobileNetV2?**
- Designed for mobile/edge: ~2–4M parameters, few MFLOPs, runs on the UNO Q's
  Linux side via TFLite (see `docs/HARDWARE.md`).
- Transfer learning works well with small produce datasets.
- Clean, well-supported TFLite conversion incl. full-integer quantisation.

**Alternatives you can drop in** (same pipeline, change `build_model` in
`training/train.py`): EfficientNet-Lite0 (a bit more accurate, similar size), or a
YOLO-nano variant if you later need *detection/localisation* rather than
whole-image classification. The brief says avoid unnecessarily large models —
MobileNetV2 is the right default.

## What the trained model predicts

The dataset layout in the brief is **condition-labelled**
(`dataset/train/fresh|ripe|spoiled/`), so the trained model predicts the
**condition**. The output class order is written to `models/labels.json` at
export time and read back at inference so the two never drift.

**Food identity** comes from:
1. an explicit `food_hint` (the operator/rig knows what it loaded — most reliable), else
2. the heuristic colour guess (low confidence; colour alone can't separate a red
   tomato from a red apple).

A dedicated **food-identity model** is a clean future addition: train a second
MobileNet on food-labelled folders and predict both. The `labels.json` `type`
field (`"condition"` today) is where that extension plugs in. It was left out on
purpose to honour the brief's "don't over-engineer before the basics work".

## Food-specific logic — `ai/food_rules.py`

Different produce ripens/spoils differently, so one colour rule can't fit all:

- **Banana**: yellow → brown flecks is *ripening*, not spoilage.
- **Tomato**: green → red is *ripening*.
- **Potato**: dull by nature; green patches/sprouting are the spoilage signal.
- **Apple**: naturally varied colour; browning/bruising is the signal.

The trained model learns this directly from per-food data; the heuristic backend
uses the `FoodRule` table. Both are configurable/extensible.

## Output normalisation

Whatever a backend emits (logits or probabilities), `ai/model.py` softmaxes if
needed and normalises to sum to 1.0; `ai/utils.to_int_percentages` then converts
to **integers that sum to exactly 100** (largest-remainder method) for the API.
