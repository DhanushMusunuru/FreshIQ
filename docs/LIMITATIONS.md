# Limitations — read this

FreshIQ is a **visual estimation** tool. Being honest about what it can and cannot
do is part of the deliverable.

## What FreshIQ does NOT do

- **It does not measure food safety.** It cannot detect pathogens, toxins,
  bacterial load, pesticide residue, internal rot with no surface sign, or chemical
  spoilage. A clean-looking item can still be unsafe; a blemished item can still be
  fine. **Do not use FreshIQ to decide whether food is safe to eat.**
- **The percentages are not physical measurements.** `"spoiled": 80` means the
  model is ~80% confident the *appearance* matches its "spoiled" class — **not**
  that the item is 80% rotten.
- **It only sees the surface.** Internal bruising, core rot, and hidden mould are
  invisible to a camera. The rotating tray helps by showing multiple angles, but
  the underside/interior can still be missed.

## Accuracy claims

- **No accuracy is claimed for the heuristic backend.** It encodes reasonable
  visual rules but has not been validated on real produce. Treat it as a baseline.
- **The neural backend's accuracy is whatever your test set shows — and nothing
  more.** Run `training/evaluate.py` and report the real confusion matrix. A model
  is only as good as its data; a small or biased dataset gives a fragile model that
  may fail on foods, lighting, or backgrounds it didn't see in training.

## Known technical limitations

- **Food identification from colour is weak.** The heuristic can't reliably tell a
  red tomato from a red apple. Provide a `food_hint` when the rig knows what it
  loaded, or train a dedicated food-identity model. Default identity confidence is
  intentionally low (~0.2–0.35) to signal this.
- **Segmentation is simple.** `ai/preprocessing.segment_food` assumes a single,
  roughly centred item on a relatively plain tray. Cluttered backgrounds, multiple
  items, or a busy tray pattern will degrade it. Cluttered/multi-item frames are
  meant to be caught by the `multiple_objects` quality check.
- **Lighting sensitivity.** Colour-based cues shift under different light. Train and
  infer under the same lighting.
- **Condition classes are coarse.** `fresh / ripe / spoiled` is a simplification;
  real ripeness is a continuum and differs per food.
- **Aggregation heuristics are tunable, not proven.** The worst-significant-view
  logic reflects the brief's intent (a mouldy back should win over a fresh front),
  but the exact thresholds in `config.aggregation` should be tuned on your data.

## Responsible use

Present FreshIQ's output as **guidance for visual sorting/triage**, always with the
disclaimer that it is an appearance-based estimate, not a food-safety
determination. `backend/service.py` attaches that disclaimer to every record.
