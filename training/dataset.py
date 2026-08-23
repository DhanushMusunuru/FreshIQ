"""
Dataset loading and augmentation for training.

TWO LAYOUTS ARE SUPPORTED, and the one you use decides what the model predicts.

  (A) FLAT -- condition only. One output head.

        dataset/train/fresh/*.jpg
        dataset/train/spoiled/*.jpg

  (B) NESTED -- condition AND food. Two output heads.

        dataset/train/fresh/tomato/*.jpg
        dataset/train/fresh/banana/*.jpg
        dataset/train/spoiled/tomato/*.jpg
        ...

Layout (B) is what notebooks/freshiq_kaggle_training.ipynb produces and what the
MultiHead backend in ai/multihead.py expects. Layout (A) still works and yields a
plain single-head model that ai/model.py's TFLiteBackend reads. Either way the
class order is captured in models/labels.json, and inference reads it back, so the
two can never silently drift apart.

The layout is DETECTED, not configured -- one less thing to get wrong.

Requires TensorFlow (only for training; heuristic inference does not need it).
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _require_tf():
    try:
        import tensorflow as tf  # noqa: F401
        return tf
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "TensorFlow is required for training. Install it with:\n"
            "    pip install -r requirements-train.txt\n"
            "(Inference with the heuristic backend does NOT need TensorFlow.)"
        ) from exc


def _subdirs(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return sorted(d for d in os.listdir(path)
                  if os.path.isdir(os.path.join(path, d)) and not d.startswith("."))


def discover_classes(split_dir: str) -> List[str]:
    """Sorted list of condition sub-folders in a split directory."""
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    classes = _subdirs(split_dir)
    if not classes:
        raise ValueError(f"No condition sub-folders found in {split_dir}")
    return classes


def detect_layout(split_dir: str) -> Tuple[str, List[str], List[str]]:
    """
    Work out whether this split is FLAT or NESTED.

    Returns (layout, conditions, foods) where layout is "flat" or "nested" and
    foods is [] for a flat layout.
    """
    conditions = discover_classes(split_dir)
    foods: List[str] = []
    nested = False
    for cond in conditions:
        subs = _subdirs(os.path.join(split_dir, cond))
        if subs:
            nested = True
            foods.extend(subs)
    if not nested:
        return "flat", conditions, []

    foods = sorted(set(foods))
    # A half-nested tree (some conditions with food folders, some without) means
    # images would be silently skipped. Fail loudly instead.
    for cond in conditions:
        if not _subdirs(os.path.join(split_dir, cond)):
            raise ValueError(
                f"Mixed layout: '{cond}' has no food sub-folders but other "
                f"conditions do. Use one layout consistently.")
    return "nested", conditions, foods


def list_files(split_dir: str, conditions: List[str],
               foods: Optional[List[str]]) -> Tuple[List[str], List[int], List[int]]:
    """Return (paths, condition_indices, food_indices). food_indices is [] if flat."""
    paths, cond_idx, food_idx = [], [], []
    for ci, cond in enumerate(conditions):
        cond_dir = os.path.join(split_dir, cond)
        if foods:
            for fi, food in enumerate(foods):
                d = os.path.join(cond_dir, food)
                if not os.path.isdir(d):
                    continue
                for f in sorted(os.listdir(d)):
                    if f.lower().endswith(IMAGE_EXTS):
                        paths.append(os.path.join(d, f))
                        cond_idx.append(ci)
                        food_idx.append(fi)
        else:
            for f in sorted(os.listdir(cond_dir)):
                if f.lower().endswith(IMAGE_EXTS):
                    paths.append(os.path.join(cond_dir, f))
                    cond_idx.append(ci)
    return paths, cond_idx, food_idx


def build_datasets(data_dir: str, image_size: int = 128, batch_size: int = 32,
                   augment: bool = True, seed: int = 1337):
    """
    Build tf.data pipelines for train/validation/test.

    Returns (train_ds, val_ds, test_ds_or_None, conditions, foods).
    ``foods`` is [] for a flat (condition-only) layout.

    PREPROCESSING CONTRACT -- this must match ai/preprocessing.prepare_for_model
    exactly: RGB, resized to image_size, scaled to [0, 1]. The [-1, 1] shift that
    MobileNetV2 wants happens INSIDE the model as a Rescaling layer. Training with
    one normalisation and inferring with another produces a model that runs fine
    and is confidently wrong -- the worst kind of bug.
    """
    tf = _require_tf()
    AUTOTUNE = tf.data.AUTOTUNE

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")
    test_dir = os.path.join(data_dir, "test")

    layout, conditions, foods = detect_layout(train_dir)
    n_cond, n_food = len(conditions), len(foods)

    # Augmentation is deliberately conservative for tray-mounted produce: flips,
    # small rotation/zoom, gentle brightness and contrast. We avoid heavy colour
    # distortion and cutout because those can FABRICATE spoilage-looking
    # artefacts on a fresh item and teach the model exactly the wrong lesson.
    aug_layers = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomBrightness(0.12, value_range=(0.0, 1.0)),
        tf.keras.layers.RandomContrast(0.12),
    ], name="augmentation")

    def decode(path):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3,
                                 expand_animations=False)
        img = tf.image.resize(img, [image_size, image_size], method="area")
        return tf.cast(img, tf.float32) / 255.0          # -> [0, 1]

    def load(split_dir, training):
        if not os.path.isdir(split_dir):
            return None
        paths, ci, fi = list_files(split_dir, conditions, foods)
        if not paths:
            return None

        yc = tf.one_hot(ci, n_cond)
        if foods:
            yf = tf.one_hot(fi, n_food)
            ds = tf.data.Dataset.from_tensor_slices((paths, yc, yf))
            ds = ds.map(lambda p, c, f: (decode(p), {"condition": c, "food": f}),
                        num_parallel_calls=AUTOTUNE)
        else:
            ds = tf.data.Dataset.from_tensor_slices((paths, yc))
            ds = ds.map(lambda p, c: (decode(p), {"condition": c}),
                        num_parallel_calls=AUTOTUNE)

        if training:
            ds = ds.shuffle(min(len(paths), 10_000), seed=seed,
                            reshuffle_each_iteration=True)
            if augment:
                ds = ds.map(lambda x, y: (aug_layers(x, training=True), y),
                            num_parallel_calls=AUTOTUNE)
        return ds.batch(batch_size).prefetch(AUTOTUNE)

    train_ds = load(train_dir, training=True)
    val_ds = load(val_dir, training=False)
    test_ds = load(test_dir, training=False)

    if train_ds is None:
        raise ValueError(f"No training images found under {train_dir}")
    if val_ds is None:
        raise ValueError(
            f"No validation images found under {val_dir}. You need a validation "
            "split for early stopping -- see docs/TRAINING.md.")

    return train_ds, val_ds, test_ds, conditions, foods


def count_images(split_dir: str):
    """
    Return (total, {condition: count}, layout) for a split, for sanity checks.

    Look at the per-class counts before training. Wildly unbalanced classes mean
    the model can score well by always guessing the majority class.
    """
    layout, conditions, foods = detect_layout(split_dir)
    paths, ci, _ = list_files(split_dir, conditions, foods or None)
    counts = {c: 0 for c in conditions}
    for i in ci:
        counts[conditions[i]] += 1
    return len(paths), counts, layout
