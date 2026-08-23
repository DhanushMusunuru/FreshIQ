"""
Training / fine-tuning entry point.

Trains a MobileNetV2 transfer-learning classifier on your own dataset and saves a
Keras model plus models/labels.json. Export to TFLite with training/export.py.

The architecture matches notebooks/freshiq_kaggle_training.ipynb exactly, so a
model trained here and a model trained there are interchangeable, and both load
in ai/model.py without code changes.

WHY THESE DEFAULTS (they are chosen for a Raspberry Pi Zero-class board, not for
leaderboard scores):

  --image-size 128   The training images are natively 128x128. Upscaling to 224
                     adds no information and costs roughly 3x the arithmetic.
  --alpha 0.35       MobileNetV2's width multiplier. 0.35 gives ~0.4M parameters
                     against 2.2M at 1.0 -- about 6x fewer operations per image.
                     ImageNet weights exist for alpha in {0.35,0.5,0.75,1.0,1.3,
                     1.4} at input sizes {96,128,160,192,224}; stick to those.

Head count follows the dataset layout automatically (see training/dataset.py):
  dataset/train/<condition>/*.jpg          -> one head  (condition)
  dataset/train/<condition>/<food>/*.jpg   -> two heads (condition + food)

Usage:
    # FROM SCRATCH (ImageNet start) -- pass 1, heads only, MobileNet frozen
    python training/train.py --data-dir dataset --epochs 25

    # ...then pass 2, gentle fine-tuning at a 10x lower learning rate
    python training/train.py --data-dir dataset --fine-tune --lr 1e-4 --epochs 15

    # IMPROVING AN EXISTING MODEL on photos from your own rig.
    # --init-from is what makes this an improvement rather than a restart:
    # without it the run rebuilds MobileNetV2 from ImageNet and discards
    # everything the current model learned.
    python training/train.py --data-dir dataset --fine-tune --lr 1e-5 --epochs 10 \
        --init-from models/freshiq_model.keras

Requires TensorFlow:  pip install -r requirements-train.txt
"""

from __future__ import annotations

import argparse
import json
import os

from dataset import build_datasets, count_images   # local import (run from training/)


def build_model(num_conditions: int, num_foods: int, image_size: int,
                alpha: float, fine_tune: bool):
    """
    MobileNetV2 body + one or two small heads.

        input (S,S,3) float [0,1]
              |
        Rescaling(2,-1)        <- [0,1] -> [-1,1], what MobileNetV2 expects
              |
        MobileNetV2(alpha)     <- ImageNet features; frozen unless --fine-tune
              |
        GlobalAveragePooling2D
              |
          Dropout(0.3)
           /        \\
     condition     food        <- second head only when num_foods > 0
    """
    import tensorflow as tf

    base = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        alpha=alpha,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = fine_tune
    if fine_tune:
        # Only unfreeze the top ~30 layers. Early layers detect universal things
        # (edges, colour blobs) that need no adjustment; late layers hold the
        # dataset-specific concepts worth retraining.
        for layer in base.layers[:-30]:
            layer.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0)(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = {"condition": tf.keras.layers.Dense(
        num_conditions, activation="softmax", dtype="float32", name="condition")(x)}
    if num_foods:
        outputs["food"] = tf.keras.layers.Dense(
            num_foods, activation="softmax", dtype="float32", name="food")(x)

    return tf.keras.Model(inputs, outputs, name="freshiq")


def shapes_match(src_weights, dst_weights) -> bool:
    """
    Can these two weight lists be copied across?

    Kept as a tiny pure function so it can be unit-tested without TensorFlow.
    """
    return (len(src_weights) == len(dst_weights)
            and all(tuple(a.shape) == tuple(b.shape)
                    for a, b in zip(src_weights, dst_weights)))


def transfer_weights(src_model, dst_model, verbose: bool = True):
    """
    Copy every layer whose weights are the same shape from src into dst.

    WHY THIS EXISTS
    ---------------
    Without it, --fine-tune rebuilds MobileNetV2 from ImageNet and throws away
    whatever you already trained. Fine-tuning on a few hundred rig photos
    starting from ImageNet gives a far worse model than continuing from one that
    already learned fresh-vs-spoiled from 97,000 images.

    Shape-checking per layer means a changed class list is handled gracefully:
    the body and any matching head transfer, and a head whose width changed is
    left freshly initialised rather than crashing the run.
    """
    src_by_name = {l.name: l for l in src_model.layers}
    copied, reinit = [], []
    for layer in dst_model.layers:
        src = src_by_name.get(layer.name)
        if src is None:
            continue
        sw, dw = src.get_weights(), layer.get_weights()
        if not dw:
            continue
        if shapes_match(sw, dw):
            layer.set_weights(sw)
            copied.append(layer.name)
        else:
            reinit.append((layer.name,
                           [tuple(w.shape) for w in sw],
                           [tuple(w.shape) for w in dw]))
    if verbose:
        print(f"[init] transferred {len(copied)} layer(s) from the existing model")
        for name, s, d in reinit:
            print(f"[init] '{name}' left randomly initialised: {s} -> {d} "
                  "(class list changed)")
    return copied, reinit


def main():
    ap = argparse.ArgumentParser(description="Train the FreshIQ classifier.")
    ap.add_argument("--data-dir", default="dataset", help="dataset root (train/validation/test)")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.35,
                    help="MobileNetV2 width multiplier; smaller = faster on edge hardware")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--init-from", default=None,
                    help="continue from an existing .keras model instead of ImageNet. "
                         "USE THIS when fine-tuning on rig photos: "
                         "--init-from models/freshiq_model.keras")
    ap.add_argument("--fine-tune", action="store_true",
                    help="unfreeze top MobileNet layers (use a lower --lr, e.g. 1e-4)")
    ap.add_argument("--food-loss-weight", type=float, default=0.5,
                    help="how much the food head matters vs condition (two-head only)")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--model-name", default="freshiq_model")
    ap.add_argument("--no-augment", action="store_true")
    args = ap.parse_args()

    import tensorflow as tf

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Resolve input size / width FIRST ------------------------------------
    # This must happen before build_datasets, because the data pipeline resizes
    # images to image_size. Resolving it afterwards would decode the dataset at
    # one size and build the model for another.
    image_size, alpha = args.image_size, args.alpha
    if args.init_from:
        if not os.path.exists(args.init_from):
            raise SystemExit(f"[error] --init-from file not found: {args.init_from}")
        lp = os.path.join(os.path.dirname(args.init_from) or ".", "labels.json")
        if os.path.exists(lp):
            with open(lp, encoding="utf-8") as fh:
                prev = json.load(fh)
            if prev.get("input_size") and prev["input_size"] != image_size:
                print(f"[init] labels.json says input_size={prev['input_size']}, "
                      f"overriding --image-size {image_size}")
                image_size = prev["input_size"]
            arch = str(prev.get("architecture", ""))
            if "alpha" in arch:
                try:
                    a = float(arch.split("alpha")[1])
                    if a != alpha:
                        print(f"[init] labels.json says alpha={a}, "
                              f"overriding --alpha {alpha}")
                        alpha = a
                except ValueError:
                    pass

    # --- Dataset sanity check ----------------------------------------------
    total_train, per_class, layout = count_images(os.path.join(args.data_dir, "train"))
    print(f"[data] layout: {layout}")
    print(f"[data] train images: {total_train}  per-condition: {per_class}")
    if total_train < 30 * max(len(per_class), 1):
        print("[warn] Very small training set. Expect weak, possibly unreliable "
              "results. See docs/TRAINING.md for recommended dataset sizes.")
    if per_class and max(per_class.values()) > 3 * max(min(per_class.values()), 1):
        print("[warn] Classes are heavily unbalanced. The model can score well by "
              "always guessing the majority class -- read the confusion matrix, "
              "not the accuracy.")

    train_ds, val_ds, test_ds, conditions, foods = build_datasets(
        args.data_dir, image_size, args.batch_size, augment=not args.no_augment)
    print(f"[data] conditions: {conditions}")
    print(f"[data] foods: {foods if foods else '(none -- single-head model)'}")

    model = build_model(len(conditions), len(foods), image_size, alpha, args.fine_tune)

    if args.init_from:
        print(f"[init] loading weights from {args.init_from}")
        previous = tf.keras.models.load_model(args.init_from, compile=False)
        transfer_weights(previous, model)
    else:
        print("[init] starting from ImageNet weights only.\n"
              "       If you meant to improve an already-trained model, stop and rerun with\n"
              "         --init-from models/freshiq_model.keras")

    losses = {"condition": "categorical_crossentropy"}
    weights = {"condition": 1.0}
    metrics = {"condition": ["accuracy"]}
    if foods:
        losses["food"] = "categorical_crossentropy"
        weights["food"] = args.food_loss_weight
        metrics["food"] = ["accuracy"]

    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss=losses, loss_weights=weights, metrics=metrics)
    model.summary()

    trainable = sum(int(tf.size(w)) for w in model.trainable_weights)
    print(f"[model] {model.count_params():,} params, {trainable:,} trainable "
          f"({100 * trainable / max(model.count_params(), 1):.1f}%)")

    keras_path = os.path.join(args.out_dir, f"{args.model_name}.keras")
    monitor = "val_condition_accuracy" if foods else "val_accuracy"
    callbacks = [
        # Overfitting protection: stop when val loss stops improving and rewind
        # to the best epoch we ever saw.
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            keras_path, monitor=monitor, mode="max", save_best_only=True),
        tf.keras.callbacks.CSVLogger(os.path.join(args.out_dir, "training_log.csv")),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    model.save(keras_path)
    print(f"[save] Keras model -> {keras_path}")

    # labels.json is how inference learns the class ORDER and input size. The
    # network only outputs bare arrays of numbers; without this file nothing
    # knows which number means "spoiled". The "type" field also tells
    # ai/model.py which backend to build.
    labels = {
        "type": "condition+food" if foods else "condition",
        "classes": conditions,
        "input_size": image_size,
        "architecture": f"MobileNetV2-alpha{alpha}",
        "preprocessing": "RGB float32 in [0,1]; model applies Rescaling(2,-1) internally",
    }
    if foods:
        labels["foods"] = foods
    labels_path = os.path.join(args.out_dir, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as fh:
        json.dump(labels, fh, indent=2)
    print(f"[save] labels -> {labels_path}")

    # --- Report on validation / test ---------------------------------------
    print(f"[eval] validation: {model.evaluate(val_ds, return_dict=True, verbose=0)}")
    if test_ds is not None:
        print(f"[eval] test: {model.evaluate(test_ds, return_dict=True, verbose=0)}")
        print("NOTE: report these HONEST numbers. Accuracy alone hides the error "
              "that matters -- run training/evaluate.py for the confusion matrix, "
              "and read the spoiled-called-fresh cell. See docs/LIMITATIONS.md.")

    print("\nNext: export to TFLite for the edge device:")
    print(f"    python training/export.py --keras {keras_path} "
          f"--image-size {image_size} --quantize")


if __name__ == "__main__":
    main()
