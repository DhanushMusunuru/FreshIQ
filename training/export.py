"""
Export a trained Keras model to TensorFlow Lite for the edge device (UNO Q).

Produces models/freshiq_model.tflite. Optionally applies int8 quantisation
(smaller + faster on edge hardware) using a representative sample of your
training images.

Usage:
    python training/export.py --keras models/freshiq_model.keras
    python training/export.py --keras models/freshiq_model.keras \
        --quantize --rep-dir dataset/train

Requires TensorFlow:  pip install tensorflow
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def representative_dataset_gen(rep_dir: str, image_size: int, max_samples: int = 200):
    import cv2

    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        files.extend(glob.glob(os.path.join(rep_dir, "**", ext), recursive=True))
    files = files[:max_samples]
    if not files:
        raise ValueError(f"No representative images found under {rep_dir}")

    def gen():
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_size, image_size))
            img = img.astype(np.float32) / 255.0
            yield [np.expand_dims(img, 0)]

    return gen


def main():
    ap = argparse.ArgumentParser(description="Export Keras -> TFLite.")
    ap.add_argument("--keras", default="models/freshiq_model.keras")
    ap.add_argument("--out", default="models/freshiq_model.tflite")
    ap.add_argument("--image-size", type=int, default=128,
                    help="must match labels.json input_size (128 by default)")
    ap.add_argument("--quantize", action="store_true",
                    help="apply int8 quantisation (smaller/faster on edge)")
    ap.add_argument("--rep-dir", default="dataset/train",
                    help="folder of representative images for quantisation")
    args = ap.parse_args()

    import tensorflow as tf

    model = tf.keras.models.load_model(args.keras)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if args.quantize:
        converter.representative_dataset = representative_dataset_gen(
            args.rep_dir, args.image_size)
        # Full-integer quantisation; keep float I/O for easy Python use.
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        print("[export] applying int8 quantisation")

    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as fh:
        fh.write(tflite_model)

    size_kb = os.path.getsize(args.out) / 1024.0
    print(f"[export] wrote {args.out} ({size_kb:.1f} KB)")

    # --- Record which output tensor is which head --------------------------
    # TFLite does NOT guarantee that output tensor order matches the order the
    # heads were declared in Keras. If those get swapped, the food probabilities
    # are read as condition probabilities and everything is quietly wrong. We
    # resolve the heads by tensor width here, once, and write the answer into
    # labels.json so inference never has to guess.
    labels_path = os.path.join(os.path.dirname(args.out) or ".", "labels.json")
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as fh:
            labels = json.load(fh)

        interp = tf.lite.Interpreter(model_content=tflite_model)
        interp.allocate_tensors()
        outs = interp.get_output_details()
        widths = [int(o["shape"][-1]) for o in outs]
        print(f"[export] output widths: {widths}")

        if labels.get("type") == "condition+food" and len(outs) >= 2:
            try:
                labels["tflite_output_index"] = {
                    "condition": widths.index(len(labels["classes"])),
                    "food": widths.index(len(labels["foods"])),
                }
            except ValueError:
                raise SystemExit(
                    f"[error] Cannot match output widths {widths} to "
                    f"{len(labels['classes'])} conditions and "
                    f"{len(labels['foods'])} foods. Did labels.json come from a "
                    "different training run than this .keras file?")
            print(f"[export] head mapping: {labels['tflite_output_index']}")

        labels["input_size"] = args.image_size
        with open(labels_path, "w", encoding="utf-8") as fh:
            json.dump(labels, fh, indent=2)
        print(f"[export] updated {labels_path}")
    else:
        print(f"[warn] {labels_path} not found. Inference cannot interpret this "
              "model without it -- re-run training/train.py.")

    print("\nInference will now auto-select the TFLite backend "
          "(config.model.backend='auto'). Run: python scripts/test_image.py <img>")


if __name__ == "__main__":
    main()
