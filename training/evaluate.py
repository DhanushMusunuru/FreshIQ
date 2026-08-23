"""
Honest evaluation of a trained model on the held-out test set (CORE AI TASK #9/#14).

Reports accuracy, per-class precision/recall/F1, and a confusion matrix, so you
can state real numbers instead of guessing. Works against the current backend
(TFLite if present, else Keras, else it will refuse — the heuristic backend has
no "trained accuracy" to report).

Usage:
    python training/evaluate.py --data-dir dataset
    python training/evaluate.py --data-dir dataset --split test

Requires TensorFlow only if evaluating a Keras model; TFLite eval needs the
tflite runtime (or tensorflow).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

# Make the repo root importable so we can reuse ai/*.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai import get_config                      # noqa: E402
from ai import model as model_module           # noqa: E402
from ai import preprocessing                    # noqa: E402


def load_split_files(data_dir: str, split: str, classes):
    """
    Collect (files, condition_labels, food_labels) from a split.

    Handles both dataset layouts (see training/dataset.py):
        flat:   dataset/<split>/<condition>/*.jpg
        nested: dataset/<split>/<condition>/<food>/*.jpg
    food_labels contains None for a flat layout.
    """
    files, conds, foods = [], [], []
    for cls in classes:
        d = os.path.join(data_dir, split, cls)
        if not os.path.isdir(d):
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for f in glob.glob(os.path.join(d, ext)):
                files.append(f); conds.append(cls); foods.append(None)
        # Nested: one more level down, the folder name is the food.
        for sub in sorted(os.listdir(d)):
            sd = os.path.join(d, sub)
            if not os.path.isdir(sd):
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                for f in glob.glob(os.path.join(sd, ext)):
                    files.append(f); conds.append(cls); foods.append(sub)
    return files, conds, foods


def main():
    ap = argparse.ArgumentParser(description="Evaluate the trained model honestly.")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--split", default="test", choices=["test", "validation", "train"])
    ap.add_argument("--out", default="models/evaluation.json")
    args = ap.parse_args()

    config = get_config()
    backend = model_module.build_backend(config)
    if backend.name == "heuristic":
        print("[error] No trained model found (backend is 'heuristic'). Train and "
              "export a model first; the heuristic backend has no test-set accuracy "
              "to report. See docs/TRAINING.md.")
        sys.exit(1)

    labels_path = config.model.labels_path
    with open(labels_path) as fh:
        classes = json.load(fh)["classes"]

    files, y_true, y_food = load_split_files(args.data_dir, args.split, classes)
    if not files:
        print(f"[error] No images in {args.data_dir}/{args.split}")
        sys.exit(1)

    print(f"[eval] backend={backend.name}  split={args.split}  images={len(files)}")

    # Confusion matrix + counts.
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    correct = 0
    food_total = food_correct = 0

    import cv2
    for f, true_cls, true_food in zip(files, y_true, y_food):
        frame = cv2.imread(f, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        pred = backend.predict(frame, config)
        probs = pred["probs"]
        pred_cls = max(probs, key=probs.get)
        if pred_cls not in idx:
            # Model predicted a class not in this test folder set; skip gracefully.
            continue
        cm[idx[true_cls], idx[pred_cls]] += 1
        if pred_cls == true_cls:
            correct += 1
        # Only score food identity when the folders actually carry a food label.
        if true_food is not None:
            food_total += 1
            food_correct += (pred.get("food") == true_food)

    n = cm.sum()
    accuracy = correct / n if n else 0.0

    # Per-class precision / recall / F1.
    report = {}
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        report[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(cm[i, :].sum()),
        }

    print(f"\n[eval] accuracy: {accuracy:.4f}  ({correct}/{n})")
    print("\nPer-class metrics:")
    print(f"{'class':<12}{'precision':>10}{'recall':>10}{'f1':>8}{'support':>9}")
    for cls in classes:
        r = report[cls]
        print(f"{cls:<12}{r['precision']:>10.3f}{r['recall']:>10.3f}"
              f"{r['f1']:>8.3f}{r['support']:>9}")

    print("\nConfusion matrix (rows=true, cols=pred):")
    print("          " + "".join(f"{c[:8]:>10}" for c in classes))
    for i, cls in enumerate(classes):
        print(f"{cls[:8]:<10}" + "".join(f"{cm[i, j]:>10}" for j in range(len(classes))))

    # --- The error that actually matters -----------------------------------
    # Not every mistake costs the same. Passing rotten food as fresh is the
    # expensive one; binning good food is merely wasteful. Report them apart.
    costly = None
    if "spoiled" in idx and "fresh" in idx:
        i_sp, i_fr = idx["spoiled"], idx["fresh"]
        n_sp = int(cm[i_sp].sum())
        if n_sp:
            missed = int(cm[i_sp, i_fr])
            costly = {
                "spoiled_recall": round(float(cm[i_sp, i_sp]) / n_sp, 4),
                "spoiled_called_fresh": missed,
                "spoiled_total": n_sp,
            }
            print(f"\n  spoiled recall       : {costly['spoiled_recall']:.4f}"
                  "   <- REPORT THIS, not accuracy")
            print(f"  spoiled called fresh : {missed:,} of {n_sp:,} "
                  f"({100 * missed / n_sp:.2f}%)  <- the costly error")
            print(f"  fresh called spoiled : {int(cm[i_fr, i_sp]):,} of "
                  f"{int(cm[i_fr].sum()):,}  (merely wasteful)")

    if food_total:
        print(f"\n  food identity accuracy: {food_correct / food_total:.4f} "
              f"({food_correct:,}/{food_total:,})")

    out = {
        "backend": backend.name,
        "split": args.split,
        "num_images": int(n),
        "accuracy": round(accuracy, 4),
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }
    if costly:
        out.update(costly)
    if food_total:
        out["food_accuracy"] = round(food_correct / food_total, 4)
        out["food_images"] = int(food_total)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[save] {args.out}")
    print("\nReport these numbers as-is. Do NOT round up or over-claim accuracy.")


if __name__ == "__main__":
    main()
