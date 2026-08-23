"""
Compare predictions against ground-truth labels (CORE AI TASK #13).

THIS IS THE SCRIPT THAT TELLS YOU IF THE MODEL ACTUALLY WORKS.

A test-set score from Kaggle measures the model on web photos of supermarket
produce. Your rig photographs food on one tray under one lamp. The only number
that means anything for YOUR machine is the one you get from YOUR photos.

Shoot some produce, sort it into folders by what it really is, and run this:

    myphotos/
        fresh/*.jpg
        spoiled/*.jpg

    python scripts/compare_ground_truth.py --dir myphotos --save report.json

Optionally nest by food to score identity too:

    myphotos/
        fresh/tomato/*.jpg
        spoiled/banana/*.jpg

Works with ANY backend (including the zero-training heuristic), so you can
measure the baseline on your real photos before trusting anything trained.
Prints accuracy, a confusion matrix, and the files it got wrong - and does NOT
pretend the numbers are better than they are.
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai import classify_frame, get_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder with one sub-folder per true class")
    ap.add_argument("--food", default=None, help="optional food hint applied to all")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    config = get_config()
    classes = [d for d in sorted(os.listdir(args.dir))
               if os.path.isdir(os.path.join(args.dir, d))]
    if not classes:
        print(f"No class sub-folders in {args.dir}")
        sys.exit(1)

    cm = defaultdict(lambda: defaultdict(int))
    total = 0
    correct = 0
    mistakes = []
    food_total = food_correct = 0
    rejected = 0

    def photos_in(d):
        """Yield (path, true_food_or_None) - flat, or one level nested by food."""
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for f in glob.glob(os.path.join(d, ext)):
                yield f, None
        for sub in sorted(os.listdir(d)):
            sd = os.path.join(d, sub)
            if os.path.isdir(sd):
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                    for f in glob.glob(os.path.join(sd, ext)):
                        yield f, sub.strip().lower().replace(" ", "_")

    for true_cls in classes:
        for f, true_food in photos_in(os.path.join(args.dir, true_cls)):
            frame = cv2.imread(f, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            res = classify_frame(frame, food_hint=args.food)
            if res.get("status") != "ok":
                # Rejected by the quality gate - NOT a wrong answer. Counted
                # separately so a batch of blurred photos does not read as a
                # model failure.
                pred = f"invalid:{res.get('reason')}"
                rejected += 1
            else:
                pred = res["condition"]
                if true_food is not None and args.food is None:
                    food_total += 1
                    food_correct += (res.get("food") == true_food)
            cm[true_cls][pred] += 1
            total += 1
            if pred == true_cls:
                correct += 1
            else:
                mistakes.append({"file": f, "true": true_cls, "pred": pred,
                                 "confidence": res.get("confidence")})

    acc = correct / total if total else 0.0
    print(f"backend/config accuracy: {acc:.3f}  ({correct}/{total})\n")

    pred_labels = sorted({p for row in cm.values() for p in row})
    print("Confusion (rows=true, cols=pred):")
    header = "          " + "".join(f"{p[:9]:>11}" for p in pred_labels)
    print(header)
    for tc in classes:
        row = "".join(f"{cm[tc].get(p, 0):>11}" for p in pred_labels)
        print(f"{tc[:9]:<10}{row}")

    # --- The error that matters ---------------------------------------------
    # Passing rotten food as fresh is the expensive mistake. Binning a good
    # tomato is merely wasteful. Report them apart, never as one accuracy.
    if "spoiled" in classes and "fresh" in classes:
        n_sp = sum(cm["spoiled"].values())
        if n_sp:
            missed = cm["spoiled"].get("fresh", 0)
            print(f"\n  spoiled recall       : {cm['spoiled'].get('spoiled', 0) / n_sp:.3f}"
                  "   <- REPORT THIS, not accuracy")
            print(f"  spoiled called fresh : {missed} of {n_sp}   <- rotten food passed as good")
            print(f"  fresh called spoiled : {cm['fresh'].get('spoiled', 0)} of "
                  f"{sum(cm['fresh'].values())}   (merely wasteful)")

    if food_total:
        print(f"\n  food identity        : {food_correct / food_total:.3f} "
              f"({food_correct}/{food_total})")

    if rejected:
        print(f"\n  {rejected} photo(s) rejected by the quality gate (blurred, dark, "
              "no food found). Those are not wrong answers - reshoot them.")

    if mistakes:
        print(f"\n{len(mistakes)} misclassified (first 10) - LOOK AT THESE FILES:")
        for m in mistakes[:10]:
            print(f"  {os.path.basename(m['file'])}: true={m['true']} pred={m['pred']}")

    if args.save:
        out = {
            "accuracy": round(acc, 4),
            "total": total,
            "correct": correct,
            "rejected": rejected,
            "confusion_matrix": {t: dict(r) for t, r in cm.items()},
            "mistakes": mistakes,
        }
        if food_total:
            out["food_accuracy"] = round(food_correct / food_total, 4)
        with open(args.save, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nsaved -> {args.save}")


if __name__ == "__main__":
    main()
