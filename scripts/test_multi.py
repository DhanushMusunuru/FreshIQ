"""
Test multiple images as different views of ONE food item (PHASE 3 & 4).

    python scripts/test_multi.py sample_data/tomato_front.jpg sample_data/tomato_back.jpg --food tomato
    python scripts/test_multi.py views/*.jpg --food tomato --save result.json --verbose

Demonstrates that the aggregation reflects the WORST significant view rather than
a blind average (e.g. fresh front + mouldy back -> spoiled).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai import inspect  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="two or more views of the same item")
    ap.add_argument("--food", default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--verbose", action="store_true", help="show per-view breakdown")
    args = ap.parse_args()

    final = inspect(args.images, food_hint=args.food)
    views = final.pop("views", [])

    print("=== FINAL INSPECTION RESULT ===")
    print(json.dumps(final, indent=2))

    if args.verbose:
        print("\n=== PER-VIEW ===")
        for i, v in enumerate(views):
            v.pop("confidence_raw", None)
            label = args.images[i] if i < len(args.images) else f"view {i}"
            print(f"\n[{i}] {os.path.basename(str(label))}")
            print(json.dumps(v, indent=2))

    if args.save:
        with open(args.save, "w") as fh:
            json.dump({"final": final, "views": views}, fh, indent=2)
        print(f"\nsaved -> {args.save}")


if __name__ == "__main__":
    main()
