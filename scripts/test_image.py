"""
Test a single image (PHASE 1).

    python scripts/test_image.py sample_data/tomato_spoiled.jpg
    python scripts/test_image.py photo.jpg --food tomato --save out.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai import classify_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--food", default=None, help="optional known food hint")
    ap.add_argument("--save", default=None, help="write result JSON to this path")
    args = ap.parse_args()

    result = classify_image(args.image, food_hint=args.food)
    result.pop("confidence_raw", None)   # internal field, not part of the API
    print(json.dumps(result, indent=2))

    if args.save:
        with open(args.save, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nsaved -> {args.save}")


if __name__ == "__main__":
    main()
