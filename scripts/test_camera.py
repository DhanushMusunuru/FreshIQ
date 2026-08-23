"""
Test the USB camera path (PHASE 2).

Single frame:
    python scripts/test_camera.py --index 0 --food tomato

Capture N rotating-tray views and run a full inspection (PHASE 3 & 4):
    python scripts/test_camera.py --index 0 --views 4 --delay 0.8 --food tomato

Save the captured frames too:
    python scripts/test_camera.py --views 4 --save-dir captures/

Requires a connected USB camera. On a headless machine this will fail to open
the device — that is expected; use scripts/test_image.py instead.
"""

import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai import classify_frame, inspect          # noqa: E402
from camera.capture import Camera, CameraError    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0, help="camera device index")
    ap.add_argument("--views", type=int, default=1, help="number of tray views to capture")
    ap.add_argument("--delay", type=float, default=0.8, help="seconds between views")
    ap.add_argument("--food", default=None)
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    try:
        with Camera(index=args.index) as cam:
            if args.views <= 1:
                frame = cam.capture()
                if args.save_dir:
                    cv2.imwrite(os.path.join(args.save_dir, "view_0.jpg"), frame)
                result = classify_frame(frame, food_hint=args.food)
                result.pop("confidence_raw", None)
                print(json.dumps(result, indent=2))
            else:
                print(f"Capturing {args.views} views "
                      f"(rotate the tray between prompts if driving it manually)...")
                frames = cam.capture_sequence(args.views, delay=args.delay)
                if args.save_dir:
                    for i, fr in enumerate(frames):
                        cv2.imwrite(os.path.join(args.save_dir, f"view_{i}.jpg"), fr)
                final = inspect(frames, food_hint=args.food)
                final.pop("views", None)
                print(json.dumps(final, indent=2))
    except CameraError as exc:
        print(f"[camera error] {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
