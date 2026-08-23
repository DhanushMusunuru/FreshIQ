"""
Measure inference time (CORE AI TASK #13).

    python scripts/benchmark.py sample_data/tomato_fresh.jpg --runs 50

Reports mean / median / p95 latency and throughput for the active backend, so you
can judge whether it is fast enough for the target hardware (see docs/HARDWARE.md).
"""

import argparse
import os
import statistics
import sys
import time

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai import classify_frame, get_config          # noqa: E402
from ai import model as model_module                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"Could not read {args.image}")
        sys.exit(1)

    config = get_config()
    backend = model_module.build_backend(config)
    print(f"Backend: {backend.name}")

    for _ in range(args.warmup):
        classify_frame(frame)

    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        classify_frame(frame)
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    p95 = times[int(len(times) * 0.95) - 1]
    print(f"runs:        {args.runs}")
    print(f"mean:        {statistics.mean(times):.1f} ms")
    print(f"median:      {statistics.median(times):.1f} ms")
    print(f"p95:         {p95:.1f} ms")
    print(f"min / max:   {times[0]:.1f} / {times[-1]:.1f} ms")
    print(f"throughput:  {1000.0 / statistics.mean(times):.1f} images/sec")


if __name__ == "__main__":
    main()
