# Hardware & performance

## What should run where

The **Arduino UNO Q** has a dual-brain design: a microcontroller (STM32) *and* a
Linux-capable application processor (Qualcomm Dragonwing) that runs Debian and
Python. The vision code targets the **Linux side**, not the MCU.

| Component | Runs on | Why |
|---|---|---|
| Tray motor / sensors | UNO Q microcontroller (Arduino sketch) | Real-time GPIO; **out of scope for this repo** |
| USB camera capture (`camera/`) | UNO Q Linux side | OpenCV `VideoCapture` |
| Quality check + heuristic backend (`ai/`) | UNO Q Linux side | Pure OpenCV/NumPy, lightweight |
| **TFLite** inference (trained model) | UNO Q Linux side (feasible) | MobileNetV2-int8 is small; see caveats |
| Training (`training/`) | A separate PC/laptop/Colab | Needs TensorFlow + more compute; **never on the UNO Q** |

### Recommended split for the prototype
- **On the UNO Q:** capture → quality check → inference (heuristic *or* a
  quantised TFLite model) → structured result → send to backend.
- **On a host PC:** collect/label the dataset, train, export the `.tflite`, then
  copy that single file to the UNO Q.

If the UNO Q's Linux side proves too slow or memory-constrained for the TFLite
model at your required throughput, the fallback is either (a) run the **heuristic
backend** on-device (very light) and/or (b) stream frames to a nearby host/mini-PC
that runs the TFLite model and returns the result. The clean `ai` interface makes
that relocation a one-line change in the caller, not a rewrite.

## Model size / speed expectations

- **MobileNetV2 @224, int8 TFLite:** roughly **2–4 MB** on disk.
- **Latency** is hardware-dependent — **measure it, don't guess**:

  ```bash
  python scripts/benchmark.py sample_data/tomato_fresh.jpg --runs 50
  ```

  On a typical laptop CPU the heuristic backend runs in the low tens of
  milliseconds per image; a quantised MobileNet is usually in a similar-to-modestly
  higher range on the same CPU, and slower on a low-power edge SoC. For a rotating
  tray taking ~4 photos per item with a second or two of settle time between
  angles, even a few hundred ms of inference per view is comfortably fast enough.

## Camera notes

- Any UVC-compliant USB camera works via OpenCV. Set resolution in
  `camera.capture.Camera(width=..., height=...)`.
- The first few frames are discarded (`warmup_frames`) while auto-exposure and
  white-balance settle — important for consistent colour analysis.
- Keep **lighting consistent** between training and inference; colour-based
  freshness cues are lighting-sensitive.
