# Deploying to the Arduino UNO Q

## What runs where

The UNO Q has two processors. **Everything in this repo runs on the Linux side**
(the Qualcomm Dragonwing MPU running Debian), not on the STM32 microcontroller.

| Part | Processor | In this repo? |
|---|---|---|
| Tray motor, sensors, timing | STM32 MCU (Arduino sketch) | No — out of scope |
| USB camera capture | Linux side | Yes, `camera/` |
| Quality check + inference | Linux side | Yes, `ai/` |
| Sending results onward | Linux side | Yes, `backend/` |
| **Training** | A laptop or Kaggle | **Never on the board** |

The board has 2–4 GB of RAM and a quad-core Kryo CPU, so it is comfortably
over-specified for a 0.65 MB model. This is not a tight fit.

---

## 1. Get the files onto the board

Copy the whole repo plus your trained model:

```bash
# from your PC, with the board on the same network
scp -r FreshIQ/ <user>@<board-ip>:~/
```

The board needs `models/` to contain all three files:

```
models/
  freshiq_model.tflite     the quantised model that actually runs
  labels.json              class order + input size - WITHOUT THIS NOTHING WORKS
  freshiq_model.keras      not needed on the board; keep it on your PC
```

---

## 2. Install the dependencies

```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv
pip3 install --break-system-packages ai-edge-litert numpy
```

### About the runtime package — this trips people up

`ai/utils.py::load_interpreter_class` tries three packages in order:

| Package | Notes |
|---|---|
| **`ai-edge-litert`** | Google's **current** name for the standalone runtime. **Install this one.** |
| `tflite-runtime` | The old name. ARM64 wheels stop at Python 3.11, and Debian's own package is pinned to TensorFlow 2.5. Often will not install on a modern board. |
| `tensorflow` | Works, but ~400 MB. Only worth it if the other two fail. |

If you see `No TFLite interpreter available`, the error message lists all three
attempts. Install `ai-edge-litert` and try again.

`python3-opencv` from apt is preferable to `pip install opencv-python` on ARM —
it is prebuilt, so you avoid a very long compile.

---

## 3. Verify, in this order

Each step isolates one thing. Do not skip ahead — if step 2 fails, step 3's
error will be misleading.

**a) The model loads and the right backend is selected**

```bash
cd ~/FreshIQ
python3 scripts/test_image.py sample_data/tomato_spoiled.jpg
```

Look at the `backend` field:

| Value | Meaning |
|---|---|
| `tflite_multihead` | Correct — your trained model is running |
| `heuristic` | The model was not found. Check `models/` has all three files. |

**b) The camera is visible**

```bash
ls /dev/video*                 # should list at least /dev/video0
python3 scripts/test_camera.py --views 1
```

If `/dev/video0` is missing, the camera is not enumerating — try a different USB
port or a powered hub. If it exists but capture fails, check permissions:
`sudo usermod -aG video $USER`, then log out and back in.

**c) The full rotating-tray path**

```bash
python3 scripts/test_camera.py --views 4 --food tomato
```

**d) Measure the real speed — do not trust any figure from a PC**

```bash
python3 scripts/benchmark.py sample_data/tomato_fresh.jpg --runs 50
```

This is the only latency number worth quoting. Kaggle's timing is on an Intel
Xeon and does not transfer.

---

## 4. Run the dashboard server

```bash
cd ~/FreshIQ
python3 backend/server.py --port 8000
```

Then from **any device on the same Wi-Fi** — your laptop, your phone — open:

```
http://<board-ip>:8000
```

Find the board's IP with `hostname -I`.

The dashboard gives you a live view of the tray, an **Inspect** button, the
verdict with confidence bars and evidence, a running history table, statistics,
and a CSV export.

### Useful flags

| Flag | What it does |
|---|---|
| `--port 8000` | Which port to listen on |
| `--host 0.0.0.0` | Default. Reachable from other devices. Use `127.0.0.1` to lock it to the board. |
| `--camera 0` | USB camera index. Try 1, 2… if 0 is wrong. |
| `--rotate-serial /dev/ttyACM0` | Send `R\n` to the STM32 between views to rotate the tray |
| `--rotate-cmd "..."` | Run a shell command between views instead |
| `--device-id rig-02` | Name this unit, so records from several rigs are distinguishable |

### It works before the hardware does

If no camera is found the server prints `DEMO MODE` and inspects the bundled
sample photos instead. The dashboard is fully usable that way — build and test
the whole interface on your laptop, then plug in hardware at the end.

### Start it automatically on boot

```bash
sudo tee /etc/systemd/system/freshiq.service >/dev/null <<'EOF'
[Unit]
Description=FreshIQ inspection server
After=network-online.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/FreshIQ
ExecStart=/usr/bin/python3 backend/server.py --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now freshiq
sudo systemctl status freshiq       # check it came up
journalctl -u freshiq -f            # watch the log
```

`Restart=always` means a crash or a power cut brings it straight back.

---

## 5. How everything connects

```
                        ┌─────────────────────────────────────────┐
                        │            ARDUINO UNO Q                │
                        │                                         │
   USB camera ─────────►│  Linux side (Debian, quad-core Kryo)    │
   (over the tray)      │                                         │
                        │   camera/capture.py                     │
                        │        │ BGR frames                     │
                        │        ▼                                │
                        │   ai/quality_check.py  ──► reject bad   │
                        │        │                                │
                        │        ▼                                │
                        │   ai/multihead.py (freshiq_model.tflite)│
                        │        │ condition + food               │
                        │        ▼                                │
                        │   ai/aggregation.py  (worst view wins)  │
                        │        │                                │
                        │        ▼                                │
                        │   backend/server.py ──► data/*.jsonl    │
                        │        │                                │
                        │   ┌────┴─────┐                          │
                        │   │ port 8000│                          │
                        └───┼──────────┼──────────────────────────┘
                            │          │
                            │          └──── serial /dev/ttyACM0 ──┐
                            │                                      │
                            ▼                                      ▼
                   your phone / laptop                    ┌──────────────────┐
                   http://<board-ip>:8000                 │  STM32 MCU       │
                   the dashboard                          │  tray motor      │
                                                          │  waits for 'R'   │
                                                          └──────────────────┘
```

**The split:** the STM32 does real-time motion, the Linux side does vision and
serves the UI. They talk over the board's internal serial link.

### The MCU side

The server writes `R\n` before each view when you pass `--rotate-serial`. Your
Arduino sketch listens for it:

```cpp
void setup() {
  Serial.begin(115200);
  // motor / stepper setup here
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'R') {
      rotateTrayOneStep();     // your motor code
      Serial.println("OK");    // optional acknowledgement
    }
  }
}
```

Needs `pyserial` on the Linux side: `pip3 install --break-system-packages pyserial`.
If it is missing the server says so and simply skips rotation — it will still
capture, just from one angle.

### Calling it from your own code instead

The dashboard is optional. The AI is three functions:

```python
from ai import classify_image, classify_frame, inspect

classify_image("photo.jpg")                          # one file
classify_frame(bgr_frame)                            # one OpenCV frame
inspect([f1, f2, f3, f4], food_hint="tomato")        # several views -> one verdict
```

And the HTTP API, if you want to drive it from elsewhere:

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | What model and camera are loaded |
| GET | `/api/snapshot` | Current camera frame as JPEG |
| POST | `/api/inspect` | `{"views": 4, "food": "tomato"}` → full record |
| GET | `/api/history?limit=50` | Past inspections, newest first |
| GET | `/api/stats` | Totals and spoiled rate |
| GET | `/api/export.csv` | Everything, as CSV |

```bash
curl -X POST http://<board-ip>:8000/api/inspect \
     -H "Content-Type: application/json" \
     -d '{"views":4,"food":"tomato"}'
```

### Sending results onward (Firebase, etc.)

`backend/service.py::build_inspection_record` returns a plain dict. Hand it to
whatever you like:

```python
from backend.service import build_inspection_record
record = build_inspection_record(frames, food_hint="tomato")
# db.collection("inspections").add(record)
```

History is also on disk at `data/inspections.jsonl` — one JSON object per line,
append-only. Readable with `tail -f`, and a power cut costs you at most the last
line rather than the whole file.

---

## 6. If it is too slow or too tight

In order of how much they buy you:

1. **Drop the input size.** Retrain with `IMG_SIZE = 96` in the notebook. Roughly
   half the arithmetic. 96 is one of the sizes with pretrained weights, so it is
   a one-line change.
2. **Use `opencv-python-headless`** instead of full OpenCV if RAM is tight — it
   drops the GUI dependencies. (Not an issue on the UNO Q; matters on a Pi Zero.)
3. **Fall back to the heuristic backend** by setting `backend: "heuristic"` in
   `ai/config.json`. No model file needed, very light, and the pipeline keeps
   working — just less accurate.

Note that shrinking the model further has limits: at α=0.35 the final 1×1
convolution to 1280 channels is about a third of all parameters. Cutting the
input size helps more than cutting α again.

---

## Do not

- **Do not install full TensorFlow** unless the other runtimes genuinely fail.
- **Do not train on the board.** Train on Kaggle or a PC, copy the `.tflite` over.
- **Do not ship without `labels.json`.** The model outputs bare arrays of numbers;
  that file is the only thing that says which number means "spoiled".
- **Do not quote the Kaggle latency figure.** Measure on the board.
