# FreshIQ — Setup, start to finish

Follow this in order. Each step ends with something you can check, so if it fails
you know exactly which step broke.

- **Part 1** — get it running on your PC (15 min)
- **Part 2** — put the trained model in (5 min)
- **Part 3** — real-world testing: hold fruit up, see the verdict (10 min)
- **Part 4** — move it onto the Arduino UNO Q
- **Part 5** — when something goes wrong

---

# Part 1 — Run it on your PC first

Do this before touching hardware. It removes half the variables.

### 1.1 Check Python

```bash
python --version
```

Needs 3.9 or newer. If `python` is not found, try `python3`.

### 1.2 Install the dependencies

```bash
cd "D:\Side Quests\Projects\FreshIQ"
pip install -r requirements.txt
```

That is OpenCV and NumPy. Nothing else is needed to run — TensorFlow is only for
training, and you already did that on Kaggle.

### 1.3 Check it works at all

```bash
python -m pytest -q
```

**Expect:** `64 passed`

If this fails, nothing else will work. Fix it here.

### 1.4 Start the server

```bash
python backend/server.py --port 8000
```

**Expect:**

```
[camera] opened index 0                     (or: not available -> DEMO MODE)
[ai]     backend: heuristic   <- no trained model found, using rules
[data]   history: .../data/inspections.jsonl

  FreshIQ dashboard -> http://localhost:8000
```

### 1.5 Open it

Go to **http://localhost:8000** in your browser.

You should see the dashboard with two orange banners — one saying no camera (if
you have none) and one saying no trained model. **Both are expected right now.**

Press **Inspect & save**. You should get a verdict, confidence bars, and a row in
the history table.

> ✅ **Checkpoint:** the dashboard loads and produces a verdict.
> The verdict will be poor — it is running hand-written rules, not your model.
> That is the next step.

---

# Part 2 — Install your trained model

### 2.1 Get the file from Kaggle

Open your notebook → right sidebar → **Output** → download **`freshiq_models.zip`**.

### 2.2 Unzip it into `models/`

Three files must end up here:

```
D:\Side Quests\Projects\FreshIQ\models\
    freshiq_model.tflite      <- the model that runs (0.65 MB)
    labels.json               <- class order. WITHOUT THIS NOTHING WORKS.
    freshiq_model.keras       <- the full-precision master, keep it safe
```

**`labels.json` is not optional.** The model outputs a bare list of numbers like
`[0.13, 0.87]`. That file is the only thing recording that position 0 means
`fresh` and position 1 means `spoiled`. Without it the numbers are meaningless.

### 2.3 Confirm it loaded

```bash
python scripts/test_image.py sample_data/tomato_spoiled.jpg
```

Look at the `backend` line in the output:

| It says | Meaning |
|---|---|
| `tflite_multihead` | ✅ Your trained model is running |
| `heuristic` | ❌ Files are in the wrong place. Check the three names above. |

### 2.4 Restart the server

Ctrl+C, then start it again. The orange "no trained model" banner should be gone.

> ✅ **Checkpoint:** the dashboard header says `model tflite_multihead`.

---

# Part 3 — Testing with real fruit

This is the part that tells you whether it actually works.

### 3.1 Point a camera at something

Any USB webcam. Laptop camera is fine to start with.

If the server said `DEMO MODE`, it did not find a camera. Try:

```bash
python backend/server.py --port 8000 --camera 1     # then 2, 3...
```

### 3.2 Press "Go live"

This is the mode you want for real-world testing:

- Classifies **one frame every 2 seconds**
- The verdict updates as you move things in front of the camera
- **Nothing is saved** — so you can experiment freely without filling history

Hold a piece of fruit up to the camera. Watch the verdict and the confidence
bars change. Move it, rotate it, cover part of it.

### 3.3 What to actually test

Do these deliberately. Each one tells you something different:

| Test | What you learn |
|---|---|
| Fresh fruit, good light | Baseline. Should read `fresh`, high confidence. |
| Visibly rotten fruit | The main job. Should read `spoiled`. |
| A ripe banana with brown flecks | The hard case. Ripe ≠ rotten. |
| Same fruit, room light vs lamp | How lighting-sensitive it is. This is usually the biggest real-world gap. |
| Move it far from the camera | Should say `too far` / poor quality, not guess |
| Cover the lens | Should reject the frame, not invent an answer |
| Two fruits at once | Should flag `multiple objects` |
| A cup, or your hand | It has no "not food" class. Watch what nonsense it produces — worth knowing. |

### 3.4 Keep the interesting ones

When you see a result worth recording, press **Keep this one**. That writes it
to history without interrupting live mode.

Everything you keep lands in `data/inspections.jsonl` and shows up in the history
table. **Export CSV** gives you the whole lot for a report.

### 3.5 Read the failures honestly

Two mistakes, and they do not cost the same:

- **Rotten called fresh** — you passed bad food. This is the expensive one.
- **Fresh called spoiled** — you binned a good tomato. Wasteful, harmless.

If you see the first one happening often, the model is not ready regardless of
what the Kaggle number said.

**Expect worse than 93.8%.** That number came from web photos of supermarket
produce. Your kitchen has different light, different backgrounds, different
cameras. A drop to 80-something is normal and is not a bug — it is the gap
between the training data and the real world.

> ✅ **Checkpoint:** you know how it behaves on real objects, including where it
> fails.

---

# Part 4 — Onto the Arduino UNO Q

Full detail in [`docs/DEPLOY.md`](docs/DEPLOY.md). The short version:

### 4.1 Copy it across

```bash
scp -r "D:\Side Quests\Projects\FreshIQ" user@<board-ip>:~/FreshIQ
```

### 4.2 Install on the board

```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv
pip3 install --break-system-packages ai-edge-litert numpy pyserial
```

**Install `ai-edge-litert`, not `tflite-runtime`.** The old package's ARM64
wheels stop at Python 3.11 and Debian's own build is pinned to TensorFlow 2.5.

### 4.3 Verify in this order — do not skip ahead

```bash
cd ~/FreshIQ
python3 -m pytest -q                                    # 1. code is intact
python3 scripts/test_image.py sample_data/tomato_spoiled.jpg   # 2. model loads
ls /dev/video*                                          # 3. camera exists
python3 scripts/test_camera.py --views 1                # 4. camera captures
python3 scripts/benchmark.py sample_data/tomato_fresh.jpg --runs 50   # 5. real speed
```

Step 5 is the only latency number worth quoting. Kaggle's 1.5 ms was an Intel
Xeon and does not transfer.

### 4.4 Start the server

```bash
python3 backend/server.py --port 8000
hostname -I                    # find the board's IP
```

Open `http://<board-ip>:8000` from your phone or laptop on the same Wi-Fi.

### 4.5 Make it start on boot

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
sudo systemctl status freshiq
```

### 4.6 Wire up the tray

```bash
python3 backend/server.py --port 8000 --rotate-serial /dev/ttyACM0
```

The server sends `R\n` before each view. The STM32 sketch listens for it — see
[`docs/DEPLOY.md`](docs/DEPLOY.md) for the sketch.

> ✅ **Checkpoint:** you can open the dashboard from your phone and inspect fruit
> on the real rig.

---

# Part 5 — When something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `backend: heuristic` after installing the model | Files in the wrong folder, or `labels.json` missing | All three files directly in `models/` |
| `DEMO MODE` with a camera plugged in | Wrong index, or another app has it | `--camera 1`, close Zoom/Teams, check `ls /dev/video*` |
| `No TFLite interpreter available` | Runtime not installed | `pip install ai-edge-litert` |
| `No module named cgi` | Python 3.13+ running the old `service.py` | Use `backend/server.py` instead — it does not use `cgi` |
| Dashboard says `offline` | Server not running, or wrong port | Check the terminal; look for a Python traceback |
| Cannot reach it from your phone | Bound to localhost, or firewall | Use `--host 0.0.0.0` (the default) and allow the port |
| Every frame says **`blurred`** | Laptop/phone cameras are softer than a tray-mounted USB camera. The default threshold is 55. | Read the number the dashboard now shows (`focus 23, needs ≥ 55`), then restart with `--min-focus 20` |
| Every frame comes back `poor` / `invalid` | Too dark, or no food found | More light; move the item closer; the gate is doing its job |
| Phone cannot reach the dashboard | Server bound to localhost, or Windows Firewall | Start with `--host 0.0.0.0` and allow Python through the firewall when prompted |
| Everything reads `spoiled` | Running the heuristic on dim images | Install the model; improve lighting |
| Live mode feels sluggish | Board busy, or too many views | Live mode already uses 1 view; try a lower camera resolution |
| History disappeared | `data/` was deleted | It is gitignored by design; back up `data/inspections.jsonl` |

### Getting more detail

```bash
python backend/server.py --port 8000        # watch the terminal
curl http://localhost:8000/api/health       # what is actually loaded
journalctl -u freshiq -f                    # if running via systemd
tail -f data/inspections.jsonl              # watch results land
```

---

## Where things live

```
FreshIQ/
  ai/            the vision system
  backend/       server.py + dashboard.html   <- the web interface
  camera/        USB camera capture
  models/        YOUR TRAINED MODEL GOES HERE
  data/          inspection history (created on first run, gitignored)
  scripts/       CLI tools for testing
  notebooks/     the Kaggle training notebook
  docs/          DEPLOY.md, FROM_SCRATCH.md, LIMITATIONS.md, ...
  tests/         64 tests
```

## Before you show it to anyone

Read [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md). The short version: this
estimates **appearance**, not safety. `spoiled: 82` means the model is 82%
confident the item *looks* like its spoiled class — not that it is 82% rotten.
It cannot see inside, and it cannot detect bacteria. Say so when you present it.
