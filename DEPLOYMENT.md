# Deploying FreshIQ to the Arduino UNO Q

Copy-paste instructions to go from the Windows laptop to a running rig.

Anything this document cannot know from the files is marked
`<USER MUST PROVIDE>`. There are only three, and they are all things only you
can see: the board's IP address, its Linux username, and your A4988 wiring.

---

## What gets copied

**~3.6 MB.** Most of the repo does not belong on the board:

| Copy | Why |
|---|---|
| `ai/ backend/ camera/ config/ hardware/` | the application |
| `models/freshiq_model.tflite`, `models/labels.json` | the trained model |
| `sample_data/` | lets the dashboard work before the camera does |
| `requirements.txt install.sh start_freshiq.sh deploy/` | setup |
| **Skip** `cad/` | 66 MB of Fusion files; the board never needs them |
| **Skip** `models/*.keras` | 9.7 MB training artifact; LiteRT runs the .tflite |
| **Skip** `training/ notebooks/ dataset/` | laptop-only |

> `models/` and `data/` are in `.gitignore`. **A `git clone` will not bring the
> model.** It must be copied explicitly, which is what the command below does.

---

## 1. Copy the files

From PowerShell on the laptop, in the project folder:

```powershell
scp -r ai backend camera config hardware firmware deploy sample_data docs requirements.txt install.sh start_freshiq.sh <USER MUST PROVIDE: user>@<USER MUST PROVIDE: board-ip>:~/FreshIQ/
```

```powershell
scp models/freshiq_model.tflite models/labels.json <USER MUST PROVIDE: user>@<USER MUST PROVIDE: board-ip>:~/FreshIQ/models/
```

If `scp` complains that `~/FreshIQ/models` does not exist, create it first:

```powershell
ssh <USER MUST PROVIDE: user>@<USER MUST PROVIDE: board-ip> "mkdir -p ~/FreshIQ/models ~/FreshIQ/data"
```

---

## 2. Install

```bash
ssh <USER MUST PROVIDE: user>@<USER MUST PROVIDE: board-ip>
```

```bash
cd ~/FreshIQ && chmod +x install.sh start_freshiq.sh && ./install.sh
```

`install.sh` checks Python, installs the system libraries OpenCV needs,
installs the Python packages, and then **verifies the model against
labels.json by reading the real tensors** — the one mismatch that would
otherwise fail silently and give you confident nonsense.

It does not install TensorFlow. The board only runs the already-trained
`.tflite` through LiteRT.

---

## 3. Tell it about your hardware

```bash
ls /dev/video* /dev/ttyACM* /dev/ttyUSB*
```

```bash
nano ~/FreshIQ/config/freshiq.json
```

Set `camera.index` (usually `0` for `/dev/video0`). Leave `motor.driver` as
`"bridge"` — that is the UNO Q's onboard MCU. `motor.port` is ignored unless
you switch the driver to `"serial"` for a separate external Arduino.
Everything else already matches the printed tray.

---

## 4. Flash the tray controller to the UNO Q's ONBOARD MCU

The UNO Q is one board with two processors. FreshIQ runs on the Linux half; the
stepper is driven by the microcontroller half. They are joined by an internal
UART which on Linux is `/dev/ttyHS1` — **already owned by the `arduino-router`
service**, which is why application code must never open it and why there is no
`/dev/ttyACM0` to look for. Linux calls the sketch by RPC through that router.

**Confirm these before flashing** — they match
`docs/schematics/freshiq-wiring.svg`:

| Constant | Value | Check |
|---|---|---|
| `PIN_STEP` / `PIN_DIR` / `PIN_ENABLE` | **2 / 3 / 5** | D2 = STEP, D3 = DIR, D5 = ENABLE (active LOW) |
| `MICROSTEPS` | 1 | your MS1/MS2/MS3 jumpers, **and** `tray.microsteps` in the config |
| `COMPARTMENTS` | 4 | the printed tray has 4 → 50 steps per compartment |
| A4988 VDD | **+3V3** | not +5 V — see the note in the schematic |
| A4988 Vref | — | set the current limit for your motor before powering it |

Flash it from the board itself:

```bash
arduino-cli core install arduino:zephyr          # once
arduino-cli lib install Arduino_RouterBridge     # once
arduino-cli compile -b arduino:zephyr:unoq firmware/freshiq_tray
arduino-cli upload  -b arduino:zephyr:unoq firmware/freshiq_tray
```

Then check the link end to end, without starting the server:

```bash
python3 tools/check_tray.py            # is the router up, is the sketch answering?
python3 tools/check_tray.py --rotate   # turn a full lap, one compartment at a time
```

Expect `READY`, then four moves of 50 steps. If the tray turns the wrong way,
flip `DIRECTION` in the sketch and re-upload.

---

## 5. Align the tray, once

The rig has **no home switch**, so software cannot know where the tray is.
Turn it by hand until compartment 1 sits under the camera. The software assumes
compartment 1 at startup. If it ever drifts, correct it from the dashboard or:

```bash
curl -X POST http://localhost:8000/api/motor/position -H "Content-Type: application/json" -d '{"position":1}'
```

---

## 6. Start it

```bash
cd ~/FreshIQ && ./start_freshiq.sh
```

Open `http://<USER MUST PROVIDE: board-ip>:8000` from a laptop or phone on the
same Wi-Fi. Press **Start scan**.

---

## 7. Start automatically on boot

```bash
sudo cp ~/FreshIQ/deploy/freshiq.service /etc/systemd/system/freshiq.service
```

```bash
sudo sed -i "s/YOUR_USER/$USER/g; s#/home/YOUR_USER/FreshIQ#$HOME/FreshIQ#g" /etc/systemd/system/freshiq.service
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now freshiq && sudo systemctl status freshiq
```

Then:

| Task | Command |
|---|---|
| Watch the log | `journalctl -u freshiq -f` |
| Restart after a config change | `sudo systemctl restart freshiq` |
| Stop | `sudo systemctl stop freshiq` |
| Disable auto-start | `sudo systemctl disable freshiq` |

---

## Verifying, in order

```bash
cd ~/FreshIQ && curl -s localhost:8000/api/system/status | python3 -m json.tool
```

Check in this order — each depends on the one before:

1. `ai.backend` is `tflite_multihead`. If it says `heuristic`, the model is missing.
2. `camera.mode` is `live`. If `demo`, no camera was found.
3. `scan_available` is `true`. If false, `scan_error` says why.
4. `tray.position` is 1.

```bash
curl -X POST localhost:8000/api/scan/compartment -H "Content-Type: application/json" -d '{"compartment":2}'
```

The tray should move one compartment, pause, and return a result.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `libGL.so.1: cannot open shared object file` | OpenCV's system deps | `sudo apt install -y libgl1 libglib2.0-0` |
| Backend says `heuristic` | model not copied (`models/` is gitignored) | re-run the `scp` in step 1 |
| `camera.mode: demo` | no `/dev/video*`, or another process holds it | replug; `sudo fuser -v /dev/video0` |
| Everything `blurred` | focus threshold tuned for a rig camera | `./start_freshiq.sh --min-focus 20` |
| `scan_unavailable` | serial port wrong, or pyserial missing | check `scan_error` in `/api/system/status` |
| Serial `Permission denied` | user not in `dialout` | `sudo usermod -aG dialout $USER`, then log out and in |
| Tray moves to the wrong compartment | lost steps, or misalignment | lower speed in the sketch, raise A4988 current, re-align |
| `[tray] UNAVAILABLE ... no serial port found` | `motor.driver` is `"serial"`, which looks for an EXTERNAL Arduino | set it to `"bridge"` in `config/freshiq.json` |
| `no router socket at /var/run/arduino-router.sock` | the router service is down | `systemctl status arduino-router` |
| `the tray sketch did not answer` | firmware not flashed to the onboard MCU | re-run the `arduino-cli` steps in section 4 |
| `msgpack` not installed | installed outside your `.venv` | `pip install msgpack` inside the venv |
| Dashboard unreachable from a phone | bound to loopback | `server.host` must be `0.0.0.0`, not `127.0.0.1` |

---

## Network exposure

The server has **no authentication**. `0.0.0.0` means anyone on your Wi-Fi can
open the dashboard and drive the tray.

That is fine for a LAN prototype. **Do not port-forward it to the internet.**
For remote access use a private tunnel — Tailscale is the least painful:

```bash
curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up
```

The board then gets a private address reachable only from your own devices,
with nothing exposed publicly.

---

## Replacing the model later

```bash
scp models/freshiq_model.tflite models/labels.json user@board:~/FreshIQ/models/ && ssh user@board "sudo systemctl restart freshiq"
```

`labels.json` must always travel with the `.tflite`. It defines the class order
and input size; the wrong one silently mislabels everything. `install.sh`
re-checks that they agree.
