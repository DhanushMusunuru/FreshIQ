# FreshIQ — Arduino UNO Q

Food-freshness inspection on a single UNO Q. A USB camera and a TFLite model run
on the Linux side (MPU); a NEMA 17 turns a 4-compartment tray through an A4988
on the microcontroller side (MCU). The two halves talk over the Arduino router.

```
CAMERA -> Linux MPU -> AI -> scan controller
                                  |
                     arduino-router (MessagePack-RPC)
                                  |
                          onboard MCU (sketch)
                                  |
                        A4988 -> NEMA 17 -> TRAY
                                  |
                              back to CAMERA
```

## Layout

This is an **Arduino App**: App Lab builds and deploys both halves together.

```
FreshIQ/
├── app.yaml              app manifest (name, icon, version)
├── START.sh              optional: run the Linux half on its own
├── docs/                 wiring schematic, architecture, deployment notes
├── sketch/               -> the MICROCONTROLLER half
│   ├── sketch.ino          tray driver: A4988 stepping, RPC methods
│   └── sketch.yaml         build profile (board + library versions)
└── python/               -> the LINUX half
    ├── main.py             App Lab entry point
    ├── requirements.txt    installed automatically on first run
    ├── ai/                 classifier, heuristics, aggregation
    ├── backend/            HTTP server, dashboard, scan state machine
    ├── camera/             OpenCV capture
    ├── config/             freshiq.json — compartments, camera, motor
    ├── hardware/           motor.py (tray) + bridge_rpc.py (MPU->MCU)
    ├── models/             freshiq_model.tflite + labels
    ├── sample_data/        demo images used when no camera is present
    └── tools/check_tray.py tray diagnostic
```

## Install

**LAPTOP TERMINAL** — copy the app onto the board:

```bash
unzip FreshIQ-UNO-Q-App.zip
ssh <user>@<board-ip> "mkdir -p ~/ArduinoApps/FreshIQ"
scp -r FreshIQ/* <user>@<board-ip>:~/ArduinoApps/FreshIQ/
```

App Lab picks the folder up automatically — it lists any app under
`~/ArduinoApps/`. If you prefer the UI, open App Lab and FreshIQ will be there.

## Run

**UNO Q SSH TERMINAL:**

```bash
arduino-app-cli app start ~/ArduinoApps/FreshIQ
arduino-app-cli app logs  ~/ArduinoApps/FreshIQ
```

`app start` compiles `sketch/` and flashes it to the MCU, installs
`python/requirements.txt`, and runs `python/main.py`. The **first run is slow**
— it downloads numpy, OpenCV and the TFLite runtime. Watch `app logs` until:

```
[tray]   BridgeTrayMotor on /var/run/arduino-router.sock, 4 compartments, 50 steps each
```

Then open **http://&lt;board-ip&gt;:8000** and press **Inspect**.

To start it automatically at power-on:

```bash
arduino-app-cli properties set default user:FreshIQ
```

Stop it with `arduino-app-cli app stop ~/ArduinoApps/FreshIQ`.

## Before the first move

Check these against your build — they are in `sketch/sketch.ino` and match
`docs/schematics/freshiq-wiring.svg`:

| A4988 | UNO Q | Note |
|---|---|---|
| STEP | **D2** | one pulse, one step |
| DIR | **D3** | flip `DIRECTION` in the sketch if the tray turns the wrong way |
| ENABLE | **D5** | active LOW |
| VDD | **+3V3** | **not +5 V** — on 5 V the logic-HIGH threshold is ~3.5 V and the UNO Q only drives 3.3 V, so STEP is never reliably seen |
| RESET + SLEEP | jumpered together | a floating RESET holds the driver in reset forever |
| MS1/MS2/MS3 | open | full step — must match `tray.microsteps` in `python/config/freshiq.json` |

Set the A4988 current limit (Vref) before powering the motor, and fit the 100 µF
capacitor across VMOT/GND.

The tray has **no home switch**, so software cannot know where it is. Turn it by
hand until compartment 1 sits under the camera before the first scan.

## Tray speed

How long one compartment of travel takes, in `python/config/freshiq.json`:

```json
"tray": {
  "seconds_per_compartment": 3.5
}
```

The MCU is told a per-step delay derived from this and `steps_per_compartment`,
so **changing the speed is a config edit and an app restart — no re-flash**.
The measured move lands within about 0.15 s of the configured value (the extra
is the brief hold that lets the platter settle before the driver is released).

If a move takes the right amount of time but the tray barely turns, the problem
is distance, not speed: your A4988 **MS1/MS2/MS3 jumpers are microstepping**
while `tray.microsteps` says `1`. At 1/16 microstepping, 50 steps moves 5.6°
instead of 90°. Either open the jumpers or set `microsteps` to match — they must
agree. `python/tools/check_tray.py --rotate` prints the angle each move should
produce so you can measure it.

## Geometry

`python/config/freshiq.json`: 4 compartments, 200 steps/rev, 1 microstep →
**50 steps per compartment**. Change `compartments` and `microsteps` together;
the loader refuses a combination that would make the tray drift each revolution.

## Diagnosing the tray

**UNO Q SSH TERMINAL:**

```bash
cd ~/ArduinoApps/FreshIQ/python
python3 tools/check_tray.py            # is the router up? is the sketch answering?
python3 tools/check_tray.py --rotate   # turn a full physical lap, one compartment at a time
```

| Message | Cause | Fix |
|---|---|---|
| `no router socket at /var/run/arduino-router.sock` | router service down | `systemctl status arduino-router` |
| `the tray sketch did not answer` | MCU half not deployed | `arduino-app-cli app start ~/ArduinoApps/FreshIQ` |
| `no serial port found (looked for /dev/ttyACM*)` | `motor.driver` was set to `serial` | set it back to `"bridge"` |
| `msgpack` not installed | running outside the app's environment | `pip install msgpack` |
| `can't open camera by index` | camera is not on index 0 | `ls /dev/video*`, then set `camera.index` to `"/dev/video0"` |

## Why RPC and not a serial port

The UNO Q is one board with two processors joined by an internal UART. On Linux
that UART is `/dev/ttyHS1`, and `arduino-router` already owns it — Arduino's
documentation says application code must not open it. The MCU is not a USB
device, so `/dev/ttyACM0` does not exist and never will. `python/hardware/`
therefore calls the sketch by name through the router instead:

```
rpc.call("tray_rotate", 50)  ->  tray_rotate(50)   turns one compartment
                             <-  returns 50        move is FINISHED
```

The call returns only once the motor has stopped, so the camera can never
photograph a moving tray.
