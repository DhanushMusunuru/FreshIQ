# FreshIQ — CAD

Parametric Fusion 360 model of the FreshIQ enclosure: rotating 5-compartment
tray on a lazy-susan bearing, NEMA 17 direct drive underneath, camera on an
overhead gantry, UNO Q + A4988 + buck in the base.

## The live bridge

Fusion runs an add-in that watches a folder in this repo. No sockets, no ports,
and nothing executable travels over the channel — a command file carries only a
command **name** plus parameters, and every command is a named function in
`freshiq_cad/`. The add-in reloads that package on each call, so editing a file
here and re-sending is the whole iteration loop.

```
%APPDATA%\Autodesk\Autodesk Fusion 360\API\AddIns\FreshIQBridge\
    FreshIQBridge.py        watcher + main-thread dispatcher
    FreshIQBridge.manifest  runOnStartup: true

cad\
    bridge.py               client — drops a command, waits for the reply
    encode.py               PNG sequences -> mp4 / gif via ffmpeg
    freshiq_cad\
        params.py           every measured dimension, one place, all mm
        build.py            layout maths + all 13 components
        inspect_design.py   reads the real B-Rep back and asserts against build
        style.py            appearances, so the animation reads as a product
        animate.py          drives joint + camera, captures viewport frames
        export.py           STL per printed part, STEP, printability checks
        handshake.py        connection probe
    out\
        bridge.json         present only while the bridge is up
        inbox\ outbox\      the channel
        anim\<scene>\       rendered frames
        export\stl|step\    exported geometry
        video\              encoded clips
```

Fusion's API is main-thread only, so the watcher thread never calls it — it
fires a custom event and the handler registered on the main thread does the
work. That's the documented-safe add-in pattern.

## Usage

```bash
python cad/bridge.py status
```

```bash
python cad/bridge.py send build --ctx '{"show_dialog": false}' --timeout 600
```

```bash
python cad/bridge.py send inspect --ctx '{"verbose": true}'
```

```bash
python cad/bridge.py send animate --ctx '{"orbit_frames": 96}' --timeout 900
```

```bash
python cad/encode.py --fps 30 --gif
```

Commands: `handshake`, `inspect`, `build`, `animate`, `export`.
Always pass `"show_dialog": false` — a Fusion message box blocks the main
thread and the call will time out waiting for someone to click OK.

The add-in's `ALLOWED` list is read once when Fusion starts, so adding a new
command name there does nothing until the add-in restarts. New work folds into
an existing command instead — that's why `style` runs as part of `build`.

Fusion also throttles its event loop when it isn't the foreground window, so a
call can sit for minutes before it is picked up even though it only takes ~40 s
to run. Use generous timeouts and read the reply from `out/outbox/` if the
client gives up first.

## Printed parts

Sized so every part prints whole on a 256 × 256 × 256 bed — no seams, no
splitting. Weights are 100 % infill upper bounds; real prints use far less.

| Part | Size mm | Notes |
|---|---|---|
| `01_Base_Shell` | 250 × 250 × 45 | shelled 3 mm, vents, front UI cutouts |
| `02_Deck` | 250 × 250 × 5 | motor + pedestal + gantry mounting |
| `03_Pedestal` | 130 × 130 × 14 | lifts the bearing to meet the hub flange |
| `04_Tray_Carrier` | 200 × 200 × 14 | bolts to the coupling hub, stays on |
| `08_Tray_Insert` | 193.5 × 193.5 × 38 | **lifts out for washing** |
| `05_Arm_Post` | 62 × 44 × 206 | gantry, cable channel up the back |
| `06_Arm_Head` | 34 × 84 × 26 | camera pocket + lens aperture |
| `07_Camera_Mount_Plate` | 44 × 40 × 3 | slotted — reprint this one when the camera is known |

If `bridge.py status` says `BRIDGE DOWN`, start Fusion. If it's already open:
Utilities → ADD-INS → Scripts and Add-Ins → Add-Ins tab → FreshIQBridge → Run.

## Units

`params.py` is **millimetres everywhere**. Fusion's API is centimetres
internally; the `MM = 0.1` factor converts at the point of use and nowhere else.
The handshake verifies this by building a 20 mm cube and asserting its volume
is 8000 mm³.

## Two API traps worth remembering

`ExtrudeFeatureInput.participantBodies` takes a plain Python **list**, not an
`ObjectCollection` — passing a collection raises a SWIG `TypeError`.
`UnitsManager.defaultLengthUnits` is read-only; set `distanceDisplayUnits`
with a `DistanceUnits` enum instead. Root component name can't be set on an
unsaved document.

## Parameter tags

| Tag | Meaning |
|---|---|
| `[DS]` | From a manufacturer datasheet — trust it |
| `[STD]` | Industry standard dimension — trust it |
| `[VAR]` | Varies by supplier — **measure yours with calipers** |
| `[DES]` | Free design choice — tune to taste |
