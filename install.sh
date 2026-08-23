#!/usr/bin/env bash
#
# FreshIQ installer for the Arduino UNO Q (Debian, Linux side).
#
#   ./install.sh
#
# Installs only what the DEVICE needs. It deliberately does NOT install
# TensorFlow: training happens on your laptop, and the board only ever runs the
# already-trained .tflite through LiteRT. TensorFlow on an ARM board is a
# multi-hundred-megabyte download that would buy nothing here.
#
# Safe to re-run.

set -u

BOLD=$'\033[1m'; RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; OFF=$'\033[0m'
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

FAIL=0
ok()   { echo "  ${GRN}ok${OFF}    $*"; }
warn() { echo "  ${YEL}warn${OFF}  $*"; }
bad()  { echo "  ${RED}FAIL${OFF}  $*"; FAIL=$((FAIL+1)); }
step() { echo; echo "${BOLD}$*${OFF}"; }

echo "${BOLD}FreshIQ install${OFF}   ($ROOT)"

# --------------------------------------------------------------------------
step "1. Python"
if ! command -v python3 >/dev/null 2>&1; then
  bad "python3 not found. Install it:  sudo apt install -y python3 python3-pip"
else
  PYV=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')
  ok "python3 $PYV"
  python3 - <<'EOF' || exit 1
import sys
if sys.version_info < (3, 9):
    sys.exit("FreshIQ needs Python 3.9 or newer")
EOF
fi

# --------------------------------------------------------------------------
step "2. System libraries"
# OpenCV wheels are not fully self-contained: they dlopen libGL and glib at
# import time, and the error you get without them ("libGL.so.1: cannot open
# shared object file") does not mention OpenCV at all.
if command -v apt-get >/dev/null 2>&1; then
  MISSING=""
  for pkg in libgl1 libglib2.0-0; do
    dpkg -s "$pkg" >/dev/null 2>&1 || MISSING="$MISSING $pkg"
  done
  if [ -n "$MISSING" ]; then
    echo "  installing:$MISSING"
    sudo apt-get update -qq && sudo apt-get install -y $MISSING >/dev/null \
      && ok "system libraries installed" \
      || bad "apt install failed. Run manually: sudo apt install -y$MISSING"
  else
    ok "libgl1, libglib2.0-0 present"
  fi
else
  warn "no apt-get; skipping system libraries (not Debian?)"
fi

# --------------------------------------------------------------------------
step "3. Python packages"
# Debian marks the system Python "externally managed" (PEP 668), so a plain
# pip install refuses. --break-system-packages is the documented escape hatch.
PIPFLAGS="--break-system-packages"
python3 -m pip install --help 2>/dev/null | grep -q break-system-packages || PIPFLAGS=""

python3 -m pip install $PIPFLAGS -q -r requirements.txt \
  && ok "requirements.txt installed" \
  || bad "pip install failed"

# --------------------------------------------------------------------------
step "4. Verify the runtime"
python3 - <<'EOF'
import sys
bad = 0
try:
    import numpy; print("  ok    numpy", numpy.__version__)
except Exception as e: print("  FAIL  numpy:", e); bad += 1
try:
    import cv2; print("  ok    opencv", cv2.__version__)
except Exception as e: print("  FAIL  opencv:", e); bad += 1

found = None
for mod, label in (("ai_edge_litert.interpreter", "ai-edge-litert"),
                   ("tflite_runtime.interpreter", "tflite-runtime"),
                   ("tensorflow.lite", "tensorflow")):
    try:
        __import__(mod); found = label; break
    except Exception:
        continue
if found:
    print("  ok    tflite runtime:", found)
else:
    print("  FAIL  no TFLite/LiteRT runtime. Try: pip3 install --break-system-packages ai-edge-litert")
    bad += 1

try:
    import serial; print("  ok    pyserial", serial.__version__)
except Exception:
    print("  warn  pyserial missing - the tray will not move.")
    print("        pip3 install --break-system-packages pyserial")
sys.exit(1 if bad else 0)
EOF
[ $? -eq 0 ] || FAIL=$((FAIL+1))

# --------------------------------------------------------------------------
step "5. Model files"
python3 - <<'EOF'
import json, os, sys
bad = 0
tfl = "models/freshiq_model.tflite"
lbl = "models/labels.json"

if not os.path.exists(tfl):
    print("  FAIL  %s is missing." % tfl)
    print("        models/ is gitignored, so it does NOT arrive with a git clone.")
    print("        Copy it from your laptop (see README, 'Deploying').")
    bad += 1
else:
    print("  ok    %s (%.1f MB)" % (tfl, os.path.getsize(tfl) / 1e6))

if not os.path.exists(lbl):
    print("  FAIL  %s is missing - a model cannot be interpreted without it." % lbl)
    bad += 1
else:
    L = json.load(open(lbl, encoding="utf-8"))
    print("  ok    labels.json: %d conditions %s, %d foods, input %s" % (
        len(L.get("classes", [])), L.get("classes"),
        len(L.get("foods", [])), L.get("input_size")))

    # The single most damaging silent failure: labels that do not describe the
    # model actually being loaded. Check the real tensors, not the docs.
    if os.path.exists(tfl):
        try:
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                from tflite_runtime.interpreter import Interpreter
            it = Interpreter(model_path=tfl); it.allocate_tensors()
            widths = sorted(int(o["shape"][-1]) for o in it.get_output_details())
            want = sorted([len(L.get("classes", [])), len(L.get("foods", []))])
            size = int(it.get_input_details()[0]["shape"][1])
            if widths != want:
                print("  FAIL  model outputs %s but labels.json describes %s" % (widths, want))
                bad += 1
            elif size != int(L.get("input_size", 0)):
                print("  FAIL  model input is %dpx but labels.json says %s" % (size, L.get("input_size")))
                bad += 1
            else:
                print("  ok    model matches labels (outputs %s, input %dpx)" % (widths, size))
        except Exception as e:
            print("  warn  could not introspect the model:", e)
sys.exit(1 if bad else 0)
EOF
[ $? -eq 0 ] || FAIL=$((FAIL+1))

# --------------------------------------------------------------------------
step "6. Device config"
python3 - <<'EOF'
import sys
try:
    from config import load_settings
    s = load_settings()
    print("  ok    %s" % s.source_path)
    print("        mode=%s  compartments=%d  steps/compartment=%d"
          % (s.mode, s.tray.compartments, s.steps_per_compartment()))
    print("        motor=%s %s   camera index=%s   port=%s"
          % (s.motor.driver, s.motor.port, s.camera.index, s.server.port))
except Exception as e:
    print("  FAIL  config/freshiq.json:", e); sys.exit(1)
EOF
[ $? -eq 0 ] || FAIL=$((FAIL+1))

# --------------------------------------------------------------------------
step "7. Data directories"
mkdir -p data/inspections && ok "data/inspections"

# --------------------------------------------------------------------------
step "8. Hardware present?"
CAMS=$(ls /dev/video* 2>/dev/null | tr '\n' ' ')
[ -n "$CAMS" ] && ok "camera device(s): $CAMS" \
               || warn "no /dev/video* found. Plug the USB camera in."

PORTS=$(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | tr '\n' ' ')
if [ -n "$PORTS" ]; then
  ok "serial port(s): $PORTS"
  echo "        Set motor.port in config/freshiq.json to the right one."
else
  warn "no /dev/ttyACM* or /dev/ttyUSB*. Flash firmware/freshiq_tray/ to the MCU."
fi

if ! id -nG "$USER" | grep -qw dialout; then
  warn "$USER is not in the 'dialout' group - opening the serial port will fail."
  echo "        sudo usermod -aG dialout $USER   then log out and back in"
fi

# --------------------------------------------------------------------------
echo
if [ "$FAIL" -eq 0 ]; then
  echo "${GRN}${BOLD}Install OK.${OFF}  Start it with:   ./start_freshiq.sh"
else
  echo "${RED}${BOLD}$FAIL check(s) failed.${OFF} Fix the FAIL lines above, then re-run ./install.sh"
  exit 1
fi
