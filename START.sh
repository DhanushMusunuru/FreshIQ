#!/usr/bin/env bash
#
#   ./START.sh
#
# OPTIONAL. The normal way to run FreshIQ on an UNO Q is as an Arduino App:
#
#     arduino-app-cli app start ~/ArduinoApps/FreshIQ
#
# which starts BOTH halves - this Python code on Linux and sketch/sketch.ino on
# the microcontroller. Use START.sh only when you want the Linux half on its own
# (for example on a laptop, or while the sketch is already running).
#
# First run installs what is missing and then starts.
# Every run after that just starts, in about two seconds.
#
# Safe to run any number of times. It installs nothing that is already there.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; O=$'\033[0m'
STAMP=".installed"

PY=python3
command -v python3 >/dev/null 2>&1 || PY=python

# Prefer a .venv in the project if one exists and is not already active.
#
# Otherwise it is entirely possible to pip-install everything into .venv and
# then start the server with the SYSTEM python, which imports none of it. The
# symptom is a working dashboard with the tray reported unavailable, because
# msgpack "is installed" - just not in the interpreter that is running.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
  echo "${G}using${O} .venv"
fi

if ! command -v "$PY" >/dev/null 2>&1 && [ ! -x "$PY" ]; then
  echo "${R}Python 3 is not installed.${O}"
  echo "  sudo apt update && sudo apt install -y python3 python3-pip"
  exit 1
fi

mkdir -p python/data/inspections

# --------------------------------------------------------------------------
# First run only
# --------------------------------------------------------------------------
if [ ! -f "$STAMP" ]; then
  echo
  echo "${B}FreshIQ — first run, setting up. This takes a few minutes.${O}"
  echo

  if command -v apt-get >/dev/null 2>&1; then
    NEED=""
    for pkg in libgl1 libglib2.0-0; do
      dpkg -s "$pkg" >/dev/null 2>&1 || NEED="$NEED $pkg"
    done
    if [ -n "$NEED" ]; then
      # OpenCV dlopens libGL at import; without it you get a confusing
      # "libGL.so.1: cannot open shared object file" that never says OpenCV.
      echo "  installing system libraries:$NEED"
      sudo apt-get update -qq >/dev/null 2>&1
      sudo apt-get install -y $NEED >/dev/null 2>&1 \
        && echo "  ${G}ok${O} system libraries" \
        || echo "  ${Y}warn${O} could not install$NEED - if it fails to start, run: sudo apt install -y$NEED"
    fi
  fi

  echo "  installing Python packages (this is the slow part)..."
  FLAGS="--break-system-packages"
  "$PY" -m pip install --help 2>/dev/null | grep -q break-system-packages || FLAGS=""
  if "$PY" -m pip install $FLAGS -q -r python/requirements.txt 2>/dev/null; then
    echo "  ${G}ok${O} Python packages"
  else
    echo "  ${R}pip install failed.${O} Try:"
    echo "     $PY -m pip install $FLAGS -r python/requirements.txt"
    exit 1
  fi

  # Only claim success once the imports actually work.
  if "$PY" - <<'EOF'
import sys
try:
    import numpy, cv2
except Exception as e:
    sys.exit("import failed: %s" % e)
for m in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter", "tensorflow.lite"):
    try:
        __import__(m); break
    except Exception:
        continue
else:
    sys.exit("no TFLite/LiteRT runtime")
EOF
  then
    touch "$STAMP"
    echo "  ${G}ok${O} verified"
    echo
  else
    echo "  ${R}Setup did not complete.${O} See the message above."
    exit 1
  fi

  # Only relevant for motor.driver = "serial" (a SEPARATE Arduino on USB). The
  # UNO Q's own MCU is reached through arduino-router, which needs no group.
  if grep -q '"driver"[[:space:]]*:[[:space:]]*"serial"' python/config/freshiq.json 2>/dev/null \
     && ! id -nG "$USER" 2>/dev/null | grep -qw dialout; then
    echo "  ${Y}note${O} adding $USER to the 'dialout' group so the tray motor can be opened."
    sudo usermod -aG dialout "$USER" 2>/dev/null \
      && echo "       done - this takes effect after your NEXT login. Until then the" \
      && echo "       dashboard works but the tray will not move."
    echo
  fi
fi

# --------------------------------------------------------------------------
# Go
# --------------------------------------------------------------------------
CONFIG="python/config/freshiq.json"
PYTHON_ONLY=0
for a in "$@"; do
  case "$a" in
    --dev|--development) CONFIG="python/config/freshiq.development.json"; PYTHON_ONLY=1 ;;
    --python-only)       PYTHON_ONLY=1 ;;
  esac
done

# --------------------------------------------------------------------------
# On an UNO Q, hand over to App Lab.
#
# THIS SCRIPT ONLY EVER STARTED THE LINUX HALF. The tray needs the OTHER half -
# sketch/sketch.ino running on the onboard MCU - and only App Lab deploys that.
# Starting Python alone left the router reachable but nothing answering
# tray_status, so the dashboard said "Tray control unavailable" and the fault
# looked like the sketch or the wiring rather than how it was launched.
#
# `arduino-app-cli app start` flashes the sketch AND runs python/main.py, so it
# is the correct single command on this board. Fall through to Python-only when
# App Lab is not installed (a laptop), or when asked explicitly.
# --------------------------------------------------------------------------
APP_DIR="$(pwd)"
if [ "$PYTHON_ONLY" -eq 0 ] && command -v arduino-app-cli >/dev/null 2>&1    && [ -f app.yaml ] && [ -f sketch/sketch.ino ]; then
  echo "${B}Arduino UNO Q detected - starting BOTH halves through App Lab${O}"
  echo "  sketch -> onboard MCU, python -> Linux side"
  echo "  (use ./START.sh --python-only to skip the MCU)"
  echo

  # A previous instance holds /dev/video0 and the router session. Leaving it
  # running is what produces "no camera found" on a camera that works.
  arduino-app-cli app stop "$APP_DIR" >/dev/null 2>&1 || true

  IP=$(hostname -I 2>/dev/null | awk '{print $1}')
  [ -n "${IP:-}" ] && echo "  open  ${B}http://$IP:8000${O}  once it is up"
  echo
  exec arduino-app-cli app start "$APP_DIR"
fi

if [ "$PYTHON_ONLY" -eq 0 ]; then
  echo "${Y}note${O}  App Lab (arduino-app-cli) not found, so only the Linux half"
  echo "      will run. The tray needs sketch/sketch.ino on the MCU; without it"
  echo "      the dashboard reports 'Tray control unavailable'."
  echo
fi

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "${B}Starting FreshIQ${O}"
[ -n "${IP:-}" ] && echo "  open  ${B}http://$IP:8000${O}  on your phone or laptop"
echo

exec "$PY" -u python/backend/server.py --config "$CONFIG" "$@"
