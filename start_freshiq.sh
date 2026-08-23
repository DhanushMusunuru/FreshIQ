#!/usr/bin/env bash
#
# Start FreshIQ.
#
#   ./start_freshiq.sh                   production (config/freshiq.json)
#   ./start_freshiq.sh --dev             no hardware needed
#   ./start_freshiq.sh --port 9000       override the port
#   ./start_freshiq.sh --no-tray         camera + AI only, motor untouched
#
# ONE process serves everything: the API, the dashboard and the scan
# controller. There is no separate frontend server, no message broker and no
# database daemon to babysit - on a single-board computer each of those is one
# more thing to fail at 2am for no benefit at this scale.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONFIG="config/freshiq.json"
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --dev|--development)
      CONFIG="config/freshiq.development.json"; shift ;;
    --config)
      CONFIG="$2"; shift 2 ;;
    *)
      EXTRA+=("$1"); shift ;;
  esac
done

if [ ! -f "$CONFIG" ]; then
  echo "config not found: $CONFIG" >&2
  exit 1
fi

if [ ! -f models/freshiq_model.tflite ]; then
  echo
  echo "  WARNING: models/freshiq_model.tflite is missing."
  echo "  FreshIQ will fall back to the rule-based backend, which is much less"
  echo "  accurate. models/ is gitignored - copy it from your laptop."
  echo
fi

PY=python3
command -v python3 >/dev/null 2>&1 || PY=python

# -u keeps the log unbuffered, so `journalctl -f` shows progress live instead of
# in 4 KB bursts.
exec "$PY" -u backend/server.py --config "$CONFIG" ${EXTRA[@]+"${EXTRA[@]}"}
