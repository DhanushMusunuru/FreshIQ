#!/usr/bin/env bash
#
#   ./make_app_bundle.sh
#
# Produces build/FreshIQ/ (and FreshIQ_app.zip) - exactly what Arduino App Lab
# imports, and nothing else.
#
# The repository holds far more than the board needs: CAD, the test suite, the
# training pipeline, documentation, the trainable .keras model. Copying all of
# that onto the UNO Q wastes space and, worse, makes it ambiguous which model
# is the deployed one. This script takes only the three things App Lab looks
# for - app.yaml, python/, sketch/ - plus the launcher scripts.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

OUT="build/FreshIQ"
rm -rf build && mkdir -p "$OUT"

cp app.yaml "$OUT"/
cp START.sh INSTALL_SERVICE.sh "$OUT"/ 2>/dev/null || true
cp -r sketch "$OUT"/
cp -r python "$OUT"/

# Strip what must not travel to the board.
find "$OUT" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$OUT" -name "*.pyc" -delete 2>/dev/null || true
find "$OUT" -name "desktop.ini" -delete 2>/dev/null || true
# The trainable Keras model and the unused float export: the device loads
# freshiq_model.tflite and nothing else. Shipping the others invites deploying
# the wrong one, which degrades accuracy silently rather than failing loudly.
rm -f "$OUT"/python/models/freshiq_model.keras
rm -f "$OUT"/python/models/freshiq_model_float.tflite
rm -f "$OUT"/python/models/training_log*.csv
# Runtime history belongs to the board, not to the bundle.
rm -f "$OUT"/python/data/inspections.jsonl
rm -rf "$OUT"/python/data/inspections/*/
mkdir -p "$OUT"/python/data/inspections

chmod +x "$OUT"/*.sh 2>/dev/null || true

# Fail loudly if the one file that actually matters did not make it.
test -f "$OUT/python/models/freshiq_model.tflite" || { echo "FATAL: model missing"; exit 1; }
test -f "$OUT/python/models/labels.json"          || { echo "FATAL: labels missing"; exit 1; }
test -f "$OUT/app.yaml"                            || { echo "FATAL: app.yaml missing"; exit 1; }
test -f "$OUT/sketch/sketch.ino"                   || { echo "FATAL: sketch missing"; exit 1; }

if command -v zip >/dev/null 2>&1; then
  (cd build && zip -qr ../FreshIQ_app.zip FreshIQ)
  echo "zip:    FreshIQ_app.zip"
fi

echo "folder: $OUT"
echo "files:  $(find "$OUT" -type f | wc -l)"
echo "size:   $(du -sm "$OUT" | cut -f1) MB"
