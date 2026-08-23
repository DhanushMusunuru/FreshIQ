"""
Test bootstrap.

The application lives under python/ because that is the layout Arduino App Lab
expects (python/ beside sketch/, with app.yaml at the root). The test suite sits
at the repository root so it is not shipped to the board. This file bridges the
two: it puts python/ on sys.path so `import backend`, `import ai`,
`import hardware` and `import config` resolve exactly as they do at runtime on
the UNO Q.

Doing it here rather than with a sys.path line in every test file means the
tests import the same modules the device does - not a copy that can drift.
"""

import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
PYDIR = os.path.join(REPO, "python")

if PYDIR not in sys.path:
    sys.path.insert(0, PYDIR)

# Config, models and sample data are addressed relative to python/ at runtime
# (main.py chdir's there), so tests must resolve them the same way.
CONFIG_JSON = os.path.join(PYDIR, "config", "freshiq.json")
MODELS_DIR = os.path.join(PYDIR, "models")
CAD_PARAMS = os.path.join(REPO, "cad", "freshiq_cad", "params.py")
