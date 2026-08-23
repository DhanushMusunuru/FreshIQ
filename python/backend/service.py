"""
Minimal backend adapter (PHASE 5).

This is the thin seam between the AI module and the rest of FreshIQ (backend ->
Firebase -> dashboard). It is intentionally small: the AI is only responsible for
image analysis, so this file just shapes an inspection into a transport-ready
record and, optionally, exposes it over HTTP for local testing.

It does NOT implement Firebase, auth, or the dashboard — those are out of scope
for the AI task. `build_inspection_record` returns a plain dict you can hand to
whatever persistence layer you use (e.g. firebase_admin's Firestore client).

Run a local test server (stdlib only, no Flask needed):
    python backend/service.py --serve --port 8000
Then POST an image:
    curl -F "image=@sample_data/tomato_spoiled.jpg" -F "food=tomato" \
         http://localhost:8000/inspect
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from typing import Dict, List, Optional, Union

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai import inspect, __version__ as ai_version  # noqa: E402


def build_inspection_record(images: List[Union[str, "np.ndarray"]],
                            food_hint: Optional[str] = None,
                            device_id: str = "freshiq-unit-01",
                            inspection_id: Optional[str] = None) -> Dict:
    """
    Run a full multi-view inspection and wrap it in a record ready to persist.

    The returned dict is JSON-serialisable and dashboard-friendly:
        {
          "inspection_id": "...",
          "device_id": "freshiq-unit-01",
          "ai_version": "0.1.0",
          "result": { ...aggregated inspection... },
          "disclaimer": "Visual estimate only; not a food-safety measurement."
        }

    A caller (the real backend) is responsible for adding a timestamp and writing
    this to Firebase/DB. We deliberately do not call time/uuid-dependent services
    beyond a random id, to keep this pure and testable.
    """
    result = inspect(images, food_hint=food_hint)
    # Trim internal per-view raw floats for a clean payload.
    for v in result.get("views", []):
        v.pop("confidence_raw", None)

    return {
        "inspection_id": inspection_id or f"insp_{uuid.uuid4().hex[:12]}",
        "device_id": device_id,
        "ai_version": ai_version,
        "result": result,
        "disclaimer": (
            "FreshIQ provides a visual estimate of appearance-based "
            "freshness/ripeness/spoilage. It is NOT a scientific measurement of "
            "food safety or chemical freshness."
        ),
    }


# --------------------------------------------------------------------------- #
# Optional local HTTP test server (stdlib only)
# --------------------------------------------------------------------------- #
def _run_server(port: int):
    """
    Removed. This was a second, standalone HTTP server that predates
    backend/server.py, and it parsed multipart uploads with the `cgi` module.

    `cgi` was REMOVED from the standard library in Python 3.13, and the UNO Q
    runs 3.13 - so this path could only ever raise ModuleNotFoundError on the
    target hardware. Rather than leave a route that cannot work, it now says so
    and points at the server that does.

    Nothing else is lost: backend/server.py serves the same inspection endpoint
    (plus the dashboard, the tray and the history) and does its own multipart
    handling without `cgi`.
    """
    raise RuntimeError(
        "backend/service.py --serve was removed: it relied on the `cgi` "
        "module, which no longer exists in Python 3.13 (the UNO Q's "
        "version). Use the real server instead:  "
        "python3 backend/server.py --port {}  "
        "build_inspection_record() in this module is unaffected and is "
        "still what the backend uses.".format(port))


def main():
    import argparse
    ap = argparse.ArgumentParser(description="FreshIQ backend adapter / test server.")
    ap.add_argument("--serve", action="store_true", help="run the local HTTP test server")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("images", nargs="*", help="if not serving, inspect these images and print the record")
    ap.add_argument("--food", default=None)
    args = ap.parse_args()

    if args.serve:
        _run_server(args.port)
    elif args.images:
        record = build_inspection_record(args.images, food_hint=args.food)
        print(json.dumps(record, indent=2))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
