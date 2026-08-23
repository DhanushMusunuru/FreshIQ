"""
FreshIQ CAD - the half of the bridge that lives in the project repo.

The Fusion 360 side is a thin loader (installed into Fusion's Scripts folder)
that does nothing but import this package and call run(). All real logic lives
here, so the model can be edited in the repo and re-run in Fusion without ever
touching the Fusion install.

Contract with the loader:
    run(command: str, ctx: dict) -> dict
The returned dict is serialised to cad/out/last_run.json.
"""

import importlib
import sys
import traceback

__version__ = "0.1.0"

# Modules reloaded on every run so edits land without restarting Fusion.
_HOT = ("params", "handshake", "build", "inspect_design", "animate", "export",
        "style")

COMMANDS = ("handshake", "inspect", "build", "style", "animate", "export")


def _reload_all():
    """Force-reload our own submodules; Fusion keeps one Python session alive."""
    for name in _HOT:
        full = "{}.{}".format(__name__, name)
        mod = sys.modules.get(full)
        if mod is not None:
            importlib.reload(mod)


def run(command, ctx=None):
    ctx = ctx or {}
    _reload_all()

    if command == "handshake":
        from . import handshake
        return handshake.run(ctx)

    if command == "inspect":
        from . import inspect_design
        return inspect_design.run(ctx)

    if command == "build":
        from . import build
        return build.run(ctx)

    if command == "style":
        from . import style
        return style.run(ctx)

    if command == "animate":
        from . import animate
        return animate.run(ctx)

    if command == "export":
        from . import export
        return export.run(ctx)

    return {
        "ok": False,
        "command": command,
        "error": "unknown command: {!r}".format(command),
        "known_commands": list(COMMANDS),
    }


def safe_run(command, ctx=None):
    """run() with the traceback captured instead of thrown, for the loader."""
    try:
        result = run(command, ctx)
    except Exception as exc:  # noqa: BLE001 - we want everything
        return {
            "ok": False,
            "command": command,
            "error": "{}: {}".format(type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
        }
    result.setdefault("ok", True)
    result.setdefault("command", command)
    return result
