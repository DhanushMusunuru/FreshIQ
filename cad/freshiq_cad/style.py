"""
Assign appearances so the explainer animation reads as a product, not a pile
of grey lumps.

Appearance names differ between Fusion releases and locales, so nothing here
hard-codes an exact string. Each slot lists keywords in priority order and
takes the first library appearance whose name contains all of them. If a slot
finds nothing it is reported and skipped rather than aborting the run.
"""

import adsk.core
import adsk.fusion

# slot -> list of candidate keyword sets, best first
PALETTE = {
    "white":  [("plastic", "matte", "white"), ("plastic", "white"), ("white",)],
    "grey":   [("plastic", "matte", "gray"), ("plastic", "matte", "grey"),
               ("plastic", "gray"), ("plastic", "grey")],
    "black":  [("plastic", "matte", "black"), ("plastic", "black"), ("black",)],
    "steel":  [("steel", "satin"), ("steel",), ("stainless",)],
    "alu":    [("aluminum", "satin"), ("aluminum",), ("aluminium",)],
    "green":  [("plastic", "matte", "green"), ("plastic", "green"), ("green",)],
    "pcb":    [("plastic", "matte", "green"), ("green",)],
}

# component name -> slot
ASSIGN = {
    "01_Base_Shell_A": "white",
    "01_Base_Shell_B": "white",
    "02_Deck_A": "white",
    "02_Deck_B": "white",
    "03_Pedestal": "grey",
    "04_Tray_Carrier": "grey",
    "08_Tray_Insert": "white",
    "05_Arm_Post": "white",
    "06_Arm_Head": "white",
    "07_Camera_Tilt_Bracket": "black",
    "REF_NEMA17_Motor": "steel",
    "REF_Coupling_Hub": "black",
    "REF_Lazy_Susan": "alu",
    "REF_Electronics": "pcb",
    "REF_Camera": "black",
}


def _find(lib_appearances, keywords):
    for ap in lib_appearances:
        low = ap.name.lower()
        if all(k in low for k in keywords):
            return ap
    return None


def run(ctx):
    app = adsk.core.Application.get()
    des = adsk.fusion.Design.cast(app.activeProduct)
    if des is None:
        return {"ok": False, "error": "no active design - run build first"}

    lib = None
    for name in ("Fusion 360 Appearance Library", "Fusion Appearance Library",
                 "Appearance Library"):
        try:
            lib = app.materialLibraries.itemByName(name)
        except Exception:  # noqa: BLE001
            lib = None
        if lib:
            break
    if lib is None and app.materialLibraries.count:
        lib = app.materialLibraries.item(app.materialLibraries.count - 1)
    if lib is None:
        return {"ok": False, "error": "no appearance library available"}

    resolved = {}
    missing = []
    for slot, candidates in PALETTE.items():
        found = None
        for kws in candidates:
            found = _find(lib.appearances, kws)
            if found:
                break
        if found is None:
            missing.append(slot)
            continue
        existing = des.appearances.itemByName(found.name)
        resolved[slot] = existing or des.appearances.addByCopy(found, found.name)

    root = des.rootComponent
    applied = {}
    skipped = []
    for occ in root.occurrences:
        name = occ.component.name
        slot = ASSIGN.get(name)
        ap = resolved.get(slot)
        if ap is None:
            skipped.append("{} (slot {!r} unavailable)".format(name, slot))
            continue
        n = 0
        for b in occ.component.bRepBodies:
            try:
                b.appearance = ap
                n += 1
            except Exception:  # noqa: BLE001
                pass
        applied[name] = {"appearance": ap.name, "bodies": n}

    return {
        "ok": bool(applied),
        "report": {
            "library": lib.name,
            "library_appearances": lib.appearances.count,
            "resolved_slots": {k: v.name for k, v in resolved.items()},
            "missing_slots": missing,
            "applied": applied,
            "skipped": skipped,
        },
    }
