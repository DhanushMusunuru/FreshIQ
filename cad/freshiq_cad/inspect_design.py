"""
Measure what Fusion actually built and check it against what was intended.

This is the test harness. build.py says where things should be; this reads the
real B-Rep back out and compares. A part that silently failed to cut a hole, or
extruded the wrong way, shows up here as a bounding box in the wrong place
rather than as a nice picture that happens to be wrong.
"""

import adsk.core
import adsk.fusion

from .build import compute_layout, MM
from .params import MOTOR, UNOQ, HUB, LAZY, BUILD as B, CARRIER, INSERT, SPLIT


def _bbox_mm(bb):
    return {
        "x": [round(bb.minPoint.x / MM, 2), round(bb.maxPoint.x / MM, 2)],
        "y": [round(bb.minPoint.y / MM, 2), round(bb.maxPoint.y / MM, 2)],
        "z": [round(bb.minPoint.z / MM, 2), round(bb.maxPoint.z / MM, 2)],
    }


def _union_bbox(detail):
    """Union of BODY boxes.

    component.boundingBox / occurrence.boundingBox include SKETCH geometry.
    The 400 x 200 trim rectangle and the panel sketches at y = +/-125 made
    split halves measure 247 mm instead of 125. Only bodies are the part.
    """
    if not detail:
        return None
    out = {}
    for ax in ("x", "y", "z"):
        lo = min(b["bbox_mm"][ax][0] for b in detail)
        hi = max(b["bbox_mm"][ax][1] for b in detail)
        out[ax] = [round(lo, 2), round(hi, 2)]
    return out


def _size_mm(bb):
    return [round((bb.maxPoint.x - bb.minPoint.x) / MM, 2),
            round((bb.maxPoint.y - bb.minPoint.y) / MM, 2),
            round((bb.maxPoint.z - bb.minPoint.z) / MM, 2)]


def run(ctx):
    app = adsk.core.Application.get()
    des = adsk.fusion.Design.cast(app.activeProduct)
    if des is None:
        return {"ok": False, "error": "no active design"}

    root = des.rootComponent
    L = compute_layout()

    comps = {}
    total_bodies = 0
    for occ in root.occurrences:
        c = occ.component
        bodies = []
        for b in c.bRepBodies:
            bodies.append({
                "name": b.name,
                "volume_mm3": round(b.volume / (MM ** 3), 1),
                "area_mm2": round(b.area / (MM ** 2), 1),
                "bbox_mm": _bbox_mm(b.boundingBox),
                "size_mm": _size_mm(b.boundingBox),
                "faces": b.faces.count,
                "edges": b.edges.count,
                "valid": b.isValid,
            })
            total_bodies += 1
        comps[c.name] = {
            "bodies": len(bodies),
            "detail": bodies,
            "bbox_mm": _union_bbox(bodies),
        }

    # ---- assertions against the intended layout -------------------------
    tests = []

    def expect(name, got, want, tol, unit="mm"):
        ok = got is not None and abs(got - want) <= tol
        tests.append({
            "test": name,
            "ok": bool(ok),
            "got": None if got is None else round(got, 2),
            "want": round(want, 2),
            "tol": tol,
            "unit": unit,
        })

    # Measure the WHOLE component, not body[0] - the reference parts are
    # multi-body (motor = body + boss + shaft), so body[0] alone understates
    # their extent and produced two false failures.
    def zrange(comp_name):
        c = comps.get(comp_name)
        if not c or not c["bbox_mm"]:
            return None, None
        z = c["bbox_mm"]["z"]
        return z[0], z[1]

    def span(comp_name, axis):
        c = comps.get(comp_name)
        if not c or not c["bbox_mm"]:
            return None
        lo, hi = c["bbox_mm"][axis]
        return hi - lo

    # Base splits front/back at Y=0; deck splits left/right at X=0.
    for suf in ("A", "B"):
        n = "01_Base_Shell_" + suf
        lo, hi = zrange(n)
        expect("base %s sits on Z0" % suf, lo, 0.0, 0.2)
        expect("base %s top" % suf, hi, L["deck_bot"], 0.2)
        expect("base %s full width in X" % suf, span(n, "x"), B["base_size"], 0.5)
        expect("base %s is a HALF in Y" % suf, span(n, "y"), B["base_size"] / 2.0, 0.6)

        n = "02_Deck_" + suf
        lo, hi = zrange(n)
        # deck bottom is the seam pad hanging below the plate, not the plate
        expect("deck %s pad bottom" % suf, lo, SPLIT["deck_pad_z0"], 0.3)
        expect("deck %s top" % suf, hi, L["deck_top"], 0.2)
        expect("deck %s is a HALF in X" % suf, span(n, "x"), B["base_size"] / 2.0, 0.6)
        expect("deck %s full depth in Y" % suf, span(n, "y"), B["base_size"], 0.5)

    lo, hi = zrange("03_Pedestal")
    expect("pedestal bottom", lo, L["deck_top"], 0.2)
    expect("pedestal top", hi, L["lazy_bot"], 0.2)
    expect("pedestal OD", span("03_Pedestal", "x"), B["pedestal_od"], 0.5)

    lo, hi = zrange("04_Tray_Carrier")
    expect("carrier bottom sits on bearing", lo, L["carrier_bot"], 0.2)
    expect("carrier rim top", hi, L["carrier_rim_top"], 0.2)
    expect("carrier diameter", span("04_Tray_Carrier", "x"), CARRIER["od"], 0.5)

    lo, hi = zrange("08_Tray_Insert")
    expect("insert sits on carrier floor", lo, L["insert_bot"], 0.2)
    expect("insert rim top", hi, L["tray_rim_top"], 0.2)
    expect("insert clears carrier bore",
           span("08_Tray_Insert", "x"),
           CARRIER["od"] - 2 * CARRIER["rim_wall"] - INSERT["fit_clear"], 0.5)

    lo, hi = zrange("05_Arm_Post")
    expect("post starts at deck", lo, L["deck_top"], 0.2)
    expect("post top", hi, L["head_top"], 0.5)

    lo, hi = zrange("06_Arm_Head")
    expect("arm head bottom = lens height", lo, L["head_bot"], 0.5)

    # Z alone is not enough: the head previously sat entirely behind the
    # camera, so the pocket and aperture cut nothing. Check Y containment.
    head = comps.get("06_Arm_Head")
    if head and head["bbox_mm"]:
        ylo, yhi = head["bbox_mm"]["y"]
        tests.append({
            "test": "arm head spans the camera in Y",
            "ok": ylo - 0.5 <= L["cam_y"] <= yhi + 0.5,
            "got": "head y {}..{}, camera y {}".format(ylo, yhi, round(L["cam_y"], 1)),
            "want": "camera inside head",
            "tol": None,
            "unit": "mm",
        })
        pxlo = comps.get("05_Arm_Post")
        if pxlo and pxlo["bbox_mm"]:
            plo, phi = pxlo["bbox_mm"]["y"]
            tests.append({
                "test": "arm head reaches back to the post",
                "ok": ylo <= plo + 0.5,
                "got": "head back {}, post back {}".format(ylo, plo),
                "want": "head >= post depth",
                "tol": None,
                "unit": "mm",
            })

    lo, hi = zrange("07_Camera_Tilt_Bracket")
    expect("camera plate bottom", lo, L["plate_bot"], 0.3)

    plate = comps.get("07_Camera_Tilt_Bracket")
    if plate and plate["bbox_mm"] and head and head["bbox_mm"]:
        plo, phi = plate["bbox_mm"]["y"]
        hlo, hhi = head["bbox_mm"]["y"]
        tests.append({
            "test": "camera plate sits under the arm head",
            "ok": plo >= hlo - 0.5 and phi <= hhi + 0.5,
            "got": "plate y {}..{} vs head {}..{}".format(plo, phi, hlo, hhi),
            "want": "plate inside head footprint",
            "tol": None,
            "unit": "mm",
        })

    lo, hi = zrange("REF_NEMA17_Motor")
    expect("motor body bottom", lo, L["motor_bot"], 0.2)
    expect("shaft top", hi, L["shaft_top"], 0.3)

    lo, hi = zrange("REF_Coupling_Hub")
    expect("hub bottom", lo, L["hub_bot"], 0.2)
    expect("hub top meets tray", hi, L["tray_bot"], 0.2)

    # The insert must be hollow - a solid puck of the same envelope would mean
    # the compartment pockets never got cut.
    ins = comps.get("08_Tray_Insert")
    if ins and ins["detail"]:
        vol = ins["detail"][0]["volume_mm3"]
        od = CARRIER["od"] - 2 * CARRIER["rim_wall"] - INSERT["fit_clear"]
        solid = 3.14159 * (od / 2.0) ** 2 * B["tray_rim_h"]
        frac = vol / solid
        tests.append({
            "test": "insert is hollowed (volume < 45% of solid puck)",
            "ok": frac < 0.45,
            "got": round(frac * 100, 1),
            "want": "< 45",
            "tol": None,
            "unit": "% of solid",
        })

    # Base shell must be shelled, not a solid brick.
    shell = comps.get("01_Base_Shell_A")
    if shell and shell["detail"]:
        vol = shell["detail"][0]["volume_mm3"]
        solid = B["base_size"] ** 2 * L["deck_bot"] / 2.0
        frac = vol / solid
        tests.append({
            "test": "base shell is hollowed (volume < 35% of solid block)",
            "ok": frac < 0.35,
            "got": round(frac * 100, 1),
            "want": "< 35",
            "tol": None,
            "unit": "% of solid",
        })

    joints = [{"name": j.name,
               "type": type(j.jointMotion).__name__,
               "suppressed": j.isSuppressed} for j in root.joints]

    all_detail = [b for c in comps.values() for b in c["detail"]]
    env = _union_bbox(all_detail) or {"x": [0, 0], "y": [0, 0], "z": [0, 0]}
    failed = [t for t in tests if not t["ok"]]

    return {
        "ok": len(failed) == 0,
        "report": {
            "document": app.activeDocument.name if app.activeDocument else None,
            "root": root.name,
            "components": len(comps),
            "bodies": total_bodies,
            "joints": joints,
            "overall_envelope_mm": [round(env[a][1] - env[a][0], 2) for a in "xyz"],
            "tests_run": len(tests),
            "tests_failed": len(failed),
            "failures": failed,
            "tests": tests if ctx.get("verbose", False) else None,
            "component_detail": comps if ctx.get("verbose", False) else
                {k: {"bodies": v["bodies"], "bbox_mm": v["bbox_mm"]}
                 for k, v in comps.items()},
        },
    }
