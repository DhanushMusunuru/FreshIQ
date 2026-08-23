"""
Connection probe. Proves the repo -> Fusion 360 link works in both directions
BEFORE we spend effort on the real model.

It answers three questions:
  1. Can Fusion see and import code from the project folder?   (import worked)
  2. Can we read the running application state?                 (read path)
  3. Can we actually create geometry?                           (write path)

Then it writes everything to cad/out/last_run.json so the answers come back.
"""

import adsk.core
import adsk.fusion

from .params import MOTOR, UNOQ, LAZY, HUB

MM = 0.1  # Fusion's internal unit is cm; 1 mm = 0.1 cm


def run(ctx):
    app = adsk.core.Application.get()
    ui = app.userInterface

    report = {
        "fusion_version": app.version,
        "user": app.currentUser.displayName if app.currentUser else None,
        "documents_open": app.documents.count,
        "params_loaded": {
            "motor_frame_mm": MOTOR["frame"],
            "unoq_pcb_mm": [UNOQ["pcb_l"], UNOQ["pcb_w"]],
            "lazy_susan_mm": LAZY["plate"],
            "hub_flange_mm": HUB["flange_dia"],
        },
    }

    # ---- read path -------------------------------------------------------
    doc = app.activeDocument
    if doc:
        report["active_document"] = doc.name
        report["active_doc_saved"] = doc.isSaved
    else:
        report["active_document"] = None

    design = adsk.fusion.Design.cast(app.activeProduct)
    report["design_workspace_active"] = design is not None
    if design:
        units = design.unitsManager
        report["default_units"] = units.defaultLengthUnits
        report["design_type"] = (
            "parametric"
            if design.designType == adsk.fusion.DesignTypes.ParametricDesignType
            else "direct"
        )

    # ---- write path ------------------------------------------------------
    # Create a scratch design and put one 20 mm cube in it. If this succeeds,
    # the full model build will succeed too.
    if ctx.get("create_probe", True):
        probe_doc = app.documents.add(
            adsk.core.DocumentTypes.FusionDesignDocumentType
        )
        probe_design = adsk.fusion.Design.cast(app.activeProduct)
        probe_design.designType = adsk.fusion.DesignTypes.ParametricDesignType
        # defaultLengthUnits is read-only; distanceDisplayUnits is the setter.
        probe_design.unitsManager.distanceDisplayUnits = (
            adsk.fusion.DistanceUnits.MillimeterDistanceUnits)

        root = probe_design.rootComponent
        # NB: root component name follows the document name and cannot be set
        # on an unsaved document, so it is left alone.

        sketch = root.sketches.add(root.xYConstructionPlane)
        lines = sketch.sketchCurves.sketchLines
        lines.addTwoPointRectangle(
            adsk.core.Point3D.create(0, 0, 0),
            adsk.core.Point3D.create(20 * MM, 20 * MM, 0),
        )
        prof = sketch.profiles.item(0)
        ext = root.features.extrudeFeatures.addSimple(
            prof,
            adsk.core.ValueInput.createByReal(20 * MM),
            adsk.fusion.FeatureOperations.NewBodyFeatureOperation,
        )

        body = ext.bodies.item(0)
        body.name = "link_ok"

        # Volume in mm^3 - a 20 mm cube must be 8000. Proves units are right.
        vol_mm3 = body.volume / (MM ** 3)
        report["probe_body_volume_mm3"] = round(vol_mm3, 3)
        report["probe_units_correct"] = abs(vol_mm3 - 8000.0) < 1.0
        report["probe_document"] = probe_doc.name

        app.activeViewport.fit()

    if ctx.get("show_dialog", True):
        ui.messageBox(
            "FreshIQ link OK\n\n"
            "Fusion {}\n"
            "Params loaded from the repo: motor frame {} mm, "
            "UNO Q {} x {} mm.\n\n"
            "You can close the probe document without saving.".format(
                app.version,
                MOTOR["frame"],
                UNOQ["pcb_l"],
                UNOQ["pcb_w"],
            ),
            "FreshIQ",
        )

    return {"ok": True, "report": report}
