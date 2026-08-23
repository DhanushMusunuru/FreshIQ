"""
Explainer animation, rendered as PNG sequences from the Fusion viewport.

Fusion's Animation workspace has no public API, so this does it the honest way:
drive the model (joint angles, occurrence transforms, camera) one frame at a
time and capture the viewport each step. ffmpeg turns the sequences into video
outside Fusion.

Three scenes:
  orbit    - hero turntable, camera circles the assembly while the tray turns
  index    - fixed 3/4 view, tray steps compartment to compartment and dwells,
             which is the actual inspection cycle the machine performs
  explode  - parts separate along Z so the stack-up reads at a glance

Frames land in cad/out/anim/<scene>/frame_####.png
"""

import math
import os
import shutil

import adsk.core
import adsk.fusion

from .build import compute_layout, MM, cm
from .params import BUILD as B


def _anim_dir(scene, clean=True):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(here, "out", "anim", scene)
    if clean and os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    return d


def _ease(t):
    """Smoothstep, so starts and stops do not look robotic."""
    return t * t * (3.0 - 2.0 * t)


def set_camera(app, az_deg, el_deg, radius, target_z, extents, perspective=True):
    """Place the camera on a sphere around the model.

    viewExtents ONLY works on an orthographic camera - setting it on a
    perspective one raises "Camera type must be orthographic for extents".
    So perspective framing is driven purely by eye-to-target distance, and
    extents is applied only in the orthographic branch.
    """
    vp = app.activeViewport
    cam = vp.camera
    cam.cameraType = (adsk.core.CameraTypes.PerspectiveCameraType if perspective
                      else adsk.core.CameraTypes.OrthographicCameraType)

    az = math.radians(az_deg)
    el = math.radians(el_deg)
    cam.target = adsk.core.Point3D.create(0.0, 0.0, cm(target_z))
    cam.eye = adsk.core.Point3D.create(
        cm(radius * math.cos(el) * math.cos(az)),
        cm(radius * math.cos(el) * math.sin(az)),
        cm(target_z + radius * math.sin(el)),
    )
    cam.upVector = adsk.core.Vector3D.create(0.0, 0.0, 1.0)
    cam.isSmoothTransition = False
    vp.camera = cam

    if not perspective:
        cam2 = vp.camera
        cam2.viewExtents = cm(extents)
        vp.camera = cam2
    return vp


def find_tray_joint(root):
    for j in root.joints:
        if j.name == "tray_rotation":
            return j
    for j in root.joints:
        if isinstance(j.jointMotion, adsk.fusion.RevoluteJointMotion):
            return j
    return None


def set_tray_angle(joint, deg):
    if joint is None:
        return False
    motion = joint.jointMotion
    if not isinstance(motion, adsk.fusion.RevoluteJointMotion):
        return False
    motion.rotationValue = math.radians(deg)
    return True


def occurrences_by_name(root):
    return {occ.component.name: occ for occ in root.occurrences}


# How far each part travels in the exploded view, in mm along Z.
EXPLODE = {
    "08_Tray_Insert": 165.0,      # lifts off first - it is the washable part
    "04_Tray_Carrier": 115.0,
    "REF_Lazy_Susan": 80.0,
    "REF_Coupling_Hub": 55.0,
    "03_Pedestal": 35.0,
    "02_Deck_A": 18.0,
    "02_Deck_B": 18.0,
    "REF_NEMA17_Motor": -75.0,
    "REF_Electronics": -45.0,
    "01_Base_Shell_A": 0.0,
    "01_Base_Shell_B": 0.0,
    "05_Arm_Post": 0.0,
    "06_Arm_Head": 70.0,
    "07_Camera_Tilt_Bracket": 105.0,
    "REF_Camera": 130.0,
}


def set_explode(occs, t):
    """t in 0..1. Returns how many occurrences actually moved."""
    moved = 0
    for name, dz in EXPLODE.items():
        occ = occs.get(name)
        if occ is None:
            continue
        m = adsk.core.Matrix3D.create()
        m.translation = adsk.core.Vector3D.create(0.0, 0.0, cm(dz * t))
        try:
            occ.transform2 = m
            moved += 1
        except Exception:  # noqa: BLE001
            try:
                occ.transform = m
                moved += 1
            except Exception:  # noqa: BLE001
                pass
    return moved


def reset_explode(occs):
    ident = adsk.core.Matrix3D.create()
    for name in EXPLODE:
        occ = occs.get(name)
        if occ is None:
            continue
        try:
            occ.transform2 = ident
        except Exception:  # noqa: BLE001
            try:
                occ.transform = ident
            except Exception:  # noqa: BLE001
                pass


def clean_view(app, des):
    """Hide everything that says 'CAD tool' rather than 'product'.

    Sketches, construction planes and origin markers all render into the
    viewport capture - the deck came out covered in little sketch circles
    until these were switched off.
    """
    notes = {}
    root = des.rootComponent

    hidden = 0
    for comp in [root] + [o.component for o in root.occurrences]:
        for attr in ("isSketchFolderLightBulbOn",
                     "isConstructionFolderLightBulbOn",
                     "isOriginFolderLightBulbOn",
                     "isJointsFolderLightBulbOn"):
            try:
                setattr(comp, attr, False)
                hidden += 1
            except Exception:  # noqa: BLE001
                pass
    notes["folders_hidden"] = hidden

    # The grid is not exposed as a plain viewport property and the name has
    # moved between releases, so discover it instead of guessing.
    hits = []
    for label, obj in (("graphics", getattr(app.preferences, "graphicsPreferences", None)),
                       ("general", getattr(app.preferences, "generalPreferences", None))):
        if obj is None:
            continue
        names = [a for a in dir(obj) if not a.startswith("_")]
        notes["{}_prefs_attrs".format(label)] = names
        for attr in names:
            if "grid" in attr.lower():
                try:
                    setattr(obj, attr, False)
                    hits.append("{}.{}".format(label, attr))
                except Exception:  # noqa: BLE001
                    pass
    notes["grid_props_set"] = hits

    # Nothing in graphicsPreferences/generalPreferences mentions the grid and
    # the obvious command names do not exist. Fusion's own error says to use
    # "?" for a listing, so ask it what it actually has and try what comes
    # back rather than guessing further.
    candidates = []
    try:
        listing = str(app.executeTextCommand("?"))
        notes["command_listing_chars"] = len(listing)
        for line in listing.splitlines():
            tok = line.strip().split()
            if tok and "grid" in tok[0].lower():
                candidates.append(tok[0])
        notes["grid_commands_found"] = candidates[:20]
    except Exception as exc:  # noqa: BLE001
        notes["command_listing_error"] = str(exc)[:120]

    tried = {}
    for name in candidates[:8]:
        for arg in (" false", " 0", ""):
            try:
                tried[name + arg] = str(app.executeTextCommand(name + arg))[:60]
                break
            except Exception as exc:  # noqa: BLE001
                tried[name + arg] = "ERR {}".format(str(exc)[:50])
    notes["grid_commands_tried"] = tried

    return notes


def capture(app, path, w, h):
    app.activeViewport.refresh()
    adsk.doEvents()
    return app.activeViewport.saveAsImageFile(path, w, h)


def run(ctx):
    app = adsk.core.Application.get()
    des = adsk.fusion.Design.cast(app.activeProduct)
    if des is None:
        return {"ok": False, "error": "no active design - run build first"}

    root = des.rootComponent
    L = compute_layout()
    occs = occurrences_by_name(root)
    joint = find_tray_joint(root)

    w = int(ctx.get("width", 1280))
    h = int(ctx.get("height", 720))
    scenes = ctx.get("scenes", ["orbit", "index", "explode"])
    n_orbit = int(ctx.get("orbit_frames", 96))
    n_index = int(ctx.get("index_frames", 120))
    n_explode = int(ctx.get("explode_frames", 90))

    # ORTHOGRAPHIC by default. The camera handed back by vp.camera already
    # carries a viewExtents value, so simply flipping its type to perspective
    # and assigning it raises "Camera type must be orthographic for extents"
    # before any framing code runs. Orthographic also gives fully
    # deterministic framing via extents, which suits a technical explainer.
    perspective = bool(ctx.get("perspective", False))
    target_z = float(ctx.get("target_z", L["head_top"] * 0.42))
    radius = float(ctx.get("radius", 620.0))
    extents = float(ctx.get("extents", 340.0))
    elev = float(ctx.get("elevation", 24.0))

    vp = app.activeViewport
    try:
        vp.visualStyle = adsk.core.VisualStyles.ShadedWithVisibleEdgesOnlyVisualStyle
    except Exception:  # noqa: BLE001
        pass

    view_notes = clean_view(app, des)

    def cam(az, el, r=None, tz=None, ex=None):
        return set_camera(app, az, el,
                          radius if r is None else r,
                          target_z if tz is None else tz,
                          extents if ex is None else ex,
                          perspective)

    report = {
        "joint_found": joint is not None,
        "occurrences": len(occs),
        "image": [w, h],
        "perspective": perspective,
        "radius": radius,
        "extents": extents,
        "view_cleanup": view_notes,
        "scenes": {},
    }

    # ---------------------------------------------------------------- orbit
    if "orbit" in scenes:
        d = _anim_dir("orbit")
        written = 0
        for i in range(n_orbit):
            t = i / float(n_orbit)
            cam(360.0 * t - 45.0, elev)
            set_tray_angle(joint, 360.0 * t * 0.5)
            if capture(app, os.path.join(d, "frame_{:04d}.png".format(i)), w, h):
                written += 1
        report["scenes"]["orbit"] = {"frames": written, "dir": d}

    # ---------------------------------------------------------------- index
    # The real inspection cycle: rotate one compartment, hold while the camera
    # would be capturing, repeat.
    if "index" in scenes:
        d = _anim_dir("index")
        cam(-55.0, 30.0)
        n_comp = B["n_compartments"]
        step_deg = 360.0 / n_comp
        per = max(1, n_index // n_comp)
        move_frames = max(1, int(per * 0.55))
        written = 0
        f = 0
        for k in range(n_comp):
            for j in range(per):
                if j < move_frames:
                    local = _ease(j / float(move_frames))
                else:
                    local = 1.0
                ang = step_deg * (k + local)
                set_tray_angle(joint, ang)
                if capture(app, os.path.join(d, "frame_{:04d}.png".format(f)), w, h):
                    written += 1
                f += 1
        report["scenes"]["index"] = {
            "frames": written, "dir": d,
            "compartments": n_comp, "step_deg": round(step_deg, 1),
        }

    # -------------------------------------------------------------- explode
    if "explode" in scenes:
        d = _anim_dir("explode")
        set_tray_angle(joint, 0.0)
        written = 0
        moved = 0
        for i in range(n_explode):
            raw = i / float(n_explode - 1) if n_explode > 1 else 1.0
            # out and back, so the clip loops cleanly
            tri = 1.0 - abs(2.0 * raw - 1.0)
            t = _ease(min(1.0, tri * 1.35))
            moved = set_explode(occs, t)
            # ORTHOGRAPHIC: zoom comes from EXTENTS, not radius. Scaling
            # radius here did nothing and the exploded stack ran off the top
            # of the frame. The stack grows from ~256 mm tall to ~416 mm, so
            # extents has to open up by roughly the same factor.
            cam(-60.0 + 25.0 * raw, 14.0 + 10.0 * t,
                tz=target_z * (1.0 + 0.65 * t),
                ex=extents * (1.0 + 0.75 * t))
            if capture(app, os.path.join(d, "frame_{:04d}.png".format(i)), w, h):
                written += 1
        reset_explode(occs)
        report["scenes"]["explode"] = {
            "frames": written, "dir": d, "parts_moved": moved,
        }

    set_tray_angle(joint, 0.0)
    cam(-45.0, 24.0)
    app.activeViewport.refresh()

    total = sum(s.get("frames", 0) for s in report["scenes"].values())
    report["total_frames"] = total
    return {"ok": total > 0, "report": report}
