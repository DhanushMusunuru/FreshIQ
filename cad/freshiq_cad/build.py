"""
FreshIQ - the model.

Everything is placed in ABSOLUTE WORLD COORDINATES:
    origin = centre of the base's bottom face, Z up, tray rotates about Z.
    +Y = front (OLED / knob face),  -Y = rear (gantry post).

That means components are created with identity transforms and geometry is
sketched where it actually lives. No occurrence transforms to get wrong, and a
revolute joint about Z still works because rotation about Z is in-place.

Two rules that keep the Fusion API from biting:
  * spt() converts a WORLD mm coordinate into sketch space via
    modelToSketchSpace, so no sketch-plane orientation ever has to be guessed.
  * add_polygon() chains lines through shared SketchPoints, so closed profiles
    are guaranteed to form instead of depending on coincident-point tolerance.
"""

import math
import traceback

import adsk.core
import adsk.fusion

from .params import (
    MOTOR, DRIVER, UNOQ, HUB, LAZY, CAMERA, POWER, BUILD as B, CAM_MOUNT as CM,
    CARRIER, INSERT, TILT, SPLIT,
)

MM = 0.1  # Fusion internal unit is cm


def cm(v):
    return v * MM


# ==========================================================================
# Layout - every Z height in the machine, derived once
# ==========================================================================

def compute_layout():
    L = {}
    L["floor_top"] = B["base_floor_t"]
    L["motor_bot"] = L["floor_top"] + B["motor_gap"]
    L["motor_face"] = L["motor_bot"] + MOTOR["body_len"]
    L["shaft_top"] = L["motor_face"] + MOTOR["shaft_len"]

    L["deck_bot"] = L["motor_face"]
    L["deck_top"] = L["deck_bot"] + B["deck_t"]

    L["hub_bot"] = L["deck_top"] + 2.0
    L["hub_top"] = L["hub_bot"] + HUB["total_len"]
    L["hub_flange_bot"] = L["hub_top"] - HUB["flange_t"]

    L["tray_bot"] = L["hub_top"]
    L["lazy_top"] = L["tray_bot"]
    L["lazy_bot"] = L["lazy_top"] - LAZY["height"]
    L["pedestal_h"] = L["lazy_bot"] - L["deck_top"]

    # Carrier: bolts to the hub flange, stays on the machine.
    L["carrier_bot"] = L["tray_bot"]
    L["carrier_floor_top"] = L["carrier_bot"] + CARRIER["floor_t"]
    L["carrier_rim_top"] = L["carrier_floor_top"] + CARRIER["rim_h"]

    # Insert: the food tray, lifts straight out of the carrier rim.
    L["insert_bot"] = L["carrier_floor_top"]
    L["tray_floor_top"] = L["insert_bot"] + B["tray_floor"]
    L["tray_rim_top"] = L["insert_bot"] + B["tray_rim_h"]
    L["tray_hub_top"] = L["carrier_floor_top"]   # kept for the hub bolt cuts

    # Camera stack, built UP from the lens plane so arm_h stays the thing you
    # actually tune. The bracket hangs below the head on the tilt pivot.
    L["lens_z"] = L["tray_floor_top"] + B["arm_h"]
    L["plate_bot"] = L["lens_z"]
    L["plate_top"] = L["plate_bot"] + CM["plate_t"]
    L["pivot_z"] = L["plate_top"] + TILT["arm_rise"]
    L["bracket_arm_top"] = L["pivot_z"] + TILT["arm_over"]
    L["head_bot"] = L["pivot_z"] - TILT["pivot_inset"]
    L["head_top"] = L["head_bot"] + B["arm_head_t"]

    # Gantry post: behind the tray, inside the base footprint.
    tray_r = B["tray_dia"] / 2.0
    base_half = B["base_size"] / 2.0
    L["post_y_inner"] = -(tray_r + B["arm_clear"])
    L["post_y_outer"] = L["post_y_inner"] - B["arm_d"]
    L["post_y"] = (L["post_y_inner"] + L["post_y_outer"]) / 2.0
    L["cam_y"] = L["post_y"] + B["arm_reach"]

    # Compartment centroid radius, for aiming the camera.
    L["compartment_r"] = (B["tray_hub_deck_d"] / 2.0 + tray_r) / 2.0

    checks = []

    def chk(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    eng = L["shaft_top"] - L["hub_bot"]
    chk("shaft engagement in hub >= 12mm", eng >= 12.0, "{:.2f} mm".format(eng))
    chk("hub clears deck", L["hub_bot"] > L["deck_top"],
        "{:.2f} mm".format(L["hub_bot"] - L["deck_top"]))
    chk("hub clears motor face", L["hub_bot"] > L["motor_face"],
        "{:.2f} mm".format(L["hub_bot"] - L["motor_face"]))
    chk("hub OD fits lazy-susan access hole",
        HUB["flange_dia"] < LAZY["access_hole"],
        "{:.1f} < {:.2f}".format(HUB["flange_dia"], LAZY["access_hole"]))
    chk("pedestal height positive", L["pedestal_h"] > 0,
        "{:.2f} mm".format(L["pedestal_h"]))
    chk("gantry post clears tray", L["post_y_inner"] <= -tray_r,
        "post inner {:.1f} vs tray edge {:.1f}".format(L["post_y_inner"], -tray_r))
    chk("gantry post inside base footprint", L["post_y_outer"] >= -base_half,
        "post outer {:.1f} vs base edge {:.1f}".format(L["post_y_outer"], -base_half))
    chk("motor fits under deck", L["motor_bot"] >= L["floor_top"],
        "{:.2f} mm above floor".format(L["motor_bot"] - L["floor_top"]))
    chk("UNO Q fits under deck",
        L["floor_top"] + 5.0 + UNOQ["pcb_t"] + UNOQ["above_max"] < L["deck_bot"],
        "top of board stack {:.1f} vs deck {:.1f}".format(
            L["floor_top"] + 5.0 + UNOQ["pcb_t"] + UNOQ["above_max"], L["deck_bot"]))

    fov = 2.0 * math.degrees(math.atan(
        (B["tray_dia"] / 2.0 * math.sin(math.pi / B["n_compartments"])) / B["arm_h"]))
    chk("camera FOV needed <= 60 deg", fov <= 60.0, "{:.1f} deg".format(fov))
    L["required_fov_deg"] = round(fov, 1)

    L["checks"] = checks
    L["all_ok"] = all(c["ok"] for c in checks)
    return L


# ==========================================================================
# Fusion helpers
# ==========================================================================

def val(v):
    return adsk.core.ValueInput.createByReal(cm(v))


def new_component(root, name):
    occ = root.occurrences.addNewComponent(adsk.core.Matrix3D.create())
    occ.component.name = name
    return occ


def offset_plane(comp, base_plane, dist, axis):
    """Offset plane that lands on the correct side regardless of normal sign."""
    planes = comp.constructionPlanes
    pin = planes.createInput()
    pin.setByOffset(base_plane, val(dist))
    pl = planes.add(pin)
    if abs(getattr(pl.geometry.origin, axis) - cm(dist)) > 1e-6:
        pl.deleteMe()
        pin = planes.createInput()
        pin.setByOffset(base_plane, val(-dist))
        pl = planes.add(pin)
    return pl


def sketch_at_z(comp, z):
    return comp.sketches.add(offset_plane(comp, comp.xYConstructionPlane, z, "z"))


def sketch_at_y(comp, y):
    return comp.sketches.add(offset_plane(comp, comp.xZConstructionPlane, y, "y"))


def sketch_at_x(comp, x):
    return comp.sketches.add(offset_plane(comp, comp.yZConstructionPlane, x, "x"))


def spt(sk, x, y, z=0.0):
    """World mm -> sketch-space point. Removes all plane-orientation guessing."""
    return sk.modelToSketchSpace(
        adsk.core.Point3D.create(cm(x), cm(y), cm(z))
    )


def add_circle(sk, x, y, z, dia):
    return sk.sketchCurves.sketchCircles.addByCenterRadius(
        spt(sk, x, y, z), cm(dia / 2.0)
    )


def add_rect(sk, cx, cy, cz, w, h, plane="xy"):
    """Axis-aligned rectangle given by world centre and world-plane extents."""
    if plane == "xy":
        a = (cx - w / 2.0, cy - h / 2.0, cz)
        b = (cx + w / 2.0, cy + h / 2.0, cz)
    elif plane == "xz":
        a = (cx - w / 2.0, cy, cz - h / 2.0)
        b = (cx + w / 2.0, cy, cz + h / 2.0)
    else:  # yz
        a = (cx, cy - w / 2.0, cz - h / 2.0)
        b = (cx, cy + w / 2.0, cz + h / 2.0)
    return sk.sketchCurves.sketchLines.addTwoPointRectangle(
        spt(sk, *a), spt(sk, *b)
    )


def add_polygon(sk, pts):
    """Closed loop through SHARED sketch points - always forms a profile."""
    lines = sk.sketchCurves.sketchLines
    p3 = [spt(sk, *p) for p in pts]
    first = lines.addByTwoPoints(p3[0], p3[1])
    prev = first
    for i in range(2, len(p3)):
        prev = lines.addByTwoPoints(prev.endSketchPoint, p3[i])
    lines.addByTwoPoints(prev.endSketchPoint, first.startSketchPoint)
    return first


def profiles_all(sk):
    col = adsk.core.ObjectCollection.create()
    for p in sk.profiles:
        col.add(p)
    return col


def profile_largest(sk):
    best, best_a = None, -1.0
    for p in sk.profiles:
        a = p.areaProperties(
            adsk.fusion.CalculationAccuracy.LowCalculationAccuracy).area
        if a > best_a:
            best, best_a = p, a
    return best


def profile_annulus(sk):
    """The profile with an outer loop and a hole - i.e. two loops."""
    for p in sk.profiles:
        if p.profileLoops.count == 2:
            return p
    return profile_largest(sk)


def extrude(comp, prof, dist, op, participants=None, symmetric=False):
    ef = comp.features.extrudeFeatures
    inp = ef.createInput(prof, op)
    if symmetric:
        inp.setSymmetricExtent(val(abs(dist)), True)
    else:
        inp.setDistanceExtent(False, val(dist))
    if participants:
        # participantBodies takes a plain Python list of BRepBody.
        # An ObjectCollection here raises a SWIG TypeError.
        inp.participantBodies = list(participants)
    return ef.add(inp)


NEW = adsk.fusion.FeatureOperations.NewBodyFeatureOperation
CUT = adsk.fusion.FeatureOperations.CutFeatureOperation
JOIN = adsk.fusion.FeatureOperations.JoinFeatureOperation


def vertical_edges(body):
    col = adsk.core.ObjectCollection.create()
    for e in body.edges:
        g = e.geometry
        if g.objectType == adsk.core.Line3D.classType():
            d = g.startPoint.vectorTo(g.endPoint)
            d.normalize()
            if abs(abs(d.z) - 1.0) < 1e-6:
                col.add(e)
    return col


def fillet_vertical(comp, body, radius):
    col = vertical_edges(body)
    if col.count == 0:
        return None
    ff = comp.features.filletFeatures
    inp = ff.createInput()
    inp.isRollingBallCorner = True
    inp.addConstantRadiusEdgeSet(col, val(radius), True)
    return ff.add(inp)


def face_by_normal(body, nx, ny, nz, pick="max"):
    best, best_v = None, None
    for f in body.faces:
        if f.geometry.objectType != adsk.core.Plane.classType():
            continue
        n = f.geometry.normal
        if abs(n.x - nx) > 1e-6 or abs(n.y - ny) > 1e-6 or abs(n.z - nz) > 1e-6:
            continue
        p = f.pointOnFace
        v = p.x * nx + p.y * ny + p.z * nz
        if best_v is None or (v > best_v if pick == "max" else v < best_v):
            best, best_v = f, v
    return best


def shell_body(comp, body, remove_face, thickness):
    col = adsk.core.ObjectCollection.create()
    col.add(remove_face)
    sf = comp.features.shellFeatures
    inp = sf.createInput(col, False)
    inp.insideThickness = val(thickness)
    return sf.add(inp)


def circular_holes(comp, sk_z, body, positions, dia, depth):
    """Cut a set of vertical holes. positions = [(x, y), ...] in world mm."""
    sk = sketch_at_z(comp, sk_z)
    for (x, y) in positions:
        add_circle(sk, x, y, sk_z, dia)
    return extrude(comp, profiles_all(sk), depth, CUT, participants=[body])


def post_foot(L):
    """Gantry foot flange + its bolt pattern.

    The flange cannot be centred on the post: the post sits 13 mm from the
    rear edge of the base, so a symmetric 44 mm flange would hang 9 mm off
    the back. It extends forward instead, toward the tray centre - harmless,
    because the flange lives at z=50 and the tray starts at z=72.
    """
    depth = 44.0
    y_back = L["post_y_outer"]                 # flush with the post's rear face
    cy = y_back + depth / 2.0
    holes = [(x, cy + dy) for x in (-22.0, 22.0) for dy in (-13.0, 13.0)]
    return {"w": 62.0, "depth": depth, "cy": cy, "holes": holes}


def bolt_circle(pcd, n, phase_deg=0.0, cx=0.0, cy=0.0):
    r = pcd / 2.0
    out = []
    for i in range(n):
        a = math.radians(phase_deg + i * 360.0 / n)
        out.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return out


# ==========================================================================
# Printed parts
# ==========================================================================

# ==========================================================================
# Splitting large parts for the print bed
# ==========================================================================

def _trim_half(c, body, axis, side):
    """Cut away everything on the far side of the seam plane."""
    z0 = -1.0
    big = 400.0
    sk = sketch_at_z(c, z0)
    if axis == "y":
        add_rect(sk, 0.0, -side * big / 4.0, z0, big, big / 2.0)
    else:
        add_rect(sk, -side * big / 4.0, 0.0, z0, big / 2.0, big)
    extrude(c, profile_largest(sk), 340.0, CUT, participants=[body])


def _join_pads(c, body, axis, centres, pad, z0):
    """Blocks straddling the seam - the 3 mm wall alone has no meat for a
    dowel, so the joint needs local thickening BEFORE the part is cut."""
    w, d, h = pad
    sk = sketch_at_z(c, z0)
    for (x, y) in centres:
        add_rect(sk, x, y, z0, w, d)
    extrude(c, profiles_all(sk), h, JOIN, participants=[body])


def _join_holes(c, body, axis, centres, side, z_dowel, z_screw):
    """Dowel + screw through the seam.

    Cut AFTER trimming: the far half is already gone, so a symmetric extrude
    only reaches into the half being kept. That sidesteps having to know the
    sign of the offset plane's normal, which differs between axes.
    """
    screw_d = SPLIT["screw_clear"] if side > 0 else SPLIT["screw_pilot"]
    screw_len = SPLIT["screw_depth"] if side > 0 else SPLIT["dowel_depth"]

    sk = sketch_at_y(c, 0.0) if axis == "y" else sketch_at_x(c, 0.0)
    for (x, y) in centres:
        if axis == "y":
            add_circle(sk, x, 0.0, z_dowel, SPLIT["dowel_dia"])
        else:
            add_circle(sk, 0.0, y, z_dowel, SPLIT["dowel_dia"])
    extrude(c, profiles_all(sk), 2.0 * SPLIT["dowel_depth"], CUT,
            participants=[body], symmetric=True)

    sk2 = sketch_at_y(c, 0.0) if axis == "y" else sketch_at_x(c, 0.0)
    for (x, y) in centres:
        if axis == "y":
            add_circle(sk2, x, 0.0, z_screw, screw_d)
        else:
            add_circle(sk2, 0.0, y, z_screw, screw_d)
    extrude(c, profiles_all(sk2), 2.0 * screw_len, CUT,
            participants=[body], symmetric=True)


def _half_suffix(side):
    return "A" if side > 0 else "B"


def build_base_shell(root, L, side=None):
    name = "01_Base_Shell" if side is None else "01_Base_Shell_" + _half_suffix(side)
    occ = new_component(root, name)
    c = occ.component

    sk = sketch_at_z(c, 0.0)
    add_rect(sk, 0, 0, 0, B["base_size"], B["base_size"])
    ext = extrude(c, profile_largest(sk), L["deck_bot"], NEW)
    body = ext.bodies.item(0)
    body.name = "base_shell"

    fillet_vertical(c, body, B["base_fillet"])

    top = face_by_normal(body, 0, 0, 1, "max")
    shell_body(c, body, top, B["base_wall"])

    half = B["base_size"] / 2.0

    # --- front face: OLED window, knob, button -------------------------
    # All X positions must stay inside the flat part of the face, i.e. within
    # +/-(half - base_fillet); beyond that the wall has started to curve away.
    skf = sketch_at_y(c, half)
    add_rect(skf, 0, half, 24.0, B["display_w"], B["display_h"], plane="xz")
    add_circle(skf, B["knob_x"], half, 24.0, B["knob_dia"])
    add_circle(skf, B["button_x"], half, 24.0, B["button_dia"])
    extrude(c, profiles_all(skf), 40.0, CUT,
            participants=[body], symmetric=True)

    # --- rear face: 12 V barrel jack + USB access ----------------------
    skr = sketch_at_y(c, -half)
    add_circle(skr, B["jack_x"], -half, 22.0, POWER["jack_thread_d"] + 0.4)
    add_rect(skr, B["usb_window_x"], -half, 22.0, 34.0, 14.0, plane="xz")
    extrude(c, profiles_all(skr), 40.0, CUT,
            participants=[body], symmetric=True)

    # --- side vents ----------------------------------------------------
    for sx in (1, -1):
        skv = sketch_at_x(c, sx * half)
        n = B["vent_slot_n"]
        for i in range(n):
            y = -((n - 1) / 2.0) * 7.0 + i * 7.0
            add_rect(skv, sx * half, y, 22.0, B["vent_slot_w"], 20.0, plane="yz")
        extrude(c, profiles_all(skv), 40.0, CUT,
                participants=[body], symmetric=True)

    # --- standoff bosses on the floor for UNO Q / A4988 / buck ---------
    bosses = _electronics_positions()
    skb = sketch_at_z(c, L["floor_top"])
    for (x, y) in bosses:
        add_circle(skb, x, y, L["floor_top"], B["screw_boss_d"])
    extrude(c, profiles_all(skb), 5.0, JOIN, participants=[body])

    skp = sketch_at_z(c, L["floor_top"] + 5.0)
    for (x, y) in bosses:
        add_circle(skp, x, y, L["floor_top"] + 5.0, B["screw_boss_id"])
    extrude(c, profiles_all(skp), -6.0, CUT, participants=[body])

    # --- deck tie bosses -------------------------------------------------
    # These carry the screws that bolt the deck down. With the parts split
    # they do structural work too: the deck spans the base's seam, so these
    # eight screws are what stop the two halves hinging apart.
    skt = sketch_at_z(c, L["floor_top"])
    for (x, y) in SPLIT["tie_positions"]:
        add_circle(skt, x, y, L["floor_top"], SPLIT["tie_boss_d"])
    extrude(c, profiles_all(skt), L["deck_bot"] - L["floor_top"], JOIN,
            participants=[body])

    circular_holes(c, L["deck_bot"], body, list(SPLIT["tie_positions"]),
                   SPLIT["tie_pilot"], -14.0)

    # --- split ------------------------------------------------------------
    if side is not None:
        pads = [(x, 0.0) for x in SPLIT["base_pad_x"]]
        _join_pads(c, body, "y", pads, SPLIT["base_pad"], L["floor_top"])
        _trim_half(c, body, "y", side)
        _join_holes(c, body, "y", pads, side,
                    L["floor_top"] + 6.0, L["floor_top"] + 14.0)

    return occ


def _unoq_origin():
    """World XY of the UNO Q board's hole-pattern origin corner."""
    cx, cy = B["unoq_xy"]
    return cx - UNOQ["pcb_l"] / 2.0, cy - UNOQ["pcb_w"] / 2.0


def _electronics_positions():
    ox, oy = _unoq_origin()
    pos = [(ox + hx, oy + hy) for (hx, hy) in UNOQ["holes"]]
    # A4988 carrier and buck converter, two screws each.
    ax, ay = B["a4988_xy"]
    bx, by = B["buck_xy"]
    pos += [(ax, ay + 7.0), (ax, ay - 7.0)]
    pos += [(bx, by + 7.0), (bx, by - 7.0)]
    return pos


def build_deck(root, L, side=None):
    name = "02_Deck" if side is None else "02_Deck_" + _half_suffix(side)
    occ = new_component(root, name)
    c = occ.component

    sk = sketch_at_z(c, L["deck_bot"])
    add_rect(sk, 0, 0, L["deck_bot"], B["base_size"], B["base_size"])
    ext = extrude(c, profile_largest(sk), B["deck_t"], NEW)
    body = ext.bodies.item(0)
    body.name = "deck"
    fillet_vertical(c, body, B["base_fillet"])

    z = L["deck_top"]
    depth = -(B["deck_t"] + 1.0)

    # Central shaft + motor-boss clearance
    circular_holes(c, z, body, [(0.0, 0.0)], B["deck_shaft_hole"], depth)

    # NEMA 17 mounting pattern
    p = MOTOR["hole_pitch"] / 2.0
    motor_holes = [(p, p), (p, -p), (-p, p), (-p, -p)]
    circular_holes(c, z, body, motor_holes, MOTOR["hole_dia"] + 0.4, depth)

    # Pedestal screws. Phased 45 deg: at 0 deg two of the four land exactly
    # on x=0, which is where the deck is now cut in half.
    circular_holes(c, z, body, bolt_circle(120.0, 4, 45.0), 3.4, depth)

    # Gantry post mounting flange
    circular_holes(c, z, body, post_foot(L)["holes"], 3.4, depth)

    # Screws down into the base shell's tie bosses
    circular_holes(c, z, body, list(SPLIT["tie_positions"]),
                   SPLIT["tie_clear"], depth)

    if side is not None:
        pads = [(0.0, y) for y in SPLIT["deck_pad_y"]]
        _join_pads(c, body, "x", pads, SPLIT["deck_pad"], SPLIT["deck_pad_z0"])
        _trim_half(c, body, "x", side)
        _join_holes(c, body, "x", pads, side,
                    SPLIT["deck_pad_z0"] + 4.0, SPLIT["deck_pad_z0"] + 11.0)

    return occ


def build_pedestal(root, L):
    occ = new_component(root, "03_Pedestal")
    c = occ.component

    sk = sketch_at_z(c, L["deck_top"])
    add_circle(sk, 0, 0, L["deck_top"], B["pedestal_od"])
    add_circle(sk, 0, 0, L["deck_top"], B["pedestal_id"])
    ext = extrude(c, profile_annulus(sk), L["pedestal_h"], NEW)
    body = ext.bodies.item(0)
    body.name = "pedestal"

    # Screws down into the deck
    circular_holes(c, L["lazy_bot"], body, bolt_circle(120.0, 4, 45.0),
                   3.4, -(L["pedestal_h"] + 1.0))

    # Pilot holes for the lazy-susan plate screws (corner pattern)
    ins = LAZY["plate"] / 2.0 - LAZY["hole_inset"]
    lazy_holes = [(x, y) for x in (-ins, ins) for y in (-ins, ins)]
    circular_holes(c, L["lazy_bot"], body, lazy_holes,
                   B["screw_boss_id"], -8.0)

    return occ


def insert_od():
    """Insert OD = carrier bore minus the running clearance."""
    return CARRIER["od"] - 2.0 * CARRIER["rim_wall"] - INSERT["fit_clear"]


def build_tray_carrier(root, L):
    """Driven platter. Bolts to the coupling hub and never comes off."""
    occ = new_component(root, "04_Tray_Carrier")
    c = occ.component

    z0 = L["carrier_bot"]
    sk = sketch_at_z(c, z0)
    add_circle(sk, 0, 0, z0, CARRIER["od"])
    ext = extrude(c, profile_largest(sk), CARRIER["floor_t"], NEW)
    body = ext.bodies.item(0)
    body.name = "tray_carrier"

    # Upstand rim that captures the insert
    z1 = L["carrier_floor_top"]
    sk2 = sketch_at_z(c, z1)
    add_circle(sk2, 0, 0, z1, CARRIER["od"])
    add_circle(sk2, 0, 0, z1, CARRIER["od"] - 2.0 * CARRIER["rim_wall"])
    extrude(c, profile_annulus(sk2), CARRIER["rim_h"], JOIN, participants=[body])

    # Coupling-hub bolt pattern
    through = -(CARRIER["floor_t"] + 1.0)
    circular_holes(c, z1, body, bolt_circle(HUB["pcd"], HUB["n_holes"], 0.0),
                   HUB["hole_dia"] + 0.3, through)
    circular_holes(c, z1, body, [(0.0, 0.0)], 8.0, through)

    return occ


def build_tray_insert(root, L):
    """The food tray. Lifts straight out of the carrier for washing."""
    occ = new_component(root, "08_Tray_Insert")
    c = occ.component

    od = insert_od()
    r_out = od / 2.0
    hub_r = B["tray_hub_deck_d"] / 2.0
    inner_r = hub_r + B["tray_wall"]
    outer_r = r_out - B["tray_wall"]

    z0 = L["insert_bot"]
    top = L["tray_rim_top"]

    sk = sketch_at_z(c, z0)
    add_circle(sk, 0, 0, z0, od)
    ext = extrude(c, profile_largest(sk), B["tray_rim_h"], NEW)
    body = ext.bodies.item(0)
    body.name = "tray_insert"

    # Annular food moat
    sk2 = sketch_at_z(c, top)
    add_circle(sk2, 0, 0, top, outer_r * 2.0)
    add_circle(sk2, 0, 0, top, inner_r * 2.0)
    extrude(c, profile_annulus(sk2), -(top - L["tray_floor_top"]), CUT,
            participants=[body])

    # Raised centre island (logo lives here, and it stops the five dividers
    # meeting in one overlapping mess at the axis)
    sk3 = sketch_at_z(c, top)
    add_circle(sk3, 0, 0, top, inner_r * 2.0)
    extrude(c, profile_largest(sk3), -(top - (z0 + 6.0)), CUT,
            participants=[body])

    # Radial dividers
    n = B["n_compartments"]
    t = B["divider_t"] / 2.0
    zf = L["tray_floor_top"]
    sk4 = sketch_at_z(c, zf)
    for i in range(n):
        a = math.radians(i * 360.0 / n)
        ca, sa = math.cos(a), math.sin(a)

        def wp(u, v):
            return (u * ca - v * sa, u * sa + v * ca, zf)

        add_polygon(sk4, [wp(hub_r, -t), wp(r_out, -t),
                          wp(r_out, t), wp(hub_r, t)])
    extrude(c, profiles_all(sk4), top - zf, JOIN, participants=[body])

    # Finger scallops front and back so it can actually be lifted out
    sk5 = sketch_at_z(c, top)
    w = INSERT["lift_slot_w"]
    for sy in (1, -1):
        add_rect(sk5, 0.0, sy * r_out, top, w, 2.0 * B["tray_wall"] + 8.0)
    extrude(c, profiles_all(sk5), -INSERT["lift_slot_h"], CUT,
            participants=[body])

    return occ


def build_arm_post(root, L):
    occ = new_component(root, "05_Arm_Post")
    c = occ.component

    z0 = L["deck_top"]
    z1 = L["head_top"]

    sk = sketch_at_z(c, z0)
    add_rect(sk, 0, L["post_y"], z0, B["arm_w"], B["arm_d"])
    ext = extrude(c, profile_largest(sk), z1 - z0, NEW)
    body = ext.bodies.item(0)
    body.name = "arm_post"
    fillet_vertical(c, body, 6.0)

    # Foot flange - offset forward so it stays inside the base footprint
    foot = post_foot(L)
    skf = sketch_at_z(c, z0)
    add_rect(skf, 0, foot["cy"], z0, foot["w"], foot["depth"])
    extrude(c, profile_largest(skf), 5.0, JOIN, participants=[body])

    circular_holes(c, z0 + 5.0, body, foot["holes"], 3.4, -(5.0 + 1.0))

    # Cable channel up the back
    skc = sketch_at_z(c, z0)
    add_rect(skc, 0, L["post_y"] - B["arm_d"] / 2.0 + 4.0, z0, 16.0, 8.0)
    extrude(c, profile_largest(skc), z1 - z0, CUT, participants=[body])

    return occ


def build_arm_head(root, L):
    occ = new_component(root, "06_Arm_Head")
    c = occ.component

    # The head must reach PAST the camera, not stop short of it. Getting this
    # sign wrong left the head spanning y -125..-89 with the camera at -65, so
    # the pocket and the lens aperture were cutting empty space.
    y_back = L["post_y_outer"]
    y_front = L["cam_y"] + CM["plate_h"] / 2.0 + 4.0
    depth = abs(y_front - y_back)
    cy = (y_back + y_front) / 2.0

    sk = sketch_at_z(c, L["head_bot"])
    add_rect(sk, 0, cy, L["head_bot"], B["arm_w"], depth)
    ext = extrude(c, profile_largest(sk), B["arm_head_t"], NEW)
    body = ext.bodies.item(0)
    body.name = "arm_head"
    fillet_vertical(c, body, 6.0)

    # The camera no longer lives inside the head - it hangs below on the tilt
    # bracket - so the old pocket and lens aperture are gone. What the head
    # needs instead is two threaded points per side: the tilt PIVOT, and one
    # LOCK hole that the bracket's chosen detent lines up with.
    half_w = B["arm_w"] / 2.0
    for sx in (1, -1):
        sk = sketch_at_x(c, sx * half_w)
        add_circle(sk, sx * half_w, L["cam_y"], L["pivot_z"], TILT["thread_dia"])
        add_circle(sk, sx * half_w, L["cam_y"] + TILT["lock_r"], L["pivot_z"],
                   TILT["thread_dia"])
        # Symmetric so the cut direction does not depend on the plane's
        # normal sign; 16 mm centred on the face leaves an 8 mm deep pilot.
        extrude(c, profiles_all(sk), 16.0, CUT,
                participants=[body], symmetric=True)

    return occ


def build_cam_bracket(root, L):
    """Tilting camera bracket: flat plate plus two side arms on an M3 pivot.

    Tilt is set by which of five detent holes is bolted to the head's single
    lock point, giving -20 / -10 / 0 / +10 / +20 degrees.
    """
    occ = new_component(root, "07_Camera_Tilt_Bracket")
    c = occ.component

    z = L["plate_bot"]
    sk = sketch_at_z(c, z)
    add_rect(sk, 0, L["cam_y"], z, CM["plate_w"], CM["plate_h"])
    ext = extrude(c, profile_largest(sk), CM["plate_t"], NEW)
    body = ext.bodies.item(0)
    body.name = "camera_tilt_bracket"
    fillet_vertical(c, body, 4.0)

    # Side arms, straddling the arm head with a running clearance
    x_in = B["arm_w"] / 2.0 + TILT["arm_gap"]
    x_out = x_in + TILT["arm_t"]
    y0 = L["cam_y"] - TILT["arm_back"]
    y1 = L["cam_y"] + TILT["arm_front"]
    z0 = L["plate_top"]
    z1 = L["bracket_arm_top"]

    sk2 = sketch_at_z(c, z0)
    for sx in (1, -1):
        add_rect(sk2, sx * (x_in + x_out) / 2.0, (y0 + y1) / 2.0, z0,
                 TILT["arm_t"], y1 - y0)
    extrude(c, profiles_all(sk2), z1 - z0, JOIN, participants=[body])

    # Pivot, plus the five detents on a bolt circle about it
    holes = [(L["cam_y"], L["pivot_z"])]
    for a in TILT["angles_deg"]:
        rad = math.radians(a)
        holes.append((L["cam_y"] + TILT["lock_r"] * math.cos(rad),
                      L["pivot_z"] + TILT["lock_r"] * math.sin(rad)))

    for sx in (1, -1):
        sk3 = sketch_at_x(c, sx * (x_in + x_out) / 2.0)
        for (hy, hz) in holes:
            add_circle(sk3, sx * (x_in + x_out) / 2.0, hy, hz, TILT["pivot_dia"])
        extrude(c, profiles_all(sk3), 2.0 * TILT["arm_t"] + 4.0, CUT,
                participants=[body], symmetric=True)

    # Lens aperture
    circular_holes(c, L["plate_top"], body, [(0.0, L["cam_y"])],
                   CM["aperture_dia"], -(CM["plate_t"] + 1.0))

    # Slot grid - fits any small module until the real camera is known
    sks = sketch_at_z(c, L["plate_top"])
    half = CM["slot_len"] / 2.0 - CM["slot_w"] / 2.0
    for sx in (-1, 1):
        for sy in (-1, 1):
            x = sx * CM["slot_pitch"]
            y = L["cam_y"] + sy * CM["slot_pitch"]
            add_polygon(sks, [
                (x - CM["slot_w"] / 2.0, y - half, L["plate_top"]),
                (x + CM["slot_w"] / 2.0, y - half, L["plate_top"]),
                (x + CM["slot_w"] / 2.0, y + half, L["plate_top"]),
                (x - CM["slot_w"] / 2.0, y + half, L["plate_top"]),
            ])
    extrude(c, profiles_all(sks), -(CM["plate_t"] + 1.0), CUT,
            participants=[body])

    return occ


# ==========================================================================
# Reference bodies for the purchased parts
# (simplified envelopes - for fit checking and the animation, not for print)
# ==========================================================================

def _block(comp, name, cx, cy, z0, w, d, h):
    sk = sketch_at_z(comp, z0)
    add_rect(sk, cx, cy, z0, w, d)
    ext = extrude(comp, profile_largest(sk), h, NEW)
    b = ext.bodies.item(0)
    b.name = name
    return b


def _cyl(comp, name, cx, cy, z0, dia, h):
    sk = sketch_at_z(comp, z0)
    add_circle(sk, cx, cy, z0, dia)
    ext = extrude(comp, profile_largest(sk), h, NEW)
    b = ext.bodies.item(0)
    b.name = name
    return b


def build_ref_motor(root, L):
    occ = new_component(root, "REF_NEMA17_Motor")
    c = occ.component
    body = _block(c, "motor_body", 0, 0, L["motor_bot"],
                  MOTOR["frame"], MOTOR["frame"], MOTOR["body_len"])
    _cyl(c, "motor_boss", 0, 0, L["motor_face"], MOTOR["boss_dia"], MOTOR["boss_h"])
    _cyl(c, "motor_shaft", 0, 0, L["motor_face"], MOTOR["shaft_dia"],
         MOTOR["shaft_len"])
    fillet_vertical(c, body, MOTOR["corner_chamfer"])
    return occ


def build_ref_hub(root, L):
    occ = new_component(root, "REF_Coupling_Hub")
    c = occ.component
    _cyl(c, "hub_boss", 0, 0, L["hub_bot"], HUB["boss_dia"],
         HUB["total_len"] - HUB["flange_t"])
    _cyl(c, "hub_flange", 0, 0, L["hub_flange_bot"], HUB["flange_dia"],
         HUB["flange_t"])
    return occ


def build_ref_lazy(root, L):
    occ = new_component(root, "REF_Lazy_Susan")
    c = occ.component
    _block(c, "lazy_lower", 0, 0, L["lazy_bot"], LAZY["plate"], LAZY["plate"], 2.0)
    _block(c, "lazy_upper", 0, 0, L["lazy_top"] - 2.0,
           LAZY["plate"], LAZY["plate"], 2.0)
    _cyl(c, "lazy_ball_race", 0, 0, L["lazy_bot"] + 2.0,
         LAZY["plate"] - 8.0, LAZY["height"] - 4.0)
    return occ


def build_ref_electronics(root, L):
    occ = new_component(root, "REF_Electronics")
    c = occ.component
    z = L["floor_top"] + 5.0

    ux, uy = B["unoq_xy"]
    ax, ay = B["a4988_xy"]
    bx, by = B["buck_xy"]

    _block(c, "arduino_uno_q", ux, uy, z,
           UNOQ["pcb_l"], UNOQ["pcb_w"], UNOQ["pcb_t"])
    _block(c, "uno_q_component_envelope", ux, uy, z + UNOQ["pcb_t"],
           UNOQ["pcb_l"] - 6.0, UNOQ["pcb_w"] - 6.0, UNOQ["above_max"])

    _block(c, "a4988_driver", ax, ay, z + DRIVER["pin_below"],
           DRIVER["pcb_l"], DRIVER["pcb_w"], DRIVER["pcb_t"])
    _block(c, "a4988_envelope", ax, ay,
           z + DRIVER["pin_below"] + DRIVER["pcb_t"],
           DRIVER["pcb_l"], DRIVER["pcb_w"], DRIVER["body_above"])

    _block(c, "buck_12v_to_5v", bx, by, z,
           POWER["buck_l"], POWER["buck_w"], POWER["buck_h"])

    _cyl(c, "dc_jack_12v", B["jack_x"],
         -B["base_size"] / 2.0 + POWER["jack_body_l"] / 2.0,
         22.0 - POWER["jack_nut_d"] / 2.0, POWER["jack_nut_d"], POWER["jack_nut_d"])
    return occ


def build_ref_camera(root, L):
    occ = new_component(root, "REF_Camera")
    c = occ.component
    _block(c, "camera_module_PLACEHOLDER", 0.0, L["cam_y"], L["plate_top"],
           CAMERA["pcb_w"], CAMERA["pcb_l"], CAMERA["pcb_t"])
    _cyl(c, "camera_lens_PLACEHOLDER", 0.0, L["cam_y"],
         L["plate_bot"], CAMERA["lens_dia"], CM["plate_t"] + CAMERA["body_h"])
    return occ


# ==========================================================================
# Motion - so the tray can be animated
# ==========================================================================

def _circular_edge_at(body, z, want_r, tol=1.5):
    for e in body.edges:
        g = e.geometry
        if g.objectType != adsk.core.Circle3D.classType():
            continue
        c = g.center
        if abs(c.z - cm(z)) > cm(0.5):
            continue
        if abs(c.x) > cm(0.5) or abs(c.y) > cm(0.5):
            continue
        if abs(g.radius - cm(want_r)) < cm(tol):
            return e
    return None


def _planar_face_at_z(body, z, normal_z=1):
    for f in body.faces:
        if f.geometry.objectType != adsk.core.Plane.classType():
            continue
        n = f.geometry.normal
        if abs(n.z - normal_z) > 1e-6:
            continue
        if abs(f.pointOnFace.z - cm(z)) < cm(0.3):
            return f
    return None


def add_rotation_joint(root, carrier_occ, lazy_occ, L):
    """Revolute joint about Z so the tray spins for the explainer animation.

    CRITICAL: joints.createInput(a, b) MOVES occurrence a onto b. Jointing the
    tray's bottom circle (z=72) to the deck's shaft hole (z=50) therefore
    teleports the tray 22 mm down through the base. Both geometries must
    already be coincident.

    The tray's bottom face and the lazy susan's upper plate top face are both
    at z = tray_bot by construction - they are the surfaces that physically
    bolt together - so jointing those two moves nothing.
    """
    tray_body = carrier_occ.component.bRepBodies.item(0)

    lazy_upper = None
    for b in lazy_occ.component.bRepBodies:
        if b.name == "lazy_upper":
            lazy_upper = b
            break
    if lazy_upper is None:
        raise RuntimeError("lazy_upper body not found")

    e_tray = _circular_edge_at(tray_body, L["carrier_bot"], CARRIER["od"] / 2.0)
    f_lazy = _planar_face_at_z(lazy_upper, L["lazy_top"], normal_z=1)
    if e_tray is None or f_lazy is None:
        raise RuntimeError("joint geometry not found (carrier_edge={}, lazy_face={})"
                           .format(e_tray is not None, f_lazy is not None))

    KP = adsk.fusion.JointKeyPointTypes.CenterKeyPoint
    g_tray = adsk.fusion.JointGeometry.createByCurve(e_tray, KP)
    g_lazy = adsk.fusion.JointGeometry.createByPlanarFace(f_lazy, None, KP)

    # Measure the BODY, not the occurrence: occurrence.boundingBox reads as
    # empty here, which made the earlier displacement guard compare 0.0 to 0.0
    # and pass vacuously.
    z_before = round(tray_body.boundingBox.minPoint.z / MM, 2)

    ji = root.joints.createInput(g_tray, g_lazy)
    ji.setAsRevoluteJointMotion(adsk.fusion.JointDirections.ZAxisJointDirection)
    joint = root.joints.add(ji)
    joint.name = "tray_rotation"

    # Verify the joint did not move the carrier. If it did, back it out rather
    # than leave a silently displaced assembly.
    z_after = round(tray_body.boundingBox.minPoint.z / MM, 2)
    if abs(z_after - z_before) > 0.5:
        joint.deleteMe()
        raise RuntimeError(
            "joint displaced the carrier {:.2f} -> {:.2f} mm; joint removed"
            .format(z_before, z_after))

    return {"joint": "tray_rotation", "carrier_z_mm": z_after, "moved": False}


# ==========================================================================
# Entry point
# ==========================================================================

PARTS = [
    ("01_Base_Shell_A", lambda r, L: build_base_shell(r, L, side=+1)),
    ("01_Base_Shell_B", lambda r, L: build_base_shell(r, L, side=-1)),
    ("02_Deck_A", lambda r, L: build_deck(r, L, side=+1)),
    ("02_Deck_B", lambda r, L: build_deck(r, L, side=-1)),
    ("03_Pedestal", build_pedestal),
    ("04_Tray_Carrier", build_tray_carrier),
    ("08_Tray_Insert", build_tray_insert),
    ("05_Arm_Post", build_arm_post),
    ("06_Arm_Head", build_arm_head),
    ("07_Camera_Tilt_Bracket", build_cam_bracket),
    ("REF_NEMA17_Motor", build_ref_motor),
    ("REF_Coupling_Hub", build_ref_hub),
    ("REF_Lazy_Susan", build_ref_lazy),
    ("REF_Electronics", build_ref_electronics),
    ("REF_Camera", build_ref_camera),
]


def run(ctx):
    app = adsk.core.Application.get()
    ui = app.userInterface

    L = compute_layout()
    report = {
        "layout_mm": {k: (round(v, 2) if isinstance(v, float) else v)
                      for k, v in L.items() if k != "checks"},
        "checks": L["checks"],
        "parts": {},
        "tracebacks": {},
    }

    if not L["all_ok"] and not ctx.get("force", False):
        failed = [c for c in L["checks"] if not c["ok"]]
        report["aborted"] = "layout checks failed"
        report["failed_checks"] = failed
        return {"ok": False, "report": report}

    app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
    des = adsk.fusion.Design.cast(app.activeProduct)
    des.designType = adsk.fusion.DesignTypes.ParametricDesignType
    # defaultLengthUnits is read-only; distanceDisplayUnits is the setter.
    des.unitsManager.distanceDisplayUnits = (
        adsk.fusion.DistanceUnits.MillimeterDistanceUnits)
    root = des.rootComponent
    # Root component name follows the document name; only settable once saved.
    try:
        root.name = "FreshIQ"
    except Exception:  # noqa: BLE001
        pass

    occs = {}
    for name, fn in PARTS:
        try:
            occs[name] = fn(root, L)
            report["parts"][name] = "ok"
        except Exception as exc:  # noqa: BLE001
            report["parts"][name] = "FAILED {}: {}".format(type(exc).__name__, exc)
            report["tracebacks"][name] = traceback.format_exc()

    # Motion for the animation
    if "04_Tray_Carrier" in occs and "REF_Lazy_Susan" in occs:
        try:
            info = add_rotation_joint(
                root, occs["04_Tray_Carrier"], occs["REF_Lazy_Susan"], L)
            report["tray_joint"] = "ok - revolute about Z, carrier z={} mm".format(
                info["carrier_z_mm"])
        except Exception as exc:  # noqa: BLE001
            report["tray_joint"] = "FAILED {}: {}".format(type(exc).__name__, exc)
            report["tracebacks"]["tray_joint"] = traceback.format_exc()

    # The insert is only located by gravity and the rim, so it has no joint of
    # its own. Rigid-group it to the carrier or it stays put while the carrier
    # spins underneath it in the animation.
    if "04_Tray_Carrier" in occs and "08_Tray_Insert" in occs:
        try:
            col = adsk.core.ObjectCollection.create()
            col.add(occs["04_Tray_Carrier"])
            col.add(occs["08_Tray_Insert"])
            rg = root.rigidGroups.add(col, False)
            rg.name = "tray_stack"
            report["tray_rigid_group"] = "ok - insert follows the carrier"
        except Exception as exc:  # noqa: BLE001
            report["tray_rigid_group"] = "FAILED {}: {}".format(
                type(exc).__name__, exc)
            report["tracebacks"]["tray_rigid_group"] = traceback.format_exc()

    # Appearances. Folded into build rather than exposed as its own command:
    # the add-in's allowlist is fixed when Fusion starts, so a new command
    # name would be rejected until the add-in is restarted.
    if ctx.get("style", True):
        try:
            from . import style
            sres = style.run({})
            report["style"] = {
                "ok": sres.get("ok"),
                "applied": len((sres.get("report") or {}).get("applied", {})),
                "missing_slots": (sres.get("report") or {}).get("missing_slots"),
            }
        except Exception as exc:  # noqa: BLE001
            report["style"] = "FAILED {}: {}".format(type(exc).__name__, exc)
            report["tracebacks"]["style"] = traceback.format_exc()

    # Measured overall envelope
    try:
        bb = root.boundingBox
        report["overall_envelope_mm"] = {
            "x": round((bb.maxPoint.x - bb.minPoint.x) / MM, 1),
            "y": round((bb.maxPoint.y - bb.minPoint.y) / MM, 1),
            "z": round((bb.maxPoint.z - bb.minPoint.z) / MM, 1),
        }
    except Exception:  # noqa: BLE001
        pass

    app.activeViewport.fit()

    ok = all(v == "ok" for v in report["parts"].values())
    if ctx.get("show_dialog", True):
        ui.messageBox(
            "FreshIQ build {}\n\n{} of {} parts built.\n"
            "See cad/out/last_run.json for details.".format(
                "complete" if ok else "finished WITH ERRORS",
                sum(1 for v in report["parts"].values() if v == "ok"),
                len(PARTS)),
            "FreshIQ")

    return {"ok": ok, "report": report}
