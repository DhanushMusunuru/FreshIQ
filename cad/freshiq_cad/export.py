"""
Export printed parts as STL, the assembly as STEP, and check printability.

An STL that exports cleanly is not the same as a part you can actually print.
This also checks, per part:

  * exactly one solid body  - two loose bodies means a feature failed to join
  * closed / manifold       - open shells slice into garbage
  * fits the print bed      - tried flat, and rotated 90 deg in Z, and on end
  * material estimate       - volume -> grams of PLA

The REF_ components are simplified envelopes of purchased parts. Exporting
those as printable STL would be misleading, so they are skipped by default.
"""

import os

import adsk.core
import adsk.fusion

PRINTED_PREFIXES = ("01_", "02_", "03_", "04_", "05_", "06_", "07_", "08_")
MM = 0.1
PLA_G_PER_CM3 = 1.24


def _export_dir(sub):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(here, "out", "export", sub)
    os.makedirs(d, exist_ok=True)
    return d


def _bodies_size_mm(bodies):
    """Size from BODIES only - component.boundingBox includes sketches."""
    xs, ys, zs = [], [], []
    for b in bodies:
        bb = b.boundingBox
        xs += [bb.minPoint.x, bb.maxPoint.x]
        ys += [bb.minPoint.y, bb.maxPoint.y]
        zs += [bb.minPoint.z, bb.maxPoint.z]
    return [round((max(xs) - min(xs)) / MM, 1),
            round((max(ys) - min(ys)) / MM, 1),
            round((max(zs) - min(zs)) / MM, 1)]


def _fits(size, bed):
    """Does size fit bed, allowing the sensible re-orientations a slicer offers?"""
    x, y, z = size
    candidates = [
        ("flat", (x, y, z)),
        ("rotated 90 in Z", (y, x, z)),
        ("on its side", (x, z, y)),
        ("on its side + 90", (z, x, y)),
        ("on end", (z, y, x)),
        ("on end + 90", (y, z, x)),
    ]
    for label, (a, b, c) in candidates:
        if a <= bed[0] and b <= bed[1] and c <= bed[2]:
            return True, label
    return False, None


def run(ctx):
    app = adsk.core.Application.get()
    des = adsk.fusion.Design.cast(app.activeProduct)
    if des is None:
        return {"ok": False, "error": "no active design - run build first"}

    root = des.rootComponent
    em = des.exportManager
    include_ref = bool(ctx.get("include_ref", False))
    bed = ctx.get("bed", [256.0, 256.0, 256.0])

    level = {
        "low": adsk.fusion.MeshRefinementSettings.MeshRefinementLow,
        "medium": adsk.fusion.MeshRefinementSettings.MeshRefinementMedium,
        "high": adsk.fusion.MeshRefinementSettings.MeshRefinementHigh,
    }.get(str(ctx.get("refinement", "high")).lower(),
          adsk.fusion.MeshRefinementSettings.MeshRefinementHigh)

    stl_dir = _export_dir("stl")
    parts = []
    problems = []
    errors = {}

    for occ in root.occurrences:
        name = occ.component.name
        if not name.startswith(PRINTED_PREFIXES) and not include_ref:
            continue
        comp = occ.component
        if comp.bRepBodies.count == 0:
            problems.append("{}: no bodies".format(name))
            continue

        bodies = list(comp.bRepBodies)
        vol_mm3 = sum(b.volume for b in bodies) / (MM ** 3)
        size = _bodies_size_mm(bodies)

        solids = [b for b in bodies if b.isSolid]
        closed = all(b.isSolid for b in bodies)
        fits, orientation = _fits(size, bed)

        entry = {
            "part": name,
            "bodies": len(bodies),
            "solid_bodies": len(solids),
            "closed_manifold": closed,
            "size_mm": size,
            "volume_mm3": round(vol_mm3, 1),
            "pla_grams": round(vol_mm3 / 1000.0 * PLA_G_PER_CM3, 1),
            "fits_bed": fits,
            "orientation": orientation,
        }

        if len(bodies) != 1:
            problems.append("{}: {} bodies, expected 1 (a feature did not join)"
                            .format(name, len(bodies)))
        if not closed:
            problems.append("{}: not a closed solid - will not slice".format(name))
        if not fits:
            problems.append("{}: {} mm does not fit a {} mm bed in any orientation"
                            .format(name, size, bed))

        path = os.path.join(stl_dir, name + ".stl")
        try:
            opts = em.createSTLExportOptions(comp, path)
            opts.meshRefinement = level
            opts.isBinaryFormat = True
            em.execute(opts)
            entry["stl"] = path
            entry["stl_kb"] = round(os.path.getsize(path) / 1024.0, 1) \
                if os.path.exists(path) else 0
        except Exception as exc:  # noqa: BLE001
            errors[name] = "{}: {}".format(type(exc).__name__, exc)
            entry["stl"] = None

        parts.append(entry)

    step_path = os.path.join(_export_dir("step"), "FreshIQ_assembly.step")
    step_ok = False
    try:
        em.execute(em.createSTEPExportOptions(step_path, root))
        step_ok = os.path.exists(step_path)
    except Exception as exc:  # noqa: BLE001
        errors["STEP"] = "{}: {}".format(type(exc).__name__, exc)

    total_g = round(sum(p["pla_grams"] for p in parts), 1)

    return {
        "ok": bool(parts) and not errors and not problems,
        "report": {
            "bed_mm": bed,
            "stl_dir": stl_dir,
            "parts": parts,
            "part_count": len(parts),
            "total_pla_grams": total_g,
            "total_pla_kg": round(total_g / 1000.0, 2),
            "step": step_path if step_ok else None,
            "step_kb": round(os.path.getsize(step_path) / 1024.0, 1) if step_ok else 0,
            "printability_problems": problems,
            "errors": errors,
        },
    }
