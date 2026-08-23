#!/usr/bin/env python
"""
Generate PARTS.md from the real measured geometry.

Reads the export reply (per-part bounding box, volume, manifold state, bed fit)
and the inspect reply (component bounding boxes), and writes a printing sheet.

    python cad/partlist.py --export out/t_export.json --inspect out/t_inspect.json

Numbers come from the B-Rep Fusion actually built, not from the design intent,
so if a feature silently failed the table shows it.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# What each part is and how to print it. Keyed by component name.
NOTES = {
    "01_Base_Shell_A": ("Enclosure, FRONT half (OLED / knob face). Seam at Y=0.",
                        "Open side up. Seam face down is fine too."),
    "01_Base_Shell_B": ("Enclosure, REAR half (DC jack / USB face). Seam at Y=0.",
                        "Open side up. Seam face down is fine too."),
    "02_Deck_A": ("Top plate, RIGHT half. Seam at X=0, crossing the base seam.",
                  "Plate down, seam pad up. 4 walls for stiffness."),
    "02_Deck_B": ("Top plate, LEFT half. Seam at X=0, crossing the base seam.",
                  "Plate down, seam pad up. 4 walls for stiffness."),
    "03_Pedestal": ("Ring that lifts the bearing so its top plate meets the hub flange.",
                    "Flat, as modelled."),
    "04_Tray_Carrier": ("Driven platter. Bolts to the coupling hub, stays on the machine.",
                        "Flat, rim up."),
    "08_Tray_Insert": ("The food tray. Lifts out for washing.",
                       "Flat, compartments up. Food-safe filament if it touches food."),
    "05_Arm_Post": ("Camera gantry. Cable channel runs up the back.",
                    "TALLEST PART. Print upright, or on its back to avoid a long thin tower."),
    "06_Arm_Head": ("Horizontal beam carrying the camera tilt pivot.",
                    "On its side so the pivot holes print round."),
    "07_Camera_Tilt_Bracket": ("Tilting camera mount, 5 detents at -20/-10/0/+10/+20 deg.",
                               "Plate down, arms up. REPRINT THIS when the camera is known."),
}


def load(path):
    if not os.path.isabs(path):
        path = os.path.join(HERE, path)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="out/t_export.json")
    ap.add_argument("--inspect", default="out/t_inspect.json")
    ap.add_argument("--out", default="PARTS.md")
    a = ap.parse_args()

    exp = load(a.export)
    er = exp.get("report") or {}
    parts = er.get("parts") or []
    if not parts:
        sys.exit("no parts in export reply: {}".format(exp.get("error")))

    try:
        ins = load(a.inspect).get("report") or {}
    except Exception:  # noqa: BLE001
        ins = {}

    lines = []
    lines.append("# FreshIQ — printed parts\n")
    lines.append("Measured from the built geometry, not the design intent. "
                 "Bed assumed {} mm.\n".format(er.get("bed_mm")))
    env = ins.get("overall_envelope_mm")
    if env:
        lines.append("Assembled envelope: **{} × {} × {} mm**.\n".format(*env))

    lines.append("\n## Dimensions\n")
    lines.append("| # | Part | X × Y × Z (mm) | Volume (cm³) | Solid | Fits bed |")
    lines.append("|---|---|---|---|---|---|")
    for p in sorted(parts, key=lambda q: q["part"]):
        x, y, z = p["size_mm"]
        lines.append("| {} | `{}` | {:.1f} × {:.1f} × {:.1f} | {:.1f} | {} | {} |".format(
            p["part"].split("_")[0], p["part"], x, y, z,
            p["volume_mm3"] / 1000.0,
            "yes" if p["closed_manifold"] and p["bodies"] == 1 else "NO",
            "yes" if p["fits_bed"] else "NO"))

    lines.append("\n## Material\n")
    lines.append("Volumes above are solid geometry. The grams below assume "
                 "**100 % infill** and are therefore an upper bound — a normal "
                 "15–20 % infill print uses far less.\n")
    lines.append("| Part | PLA @100 % (g) | ~PLA @20 % (g) |")
    lines.append("|---|---|---|")
    tot100 = tot20 = 0.0
    for p in sorted(parts, key=lambda q: q["part"]):
        g100 = p["pla_grams"]
        # rough: perimeters+top/bottom dominate, ~35 % of solid at 20 % infill
        g20 = g100 * 0.35
        tot100 += g100
        tot20 += g20
        lines.append("| `{}` | {:.0f} | ~{:.0f} |".format(p["part"], g100, g20))
    lines.append("| **Total** | **{:.0f}** | **~{:.0f}** |".format(tot100, tot20))

    lines.append("\n## Printing notes\n")
    for p in sorted(parts, key=lambda q: q["part"]):
        what, how = NOTES.get(p["part"], ("", ""))
        lines.append("**`{}`** — {}  \n{}  \nOrientation: {}\n".format(
            p["part"], what, how, p.get("orientation") or "n/a"))

    lines.append("\n## Files\n")
    lines.append("STL: `{}`  \nSTEP: `{}`\n".format(
        er.get("stl_dir"), er.get("step")))

    text = "\n".join(lines) + "\n"
    out = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print("written -> {}".format(out))


if __name__ == "__main__":
    main()
