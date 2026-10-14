# FreshIQ — print specification

Give the print shop this file plus the 10 STL files. Nothing else is needed.
Send the STEP only if they ask — it is for CAD editing, not for printing.

## What to send

| Send | File | Why |
|---|---|---|
| ✅ | `stl/*.stl` (10 files) | The parts. Universal, every service accepts them. |
| ✅ | this file | Material, settings, tolerances. |
| ✅ | `PARTS.md` | Per-part sizes and orientation notes. |
| ➖ | `step/FreshIQ_assembly.step` | Only if they want to inspect or modify the CAD. |

**Quantity: 1 of each part.**

## Material

| Part | Material | Why |
|---|---|---|
| `08_Tray_Insert` | **PETG** | Touches food, gets washed, more heat- and scratch-resistant than PLA |
| Everything else | PLA or PETG | Structural only, no heat load |

PETG throughout is simplest if they would rather not switch filament.

### Read this before using it with food

An FDM print is **not** food-safe in the regulatory sense, whatever filament is
used. The layer lines are microscopic crevices that trap moisture and bacteria
and survive normal washing, and most filament is not food-grade certified —
brass nozzles can also leave lead traces.

For **whole uncut fruit** the practical risk is low, and this is what the tray
is designed for. If you want to be careful:

- Keep the insert for whole fruit only — never cut fruit, never wet produce left sitting.
- Wash by hand in cool water and dry fully. **Do not put it in a dishwasher** —
  PLA deforms around 60 °C, and PETG will soften on a hot cycle.
- Or drop a food-grade silicone mat or greaseproof liner into each compartment.
- Or seal it with a food-safe epoxy after printing.

That is why the insert is a separate lift-out part: it can be replaced cheaply
(~65 g of filament) when it wears, and it never has to go near the motor.

## Print settings

| Setting | Value | Note |
|---|---|---|
| Layer height | 0.2 mm | 0.15 mm on `07_Camera_Tilt_Bracket` if they want cleaner detents |
| Walls / perimeters | 3 | **4 on `02_Deck`** — it spans 250 mm and carries the motor |
| Top / bottom layers | 5 | |
| Infill | 15–20 % gyroid | |
| Supports | **None needed** | See below |
| Brim | On for `05_Arm_Post` | Tall and narrow, benefits from adhesion |
| Nozzle | 0.4 mm | |

### Supports

No part needs supports in the orientation given in `PARTS.md`. The only
overhangs are the panel cutouts in the base side walls (display window, knob
and button holes, vents, DC jack). Those are horizontal holes in a vertical
wall, so the top of each is a short bridge — up to 26 mm on the knob hole.
Any decent printer bridges that. A little droop at the top of the knob hole is
cosmetic and hidden by the knob.

## Tolerances — please do not "fix" these

**Do not scale the parts.** Several fits are dimensioned:

| Fit | Clearance | Where |
|---|---|---|
| Insert into carrier rim | 0.5 mm diametral | `08` sits inside `04` |
| Tilt bracket arms over arm head | 0.4 mm per side | `07` straddles `06` |
| M3 clearance holes | Ø3.2 / Ø3.4 | pivot, detents, deck mounts |
| M3 self-tap pilots | Ø2.5 | arm head sides, standoff bosses |
| M4 clearance | Ø4.5 | coupling hub bolts in `04` |
| NEMA 17 mount holes | Ø3.4 | `02_Deck` |
| Motor shaft / boss clearance | Ø26 | `02_Deck` centre |

If their slicer offers "hole compensation" or "auto-repair holes", ask them to
leave it off. The clearances already account for typical FDM elephant-foot and
hole shrinkage.

## Hardware you supply separately (not printed)

| Item | Qty | Use |
|---|---|---|
| M3 × 8 mm | 4 | NEMA 17 to deck |
| M3 × 10 mm | 4 | gantry post foot to deck |
| M3 × 12 mm | 4 | pedestal to deck |
| M3 × 16 mm | 2 | camera tilt pivot + lock |
| M3 × 6 mm | 8 | electronics standoffs |
| M4 × 10 mm | 6 | coupling hub to tray carrier |
| #8 × 1/2 in pan head | 8 | lazy susan to pedestal and carrier |
| M2.5 × 6 mm | 4 | camera module to bracket |
| M3 × 10 mm | 8 | deck to base shell tie screws |
| M3 × 20 mm | 4 | across the two seams (2 per seam) |
| Ø4 × 20 mm dowel pin | 4 | seam alignment (2 per seam), steel or printed |

## Assembling the split parts

The base prints as a front half and a rear half (seam at Y=0); the deck prints
as a left half and a right half (seam at X=0). **The seams deliberately cross
each other** — the deck bridges the base's joint and the base bridges the
deck's, so each part clamps the other's seam. Do not re-orient them to make
both seams parallel.

1. Push the Ø4 dowels into one half of each pair, then mate the halves.
2. Drive the M3 × 20 seam screws — the clearance hole is in the `_A` half and
   the self-tapping pilot in the `_B` half, so start from the A side.
3. Sit the assembled deck on the assembled base and drive the eight M3 × 10
   tie screws into the bosses. These do the real structural work.
4. Only then mount the motor, pedestal and gantry.

A 0.1–0.2 mm step at a seam is normal and disappears once the tie screws pull
everything flat. Glue is optional; the screws are sufficient.

## Colour

If they ask: white or light grey for the base, deck, post, head and tray
insert; darker grey for the carrier; black for the tilt bracket. Purely
cosmetic — it matches the reference renders.
