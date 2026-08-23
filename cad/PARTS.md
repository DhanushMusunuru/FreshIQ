# FreshIQ — printed parts

Measured from the built geometry, not the design intent. Bed assumed [256, 256, 256] mm.

Assembled envelope: **250.0 × 250.0 × 275.0 mm**.


## Dimensions

| # | Part | X × Y × Z (mm) | Volume (cm³) | Solid | Fits bed |
|---|---|---|---|---|---|
| 01 | `01_Base_Shell_A` | 250.0 × 125.0 × 45.0 | 160.1 | yes | yes |
| 01 | `01_Base_Shell_B` | 250.0 × 125.0 × 45.0 | 162.2 | yes | yes |
| 02 | `02_Deck_A` | 125.0 × 250.0 × 20.0 | 154.7 | yes | yes |
| 02 | `02_Deck_B` | 125.0 × 250.0 × 20.0 | 154.8 | yes | yes |
| 03 | `03_Pedestal` | 130.0 × 130.0 × 14.1 | 75.6 | yes | yes |
| 04 | `04_Tray_Carrier` | 208.0 × 208.0 × 14.0 | 218.5 | yes | yes |
| 05 | `05_Arm_Post` | 62.0 × 44.0 × 225.0 | 105.0 | yes | yes |
| 06 | `06_Arm_Head` | 34.0 × 88.6 × 26.0 | 77.4 | yes | yes |
| 07 | `07_Camera_Tilt_Bracket` | 44.0 × 44.0 × 40.0 | 12.1 | yes | yes |
| 08 | `08_Tray_Insert` | 201.5 × 201.5 × 38.0 | 149.4 | yes | yes |

## Material

Volumes above are solid geometry. The grams below assume **100 % infill** and are therefore an upper bound — a normal 15–20 % infill print uses far less.

| Part | PLA @100 % (g) | ~PLA @20 % (g) |
|---|---|---|
| `01_Base_Shell_A` | 199 | ~70 |
| `01_Base_Shell_B` | 201 | ~70 |
| `02_Deck_A` | 192 | ~67 |
| `02_Deck_B` | 192 | ~67 |
| `03_Pedestal` | 94 | ~33 |
| `04_Tray_Carrier` | 271 | ~95 |
| `05_Arm_Post` | 130 | ~46 |
| `06_Arm_Head` | 96 | ~34 |
| `07_Camera_Tilt_Bracket` | 15 | ~5 |
| `08_Tray_Insert` | 185 | ~65 |
| **Total** | **1574** | **~551** |

## Printing notes

**`01_Base_Shell_A`** — Enclosure, FRONT half (OLED / knob face). Seam at Y=0.  
Open side up. Seam face down is fine too.  
Orientation: flat

**`01_Base_Shell_B`** — Enclosure, REAR half (DC jack / USB face). Seam at Y=0.  
Open side up. Seam face down is fine too.  
Orientation: flat

**`02_Deck_A`** — Top plate, RIGHT half. Seam at X=0, crossing the base seam.  
Plate down, seam pad up. 4 walls for stiffness.  
Orientation: flat

**`02_Deck_B`** — Top plate, LEFT half. Seam at X=0, crossing the base seam.  
Plate down, seam pad up. 4 walls for stiffness.  
Orientation: flat

**`03_Pedestal`** — Ring that lifts the bearing so its top plate meets the hub flange.  
Flat, as modelled.  
Orientation: flat

**`04_Tray_Carrier`** — Driven platter. Bolts to the coupling hub, stays on the machine.  
Flat, rim up.  
Orientation: flat

**`05_Arm_Post`** — Camera gantry. Cable channel runs up the back.  
TALLEST PART. Print upright, or on its back to avoid a long thin tower.  
Orientation: flat

**`06_Arm_Head`** — Horizontal beam carrying the camera tilt pivot.  
On its side so the pivot holes print round.  
Orientation: flat

**`07_Camera_Tilt_Bracket`** — Tilting camera mount, 5 detents at -20/-10/0/+10/+20 deg.  
Plate down, arms up. REPRINT THIS when the camera is known.  
Orientation: flat

**`08_Tray_Insert`** — The food tray. Lifts out for washing.  
Flat, compartments up. Food-safe filament if it touches food.  
Orientation: flat


## Files

STL: `D:\Side Quests\Projects\FreshIQ\cad\out\export\stl`  
STEP: `D:\Side Quests\Projects\FreshIQ\cad\out\export\step\FreshIQ_assembly.step`

