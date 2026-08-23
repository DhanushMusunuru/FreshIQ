"""
FreshIQ - measured/sourced parameters for the CAD model.

UNITS: every value in this file is MILLIMETRES unless the name says otherwise.
Fusion's API works in centimetres internally; conversion happens in build.py,
never here. Keep this file pure data so it stays readable.

Each entry is tagged:
  [DS]  from a manufacturer datasheet / official drawing  -> trust it
  [STD] an industry standard dimension                    -> trust it
  [VAR] varies by supplier                                -> MEASURE YOURS
  [DES] a free design choice made for this enclosure      -> tune to taste
"""

# --------------------------------------------------------------------------
# 1. NEMA 17 stepper motor  (17HS4401 class)
# --------------------------------------------------------------------------
MOTOR = dict(
    frame            = 42.3,   # [STD] body square across flats
    body_len         = 40.0,   # [DS]  17HS4401 = 40 mm (38/34/48 variants exist)
    corner_chamfer   = 5.0,    # [STD] 45 deg corner cut on the 42 mm frame
    shaft_dia        = 5.0,    # [DS]
    shaft_len        = 24.0,   # [DS]  measured from the front face
    shaft_flat_len   = 20.0,   # [DS]  D-cut length
    shaft_flat_depth = 0.5,    # [STD] D-cut depth (5.0 -> 4.5 across the flat)
    boss_dia         = 22.0,   # [STD] front pilot boss
    boss_h           = 2.0,    # [STD]
    hole_pitch       = 31.0,   # [STD] 31 x 31 mm square pattern
    hole_dia         = 3.0,    # [STD] M3
    hole_depth       = 4.5,    # [STD] blind, tapped
    weight_g         = 300.0,  # [DS]
)

# --------------------------------------------------------------------------
# 2. A4988 stepper driver carrier (Pololu-format module)
# --------------------------------------------------------------------------
DRIVER = dict(
    pcb_w        = 15.24,  # [DS] 0.6 in
    pcb_l        = 20.32,  # [DS] 0.8 in
    pcb_t        = 1.6,    # [STD]
    pin_rows     = 2,      # [DS] 2 x 8 pins, 0.1 in pitch
    pin_pitch    = 2.54,   # [DS]
    row_spacing  = 12.7,   # [DS] 0.5 in between the two header rows
    pin_below    = 3.0,    # [STD] male header tail under the PCB
    body_above   = 11.0,   # [DES] IC + heatsink + trimpot clearance envelope
)

# --------------------------------------------------------------------------
# 3. Arduino UNO Q (ABX00162 / ABX00173) - the SBC
# --------------------------------------------------------------------------
# NOTE: the UNO Q has NO barrel jack. 7-24 V DC IN arrives on the VIN pin of
# the JMEDIA / JANALOG headers. That drives the power plan in section 7.
UNOQ = dict(
    pcb_l        = 68.58,  # [DS] UNO form factor
    pcb_w        = 53.34,  # [DS]
    pcb_t        = 1.6,    # [STD]
    corner_r     = 3.0,    # [STD] UNO outline radius
    hole_dia     = 3.2,    # [DS] 4 x R1.6 mm
    # [DS] canonical UNO hole pattern, origin = board corner nearest USB-C
    holes        = [(13.97, 2.54), (15.24, 50.80), (66.04, 7.62), (66.04, 35.56)],
    below_max    = 2.0,    # [DS] bottom-side parts kept under 2 mm
    above_max    = 16.0,   # [DES] header + USB-C + shielding envelope
    vin_min      = 7.0,    # [DS] V
    vin_max      = 24.0,   # [DS] V
    usb_c_v      = 5.0,    # [DS] 5 V / 3 A only, no higher PD profiles
    usb_c_a      = 3.0,    # [DS]
)

# --------------------------------------------------------------------------
# 4. NEMA17 coupling hub, 5 mm ID (EasyMech "Small")
# --------------------------------------------------------------------------
# This is the part in your reference photo: flanged hub, 6 outer holes,
# 2 grub screws onto the motor shaft.
HUB = dict(
    flange_dia   = 47.0,  # [VAR] supplier-listed OD
    bore_dia     = 5.0,   # [DS]  slip fit on the motor shaft
    pcd          = 35.0,  # [VAR] bolt circle for the 6 outer holes
    n_holes      = 6,     # [VAR] photo shows 6
    hole_dia     = 4.2,   # [VAR] clearance for M4
    total_len    = 20.0,  # [VAR] flange + boss stack height
    flange_t     = 6.0,   # [VAR] <-- MEASURE: drives tray-deck thickness
    boss_dia     = 16.0,  # [VAR] <-- MEASURE
    grub_screw   = "M4",  # [VAR] 2 off, 90 deg apart
)

# --------------------------------------------------------------------------
# 5. Lazy susan turntable bearing, 4 in square
# --------------------------------------------------------------------------
LAZY = dict(
    plate         = 101.6,  # [STD] 4 in square, both plates
    height        = 7.94,   # [STD] 5/16 in assembled stack height
    access_hole   = 57.15,  # [STD] 2.25 in clearance hole in the upper plate
    load_lb       = 300.0,  # [STD]
    screw_size    = "#8",   # [STD] pan head
    # [VAR] hole pattern is NOT standardised across brands.
    # Typical: 4 holes per plate, inset from each corner, rotated 45 deg
    # between plates so they never line up. MEASURE YOURS.
    hole_dia      = 4.5,
    hole_inset    = 10.0,   # from each plate edge
)

# --------------------------------------------------------------------------
# 6. Camera - 1/4 in CMOS module
# --------------------------------------------------------------------------
# [VAR] PENDING CONFIRMATION. Defaults below = OV5647-class 1/4 in module.
# Swap CAMERA_VARIANT once you confirm which board you actually have.
CAMERA_VARIANT = "csi_ov5647"

CAMERA = dict(
    pcb_w        = 25.0,   # [VAR] OV5647 CSI module
    pcb_l        = 24.0,   # [VAR]
    pcb_t        = 1.0,    # [VAR]
    body_h       = 9.0,    # [VAR] incl. lens barrel
    lens_dia     = 8.0,    # [VAR] M8 barrel typical
    hole_dia     = 2.2,    # [VAR] M2
    hole_pitch_x = 21.0,   # [VAR]
    hole_pitch_y = 12.5,   # [VAR]
    sensor_fmt   = "1/4 in",  # [DS] 3.6 x 2.7 mm active area
    sensor_w     = 3.6,
    sensor_h     = 2.7,
)

# --------------------------------------------------------------------------
# 7. Power - 12 V adapter
# --------------------------------------------------------------------------
# 12 V feeds the A4988 motor rail directly and a buck converter for the UNO Q.
# It can NOT go straight into the UNO Q's USB-C.
POWER = dict(
    volts          = 12.0,
    amps           = 3.0,   # [DES] 2 A motor + 1 A logic headroom
    jack_thread_d  = 8.0,   # [STD] panel-mount barrel jack thread
    jack_nut_d     = 11.0,  # [STD]
    jack_body_l    = 20.0,  # [STD] depth needed behind the panel
    plug           = "5.5 x 2.1 mm centre-positive",  # [STD]
    buck_l         = 43.0,  # [STD] MP1584 / LM2596 mini buck module
    buck_w         = 21.0,
    buck_h         = 14.0,
)

# --------------------------------------------------------------------------
# 8. Enclosure + tray - design choices, all [DES], tune freely
# --------------------------------------------------------------------------
# FORM: flat rounded-square base (reference image 3).
# Chosen because the NEMA 17 (42 mm) and the UNO Q (68.6 x 53.3 mm) sit
# side-by-side on one floor instead of stacking, which keeps the whole thing
# low and stable under a 260 mm tray, and leaves a flat front face for the
# OLED and knob.
BUILD = dict(
    # ---- rotating tray ----
    # SIZED FOR A 256 mm PRINT BED, everything in one piece, no seams.
    #   base_half must be >= tray_r + arm_clear + arm_d
    #   125       >= 104   + 4         + 17      -> post flush at the back
    # 4 compartments not 5: 90 deg sectors give 70.2 cm2 of floor each,
    # where 5 x 72 deg on the same tray only managed 47.5 cm2.
    tray_dia         = 208.0,  # rotating tray outer diameter
    tray_wall        = 2.4,    # printed wall thickness
    tray_floor       = 2.0,
    tray_rim_h       = 38.0,   # compartment wall height
    tray_rim_draft   = 6.0,    # deg outward flare, so fruit lifts out easily
    n_compartments   = 4,      # 4 x 90 deg sectors give 70 cm2 each;
                               # 5 x 72 deg only managed 47.5 cm2
    divider_t        = 2.4,
    tray_hub_deck_t  = 6.0,    # thickened centre disc that bolts to the hub
    tray_hub_deck_d  = 40.0,   # centre island on the INSERT only - the hub
                               # bolts are in the carrier, so shrinking this
                               # buys 4.6 cm2 per compartment for free

    # ---- base ----
    base_size        = 250.0,  # rounded-square footprint, across flats
    base_fillet      = 45.0,   # corner radius -> squircle, matches the render
    # Panel features must stay on the FLAT part of each face, i.e. within
    # +/-(base_size/2 - base_fillet) = +/-80 mm of centre.
    # x=70 not 80: the base's seam pad occupies x 108..122, and at x=80
    # the board reached 114.3 and fouled it.
    unoq_xy          = (70.0, 0.0),
    a4988_xy         = (-78.0, 45.0),
    buck_xy          = (-78.0, -45.0),
    jack_x           = 70.0,
    usb_window_x     = -45.0,
    knob_x           = 62.0,
    button_x         = -62.0,
    base_wall        = 3.0,
    base_floor_t     = 3.0,
    deck_t           = 5.0,    # top deck; motor bolts to its underside
    motor_gap        = 2.0,    # floor-to-motor-body clearance
    deck_shaft_hole  = 26.0,   # clearance for shaft + motor boss counterbore
    vent_slot_w      = 3.0,
    vent_slot_n      = 14,

    # ---- lazy-susan pedestal ----
    # Raises the bearing so its TOP plate lands exactly at the hub flange top.
    # Height is COMPUTED in build.compute_layout(), not hard-coded here.
    # OD 130 fully supports the 101.6 mm square bearing plate; ID 100 clears
    # the Ø47 hub and still leaves ring material under the corner screws.
    pedestal_od      = 130.0,
    pedestal_id      = 100.0,

    # ---- camera gantry ----
    # 150 mm above the tray gives a ~52 deg FOV requirement to frame one
    # compartment - comfortably inside any 1/4 in module - while keeping the
    # printed post under 190 mm so it fits a normal print volume.
    arm_h            = 150.0,  # camera lens height above the tray surface
    arm_w            = 34.0,   # post cross-section, width
    arm_d            = 17.0,   # post cross-section, depth
    arm_reach        = 56.1,   # post centre -> compartment centroid (r 60.4)
    arm_clear        = 4.0,    # gap between tray edge and post inner face
    arm_head_t       = 26.0,   # thickness of the camera head block
    cam_tilt_deg     = 0.0,    # 0 = straight down at one compartment

    # ---- front panel UI ----
    display_w        = 30.0,   # 0.96 in OLED window
    display_h        = 15.0,
    knob_dia         = 26.0,   # rotary encoder knob
    button_dia       = 16.0,

    # ---- print / assembly ----
    fit_clearance    = 0.25,   # printed part-to-part slip fit
    screw_boss_d     = 8.0,    # M3 boss outer diameter
    screw_boss_id    = 2.5,    # M3 self-tapping pilot
    nut_pocket_af    = 5.6,    # M3 nut across-flats + fit
)

# --------------------------------------------------------------------------
# 9. Camera mount - generic until the module is identified
# --------------------------------------------------------------------------
# Until the exact camera board is confirmed, the arm carries a flat mount
# PLATE with a slotted grid. Any small module bolts to it, and it is a single
# separate printed part - so swapping in the real footprint later means
# reprinting one 12 g part, not the whole arm.
CAM_MOUNT = dict(
    plate_w      = 44.0,
    plate_h      = 40.0,
    plate_t      = 3.0,
    slot_w       = 2.6,    # M2.5 clearance slots
    slot_len     = 14.0,
    slot_pitch   = 13.0,
    aperture_dia = 14.0,   # lens clear-through hole
)

# --------------------------------------------------------------------------
# 11. Camera tilt
# --------------------------------------------------------------------------
# The bracket hangs from the arm head on an M3 pivot running along X and is
# locked by a second M3 into ONE threaded hole in the head. The bracket
# carries five holes on a bolt circle about the pivot, so the angle is set by
# which hole you line up.
#
# Detents rather than an arc slot on purpose: a 3.4 mm slot through a 3 mm
# printed arm is a fragile layer-delamination line, it creeps under vibration
# once the clamp relaxes, and you cannot return to the same angle twice.
SPLIT = dict(
    # CROSS-LAPPED on purpose. The base splits front/back (Y=0) and the deck
    # splits left/right (X=0), so the deck bolts across the base's seam and
    # the base bolts across the deck's seam. Each part clamps the other's
    # joint; split them both the same way and the whole box hinges.
    #
    # NOTE: a half of a 250 x 250 part is 250 x 125 - the LONG side is
    # unchanged. This buys warp resistance and brim room, not a smaller
    # maximum dimension. Set quarters=True if you need 125 x 125.
    quarters       = False,

    base_pad       = (14.0, 24.0, 20.0),   # X, Y, Z - straddles the Y=0 seam
    base_pad_x     = (115.0, -115.0),      # inside the left and right walls
    deck_pad       = (24.0, 14.0, 15.0),   # X, Y, Z - straddles the X=0 seam
    deck_pad_y     = (60.0, -60.0),        # clear of the motor and electronics
    deck_pad_z0    = 30.0,                 # pad hangs below the deck underside

    dowel_dia      = 4.2,   # Ø4 pin, printed or steel
    dowel_depth    = 10.0,
    screw_clear    = 3.4,   # M3 clearance, on the A half
    screw_pilot    = 2.5,   # M3 self-tap, on the B half
    screw_depth    = 13.0,

    # Deck-to-shell screws. Deliberately off both axes so no boss lands on a
    # seam, and spread so each half of each part gets four.
    tie_positions  = ((95.0, 95.0), (-95.0, 95.0), (95.0, -95.0), (-95.0, -95.0),
                      (110.0, 40.0), (-110.0, 40.0), (110.0, -40.0), (-110.0, -40.0)),
    tie_boss_d     = 9.0,
    tie_pilot      = 2.5,
    tie_clear      = 3.4,
)

TILT = dict(
    angles_deg   = (-20.0, -10.0, 0.0, 10.0, 20.0),
    lock_r       = 18.0,   # detent bolt circle radius about the pivot
    arm_t        = 3.0,    # side arm thickness
    arm_gap      = 0.4,    # per side clearance against the arm head
    arm_rise     = 29.0,   # plate top -> pivot centre
    pivot_inset  = 13.0,   # pivot centre above the arm head underside
    pivot_dia    = 3.2,    # M3 clearance in the bracket
    thread_dia   = 2.5,    # M3 self-tapping pilot in the head
    arm_back     = 12.0,   # arm extent behind the pivot, in Y
    arm_front    = 24.0,   # arm extent in front of the pivot, in Y
    arm_over     = 8.0,    # arm material above the pivot
)

# --------------------------------------------------------------------------
# 10. Rotating tray - split into a driven carrier and a lift-out food insert
# --------------------------------------------------------------------------
# The insert is the part food touches, so it has to come off for washing
# without unbolting anything from the motor. The carrier stays bolted to the
# coupling hub; the insert drops into its rim and is driven by three lugs.
CARRIER = dict(
    od           = 208.0,  # matches tray_dia - carrier and insert are flush
    floor_t      = 6.0,    # thick enough for the M4 hub bolts to seat
    rim_h        = 8.0,    # upstand that locates the insert
    rim_wall     = 3.0,
)
# No drive lugs. They were drawn in and then removed: a lug tall enough to
# grip needs ~5 mm, and the insert's floor over the moat is only 2 mm, so the
# mating slot would punch straight through into the food. The rim locates the
# insert and friction plus the fruit's own weight drives it - at ~10 rpm with
# gentle acceleration there is nothing for a lug to do.

INSERT = dict(
    fit_clear    = 0.5,    # diametral clearance inside the carrier rim
    lug_clear    = 0.4,    # per side, on the drive-lug slots
    lift_slot_w  = 26.0,   # finger scallops in the rim so it lifts out
    lift_slot_h  = 12.0,
)

