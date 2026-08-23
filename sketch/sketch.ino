/*
 * FreshIQ tray controller  -  Arduino UNO Q onboard MCU (STM32U585)
 * =================================================================
 * Drives a NEMA 17 through an A4988. Flashed to the MICROCONTROLLER half of
 * the UNO Q; FreshIQ itself runs on the Linux half of the same board.
 *
 * HOW THE TWO HALVES TALK  (this is what changed, and why)
 * --------------------------------------------------------
 * The UNO Q is ONE board with TWO processors joined by an internal UART. On
 * Linux that UART is /dev/ttyHS1 and it is already owned by the `arduino-router`
 * service - Arduino's documentation says application code must not open it.
 * So there is no /dev/ttyACM0 to talk to and never will be: this MCU is not a
 * USB peripheral.
 *
 * The supported path is RPC through the router. This sketch REGISTERS named
 * functions; Linux CALLS them by name:
 *
 *     Python (hardware/motor.py)          this sketch
 *     ---------------------------         --------------------------------
 *     rpc.call("tray_rotate", 50, 70000)                    turns the tray
 *                              -->  tray_rotate(50, 70000)  50 steps, 70 ms each
 *                              <--  returns 50              move is FINISHED
 *
 * The return value is sent only AFTER motion has stopped. That is the whole
 * point: backend/scan.py waits for it instead of guessing with a fixed sleep,
 * so the camera can never photograph a moving tray.
 *
 * ---------------------------------------------------------------------------
 * >>> CONFIRM THESE THREE THINGS BEFORE FLASHING <<<
 *
 *   1. PIN_STEP / PIN_DIR / PIN_ENABLE below match how you actually wired the
 *      A4988. They agree with docs/schematics/freshiq-wiring.svg: D2 = STEP,
 *      D3 = DIR, D5 = ENABLE.
 *   2. MICROSTEPS matches your MS1/MS2/MS3 jumpers (all open = 1 = full step).
 *      It must also match "microsteps" in config/freshiq.json.
 *   3. The A4988 current limit (Vref trimpot) is set for your motor. Too high
 *      cooks the driver, too low skips steps under the tray's inertia.
 *
 *   Also: A4988 VDD must be +3V3, NOT +5V. The A4988 reads logic HIGH at about
 *   0.7 x VDD; on 5 V that is 3.5 V and the UNO Q's outputs only reach 3.3 V,
 *   so STEP would never be reliably seen.
 * ---------------------------------------------------------------------------
 *
 * FLASHING
 * You do not flash this by hand. It is the MCU half of an Arduino App: both
 * halves are built and deployed together by
 *
 *     arduino-app-cli app start ~/ArduinoApps/FreshIQ
 *
 * or by pressing Run in Arduino App Lab. The build profile (board and library
 * versions) is in sketch.yaml next to this file.
 *
 * RPC METHODS (must match hardware/motor.py BridgeTrayMotor)
 *
 *     tray_rotate(steps, step_delay_us) -> int
 *                                 turn `steps` steps, pausing step_delay_us
 *                                 between each, and return when stopped.
 *                                 The returned value is the number of steps
 *                                 ACTUALLY taken; fewer than asked means the
 *                                 move was aborted, so Linux knows the tray is
 *                                 no longer aligned instead of assuming it is.
 *     tray_stop()        -> int   de-energise the A4988 immediately, return 0
 *     tray_status()      -> int   0 = READY, 1 = BUSY
 */

#include "Arduino_RouterBridge.h"

// ---- wiring -------------------------------------------------------- VERIFY
// Matches docs/schematics/freshiq-wiring.svg and DEPLOYMENT.md.
const int PIN_STEP   = 2;
const int PIN_DIR    = 3;
const int PIN_ENABLE = 5;    // A4988 /EN is ACTIVE LOW: LOW = motor energised

// ---- geometry ------------------------------------------------------ VERIFY
const int  MICROSTEPS           = 1;    // 1, 2, 4, 8 or 16 - match your jumpers
const long STEPS_PER_REVOLUTION = 200L; // NEMA 17, 1.8 deg/step
const int  COMPARTMENTS         = 4;    // matches the printed tray
const long STEPS_PER_COMPARTMENT =
      (STEPS_PER_REVOLUTION * MICROSTEPS) / COMPARTMENTS;   // = 50 at 1x

// Refuse absurd requests rather than spinning the tray for a minute because a
// bad value reached the RPC call.
const long MAX_STEPS_PER_CALL = STEPS_PER_REVOLUTION * MICROSTEPS * 2L;

// ---- motion profile -------------------------------------------------------
// SPEED IS SET FROM LINUX, not here: config/freshiq.json carries
// tray.seconds_per_compartment, and hardware/motor.py turns it into a
// per-step delay that arrives as the second argument to tray_rotate(). Tuning
// how fast the tray turns is therefore a config edit, not a re-flash.
//
// A 200 mm platter loaded with fruit has real inertia, so the first and last
// few steps of a move are stretched. At the slow speeds this tray runs the
// ramp barely matters, but it costs nothing and protects a fast setting.
const unsigned int PULSE_US = 3;          // STEP high time

const unsigned long DEFAULT_STEP_DELAY_US = 70000UL;  // 50 steps ~ 3.5 s
const unsigned long MIN_STEP_DELAY_US     = 150UL;    // fastest allowed
const unsigned long MAX_STEP_DELAY_US     = 200000UL; // slowest allowed
const int           RAMP_STEPS_MAX        = 10;

const bool DIRECTION = true;  // flip if the tray turns the wrong way

volatile bool g_busy  = false;
volatile bool g_abort = false;

void enableDriver()  { digitalWrite(PIN_ENABLE, LOW);  }
void disableDriver() { digitalWrite(PIN_ENABLE, HIGH); }

/*
 * Wait between steps.
 *
 * delayMicroseconds() is only accurate below roughly 16 ms - beyond that it
 * silently misbehaves. A tray told to take 3.5 s over 50 steps needs 70 ms per
 * step, which is far outside that range, so anything long is spent in delay()
 * and only the sub-millisecond remainder goes to delayMicroseconds().
 */
void stepSleep(unsigned long us) {
  if (us >= 16000UL) {
    delay(us / 1000UL);
    us %= 1000UL;
  }
  if (us > 0) delayMicroseconds((unsigned int)us);
}

/* Ease in and out of `cruise`, which is whatever Linux asked for. */
unsigned long delayForStep(long i, long total, unsigned long cruise, int ramp) {
  if (ramp <= 0) return cruise;
  long fromEnd = total - i - 1;
  long edge = (i < fromEnd) ? i : fromEnd;
  if (edge >= (long)ramp) return cruise;
  unsigned long extra = cruise * 3UL / 5UL;      // start 60% slower than cruise
  return cruise + extra - (extra * (unsigned long)edge) / (unsigned long)ramp;
}

/*
 * Turn the tray and report how far it actually got.
 *
 * NOTHING IS PRINTED IN HERE. Arduino_RouterBridge documents that calling
 * Bridge.call() or printing from inside an RPC handler can deadlock the
 * MCU-CPU IPC, so this function only touches GPIO.
 */
int tray_rotate(int steps, int step_delay_us) {
  if (steps <= 0) return 0;
  if (steps > MAX_STEPS_PER_CALL) steps = (int)MAX_STEPS_PER_CALL;

  // Clamp rather than trust: a bad value here either stalls the tray for
  // minutes or drives the A4988 faster than it can follow.
  unsigned long cruise = (step_delay_us > 0)
      ? (unsigned long)step_delay_us : DEFAULT_STEP_DELAY_US;
  if (cruise < MIN_STEP_DELAY_US) cruise = MIN_STEP_DELAY_US;
  if (cruise > MAX_STEP_DELAY_US) cruise = MAX_STEP_DELAY_US;

  int ramp = (int)(steps / 5);
  if (ramp > RAMP_STEPS_MAX) ramp = RAMP_STEPS_MAX;

  g_abort = false;
  g_busy = true;

  enableDriver();
  digitalWrite(PIN_DIR, DIRECTION ? HIGH : LOW);
  delayMicroseconds(5);                  // DIR setup time before the first STEP

  long done = 0;
  for (long i = 0; i < steps; i++) {
    if (g_abort) break;                  // tray_stop() came in mid-move
    digitalWrite(PIN_STEP, HIGH);
    delayMicroseconds(PULSE_US);
    digitalWrite(PIN_STEP, LOW);
    stepSleep(delayForStep(i, steps, cruise, ramp));
    done++;
  }

  // Hold position briefly so the platter settles against the detent of the
  // energised coil, then release so the driver does not sit there heating up.
  delay(120);
  disableDriver();

  g_busy = false;
  g_abort = false;
  return (int)done;
}

/*
 * Stop now.
 *
 * De-energising is unconditional and immediate, so this is safe even if a move
 * is in flight. One compartment is only ~50 steps (well under a second), so in
 * the worst case the abort flag is seen almost at once anyway.
 */
int tray_stop() {
  g_abort = true;
  disableDriver();
  return 0;
}

/* 0 = READY, 1 = BUSY. Used by hardware/motor.py's ping(). */
int tray_status() {
  return g_busy ? 1 : 0;
}

void setup() {
  pinMode(PIN_STEP, OUTPUT);
  pinMode(PIN_DIR, OUTPUT);
  pinMode(PIN_ENABLE, OUTPUT);

  digitalWrite(PIN_STEP, LOW);
  digitalWrite(PIN_DIR, DIRECTION ? HIGH : LOW);
  disableDriver();              // start de-energised and cool

  // NO Serial.begin() and NO 'while (!Serial)' here. The link to Linux is the
  // router bridge, not a USB CDC port - and waiting on a Serial that never
  // opens would hang setup() forever, leaving the tray permanently dead.
  Bridge.begin();

  Bridge.provide("tray_rotate", tray_rotate);
  Bridge.provide("tray_stop",   tray_stop);
  Bridge.provide("tray_status", tray_status);
}

void loop() {
  // Handlers registered with Bridge.provide() are served by the library's own
  // thread, so there is nothing to pump here. Yield instead of spinning.
  delay(10);
}
