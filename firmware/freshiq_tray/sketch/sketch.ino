/*
 * FreshIQ tray controller — Arduino UNO Q on-board MCU
 * ====================================================
 * Runs on the UNO Q's STM32. Drives a NEMA 17 through an A4988.
 *
 * WHY THIS IS NOT A SERIAL SKETCH
 * -------------------------------
 * On the UNO Q the Linux side and this MCU are on one board, joined by an
 * internal UART. That UART is owned by the `arduino-router` service:
 *
 *     arduino-router --serial-port /dev/ttyHS1 --unix-port /var/run/arduino-router.sock
 *
 * Arduino's documentation states plainly that /dev/ttyHS1 on Linux and Serial1
 * here are reserved for the router and must not be opened directly. So this
 * sketch does not read Serial at all: it registers RPC handlers and the router
 * delivers calls made from Python.
 *
 * PROVIDED FUNCTIONS  (called from hardware/motor.py)
 *
 *     tray_rotate(steps) -> int    turn `steps`, return steps actually done
 *     tray_stop()        -> int    de-energise now, return 0
 *     tray_status()      -> int    1 = busy, 0 = idle
 *
 * provide_safe, not provide: these handlers call digitalWrite, and the docs
 * require provide_safe for anything using standard Arduino APIs — it runs the
 * handler in the main-loop context instead of the RPC background thread.
 *
 * Never call Bridge.call(), Serial.print() or Monitor.print() inside a
 * provided function. Initiating traffic while answering deadlocks the bridge.
 *
 * ---------------------------------------------------------------------------
 * >>> CONFIRM BEFORE FLASHING <<<
 *   1. PIN_STEP / PIN_DIR / PIN_ENABLE match your A4988 wiring.
 *   2. MICROSTEPS matches your MS1/MS2/MS3 jumpers AND "microsteps" in
 *      config/freshiq.json.
 *   3. The A4988 Vref current limit is set. Too high cooks the driver, too low
 *      skips steps under the tray's inertia.
 * ---------------------------------------------------------------------------
 */

#include "Arduino_RouterBridge.h"

// ---- wiring -------------------------------------------------------- VERIFY
const int PIN_STEP   = 2;
const int PIN_DIR    = 3;
const int PIN_ENABLE = 5;    // A4988 /EN is ACTIVE LOW: LOW = motor energised

// ---- geometry ------------------------------------------------------ VERIFY
const int  MICROSTEPS           = 1;    // 1, 2, 4, 8 or 16 - match your jumpers
const long STEPS_PER_REVOLUTION = 200L; // NEMA 17, 1.8 deg/step
const int  COMPARTMENTS         = 4;    // matches the printed tray
const long STEPS_PER_COMPARTMENT =
      (STEPS_PER_REVOLUTION * MICROSTEPS) / COMPARTMENTS;   // = 50 at 1x

// A move is bounded so a bad argument cannot spin the tray indefinitely while
// the main loop is blocked inside the handler.
const long MAX_STEPS_PER_CALL = STEPS_PER_REVOLUTION * MICROSTEPS;

// ---- motion profile -------------------------------------------------------
// A 200 mm platter loaded with fruit has real inertia. Starting at full speed
// makes an A4988 stall and lose steps, and lost steps are invisible to the
// software because this rig has no encoder. So ramp up and down.
const unsigned int PULSE_US       = 3;    // STEP high time
const unsigned int START_DELAY_US = 1600; // slowest (start/end of a move)
const unsigned int MIN_DELAY_US   = 500;  // fastest (cruise)
const unsigned int RAMP_STEPS     = 24;   // steps spent accelerating

const bool DIRECTION = true;  // flip if the tray turns the wrong way

volatile bool busy    = false;
volatile bool abortNow = false;

void enableDriver()  { digitalWrite(PIN_ENABLE, LOW);  }
void disableDriver() { digitalWrite(PIN_ENABLE, HIGH); }

unsigned int delayForStep(long i, long total) {
  // Linear ramp up, cruise, linear ramp down.
  long fromEnd = total - i - 1;
  long edge = (i < fromEnd) ? i : fromEnd;
  if (edge >= (long)RAMP_STEPS) return MIN_DELAY_US;
  long span = START_DELAY_US - MIN_DELAY_US;
  return (unsigned int)(START_DELAY_US - (span * edge) / RAMP_STEPS);
}

// ---- RPC handlers ---------------------------------------------------------

int tray_rotate(int steps) {
  if (steps <= 0) return 0;
  if (steps > MAX_STEPS_PER_CALL) steps = MAX_STEPS_PER_CALL;

  busy = true;
  abortNow = false;
  enableDriver();
  digitalWrite(PIN_DIR, DIRECTION ? HIGH : LOW);
  delayMicroseconds(5);            // DIR setup time before the first STEP

  long done = 0;
  for (long i = 0; i < steps; i++) {
    if (abortNow) break;
    digitalWrite(PIN_STEP, HIGH);
    delayMicroseconds(PULSE_US);
    digitalWrite(PIN_STEP, LOW);
    delayMicroseconds(delayForStep(i, steps));
    done++;
  }

  // Hold briefly so the platter settles against the energised coil's detent,
  // then release so the driver is not left heating up between compartments.
  delay(120);
  disableDriver();
  busy = false;
  return (int)done;                // the reply Python waits on
}

int tray_stop() {
  abortNow = true;
  disableDriver();
  busy = false;
  return 0;
}

int tray_status() {
  return busy ? 1 : 0;
}

// ---- lifecycle ------------------------------------------------------------

void setup() {
  pinMode(PIN_STEP, OUTPUT);
  pinMode(PIN_DIR, OUTPUT);
  pinMode(PIN_ENABLE, OUTPUT);

  digitalWrite(PIN_STEP, LOW);
  digitalWrite(PIN_DIR, DIRECTION ? HIGH : LOW);
  disableDriver();                 // start de-energised and cool

  Bridge.begin();
  Bridge.provide_safe("tray_rotate", tray_rotate);
  Bridge.provide_safe("tray_stop",   tray_stop);
  Bridge.provide_safe("tray_status", tray_status);
}

void loop() {
  // Deliberately empty. provide_safe handlers are serviced from the main loop
  // by the bridge, so anything added here delays tray moves.
}
