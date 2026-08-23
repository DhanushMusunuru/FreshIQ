/*
 * FreshIQ tray controller
 * =======================
 * Runs on the UNO Q's microcontroller. Drives a NEMA 17 through an A4988.
 *
 * The Linux side decides WHICH compartment to inspect; this sketch only knows
 * how to turn the tray by a given number of steps and how to say when it has
 * finished. Real-time stepping belongs here because Linux is not a real-time
 * OS - a scheduler hiccup in a Python step loop means missed steps and a tray
 * that silently drifts out of alignment.
 *
 * ---------------------------------------------------------------------------
 * >>> CONFIRM THESE THREE THINGS BEFORE FLASHING <<<
 *
 *   1. PIN_STEP / PIN_DIR / PIN_ENABLE below match how you actually wired the
 *      A4988. No wiring diagram existed for this build, so these are the
 *      conventional choices, NOT measured facts.
 *   2. MICROSTEPS matches your MS1/MS2/MS3 jumpers (all open = 1 = full step).
 *      It must also match "microsteps" in config/freshiq.json.
 *   3. The A4988 current limit (Vref trimpot) is set for your motor. Too high
 *      cooks the driver, too low skips steps under the tray's inertia.
 * ---------------------------------------------------------------------------
 *
 * PROTOCOL (matches hardware/motor.py)
 *
 *   R<steps>\n   rotate that many steps, then reply "OK"
 *   R\n          rotate STEPS_PER_COMPARTMENT, then reply "OK"
 *   S\n          stop immediately, disable the driver, reply "OK"
 *   ?\n          reply "READY" when idle, "BUSY" while moving
 *
 *   Anything unrecognised gets "ERR unknown command".
 *
 * The reply is sent only AFTER motion has finished. That is the whole point:
 * the Python side waits for it instead of guessing with a fixed sleep, so the
 * camera can never photograph a moving tray.
 */

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

// ---- motion profile -------------------------------------------------------
// A 200 mm platter loaded with fruit has real inertia. Starting at full speed
// makes an A4988 stall and lose steps, and lost steps are invisible to the
// software because this rig has no encoder. So ramp up and down.
const unsigned int PULSE_US      = 3;     // STEP high time
const unsigned int START_DELAY_US = 1600; // slowest (start/end of a move)
const unsigned int MIN_DELAY_US   = 500;  // fastest (cruise)
const unsigned int RAMP_STEPS     = 24;   // steps spent accelerating

const bool DIRECTION = true;  // flip if the tray turns the wrong way

volatile bool busy = false;

void setup() {
  pinMode(PIN_STEP, OUTPUT);
  pinMode(PIN_DIR, OUTPUT);
  pinMode(PIN_ENABLE, OUTPUT);

  digitalWrite(PIN_STEP, LOW);
  digitalWrite(PIN_DIR, DIRECTION ? HIGH : LOW);
  disableDriver();              // start de-energised and cool

  Serial.begin(115200);
  while (!Serial) { ; }
  Serial.println("FreshIQ tray ready");
}

void enableDriver()  { digitalWrite(PIN_ENABLE, LOW);  }
void disableDriver() { digitalWrite(PIN_ENABLE, HIGH); }

unsigned int delayForStep(long i, long total) {
  // Linear ramp up, cruise, linear ramp down.
  long fromEnd = total - i - 1;
  long edge = (i < fromEnd) ? i : fromEnd;
  if (edge >= RAMP_STEPS) return MIN_DELAY_US;
  long span = START_DELAY_US - MIN_DELAY_US;
  return (unsigned int)(START_DELAY_US - (span * edge) / RAMP_STEPS);
}

void rotateSteps(long steps) {
  if (steps <= 0) return;
  busy = true;
  enableDriver();
  digitalWrite(PIN_DIR, DIRECTION ? HIGH : LOW);
  delayMicroseconds(5);                  // DIR setup time before the first STEP

  for (long i = 0; i < steps; i++) {
    // An 'S' arriving mid-move aborts it. Checked every step so a stop is
    // acted on in milliseconds, not at the end of the move.
    if (Serial.available() && Serial.peek() == 'S') {
      while (Serial.available()) Serial.read();
      break;
    }
    digitalWrite(PIN_STEP, HIGH);
    delayMicroseconds(PULSE_US);
    digitalWrite(PIN_STEP, LOW);
    delayMicroseconds(delayForStep(i, steps));
  }

  // Hold position briefly so the platter settles against the detent of the
  // energised coil, then release so the driver does not sit there heating up.
  delay(120);
  disableDriver();
  busy = false;
}

void handleLine(String line) {
  line.trim();
  if (line.length() == 0) return;

  char cmd = line.charAt(0);

  if (cmd == 'R') {
    long steps = STEPS_PER_COMPARTMENT;
    if (line.length() > 1) {
      long parsed = line.substring(1).toInt();
      if (parsed > 0) {
        steps = parsed;
      } else {
        Serial.println("ERR bad step count");
        return;
      }
    }
    rotateSteps(steps);
    Serial.println("OK");            // sent only once motion has stopped
  }
  else if (cmd == 'S') {
    disableDriver();
    busy = false;
    Serial.println("OK");
  }
  else if (cmd == '?') {
    Serial.println(busy ? "BUSY" : "READY");
  }
  else {
    Serial.println("ERR unknown command");
  }
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    handleLine(line);
  }
}
