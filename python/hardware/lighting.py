"""
Enclosure lighting.

CONFIGURED HARDWARE: a white LED strip wired directly to the 12 V rail. It is
on whenever the device is powered, and there is no software control.

That is a legitimate design, not a missing feature. The model judges food by
colour, and it was trained on consistently lit photos - the worst thing you can
do to accuracy is vary the light between compartments. A strip that is simply
always on cannot drift.

This module therefore exists to do three things:

  1. report lighting state honestly to the dashboard and to /api/health
  2. give the scan controller a place to call on()/off() that is a no-op today
     but becomes real if the strip is ever moved onto an MCU pin
  3. warn when the configuration claims a capability the hardware lacks

If you later switch to a PWM-driven strip, add an MCU command and implement
PwmLighting below - nothing else in the codebase has to change.
"""

from __future__ import annotations

from typing import Optional


class Lighting:
    """No-op base: always-on strip with no control line."""

    mode = "always_on"
    controllable = False

    def __init__(self, note: str = ""):
        self.note = note or "White LED strip on the 12 V rail; no software control."

    def on(self) -> None:
        """Called before capture. Nothing to do - the strip is already on."""

    def off(self) -> None:
        """Called after a scan. Deliberately does nothing for a hardwired strip."""

    def set_brightness(self, level: float) -> None:
        raise NotImplementedError(
            "This strip is hardwired to 12 V and cannot be dimmed in software. "
            "Set lighting.mode to 'pwm' and wire the strip to an MCU pin first.")

    def status(self) -> dict:
        return {
            "mode": self.mode,
            "controllable": self.controllable,
            "state": "on",
            "note": self.note,
        }


class NoLighting(Lighting):
    """No strip fitted. Reports honestly so the dashboard can warn."""

    mode = "none"
    controllable = False

    def __init__(self):
        super().__init__(
            "No enclosure lighting fitted. The model is colour-sensitive, so "
            "results will vary with ambient light and time of day.")

    def status(self) -> dict:
        s = super().status()
        s["state"] = "off"
        s["warning"] = "no controlled lighting - expect inconsistent colour"
        return s


class PwmLighting(Lighting):
    """
    Placeholder for a strip driven from an MCU PWM pin.

    Not wired on this build. Left as a clearly-marked stub so nobody has to
    guess at the interface later; it raises rather than silently pretending.
    """

    mode = "pwm"
    controllable = True

    def __init__(self, motor=None, brightness: float = 1.0):
        super().__init__("PWM strip on an MCU pin.")
        self.motor = motor
        self.brightness = brightness

    def _send(self, level: float) -> None:
        raise NotImplementedError(
            "PWM lighting is not implemented: this rig's strip is hardwired to "
            "12 V. To add it, define an 'L<0-255>' command in "
            "firmware/freshiq_tray/freshiq_tray.ino and send it here.")

    def on(self) -> None:
        self._send(self.brightness)

    def off(self) -> None:
        self._send(0.0)

    def set_brightness(self, level: float) -> None:
        self.brightness = max(0.0, min(1.0, float(level)))
        self._send(self.brightness)


def build_lighting(settings, motor=None) -> Lighting:
    mode = str(getattr(settings.lighting, "mode", "always_on")).lower()
    if mode == "none":
        return NoLighting()
    if mode == "pwm":
        return PwmLighting(motor=motor)
    return Lighting(note=getattr(settings.lighting, "note", ""))
