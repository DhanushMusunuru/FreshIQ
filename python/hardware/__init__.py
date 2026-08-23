"""
Hardware abstraction layer.

Everything that touches a physical device lives here, so the AI and the backend
never talk to a serial port or a GPIO pin directly. That separation is what
makes it possible to run the entire scan on a laptop with no rig attached.

    motor.py      tray rotation (UNO Q onboard MCU, external serial MCU,
                  or a simulator)
    bridge_rpc.py MPU -> MCU transport for the UNO Q (arduino-router RPC)
    lighting.py   enclosure LEDs (hardwired 12 V on this build)

The CAMERA deliberately does NOT live here. camera/capture.py already exists,
works, and is covered by the test suite; duplicating it into hardware/ would
mean two camera implementations drifting apart. The scan controller takes any
object exposing .snapshot(), which both camera/capture.Camera and the server's
CameraSource already do.
"""

from .lighting import Lighting, NoLighting, PwmLighting, build_lighting
from .bridge_rpc import BridgeError, build_rpc
from .motor import (
    BridgeTrayMotor,
    MotorError,
    SerialTrayMotor,
    SimulatedTrayMotor,
    TrayMotor,
    build_motor,
)

__all__ = [
    "MotorError",
    "BridgeError",
    "build_rpc",
    "TrayMotor",
    "BridgeTrayMotor",
    "SerialTrayMotor",
    "SimulatedTrayMotor",
    "build_motor",
    "Lighting",
    "NoLighting",
    "PwmLighting",
    "build_lighting",
]
