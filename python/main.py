"""
App Lab entry point.

`arduino-app-cli app start` runs THIS file on the Linux side (MPU) and flashes
../sketch/sketch.ino to the microcontroller (MCU). All the real code lives in
the packages next to this file; main.py only sets the working directory and
hands over to the existing server, so there is exactly one implementation of
FreshIQ rather than an App Lab copy that drifts from the standalone one.

The tray is reached from here through hardware/bridge_rpc.py, which speaks
MessagePack-RPC to arduino-router - the same service App Lab itself uses to
reach the sketch. See python/hardware/motor.py for why that, and not a serial
port, is the correct path on an UNO Q.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Imports must resolve to the packages beside this file, and relative paths in
# config/freshiq.json (data/, models/, sample_data/) are resolved from here.
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.chdir(HERE)

# backend/server.py parses sys.argv. App Lab may append its own arguments, and
# an unrecognised one would make argparse exit before the server ever starts,
# so hand it a clean argv and let config/freshiq.json supply every setting.
sys.argv = [sys.argv[0]]

from backend import server  # noqa: E402

if __name__ == "__main__":
    server.main()
