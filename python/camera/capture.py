"""
USB camera capture (CORE AI TASK / PHASE 2 & 3).

Thin OpenCV wrapper for grabbing frames from the USB camera mounted over the
FreshIQ rotating tray. Kept separate from the AI so the model never depends on a
camera being present (tests and the backend can feed images directly).

Typical use:

    from camera.capture import Camera
    from ai import classify_frame, inspect

    with Camera(index=0) as cam:
        frame = cam.capture()               # one frame
        print(classify_frame(frame))

        frames = cam.capture_sequence(4, delay=0.8)   # 4 rotating-tray views
        print(inspect(frames, food_hint="tomato"))

For the real rig, trigger the tray rotation between captures (a callback hook is
provided). Motor/Arduino control is intentionally OUT OF SCOPE here — pass a
callable if you want to rotate between shots.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional

import cv2
import numpy as np


class CameraError(RuntimeError):
    pass



def _open_failure_hint(index) -> str:
    """
    Say WHICH failure this is.

    "Could not open camera" covers two different faults in one sentence, and
    they need opposite fixes: no camera attached, versus a camera that is
    attached and working but already claimed by another process. The second is
    the common one on this rig - an earlier instance of the app still holding
    /dev/video0 - and calling that "no camera found" sends you off to check the
    USB cable on a camera that is perfectly fine.
    """
    import glob
    nodes = sorted(glob.glob("/dev/video*"))

    if not nodes:
        return (
            "No camera found: there are no /dev/video* devices at all, so the "
            "system cannot see a camera."
            "\n  Plug the USB camera in, then check:  ls -l /dev/video*"
        )

    return (
        "Camera {index!r} could not be opened, but {nodes} {verb} present - a "
        "camera IS attached, so this is almost certainly not a hardware fault."
        "\n  Most likely another process already holds the device, usually an "
        "earlier instance of FreshIQ that was never stopped."
        "\n  Find the holder:   sudo fuser -v {first}"
        "\n  Stop a stale app:  arduino-app-cli app stop ~/ArduinoApps/FreshIQ"
        "\n  A UVC camera often exposes two nodes (video0 = frames, video1 = "
        "metadata only); if index 0 refuses, try the next one in "
        "python/config/freshiq.json."
    ).format(index=index, nodes=", ".join(nodes),
             verb="is" if len(nodes) == 1 else "are", first=nodes[0])


class Camera:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720,
                 warmup_frames: int = 5, backend: Optional[int] = None):
        self.index = index
        self.width = width
        self.height = height
        self.warmup_frames = warmup_frames
        self.backend = backend
        self.cap: Optional[cv2.VideoCapture] = None

    def _try_open(self, source, backend=None):
        """Open one (source, backend) pair. Returns a live capture or None."""
        try:
            cap = cv2.VideoCapture(source, backend) if backend is not None \
                else cv2.VideoCapture(source)
        except Exception:  # noqa: BLE001
            return None
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()
        return None

    def _candidates(self):
        """
        What to try, in order.

        `index` may be an int (OpenCV index) or a device path such as
        "/dev/video0" - a UVC camera does not always land on the index its
        device node number suggests, and on some boards index 0 fails outright
        while the explicit path works. Trying V4L2 by name after the plain
        index costs nothing and removes a whole class of "can't open camera by
        index" failures.
        """
        v4l2 = getattr(cv2, "CAP_V4L2", None)
        raw = self.index

        if isinstance(raw, str):
            yield raw, None
            if v4l2 is not None:
                yield raw, v4l2
            return

        yield raw, None
        if v4l2 is not None:
            yield raw, v4l2
            yield "/dev/video{}".format(raw), v4l2

    def open(self) -> "Camera":
        if self.backend is not None:
            # An explicit backend is an instruction, not a hint: honour it alone.
            self.cap = self._try_open(self.index, self.backend)
        else:
            self.cap = None
            for source, backend in self._candidates():
                self.cap = self._try_open(source, backend)
                if self.cap is not None:
                    break

        if not self.cap or not self.cap.isOpened():
            raise CameraError(_open_failure_hint(self.index))
        # Opening is NOT proof of anything. isOpened() only says the device
        # node was claimed; whether the camera will actually deliver frames in
        # the requested mode is a separate question, and _prepare() answers it.
        if self._prepare(self.cap, set_size=True):
            return self

        # The camera opened but will not read at the requested size. Cheap USB2
        # cameras advertise modes their bandwidth cannot sustain, so the set()
        # calls "succeed", every read() then fails, and the symptom is a log
        # line saying the camera opened followed by a permanently blank preview.
        # Fall back to whatever mode the camera offers by default rather than
        # serving nothing. When the requested size DOES work this branch never
        # runs, so nothing changes for a camera that was already fine.
        self.cap.release()
        self.cap = None
        for source, backend in self._candidates():
            cap = self._try_open(source, backend)
            if cap is None:
                continue
            if self._prepare(cap, set_size=False):
                self.cap = cap
                got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                print("[camera] {}x{} is not usable on this camera; fell back to "
                      "its default mode {}x{}".format(self.width, self.height,
                                                      got[0], got[1]), flush=True)
                return self
            cap.release()

        raise CameraError(
            f"Camera {self.index!r} opens but will not deliver frames, at "
            f"{self.width}x{self.height} or at its own default. Check what modes "
            "it actually supports with:  v4l2-ctl --list-formats-ext -d /dev/video0")

    def _prepare(self, cap, set_size: bool) -> bool:
        """
        Configure a capture and PROVE it can read. Returns False if it cannot.

        The old code set the resolution, did the warmup reads, threw away
        whether they worked, and returned success unconditionally. That is why
        a camera that opened but never read still logged "opened index 0" and
        then starved the preview forever with nothing explaining it.
        """
        if set_size:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Keep only the newest frame in the driver queue.
        #
        # By default OpenCV buffers several frames. A caller that reads slower
        # than the camera produces then receives the OLDEST queued frame, so the
        # view trails reality by however many are stacked up - it looks exactly
        # like network lag on a purely local camera. Not every backend honours
        # this, which is why backend/server.py also drains continuously.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        # These double as the warm-up that lets auto-exposure and white balance
        # settle, so the frames are discarded either way - only whether ANY of
        # them arrived is kept.
        ok = False
        for _ in range(max(1, self.warmup_frames)):
            got, _frame = cap.read()
            ok = ok or bool(got)
        return ok

    def capture(self) -> np.ndarray:
        """Grab a single BGR frame. Raises CameraError on failure."""
        if self.cap is None:
            self.open()
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise CameraError("Failed to read a frame from the camera.")
        return frame

    def capture_sequence(self, count: int, delay: float = 0.8,
                         on_before_capture: Optional[Callable[[int], None]] = None
                         ) -> List[np.ndarray]:
        """
        Capture ``count`` frames, e.g. one per tray angle.

        Args:
            count: number of views to capture.
            delay: seconds to wait between captures (let the tray settle).
            on_before_capture: optional callback(index) called before each shot —
                use it to rotate the tray (drive the motor / signal the Arduino).
                Rotation logic itself lives outside the AI module.
        """
        frames: List[np.ndarray] = []
        for i in range(count):
            if on_before_capture is not None:
                on_before_capture(i)
                # small settle after motion
                time.sleep(min(delay, 0.3))
            frames.append(self.capture())
            if i < count - 1:
                time.sleep(delay)
        return frames

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # Context-manager sugar.
    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()


def capture_single(index: int = 0) -> np.ndarray:
    """Convenience: open a camera, grab one frame, close it."""
    with Camera(index=index) as cam:
        return cam.capture()
