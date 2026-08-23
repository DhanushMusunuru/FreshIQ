"""USB camera capture utilities for the FreshIQ rotating-tray rig."""

from .capture import Camera, CameraError, capture_single

__all__ = ["Camera", "CameraError", "capture_single"]
