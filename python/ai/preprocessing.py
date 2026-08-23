"""
Image loading, segmentation and normalisation utilities.

Everything here works on OpenCV BGR ``numpy`` arrays (the format returned by
``cv2.VideoCapture.read`` and ``cv2.imread``) so the same helpers serve both the
"single image file" path and the "live camera frame" path.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """Load an image file as a BGR array. Raises ValueError if it cannot be read."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {image_path!r}")
    return img


def ensure_bgr(frame: np.ndarray) -> np.ndarray:
    """
    Normalise an input array to a 3-channel BGR uint8 image.

    Accepts grayscale, BGRA, or BGR. Raises on anything else.
    """
    if frame is None:
        raise ValueError("frame is None")
    if frame.dtype != np.uint8:
        # Assume a float image in [0,1] or [0,255]; scale sensibly.
        f = frame.astype(np.float32)
        if f.max() <= 1.0:
            f = f * 255.0
        frame = np.clip(f, 0, 255).astype(np.uint8)
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame
    raise ValueError(f"Unsupported frame shape: {frame.shape}")


def resize_max(frame: np.ndarray, max_side: int = 640) -> np.ndarray:
    """Downscale so the longest side is at most ``max_side`` (keeps aspect ratio)."""
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame
    scale = max_side / float(longest)
    return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def segment_food(frame: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """
    Best-effort separation of the food item from the tray/background.

    Strategy (fast, no ML): threshold on colour saturation to find the coloured
    object against a plainer tray, keep the largest connected blob, and clean it
    up morphologically. If that fails to produce a plausible object (area too
    small or nearly the whole frame), fall back to a centred rectangle so
    downstream code still gets a region to analyse.

    Returns:
        mask: uint8 array (0 or 255), 255 = food.
        bbox: (x, y, w, h) of the food region, or None if a full-frame fallback
              was used.

    This is intentionally simple and documented as a limitation: it assumes a
    single, roughly centred item on a relatively plain tray, which matches the
    FreshIQ rotating-tray setup. It is not a general-purpose segmenter.
    """
    h, w = frame.shape[:2]
    area = h * w
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Foreground guess: saturated OR clearly not-background-bright pixels.
    _, sat_mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Also treat very dark regions (possible dark/rotten food) as foreground.
    dark_mask = (v < 60).astype(np.uint8) * 255
    mask = cv2.bitwise_or(sat_mask, dark_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        blob_area = cv2.contourArea(largest)
        frac = blob_area / float(area)
        if 0.02 <= frac <= 0.98:
            clean = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
            x, y, bw, bh = cv2.boundingRect(largest)
            return clean, (x, y, bw, bh)

    # Fallback: centred rectangle covering the middle 70%.
    fallback = np.zeros((h, w), dtype=np.uint8)
    x0, y0 = int(w * 0.15), int(h * 0.15)
    x1, y1 = int(w * 0.85), int(h * 0.85)
    fallback[y0:y1, x0:x1] = 255
    return fallback, None


def list_food_contours(frame: np.ndarray, min_area_fraction: float) -> list:
    """
    Return the list of connected foreground blobs larger than ``min_area_fraction``
    of the frame. Used by the quality check to detect "multiple objects".
    """
    h, w = frame.shape[:2]
    area = h * w
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    _, sat_mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel)
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in contours if cv2.contourArea(c) >= min_area_fraction * area]
    return big


def prepare_for_model(frame: np.ndarray, input_size: int) -> np.ndarray:
    """
    Convert a BGR frame into a normalised RGB tensor of shape
    (1, input_size, input_size, 3), float32 in [0, 1] — the format the
    trained MobileNet/EfficientNet-Lite model expects.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0)
