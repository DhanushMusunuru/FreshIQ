"""
FreshIQ AI package.

Visual food inspection: identify a fruit/vegetable and estimate its
fresh/ripe/spoiled condition from one or more camera views.

Public API (this is all the backend needs):

    from ai import classify_image, classify_frame, inspect

    classify_image(path)        -> single-view result dict
    classify_frame(bgr_frame)   -> single-view result dict
    inspect([...])              -> aggregated multi-view result dict

IMPORTANT: The percentages returned are the model's *visual classification
confidence*, not a scientific measurement of physical freshness or food safety.
See docs/LIMITATIONS.md.
"""

from .classifier import (
    classify_image,
    classify_frame,
    inspect,
    reset_backend,
)
from .config import Config, get_config, load_config

__all__ = [
    "classify_image",
    "classify_frame",
    "inspect",
    "reset_backend",
    "Config",
    "get_config",
    "load_config",
]

__version__ = "0.1.0"
