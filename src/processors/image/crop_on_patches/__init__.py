"""
Patch-based cropping package.

This package contains CropOnPatchesCommon and its subclasses for detecting
and aligning OMR sheets using various patch detection methods (dots, lines, markers).
"""

from src.processors.image.crop_on_patches.common import CropOnPatchesCommon
from src.processors.image.crop_on_patches.custom_markers import CropOnCustomMarkers
from src.processors.image.crop_on_patches.dot_lines import CropOnDotLines
from src.processors.image.crop_on_patches.l_markers import CropOnLMarkers

__all__ = [
    "CropOnPatchesCommon",
    "CropOnCustomMarkers",
    "CropOnDotLines",
    "CropOnLMarkers",
]
