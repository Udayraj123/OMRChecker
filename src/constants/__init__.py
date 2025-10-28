"""
Constants package for OMRChecker.

Note: __all__ was intentionally removed to allow importing all
constants from the submodule without restricting the exported
names. Use `from src.constants.image_processing import *` or
import specific names as needed.
"""
from .image_processing import *  # noqa

# Summary of recent changes:
# - Removed restrictive __all__ list to avoid limiting imports
#   from the constants package.