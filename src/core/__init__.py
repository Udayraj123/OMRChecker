"""Core OMRChecker library - Pure processing logic without CLI dependencies.

This module provides the main OMRProcessor class which can be used both
by the CLI and as a library API.
"""

from src.core.omr_processor import OMRProcessor
from src.core.types import OMRResult, ProcessorConfig

__all__ = ["OMRProcessor", "OMRResult", "ProcessorConfig"]
