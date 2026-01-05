"""Unified processor infrastructure for OMR processing.

This module provides a unified architecture for all image processing,
alignment, and OMR detection operations.

Note: Some processors are lazy-loaded to avoid circular import dependencies.
"""

from typing import Any

# Core processor infrastructure (no circular dependencies)
from src.processors.base import ProcessingContext, Processor
from src.processors.pipeline import ProcessingPipeline


# Lazy import via __getattr__ to avoid circular dependencies
def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy-load processors that have circular dependencies."""
    # Import statements are intentionally not at top-level to avoid circular imports
    if name == "ImageTemplatePreprocessor":
        from src.processors.image import ImageTemplatePreprocessor

        return ImageTemplatePreprocessor
    if name == "PreprocessingProcessor":
        from src.processors.image import PreprocessingProcessor

        return PreprocessingProcessor
    if name == "AlignmentProcessor":
        from src.processors.alignment import AlignmentProcessor

        return AlignmentProcessor
    if name == "ReadOMRProcessor":
        from src.processors.detection import ReadOMRProcessor

        return ReadOMRProcessor
    if name == "EvaluationProcessor":
        from src.processors.evaluation import EvaluationProcessor

        return EvaluationProcessor
    if name == "FileOrganizerProcessor":
        from src.processors.organization import FileOrganizerProcessor

        return FileOrganizerProcessor
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "AlignmentProcessor",
    "EvaluationProcessor",
    "FileOrganizerProcessor",
    "ImageTemplatePreprocessor",
    "PreprocessingProcessor",
    "ProcessingContext",
    "ProcessingPipeline",
    "Processor",
    "ReadOMRProcessor",
]
