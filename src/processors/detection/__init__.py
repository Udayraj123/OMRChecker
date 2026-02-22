"""Detection processors and field detection algorithms."""

from src.processors.detection.detection_repository import DetectionRepository
from src.processors.detection.processor import ReadOMRProcessor

__all__ = ["ReadOMRProcessor", "DetectionRepository"]
