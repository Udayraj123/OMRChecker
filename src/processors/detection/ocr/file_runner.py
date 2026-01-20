from src.processors.constants import FieldDetectionType
from src.processors.detection.base.file_runner import FieldTypeFileLevelRunner
from src.processors.detection.ocr.detection_pass import OCRDetectionPass
from src.processors.detection.ocr.interpretation_pass import (
    OCRInterpretationPass,
)
from src.processors.repositories.detection_repository import DetectionRepository


class OCRFileRunner(FieldTypeFileLevelRunner):
    def __init__(self, tuning_config, repository: DetectionRepository) -> None:
        field_detection_type = FieldDetectionType.OCR
        detection_pass = OCRDetectionPass(
            tuning_config, field_detection_type, repository=repository
        )
        interpretation_pass = OCRInterpretationPass(
            tuning_config, field_detection_type, repository=repository
        )
        super().__init__(
            tuning_config, field_detection_type, detection_pass, interpretation_pass
        )
        self.repository = repository
