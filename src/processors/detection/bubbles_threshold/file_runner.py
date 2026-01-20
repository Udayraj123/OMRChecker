from src.processors.constants import FieldDetectionType
from src.processors.detection.base.file_runner import FieldTypeFileLevelRunner
from src.processors.detection.bubbles_threshold.detection_pass import (
    BubblesThresholdDetectionPass,
)
from src.processors.detection.bubbles_threshold.interpretation_pass import (
    BubblesThresholdInterpretationPass,
)
from src.processors.repositories.detection_repository import DetectionRepository


class BubblesThresholdFileRunner(FieldTypeFileLevelRunner):
    def __init__(self, tuning_config, repository: DetectionRepository) -> None:
        field_detection_type = FieldDetectionType.BUBBLES_THRESHOLD
        detection_pass = BubblesThresholdDetectionPass(
            tuning_config, field_detection_type, repository=repository
        )
        interpretation_pass = BubblesThresholdInterpretationPass(
            tuning_config, field_detection_type, repository=repository
        )
        super().__init__(
            tuning_config, field_detection_type, detection_pass, interpretation_pass
        )
        self.repository = repository
