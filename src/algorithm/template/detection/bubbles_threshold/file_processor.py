from src.algorithm.template.detection.base.file_processor import FieldTypeFileProcessor
from src.algorithm.template.detection.bubbles_threshold.detection_pass import (
    BubblesThresholdDetectionPass,
)
from src.algorithm.template.detection.bubbles_threshold.interpretation_pass import (
    BubblesThresholdInterpretationPass,
)
from src.processors.constants import FieldDetectionType


class BubblesThresholdFileProcessor(FieldTypeFileProcessor):
    def __init__(self, tuning_config):
        field_detection_type = FieldDetectionType.BUBBLES_THRESHOLD
        detection_pass = BubblesThresholdDetectionPass(
            tuning_config, field_detection_type
        )
        interpretation_pass = BubblesThresholdInterpretationPass(
            tuning_config, field_detection_type
        )
        super().__init__(
            tuning_config, field_detection_type, detection_pass, interpretation_pass
        )
