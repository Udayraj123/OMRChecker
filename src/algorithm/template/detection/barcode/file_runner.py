from src.algorithm.template.detection.barcode.detection_pass import BarcodeDetectionPass
from src.algorithm.template.detection.barcode.interpretation_pass import (
    BarcodeInterpretationPass,
)
from src.algorithm.template.detection.base.file_runner import FieldTypeFileLevelRunner
from src.processors.constants import FieldDetectionType


class BarcodeFileRunner(FieldTypeFileLevelRunner):
    def __init__(self, tuning_config):
        field_detection_type = FieldDetectionType.Barcode
        detection_pass = BarcodeDetectionPass(tuning_config, field_detection_type)
        interpretation_pass = BarcodeInterpretationPass(
            tuning_config, field_detection_type
        )
        super().__init__(
            tuning_config, field_detection_type, detection_pass, interpretation_pass
        )
