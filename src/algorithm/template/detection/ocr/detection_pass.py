from src.algorithm.template.detection.base.detection_pass import FieldTypeDetectionPass
from src.algorithm.template.detection.ocr.detection import OCRFieldDetection
from src.algorithm.template.template_layout import Field


class OCRDetectionPass(FieldTypeDetectionPass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Note: This is used by parent to generate the detection
    def get_field_detection(
        self, field: Field, gray_image, colored_image
    ) -> OCRFieldDetection:
        return OCRFieldDetection(field, gray_image, colored_image)

    def initialize_directory_level_aggregates(self, initial_directory_path):
        super().initialize_directory_level_aggregates(initial_directory_path)
        self.insert_directory_level_aggregates(
            {
                # TODO: check if any insert needed
            }
        )

    def initialize_file_level_aggregates(self, file_path):
        super().initialize_file_level_aggregates(file_path)
        self.insert_file_level_aggregates(
            {
                # TODO: check if any insert needed
            }
        )

    def update_field_level_aggregates_on_processed_field_detection(
        self, field: Field, field_detection: OCRFieldDetection
    ):
        super().update_field_level_aggregates_on_processed_field_detection(
            field, field_detection
        )
        self.insert_field_level_aggregates(
            {"detected_texts": field_detection.detected_texts}
        )

    def update_file_level_aggregates_on_processed_field_detection(
        self,
        field: Field,
        field_detection: OCRFieldDetection,
        field_level_aggregates,
    ):
        super().update_file_level_aggregates_on_processed_field_detection(
            field, field_detection, field_level_aggregates
        )
        # TODO: check if any insert needed
