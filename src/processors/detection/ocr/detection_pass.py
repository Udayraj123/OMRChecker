from src.processors.detection.base.detection_pass import FieldTypeDetectionPass
from src.processors.detection.ocr.detection import OCRFieldDetection
from src.processors.layout.field.base import Field
from src.processors.detection.detection_repository import DetectionRepository


class OCRDetectionPass(FieldTypeDetectionPass):
    def __init__(self, *args, repository: DetectionRepository, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.repository = repository

    # Note: This is used by parent to generate the detection
    def get_field_detection(
        self, field: Field, gray_image, colored_image
    ) -> OCRFieldDetection:
        return OCRFieldDetection(field, gray_image, colored_image)

    def initialize_directory_level_aggregates(self, initial_directory_path) -> None:
        super().initialize_directory_level_aggregates(initial_directory_path)
        self.insert_directory_level_aggregates(
            {
                # TODO: check if any insert needed
            }
        )

    def initialize_file_level_aggregates(self, file_path) -> None:
        super().initialize_file_level_aggregates(file_path)
        self.insert_file_level_aggregates(
            {
                # TODO: check if any insert needed
            }
        )

    def update_field_level_aggregates_on_processed_field_detection(
        self, field: Field, field_detection: OCRFieldDetection
    ) -> None:
        super().update_field_level_aggregates_on_processed_field_detection(
            field, field_detection
        )

        # Save to repository
        self.repository.save_ocr_field(field.id, field_detection.result)

        self.insert_field_level_aggregates({"detections": field_detection.detections})

    def update_file_level_aggregates_on_processed_field_detection(
        self,
        field: Field,
        field_detection: OCRFieldDetection,
        field_level_aggregates,
    ) -> None:
        # Skip populating field_label_wise_aggregates (using repository)
        # Just update fields_count for statistics
        file_level_aggregates = self.get_file_level_aggregates()
        file_level_aggregates["fields_count"].push("processed")
