import os
from typing import List

from src.algorithm.template.detection.base.file_processor import FieldTypeFileProcessor
from src.algorithm.template.detection.bubbles_threshold.file_processor import (
    BubblesThresholdFileProcessor,
)
from src.algorithm.template.detection.ocr.file_processor import OCRFileProcessor
from src.algorithm.template.template_layout import Field
from src.processors.constants import FieldDetectionType
from src.utils.stats import StatsByLabel

"""
Template Detector takes care of detections of an image file using a single template
We create one instance of TemplateDetector per Template.
Note: a Template may get reused for multiple directories(in nested case)
"""


class TemplateDetector:
    field_type_to_processor = {
        FieldDetectionType.BUBBLES_THRESHOLD: BubblesThresholdFileProcessor,
        FieldDetectionType.OCR: OCRFileProcessor,
        # FieldDetectionType.BARCODE_QR: BarcodeQRProcessor,
        # FieldDetectionType.BUBBLES_BLOB: BubblesBlobProcessor,
    }

    def __init__(self, template):
        self.template = template
        self.all_fields = template.all_fields
        self.prepare_field_type_processors()
        initial_directory_path = os.path.dirname(template.path)
        self.initialize_directory_level_aggregates(initial_directory_path)

    def prepare_field_type_processors(self):
        template = self.template
        # Create instances of all required field type processors
        self.field_type_processors = {
            field_detection_type: self.get_field_detection_type_processor(
                field_detection_type
            )
            for field_detection_type in template.all_field_detection_types
        }
        # List of fields and their mapped processors
        all_fields: List[Field] = template.all_fields
        self.all_field_type_processors = [
            (field, self.field_type_processors[field.field_detection_type])
            for field in all_fields
        ]

    def get_field_detection_type_processor(
        self, field_detection_type
    ) -> FieldTypeFileProcessor:
        tuning_config = self.template.tuning_config
        FieldTypeProcessorClass = TemplateDetector.field_type_to_processor[
            field_detection_type
        ]
        return FieldTypeProcessorClass(tuning_config)

    def initialize_directory_level_aggregates(self, initial_directory_path):
        self.directory_level_aggregates = {
            "directory_level": {
                "initial_directory_path": initial_directory_path,
                "interpreted_files_count": StatsByLabel("processed", "multi_marked"),
            },
            "field_detection_type_level_detection_aggregates": {
                # This count would get different in conditional sets
                field_detection_type: {
                    "fields_count": StatsByLabel("processed"),
                    # More get added from the field type
                }
                for field_detection_type in self.template.all_field_detection_types
            },
            "field_detection_type_level_interpretation_aggregates": {
                field_detection_type: {
                    "fields_count": StatsByLabel("processed"),
                    # TODO: More get added from the field type
                }
                for field_detection_type in self.template.all_field_detection_types
            },
            # field_label_wise_interpretation_aggregates
            "file_level_detection_aggregates": {
                "field_label_wise_detection_aggregates": None,
                "field_detection_type_wise_detection_aggregates": None,
            },
            "file_level_interpretation_aggregates": {},
        }
        for field_type_processor in self.field_type_processors.values():
            field_type_processor.initialize_directory_level_aggregates()

    def read_omr_and_update_metrics(self, file_path, gray_image, colored_image):
        # First pass to compute aggregates like global threshold
        # TODO: populate local thresholds even in first pass? (to enable multiple passes)

        # populate detections
        self.run_file_level_detection(file_path, gray_image, colored_image)

        # populate interpretations
        omr_response = self.run_file_level_interpretation(
            file_path, gray_image, colored_image
        )

        return omr_response

    def run_file_level_detection(self, file_path, gray_image, colored_image):
        self.initialize_file_level_detection_aggregates(file_path)

        # Perform detection step for each field
        for field, field_type_processor in self.all_field_type_processors:
            # TODO: see where the conditional sets logic can fit in this loop (or at a wrapper level?)
            # TODO: field object should contain the corresponding runtime config for the detection
            field_type_processor.run_field_level_detection(
                field, gray_image, colored_image
            )
            self.update_field_level_detection_aggregates(
                file_path, field, field_type_processor
            )

        self.update_detection_aggregates_on_processed_file(file_path)

    def update_field_level_detection_aggregates(
        self, file_path, field: Field, field_type_processor: FieldTypeFileProcessor
    ):
        field_detection_type = field_type_processor.field_detection_type

        field_label_wise_detection_aggregates = self.file_level_detection_aggregates[
            "field_label_wise_detection_aggregates"
        ]
        field_label_wise_detection_aggregates[
            field.field_label
        ] = field_type_processor.get_field_level_detection_aggregates()

        field_detection_type_level_detection_aggregates = (
            self.directory_level_aggregates[
                "field_detection_type_level_detection_aggregates"
            ][field_detection_type]
        )
        # Update the processed field count for that processor
        field_detection_type_level_detection_aggregates["fields_count"].push(
            "processed"
        )

    def update_detection_aggregates_on_processed_file(self, file_path):
        # Updating field detection type level aggregates
        field_detection_type_wise_detection_aggregates = {}
        for field_type_processor in self.field_type_processors.values():
            field_type_processor.update_detection_aggregates_on_processed_file(
                file_path
            )
            field_detection_type_wise_detection_aggregates[
                field_type_processor.field_detection_type
            ] = field_type_processor.get_file_level_detection_aggregates()

        # Update aggregates in state
        self.file_level_detection_aggregates[
            "field_detection_type_wise_detection_aggregates"
        ] = field_detection_type_wise_detection_aggregates
        self.directory_level_aggregates["file_level_detection_aggregates"][
            file_path
        ] = self.file_level_detection_aggregates

        # TODO: uncomment if needed for directory level graphs
        # self.directory_level_aggregates["field_label_wise_detection_aggregates"][file_path] = self.field_label_wise_detection_aggregates

    def initialize_file_level_detection_aggregates(self, file_path):
        self.file_level_detection_aggregates = {
            "file_path": file_path,
            "field_label_wise_detection_aggregates": {},
            "field_detection_type_wise_detection_aggregates": {},
        }

        # Setup field type wise metrics
        for field_type_processor in self.field_type_processors.values():
            field_type_processor.initialize_file_level_detection_aggregates(file_path)

    # TODO: move into template_interpreter as a subclass of TemplatePass?
    def run_file_level_interpretation(self, file_path, gray_image, colored_image):
        self.initialize_file_level_interpretation_aggregates(file_path)

        current_omr_response = {}
        # Perform interpretation step for each field
        for field, field_type_processor in self.all_field_type_processors:
            detected_string = field_type_processor.run_field_level_interpretation(
                field, gray_image, colored_image
            )

            self.update_field_level_interpretation_aggregates(
                current_omr_response, field, field_type_processor
            )

            field_label = field.field_label
            current_omr_response[field_label] = detected_string

        self.update_interpretation_aggregates_on_processed_file(
            file_path, current_omr_response
        )

        return current_omr_response

    def initialize_file_level_interpretation_aggregates(self, file_path):
        self.file_level_interpretation_aggregates = {
            "file_path": file_path,
            "read_response_flags": {
                "is_multi_marked": False,
                "multi_marked_fields": [],
                "is_identifier_multi_marked": False,
            },
            "field_label_wise_interpretation_aggregates": {},
            "field_detection_type_wise_interpretation_aggregates": {},
        }

        # Interpretation loop needs access to the file level detection aggregates
        # TODO: get this as a getter inside template_interpreter(self.detector_ref)
        all_file_level_detection_aggregates = self.directory_level_aggregates[
            "file_level_detection_aggregates"
        ][file_path]
        field_detection_type_wise_detection_aggregates = (
            all_file_level_detection_aggregates[
                "field_detection_type_wise_detection_aggregates"
            ]
        )
        field_label_wise_detection_aggregates = all_file_level_detection_aggregates[
            "field_label_wise_detection_aggregates"
        ]

        # Setup field type wise metrics
        for field_type_processor in self.field_type_processors.values():
            field_type_processor.initialize_file_level_interpretation_aggregates(
                file_path,
                field_detection_type_wise_detection_aggregates,
                field_label_wise_detection_aggregates,
            )

    def update_field_level_interpretation_aggregates(
        self,
        current_omr_response,
        field: Field,
        field_type_processor: FieldTypeFileProcessor,
    ):
        field_label = field.field_label
        field_label_wise_interpretation_aggregates = (
            self.file_level_interpretation_aggregates[
                "field_label_wise_interpretation_aggregates"
            ]
        )

        field_level_interpretation_aggregates = (
            field_type_processor.get_field_level_interpretation_aggregates()
        )
        field_label_wise_interpretation_aggregates[
            field_label
        ] = field_level_interpretation_aggregates

        read_response_flags = self.file_level_interpretation_aggregates[
            "read_response_flags"
        ]

        if field_level_interpretation_aggregates["is_multi_marked"]:
            read_response_flags["is_multi_marked"] = True
            read_response_flags["multi_marked_fields"].push(field)
            # TODO: define identifier_labels
            # if field.is_part_of_identifier():
            # if field_label in self.template.identifier_labels:
            #     read_response_flags["is_identifier_multi_marked"] = True

        # TODO: support for more validations here?

        field_detection_type = field_type_processor.field_detection_type

        field_detection_type_level_interpretation_aggregates = (
            self.directory_level_aggregates[
                "field_detection_type_level_interpretation_aggregates"
            ][field_detection_type]
        )
        # Update the processed field count for that processor
        field_detection_type_level_interpretation_aggregates["fields_count"].push(
            "processed"
        )

    def get_file_level_interpretation_aggregates(self):
        return self.file_level_interpretation_aggregates

    def update_interpretation_aggregates_on_processed_file(
        self, file_path, current_omr_response
    ):
        # Updating field interpretation type level aggregates
        field_detection_type_wise_interpretation_aggregates = {}
        for field_type_processor in self.field_type_processors.values():
            # Note: This would contain confidence_metrics_for_file
            field_type_processor.update_interpretation_aggregates_on_processed_file(
                file_path
            )
            field_detection_type_wise_interpretation_aggregates[
                field_type_processor.field_detection_type
            ] = field_type_processor.get_file_level_interpretation_aggregates()

        # Update aggregates in state
        self.file_level_interpretation_aggregates[
            "field_detection_type_wise_interpretation_aggregates"
        ] = field_detection_type_wise_interpretation_aggregates
        self.directory_level_aggregates["file_level_interpretation_aggregates"][
            file_path
        ] = self.file_level_interpretation_aggregates

        # Update read_response_flags
        read_response_flags = self.file_level_interpretation_aggregates[
            "read_response_flags"
        ]

        self.directory_level_aggregates["directory_level"][
            "interpreted_files_count"
        ].push("processed")
        if read_response_flags["is_multi_marked"]:
            self.directory_level_aggregates["directory_level"][
                "interpreted_files_count"
            ].push("multi_marked")

        # confidence_metrics_for_file = self.file_level_interpretation_aggregates["field_detection_type_wise_interpretation_aggregates"]["confidence_metrics_for_file"]

    def finalize_directory_metrics(self):
        # TODO: get_directory_level_confidence_metrics()

        # output_metrics = self.directory_level_aggregates
        # TODO: export directory level stats here
        pass
