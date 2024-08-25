import os

from src.algorithm.detection.barcode_qr.barcode_qr_detector import BarcodeQRDetector
from src.algorithm.detection.bubbles_blob.bubbles_blob_detector import (
    BubblesBlobDetector,
)
from src.algorithm.detection.bubbles_threshold.bubbles_threshold_detector import (
    BubblesThresholdDetector,
)
from src.algorithm.detection.ocr.ocr_detector import OCRDetector
from src.processors.constants import FieldDetectionType
from src.utils.stats import StatsByLabel

"""
Template Detector takes care of detections of an image file using a single template
We create one instance of TemplateDetector per Template.
Note: a Template may get reused for multiple directories(in nested case)
"""


class TemplateDetector:
    field_type_to_detector = {
        FieldDetectionType.BUBBLES_THRESHOLD: BubblesThresholdDetector,
        FieldDetectionType.BUBBLES_BLOB: BubblesBlobDetector,
        FieldDetectionType.OCR: OCRDetector,
        FieldDetectionType.BARCODE_QR: BarcodeQRDetector,
    }

    def __init__(self, template):
        self.template = template
        self.all_fields = template.all_fields
        self.prepare_field_type_detectors()
        initial_directory_path = os.path.dirname(template.path)
        self.reset_directory_level_aggregates(initial_directory_path)

    def prepare_field_type_detectors(self):
        template = self.template
        # Create instances of all required field type detectors
        self.field_type_detectors = {
            field_detection_type: self.get_field_detector_instance(field_detection_type)
            for field_detection_type in template.all_field_detection_types
        }
        # List of fields and their mapped detectors
        self.all_field_detectors = [
            (field, self.field_type_detectors[field.field_detection_type])
            for field in template.all_fields
        ]

    def get_field_detector_instance(self, field_detection_type):
        tuning_config = self.template.tuning_config
        # Note: detector level config would be passed at runtime! Because aggregates need to happen per field type
        # Instances per different config may need to be created.
        # That's why "reader" may be multiple, but "detector" will be single per directory
        # Note: field object can have the corresponding runtime config for the detection
        FieldDetectorClass = TemplateDetector.field_type_to_detector[
            field_detection_type
        ]
        return FieldDetectorClass(tuning_config)

    def reset_directory_level_aggregates(self, initial_directory_path):
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
        for field_type_detector in self.field_type_detectors.values():
            field_type_detector.reset_directory_level_aggregates()

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
        self.reset_file_level_detection_aggregates(file_path)

        # TODO: see where the conditional sets logic can fit in this loop (or at a wrapper level?)
        # Perform detection step for each field
        for field, field_type_detector in self.all_field_detectors:
            # TODO: field object should contain the corresponding runtime config for the detection
            field_type_detector.run_field_level_detection(
                field, gray_image, colored_image
            )
            self.update_field_level_detection_aggregates(
                file_path, field, field_type_detector
            )

        self.update_file_level_detection_aggregates(file_path)

    def update_field_level_detection_aggregates(
        self, file_path, field, field_type_detector
    ):
        field_detection_type = field_type_detector.field_detection_type

        field_label_wise_detection_aggregates = self.file_level_detection_aggregates[
            "field_label_wise_detection_aggregates"
        ]
        field_label_wise_detection_aggregates[
            field.field_label
        ] = field_type_detector.get_field_level_detection_aggregates()

        field_detection_type_level_detection_aggregates = (
            self.directory_level_aggregates[
                "field_detection_type_level_detection_aggregates"
            ][field_detection_type]
        )
        # Update the processed field count for that detector
        field_detection_type_level_detection_aggregates["fields_count"].push(
            "processed"
        )

    def update_file_level_detection_aggregates(self, file_path):
        # Updating field detection type level aggregates
        field_detection_type_wise_detection_aggregates = {}
        for field_type_detector in self.field_type_detectors.values():
            field_type_detector.finalize_file_level_detection_aggregates()
            field_detection_type_wise_detection_aggregates[
                field_type_detector.field_detection_type
            ] = field_type_detector.get_file_level_detection_aggregates()

        # Update aggregates in state
        self.file_level_detection_aggregates[
            "field_detection_type_wise_detection_aggregates"
        ] = field_detection_type_wise_detection_aggregates
        self.directory_level_aggregates["file_level_detection_aggregates"][
            file_path
        ] = self.file_level_detection_aggregates

        # TODO: uncomment if needed for directory level graphs
        # self.directory_level_aggregates["field_label_wise_detection_aggregates"][file_path] = self.field_label_wise_detection_aggregates

    def reset_file_level_detection_aggregates(self, file_path):
        self.file_level_detection_aggregates = {
            "file_path": file_path,
            "field_label_wise_detection_aggregates": {},
            "field_detection_type_wise_detection_aggregates": {},
        }

        # Setup field type wise metrics
        for field_type_detector in self.field_type_detectors.values():
            field_type_detector.reset_file_level_detection_aggregates(file_path)

    def run_file_level_interpretation(self, file_path, gray_image, colored_image):
        self.reset_file_level_interpretation_aggregates(file_path)

        current_omr_response = {}
        # Perform interpretation step for each field
        for field, field_type_detector in self.all_field_detectors:
            # TODO: field object should contain the corresponding runtime config for the detection
            field_type_detector.reset_field_level_interpretation_aggregates()
            detected_string = field_type_detector.run_field_level_interpretation(
                field, gray_image, colored_image
            )

            self.update_field_level_interpretation_aggregates(
                current_omr_response, field, field_type_detector
            )

            field_label = field.field_label
            current_omr_response[field_label] = detected_string

        self.update_file_level_interpretation_aggregates(
            file_path, current_omr_response
        )

        return current_omr_response

    def reset_file_level_interpretation_aggregates(self, file_path):
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
        for field_type_detector in self.field_type_detectors.values():
            field_type_detector.reinitialize_file_level_interpretation_aggregates(
                file_path,
                field_detection_type_wise_detection_aggregates,
                field_label_wise_detection_aggregates,
            )

    def update_field_level_interpretation_aggregates(
        self, current_omr_response, field, field_type_detector
    ):
        field_label = field.field_label
        field_label_wise_interpretation_aggregates = (
            self.file_level_interpretation_aggregates[
                "field_label_wise_interpretation_aggregates"
            ]
        )

        field_level_interpretation_aggregates = (
            field_type_detector.get_field_level_interpretation_aggregates()
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

        field_detection_type = field_type_detector.field_detection_type

        field_detection_type_level_interpretation_aggregates = (
            self.directory_level_aggregates[
                "field_detection_type_level_interpretation_aggregates"
            ][field_detection_type]
        )
        # Update the processed field count for that detector
        field_detection_type_level_interpretation_aggregates["fields_count"].push(
            "processed"
        )

    def get_file_level_interpretation_aggregates(self):
        return self.file_level_interpretation_aggregates

    def update_file_level_interpretation_aggregates(
        self, file_path, current_omr_response
    ):
        # Updating field interpretation type level aggregates
        field_detection_type_wise_interpretation_aggregates = {}
        for field_type_detector in self.field_type_detectors.values():
            # Note: This would contain confidence_metrics_for_file
            field_type_detector.finalize_file_level_interpretation_aggregates(file_path)
            field_detection_type_wise_interpretation_aggregates[
                field_type_detector.field_detection_type
            ] = field_type_detector.get_file_level_interpretation_aggregates()

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
