import os
import numpy as np
from src.algorithm.detection.barcode_qr.barcode_qr_detector import BarcodeQRDetector
from src.algorithm.detection.bubbles_blob.bubbles_blob_detector import BubblesBlobDetector
from src.algorithm.detection.bubbles_threshold.bubbles_threshold_detector import BubblesThresholdDetector
from src.algorithm.detection.ocr.ocr_detector import OCRDetector
from src.processors.constants import FieldDetectionType


"""
Template Detector takes care of detections of an image file using a single template
We create one instance of TemplateDetector per Template. 
Note: a Template may get reused for multiple directories(in nested case)
"""
class TemplateDetector:
    # Currently two passes are enough for all detectors to work
    NUM_PASSES = 2

    detectors_map = {
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
    
    def get_detector_instance(self, field_type):
        return TemplateDetector.detectors_map[field_type]()

    def prepare_field_type_detectors(self):
        template = self.template
        # TODO: create instances of all required field type detectors
        self.field_type_detectors = {
            field_type: self.get_detector_instance(field_type) for field_type in template.all_field_types
        }
        self.all_field_detectors = [(field, self.field_type_detectors[field.field_type]) for field in template.all_fields]


    def reset_directory_level_aggregates(self, initial_directory_path):
        self.directory_level_aggregates = {
            "directory_level": {
                "initial_directory_path": initial_directory_path,
                "processed_files_count": 0,
                "multi_marked_count": 0,
            },
            "field_type_level": {
                # [BUBBLES_THRESHOLD] : {}
                # [BARCODE_QR]: {}
            },
            "file_level": {
                # filepath: {
                # "all_fields_threshold_for_file"
                # "read_response_flags": []
                # }
            },
        }
        for field_type_detector in self.field_type_detectors.values():
            field_type_detector.setup_directory_metrics()

    def read_omr_and_update_metrics(self, file_path, gray_image, colored_image):
        # First pass to compute aggregates like global threshold
        # TODO: populate local thresholds even in first pass? (to enable multiple passes)
        
        self.reset_file_level_aggregates(file_path)
        
        for pass_index in range(TemplateDetector.NUM_PASSES):
            omr_response = self.run_detection_pass(
                pass_index, gray_image, colored_image
            )

        # self.update_file_level_validation(omr_response)
        
        # TODO: call from parents?
        self.update_directory_level_metrics(file_path, omr_response)
        
        file_level_aggregates = self.file_level_aggregates

        return omr_response, file_level_aggregates
    
    def reset_file_level_aggregates(self):
        self.file_level_aggregates = {}

        # Setup field type wise metrics
        for field_type_detector in self.field_type_detectors.values():
            field_type_detector.reset_file_level_aggregates()


    def reset_field_type_level_aggregates(self):
        for field_type_detector in self.field_type_detectors.values():
            field_type_detector.reset_field_type_level_aggregates()
    
    def get_file_level_aggregates_for_pass(self, pass_index):
        if(pass_index == -1):
            return {}
        return self.file_level_aggregates[pass_index]
    
    def set_file_level_aggregates(self, pass_index, current_file_level_aggregates):
        self.file_level_aggregates[pass_index] = current_file_level_aggregates

    def run_detection_pass(self, pass_index, gray_image, colored_image):
        self.reset_field_type_level_aggregates()
        # For fallbacks
        previous_file_level_aggregates = self.get_file_level_aggregates_for_pass(pass_index - 1)

        read_response_flags = {
            "multi_marked": False,
            "multi_marked_fields": {},
            "multi_id": False,
        }

        current_omr_response = {}
        for field, field_detector in self.all_field_detectors:
            # reading + detection + metrics calculation
            detected_string = field_detector.detect_and_update_field_level_aggregates(previous_file_level_aggregates, gray_image, colored_image)
            
            field_label = field.field_label

            # Run validation before updating omr response
            self.update_validation_for_field(pass_index, current_omr_response, field_label, detected_string)

            current_omr_response[field_label] = detected_string
            
        # Updating aggregates 
        field_level_aggregates = {}
        for field, field_detector in self.all_field_detectors:
            field_level_aggregates[field.field_label] = field_detector.get_field_type_level_aggregates()
        
        field_type_level_aggregates = {}
        for field_type_detector in self.field_type_detectors.values():
            field_type_level_aggregates[field_type_detector.type] = field_type_detector.get_field_type_level_aggregates()
        
        current_file_level_aggregates = {
            "read_response_flags": read_response_flags,
            "field_level_aggregates": field_level_aggregates,
            "field_type_level_aggregates": field_type_level_aggregates,
        }

        self.set_file_level_aggregates(pass_index, current_file_level_aggregates)

        return current_omr_response
    
    def update_validation_for_field(
        self, read_response_flags, current_omr_response, field_label
    ):
        # TODO: define identifier_labels
        if field_label in current_omr_response:
            if field_label in self.template.identifier_labels:
                read_response_flags["multi_id"] = True

        # TODO: evaluate more validations here


    def update_directory_level_metrics(
        self,
        file_path,
        file_aggregate_params,
        omr_response,
    ):
        # TODO: get_file_level_confidence_metrics()
        # TODO: for plotting underconfident_fields = filter(lambda x: x.confidence < 0.8, fields_confidence)

        # TODO add metrics from other FieldDetectionTypes too (loop)

        file_wise_aggregates, directory_aggregates = map(
            self.field_type_detectors[
                FieldDetectionType.BUBBLES_THRESHOLD
            ].directory_level_aggregates.get,
            ["file_wise_aggregates", "directory_aggregates"],
        )
        file_wise_aggregates[file_path] = file_aggregate_params

        all_fields_threshold_for_file, read_response_flags = map(
            file_aggregate_params.get,
            ["all_fields_threshold_for_file", "read_response_flags"],
        )

        if read_response_flags["multi_marked"]:
            directory_aggregates["multi_marked_count"] += 1

        directory_aggregates["omr_thresholds_sum"] += all_fields_threshold_for_file

        # To be used as a better fallback!
        directory_aggregates["running_omr_threshold"] = (
            directory_aggregates["omr_thresholds_sum"]
            / directory_aggregates["processed_files_count"]
        )

        directory_aggregates["per_omr_thresholds"].append(all_fields_threshold_for_file)

    def get_omr_metrics_for_file(self, file_path):
        (
            multi_marked,
            field_number_to_field_bubble_means,
            all_fields_threshold_for_file,
            confidence_metrics_for_file,
            field_wise_means_and_refs
        ) = self.aggregate_params[file_path]
        # TODO: update according to the finalized aggregates schema
        return {
            "template": template,
            multi_marked,
            field_number_to_field_bubble_means,
            all_fields_threshold_for_file,
            confidence_metrics_for_file,
        }

    def finalize_directory_metrics(self):
        # TODO: get_directory_level_confidence_metrics()

        output_metrics = self.directory_level_aggregates
        per_omr_thresholds = output_metrics["directory_aggregates"][
            "per_omr_thresholds"
        ]
        per_omr_threshold_avg = np.mean(per_omr_thresholds)
        output_metrics["directory_aggregates"][
            "per_omr_threshold_avg"
        ] = per_omr_threshold_avg
