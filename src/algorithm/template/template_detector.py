import os
import numpy as np
from src.algorithm.detection.barcode_qr.barcode_qr_detector import BarcodeQRDetector
from src.algorithm.detection.bubbles_blob.bubbles_blob_detector import BubblesBlobDetector
from src.algorithm.detection.bubbles_threshold.bubbles_threshold_detector import BubblesThresholdDetector
from src.algorithm.detection.ocr.ocr_detector import OCRDetector
from src.processors.constants import FieldDetectionType
from src.utils.parsing import default_dump



class StatsByLabel:
    def __init__(self, *labels):
        self.label_counts = {
            label: 0 for label in labels
        }
    
    def push(self, label, number = 1):
        if label not in self.label_counts:
            raise Exception(f"Unknown label passed to stats by label: {label}, allowed labels: {self.label_counts.keys()}")
            # self.label_counts[label] = []

        self.label_counts[label].push(number)
    
    def to_json(self):
        return {
            key: default_dump(getattr(self, key))
            for key in [
                "label_counts",
            ]
        }
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
        # TODO: create instances of all required field type detectors
        self.field_type_detectors = {
            field_type: self.get_field_detector_instance(field_type) for field_type in template.all_field_detection_types
        }
        self.all_field_detectors = [(field, self.field_type_detectors[field.field_type]) for field in template.all_fields]
  
    def get_field_detector_instance(self, field_type):
        config = self.template.tuning_config
        # Note: detector level config would be passed at runtime! Because aggregates need to happen per field type
        # Instances per different config may need to be created.
        # That's why "reader" may be multiple, but "detector" will be single per directory
            # Note: field object can have the corresponding runtime config for the detection
        FieldDetectorClass = TemplateDetector.field_type_to_detector[field_type]
        return FieldDetectorClass(config)

    def reset_directory_level_aggregates(self, initial_directory_path):
        self.directory_level_aggregates = {
            "directory_level": {
                "initial_directory_path": initial_directory_path,
                "files_count": StatsByLabel("processed", "multi_marked")
            },
            "field_detection_type_wise": {
                # This count would get different in conditional sets
                field_type: {
                    "files_count": StatsByLabel("processed"),
                    # More get added from the field type
                } for field_type in self.template.all_field_detection_types
            },
            "file_level": {
                # filepath: {
                # "field_detection_type_wise"
                # "read_response_flags": []
                # }
            },
        }
        for field_type_detector in self.field_type_detectors.values():
            field_type_detector.reset_directory_level_aggregates()

    def read_omr_and_update_metrics(self, file_path, gray_image, colored_image):
        # First pass to compute aggregates like global threshold
        # TODO: populate local thresholds even in first pass? (to enable multiple passes)
        
        
        # populate detections
        self.run_file_level_detection(
            file_path, gray_image, colored_image
        )

        # populate interpretations
        omr_response = self.run_file_level_interpretation(
            file_path, gray_image, colored_image
        )

        # self.update_file_level_validation(omr_response)
        
        # TODO: call from parents?
        self.update_directory_aggregates_for_file(file_path, omr_response)
        
        file_level_aggregates = self.file_level_interpretation_aggregates

        return omr_response, file_level_aggregates
    

    # def reset_field_detection_type_wise_aggregates(self):
    #     for field_type_detector in self.field_type_detectors.values():
    #         field_type_detector.reset_field_detection_type_wise_aggregates()
    
    def run_file_level_detection(self, file_path, gray_image, colored_image):
        self.reinitialize_file_level_detection_aggregates(file_path)
        
        # Setup field type wise metrics
        for field_type_detector in self.field_type_detectors.values():
            field_type_detector.reinitialize_file_level_detection_aggregates(file_path)

        for field, field_type_detector in self.all_field_detectors:
            # Note: field object can have the corresponding runtime config for the detection
            # reading + detection
            field_type_detector.run_field_level_detection(field, gray_image, colored_image)
        
        # Updating aggregates 
        field_detection_type_wise_aggregates = {}
        for field_type_detector in self.field_type_detectors.values():
            field_detection_type_wise_aggregates[field_type_detector.detection_type] = field_type_detector.generate_file_level_detection_aggregates(file_path)
            
        # field_level_aggregates = {}
        # for field, field_detector in self.all_field_detectors:
        #     field_level_aggregates[field.field_label] = field_detector.get_field_level_aggregates()
        
        self.file_level_detection_aggregates = {
            # "field_level_aggregates": field_level_aggregates,
            "field_detection_type_wise_aggregates": field_detection_type_wise_aggregates,
        }

    def reinitialize_file_level_detection_aggregates(self,file_path):
        self.file_level_detection_aggregates = {"file_path": file_path}

    def update_validation_for_field(
        self, read_response_flags, current_omr_response, field_label
    ):
        # TODO: define identifier_labels
        if field_label in current_omr_response:
            if field_label in self.template.identifier_labels:
                read_response_flags["multi_id"] = True

        # TODO: evaluate more validations here

    def run_file_level_interpretation(self, file_path, gray_image, colored_image):
        self.reinitialize_file_level_interpretation_aggregates(file_path)

        # For fallbacks
        all_file_level_detection_aggregates = self.file_level_detection_aggregates

        # Setup field type wise metrics
        for field_type_detector in self.field_type_detectors.values():
            field_type_detector.reinitialize_file_level_interpretation_aggregates(file_path, all_file_level_detection_aggregates)

        file_level_interpretation_aggregates = {
            "read_response_flags": {
            "multi_marked": False,
            "multi_marked_fields": {},
            "multi_id": False,
        },
            "field_level_aggregates": None,
            "field_detection_type_wise_aggregates": None,
        }

        current_omr_response = {}
        for field, field_detector in self.all_field_detectors:
            # reading + detection + metrics calculation
            detected_string = field_detector.run_field_level_interpretation(field, gray_image, colored_image)
            
            field_label = field.field_label

            # Run validation before updating omr response
            self.update_validation_for_field(field_label, file_level_interpretation_aggregates, current_omr_response, detected_string)

            current_omr_response[field_label] = detected_string

        # Updating aggregates 
        field_detection_type_wise_aggregates = {}
        for field_type_detector in self.field_type_detectors.values():
            interpretation_aggregates = field_type_detector.generate_file_level_interpretation_aggregates(file_path)
            field_detection_type_wise_aggregates[field_type_detector.detection_type] = interpretation_aggregates
            
        # field_level_aggregates = {}
        # for field, field_detector in self.all_field_detectors:
        #     field_level_aggregates[field.field_label] 
        file_level_interpretation_aggregates = {
            "file_path": file_path,
            "read_response_flags": read_response_flags,
        }

        self.file_level_interpretation_aggregates = file_level_interpretation_aggregates

        return current_omr_response

    def reinitialize_file_level_interpretation_aggregates(self, file_path):
        self.file_level_interpretation_aggregates = {"file_path": file_path}

    def update_directory_aggregates_for_file(
        self,
        file_path,
        file_aggregate_params,
        omr_response,
    ):
        self.file_level_detection_aggregates
        self.file_level_interpretation_aggregates
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

        if file_aggregate_params["read_response_flags"]["multi_marked"]:
            # TODO: directory_aggregates["files_count"].push("multi_marked")
            directory_aggregates["multi_marked_count"] += 1

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
