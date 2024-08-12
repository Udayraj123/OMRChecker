from src.processors.constants import FieldDetectionType


class TemplateDetector:
    def __init__(self, template):
        self.template = template
        self.prepare_field_type_detectors()
        self.prepare_output_metrics()

    def prepare_output_metrics(self):
        self.output_metrics = {
            "directory_aggregates": {
                "processed_files_count": 0,
                "multi_marked_count": 0,
            },
            "file_wise_aggregates": {
                # filepath: {
                # "all_fields_threshold_for_file"
                # "read_response_flags": []
                # }
            },
        }
        for field_type_detector in self.field_type_detectors.values():
            field_type_detector.setup_directory_metrics(
                self.output_metrics["directory_aggregates"]
            )

    def prepare_field_type_detectors(self):
        template = self.template
        # TODO: create instances of all required field type detectors
        self.field_type_detectors = {
            FieldDetectionType.BUBBLES_THRESHOLD: BubblesThresholdDetector()
        }

    def read_omr_and_update_metrics(self, file_path, gray_image, colored_image):
        # First pass to compute aggregates like global threshold
        # TODO: populate local thresholds even in first pass?
        # for field_detector in self.all_field_detectors:
        self.prepare_metrics_for_file(file_path)
        file_aggregate_params = self.populate_file_aggregate_params(
            gray_image, colored_image
        )
        omr_response = {}

        # Second pass with computed aggregates
        for field_detector in self.all_field_detectors:
            field_label = field_detector.field_label
            final_field_interpretation = field_detector.get_field_interpretation(
                gray_image, colored_image, file_aggregate_params
            )

            final_field_interpretation.update_field_level_aggregates(
                file_aggregate_params, omr_response
            )
            self.update_file_level_aggregates(
                file_aggregate_params, final_field_interpretation, omr_response
            )

            omr_response[field_label] = final_field_interpretation.get_detected_string()

        self.update_metrics_for_file(file_path, file_aggregate_params, omr_response)

        return omr_response, file_aggregate_params

    def update_file_level_aggregates(
        self, file_aggregate_params, field_interpretation, omr_response
    ):
        field_label = field_interpretation.field_label
        confidence_metrics = field_interpretation.get_field_level_confidence_metrics()

        file_aggregate_params["confidence_metrics_for_file"][
            field_label
        ] = confidence_metrics

        # TODO: uncomment this
        # if field_label in omr_response:
        #     if field_label in self.template.identifier_labels:
        #         file_aggregate_params["read_response_flags"]["multi_id"] = True

        # TODO: evaluate more validations here

    def populate_file_aggregate_params(self, gray_image, colored_image):
        for field_type_detector in self.field_type_detectors.values():
            field_type_detector.reset_field_type_aggregates(
                file_aggregate_params, gray_image, colored_image
            )

        for field in self.all_fields:
            field_detector = self.field_type_detectors[field.type]
            # TODO: debug the f
            field_label = field_detector.field_label
            initial_field_interpretation = field_detector.get_field_interpretation(
                field, gray_image, colored_image
            )
            field_detector.update_field_type_aggregates(
                field, initial_field_interpretation
            )

        # For each fieldReaderType we need to populate aggregates
        for field_type_detector in self.field_type_detectors.values():
            field_type_detector.populate_field_type_aggregates(
                file_aggregate_params, gray_image, colored_image
            )

        file_aggregate_params = {
            "read_response_flags": {
                "multi_marked": False,
                "multi_marked_fields": {},
                "multi_id": False,
            },
            # >> TODO: move inside bubble_thresholding -
            "first_pass": {
                "global_bubble_means_and_refs": global_bubble_means_and_refs,
                "all_outlier_deviations": all_outlier_deviations,
            },
        }
        self.file_aggregate_params = file_aggregate_params

        return file_aggregate_params

    def get_metrics_for_file(self, file_path):
        (
            multi_marked,
            field_number_to_field_bubble_means,
            all_fields_threshold_for_file,
            confidence_metrics_for_file,
        ) = self.aggregate_params[file_path]
        # TODO: update according to the finalized aggregates schema
        return (
            multi_marked,
            field_number_to_field_bubble_means,
            all_fields_threshold_for_file,
            confidence_metrics_for_file,
        )

    def update_metrics_for_file(
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
            ].output_metrics.get,
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

    def finalize_directory_metrics(self):
        # TODO: get_directory_level_confidence_metrics()

        output_metrics = self.output_metrics
        per_omr_thresholds = output_metrics["directory_aggregates"][
            "per_omr_thresholds"
        ]
        per_omr_threshold_avg = np.mean(per_omr_thresholds)
        output_metrics["directory_aggregates"][
            "per_omr_threshold_avg"
        ] = per_omr_threshold_avg

    def get_omr_metrics_for_file(self):
        # TODO: use self as much as possible
        return {
            "all_fields_threshold_for_file": all_fields_threshold_for_file,
            "template": template,
            "global_bubble_means_and_refs": global_bubble_means_and_refs,
            "confidence_metrics_for_file": confidence_metrics_for_file,
        }
