import json
from pathlib import Path

from src.algorithm.template.detection.template_file_runner import TemplateFileRunner
from src.algorithm.template.directory_handler import TemplateDirectoryHandler
from src.algorithm.template.layout.template_drawing import TemplateDrawing
from src.algorithm.template.layout.template_layout import TemplateLayout
from src.processors.constants import FieldDetectionType
from src.utils.file import PathUtils, SaveImageOps
from src.utils.image import ImageUtils
from src.utils.logger import logger

"""
The main interface for interacting with all template json related operations
"""


class Template:
    def __init__(self, template_path, tuning_config) -> None:
        # TODO: load template json file at this level?
        self.tuning_config = tuning_config
        # template_json =
        self.save_image_ops = SaveImageOps(tuning_config)
        self.template_layout = TemplateLayout(self, template_path, tuning_config)
        # TODO: find a better way to couple these
        self.path = self.template_layout.path
        self.alignment = self.template_layout.alignment
        self.all_fields = self.template_layout.all_fields
        self.global_empty_val = self.template_layout.global_empty_val
        self.output_columns = self.template_layout.output_columns
        self.all_field_detection_types = self.template_layout.all_field_detection_types

        # re-export references for external use
        self.field_blocks = self.template_layout.field_blocks
        # TODO: see if get_concatenated_omr_response should move to template_file_runner instead
        self.get_concatenated_omr_response = (
            self.template_layout.get_concatenated_omr_response
        )
        self.template_dimensions = self.template_layout.template_dimensions
        # TODO: check Traits pattern?
        self.drawing = TemplateDrawing(self)

        self.template_file_runner = TemplateFileRunner(self)
        self.directory_handler = TemplateDirectoryHandler(self)

    # TODO: move some other functions here
    def apply_preprocessors(self, file_path, gray_image, colored_image):
        (
            gray_image,
            colored_image,
            next_template_layout,
        ) = self.template_layout.apply_preprocessors(
            file_path, gray_image, colored_image
        )
        self.template_layout = next_template_layout
        # TODO: decide how shallow copy is handled now.
        next_template = self
        return gray_image, colored_image, next_template

    def __str__(self) -> str:
        return str(self.path)

    def reset_and_setup_for_directory(self, output_dir) -> None:
        """Reset all mutations to the template and setup output directories."""
        self.template_layout.reset_all_shifts()
        self.reset_and_setup_outputs(output_dir)

    def reset_and_setup_outputs(self, output_dir) -> None:
        output_mode = self.tuning_config.outputs.output_mode
        self.directory_handler.reset_path_utils(output_dir, output_mode)

    def get_exclude_files(self):
        excluded_files = self.template_layout.get_exclude_files()

        for pp in self.get_pre_processors():
            excluded_files.extend(p for p in pp.exclude_files())

        return excluded_files

    # TODO: move consumers of this function inside
    def get_pre_processors(self):
        # return self.template_preprocessing.pre_processors
        return self.template_layout.pre_processors

    def get_pre_processor_names(self):
        return [pp.__class__.__name__ for pp in self.get_pre_processors()]

    # TODO: reduce the number of these getter
    def get_processing_image_shape(self):
        return self.template_layout.processing_image_shape

    def get_empty_response_array(self):
        return self.directory_handler.get_empty_response_array()

    def append_output_omr_response(self, _file_name, output_omr_response):
        return [
            output_omr_response[field]
            for field in self.directory_handler.omr_response_columns
        ]

    def get_results_file(self):
        return self.directory_handler.output_files["Results"]

    # TODO: replace these utils with more dynamic flagged files (FileStatsByLabel?)
    def get_multi_marked_file(self):
        return self.directory_handler.output_files["MultiMarked"]

    def get_errors_file(self):
        return self.directory_handler.output_files["Errors"]

    def finish_processing_directory(self):
        self.directory_handler.finish_processing_directory()
        return self.template_file_runner.finish_processing_directory()

    def get_save_marked_dir(self):
        return self.directory_handler.path_utils.save_marked_dir

    def get_multi_marked_dir(self):
        return self.directory_handler.path_utils.multi_marked_dir

    def get_errors_dir(self):
        return self.directory_handler.path_utils.errors_dir

    def get_evaluations_dir(self):
        return self.directory_handler.path_utils.evaluations_dir

    def read_omr_response(self, input_gray_image, colored_image, file_path):
        # Convert posix path to string
        file_path = str(file_path)

        # Note: resize also creates a copy
        gray_image, colored_image = ImageUtils.resize_to_dimensions(
            self.template_dimensions, input_gray_image, colored_image
        )
        # Resize to template dimensions for saved outputs
        self.save_image_ops.append_save_image(
            "Resized Image", range(3, 7), gray_image, colored_image
        )

        gray_image, colored_image = ImageUtils.normalize(gray_image, colored_image)

        raw_omr_response = self.template_file_runner.read_omr_and_update_metrics(
            file_path, gray_image, colored_image
        )

        concatenated_omr_response = self.get_concatenated_omr_response(raw_omr_response)

        return concatenated_omr_response, raw_omr_response

    # TODO: move inside template runner
    def get_omr_metrics_for_file(self, file_path):
        # This can be used for drawing the bubbles etc
        directory_level_interpretation_aggregates = (
            self.template_file_runner.get_directory_level_interpretation_aggregates()
        )

        template_file_level_interpretation_aggregates = (
            directory_level_interpretation_aggregates["file_wise_aggregates"][file_path]
        )

        is_multi_marked = template_file_level_interpretation_aggregates[
            "read_response_flags"
        ]["is_multi_marked"]

        field_id_to_interpretation = template_file_level_interpretation_aggregates[
            "field_id_to_interpretation"
        ]
        return is_multi_marked, field_id_to_interpretation

    # TODO: figure out a structure to output directory metrics apart from from this file one.
    # directory_metrics_path = self.path_utils.image_metrics_dir.joinpath()
    # def export_omr_metrics_for_directory()
    def export_omr_metrics_for_file(
        self, file_path, evaluation_meta, field_id_to_interpretation
    ) -> None:
        # TODO: move these inside self.template_file_runner.get_export_omr_metrics_for_file
        # This can be used for drawing the bubbles etc
        directory_level_interpretation_aggregates = (
            # TODO: get from file_level_interpretation_aggregates.field_detection_runner?
            self.template_file_runner.get_directory_level_interpretation_aggregates()
        )

        template_file_level_interpretation_aggregates = (
            directory_level_interpretation_aggregates["file_wise_aggregates"][file_path]
        )

        file_name = PathUtils.remove_non_utf_characters(file_path.name)

        is_multi_marked = template_file_level_interpretation_aggregates[
            "read_response_flags"
        ]["is_multi_marked"]

        bubbles_threshold_interpretation_aggregates = (
            template_file_level_interpretation_aggregates[
                "field_detection_type_wise_aggregates"
            ][FieldDetectionType.BUBBLES_THRESHOLD]
        )

        confidence_metrics_for_file = bubbles_threshold_interpretation_aggregates[
            "confidence_metrics_for_file"
        ]
        file_level_fallback_threshold = bubbles_threshold_interpretation_aggregates[
            "file_level_fallback_threshold"
        ]

        field_wise_means_and_refs = []
        # self.all_fields contains fields in the reference order
        for field in self.all_fields:
            bubble_interpretations = field_id_to_interpretation[field.field_label]
            field_wise_means_and_refs.extend(bubble_interpretations)
        # sorted_global_bubble_means_and_refs = sorted(field_wise_means_and_refs)

        image_metrics_path = self.path_utils.image_metrics_dir.joinpath(
            f"metrics-{file_name}.js"
        )

        template_meta = {
            # "template": template,
            "is_multi_marked": is_multi_marked,
            "field_id_to_interpretation": field_id_to_interpretation,
            "file_level_fallback_threshold": file_level_fallback_threshold,
            "confidence_metrics_for_file": confidence_metrics_for_file,
        }
        # evaluation_meta = evaluation.get_omr_metrics_for_file()

        image_metrics = {
            "template_meta": template_meta,
            "evaluation_meta": (evaluation_meta if evaluation_meta is not None else {}),
        }
        with Path.open(
            image_metrics_path,
            "w",
        ) as f:
            json_string = json.dumps(
                image_metrics,
                default=lambda x: x.to_json(),
                indent=4,
            )
            f.write(f"export default {json_string}")
            logger.info(f"Exported image metrics to: {image_metrics_path}")
