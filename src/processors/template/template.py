import json
from pathlib import Path

from src.processors import ProcessingPipeline
from src.processors.constants import FieldDetectionType
from src.processors.layout.template_drawing import TemplateDrawing
from src.processors.layout.template_layout import TemplateLayout
from src.processors.template.directory_handler import TemplateDirectoryHandler
from src.utils.file import PathUtils, SaveImageOps
from src.utils.logger import logger

"""
The main interface for interacting with all template json related operations
"""


class Template:
    def __init__(self, template_path, tuning_config, args: dict | None = None) -> None:
        # TODO: load template json file at this level?
        self.tuning_config = tuning_config
        self.args = args or {}
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

        self.directory_handler = TemplateDirectoryHandler(self)

        # Initialize the unified processor pipeline
        # Note: TemplateFileRunner is now instantiated in ReadOMRProcessor
        self.pipeline = ProcessingPipeline(self, args=self.args)

    # TODO: move some other functions here
    def __str__(self) -> str:
        return str(self.path)

    def reset_and_setup_for_directory(self, output_dir: Path) -> None:
        """Reset all mutations to the template and setup output directories."""
        self.template_layout.reset_all_shifts()
        self.reset_and_setup_outputs(output_dir)

    def reset_and_setup_outputs(self, output_dir: Path) -> None:
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
        """Finish processing directory and get aggregated results.

        Note: This delegates to the ReadOMRProcessor via the pipeline.
        """
        self.directory_handler.finish_processing_directory()
        # Get the ReadOMRProcessor from the pipeline
        read_omr_processor = self.pipeline.processors[-1]  # Last processor
        return read_omr_processor.finish_processing_directory()

    def get_save_marked_dir(self):
        return self.directory_handler.path_utils.save_marked_dir

    def get_multi_marked_dir(self):
        return self.directory_handler.path_utils.multi_marked_dir

    def get_errors_dir(self):
        return self.directory_handler.path_utils.errors_dir

    def get_evaluations_dir(self):
        return self.directory_handler.path_utils.evaluations_dir

    def process_file(self, file_path, gray_image, colored_image):
        """Process a file through the entire pipeline (preprocessing, alignment, detection).

        Uses the unified processor-based pipeline where all processors work with
        the same ProcessingContext interface.

        Args:
            file_path: Path to the file being processed
            gray_image: Input grayscale image
            colored_image: Input colored image

        Returns:
            ProcessingContext containing omr_response, is_multi_marked, and all other results
        """
        return self.pipeline.process_file(file_path, gray_image, colored_image)

    def get_omr_metrics_for_file(self, file_path, context=None):
        """Get OMR metrics for a file from the processing context.

        Args:
            file_path: Path to the file
            context: ProcessingContext from pipeline (if available)

        Returns:
            Tuple of (is_multi_marked, field_id_to_interpretation)
        """
        if context is not None:
            # Get from context if available (preferred)
            return context.is_multi_marked, context.field_id_to_interpretation

        # Fallback: get from ReadOMRProcessor
        read_omr_processor = self.pipeline.processors[-1]
        directory_level_interpretation_aggregates = read_omr_processor.template_file_runner.get_directory_level_interpretation_aggregates()

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

    def export_omr_metrics_for_file(
        self, file_path, evaluation_meta, field_id_to_interpretation, context=None
    ) -> None:
        """Export OMR metrics for a file.

        Args:
            file_path: Path to the file
            evaluation_meta: Evaluation metadata
            field_id_to_interpretation: Field interpretations
            context: ProcessingContext from pipeline (optional, for accessing aggregates)
        """
        # Get aggregates from context if available, otherwise from ReadOMRProcessor
        if (
            context is not None
            and "directory_level_interpretation_aggregates" in context.metadata
        ):
            directory_level_interpretation_aggregates = context.metadata[
                "directory_level_interpretation_aggregates"
            ]
        else:
            # Fallback: get from ReadOMRProcessor
            read_omr_processor = self.pipeline.processors[-1]
            directory_level_interpretation_aggregates = read_omr_processor.template_file_runner.get_directory_level_interpretation_aggregates()

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
