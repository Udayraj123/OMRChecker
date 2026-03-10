from typing import ClassVar

from src.processors.constants import FieldDetectionType
from src.processors.detection.barcode.file_runner import BarcodeFileRunner
from src.processors.detection.base.detection_pass import TemplateDetectionPass
from src.processors.detection.base.file_runner import (
    FieldTypeFileLevelRunner,
    FileLevelRunner,
)
from src.processors.detection.base.interpretation_pass import (
    TemplateInterpretationPass,
)
from src.processors.detection.bubbles_threshold.file_runner import (
    BubblesThresholdFileRunner,
)
from src.processors.detection.ocr.file_runner import OCRFileRunner
from src.processors.layout.field.base import Field
from src.processors.detection.detection_repository import DetectionRepository


class TemplateFileRunner(
    FileLevelRunner[TemplateDetectionPass, TemplateInterpretationPass]
):
    """Template File Runner.

    It is responsible for running the file level detection and interpretation steps.
    It maintains own template level runners as well as all the field detection type level runners.
    We create one instance of TemplateFileRunner per Template - thus it is reused for all images mapped to that Template.
    Note: a Template may get reused for multiple directories(in nested case).

    Thread-safety: this class is NOT thread-safe on its own.  All concurrent
    access must be serialised by the caller (ReadOMRProcessor acquires
    ``self._lock`` around every call into this runner).
    """

    field_detection_type_to_runner: ClassVar = {
        FieldDetectionType.BUBBLES_THRESHOLD: BubblesThresholdFileRunner,
        FieldDetectionType.OCR: OCRFileRunner,
        FieldDetectionType.BARCODE_QR: BarcodeFileRunner,
        # FieldDetectionType.BUBBLES_BLOB: BubblesBlobRunner,
    }

    def __init__(self, template) -> None:
        self.template = template
        tuning_config = template.tuning_config
        detection_pass = TemplateDetectionPass(tuning_config)
        interpretation_pass = TemplateInterpretationPass(tuning_config)
        super().__init__(tuning_config, detection_pass, interpretation_pass)
        # Create repository for new typed models pipeline
        self.repository = DetectionRepository()
        self.initialize_field_file_runners(template)
        self.initialize_directory_level_aggregates(template)

    def initialize_field_file_runners(self, template) -> None:
        self.all_fields: list[Field] = template.all_fields
        self.all_field_detection_types = self.template.all_field_detection_types

        # Create instances of all required field type processors
        self.field_detection_type_file_runners = {
            field_detection_type: self.get_field_detection_type_file_runner(
                field_detection_type
            )
            for field_detection_type in self.all_field_detection_types
        }

    def get_field_detection_type_file_runner(
        self, field_detection_type
    ) -> FieldTypeFileLevelRunner:
        tuning_config = self.tuning_config
        # ruff: noqa: N806
        FieldTypeProcessorClass = self.field_detection_type_to_runner[
            field_detection_type
        ]
        # Pass repository to all field type runners (all now support repository)
        return FieldTypeProcessorClass(tuning_config, self.repository)

    def read_omr_and_update_metrics(self, file_path, gray_image, colored_image):
        # First pass to compute aggregates like global threshold

        # populate detections
        self.run_file_level_detection(file_path, gray_image, colored_image)

        # populate interpretations
        return self.run_file_level_interpretation(file_path, gray_image, colored_image)

    # FieldTypeFileLevelRunner::run_field_level_detection
    def run_file_level_detection(self, file_path, gray_image, colored_image) -> None:
        self.initialize_file_level_detection_aggregates(file_path)

        # Perform detection step for each field
        # TODO: see where the conditional sets logic can fit in this loop (or at a wrapper level?)
        for field in self.all_fields:
            self.run_field_level_detection(field, gray_image, colored_image)

        self.update_detection_aggregates_on_processed_file(file_path)

    def run_field_level_detection(
        self, field: Field, gray_image, colored_image
    ) -> None:
        field_detection_type_file_runner = self.field_detection_type_file_runners[
            field.field_detection_type
        ]

        field_detection = field_detection_type_file_runner.run_field_level_detection(
            field, gray_image, colored_image
        )

        # initialize_field_level_aggregates is now called automatically by run_field_level_detection
        self.detection_pass.run_field_level_detection(field, field_detection)

    # Overrides
    def initialize_directory_level_aggregates(self, template) -> None:
        initial_directory_path = template.path.parent

        # super().initialize_directory_level_aggregates(initial_directory_path)

        # Initialize repository for new directory
        self.repository.initialize_directory(str(initial_directory_path))

        self.detection_pass.initialize_directory_level_aggregates(
            initial_directory_path, self.all_field_detection_types
        )
        self.interpretation_pass.initialize_directory_level_aggregates(
            initial_directory_path, self.all_field_detection_types
        )

        for (
            field_detection_type_file_runner
        ) in self.field_detection_type_file_runners.values():
            field_detection_type_file_runner.initialize_directory_level_aggregates(
                initial_directory_path
            )

    def initialize_file_level_detection_aggregates(self, file_path) -> None:
        # super().initialize_file_level_detection_aggregates(file_path)
        # Initialize repository for new file
        self.repository.initialize_file(file_path)

        self.detection_pass.initialize_file_level_aggregates(
            file_path, self.all_field_detection_types
        )

        # Setup field type wise metrics
        for (
            field_detection_type_file_runner
        ) in self.field_detection_type_file_runners.values():
            field_detection_type_file_runner.initialize_file_level_detection_aggregates(
                file_path
            )

    def update_detection_aggregates_on_processed_file(self, file_path) -> None:
        for (
            field_detection_type_file_runner
        ) in self.field_detection_type_file_runners.values():
            field_detection_type_file_runner.update_detection_aggregates_on_processed_file(
                file_path
            )

        # Finalize repository for current file (must be done before populating aggregates)
        self.repository.finalize_file()

        # Note: we update file level after field levels are updated
        # This populates bubble_fields, ocr_fields, barcode_fields from repository
        self.detection_pass.update_aggregates_on_processed_file(
            file_path, self.field_detection_type_file_runners
        )

    def run_file_level_interpretation(self, file_path, _gray_image, _colored_image):
        self.initialize_file_level_interpretation_aggregates(file_path)

        current_omr_response = {}
        # Perform interpretation step for each field
        for field in self.all_fields:
            # Intentional arg mutation
            self.run_field_level_interpretation(field, current_omr_response)

        self.update_interpretation_aggregates_on_processed_file(file_path)

        return current_omr_response

    def run_field_level_interpretation(self, field, current_omr_response) -> None:
        field_detection_type_file_runner = self.field_detection_type_file_runners[
            field.field_detection_type
        ]

        # Get file-level detection aggregates from template-level detection pass
        # (contains bubble_fields, ocr_fields, barcode_fields populated from repository)
        file_level_detection_aggregates = (
            self.detection_pass.get_file_level_aggregates()
        )

        # Run field-level interpretation with template-level aggregates
        # initialize_field_level_aggregates is called automatically inside run_field_level_interpretation
        field_interpretation = field_detection_type_file_runner.interpretation_pass.run_field_level_interpretation(
            field, file_level_detection_aggregates
        )

        field_type_runner_field_level_aggregates = (
            field_detection_type_file_runner.get_field_level_interpretation_aggregates()
        )
        # initialize_field_level_aggregates is called automatically inside run_field_level_interpretation
        self.interpretation_pass.run_field_level_interpretation(
            field,
            field_interpretation,
            field_type_runner_field_level_aggregates,
            current_omr_response,
        )

        current_omr_response[field.field_label] = (
            field_interpretation.get_field_interpretation_string()
        )

    # This overrides parent definition -
    def initialize_file_level_interpretation_aggregates(self, file_path) -> None:
        # Interpretation passes now use repository directly, no need to pass aggregates
        self.interpretation_pass.initialize_file_level_aggregates(
            file_path,
            self.all_field_detection_types,
        )

        # Setup field type wise metrics
        for (
            field_detection_type_file_runner
        ) in self.field_detection_type_file_runners.values():
            field_detection_type_file_runner.initialize_file_level_interpretation_aggregates(
                file_path,
            )

    def update_interpretation_aggregates_on_processed_file(self, file_path) -> None:
        for (
            field_detection_type_file_runner
        ) in self.field_detection_type_file_runners.values():
            field_detection_type_file_runner.update_interpretation_aggregates_on_processed_file(
                file_path
            )

        # Note: we update file level after field levels are updated
        self.interpretation_pass.update_aggregates_on_processed_file(
            file_path, self.field_detection_type_file_runners
        )

    def finish_processing_directory(self) -> None:
        # TODO: get_directory_level_confidence_metrics()

        # output_metrics = self.directory_level_aggregates
        # TODO: update export directory level stats here
        pass

    def get_export_omr_metrics_for_file(self) -> None:
        pass
