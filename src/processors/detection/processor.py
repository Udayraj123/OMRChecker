"""ReadOMR Processor for OMR detection and interpretation."""

import threading
from pathlib import Path

from src.processors.base import ProcessingContext, Processor
from src.processors.detection.template_file_runner import TemplateFileRunner
from src.utils.image import ImageUtils
from src.utils.logger import logger


class ReadOMRProcessor(Processor):
    """Processor that performs OMR detection and interpretation.

    This processor:
    1. Creates a TemplateFileRunner for the template
    2. Resizes images to template dimensions
    3. Normalizes the images
    4. Runs field detection (bubbles, OCR, barcodes)
    5. Interprets the detected data
    6. Optionally uses ML fallback for low-confidence detections
    7. Stores results in context
    """

    def __init__(self, template, ml_model_path: str | Path | None = None) -> None:
        """Initialize the ReadOMR processor.

        Args:
            template: The template containing field definitions and layout
            ml_model_path: Optional path to trained ML model for fallback detection
        """
        self.template = template
        self.tuning_config = template.tuning_config

        # Instantiate the TemplateFileRunner here instead of in Template
        # This decouples Template from processing logic
        self.template_file_runner = TemplateFileRunner(template)

        # Reentrant lock so that process_single_file and process() can both
        # acquire it from the same thread without deadlocking.  This serialises
        # access to template_file_runner aggregates and save_image_ops across
        # threads when parallel processing is enabled.
        self._lock = threading.RLock()

        # Optional ML fallback
        self.ml_detector = None
        self.hybrid_strategy = None
        if ml_model_path:
            self._initialize_ml_fallback(ml_model_path)

    def _initialize_ml_fallback(self, ml_model_path: str | Path) -> None:
        """Initialize ML fallback detector.

        Args:
            ml_model_path: Path to trained ML model
        """
        try:
            from src.processors.detection.ml_detector import (
                HybridDetectionStrategy,
                MLBubbleDetector,
            )

            self.ml_detector = MLBubbleDetector(ml_model_path)

            # Initialize hybrid strategy
            confidence_threshold = (
                getattr(self.tuning_config.ml, "confidence_threshold", 0.75)
                if hasattr(self.tuning_config, "ml")
                else 0.75
            )

            self.hybrid_strategy = HybridDetectionStrategy(
                self.ml_detector, confidence_threshold=confidence_threshold
            )

            logger.info(f"ML fallback enabled with model: {ml_model_path}")

        except Exception as e:
            logger.warning(f"Failed to initialize ML fallback: {e}")
            self.ml_detector = None
            self.hybrid_strategy = None

    def get_name(self) -> str:
        """Get the name of this processor."""
        return "ReadOMR"

    def finish_processing_directory(self):
        """Finish processing directory and get aggregated results."""
        results = self.template_file_runner.finish_processing_directory()

        # Log ML fallback statistics if enabled
        if self.hybrid_strategy:
            stats = self.hybrid_strategy.get_statistics()
            logger.info("=" * 60)
            logger.info("ML Fallback Statistics")
            logger.info("=" * 60)
            logger.info(f"Total fields processed: {stats['total_fields']}")
            logger.info(f"High confidence fields: {stats['high_confidence_fields']}")
            logger.info(f"Low confidence fields: {stats['low_confidence_fields']}")
            logger.info(f"ML fallback used: {stats['ml_fallback_used']} times")
            logger.info("=" * 60)

        return results

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Execute OMR detection and interpretation.

        Args:
            context: Processing context with preprocessed and aligned images

        Returns:
            Updated context with OMR response and interpretation metrics
        """
        logger.debug(f"Starting {self.get_name()} processor")

        # Check if shift detection already ran and populated results
        shift_detection_meta = context.metadata.get("shift_detection")

        if shift_detection_meta:
            # Shifts already applied and validated, results already in context
            logger.debug(
                "Using shift-validated detection results from ShiftDetectionProcessor"
            )
            return context

        template = context.template
        file_path = context.file_path
        input_gray_image = context.gray_image
        colored_image = context.colored_image

        # Acquire the per-processor lock so that concurrent threads cannot
        # interleave writes to template_file_runner aggregates or
        # save_image_ops.  process_single_file also acquires this lock before
        # reset_all_save_img() so the whole sequence is atomic per file.
        with self._lock:
            # Resize to template dimensions for saved outputs
            gray_image, colored_image = ImageUtils.resize_to_dimensions(
                template.template_dimensions, input_gray_image, colored_image
            )

            # Save resized image
            template.save_image_ops.append_save_image(
                "Resized Image", range(3, 7), gray_image, colored_image
            )

            # Normalize images
            gray_image, colored_image = ImageUtils.normalize(gray_image, colored_image)

            # Run detection and interpretation via TemplateFileRunner
            raw_omr_response = self.template_file_runner.read_omr_and_update_metrics(
                file_path, gray_image, colored_image
            )

            # Get concatenated response (handles custom labels)
            concatenated_omr_response = template.get_concatenated_omr_response(
                raw_omr_response
            )

            # Extract interpretation metrics
            directory_level_interpretation_aggregates = self.template_file_runner.get_directory_level_interpretation_aggregates()

            template_file_level_interpretation_aggregates = (
                directory_level_interpretation_aggregates["file_wise_aggregates"][
                    file_path
                ]
            )

            is_multi_marked = template_file_level_interpretation_aggregates[
                "read_response_flags"
            ]["is_multi_marked"]

            field_id_to_interpretation = template_file_level_interpretation_aggregates[
                "field_id_to_interpretation"
            ]

            # Update context with results
            context.omr_response = concatenated_omr_response
            context.is_multi_marked = is_multi_marked
            context.field_id_to_interpretation = field_id_to_interpretation
            context.gray_image = gray_image
            context.colored_image = colored_image

            # Store raw response and aggregates in metadata
            context.metadata["raw_omr_response"] = raw_omr_response
            context.metadata["directory_level_interpretation_aggregates"] = (
                directory_level_interpretation_aggregates
            )

            # Check for low-confidence fields and use ML fallback if needed
            if self.hybrid_strategy and self.hybrid_strategy.should_use_ml_fallback(
                context
            ):
                logger.info(
                    f"Using ML fallback for low-confidence fields in {Path(file_path).name}"
                )
                self.ml_detector.enable_for_low_confidence()
                context = self.ml_detector.process(context)
                self.ml_detector.disable()
                self.hybrid_strategy.stats["ml_fallback_used"] += 1

        logger.debug(f"Completed {self.get_name()} processor")

        return context
