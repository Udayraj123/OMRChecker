"""Simplified processing pipeline using unified Processor interface."""

from pathlib import Path

from cv2.typing import MatLike

from src.processors.base import ProcessingContext, Processor
from src.utils.logger import logger


class ProcessingPipeline:
    """Simplified pipeline that orchestrates processors.

    This pipeline provides a clean, testable interface for processing
    OMR images through multiple processors with a unified interface:
    1. Preprocessing (rotation, cropping, filtering)
    2. Alignment (feature-based alignment)
    3. ReadOMR (detection & interpretation)

    Benefits:
    - All processors use the same interface
    - Easy to test each processor independently
    - Simple to extend with new processors
    - Type-safe ProcessingContext
    - Consistent error handling
    """

    def __init__(self, template, args: dict | None = None) -> None:
        """Initialize the pipeline with a template.

        Args:
            template: The template containing all configuration and runners
            args: CLI arguments (for training data collection, ML fallback, etc.)
        """
        self.template = template
        self.tuning_config = template.tuning_config
        self.args = args or {}

        # Lazy import processors to avoid circular dependencies
        # These imports are intentionally not at top-level
        from src.processors.detection.processor import ReadOMRProcessor
        from src.processors.image.coordinator import (
            PreprocessingCoordinator,
        )

        # Check for ML model paths from args
        ml_model_path = self.args.get("ml_model_path")
        field_block_model_path = self.args.get("field_block_model_path")
        use_field_block_detection = self.args.get("use_field_block_detection", False)

        # Initialize core processors
        self.processors: list[Processor] = [
            PreprocessingCoordinator(template),
        ]

        # Add alignment processor if enabled and configured
        if self._should_enable_alignment():
            from src.processors.image.alignment.processor import AlignmentProcessor

            self.processors.append(AlignmentProcessor(template))
            logger.info("Template alignment enabled")

        # Add ML field block detector if enabled (Stage 1)
        if use_field_block_detection and field_block_model_path:
            from src.processors.detection.ml_field_block_detector import (
                MLFieldBlockDetector,
            )

            self.processors.append(
                MLFieldBlockDetector(
                    field_block_model_path,
                    confidence_threshold=self.tuning_config.ml.field_block_confidence_threshold,
                )
            )
            logger.info("ML Field Block Detection (Stage 1) enabled")

            # Add shift detection processor if configured or enabled via CLI
            shift_config = self.tuning_config.ml.shift_detection
            enable_shift_detection = self.args.get("enable_shift_detection", False)

            if shift_config.enabled or enable_shift_detection:
                from src.processors.detection.shift_detection_processor import (
                    ShiftDetectionProcessor,
                )

                self.processors.append(ShiftDetectionProcessor(template, shift_config))
                logger.info("ML-based shift detection enabled")

        # Add traditional + ML bubble detection (Stage 2)
        read_omr = ReadOMRProcessor(template, ml_model_path=ml_model_path)
        self.read_omr_processor = read_omr
        self.processors.append(read_omr)

        # Add experimental training data collector if enabled
        if self._should_enable_training_collection():
            self._add_training_data_collector()

        # Snapshot how many processors belong to the core pipeline.
        # Processors added later (e.g. FileOrganizerProcessor per-directory)
        # are reset between directories via reset_extra_processors().
        self._base_processor_count = len(self.processors)

    def _should_enable_alignment(self) -> bool:
        """Check if alignment processor should be enabled.

        Alignment is enabled if:
        1. Config has alignment enabled (default: True for backward compatibility)
        2. Template has alignment reference image configured

        Returns:
            True if alignment should be enabled
        """
        # Check config flag
        alignment_config = self.tuning_config.alignment
        if not alignment_config.enabled:
            return False

        # Check if template has alignment data
        has_alignment_data = "gray_alignment_image" in self.template.alignment
        return has_alignment_data

    def _should_enable_training_collection(self) -> bool:
        """Check if experimental training data collection should be enabled.

        Training collection is enabled if:
        1. Config has experimental.enable_training_collection = True
        2. OR CLI args have collect_training_data = True (backward compatibility)

        Returns:
            True if training collection should be enabled
        """
        # Check config flag first
        experimental_config = self.tuning_config.experimental
        if experimental_config.enable_training_collection:
            return True

        # Backward compatibility: check CLI args
        if self.args.get("collect_training_data", False):
            return True

        return False

    def _add_training_data_collector(self) -> None:
        """Add experimental training data collector processor."""
        try:
            from src.processors.experimental.training import TrainingDataCollector

            # Get config from experimental namespace
            experimental_config = self.tuning_config.experimental

            # CLI args override config (for backward compatibility)
            confidence_threshold = self.args.get(
                "confidence_threshold",
                experimental_config.training_confidence_threshold,
            )
            training_data_dir = self.args.get(
                "training_data_dir", str(experimental_config.training_data_dir)
            )

            collector = TrainingDataCollector(
                self.template,
                confidence_threshold=confidence_threshold,
                export_dir=training_data_dir,
            )

            self.processors.append(collector)
            logger.warning(
                f"⚠️  Experimental training collection enabled (confidence: {confidence_threshold})"
            )

        except ImportError as e:
            logger.warning(f"Failed to add training data collector: {e}")

    def process_file(
        self,
        file_path: Path | str,
        gray_image: MatLike,
        colored_image: MatLike,
    ) -> ProcessingContext:
        """Process a single OMR file through all processors.

        Args:
            file_path: Path to the file being processed
            gray_image: Grayscale input image
            colored_image: Colored input image

        Returns:
            ProcessingContext containing all results (omr_response, metrics, etc.)
        """
        logger.info(f"Starting pipeline for file: {file_path}")

        # Create initial context
        context = ProcessingContext(
            file_path=file_path,
            gray_image=gray_image,
            colored_image=colored_image,
            template=self.template,
        )

        # Execute each processor in sequence
        for processor in self.processors:
            processor_name = processor.get_name()
            logger.debug(f"Executing processor: {processor_name}")
            context = processor.process(context)

        logger.info(f"Completed pipeline for file: {file_path}")

        return context

    def reset_extra_processors(self) -> None:
        """Remove processors added after pipeline initialisation.

        Called by reset_and_setup_for_directory so that per-directory
        processors (e.g. FileOrganizerProcessor) don't accumulate across
        directories when the same template is reused.
        """
        self.processors = self.processors[: self._base_processor_count]

    def add_processor(self, processor: Processor) -> None:
        """Add a custom processor to the pipeline.

        This allows for extensibility - users can add their own processors
        for custom processing requirements.

        Args:
            processor: The processor to add to the pipeline
        """
        self.processors.append(processor)

    def remove_processor(self, processor_name: str) -> None:
        """Remove a processor from the pipeline by name.

        Args:
            processor_name: Name of the processor to remove
        """
        self.processors = [p for p in self.processors if p.get_name() != processor_name]

    def get_processor_names(self) -> list[str]:
        """Get the names of all processors in the pipeline.

        Returns:
            List of processor names
        """
        return [processor.get_name() for processor in self.processors]
