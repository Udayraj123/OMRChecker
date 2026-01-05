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
        from src.processors.alignment.processor import (
            AlignmentProcessor,
        )
        from src.processors.detection.processor import ReadOMRProcessor
        from src.processors.image.coordinator import (
            PreprocessingProcessor,
        )

        # Check for ML model paths from args
        ml_model_path = self.args.get("ml_model_path")
        field_block_model_path = self.args.get("field_block_model_path")
        use_field_block_detection = self.args.get("use_field_block_detection", False)

        # Initialize all processors with unified interface
        self.processors: list[Processor] = [
            PreprocessingProcessor(template),
            AlignmentProcessor(template),
        ]

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
        self.processors.append(ReadOMRProcessor(template, ml_model_path=ml_model_path))

        # Add training data collector if enabled
        if self.args.get("collect_training_data", False):
            self._add_training_data_collector()

    def _add_training_data_collector(self) -> None:
        """Add training data collector processor."""
        try:
            from src.processors.training import TrainingDataCollector

            confidence_threshold = self.args.get("confidence_threshold", 0.85)
            training_data_dir = self.args.get(
                "training_data_dir", "outputs/training_data"
            )

            collector = TrainingDataCollector(
                self.template,
                confidence_threshold=confidence_threshold,
                export_dir=training_data_dir,
            )

            self.processors.append(collector)
            logger.info(
                f"Training data collection enabled (confidence threshold: {confidence_threshold})"
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

            try:
                context = processor.process(context)
            except Exception as e:
                logger.error(f"Error in processor {processor_name}: {e}")
                raise

        logger.info(f"Completed pipeline for file: {file_path}")

        return context

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
