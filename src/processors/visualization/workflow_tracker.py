"""Workflow tracking for visualization and debugging.

This module provides tracking capabilities to capture the state of images
as they flow through the processing pipeline, enabling visualization and replay.
"""

import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.processors.base import ProcessingContext
from src.processors.visualization.workflow_session import (
    ImageEncoder,
    ProcessorState,
    WorkflowSession,
)
from src.utils.logger import logger


class WorkflowTracker:
    """Tracks workflow execution for visualization purposes.

    This class wraps around the pipeline execution and captures:
    - Processor execution order and timing
    - Input/output images at each stage
    - Processor metadata and success/failure states
    - Workflow graph structure

    Attributes:
        session: WorkflowSession containing all captured data
        capture_processors: List of processor names to capture (or ["all"])
        max_image_width: Maximum width for captured images
        include_colored: Whether to capture colored images
        image_quality: JPEG quality for captured images (0-100)
        _start_times: Track start times for each processor
    """

    def __init__(  # noqa: PLR0913
        self,
        file_path: Path | str,
        template_name: str = "Unknown",
        capture_processors: list[str] | None = None,
        max_image_width: int = 800,
        include_colored: bool = True,
        image_quality: int = 85,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the workflow tracker.

        Args:
            file_path: Path to the file being processed
            template_name: Name of the template being used
            capture_processors: List of processor names to capture (None or ["all"] for all)
            max_image_width: Maximum width for resizing captured images
            include_colored: Whether to capture colored images
            image_quality: JPEG quality for captured images (0-100)
            config: Configuration dict to store with session
        """
        session_id = f"session_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now(UTC).isoformat()

        self.session = WorkflowSession(
            session_id=session_id,
            file_path=str(file_path),
            template_name=template_name,
            start_time=start_time,
            config=config or {},
        )

        self.capture_processors = capture_processors or ["all"]
        self.max_image_width = max_image_width
        self.include_colored = include_colored
        self.image_quality = image_quality

        self._start_times: dict[str, float] = {}
        self._execution_order = 0

        logger.info(f"Initialized workflow tracker: {session_id}")

    def should_capture(self, processor_name: str) -> bool:
        """Check if a processor should be captured.

        Args:
            processor_name: Name of the processor

        Returns:
            True if the processor should be captured
        """
        if not self.capture_processors or "all" in self.capture_processors:
            return True
        return processor_name in self.capture_processors

    def start_processor(self, processor_name: str) -> None:
        """Mark the start of a processor execution.

        Args:
            processor_name: Name of the processor starting
        """
        if self.should_capture(processor_name):
            self._start_times[processor_name] = time.time()
            logger.debug(f"[Tracker] Started tracking: {processor_name}")

    def capture_state(
        self,
        processor_name: str,
        context: ProcessingContext,
        metadata: dict[str, Any] | None = None,
        success: bool = True,
        error_message: str | None = None,
    ) -> None:
        """Capture the current state after a processor execution.

        Args:
            processor_name: Name of the processor
            context: Current processing context
            metadata: Additional processor-specific metadata
            success: Whether the processor executed successfully
            error_message: Error message if processor failed
        """
        if not self.should_capture(processor_name):
            logger.debug(f"[Tracker] Skipping capture for: {processor_name}")
            return

        # Calculate duration
        start_time = self._start_times.get(processor_name)
        duration_ms = 0.0
        if start_time is not None:
            duration_ms = (time.time() - start_time) * 1000

        # Encode images
        gray_image_base64 = None
        colored_image_base64 = None
        image_shape = (0, 0)

        if context.gray_image is not None:
            gray_image_base64 = ImageEncoder.encode_image(
                context.gray_image,
                max_width=self.max_image_width,
                quality=self.image_quality,
            )
            image_shape = context.gray_image.shape

        if self.include_colored and context.colored_image is not None:
            colored_image_base64 = ImageEncoder.encode_image(
                context.colored_image,
                max_width=self.max_image_width,
                quality=self.image_quality,
            )

        # Create processor state
        state = ProcessorState(
            name=processor_name,
            order=self._execution_order,
            timestamp=datetime.now(UTC).isoformat(),
            duration_ms=duration_ms,
            image_shape=image_shape,
            gray_image_base64=gray_image_base64,
            colored_image_base64=colored_image_base64,
            metadata=metadata or {},
            success=success,
            error_message=error_message,
        )

        # Add to session
        self.session.add_processor_state(state)
        self._execution_order += 1

        logger.info(
            f"[Tracker] Captured state: {processor_name} "
            f"(order={state.order}, duration={duration_ms:.2f}ms)"
        )

    def build_graph(self, processor_names: list[str]) -> None:
        """Build the workflow graph structure from processor names.

        Args:
            processor_names: List of processor names in execution order
        """
        # Add input node
        self.session.graph.add_node(
            node_id="input",
            label="Input Image",
            metadata={"type": "input", "file_path": self.session.file_path},
        )

        # Add processor nodes
        for i, name in enumerate(processor_names):
            node_id = f"processor_{i}"
            self.session.graph.add_node(
                node_id=node_id, label=name, metadata={"type": "processor", "order": i}
            )

        # Add output node
        self.session.graph.add_node(
            node_id="output", label="Output", metadata={"type": "output"}
        )

        # Add edges
        if len(processor_names) > 0:
            # Input to first processor
            self.session.graph.add_edge("input", "processor_0")

            # Between processors
            for i in range(len(processor_names) - 1):
                self.session.graph.add_edge(f"processor_{i}", f"processor_{i + 1}")

            # Last processor to output
            self.session.graph.add_edge(
                f"processor_{len(processor_names) - 1}", "output"
            )
        else:
            # Direct input to output if no processors
            self.session.graph.add_edge("input", "output")

        logger.info(f"[Tracker] Built graph with {len(processor_names)} processors")

    def finalize(self) -> WorkflowSession:
        """Finalize the tracking session and return the complete session data.

        Returns:
            Complete WorkflowSession with all captured data
        """
        end_time = datetime.now(UTC).isoformat()
        start_dt = datetime.fromisoformat(self.session.start_time)
        end_dt = datetime.fromisoformat(end_time)
        total_duration_ms = (end_dt - start_dt).total_seconds() * 1000

        self.session.finalize(end_time, total_duration_ms)

        logger.info(
            f"[Tracker] Finalized session: {self.session.session_id} "
            f"(duration={total_duration_ms:.2f}ms, "
            f"processors={len(self.session.processor_states)})"
        )

        return self.session


def track_workflow(  # noqa: PLR0913
    file_path: Path | str,
    template_path: Path | str,
    config_path: Path | str | None = None,
    capture_processors: list[str] | None = None,
    max_image_width: int = 800,
    include_colored: bool = True,
    image_quality: int = 85,
) -> WorkflowSession:
    """High-level function to track a complete workflow execution.

    This function runs the entire OMR processing pipeline with tracking enabled,
    capturing intermediate states for visualization.

    Args:
        file_path: Path to the input OMR image
        template_path: Path to the template JSON file
        config_path: Path to the config JSON file (optional)
        capture_processors: List of processor names to capture (None for all)
        max_image_width: Maximum width for captured images
        include_colored: Whether to capture colored images
        image_quality: JPEG quality for captured images (0-100)

    Returns:
        Complete WorkflowSession with all captured data

    Example:
        >>> session = track_workflow(
        ...     file_path="inputs/sample1/sample1.jpg",
        ...     template_path="inputs/sample1/template.json",
        ...     capture_processors=["AutoRotate", "ReadOMR"]
        ... )
        >>> print(f"Captured {len(session.processor_states)} processors")
    """
    from src.processors.pipeline import ProcessingPipeline
    from src.processors.template.template import Template
    from src.utils.image import ImageUtils
    from src.utils.parsing import open_config_with_defaults

    # Load config
    if config_path:
        # Use the proper config loading function with defaults
        config = open_config_with_defaults(
            Path(config_path), {"outputMode": "default", "debug": False}
        )
    else:
        from src.schemas.defaults import CONFIG_DEFAULTS

        config = CONFIG_DEFAULTS

    # Load template
    template = Template(Path(template_path), config)

    # Load image
    gray_image, colored_image = ImageUtils.read_image_util(Path(file_path), config)

    # Initialize tracker
    tracker = WorkflowTracker(
        file_path=file_path,
        template_name=Path(template_path).stem,  # Use template filename as name
        capture_processors=capture_processors,
        max_image_width=max_image_width,
        include_colored=include_colored,
        image_quality=image_quality,
        config={"template_path": str(template_path), "config_path": str(config_path)},
    )

    # Create pipeline
    pipeline = ProcessingPipeline(template)

    # Get processor names for graph
    processor_names = pipeline.get_processor_names()
    tracker.build_graph(processor_names)

    # Track initial state
    initial_context = ProcessingContext(
        file_path=file_path,
        gray_image=gray_image,
        colored_image=colored_image,
        template=template,
    )
    tracker.capture_state("Input", initial_context, metadata={"stage": "initial"})

    # Execute pipeline with tracking
    for processor in pipeline.processors:
        processor_name = processor.get_name()

        try:
            tracker.start_processor(processor_name)
            initial_context = processor.process(initial_context)
            tracker.capture_state(
                processor_name,
                initial_context,
                metadata={"stage": "processing"},
                success=True,
            )
        except Exception as e:
            logger.error(f"Error in processor {processor_name}: {e}")
            tracker.capture_state(
                processor_name,
                initial_context,
                metadata={"stage": "processing"},
                success=False,
                error_message=str(e),
            )
            raise

    # Finalize and return
    return tracker.finalize()
