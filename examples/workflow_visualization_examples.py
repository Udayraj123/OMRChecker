"""Example usage of the workflow visualization tool.

This script demonstrates various ways to use the visualization functionality.
"""

import contextlib
from pathlib import Path

from src.processors.visualization import (
    WorkflowTracker,
    export_to_html,
    replay_from_json,
    track_workflow,
)


def example_1_basic_usage():
    """Example 1: Basic usage with high-level function."""
    session = track_workflow(
        file_path="inputs/sample1/sample1.jpg",
        template_path="inputs/sample1/template.json",
        config_path="inputs/sample1/config.json",
    )

    export_to_html(
        session,
        output_dir="outputs/visualization/example1",
        title="Example 1: Basic Workflow",
    )


def example_2_specific_processors():
    """Example 2: Capture only specific processors."""
    # Only capture key stages
    session = track_workflow(
        file_path="inputs/sample1/sample1.jpg",
        template_path="inputs/sample1/template.json",
        capture_processors=["AutoRotate", "CropOnMarkers", "ReadOMR"],
    )

    export_to_html(
        session,
        output_dir="outputs/visualization/example2",
        title="Example 2: Key Processors Only",
        open_browser=False,
    )


def example_3_custom_settings():
    """Example 3: Custom image quality and size settings."""
    session = track_workflow(
        file_path="inputs/sample1/sample1.jpg",
        template_path="inputs/sample1/template.json",
        max_image_width=1200,  # Larger images
        image_quality=95,  # Higher quality
        include_colored=True,  # Include colored images
    )

    export_to_html(
        session,
        output_dir="outputs/visualization/example3",
        title="Example 3: High Quality Images",
        open_browser=False,
    )


def example_4_manual_tracking():
    """Example 4: Manual tracking with WorkflowTracker."""
    from src.processors.base import ProcessingContext
    from src.processors.pipeline import ProcessingPipeline
    from src.processors.template.template import Template
    from src.schemas.models.config import Config
    from src.utils.image import ImageUtils

    # Load components
    config = Config()
    template = Template(Path("inputs/sample1/template.json"), config)
    gray_image, colored_image = ImageUtils.read_image_util(
        Path("inputs/sample1/sample1.jpg"), config
    )

    # Create tracker
    tracker = WorkflowTracker(
        file_path="inputs/sample1/sample1.jpg",
        template_name=template.template_name,
        capture_processors=["all"],
    )

    # Build graph
    pipeline = ProcessingPipeline(template)
    tracker.build_graph(pipeline.get_processor_names())

    # Track each processor manually
    context = ProcessingContext(
        file_path="inputs/sample1/sample1.jpg",
        gray_image=gray_image,
        colored_image=colored_image,
        template=template,
    )

    # Capture initial state
    tracker.capture_state("Input", context, metadata={"stage": "initial"})

    for processor in pipeline.processors:
        processor_name = processor.get_name()
        tracker.start_processor(processor_name)

        try:
            context = processor.process(context)
            tracker.capture_state(
                processor_name, context, metadata={"stage": "processing"}, success=True
            )
        except Exception:
            tracker.capture_state(
                processor_name,
                context,
                metadata={"stage": "processing"},
                success=False,
                error_message=str(Exception),
            )
            break

    # Finalize
    session = tracker.finalize()

    # Export
    export_to_html(
        session,
        output_dir="outputs/visualization/example4",
        title="Example 4: Manual Tracking",
        open_browser=False,
    )


def example_5_replay_session():
    """Example 5: Replay a previously saved session."""
    # First, create and save a session
    session = track_workflow(
        file_path="inputs/sample1/sample1.jpg",
        template_path="inputs/sample1/template.json",
    )

    export_to_html(
        session,
        output_dir="outputs/visualization/example5_original",
        title="Example 5: Original Session",
        open_browser=False,
        export_json=True,
    )

    # Find the JSON file
    json_file = (
        Path("outputs/visualization/example5_original/sessions")
        / f"{session.session_id}.json"
    )

    if json_file.exists():
        # Replay it
        replay_from_json(
            json_file,
            output_dir="outputs/visualization/example5_replayed",
            title="Example 5: Replayed Session",
            open_browser=False,
        )


def example_6_custom_metadata():
    """Example 6: Adding custom metadata to states."""
    from src.processors.base import ProcessingContext
    from src.processors.template.template import Template
    from src.schemas.models.config import Config
    from src.utils.image import ImageUtils

    # Load components
    config = Config()
    template = Template(Path("inputs/sample1/template.json"), config)
    gray_image, colored_image = ImageUtils.read_image_util(
        Path("inputs/sample1/sample1.jpg"), config
    )

    # Create tracker
    tracker = WorkflowTracker(
        file_path="inputs/sample1/sample1.jpg", template_name=template.template_name
    )

    # Create context
    context = ProcessingContext(
        file_path="inputs/sample1/sample1.jpg",
        gray_image=gray_image,
        colored_image=colored_image,
        template=template,
    )

    # Capture with custom metadata
    tracker.capture_state(
        "CustomProcessor",
        context,
        metadata={
            "threshold_value": 180,
            "bubbles_detected": 42,
            "confidence_score": 0.95,
            "processing_mode": "adaptive",
            "custom_flag": True,
            "notes": "This processor uses adaptive thresholding",
        },
    )

    # Finalize and export
    session = tracker.finalize()
    export_to_html(
        session,
        output_dir="outputs/visualization/example6",
        title="Example 6: Custom Metadata",
        open_browser=False,
    )


def main():
    """Run all examples."""
    with contextlib.suppress(Exception):
        example_1_basic_usage()

    with contextlib.suppress(Exception):
        example_2_specific_processors()

    with contextlib.suppress(Exception):
        example_3_custom_settings()

    with contextlib.suppress(Exception):
        example_4_manual_tracking()

    with contextlib.suppress(Exception):
        example_5_replay_session()

    with contextlib.suppress(Exception):
        example_6_custom_metadata()


if __name__ == "__main__":
    main()
