"""Tests for the unified processor architecture."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.processors import (
    AlignmentProcessor,
    ProcessingContext,
    ProcessingPipeline,
    Processor,
    ReadOMRProcessor,
)


@pytest.fixture
def mock_template():
    """Create a mock template for testing."""
    template = Mock()
    template.tuning_config = Mock()
    template.tuning_config.outputs = Mock()
    template.tuning_config.outputs.colored_outputs_enabled = True
    template.tuning_config.outputs.show_preprocessors_diff = {}
    template.tuning_config.outputs.show_image_level = 1

    # Add alignment config (disabled by default for cleaner tests)
    template.tuning_config.alignment = Mock()
    template.tuning_config.alignment.enabled = False

    template.template_layout = Mock()
    template.template_layout.processing_image_shape = [800, 600]
    template.template_layout.output_image_shape = None
    template.template_layout.pre_processors = []
    template.template_layout.get_copy_for_shifting = Mock(
        return_value=template.template_layout
    )
    template.template_layout.reset_all_shifts = Mock()

    template.alignment = {}
    template.template_dimensions = [1000, 800]
    template.save_image_ops = Mock()
    template.save_image_ops.append_save_image = Mock()

    # Mock fields and detection types for TemplateFileRunner
    template.all_fields = []
    template.all_field_detection_types = []
    template.path = Mock()
    template.path.parent = Path("/tmp")

    template.template_file_runner = Mock()
    template.template_file_runner.read_omr_and_update_metrics = Mock(
        return_value={"Q1": "A", "Q2": "B"}
    )
    template.template_file_runner.get_directory_level_interpretation_aggregates = Mock(
        return_value={
            "file_wise_aggregates": {
                "test.jpg": {
                    "read_response_flags": {"is_multi_marked": False},
                    "field_id_to_interpretation": {},
                }
            }
        }
    )

    template.get_concatenated_omr_response = Mock(return_value={"Q1": "A", "Q2": "B"})

    return template


@pytest.fixture
def mock_images():
    """Create mock images for testing."""
    gray_image = np.zeros((1000, 800), dtype=np.uint8)
    colored_image = np.zeros((1000, 800, 3), dtype=np.uint8)
    return gray_image, colored_image


class TestProcessingContext:
    """Tests for ProcessingContext."""

    def test_context_initialization(self, mock_template, mock_images):
        """Test that context initializes correctly."""
        gray_image, colored_image = mock_images
        context = ProcessingContext(
            file_path="test.jpg",
            gray_image=gray_image,
            colored_image=colored_image,
            template=mock_template,
        )

        assert context.file_path == "test.jpg"
        assert context.gray_image is gray_image
        assert context.colored_image is colored_image
        assert context.template is mock_template
        assert context.omr_response == {}
        assert context.is_multi_marked is False

    def test_context_path_conversion(self, mock_template, mock_images):
        """Test that Path objects are converted to strings."""
        gray_image, colored_image = mock_images
        context = ProcessingContext(
            file_path=Path("test.jpg"),
            gray_image=gray_image,
            colored_image=colored_image,
            template=mock_template,
        )

        assert isinstance(context.file_path, str)
        assert context.file_path == "test.jpg"


class TestReadOMRProcessor:
    """Tests for ReadOMRProcessor."""

    @patch("src.processors.detection.processor.ImageUtils")
    def test_readomr_processor_flow(self, mock_image_utils, mock_template, mock_images):
        """Test ReadOMR processor execution."""
        gray_image, colored_image = mock_images

        # Mock ImageUtils methods
        mock_image_utils.resize_to_dimensions.return_value = (gray_image, colored_image)
        mock_image_utils.normalize.return_value = (gray_image, colored_image)

        processor = ReadOMRProcessor(mock_template)

        context = ProcessingContext(
            file_path="test.jpg",
            gray_image=gray_image,
            colored_image=colored_image,
            template=mock_template,
        )

        result = processor.process(context)

        assert result is context
        assert processor.get_name() == "ReadOMR"
        assert result.omr_response == {"Q1": "A", "Q2": "B"}
        assert result.is_multi_marked is False
        assert "raw_omr_response" in result.metadata


class TestAlignmentProcessor:
    """Tests for AlignmentProcessor."""

    @patch("src.processors.image.alignment.processor.apply_template_alignment")
    def test_alignment_with_reference_image(
        self, mock_apply_alignment, mock_template, mock_images
    ):
        """Test alignment when reference image is configured."""
        gray_image, colored_image = mock_images

        # Configure alignment
        mock_template.alignment = {"gray_alignment_image": np.zeros((100, 100))}

        # Mock the alignment function
        mock_apply_alignment.return_value = (gray_image, colored_image, mock_template)

        processor = AlignmentProcessor(mock_template)

        context = ProcessingContext(
            file_path="test.jpg",
            gray_image=gray_image,
            colored_image=colored_image,
            template=mock_template,
        )

        result = processor.process(context)

        assert result is context
        assert processor.get_name() == "Alignment"
        # Verify alignment was called
        mock_apply_alignment.assert_called_once()


class TestProcessingPipeline:
    """Tests for ProcessingPipeline."""

    @patch("src.processors.detection.processor.ImageUtils")
    @patch("src.processors.image.coordinator.ImageUtils")
    def test_full_pipeline_execution(
        self, mock_preproc_utils, mock_detect_utils, mock_template, mock_images
    ):
        """Test complete pipeline execution."""
        gray_image, colored_image = mock_images

        # Mock all ImageUtils methods
        mock_preproc_utils.resize_to_shape.side_effect = lambda _shape, img: img
        mock_detect_utils.resize_to_dimensions.return_value = (
            gray_image,
            colored_image,
        )
        mock_detect_utils.normalize.return_value = (gray_image, colored_image)

        pipeline = ProcessingPipeline(mock_template)

        result = pipeline.process_file("test.jpg", gray_image, colored_image)

        assert isinstance(result, ProcessingContext)
        assert result.file_path == "test.jpg"
        assert result.omr_response == {"Q1": "A", "Q2": "B"}
        assert result.is_multi_marked is False

    def test_pipeline_processor_management(self, mock_template):
        """Test adding and removing processors."""
        pipeline = ProcessingPipeline(mock_template)

        # Check initial processors (Alignment not included by default)
        processor_names = pipeline.get_processor_names()
        assert "Preprocessing" in processor_names
        assert "ReadOMR" in processor_names

        # Add a custom processor
        custom_processor = Mock(spec=Processor)
        custom_processor.get_name.return_value = "CustomProcessor"
        pipeline.add_processor(custom_processor)

        processor_names = pipeline.get_processor_names()
        assert "CustomProcessor" in processor_names

        # Remove a processor
        pipeline.remove_processor("CustomProcessor")
        processor_names = pipeline.get_processor_names()
        assert "CustomProcessor" not in processor_names

    def test_pipeline_with_alignment_enabled(self, mock_template, mock_images):
        """Test pipeline includes alignment when enabled and configured."""
        gray_image, colored_image = mock_images

        # Enable alignment and provide alignment data
        mock_template.tuning_config.alignment.enabled = True
        mock_template.alignment = {"gray_alignment_image": np.zeros((100, 100))}

        pipeline = ProcessingPipeline(mock_template)

        processor_names = pipeline.get_processor_names()
        assert "Alignment" in processor_names
        assert "Preprocessing" in processor_names
        assert "ReadOMR" in processor_names

    def test_pipeline_without_alignment_data(self, mock_template):
        """Test pipeline excludes alignment when no alignment data exists."""
        # Enable in config but no alignment data
        mock_template.tuning_config.alignment.enabled = True
        mock_template.alignment = {}  # No gray_alignment_image

        pipeline = ProcessingPipeline(mock_template)

        processor_names = pipeline.get_processor_names()
        assert "Alignment" not in processor_names
