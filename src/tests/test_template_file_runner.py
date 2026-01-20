"""Comprehensive tests for TemplateFileRunner class.

Tests multi-pass detection and interpretation architecture.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.processors.detection.template_file_runner import TemplateFileRunner
from src.processors.layout.field.bubble_field import BubbleField
from src.processors.layout.field_block.base import FieldBlock
from src.schemas.defaults import CONFIG_DEFAULTS


@pytest.fixture
def mock_template():
    """Create a mock template for testing."""
    template = Mock()
    template.tuning_config = CONFIG_DEFAULTS
    template.all_fields = []
    template.all_field_detection_types = []
    return template


@pytest.fixture
def sample_template_with_fields(mock_template, tmp_path):
    """Create a template with sample fields."""
    # Set up template path (required for initialize_directory_level_aggregates)
    mock_template.path = tmp_path / "template.json"
    mock_template.path.parent.mkdir(parents=True, exist_ok=True)

    # Create mock field blocks
    field_block1 = Mock(spec=FieldBlock)
    field_block1.name = "block1"
    field_block1.origin = [100, 100]

    # Create mock fields
    field1 = Mock(spec=BubbleField)
    field1.id = "block1::q1"
    field1.field_label = "q1"
    field1.field_detection_type = "BUBBLES_THRESHOLD"
    field1.field_block = field_block1
    field1.empty_value = ""

    field2 = Mock(spec=BubbleField)
    field2.id = "block1::q2"
    field2.field_label = "q2"
    field2.field_detection_type = "BUBBLES_THRESHOLD"
    field2.field_block = field_block1
    field2.empty_value = ""

    mock_template.all_fields = [field1, field2]
    mock_template.all_field_detection_types = ["BUBBLES_THRESHOLD"]

    return mock_template


@pytest.fixture
def sample_images():
    """Create sample images for testing."""
    gray_image = np.zeros((800, 1000), dtype=np.uint8)
    colored_image = np.zeros((800, 1000, 3), dtype=np.uint8)
    return gray_image, colored_image


def create_mock_bubble_fields(fields):
    """Create mock bubble_fields dictionary for detection aggregates."""
    from src.processors.detection.models.detection_results import (
        BubbleFieldDetectionResult,
        BubbleMeanValue,
    )

    bubble_fields = {}
    for field in fields:
        # Create minimal bubble means for testing
        bubble_means = [
            BubbleMeanValue(mean_value=50.0, unit_bubble=None),
            BubbleMeanValue(mean_value=200.0, unit_bubble=None),
        ]
        bubble_result = BubbleFieldDetectionResult(
            field_id=field.id,
            field_label=field.field_label,
            bubble_means=bubble_means,
        )
        bubble_fields[field.field_label] = bubble_result

    return bubble_fields


class TestTemplateFileRunnerInitialization:
    """Test TemplateFileRunner initialization."""

    def test_initialization_with_template(self, sample_template_with_fields):
        """Test initialization with a template."""
        runner = TemplateFileRunner(sample_template_with_fields)

        assert runner.template == sample_template_with_fields
        assert runner.all_fields == sample_template_with_fields.all_fields
        assert len(runner.all_fields) == 2

    def test_initialize_field_file_runners(self, sample_template_with_fields):
        """Test initialization of field file runners."""
        runner = TemplateFileRunner(sample_template_with_fields)

        assert "BUBBLES_THRESHOLD" in runner.field_detection_type_file_runners
        assert len(runner.field_detection_type_file_runners) == 1

    def test_initialize_directory_level_aggregates(self, sample_template_with_fields):
        """Test initialization of directory level aggregates."""
        # Template path is already set up in fixture
        runner = TemplateFileRunner(sample_template_with_fields)

        # Check that aggregates are initialized for both passes
        detection_aggregates = runner.get_directory_level_detection_aggregates()
        interpretation_aggregates = (
            runner.get_directory_level_interpretation_aggregates()
        )

        assert detection_aggregates is not None
        assert interpretation_aggregates is not None
        assert "initial_directory_path" in detection_aggregates
        assert "initial_directory_path" in interpretation_aggregates


class TestReadOmrAndUpdateMetrics:
    """Test read_omr_and_update_metrics method."""

    def test_read_omr_and_update_metrics_two_pass(
        self, sample_template_with_fields, sample_images, tmp_path
    ):
        """Test two-pass processing (detection then interpretation)."""
        # Template path is already set up in fixture
        runner = TemplateFileRunner(sample_template_with_fields)

        gray_image, colored_image = sample_images
        file_path = str(tmp_path / "test.jpg")

        # Mock the field runners to return responses
        with (
            patch.object(
                runner.field_detection_type_file_runners["BUBBLES_THRESHOLD"],
                "run_field_level_detection",
            ) as mock_detection,
            patch.object(
                runner.field_detection_type_file_runners["BUBBLES_THRESHOLD"],
                "run_field_level_interpretation",
            ) as mock_interpretation,
            patch.object(
                runner.field_detection_type_file_runners["BUBBLES_THRESHOLD"],
                "get_field_level_interpretation_aggregates",
            ) as mock_get_aggregates,
            patch.object(
                runner.detection_pass,
                "get_file_level_aggregates",
            ) as mock_get_detection_aggregates,
        ):
            mock_detection.return_value = Mock()
            mock_interpretation_result = Mock()
            mock_interpretation_result.get_field_interpretation_string = Mock(
                return_value="A"
            )
            mock_interpretation.return_value = mock_interpretation_result
            mock_get_aggregates.return_value = {
                "field": sample_template_with_fields.all_fields[0],
                "is_multi_marked": False,
            }
            # Mock bubble_fields in detection aggregates
            bubble_fields = create_mock_bubble_fields(
                sample_template_with_fields.all_fields
            )
            mock_get_detection_aggregates.return_value = {
                "bubble_fields": bubble_fields,
                "ocr_fields": {},
                "barcode_fields": {},
            }

            response = runner.read_omr_and_update_metrics(
                file_path, gray_image, colored_image
            )

            assert response is not None
            assert isinstance(response, dict)


class TestRunFileLevelDetection:
    """Test run_file_level_detection method."""

    def test_run_file_level_detection_all_fields(
        self, sample_template_with_fields, sample_images, tmp_path
    ):
        """Test detection pass for all fields."""
        # Template path is already set up in fixture
        runner = TemplateFileRunner(sample_template_with_fields)

        gray_image, colored_image = sample_images
        file_path = str(tmp_path / "test.jpg")

        # Mock field detection
        with patch.object(
            runner.field_detection_type_file_runners["BUBBLES_THRESHOLD"],
            "run_field_level_detection",
        ) as mock_detection:
            mock_detection.return_value = Mock()

            runner.run_file_level_detection(file_path, gray_image, colored_image)

            # Should have called detection for each field
            assert mock_detection.call_count == 2  # Two fields

    def test_update_detection_aggregates_on_processed_file(
        self, sample_template_with_fields, sample_images, tmp_path
    ):
        """Test updating detection aggregates after processing file."""
        # Template path is already set up in fixture
        runner = TemplateFileRunner(sample_template_with_fields)

        gray_image, colored_image = sample_images
        file_path = str(tmp_path / "test.jpg")

        with patch.object(
            runner.field_detection_type_file_runners["BUBBLES_THRESHOLD"],
            "run_field_level_detection",
        ) as mock_detection:
            mock_detection.return_value = Mock()
            runner.run_file_level_detection(file_path, gray_image, colored_image)

        # Check that aggregates were updated
        aggregates = runner.get_directory_level_detection_aggregates()
        assert file_path in aggregates["file_wise_aggregates"]


class TestRunFieldLevelDetection:
    """Test run_field_level_detection method."""

    def test_run_field_level_detection_bubbles(
        self, sample_template_with_fields, sample_images, tmp_path
    ):
        """Test field-level detection for bubble fields."""
        # Template path is already set up in fixture
        runner = TemplateFileRunner(sample_template_with_fields)

        gray_image, colored_image = sample_images
        file_path = str(tmp_path / "test.jpg")
        field = sample_template_with_fields.all_fields[0]

        runner.initialize_file_level_detection_aggregates(file_path)

        with patch.object(
            runner.field_detection_type_file_runners["BUBBLES_THRESHOLD"],
            "run_field_level_detection",
        ) as mock_detection:
            mock_detection.return_value = Mock()

            runner.run_field_level_detection(field, gray_image, colored_image)

            mock_detection.assert_called_once()


class TestRunFileLevelInterpretation:
    """Test run_file_level_interpretation method."""

    def test_run_file_level_interpretation(
        self, sample_template_with_fields, sample_images, tmp_path
    ):
        """Test interpretation pass for all fields."""
        # Template path is already set up in fixture
        runner = TemplateFileRunner(sample_template_with_fields)

        gray_image, colored_image = sample_images
        file_path = str(tmp_path / "test.jpg")

        # Mock detection results first
        with patch.object(
            runner.field_detection_type_file_runners["BUBBLES_THRESHOLD"],
            "run_field_level_detection",
        ) as mock_detection:
            mock_detection.return_value = Mock()
            runner.run_file_level_detection(file_path, gray_image, colored_image)

        # Mock interpretation
        with (
            patch.object(
                runner.field_detection_type_file_runners[
                    "BUBBLES_THRESHOLD"
                ].interpretation_pass,
                "run_field_level_interpretation",
            ) as mock_interpretation,
            patch.object(
                runner.field_detection_type_file_runners["BUBBLES_THRESHOLD"],
                "get_field_level_interpretation_aggregates",
            ) as mock_get_aggregates,
            patch.object(
                runner.detection_pass,
                "get_file_level_aggregates",
            ) as mock_get_detection_aggregates,
        ):
            mock_interpretation_result = Mock()
            mock_interpretation_result.get_field_interpretation_string = Mock(
                return_value="A"
            )
            mock_interpretation.return_value = mock_interpretation_result
            mock_get_aggregates.return_value = {
                "field": sample_template_with_fields.all_fields[0],
                "is_multi_marked": False,
            }
            # Mock bubble_fields in detection aggregates
            bubble_fields = create_mock_bubble_fields(
                sample_template_with_fields.all_fields
            )
            mock_get_detection_aggregates.return_value = {
                "bubble_fields": bubble_fields,
                "ocr_fields": {},
                "barcode_fields": {},
            }

            response = runner.run_file_level_interpretation(
                file_path, gray_image, colored_image
            )

            assert response is not None
            assert isinstance(response, dict)
            assert mock_interpretation.call_count == 2  # Two fields


class TestRunFieldLevelInterpretation:
    """Test run_field_level_interpretation method."""

    def test_run_field_level_interpretation(
        self, sample_template_with_fields, sample_images, tmp_path
    ):
        """Test field-level interpretation."""
        # Template path is already set up in fixture
        runner = TemplateFileRunner(sample_template_with_fields)

        gray_image, colored_image = sample_images
        file_path = str(tmp_path / "test.jpg")
        field = sample_template_with_fields.all_fields[0]

        # Run detection first to populate aggregates
        with patch.object(
            runner.field_detection_type_file_runners["BUBBLES_THRESHOLD"],
            "run_field_level_detection",
        ) as mock_detection:
            mock_detection.return_value = Mock()
            runner.run_file_level_detection(file_path, gray_image, colored_image)

        # Initialize interpretation aggregates (only needs file_path, gets aggregates internally)
        runner.initialize_file_level_interpretation_aggregates(file_path)

        current_omr_response = {}

        with (
            patch.object(
                runner.field_detection_type_file_runners[
                    "BUBBLES_THRESHOLD"
                ].interpretation_pass,
                "run_field_level_interpretation",
            ) as mock_interpretation,
            patch.object(
                runner.field_detection_type_file_runners["BUBBLES_THRESHOLD"],
                "get_field_level_interpretation_aggregates",
            ) as mock_get_aggregates,
            patch.object(
                runner.detection_pass,
                "get_file_level_aggregates",
            ) as mock_get_detection_aggregates,
        ):
            mock_interpretation_result = Mock()
            mock_interpretation_result.get_field_interpretation_string = Mock(
                return_value="A"
            )
            mock_interpretation.return_value = mock_interpretation_result
            mock_get_aggregates.return_value = {
                "field": field,
                "is_multi_marked": False,
            }
            # Mock bubble_fields in detection aggregates
            bubble_fields = create_mock_bubble_fields(
                sample_template_with_fields.all_fields
            )
            mock_get_detection_aggregates.return_value = {
                "bubble_fields": bubble_fields,
                "ocr_fields": {},
                "barcode_fields": {},
            }

            runner.run_field_level_interpretation(field, current_omr_response)

            assert field.field_label in current_omr_response
            mock_interpretation.assert_called_once()


class TestAggregateManagement:
    """Test aggregate collection and management."""

    def test_aggregate_collection_across_files(
        self, sample_template_with_fields, sample_images, tmp_path
    ):
        """Test aggregate collection across multiple files."""
        # Template path is already set up in fixture
        runner = TemplateFileRunner(sample_template_with_fields)

        gray_image, colored_image = sample_images

        # Process multiple files
        for i in range(3):
            file_path = str(tmp_path / f"test_{i}.jpg")

            with (
                patch.object(
                    runner.field_detection_type_file_runners["BUBBLES_THRESHOLD"],
                    "run_field_level_detection",
                ) as mock_detection,
                patch.object(
                    runner.field_detection_type_file_runners["BUBBLES_THRESHOLD"],
                    "run_field_level_interpretation",
                ) as mock_interpretation,
                patch.object(
                    runner.field_detection_type_file_runners["BUBBLES_THRESHOLD"],
                    "get_field_level_interpretation_aggregates",
                ) as mock_get_aggregates,
                patch.object(
                    runner.detection_pass,
                    "get_file_level_aggregates",
                ) as mock_get_detection_aggregates,
            ):
                mock_detection.return_value = Mock()
                mock_interpretation.return_value = Mock()
                mock_interpretation.return_value.get_field_interpretation_string = Mock(
                    return_value="A"
                )
                mock_get_aggregates.return_value = {
                    "field": sample_template_with_fields.all_fields[0],
                    "is_multi_marked": False,
                }
                # Mock bubble_fields in detection aggregates
                bubble_fields = create_mock_bubble_fields(
                    sample_template_with_fields.all_fields
                )
                mock_get_detection_aggregates.return_value = {
                    "bubble_fields": bubble_fields,
                    "ocr_fields": {},
                    "barcode_fields": {},
                }

                runner.run_file_level_detection(file_path, gray_image, colored_image)
                runner.run_file_level_interpretation(
                    file_path, gray_image, colored_image
                )

        # Check that all files are in aggregates
        aggregates = runner.get_directory_level_detection_aggregates()
        assert len(aggregates["file_wise_aggregates"]) == 3

    def test_finish_processing_directory(self, sample_template_with_fields, tmp_path):
        """Test finishing directory processing."""
        # Template path is already set up in fixture
        runner = TemplateFileRunner(sample_template_with_fields)

        # Should not raise
        runner.finish_processing_directory()
        assert True

    def test_get_export_omr_metrics_for_file(
        self, sample_template_with_fields, sample_images, tmp_path
    ):
        """Test getting export metrics for a file."""
        # Template path is already set up in fixture
        runner = TemplateFileRunner(sample_template_with_fields)

        gray_image, colored_image = sample_images
        file_path = str(tmp_path / "test.jpg")

        with (
            patch.object(
                runner.field_detection_type_file_runners["BUBBLES_THRESHOLD"],
                "run_field_level_detection",
            ) as mock_detection,
            patch.object(
                runner.field_detection_type_file_runners["BUBBLES_THRESHOLD"],
                "run_field_level_interpretation",
            ) as mock_interpretation,
            patch.object(
                runner.field_detection_type_file_runners["BUBBLES_THRESHOLD"],
                "get_field_level_interpretation_aggregates",
            ) as mock_get_aggregates,
            patch.object(
                runner.detection_pass,
                "get_file_level_aggregates",
            ) as mock_get_detection_aggregates,
        ):
            mock_detection.return_value = Mock()
            mock_interpretation.return_value = Mock()
            mock_interpretation.return_value.get_field_interpretation_string = Mock(
                return_value="A"
            )
            mock_get_aggregates.return_value = {
                "field": sample_template_with_fields.all_fields[0],
                "is_multi_marked": False,
            }
            # Mock bubble_fields in detection aggregates
            bubble_fields = create_mock_bubble_fields(
                sample_template_with_fields.all_fields
            )
            mock_get_detection_aggregates.return_value = {
                "bubble_fields": bubble_fields,
                "ocr_fields": {},
                "barcode_fields": {},
            }

            runner.run_file_level_detection(file_path, gray_image, colored_image)
            runner.run_file_level_interpretation(file_path, gray_image, colored_image)

        # Note: get_export_omr_metrics_for_file currently returns None (not implemented)
        runner.get_export_omr_metrics_for_file()
        # Just verify method doesn't crash
        assert True
