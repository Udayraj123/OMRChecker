"""Comprehensive tests for TemplateFileRunner class.

Tests multi-pass detection and interpretation architecture.
"""

from contextlib import ExitStack, contextmanager
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.processors.detection.template_file_runner import TemplateFileRunner
from src.processors.layout.field.bubble_field import BubbleField
from src.processors.layout.field_block.base import FieldBlock
from src.utils.stats import StatsByLabel


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


def create_mock_file_results(fields, file_path):
    """Create mock FileDetectionResults for repository."""
    from src.processors.detection.models.detection_results import (
        FileDetectionResults,
    )

    file_results = FileDetectionResults(file_path=file_path)
    bubble_fields = create_mock_bubble_fields(fields)

    # Add bubble fields to repository format (keyed by field_id)
    for field in fields:
        bubble_result = bubble_fields[field.field_label]
        file_results.bubble_fields[field.id] = bubble_result

    return file_results


@contextmanager
def mock_field_runner_methods(
    runner,
    field_type="BUBBLES_THRESHOLD",
    include_detection=True,
    include_interpretation=True,
    include_aggregates=True,
    include_repository=True,
):
    """Context manager to mock common field runner methods.

    Yields a MockConfig object with all mocks for easy configuration.

    Args:
        runner: TemplateFileRunner instance
        field_type: Field detection type to mock (default: "BUBBLES_THRESHOLD")
        include_detection: Whether to mock detection methods
        include_interpretation: Whether to mock interpretation methods
        include_aggregates: Whether to mock aggregate getters
        include_repository: Whether to mock repository methods

    Yields:
        MockConfig object with attributes for each mocked method
    """
    field_runner = runner.field_detection_type_file_runners[field_type]

    patches = {}
    with ExitStack() as stack:
        if include_detection:
            patches["detection"] = stack.enter_context(
                patch.object(field_runner, "run_field_level_detection")
            )

        if include_interpretation:
            patches["interpretation"] = stack.enter_context(
                patch.object(field_runner, "run_field_level_interpretation")
            )
            patches["interpretation_pass"] = stack.enter_context(
                patch.object(
                    field_runner.interpretation_pass, "run_field_level_interpretation"
                )
            )

        if include_aggregates:
            patches["get_interpretation_aggregates"] = stack.enter_context(
                patch.object(field_runner, "get_field_level_interpretation_aggregates")
            )
            patches["get_detection_aggregates"] = stack.enter_context(
                patch.object(runner.detection_pass, "get_file_level_aggregates")
            )

        if include_repository:
            patches["get_file_results"] = stack.enter_context(
                patch.object(field_runner.repository, "get_file_results")
            )

        # Create a simple namespace object for easy access
        class MockConfig:
            pass

        config = MockConfig()
        for key, mock in patches.items():
            setattr(config, key, mock)

        yield config


@contextmanager
def mock_detection_only(runner, field_type="BUBBLES_THRESHOLD"):
    """Context manager for tests that only need detection mocks."""
    with mock_field_runner_methods(
        runner,
        field_type,
        include_detection=True,
        include_interpretation=False,
        include_aggregates=False,
        include_repository=False,
    ) as mocks:
        yield mocks


@contextmanager
def mock_interpretation_only(runner, field_type="BUBBLES_THRESHOLD"):
    """Context manager for tests that only need interpretation mocks."""
    with mock_field_runner_methods(
        runner,
        field_type,
        include_detection=False,
        include_interpretation=True,
        include_aggregates=True,
        include_repository=False,
    ) as mocks:
        yield mocks


def setup_default_mock_responses(
    config, fields, file_path, interpretation_value="A", is_multi_marked=False
):
    """Configure mocks with default return values.

    Args:
        config: MockConfig object from mock_field_runner_methods
        fields: List of field objects
        file_path: File path string
        interpretation_value: Value to return for interpretation (default: "A")
        is_multi_marked: Whether field is multi-marked (default: False)
    """
    if hasattr(config, "detection"):
        config.detection.return_value = Mock()

    if hasattr(config, "interpretation"):
        mock_interpretation_result = Mock()
        mock_interpretation_result.get_field_interpretation_string = Mock(
            return_value=interpretation_value
        )
        config.interpretation.return_value = mock_interpretation_result

    if hasattr(config, "interpretation_pass"):
        mock_interpretation_result = Mock()
        mock_interpretation_result.get_field_interpretation_string = Mock(
            return_value=interpretation_value
        )
        config.interpretation_pass.return_value = mock_interpretation_result

    if hasattr(config, "get_interpretation_aggregates"):
        config.get_interpretation_aggregates.return_value = {
            "field": fields[0] if fields else None,
            "is_multi_marked": is_multi_marked,
        }

    if hasattr(config, "get_detection_aggregates"):
        bubble_fields = create_mock_bubble_fields(fields)
        config.get_detection_aggregates.return_value = {
            "bubble_fields": bubble_fields,
            "ocr_fields": {},
            "barcode_fields": {},
            "fields_count": StatsByLabel("processed"),
        }

    if hasattr(config, "get_file_results"):
        mock_file_results = create_mock_file_results(fields, file_path)
        config.get_file_results.return_value = mock_file_results


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

        with mock_field_runner_methods(runner) as mocks:
            setup_default_mock_responses(
                mocks,
                sample_template_with_fields.all_fields,
                file_path,
            )

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

        with mock_detection_only(runner) as mocks:
            mocks.detection.return_value = Mock()

            runner.run_file_level_detection(file_path, gray_image, colored_image)

            # Should have called detection for each field
            assert mocks.detection.call_count == 2  # Two fields

    def test_update_detection_aggregates_on_processed_file(
        self, sample_template_with_fields, sample_images, tmp_path
    ):
        """Test updating detection aggregates after processing file."""
        # Template path is already set up in fixture
        runner = TemplateFileRunner(sample_template_with_fields)

        gray_image, colored_image = sample_images
        file_path = str(tmp_path / "test.jpg")

        with mock_detection_only(runner) as mocks:
            mocks.detection.return_value = Mock()
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

        with mock_detection_only(runner) as mocks:
            mocks.detection.return_value = Mock()

            runner.run_field_level_detection(field, gray_image, colored_image)

            mocks.detection.assert_called_once()


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
        with mock_detection_only(runner) as detection_mocks:
            detection_mocks.detection.return_value = Mock()
            runner.run_file_level_detection(file_path, gray_image, colored_image)

        # Mock interpretation
        with mock_interpretation_only(runner) as interpretation_mocks:
            setup_default_mock_responses(
                interpretation_mocks,
                sample_template_with_fields.all_fields,
                file_path,
            )

            response = runner.run_file_level_interpretation(
                file_path, gray_image, colored_image
            )

            assert response is not None
            assert isinstance(response, dict)
            assert (
                interpretation_mocks.interpretation_pass.call_count == 2
            )  # Two fields


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
        with mock_detection_only(runner) as detection_mocks:
            detection_mocks.detection.return_value = Mock()
            runner.run_file_level_detection(file_path, gray_image, colored_image)

        # Initialize interpretation aggregates (only needs file_path, gets aggregates internally)
        runner.initialize_file_level_interpretation_aggregates(file_path)

        current_omr_response = {}

        with mock_interpretation_only(runner) as interpretation_mocks:
            setup_default_mock_responses(
                interpretation_mocks,
                sample_template_with_fields.all_fields,
                file_path,
            )

            runner.run_field_level_interpretation(field, current_omr_response)

            assert field.field_label in current_omr_response
            interpretation_mocks.interpretation_pass.assert_called_once()


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

            with mock_field_runner_methods(runner) as mocks:
                setup_default_mock_responses(
                    mocks,
                    sample_template_with_fields.all_fields,
                    file_path,
                )

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

        with mock_field_runner_methods(runner) as mocks:
            setup_default_mock_responses(
                mocks,
                sample_template_with_fields.all_fields,
                file_path,
            )

            runner.run_file_level_detection(file_path, gray_image, colored_image)
            runner.run_file_level_interpretation(file_path, gray_image, colored_image)

        # Note: get_export_omr_metrics_for_file currently returns None (not implemented)
        runner.get_export_omr_metrics_for_file()
        # Just verify method doesn't crash
        assert True
