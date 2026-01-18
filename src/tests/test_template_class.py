"""Comprehensive tests for Template class.

Tests Template lifecycle, preprocessor application, and response handling.
"""

import json
from unittest.mock import Mock

import numpy as np
import pytest

from src.processors.template.template import Template
from src.schemas.defaults import CONFIG_DEFAULTS


@pytest.fixture
def temp_template_path(tmp_path):
    """Create a temporary template JSON file."""
    template_path = tmp_path / "template.json"
    template_data = {
        "templateDimensions": [1000, 800],
        "bubbleDimensions": [20, 20],
        "emptyValue": "",
        "fieldBlocksOffset": [0, 0],
        "fieldBlocks": {
            "block1": {
                "fieldDetectionType": "BUBBLES_THRESHOLD",
                "origin": [100, 100],
                "fieldLabels": ["q1", "q2"],
                "bubbleFieldType": "QTYPE_MCQ4",
                "bubblesGap": 30,
                "labelsGap": 50,
            }
        },
        "preProcessors": [],
        "alignment": {"referenceImagePath": None},
        "customBubbleFieldTypes": {},
        "customLabels": {},
        "outputColumns": [],
    }
    with open(template_path, "w") as f:
        json.dump(template_data, f)
    return template_path


@pytest.fixture
def mock_tuning_config():
    """Create a mock tuning config."""
    return CONFIG_DEFAULTS


@pytest.fixture
def minimal_args():
    """Minimal args dictionary."""
    return {
        "debug": False,
        "outputMode": "default",
        "setLayout": False,
        "mode": "process",
    }


@pytest.fixture
def sample_template(temp_template_path, mock_tuning_config, minimal_args):
    """Create a Template instance for testing."""
    return Template(temp_template_path, mock_tuning_config, minimal_args)


class TestTemplateInitialization:
    """Test Template initialization."""

    def test_template_initialization(
        self, temp_template_path, mock_tuning_config, minimal_args
    ):
        """Test basic template initialization."""
        template = Template(temp_template_path, mock_tuning_config, minimal_args)

        assert template.path == temp_template_path
        assert template.tuning_config == mock_tuning_config
        assert len(template.all_fields) == 2  # q1 and q2

    def test_template_with_custom_preprocessors(
        self, tmp_path, mock_tuning_config, minimal_args
    ):
        """Test template initialization with custom preprocessors."""
        template_path = tmp_path / "template.json"
        template_data = {
            "templateDimensions": [1000, 800],
            "bubbleDimensions": [20, 20],
            "emptyValue": "",
            "fieldBlocksOffset": [0, 0],
            "fieldBlocks": {
                "block1": {
                    "name": "block1",
                    "origin": [100, 100],
                    "fieldLabels": ["q1"],
                    "bubbleFieldType": "QTYPE_MCQ4",
                    "bubblesGap": 30,
                    "labelsGap": 50,
                }
            },
            "preProcessors": [{"name": "GaussianBlur", "kernelSize": 5}],
            "alignment": {"referenceImagePath": None},
            "customBubbleFieldTypes": {},
            "customLabels": {},
            "outputColumns": [],
        }
        with open(template_path, "w") as f:
            json.dump(template_data, f)

        template = Template(template_path, mock_tuning_config, minimal_args)

        assert len(template.get_pre_processors()) == 1
        assert template.get_pre_processor_names() == ["GaussianBlur"]


class TestApplyPreprocessors:
    """Test apply_preprocessors method."""

    def test_apply_preprocessors_sequence(self, sample_template, tmp_path):
        """Test applying preprocessors in sequence."""
        gray_image = np.zeros((800, 1000), dtype=np.uint8)
        colored_image = np.zeros((800, 1000, 3), dtype=np.uint8)
        file_path = tmp_path / "test.jpg"

        processed_gray, processed_colored, updated_template = (
            sample_template.apply_preprocessors(
                str(file_path), gray_image, colored_image
            )
        )

        assert processed_gray is not None
        assert processed_colored is not None
        assert updated_template is not None

    def test_apply_preprocessors_with_alignment(
        self, tmp_path, mock_tuning_config, minimal_args
    ):
        """Test applying preprocessors with alignment."""
        template_path = tmp_path / "template.json"
        template_data = {
            "templateDimensions": [1000, 800],
            "bubbleDimensions": [20, 20],
            "emptyValue": "",
            "fieldBlocksOffset": [0, 0],
            "fieldBlocks": {
                "block1": {
                    "name": "block1",
                    "origin": [100, 100],
                    "fieldLabels": ["q1"],
                    "bubbleFieldType": "QTYPE_MCQ4",
                    "bubblesGap": 30,
                    "labelsGap": 50,
                }
            },
            "preProcessors": [{"name": "GaussianBlur", "kernelSize": 5}],
            "alignment": {"referenceImagePath": None},
            "customBubbleFieldTypes": {},
            "customLabels": {},
            "outputColumns": [],
        }
        with open(template_path, "w") as f:
            json.dump(template_data, f)

        template = Template(template_path, mock_tuning_config, minimal_args)

        gray_image = np.zeros((800, 1000), dtype=np.uint8)
        colored_image = np.zeros((800, 1000, 3), dtype=np.uint8)
        file_path = tmp_path / "test.jpg"

        processed_gray, processed_colored, updated_template = (
            template.apply_preprocessors(str(file_path), gray_image, colored_image)
        )

        assert processed_gray is not None
        assert processed_colored is not None


class TestResetAndSetupForDirectory:
    """Test reset_and_setup_for_directory method."""

    def test_reset_and_setup_for_directory(self, sample_template, tmp_path):
        """Test resetting and setting up for a directory."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Should not raise
        sample_template.reset_and_setup_for_directory(output_dir)
        assert True

    def test_reset_and_setup_outputs(self, sample_template, tmp_path):
        """Test resetting and setting up outputs."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sample_template.reset_and_setup_outputs(output_dir)
        assert True


class TestGetExcludeFiles:
    """Test get_exclude_files method."""

    def test_get_exclude_files_without_preprocessors(self, sample_template):
        """Test getting exclude files without preprocessors."""
        excluded = sample_template.get_exclude_files()

        assert isinstance(excluded, list)

    def test_get_exclude_files_with_preprocessors(
        self, tmp_path, mock_tuning_config, minimal_args
    ):
        """Test getting exclude files with preprocessors that exclude files."""
        template_path = tmp_path / "template.json"
        template_data = {
            "templateDimensions": [1000, 800],
            "bubbleDimensions": [20, 20],
            "emptyValue": "",
            "fieldBlocksOffset": [0, 0],
            "fieldBlocks": {
                "block1": {
                    "name": "block1",
                    "origin": [100, 100],
                    "fieldLabels": ["q1"],
                    "bubbleFieldType": "QTYPE_MCQ4",
                    "bubblesGap": 30,
                    "labelsGap": 50,
                }
            },
            "preProcessors": [],
            "alignment": {"referenceImagePath": None},
            "customBubbleFieldTypes": {},
            "customLabels": {},
            "outputColumns": [],
        }
        with open(template_path, "w") as f:
            json.dump(template_data, f)

        template = Template(template_path, mock_tuning_config, minimal_args)

        # Mock preprocessor with exclude_files method
        mock_preprocessor = Mock()
        mock_preprocessor.exclude_files = Mock(return_value=["excluded1.jpg"])
        template.template_layout.pre_processors = [mock_preprocessor]

        excluded = template.get_exclude_files()

        assert isinstance(excluded, list)


class TestGetPreProcessors:
    """Test get_pre_processors method."""

    def test_get_pre_processors(self, sample_template):
        """Test getting preprocessors."""
        preprocessors = sample_template.get_pre_processors()

        assert isinstance(preprocessors, list)

    def test_get_pre_processor_names(self, sample_template):
        """Test getting preprocessor names."""
        names = sample_template.get_pre_processor_names()

        assert isinstance(names, list)


class TestGetConcatenatedOmrResponse:
    """Test get_concatenated_omr_response method."""

    def test_get_concatenated_omr_response_single_column(self, sample_template):
        """Test getting concatenated response for single column."""
        raw_response = {"q1": "A", "q2": "B"}
        concatenated = sample_template.get_concatenated_omr_response(raw_response)

        assert isinstance(concatenated, dict)
        assert "q1" in concatenated
        assert "q2" in concatenated

    def test_get_concatenated_omr_response_multi_column(self, sample_template):
        """Test getting concatenated response for multiple columns."""
        raw_response = {"q1": "A", "q2": "B"}
        concatenated = sample_template.get_concatenated_omr_response(raw_response)

        assert isinstance(concatenated, dict)
        assert len(concatenated) == 2

    def test_get_concatenated_omr_response_custom_labels(
        self, tmp_path, mock_tuning_config, minimal_args
    ):
        """Test getting concatenated response with custom labels."""
        template_path = tmp_path / "template.json"
        template_data = {
            "templateDimensions": [1000, 800],
            "bubbleDimensions": [20, 20],
            "emptyValue": "",
            "fieldBlocksOffset": [0, 0],
            "fieldBlocks": {
                "block1": {
                    "fieldDetectionType": "BUBBLES_THRESHOLD",
                    "origin": [100, 100],
                    "fieldLabels": ["q1", "q2"],
                    "bubbleFieldType": "QTYPE_MCQ4",
                    "bubblesGap": 30,
                    "labelsGap": 50,
                }
            },
            "preProcessors": [],
            "alignment": {},
            "customBubbleFieldTypes": {},
            "customLabels": {"CUSTOM_1": ["q1", "q2"]},
            "outputColumns": {},
        }
        with open(template_path, "w") as f:
            json.dump(template_data, f)

        template = Template(template_path, mock_tuning_config, minimal_args)
        raw_response = {"q1": "A", "q2": "B"}
        concatenated = template.get_concatenated_omr_response(raw_response)

        assert "CUSTOM_1" in concatenated
        assert concatenated["CUSTOM_1"] == "AB"  # Concatenated


class TestGetProcessingImageShape:
    """Test get_processing_image_shape method."""

    def test_get_processing_image_shape(self, sample_template):
        """Test getting processing image shape."""
        shape = sample_template.get_processing_image_shape()

        assert isinstance(shape, list)
        assert len(shape) == 2


class TestGetEmptyResponseArray:
    """Test get_empty_response_array method."""

    def test_get_empty_response_array(self, sample_template):
        """Test getting empty response array."""
        empty_array = sample_template.get_empty_response_array()

        assert isinstance(empty_array, list)


class TestAppendOutputOmrResponse:
    """Test append_output_omr_response method."""

    def test_append_output_omr_response(self, sample_template):
        """Test appending output OMR response."""
        output_omr_response = {"q1": "A", "q2": "B"}
        result = sample_template.append_output_omr_response(
            "test.jpg", output_omr_response
        )

        assert isinstance(result, list)
        assert len(result) > 0


class TestToString:
    """Test __str__ method."""

    def test_to_string(self, sample_template):
        """Test string representation."""
        str_repr = str(sample_template)

        assert isinstance(str_repr, str)
        assert len(str_repr) > 0
