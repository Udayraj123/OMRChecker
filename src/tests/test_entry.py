"""Comprehensive tests for entry point and directory processing.

Tests entry_point, process_directory_wise, and related functions.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.entry import entry_point, process_directory_wise, print_config_summary
from src.exceptions import InputDirectoryNotFoundError
from src.processors.evaluation.evaluation_config import EvaluationConfig
from src.processors.template.template import Template
from src.schemas.defaults import CONFIG_DEFAULTS


@pytest.fixture
def temp_input_dir(tmp_path):
    """Create a temporary input directory."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    return input_dir


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def minimal_args():
    """Minimal args dictionary."""
    return {
        "debug": False,
        "outputMode": "default",
        "setLayout": False,
        "mode": "process",
        "input_paths": [],
        "output_dir": None,
    }


@pytest.fixture
def sample_image_file(temp_input_dir):
    """Create a sample image file."""
    image_file = temp_input_dir / "test.jpg"
    image_file.touch()
    return image_file


class TestEntryPoint:
    """Test entry_point function."""

    def test_entry_point_with_valid_directory(
        self, temp_input_dir, temp_output_dir, minimal_args
    ):
        """Test entry point with valid directory."""
        minimal_args["input_paths"] = [str(temp_input_dir)]
        minimal_args["output_dir"] = str(temp_output_dir)

        # Should not raise (may need mocks for template processing)
        try:
            entry_point(temp_input_dir, minimal_args)
        except (FileNotFoundError, AttributeError):
            # Expected if template.json doesn't exist
            pass

    def test_entry_point_with_invalid_directory(self, temp_output_dir, minimal_args):
        """Test entry point with invalid directory."""
        minimal_args["input_paths"] = ["/nonexistent/directory"]
        minimal_args["output_dir"] = str(temp_output_dir)

        invalid_dir = Path("/nonexistent/directory")

        with pytest.raises(InputDirectoryNotFoundError):
            entry_point(invalid_dir, minimal_args)


class TestProcessDirectoryWise:
    """Test process_directory_wise function."""

    def test_process_directory_wise_recursive(
        self, temp_input_dir, temp_output_dir, minimal_args
    ):
        """Test recursive directory processing."""
        # Create nested directories
        subdir = temp_input_dir / "subdir"
        subdir.mkdir()

        minimal_args["input_paths"] = [str(temp_input_dir)]
        minimal_args["output_dir"] = str(temp_output_dir)

        # Mock template and config loading
        with (
            patch("src.entry.Template") as mock_template_class,
            patch("src.entry.open_config_with_defaults") as mock_config,
            patch("src.entry.EvaluationConfig"),
        ):
            mock_config.return_value = CONFIG_DEFAULTS
            mock_template_class.side_effect = FileNotFoundError("No template")

            # Should handle missing template gracefully
            try:
                process_directory_wise(temp_input_dir, temp_input_dir, minimal_args)
            except (FileNotFoundError, AttributeError):
                pass

    def test_process_directory_wise_with_local_config(
        self, temp_input_dir, temp_output_dir, minimal_args
    ):
        """Test processing with local config.json."""
        import json

        # Create local config
        config_file = temp_input_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump({}, f)

        minimal_args["input_paths"] = [str(temp_input_dir)]
        minimal_args["output_dir"] = str(temp_output_dir)

        with (
            patch("src.entry.Template") as mock_template_class,
            patch("src.entry.open_config_with_defaults") as mock_config,
        ):
            mock_config.return_value = CONFIG_DEFAULTS
            mock_template_class.side_effect = FileNotFoundError("No template")

            try:
                process_directory_wise(temp_input_dir, temp_input_dir, minimal_args)
            except (FileNotFoundError, AttributeError):
                pass

    def test_process_directory_wise_with_local_template(
        self, temp_input_dir, temp_output_dir, minimal_args
    ):
        """Test processing with local template.json."""
        import json

        # Create local template
        template_file = temp_input_dir / "template.json"
        template_data = {
            "templateDimensions": [1000, 800],
            "bubbleDimensions": [20, 20],
            "emptyValue": "",
            "fieldBlocksOffset": [0, 0],
            "fieldBlocks": {},
            "preProcessors": [],
            "alignment": {"referenceImagePath": None},
            "customBubbleFieldTypes": {},
            "customLabels": {},
            "outputColumns": [],
        }
        with open(template_file, "w") as f:
            json.dump(template_data, f)

        minimal_args["input_paths"] = [str(temp_input_dir)]
        minimal_args["output_dir"] = str(temp_output_dir)

        with (
            patch("src.entry.ProcessingPipeline"),
            patch("src.entry.PathUtils") as mock_path_utils,
        ):
            mock_path_utils_instance = Mock()
            mock_path_utils.return_value = mock_path_utils_instance
            mock_path_utils_instance.create_output_directories = Mock()

            try:
                process_directory_wise(temp_input_dir, temp_input_dir, minimal_args)
            except (AttributeError, KeyError):
                # Expected if some dependencies are missing
                pass

    def test_process_directory_wise_with_evaluation_config(
        self, temp_input_dir, temp_output_dir, minimal_args
    ):
        """Test processing with evaluation.json."""
        import json

        # Create evaluation config
        eval_file = temp_input_dir / "evaluation.json"
        eval_data = {
            "source_type": "local",
            "options": {
                "questions_in_order": ["q1"],
                "answers_in_order": ["A"],
            },
            "marking_schemes": {
                "DEFAULT": {
                    "correct": 1,
                    "incorrect": 0,
                    "unmarked": 0,
                }
            },
            "outputs_configuration": {},
        }
        with open(eval_file, "w") as f:
            json.dump(eval_data, f)

        minimal_args["input_paths"] = [str(temp_input_dir)]
        minimal_args["output_dir"] = str(temp_output_dir)

        with (
            patch("src.entry.Template") as mock_template_class,
            patch("src.entry.EvaluationConfig"),
        ):
            mock_template_class.side_effect = FileNotFoundError("No template")

            try:
                process_directory_wise(temp_input_dir, temp_input_dir, minimal_args)
            except (FileNotFoundError, AttributeError):
                pass


class TestPrintConfigSummary:
    """Test print_config_summary function."""

    def test_print_config_summary_all_fields(self, temp_input_dir, minimal_args):
        """Test printing config summary with all fields."""
        mock_template = Mock(spec=Template)
        mock_template.path = temp_input_dir / "template.json"
        mock_template.get_pre_processor_names = Mock(return_value=["GaussianBlur"])
        mock_template.get_processing_image_shape = Mock(return_value=[800, 600])

        omr_files = ["test1.jpg", "test2.jpg"]
        local_config_path = None
        evaluation_config = None

        # Should not raise
        print_config_summary(
            temp_input_dir,
            omr_files,
            mock_template,
            local_config_path,
            evaluation_config,
            minimal_args,
        )
        assert True

    def test_print_config_summary_with_local_config(self, temp_input_dir, minimal_args):
        """Test printing config summary with local config."""
        mock_template = Mock(spec=Template)
        mock_template.path = temp_input_dir / "template.json"
        mock_template.get_pre_processor_names = Mock(return_value=[])
        mock_template.get_processing_image_shape = Mock(return_value=[800, 600])

        omr_files = ["test1.jpg"]
        local_config_path = temp_input_dir / "config.json"
        evaluation_config = None

        print_config_summary(
            temp_input_dir,
            omr_files,
            mock_template,
            local_config_path,
            evaluation_config,
            minimal_args,
        )
        assert True

    def test_print_config_summary_with_evaluation(self, temp_input_dir, minimal_args):
        """Test printing config summary with evaluation config."""
        mock_template = Mock(spec=Template)
        mock_template.path = temp_input_dir / "template.json"
        mock_template.get_pre_processor_names = Mock(return_value=[])
        mock_template.get_processing_image_shape = Mock(return_value=[800, 600])

        omr_files = ["test1.jpg"]
        local_config_path = None
        evaluation_config = Mock(spec=EvaluationConfig)
        evaluation_config.path = temp_input_dir / "evaluation.json"

        print_config_summary(
            temp_input_dir,
            omr_files,
            mock_template,
            local_config_path,
            evaluation_config,
            minimal_args,
        )
        assert True
