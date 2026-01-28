"""Tests for main.py entry point.

Tests the main CLI interface and argument parsing.
"""

import sys
from unittest.mock import patch

import pytest


class TestParseArgs:
    """Test parse_args function from main.py."""

    def test_parse_args_with_defaults(self):
        """Test parsing with default arguments."""
        from main import parse_args

        test_args = []
        sys.argv = ["main.py"] + test_args

        args = parse_args()

        assert args["input_paths"] == ["./inputs"]
        assert args["output_dir"] == "./outputs"
        assert args["debug"] is False
        assert args["setLayout"] is False
        assert args["outputMode"] == "default"
        assert args["mode"] == "process"
        assert args["collect_training_data"] is False
        assert args["confidence_threshold"] == 0.8

    def test_parse_args_with_custom_input(self):
        """Test parsing with custom input directory."""
        from main import parse_args

        test_args = ["-i", "custom/input"]
        sys.argv = ["main.py"] + test_args

        args = parse_args()

        assert args["input_paths"] == ["custom/input"]

    def test_parse_args_with_multiple_inputs(self):
        """Test parsing with multiple input directories."""
        from main import parse_args

        test_args = ["-i", "input1", "input2", "input3"]
        sys.argv = ["main.py"] + test_args

        args = parse_args()

        assert args["input_paths"] == ["input1", "input2", "input3"]

    def test_parse_args_with_output_dir(self):
        """Test parsing with custom output directory."""
        from main import parse_args

        test_args = ["-o", "custom/output"]
        sys.argv = ["main.py"] + test_args

        args = parse_args()

        assert args["output_dir"] == "custom/output"

    def test_parse_args_with_setLayout(self):
        """Test parsing with setLayout flag."""
        from main import parse_args

        test_args = ["--setLayout"]
        sys.argv = ["main.py"] + test_args

        args = parse_args()

        assert args["setLayout"] is True

    def test_parse_args_with_debug(self):
        """Test parsing with debug flag."""
        from main import parse_args

        test_args = ["--debug"]
        sys.argv = ["main.py"] + test_args

        args = parse_args()

        assert args["debug"] is True

    def test_parse_args_with_output_mode(self):
        """Test parsing with output mode."""
        from main import parse_args

        test_args = ["--outputMode", "csv"]
        sys.argv = ["main.py"] + test_args

        args = parse_args()

        assert args["outputMode"] == "csv"

    def test_parse_args_with_mode(self):
        """Test parsing with processing mode."""
        from main import parse_args

        test_args = ["--mode", "auto-train"]
        sys.argv = ["main.py"] + test_args

        args = parse_args()

        assert args["mode"] == "auto-train"

    def test_parse_args_with_training_data_collection(self):
        """Test parsing with training data collection flag."""
        from main import parse_args

        test_args = ["--collect-training-data", "--confidence-threshold", "0.9"]
        sys.argv = ["main.py"] + test_args

        args = parse_args()

        assert args["collect_training_data"] is True
        assert args["confidence_threshold"] == 0.9

    def test_parse_args_with_all_options(self):
        """Test parsing with all options combined."""
        from main import parse_args

        test_args = [
            "-i",
            "custom/input1",
            "custom/input2",
            "-o",
            "custom/output",
            "--debug",
            "--setLayout",
            "--outputMode",
            "moderation",
            "--mode",
            "auto-train",
            "--collect-training-data",
            "--confidence-threshold",
            "0.95",
        ]
        sys.argv = ["main.py"] + test_args

        args = parse_args()

        assert args["input_paths"] == ["custom/input1", "custom/input2"]
        assert args["output_dir"] == "custom/output"
        assert args["debug"] is True
        assert args["setLayout"] is True
        assert args["outputMode"] == "moderation"
        assert args["mode"] == "auto-train"
        assert args["collect_training_data"] is True
        assert args["confidence_threshold"] == 0.95


class TestValidatePaths:
    """Test validate_paths function from main.py."""

    def test_validate_paths_with_existing_input(self, tmp_path):
        """Test validation with existing input directory."""
        from main import validate_paths

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        args = {
            "input_paths": [str(input_dir)],
            "output_dir": str(tmp_path / "output"),
        }

        # Should not raise
        validate_paths(args)

    def test_validate_paths_with_nonexistent_input(self, tmp_path):
        """Test validation with non-existent input directory."""
        from main import validate_paths

        args = {
            "input_paths": [str(tmp_path / "nonexistent")],
            "output_dir": str(tmp_path / "output"),
        }

        # Should exit with error
        with pytest.raises(SystemExit):
            validate_paths(args)

    def test_validate_paths_with_file_as_output(self, tmp_path):
        """Test validation with file as output directory."""
        from main import validate_paths

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        output_file = tmp_path / "output"
        output_file.touch()  # Create as file, not directory

        args = {
            "input_paths": [str(input_dir)],
            "output_dir": str(output_file),
        }

        # Should exit with error
        with pytest.raises(SystemExit):
            validate_paths(args)


class TestMain:
    """Test main function from main.py."""

    def test_main_with_valid_paths(self, tmp_path):
        """Test main function with valid paths."""
        from main import main

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        test_args = ["-i", str(input_dir), "-o", str(output_dir)]
        sys.argv = ["main.py"] + test_args

        with patch("main.run_cli") as mock_run_cli:
            exit_code = main()

            assert exit_code == 0
            mock_run_cli.assert_called_once()

    def test_main_with_keyboard_interrupt(self, tmp_path):
        """Test main function with keyboard interrupt."""
        from main import main

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        test_args = ["-i", str(input_dir)]
        sys.argv = ["main.py"] + test_args

        with patch("main.run_cli", side_effect=KeyboardInterrupt):
            exit_code = main()

            assert exit_code == 130

    def test_main_with_exception(self, tmp_path):
        """Test main function with exception."""
        from main import main

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        test_args = ["-i", str(input_dir)]
        sys.argv = ["main.py"] + test_args

        with patch("main.run_cli", side_effect=Exception("Test error")):
            exit_code = main()

            assert exit_code == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
