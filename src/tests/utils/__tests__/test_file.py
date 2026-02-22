"""Tests for file utility functions."""

import json
import tempfile
from pathlib import Path

import pytest

from src.utils.exceptions import ConfigLoadError, InputFileNotFoundError
from src.utils.file import PathUtils, load_json


class TestFileUtils:
    """Test suite for file utility functions."""

    def test_load_json_success(self) -> None:
        """Test loading valid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            test_data = {"key": "value", "number": 42}
            json.dump(test_data, tmp)
            tmp_path = tmp.name

        try:
            result = load_json(tmp_path)
            assert result == test_data
        finally:
            Path(tmp_path).unlink()

    def test_load_json_file_not_found(self) -> None:
        """Test loading non-existent JSON file."""
        with pytest.raises(InputFileNotFoundError):
            load_json("nonexistent_file.json")

    def test_load_json_invalid_json(self) -> None:
        """Test loading invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp.write("{invalid json content")
            tmp_path = tmp.name

        try:
            with pytest.raises(ConfigLoadError):
                load_json(tmp_path)
        finally:
            Path(tmp_path).unlink()

    def test_remove_non_utf_characters(self) -> None:
        """Test removing non-UTF characters from path."""
        path = "test/path/file.txt"
        result = PathUtils.remove_non_utf_characters(path)
        assert result == "test/path/file.txt"

    def test_sep_based_posix_path(self) -> None:
        """Test converting path to POSIX format."""
        path = "test/path/file.txt"
        result = PathUtils.sep_based_posix_path(path)
        assert "/" in result or result == "test/path/file.txt"

    def test_path_utils_initialization(self) -> None:
        """Test PathUtils initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            path_utils = PathUtils(output_dir)

            assert path_utils.output_dir == output_dir
            assert path_utils.save_marked_dir == output_dir / "CheckedOMRs"
            assert path_utils.results_dir == output_dir / "Results"
            assert path_utils.manual_dir == output_dir / "Manual"

    def test_create_output_directories(self) -> None:
        """Test creating output directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_output"
            path_utils = PathUtils(output_dir)
            path_utils.create_output_directories()

            assert path_utils.save_marked_dir.exists()
            assert path_utils.results_dir.exists()
            assert path_utils.manual_dir.exists()
