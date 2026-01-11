"""Tests for CSV utility functions."""

import tempfile
from pathlib import Path

import pandas as pd

from src.utils.csv import thread_safe_csv_append


class TestCSVUtils:
    """Test suite for CSV utility functions."""

    def test_thread_safe_csv_append_basic(self) -> None:
        """Test basic CSV append operation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            data_line = ["Alice", "30", "Engineer"]
            thread_safe_csv_append(tmp_path, data_line)

            # Verify file was created and data written
            assert Path(tmp_path).exists()

            # Read and verify content - pandas infers types
            df = pd.read_csv(tmp_path, header=None)
            assert df.shape[0] == 1
            assert df.iloc[0, 0] == "Alice"
            assert df.iloc[0, 1] == 30  # Pandas infers as int
            assert df.iloc[0, 2] == "Engineer"
        finally:
            Path(tmp_path).unlink()

    def test_thread_safe_csv_append_multiple(self) -> None:
        """Test appending multiple lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # First append
            thread_safe_csv_append(tmp_path, ["Alice", "30", "Engineer"])
            # Second append
            thread_safe_csv_append(tmp_path, ["Bob", "25", "Designer"])

            # Verify both lines exist - pandas infers types
            df = pd.read_csv(tmp_path, header=None)
            assert df.shape[0] == 2
            assert df.iloc[0, 0] == "Alice"
            assert df.iloc[0, 1] == 30
            assert df.iloc[1, 0] == "Bob"
            assert df.iloc[1, 1] == 25
        finally:
            Path(tmp_path).unlink()

    def test_thread_safe_csv_append_numeric(self) -> None:
        """Test appending numeric data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            data_line = ["100", "200", "300"]
            thread_safe_csv_append(tmp_path, data_line)

            df = pd.read_csv(tmp_path, header=None)
            assert df.shape[0] == 1
            # Values are inferred as integers by pandas
            assert list(df.iloc[0]) == [100, 200, 300]
        finally:
            Path(tmp_path).unlink()

    def test_thread_safe_csv_append_empty_line(self) -> None:
        """Test appending empty line."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            thread_safe_csv_append(tmp_path, [])

            assert Path(tmp_path).exists()
        finally:
            Path(tmp_path).unlink()
