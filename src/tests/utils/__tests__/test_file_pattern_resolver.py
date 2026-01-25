"""Tests for FilePatternResolver utility."""

import tempfile
from pathlib import Path

import pytest

from src.utils.file_pattern_resolver import FilePatternResolver


class TestFilePatternResolver:
    """Test suite for FilePatternResolver."""

    def test_simple_pattern_formatting(self):
        """Test basic pattern formatting with fields."""
        resolver = FilePatternResolver()
        fields = {"roll": "12345", "score": "95"}

        path = resolver.resolve_pattern(
            "roll_{roll}_score_{score}.jpg", fields, collision_strategy="overwrite"
        )

        assert path is not None
        assert path.name == "roll_12345_score_95.jpg"

    def test_pattern_with_folders(self):
        """Test pattern with nested folder structure."""
        resolver = FilePatternResolver()
        fields = {"booklet": "A", "batch": "morning", "roll": "12345"}

        path = resolver.resolve_pattern(
            "booklet_{booklet}/batch_{batch}/{roll}",
            fields,
            original_path=Path("test.jpg"),
            collision_strategy="overwrite",
        )

        assert path is not None
        assert str(path) == "booklet_A/batch_morning/12345.jpg"

    def test_extension_preservation(self):
        """Test that original file extension is preserved when not in pattern."""
        resolver = FilePatternResolver()
        fields = {"roll": "12345"}

        path = resolver.resolve_pattern(
            "student_{roll}",
            fields,
            original_path=Path("original.png"),
            collision_strategy="overwrite",
        )

        assert path is not None
        assert path.suffix == ".png"
        assert path.name == "student_12345.png"

    def test_extension_in_pattern_overrides(self):
        """Test that extension in pattern overrides original extension."""
        resolver = FilePatternResolver()
        fields = {"roll": "12345"}

        path = resolver.resolve_pattern(
            "student_{roll}.jpg",
            fields,
            original_path=Path("original.png"),
            collision_strategy="overwrite",
        )

        assert path is not None
        assert path.suffix == ".jpg"
        assert path.name == "student_12345.jpg"

    def test_base_directory(self):
        """Test that base directory is prepended to resolved paths."""
        base_dir = Path("/tmp/organized")
        resolver = FilePatternResolver(base_dir=base_dir)
        fields = {"roll": "12345"}

        path = resolver.resolve_pattern(
            "student_{roll}", fields, collision_strategy="overwrite"
        )

        assert path is not None
        assert str(path).startswith(str(base_dir))

    def test_path_sanitization(self):
        """Test that invalid characters are sanitized from paths."""
        resolver = FilePatternResolver()
        fields = {"name": "John<>Doe", "code": "A|B*C"}

        path = resolver.resolve_pattern(
            "{name}/{code}", fields, collision_strategy="overwrite"
        )

        assert path is not None
        # Invalid chars should be replaced with underscores
        assert "<" not in str(path)
        assert ">" not in str(path)
        assert "|" not in str(path)
        assert "*" not in str(path)

    def test_missing_field_returns_none(self):
        """Test that missing field in pattern returns None."""
        resolver = FilePatternResolver()
        fields = {"roll": "12345"}

        path = resolver.resolve_pattern(
            "student_{name}", fields, collision_strategy="overwrite"
        )

        assert path is None

    def test_collision_skip_strategy(self):
        """Test that skip strategy returns None when file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            resolver = FilePatternResolver(base_dir=base_dir)
            fields = {"roll": "12345"}

            # Create a file that will collide
            existing_file = base_dir / "student_12345.jpg"
            existing_file.touch()

            path = resolver.resolve_pattern(
                "student_{roll}",
                fields,
                original_path=Path("test.jpg"),
                collision_strategy="skip",
            )

            assert path is None

    def test_collision_increment_strategy(self):
        """Test that increment strategy adds counter when file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            resolver = FilePatternResolver(base_dir=base_dir)
            fields = {"roll": "12345"}

            # Create files that will collide
            (base_dir / "student_12345.jpg").touch()
            (base_dir / "student_12345_001.jpg").touch()

            path = resolver.resolve_pattern(
                "student_{roll}",
                fields,
                original_path=Path("test.jpg"),
                collision_strategy="increment",
            )

            assert path is not None
            assert path.name == "student_12345_002.jpg"

    def test_collision_overwrite_strategy(self):
        """Test that overwrite strategy returns path even if file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            resolver = FilePatternResolver(base_dir=base_dir)
            fields = {"roll": "12345"}

            # Create a file that will collide
            existing_file = base_dir / "student_12345.jpg"
            existing_file.touch()

            path = resolver.resolve_pattern(
                "student_{roll}",
                fields,
                original_path=Path("test.jpg"),
                collision_strategy="overwrite",
            )

            assert path is not None
            assert path.name == "student_12345.jpg"

    def test_complex_pattern_with_multiple_fields(self):
        """Test complex pattern with many fields and nested folders."""
        resolver = FilePatternResolver()
        fields = {
            "region": "north",
            "school": "ABC",
            "class": "10A",
            "roll": "12345",
            "name": "John_Doe",
            "score": "95",
        }

        path = resolver.resolve_pattern(
            "region_{region}/school_{school}/class_{class}/{name}_roll_{roll}_score_{score}",
            fields,
            original_path=Path("test.jpg"),
            collision_strategy="overwrite",
        )

        assert path is not None
        assert (
            str(path)
            == "region_north/school_ABC/class_10A/John_Doe_roll_12345_score_95.jpg"
        )

    def test_empty_fields(self):
        """Test pattern with empty field values."""
        resolver = FilePatternResolver()
        fields = {"roll": "", "name": ""}

        path = resolver.resolve_pattern(
            "{roll}/{name}", fields, collision_strategy="overwrite"
        )

        # Should still resolve, but with empty values
        assert path is not None

    def test_special_characters_in_field_values(self):
        """Test that special characters in values are sanitized."""
        resolver = FilePatternResolver()
        fields = {"name": "John/Doe\\Test", "code": "A:B"}

        path = resolver.resolve_pattern(
            "{name}_{code}", fields, collision_strategy="overwrite"
        )

        assert path is not None
        # Slashes and backslashes should be sanitized
        assert "/" not in path.name or path.name.count("/") == 0
        assert "\\" not in path.name

    def test_resolve_batch(self):
        """Test batch resolution of multiple patterns."""
        resolver = FilePatternResolver()

        patterns_and_fields = [
            ("student_{roll}", {"roll": "001"}, Path("test1.jpg")),
            ("student_{roll}", {"roll": "002"}, Path("test2.jpg")),
            ("student_{roll}", {"roll": "003"}, Path("test3.jpg")),
        ]

        results = resolver.resolve_batch(patterns_and_fields, "overwrite")

        assert len(results) == 3
        assert all(path is not None for path, _ in results)
        assert results[0][0].name == "student_001.jpg"
        assert results[1][0].name == "student_002.jpg"
        assert results[2][0].name == "student_003.jpg"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
