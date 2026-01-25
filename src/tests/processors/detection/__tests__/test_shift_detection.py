"""Tests for ML-based shift detection processor."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.processors.base import ProcessingContext
from src.processors.detection.shift_detection_processor import ShiftDetectionProcessor
from src.schemas.models.config import ShiftDetectionConfig, ThresholdingConfig


class MockTuningConfig:
    """Mock tuning config for testing."""

    def __init__(self) -> None:
        self.thresholding = ThresholdingConfig()


class MockTemplate:
    """Mock template for testing."""

    def __init__(self) -> None:
        self.field_blocks = [
            MockFieldBlock("MCQBlock1a1"),
            MockFieldBlock("MCQBlock1a2"),
            MockFieldBlock("Booklet_No"),
        ]
        self.tuning_config = MockTuningConfig()
        self.all_fields = []
        self.all_field_detection_types = []
        self.path = Path("test_template.json")


class MockFieldBlock:
    """Mock field block for testing."""

    def __init__(self, name) -> None:
        self.name = name
        self.shifts = [0, 0]
        self.origin = [100, 200]
        self.bounding_box_dimensions = [400, 300]

    def reset_all_shifts(self):
        self.shifts = [0, 0]

    def get_shifted_origin(self):
        return [self.origin[0] + self.shifts[0], self.origin[1] + self.shifts[1]]


class TestShiftDetectionProcessor:
    """Test suite for ShiftDetectionProcessor."""

    @pytest.fixture
    def create_processor(self):
        """Create a processor with mocked TemplateFileRunner."""

        def _create(config: ShiftDetectionConfig) -> ShiftDetectionProcessor:
            template = MockTemplate()
            processor = ShiftDetectionProcessor.__new__(ShiftDetectionProcessor)
            processor.template = template
            processor.shift_config = config
            processor.template_file_runner = Mock()  # Mock the file runner
            processor.stats = {
                "shifts_applied": 0,
                "shifts_rejected": 0,
                "mismatches_detected": 0,
                "confidence_reductions": [],
            }
            return processor

        return _create

    def test_validate_shifts_within_global_margin(self, create_processor):
        """Test that shifts within global margin are accepted."""
        config = ShiftDetectionConfig(
            enabled=True,
            global_max_shift_pixels=50,
            per_block_max_shift_pixels={},
        )
        processor = create_processor(config)

        ml_alignments = {
            "MCQBlock1a1": {"shift": [20, 30]},
            "MCQBlock1a2": {"shift": [-15, 25]},
        }

        validated = processor._validate_shifts(ml_alignments)

        assert len(validated) == 2
        assert "MCQBlock1a1" in validated
        assert "MCQBlock1a2" in validated
        assert validated["MCQBlock1a1"]["dx"] == 20
        assert validated["MCQBlock1a1"]["dy"] == 30
        assert processor.stats["shifts_applied"] == 2
        assert processor.stats["shifts_rejected"] == 0

    def test_validate_shifts_exceeds_global_margin(self, create_processor):
        """Test that shifts exceeding global margin are rejected."""
        config = ShiftDetectionConfig(
            enabled=True,
            global_max_shift_pixels=30,
            per_block_max_shift_pixels={},
        )
        processor = create_processor(config)

        ml_alignments = {
            "MCQBlock1a1": {"shift": [20, 20]},  # magnitude = 28.3 < 30 ✓
            "MCQBlock1a2": {"shift": [40, 40]},  # magnitude = 56.6 > 30 ✗
        }

        validated = processor._validate_shifts(ml_alignments)

        assert len(validated) == 1
        assert "MCQBlock1a1" in validated
        assert "MCQBlock1a2" not in validated
        assert processor.stats["shifts_applied"] == 1
        assert processor.stats["shifts_rejected"] == 1

    def test_validate_shifts_per_block_override(self, create_processor):
        """Test that per-block margins override global margin."""
        config = ShiftDetectionConfig(
            enabled=True,
            global_max_shift_pixels=30,
            per_block_max_shift_pixels={
                "MCQBlock1a1": 20,  # More restrictive
                "MCQBlock1a2": 50,  # More permissive
            },
        )
        processor = create_processor(config)

        ml_alignments = {
            "MCQBlock1a1": {"shift": [15, 15]},  # magnitude = 21.2 > 20 ✗
            "MCQBlock1a2": {"shift": [30, 30]},  # magnitude = 42.4 < 50 ✓
            "Booklet_No": {"shift": [20, 20]},  # magnitude = 28.3 < 30 (global) ✓
        }

        validated = processor._validate_shifts(ml_alignments)

        assert len(validated) == 2
        assert "MCQBlock1a1" not in validated
        assert "MCQBlock1a2" in validated
        assert "Booklet_No" in validated
        assert processor.stats["shifts_applied"] == 2
        assert processor.stats["shifts_rejected"] == 1

    def test_compare_bubbles_identical(self, create_processor):
        """Test bubble comparison with identical values."""
        config = ShiftDetectionConfig(enabled=True)
        processor = create_processor(config)

        shifted = {"bubble_values": [0, 1, 0, 1, 0]}
        baseline = {"bubble_values": [0, 1, 0, 1, 0]}

        diffs = processor._compare_bubbles(shifted, baseline)

        assert len(diffs) == 0

    def test_compare_bubbles_differences(self, create_processor):
        """Test bubble comparison with differences."""
        config = ShiftDetectionConfig(enabled=True)
        processor = create_processor(config)

        shifted = {"bubble_values": [0, 1, 0, 1, 0]}
        baseline = {"bubble_values": [0, 0, 0, 1, 1]}

        diffs = processor._compare_bubbles(shifted, baseline)

        assert len(diffs) == 2
        assert diffs[0] == {"index": 1, "shifted": 1, "baseline": 0}
        assert diffs[1] == {"index": 4, "shifted": 0, "baseline": 1}

    def test_compare_field_responses_identical(self, create_processor):
        """Test field response comparison with identical responses."""
        config = ShiftDetectionConfig(enabled=True)
        processor = create_processor(config)

        shifted = {"response": "A"}
        baseline = {"response": "A"}

        diff = processor._compare_field_responses(shifted, baseline)

        assert diff is None

    def test_compare_field_responses_different(self, create_processor):
        """Test field response comparison with different responses."""
        config = ShiftDetectionConfig(enabled=True)
        processor = create_processor(config)

        shifted = {"response": "B"}
        baseline = {"response": "A"}

        diff = processor._compare_field_responses(shifted, baseline)

        assert diff is not None
        assert diff["shifted"] == "B"
        assert diff["baseline"] == "A"

    def test_calculate_confidence_reduction(self, create_processor):
        """Test confidence reduction calculation."""
        config = ShiftDetectionConfig(
            enabled=True,
            confidence_reduction_min=0.1,
            confidence_reduction_max=0.5,
        )
        processor = create_processor(config)

        # Test minimum (no mismatch)
        assert processor._calculate_confidence_reduction(0.0) == pytest.approx(0.1)

        # Test maximum (complete mismatch)
        assert processor._calculate_confidence_reduction(1.0) == pytest.approx(0.5)

        # Test middle (50% mismatch)
        assert processor._calculate_confidence_reduction(0.5) == pytest.approx(0.3)

        # Test quarter (25% mismatch)
        assert processor._calculate_confidence_reduction(0.25) == pytest.approx(0.2)

    def test_find_block_by_name(self, create_processor):
        """Test finding blocks by name."""
        config = ShiftDetectionConfig(enabled=True)
        processor = create_processor(config)

        # Find existing block
        block = processor._find_block_by_name("MCQBlock1a1")
        assert block is not None
        assert block.name == "MCQBlock1a1"

        # Find non-existent block
        block = processor._find_block_by_name("NonExistent")
        assert block is None

    def test_process_no_ml_alignments(self, create_processor):
        """Test processor with no ML alignments in context."""
        config = ShiftDetectionConfig(enabled=True)
        processor = create_processor(config)

        context = ProcessingContext(
            file_path="test.jpg",
            gray_image=None,
            colored_image=None,
            template=processor.template,
            metadata={},
        )

        result = processor.process(context)

        # Should return context unchanged
        assert result is context
        assert "shift_detection" not in result.metadata

    def test_process_shift_detection_disabled(self, create_processor):
        """Test processor with shift detection disabled."""
        config = ShiftDetectionConfig(enabled=False)
        processor = create_processor(config)

        context = ProcessingContext(
            file_path="test.jpg",
            gray_image=None,
            colored_image=None,
            template=processor.template,
            metadata={
                "ml_block_alignments": {
                    "MCQBlock1a1": {"shift": [10, 15]},
                }
            },
        )

        result = processor.process(context)

        # Should return context unchanged even with alignments
        assert result is context
        assert "shift_detection" not in result.metadata

    def test_processor_name(self, create_processor):
        """Test processor name."""
        config = ShiftDetectionConfig(enabled=True)
        processor = create_processor(config)

        assert processor.get_name() == "ShiftDetection"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
