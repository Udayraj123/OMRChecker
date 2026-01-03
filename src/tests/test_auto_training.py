"""Tests for auto-training feature."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestTrainingDataCollector:
    """Tests for training data collection."""

    def test_collector_initialization(self):
        """Test collector initializes with correct parameters."""
        from src.processors.training.data_collector import TrainingDataCollector

        template = MagicMock()
        collector = TrainingDataCollector(template, confidence_threshold=0.9)

        assert collector.confidence_threshold == 0.9
        assert collector.export_dir == Path("outputs/training_data")
        assert collector.stats["total_processed"] == 0

    def test_confidence_filtering(self):
        """Test that only high-confidence samples are collected."""
        from src.processors.base import ProcessingContext
        from src.processors.training.data_collector import TrainingDataCollector

        template = MagicMock()
        collector = TrainingDataCollector(template, confidence_threshold=0.8)

        # Mock interpretation with low confidence
        low_conf_interp = MagicMock()
        low_conf_interp.field_level_confidence_metrics = {
            "overall_confidence_score": 0.5
        }

        # Mock interpretation with high confidence
        high_conf_interp = MagicMock()
        high_conf_interp.field_level_confidence_metrics = {
            "overall_confidence_score": 0.9
        }

        context = ProcessingContext(
            file_path="test.jpg",
            gray_image=MagicMock(),
            colored_image=MagicMock(),
            template=template,
        )
        context.field_id_to_interpretation = {
            "field1": low_conf_interp,
            "field2": high_conf_interp,
        }

        # Process
        result = collector.process(context)

        # Should skip low confidence
        assert collector.stats["total_processed"] == 1


class TestYOLOAnnotationExporter:
    """Tests for YOLO format export."""

    def test_roi_to_yolo_conversion(self):
        """Test conversion of ROI to YOLO format."""
        from src.processors.training.yolo_exporter import YOLOAnnotationExporter

        exporter = YOLOAnnotationExporter(Path("test_dataset"))

        roi = {
            "bbox": {"x": 100, "y": 200, "width": 50, "height": 30},
            "class": "bubble_filled",
        }

        yolo_line = exporter._convert_roi_to_yolo(roi, img_width=1000, img_height=1000)

        assert yolo_line is not None
        parts = yolo_line.split()
        assert len(parts) == 5  # class_id, x_center, y_center, width, height
        assert parts[0] == "1"  # bubble_filled class
        # Verify normalized coordinates
        x_center = float(parts[1])
        y_center = float(parts[2])
        assert 0.0 <= x_center <= 1.0
        assert 0.0 <= y_center <= 1.0

    def test_invalid_bbox_handling(self):
        """Test that invalid bboxes are skipped."""
        from src.processors.training.yolo_exporter import YOLOAnnotationExporter

        exporter = YOLOAnnotationExporter(Path("test_dataset"))

        # Invalid bbox (zero width)
        invalid_roi = {
            "bbox": {"x": 100, "y": 200, "width": 0, "height": 30},
            "class": "bubble_empty",
        }

        result = exporter._convert_roi_to_yolo(invalid_roi, 1000, 1000)
        assert result is None


class TestMLBubbleDetector:
    """Tests for ML bubble detector."""

    @patch("src.processors.detection.ml_detector.YOLO")
    def test_ml_detector_initialization(self, mock_yolo):
        """Test ML detector initializes correctly."""
        from src.processors.detection.ml_detector import MLBubbleDetector

        model_path = Path("test_model.pt")
        detector = MLBubbleDetector(model_path, confidence_threshold=0.75)

        assert detector.confidence_threshold == 0.75
        assert not detector.enabled  # Starts disabled

    def test_ml_detector_enable_disable(self):
        """Test enabling/disabling ML detector."""
        from src.processors.detection.ml_detector import MLBubbleDetector

        detector = MLBubbleDetector("nonexistent.pt")

        assert not detector.enabled
        detector.enable_for_low_confidence()
        assert detector.enabled
        detector.disable()
        assert not detector.enabled


class TestHybridDetectionStrategy:
    """Tests for hybrid detection strategy."""

    def test_low_confidence_identification(self):
        """Test identification of low-confidence fields."""
        from src.processors.detection.ml_detector import HybridDetectionStrategy

        strategy = HybridDetectionStrategy(None, confidence_threshold=0.75)

        # Mock interpretations
        high_conf = MagicMock()
        high_conf.field_level_confidence_metrics = {"overall_confidence_score": 0.9}

        low_conf = MagicMock()
        low_conf.field_level_confidence_metrics = {"overall_confidence_score": 0.6}

        field_id_to_interpretation = {
            "field1": high_conf,
            "field2": low_conf,
            "field3": high_conf,
        }

        low_conf_fields = strategy.identify_low_confidence_fields(
            field_id_to_interpretation
        )

        assert len(low_conf_fields) == 1
        assert low_conf_fields[0][0] == "field2"
        assert low_conf_fields[0][1] == 0.6

    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        from src.processors.detection.ml_detector import HybridDetectionStrategy

        strategy = HybridDetectionStrategy(None, confidence_threshold=0.75)

        high_conf = MagicMock()
        high_conf.field_level_confidence_metrics = {"overall_confidence_score": 0.9}

        low_conf = MagicMock()
        low_conf.field_level_confidence_metrics = {"overall_confidence_score": 0.6}

        field_id_to_interpretation = {
            "field1": high_conf,
            "field2": low_conf,
        }

        strategy.identify_low_confidence_fields(field_id_to_interpretation)

        stats = strategy.get_statistics()
        assert stats["total_fields"] == 2
        assert stats["high_confidence_fields"] == 1
        assert stats["low_confidence_fields"] == 1


class TestConfidenceScoreCalculation:
    """Tests for confidence score calculation in interpretation."""

    def test_confidence_score_present(self):
        """Test that confidence score is calculated."""
        # This would require mocking the full interpretation pipeline
        # For now, we verify the structure is in place
        from src.processors.detection.bubbles_threshold.interpretation import (
            BubblesFieldInterpretation,
        )

        # Verify method exists
        assert hasattr(
            BubblesFieldInterpretation, "_calculate_overall_confidence_score"
        )


class TestCLIArguments:
    """Tests for CLI argument parsing."""

    def test_training_args_parsed(self):
        """Test that training-related arguments are parsed correctly."""
        import sys
        from pathlib import Path

        # Add project root to path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))

        from main import parse_args

        # Test collect-training-data flag
        test_args = ["--collect-training-data", "--confidence-threshold", "0.9"]
        sys.argv = ["main.py"] + test_args

        args = parse_args()
        assert args["collect_training_data"] is True
        assert args["confidence_threshold"] == 0.9

    def test_mode_selection(self):
        """Test that different modes are parsed correctly."""
        import sys
        from pathlib import Path

        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))

        from main import parse_args

        # Test auto-train mode
        test_args = ["--mode", "auto-train"]
        sys.argv = ["main.py"] + test_args

        args = parse_args()
        assert args["mode"] == "auto-train"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
