"""Tests for drawing utility functions."""

import numpy as np

from src.utils.drawing import DrawingUtils


class TestDrawingUtils:
    """Test suite for drawing utility functions."""

    def test_draw_box_diagonal(self) -> None:
        """Test drawing box with diagonal corners."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        position = (10, 10)
        position_diagonal = (90, 90)
        DrawingUtils.draw_box_diagonal(image, position, position_diagonal)
        # Verify image was modified (not all zeros anymore)
        assert np.any(image > 0)

    def test_draw_box_hollow(self) -> None:
        """Test drawing hollow box."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        position = (10, 10)
        dimensions = (50, 50)
        DrawingUtils.draw_box(image, position, dimensions, style="BOX_HOLLOW")
        assert np.any(image > 0)

    def test_draw_box_filled(self) -> None:
        """Test drawing filled box."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        position = (10, 10)
        dimensions = (50, 50)
        DrawingUtils.draw_box(image, position, dimensions, style="BOX_FILLED")
        assert np.any(image > 0)

    def test_draw_box_centered(self) -> None:
        """Test drawing centered box."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        position = (50, 50)
        dimensions = (30, 30)
        DrawingUtils.draw_box(
            image, position, dimensions, style="BOX_HOLLOW", centered=True
        )
        assert np.any(image > 0)

    def test_draw_text(self) -> None:
        """Test drawing text on image."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        text = "Test"
        position = (10, 50)
        color = (255, 255, 255)  # White color
        DrawingUtils.draw_text(image, text, position, color=color)
        assert np.any(image > 0)

    def test_draw_text_centered(self) -> None:
        """Test drawing centered text."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        text = "Test"
        position = (50, 50)
        color = (255, 255, 255)  # White color
        DrawingUtils.draw_text(image, text, position, centered=True, color=color)
        assert np.any(image > 0)

    def test_draw_line(self) -> None:
        """Test drawing line."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        start = (10, 10)
        end = (90, 90)
        color = (255, 255, 255)  # White color
        DrawingUtils.draw_line(image, start, end, color=color)
        assert np.any(image > 0)

    def test_draw_polygon_closed(self) -> None:
        """Test drawing closed polygon."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        points = [(20, 20), (80, 20), (80, 80), (20, 80)]
        color = (255, 255, 255)  # White color
        DrawingUtils.draw_polygon(image, points, closed=True, color=color)
        assert np.any(image > 0)

    def test_draw_polygon_open(self) -> None:
        """Test drawing open polygon."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        points = [(20, 20), (80, 20), (80, 80)]
        color = (255, 255, 255)  # White color
        DrawingUtils.draw_polygon(image, points, closed=False, color=color)
        assert np.any(image > 0)

    def test_draw_contour(self) -> None:
        """Test drawing contour."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        contour = np.array([[10, 10], [90, 10], [90, 90], [10, 90]])
        DrawingUtils.draw_contour(image, contour)
        assert np.any(image > 0)
