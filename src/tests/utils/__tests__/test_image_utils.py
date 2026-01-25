"""Tests for ImageUtils.load_image consolidation."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.exceptions import ImageReadError
from src.utils.image import ImageUtils


def test_load_image_grayscale_success(tmp_path):
    """Test loading a grayscale image successfully."""
    # Create a simple test image
    test_image_path = tmp_path / "test_gray.png"
    gray_image = np.zeros((100, 100), dtype=np.uint8)
    cv2.imwrite(str(test_image_path), gray_image)

    # Load it using the utility
    loaded = ImageUtils.load_image(test_image_path, cv2.IMREAD_GRAYSCALE)

    assert loaded is not None
    assert loaded.shape == (100, 100)


def test_load_image_color_success(tmp_path):
    """Test loading a color image successfully."""
    # Create a simple test image
    test_image_path = tmp_path / "test_color.png"
    color_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(test_image_path), color_image)

    # Load it using the utility
    loaded = ImageUtils.load_image(test_image_path, cv2.IMREAD_COLOR)

    assert loaded is not None
    assert len(loaded.shape) == 3
    assert loaded.shape[2] == 3


def test_load_image_nonexistent_file():
    """Test that loading a non-existent file raises ImageReadError."""
    nonexistent_path = Path("/nonexistent/path/to/image.png")

    with pytest.raises(ImageReadError) as exc_info:
        ImageUtils.load_image(nonexistent_path, cv2.IMREAD_GRAYSCALE)

    assert "OpenCV returned None" in str(exc_info.value)
    assert "grayscale" in str(exc_info.value)


def test_load_image_invalid_file(tmp_path):
    """Test that loading an invalid image file raises ImageReadError."""
    # Create a text file, not an image
    invalid_path = tmp_path / "not_an_image.png"
    invalid_path.write_text("This is not an image")

    with pytest.raises(ImageReadError) as exc_info:
        ImageUtils.load_image(invalid_path, cv2.IMREAD_GRAYSCALE)

    assert "OpenCV returned None" in str(exc_info.value)


def test_load_image_unchanged_mode(tmp_path):
    """Test loading an image with IMREAD_UNCHANGED mode."""
    # Create a test image with alpha channel
    test_image_path = tmp_path / "test_unchanged.png"
    image_with_alpha = np.zeros((100, 100, 4), dtype=np.uint8)
    cv2.imwrite(str(test_image_path), image_with_alpha)

    # Load it using the utility
    loaded = ImageUtils.load_image(test_image_path, cv2.IMREAD_UNCHANGED)

    assert loaded is not None
    assert loaded.shape[:2] == (100, 100)


def test_load_image_context_in_exception():
    """Test that ImageReadError contains proper context."""
    nonexistent_path = Path("/test/missing.png")

    with pytest.raises(ImageReadError) as exc_info:
        ImageUtils.load_image(nonexistent_path, cv2.IMREAD_COLOR)

    exception = exc_info.value
    assert exception.path == nonexistent_path
    assert "color" in str(exception)
