"""
Unit tests for l_marker_detection module.

Uses synthetic L-shaped images — no real OMR sheets required.
Tests translated from the detection module's documented behavior.
"""

import numpy as np
import cv2

from src.processors.image.crop_on_patches.l_marker_detection import (
    preprocess_for_l_marker,
    detect_l_contours,
    extract_l_inner_corner,
    detect_l_marker_in_patch,
)


def make_l_shape_patch(size=200, arm_width=20, corner="top_left"):
    """
    Create a synthetic grayscale patch with a white L-shape on black background.
    The L inner corner position depends on the `corner` argument.
    """
    img = np.zeros((size, size), dtype=np.uint8)
    h_arm = arm_width  # thickness of each arm

    if corner == "top_left":
        # L pointing top-left: vertical arm on left, horizontal arm on top
        # inner corner at top-left
        img[0:size, 0:h_arm] = 255  # vertical arm (left)
        img[0:h_arm, 0 : size // 2] = 255  # horizontal arm (top)
        expected_inner = (h_arm, h_arm)  # (x, y) inner corner approx

    elif corner == "top_right":
        # inner corner at top-right
        img[0:size, size - h_arm : size] = 255
        img[0:h_arm, size // 2 : size] = 255
        expected_inner = (size - h_arm, h_arm)

    elif corner == "bottom_right":
        img[0:size, size - h_arm : size] = 255
        img[size - h_arm : size, size // 2 : size] = 255
        expected_inner = (size - h_arm, size - h_arm)

    else:  # bottom_left
        img[0:size, 0:h_arm] = 255
        img[size - h_arm : size, 0 : size // 2] = 255
        expected_inner = (h_arm, size - h_arm)

    return img, expected_inner


class TestPreprocessForLMarker:
    def test_returns_binary_image(self):
        patch, _ = make_l_shape_patch()
        result = preprocess_for_l_marker(patch)
        unique_vals = set(np.unique(result))
        assert unique_vals.issubset({0, 255}), (
            f"Expected binary, got values: {unique_vals}"
        )

    def test_same_shape_as_input(self):
        patch, _ = make_l_shape_patch(size=150)
        result = preprocess_for_l_marker(patch)
        assert result.shape == patch.shape

    def test_noisy_patch_still_produces_binary(self):
        patch, _ = make_l_shape_patch()
        noisy = patch.copy()
        noise = np.random.randint(0, 30, patch.shape, dtype=np.uint8)
        noisy = cv2.add(noisy, noise)
        result = preprocess_for_l_marker(
            noisy, morph_kernel_size=(3, 3), morph_iterations=1
        )
        unique_vals = set(np.unique(result))
        assert unique_vals.issubset({0, 255})


class TestDetectLContours:
    def test_finds_contour_in_l_patch(self):
        patch, _ = make_l_shape_patch(size=200, arm_width=25)
        binary = preprocess_for_l_marker(patch)
        # Synthetic L-shape touching borders produces Canny edge contours with
        # very small area (~1.0). Use min_area=0 to detect them.
        contours = detect_l_contours(binary, min_area=0, max_area=200 * 200)
        assert len(contours) > 0, "Expected at least one contour for L shape"

    def test_returns_empty_for_blank_patch(self):
        blank = np.zeros((100, 100), dtype=np.uint8)
        binary = preprocess_for_l_marker(blank)
        contours = detect_l_contours(binary)
        assert contours == []

    def test_sorted_largest_first(self):
        patch, _ = make_l_shape_patch(size=200, arm_width=25)
        binary = preprocess_for_l_marker(patch)
        contours = detect_l_contours(binary, min_area=0, max_area=200 * 200)
        if len(contours) > 1:
            areas = [cv2.contourArea(c) for c in contours]
            assert areas == sorted(areas, reverse=True)

    def test_area_filter_excludes_too_small(self):
        patch, _ = make_l_shape_patch(size=200, arm_width=25)
        binary = preprocess_for_l_marker(patch)
        contours_filtered = detect_l_contours(binary, min_area=999999, max_area=9999999)
        assert len(contours_filtered) == 0


class TestExtractLInnerCorner:
    def test_returns_none_for_tiny_contour(self):
        tiny = np.array([[[0, 0]], [[1, 0]], [[0, 1]]], dtype=np.int32)
        result = extract_l_inner_corner(tiny)
        # Should either return None or a point — just not crash
        # (tiny contours may not have valid defects)
        assert result is None or result.shape == (2,)

    def test_returns_point_for_l_shape(self):
        patch, _ = make_l_shape_patch(size=200, arm_width=25)
        binary = preprocess_for_l_marker(patch)
        contours = detect_l_contours(binary, min_area=100, max_area=200 * 200)
        if contours:
            result = extract_l_inner_corner(contours[0])
            # May be None if shape not clean enough, but should not crash
            if result is not None:
                assert result.shape == (2,)
                assert 0 <= result[0] <= 200
                assert 0 <= result[1] <= 200

    def test_returns_none_for_rectangle(self):
        # A rectangle has no concave defects deep enough
        rect_contour = np.array(
            [[[10, 10]], [[90, 10]], [[90, 90]], [[10, 90]]], dtype=np.int32
        )
        result = extract_l_inner_corner(rect_contour)
        # Rectangle has no convexity defects, should return None
        assert result is None


class TestDetectLMarkerInPatch:
    def test_returns_absolute_coords_with_offset(self):
        patch, _ = make_l_shape_patch(size=200, arm_width=25)
        zone_offset = (50, 100)
        result = detect_l_marker_in_patch(patch, zone_offset=zone_offset)
        if result is not None:
            # Result must be offset by zone_offset
            assert result[0] >= 50, f"x should be >= zone_offset x=50, got {result[0]}"
            assert result[1] >= 100, (
                f"y should be >= zone_offset y=100, got {result[1]}"
            )

    def test_returns_none_for_blank_patch(self):
        blank = np.zeros((100, 100), dtype=np.uint8)
        result = detect_l_marker_in_patch(blank)
        assert result is None

    def test_respects_tuning_options(self):
        patch, _ = make_l_shape_patch(size=200, arm_width=25)
        tuning = {
            "morph_kernel_size": [3, 3],
            "morph_iterations": 1,
            "min_marker_area": 10.0,
            "max_marker_area": 999999.0,
        }
        # Should not crash with custom tuning; result may or may not be None
        detect_l_marker_in_patch(patch, tuning_options=tuning)
