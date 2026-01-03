"""Tests for data augmentation functionality."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from augment_data import DataAugmenter


class TestDataAugmenter:
    """Test suite for DataAugmenter."""

    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Create temporary directories for testing."""
        source_images = tmp_path / "source_images"
        source_labels = tmp_path / "source_labels"
        output = tmp_path / "output"

        source_images.mkdir()
        source_labels.mkdir()
        output.mkdir()

        return {
            "source_images": source_images,
            "source_labels": source_labels,
            "output": output,
        }

    @pytest.fixture
    def sample_image(self):
        """Create a sample OMR-like image."""
        # Create a white background image with a dark field block
        image = np.ones((1000, 800, 3), dtype=np.uint8) * 255

        # Add a dark field block (simulating MCQ answers)
        cv2.rectangle(image, (100, 200), (500, 500), (200, 200, 200), -1)

        # Add some bubbles
        cv2.circle(image, (150, 250), 15, (50, 50, 50), -1)
        cv2.circle(image, (200, 250), 15, (50, 50, 50), -1)
        cv2.circle(image, (150, 300), 15, (50, 50, 50), -1)

        return image

    @pytest.fixture
    def sample_labels(self):
        """Create sample label data."""
        return {
            "rois": [
                {
                    "bbox": {"x": 100, "y": 200, "width": 400, "height": 300},
                    "label": "MCQBlock1",
                    "field_type": "bubbles",
                }
            ]
        }

    def test_augmenter_initialization(self, temp_dirs):
        """Test DataAugmenter initialization."""
        augmenter = DataAugmenter(
            source_images_dir=temp_dirs["source_images"],
            source_labels_dir=temp_dirs["source_labels"],
            output_dir=temp_dirs["output"],
        )

        assert augmenter.source_images_dir == temp_dirs["source_images"]
        assert augmenter.augmented_images_dir.exists()
        assert augmenter.augmented_labels_dir.exists()

    def test_background_detection(self, sample_image):
        """Test background color detection."""
        augmenter = DataAugmenter(
            source_images_dir=Path("dummy"),
            source_labels_dir=Path("dummy"),
            output_dir=Path("dummy"),
        )

        bg_color = augmenter._detect_background_color(sample_image)

        # Should detect white background (or close to it)
        assert bg_color.shape == (3,)
        assert np.all(bg_color > 200)  # Close to white (255)

    def test_field_block_shifting(self, sample_image, sample_labels):
        """Test field block shifting augmentation."""
        augmenter = DataAugmenter(
            source_images_dir=Path("dummy"),
            source_labels_dir=Path("dummy"),
            output_dir=Path("dummy"),
        )

        shifted_image, shifted_labels = augmenter._shift_field_blocks(
            sample_image.copy(), sample_labels, max_shift=20
        )

        # Check that image is modified
        assert shifted_image.shape == sample_image.shape
        assert not np.array_equal(shifted_image, sample_image)

        # Check that labels are updated
        assert "rois" in shifted_labels
        roi = shifted_labels["rois"][0]

        # Check shift metadata exists
        assert "shift" in roi
        assert "dx" in roi["shift"]
        assert "dy" in roi["shift"]

        # Check shift is within bounds
        assert abs(roi["shift"]["dx"]) <= 20
        assert abs(roi["shift"]["dy"]) <= 20

        # Check bbox is updated
        original_x = sample_labels["rois"][0]["bbox"]["x"]
        original_y = sample_labels["rois"][0]["bbox"]["y"]
        new_x = roi["bbox"]["x"]
        new_y = roi["bbox"]["y"]

        assert new_x == original_x + roi["shift"]["dx"]
        assert new_y == original_y + roi["shift"]["dy"]

    def test_field_block_shifting_boundary_check(self, sample_image, sample_labels):
        """Test that field blocks don't shift outside image boundaries."""
        augmenter = DataAugmenter(
            source_images_dir=Path("dummy"),
            source_labels_dir=Path("dummy"),
            output_dir=Path("dummy"),
        )

        # Create labels with block near edge
        edge_labels = {
            "rois": [
                {
                    "bbox": {"x": 10, "y": 10, "width": 100, "height": 100},
                    "label": "EdgeBlock",
                }
            ]
        }

        shifted_image, shifted_labels = augmenter._shift_field_blocks(
            sample_image.copy(), edge_labels, max_shift=50
        )

        roi = shifted_labels["rois"][0]
        bbox = roi["bbox"]

        # Check boundaries
        assert bbox["x"] >= 0
        assert bbox["y"] >= 0
        assert bbox["x"] + bbox["width"] <= sample_image.shape[1]
        assert bbox["y"] + bbox["height"] <= sample_image.shape[0]

    def test_brightness_adjustment(self, sample_image):
        """Test brightness adjustment."""
        augmenter = DataAugmenter(
            source_images_dir=Path("dummy"),
            source_labels_dir=Path("dummy"),
            output_dir=Path("dummy"),
        )

        bright = augmenter._adjust_brightness(sample_image.copy(), factor=1.3)
        dark = augmenter._adjust_brightness(sample_image.copy(), factor=0.7)

        # Bright image should have higher mean
        assert bright.mean() > sample_image.mean()
        assert dark.mean() < sample_image.mean()

    def test_contrast_adjustment(self, sample_image):
        """Test contrast adjustment."""
        augmenter = DataAugmenter(
            source_images_dir=Path("dummy"),
            source_labels_dir=Path("dummy"),
            output_dir=Path("dummy"),
        )

        high_contrast = augmenter._adjust_contrast(sample_image.copy(), factor=1.5)
        low_contrast = augmenter._adjust_contrast(sample_image.copy(), factor=0.5)

        # High contrast should have higher standard deviation
        assert high_contrast.std() > sample_image.std()
        assert low_contrast.std() < sample_image.std()

    def test_gaussian_noise(self, sample_image):
        """Test Gaussian noise addition."""
        augmenter = DataAugmenter(
            source_images_dir=Path("dummy"),
            source_labels_dir=Path("dummy"),
            output_dir=Path("dummy"),
        )

        noisy = augmenter._add_gaussian_noise(sample_image.copy(), sigma=10)

        # Should be different from original
        assert not np.array_equal(noisy, sample_image)
        # But similar mean
        assert abs(noisy.mean() - sample_image.mean()) < 20

    def test_blur(self, sample_image):
        """Test blur augmentation."""
        augmenter = DataAugmenter(
            source_images_dir=Path("dummy"),
            source_labels_dir=Path("dummy"),
            output_dir=Path("dummy"),
        )

        blurred = augmenter._add_blur(sample_image.copy(), kernel_size=5)

        # Blurred image should have lower variance
        assert blurred.std() < sample_image.std()

    def test_rotation(self, sample_image, sample_labels):
        """Test rotation augmentation."""
        augmenter = DataAugmenter(
            source_images_dir=Path("dummy"),
            source_labels_dir=Path("dummy"),
            output_dir=Path("dummy"),
        )

        rotated, rotated_labels = augmenter._rotate_image(
            sample_image.copy(), sample_labels, angle=5
        )

        # Should be different from original
        assert not np.array_equal(rotated, sample_image)

        # Labels should be updated
        assert "rois" in rotated_labels
        original_bbox = sample_labels["rois"][0]["bbox"]
        rotated_bbox = rotated_labels["rois"][0]["bbox"]

        # Position should have changed (approximate check)
        pos_changed = (
            original_bbox["x"] != rotated_bbox["x"]
            or original_bbox["y"] != rotated_bbox["y"]
        )
        assert pos_changed

    def test_augmentation_types(self, temp_dirs, sample_image, sample_labels):
        """Test that all augmentation types are applied."""
        import json

        # Save sample data
        image_path = temp_dirs["source_images"] / "test.jpg"
        label_path = temp_dirs["source_labels"] / "test.json"

        cv2.imwrite(str(image_path), sample_image)
        with label_path.open("w") as f:
            json.dump(sample_labels, f)

        # Run augmentation
        augmenter = DataAugmenter(
            source_images_dir=temp_dirs["source_images"],
            source_labels_dir=temp_dirs["source_labels"],
            output_dir=temp_dirs["output"],
        )

        # Generate 14 augmentations (1 more than combination count to cycle)
        stats = augmenter.augment_dataset(target_count=15)

        # Check that augmentations were generated
        assert stats["original_count"] == 1
        assert stats["generated_count"] == 14  # 15 - 1 original
        assert stats["combination_count"] == 13  # Should report 13 combinations

        # Check that files were created
        aug_images = list(augmenter.augmented_images_dir.glob("*.jpg"))
        aug_labels = list(augmenter.augmented_labels_dir.glob("*.json"))

        assert len(aug_images) == 14
        assert len(aug_labels) == 14

        # Verify normalization happened on most samples (should be ~85%)
        # With 14 samples, expect 10-13 to be normalized
        assert stats["normalized_count"] >= 8, (
            f"Expected ~85% normalized, got {stats['normalized_count']}/14"
        )

        # Verify shift metadata exists in shift combinations (types 7, 8, 9)
        shift_found = 0
        for label_file in aug_labels:
            with label_file.open() as f:
                data = json.load(f)
                if data.get("rois") and "shift" in data["rois"][0]:
                    shift_found += 1

        # Should find at least one shift (type 7, 8, or 9 appear once each in 13)
        assert shift_found >= 1, "No augmentation with shift metadata found"

    # Normalization tests
    def test_normalize_field_block(self, sample_image, sample_labels):
        """Test single field block normalization."""
        augmenter = DataAugmenter(
            source_images_dir=Path("dummy"),
            source_labels_dir=Path("dummy"),
            output_dir=Path("dummy"),
        )

        roi = sample_labels["rois"][0]
        normalized = augmenter._normalize_field_block(sample_image.copy(), roi)

        # Check that image is modified within the ROI
        bbox = roi["bbox"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

        # The field block area should be uniform
        block_region = normalized[y : y + h, x : x + w]
        # Check if region has low variance (uniform)
        assert block_region.std() < 10, "Normalized block should be uniform"

        # Rest of image should be unchanged
        assert np.array_equal(normalized[0:y, :], sample_image[0:y, :]), (
            "Area above block should be unchanged"
        )

    def test_normalize_sample(self, sample_image, sample_labels):
        """Test full sample normalization."""
        augmenter = DataAugmenter(
            source_images_dir=Path("dummy"),
            source_labels_dir=Path("dummy"),
            output_dir=Path("dummy"),
        )

        # Force normalization (prob=1.0)
        normalized, labels = augmenter._normalize_sample(
            sample_image.copy(), sample_labels.copy(), normalization_prob=1.0
        )

        # Should be different from original
        assert not np.array_equal(normalized, sample_image)

        # Labels should be unchanged
        assert labels == sample_labels

    def test_normalization_probability(self, sample_image, sample_labels):
        """Test normalization probability works."""
        augmenter = DataAugmenter(
            source_images_dir=Path("dummy"),
            source_labels_dir=Path("dummy"),
            output_dir=Path("dummy"),
        )

        # Test with 0% probability (should never normalize)
        normalized, _ = augmenter._normalize_sample(
            sample_image.copy(), sample_labels.copy(), normalization_prob=0.0
        )
        assert np.array_equal(normalized, sample_image)

        # Test with 100% probability (should always normalize)
        normalized, _ = augmenter._normalize_sample(
            sample_image.copy(), sample_labels.copy(), normalization_prob=1.0
        )
        assert not np.array_equal(normalized, sample_image)

    def test_normalized_preserves_labels(self, sample_image, sample_labels):
        """Test that normalization doesn't modify labels."""
        augmenter = DataAugmenter(
            source_images_dir=Path("dummy"),
            source_labels_dir=Path("dummy"),
            output_dir=Path("dummy"),
        )

        original_labels = sample_labels.copy()
        _, normalized_labels = augmenter._normalize_sample(
            sample_image.copy(), sample_labels.copy(), normalization_prob=1.0
        )

        # Labels should be identical
        assert normalized_labels == original_labels

    def test_background_color_for_normalization(self, sample_image, sample_labels):
        """Test that correct fill color is detected for normalization."""
        augmenter = DataAugmenter(
            source_images_dir=Path("dummy"),
            source_labels_dir=Path("dummy"),
            output_dir=Path("dummy"),
        )

        bg_color = augmenter._detect_background_color(sample_image)

        # Should be white or close to white
        assert np.all(bg_color > 200)

        # Normalize with detected background
        roi = sample_labels["rois"][0]
        normalized = augmenter._normalize_field_block(
            sample_image.copy(), roi, fill_color=bg_color
        )

        bbox = roi["bbox"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        block_region = normalized[y : y + h, x : x + w]

        # Check that block is filled with background color
        assert np.allclose(block_region.mean(), bg_color.mean(), atol=5)

    # Combined augmentation tests
    def test_combined_augmentation_2_types(self, sample_image, sample_labels):
        """Test 2-type combinations work correctly."""
        augmenter = DataAugmenter(
            source_images_dir=Path("dummy"),
            source_labels_dir=Path("dummy"),
            output_dir=Path("dummy"),
        )

        # Test brightness + noise (combination type 0)
        aug_image, aug_labels = augmenter._apply_augmentation(
            sample_image.copy(), sample_labels, aug_idx=0
        )

        assert not np.array_equal(aug_image, sample_image)
        assert aug_labels is not None

    def test_combined_augmentation_3_types(self, sample_image, sample_labels):
        """Test 3-type combinations work correctly."""
        augmenter = DataAugmenter(
            source_images_dir=Path("dummy"),
            source_labels_dir=Path("dummy"),
            output_dir=Path("dummy"),
        )

        # Test brightness + noise + blur (combination type 4)
        aug_image, aug_labels = augmenter._apply_augmentation(
            sample_image.copy(), sample_labels, aug_idx=4
        )

        assert not np.array_equal(aug_image, sample_image)
        assert aug_labels is not None

    def test_combined_augmentation_4_types(self, sample_image, sample_labels):
        """Test 4-type combination works correctly."""
        augmenter = DataAugmenter(
            source_images_dir=Path("dummy"),
            source_labels_dir=Path("dummy"),
            output_dir=Path("dummy"),
        )

        # Test brightness + contrast + noise + blur (combination type 12)
        aug_image, aug_labels = augmenter._apply_augmentation(
            sample_image.copy(), sample_labels, aug_idx=12
        )

        assert not np.array_equal(aug_image, sample_image)
        assert aug_labels is not None

    def test_shift_with_photometric(self, sample_image, sample_labels):
        """Test shift combined with photometric augmentations."""
        augmenter = DataAugmenter(
            source_images_dir=Path("dummy"),
            source_labels_dir=Path("dummy"),
            output_dir=Path("dummy"),
        )

        # Test shift + brightness (combination type 7)
        aug_image, aug_labels = augmenter._apply_augmentation(
            sample_image.copy(), sample_labels, aug_idx=7
        )

        # Should have shift metadata
        assert "rois" in aug_labels
        assert "shift" in aug_labels["rois"][0]
        # Image should be different
        assert not np.array_equal(aug_image, sample_image)

    def test_rotation_not_with_shift(self, sample_image, sample_labels):
        """Verify rotation and shift don't combine."""
        augmenter = DataAugmenter(
            source_images_dir=Path("dummy"),
            source_labels_dir=Path("dummy"),
            output_dir=Path("dummy"),
        )

        # Check all combinations
        for combination in augmenter.AUGMENTATION_COMBINATIONS:
            # Rotation and shift should never appear together
            has_rotation = "rotation" in combination
            has_shift = "shift" in combination
            assert not (has_rotation and has_shift), (
                f"Rotation and shift both in: {combination}"
            )

    def test_augmentation_order(self, sample_image, sample_labels):
        """Test that geometric augmentations are applied after photometric."""
        augmenter = DataAugmenter(
            source_images_dir=Path("dummy"),
            source_labels_dir=Path("dummy"),
            output_dir=Path("dummy"),
        )

        # Test with shift + brightness + noise (type 9)
        aug_image, aug_labels = augmenter._apply_augmentation(
            sample_image.copy(), sample_labels, aug_idx=9
        )

        # Should have shift metadata (geometric was applied)
        assert "shift" in aug_labels["rois"][0]
        # Image should be different (all augmentations applied)
        assert not np.array_equal(aug_image, sample_image)

    def test_shift_metadata_in_combination(self, sample_image, sample_labels):
        """Test that shift metadata is preserved in combinations."""
        augmenter = DataAugmenter(
            source_images_dir=Path("dummy"),
            source_labels_dir=Path("dummy"),
            output_dir=Path("dummy"),
        )

        # Test shift combinations (types 7, 8, 9)
        for aug_idx in [7, 8, 9]:
            aug_image, aug_labels = augmenter._apply_augmentation(
                sample_image.copy(), sample_labels, aug_idx=aug_idx
            )

            # Should have shift metadata
            assert "shift" in aug_labels["rois"][0]
            assert "dx" in aug_labels["rois"][0]["shift"]
            assert "dy" in aug_labels["rois"][0]["shift"]

    def test_empty_labels_handling(self, sample_image):
        """Test handling of images with no ROIs."""
        augmenter = DataAugmenter(
            source_images_dir=Path("dummy"),
            source_labels_dir=Path("dummy"),
            output_dir=Path("dummy"),
        )

        empty_labels = {"rois": []}

        # Should not crash
        shifted, labels = augmenter._shift_field_blocks(
            sample_image.copy(), empty_labels, max_shift=20
        )

        assert np.array_equal(shifted, sample_image)
        assert labels == empty_labels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
