"""Data augmentation utility to artificially expand training dataset.

Creates synthetic variations of existing samples using various transformations
while preserving annotations (bounding boxes, labels).
"""

import json
import random
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np

from src.utils.logger import logger


class DataAugmenter:
    """Augments OMR training data with realistic transformations."""

    # Define 13 augmentation combinations (2-4 types per sample)
    AUGMENTATION_COMBINATIONS: ClassVar[list[list[str]]] = [
        # 2-type combinations (non-geometric)
        ["brightness", "noise"],  # Type 0: Common scan variations
        ["contrast", "blur"],  # Type 1: Poor focus
        ["brightness", "contrast"],  # Type 2: Lighting issues
        ["noise", "blur"],  # Type 3: Low quality scan
        # 3-type combinations
        ["brightness", "noise", "blur"],  # Type 4: Multiple quality issues
        ["contrast", "noise", "blur"],  # Type 5: Degraded scan
        ["brightness", "contrast", "noise"],  # Type 6: Complex lighting + noise
        # Shift combinations (with non-geometric only)
        ["shift", "brightness"],  # Type 7: Misaligned + lighting
        ["shift", "noise"],  # Type 8: Misaligned + noise
        ["shift", "brightness", "noise"],  # Type 9: Misaligned + quality issues
        # Rotation combinations (not with shift)
        ["rotation", "brightness"],  # Type 10: Rotated + lighting
        ["rotation", "noise"],  # Type 11: Rotated + noise
        # 4-type combination (rare but realistic)
        ["brightness", "contrast", "noise", "blur"],  # Type 12: Worst case
    ]

    # Parameter ranges for each augmentation type
    AUGMENTATION_PARAMS: ClassVar[dict] = {
        "brightness": {"factor_range": (0.7, 1.3)},
        "contrast": {"factor_range": (0.8, 1.2)},
        "noise": {"sigma_range": (5, 15)},
        "blur": {"kernel_sizes": [3, 5]},
        "rotation": {"angle_range": (-3, 3)},
        "shift": {"max_shift_range": (10, 40)},
    }

    # Augmentation metadata for strategy pattern (reduces branches)
    AUGMENTATION_METADATA: ClassVar[dict] = {
        "brightness": {
            "type": "photometric",
            "method": "_adjust_brightness",
            "param_key": "factor_range",
        },
        "contrast": {
            "type": "photometric",
            "method": "_adjust_contrast",
            "param_key": "factor_range",
        },
        "noise": {
            "type": "photometric",
            "method": "_add_gaussian_noise",
            "param_key": "sigma_range",
        },
        "blur": {
            "type": "photometric",
            "method": "_add_blur",
            "param_key": "kernel_sizes",
        },
        "rotation": {
            "type": "geometric",
            "method": "_rotate_image",
            "param_key": "angle_range",
        },
        "shift": {
            "type": "geometric",
            "method": "_shift_field_blocks",
            "param_key": "max_shift_range",
        },
    }

    def __init__(
        self, source_images_dir: Path, source_labels_dir: Path, output_dir: Path
    ) -> None:
        """Initialize augmenter.

        Args:
            source_images_dir: Directory with original images
            source_labels_dir: Directory with original JSON labels
            output_dir: Directory to save augmented data
        """
        self.source_images_dir = Path(source_images_dir)
        self.source_labels_dir = Path(source_labels_dir)
        self.output_dir = Path(output_dir)

        self.augmented_images_dir = self.output_dir / "images"
        self.augmented_labels_dir = self.output_dir / "labels"

        self.augmented_images_dir.mkdir(parents=True, exist_ok=True)
        self.augmented_labels_dir.mkdir(parents=True, exist_ok=True)

    def augment_dataset(self, target_count: int = 200) -> dict:
        """Generate augmented dataset with normalization and combinations.

        Args:
            target_count: Target number of total samples (original + augmented)

        Returns:
            Statistics dictionary including normalization stats
        """
        # Get original samples
        original_images = list(self.source_images_dir.glob("*.jpg"))
        original_count = len(original_images)

        if original_count == 0:
            msg = f"No images found in {self.source_images_dir}"
            raise ValueError(msg)

        augmentations_per_image = (target_count - original_count) // original_count

        # Warn if we can't show all 13 combinations
        combination_count = len(self.AUGMENTATION_COMBINATIONS)
        if augmentations_per_image < combination_count:
            logger.warning(
                f"Target count yields {augmentations_per_image} augs/image, "
                f"less than {combination_count} combinations. "
                f"Consider increasing target to {original_count * (combination_count + 1)}"
            )

        logger.info("Starting data augmentation:")
        logger.info(f"  Original samples: {original_count}")
        logger.info(f"  Target samples: {target_count}")
        logger.info(f"  Augmentations per image: {augmentations_per_image}")
        logger.info(f"  Combination types: {combination_count}")

        stats = {
            "original_count": original_count,
            "target_count": target_count,
            "augmentations_per_image": augmentations_per_image,
            "combination_count": combination_count,
            "generated_count": 0,
            "normalized_count": 0,
        }

        for idx, image_path in enumerate(original_images):
            # Load original image and labels
            image = cv2.imread(str(image_path))
            label_path = self.source_labels_dir / f"{image_path.stem}.json"

            if not label_path.exists():
                logger.warning(f"No labels found for {image_path.name}, skipping")
                continue

            with label_path.open() as f:
                labels = json.load(f)

            # Generate augmentations
            for aug_idx in range(augmentations_per_image):
                # Step 1: Normalize (85% probability)
                normalized_image, normalized_labels = self._normalize_sample(
                    image.copy(), labels.copy(), normalization_prob=0.85
                )

                # Track if this sample was normalized
                if not np.array_equal(normalized_image, image):
                    stats["normalized_count"] += 1

                # Step 2: Apply combined augmentations
                aug_image, aug_labels = self._apply_augmentation(
                    normalized_image, normalized_labels, aug_idx
                )

                # Save augmented sample
                aug_name = f"{image_path.stem}_aug{aug_idx:03d}"
                cv2.imwrite(
                    str(self.augmented_images_dir / f"{aug_name}.jpg"), aug_image
                )

                with (self.augmented_labels_dir / f"{aug_name}.json").open("w") as f:
                    json.dump(aug_labels, f, indent=2)

                stats["generated_count"] += 1

            if (idx + 1) % 2 == 0:
                logger.info(f"  Processed {idx + 1}/{original_count} images...")

        normalization_rate = (
            stats["normalized_count"] / stats["generated_count"] * 100
            if stats["generated_count"] > 0
            else 0
        )
        logger.info("✅ Augmentation complete!")
        logger.info(f"  Generated {stats['generated_count']} new samples")
        logger.info(
            f"  Normalized: {stats['normalized_count']} ({normalization_rate:.1f}%)"
        )
        logger.info(f"  Total samples: {original_count + stats['generated_count']}")

        return stats

    def _apply_augmentation(
        self, image: np.ndarray, labels: dict, aug_idx: int
    ) -> tuple[np.ndarray, dict]:
        """Apply combination of augmentation types using strategy pattern.

        Applies 2-4 augmentation types simultaneously for realistic training data.
        Order: photometric first (brightness, contrast, noise, blur),
        then geometric (rotation, shift) to preserve coordinate system.

        Uses strategy pattern to eliminate branches and reduce complexity.

        Args:
            image: Input image
            labels: Original labels dictionary
            aug_idx: Augmentation index (for varying transformations)

        Returns:
            Tuple of (augmented_image, augmented_labels)
        """
        aug_labels = labels.copy()

        # Select combination based on index
        combination_idx = aug_idx % len(self.AUGMENTATION_COMBINATIONS)
        aug_types = self.AUGMENTATION_COMBINATIONS[combination_idx]

        # Separate augmentations by type using metadata (no hardcoded lists)
        photometric = [
            t
            for t in aug_types
            if self.AUGMENTATION_METADATA[t]["type"] == "photometric"
        ]
        geometric = [
            t for t in aug_types if self.AUGMENTATION_METADATA[t]["type"] == "geometric"
        ]

        # Apply photometric augmentations first (0 branches!)
        for aug_type in photometric:
            image = self._apply_single_augmentation(image, aug_type)

        # Apply geometric augmentations last (0 branches!)
        for aug_type in geometric:
            image, aug_labels = self._apply_single_augmentation_geometric(
                image, aug_labels, aug_type
            )

        return image, aug_labels

    def _apply_single_augmentation(
        self, image: np.ndarray, aug_type: str
    ) -> np.ndarray:
        """Apply single photometric augmentation without branches.

        Args:
            image: Input image
            aug_type: Augmentation type name

        Returns:
            Augmented image
        """
        metadata = self.AUGMENTATION_METADATA[aug_type]
        params = self.AUGMENTATION_PARAMS[aug_type]
        param_key = metadata["param_key"]

        # Generate parameter value
        param_values = params[param_key]
        if isinstance(param_values, tuple):
            # Range tuple - use uniform or randint
            if aug_type in ["brightness", "contrast", "noise", "rotation"]:
                param_value = random.uniform(*param_values)  # noqa: S311
            else:
                param_value = random.randint(*param_values)  # noqa: S311
        else:
            # List - use choice
            param_value = random.choice(param_values)  # noqa: S311

        # Call method dynamically
        method = getattr(self, metadata["method"])
        return method(image, param_value)

    def _apply_single_augmentation_geometric(
        self, image: np.ndarray, labels: dict, aug_type: str
    ) -> tuple[np.ndarray, dict]:
        """Apply single geometric augmentation without branches.

        Args:
            image: Input image
            labels: Label dictionary
            aug_type: Augmentation type name

        Returns:
            Tuple of (augmented_image, updated_labels)
        """
        metadata = self.AUGMENTATION_METADATA[aug_type]
        params = self.AUGMENTATION_PARAMS[aug_type]
        param_key = metadata["param_key"]

        # Generate parameter value
        param_values = params[param_key]
        if isinstance(param_values, tuple):
            # Range tuple - use uniform or randint
            if aug_type == "rotation":
                param_value = random.uniform(*param_values)  # noqa: S311
            else:  # shift
                param_value = random.randint(*param_values)  # noqa: S311
        else:
            param_value = random.choice(param_values)  # noqa: S311

        # Call method dynamically
        method = getattr(self, metadata["method"])
        return method(image, labels, param_value)

    def _adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image contrast."""
        mean = image.mean()
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

    def _add_gaussian_noise(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """Add Gaussian noise to simulate scan quality."""
        rng = np.random.default_rng()
        noise = rng.normal(0, sigma, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def _add_blur(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Add blur to simulate camera/scanner motion."""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def _rotate_image(
        self, image: np.ndarray, labels: dict, angle: float
    ) -> tuple[np.ndarray, dict]:
        """Rotate image and adjust ROI coordinates.

        Args:
            image: Input image
            labels: Original labels
            angle: Rotation angle in degrees

        Returns:
            Tuple of (rotated_image, updated_labels)
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Rotate image
        rotated = cv2.warpAffine(
            image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE
        )

        # Update ROI coordinates
        updated_labels = labels.copy()
        if "rois" in updated_labels:
            updated_rois = []
            for roi in updated_labels["rois"]:
                bbox = roi["bbox"]
                x, y, width, height = (
                    bbox["x"],
                    bbox["y"],
                    bbox["width"],
                    bbox["height"],
                )

                # Transform center point
                center_x = x + width / 2
                center_y = y + height / 2

                # Apply rotation matrix
                new_center = rotation_matrix @ np.array([center_x, center_y, 1])

                # Update bbox (approximate - keeps same size)
                new_x = int(new_center[0] - width / 2)
                new_y = int(new_center[1] - height / 2)

                updated_roi = roi.copy()
                updated_roi["bbox"] = {
                    "x": new_x,
                    "y": new_y,
                    "width": int(width),
                    "height": int(height),
                }
                updated_rois.append(updated_roi)

            updated_labels["rois"] = updated_rois

        return rotated, updated_labels

    def _shift_field_blocks(
        self, image: np.ndarray, labels: dict, max_shift: int = 30
    ) -> tuple[np.ndarray, dict]:
        """Shift field blocks to simulate misalignment while preserving white background.

        This augmentation creates realistic training data for the shift detection system
        by moving field blocks within a margin, using the white background to fill gaps.

        Args:
            image: Input image
            labels: Original labels with ROI information
            max_shift: Maximum shift in pixels (default: 30)

        Returns:
            Tuple of (shifted_image, updated_labels)
        """
        if "rois" not in labels or not labels["rois"]:
            return image, labels

        h, w = image.shape[:2]
        shifted_image = image.copy()
        updated_labels = labels.copy()
        updated_rois = []

        # Detect background color (assume white background for OMR sheets)
        # Sample from corners to get background color
        bg_color = self._detect_background_color(image)

        for roi in labels["rois"]:
            bbox = roi["bbox"]
            x, y, width, height = (
                bbox["x"],
                bbox["y"],
                bbox["width"],
                bbox["height"],
            )

            # Generate random shift within margin
            shift_x = random.randint(-max_shift, max_shift)  # noqa: S311
            shift_y = random.randint(-max_shift, max_shift)  # noqa: S311

            # Calculate new position (with boundary checking)
            new_x = max(0, min(w - width, x + shift_x))
            new_y = max(0, min(h - height, y + shift_y))

            # Extract the field block
            field_block = image[y : y + height, x : x + width].copy()

            # Fill the old position with background color
            cv2.rectangle(
                shifted_image,
                (x, y),
                (x + width, y + height),
                bg_color.tolist(),
                -1,  # Fill
            )

            # Place the field block at new position
            shifted_image[new_y : new_y + height, new_x : new_x + width] = field_block

            # Update ROI coordinates
            updated_roi = roi.copy()
            updated_roi["bbox"] = {
                "x": new_x,
                "y": new_y,
                "width": width,
                "height": height,
            }
            # Store the shift amount for validation (optional metadata)
            updated_roi["shift"] = {"dx": new_x - x, "dy": new_y - y}
            updated_rois.append(updated_roi)

        updated_labels["rois"] = updated_rois
        return shifted_image, updated_labels

    def _detect_background_color(self, image: np.ndarray) -> np.ndarray:
        """Detect background color by sampling image corners.

        Args:
            image: Input image

        Returns:
            Background color as numpy array (BGR)
        """
        h, w = image.shape[:2]
        margin = 20  # Sample area size

        # Sample from all four corners
        corners = [
            image[0:margin, 0:margin],  # Top-left
            image[0:margin, w - margin : w],  # Top-right
            image[h - margin : h, 0:margin],  # Bottom-left
            image[h - margin : h, w - margin : w],  # Bottom-right
        ]

        # Calculate median color from all corners
        corner_colors = [corner.mean(axis=(0, 1)) for corner in corners]
        return np.median(corner_colors, axis=0).astype(np.uint8)

    def _normalize_field_block(
        self, image: np.ndarray, roi: dict, fill_color: np.ndarray | None = None
    ) -> np.ndarray:
        """Normalize field block content with uniform fill.

        This removes bubble marking patterns to prevent the field block detector
        from learning specific answer patterns instead of structural boundaries.

        Args:
            image: Input image
            roi: ROI dictionary with bbox
            fill_color: Color to fill with (default: detected background)

        Returns:
            Image with normalized field block
        """
        normalized = image.copy()
        bbox = roi["bbox"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

        # Detect fill color if not provided (use background or light gray)
        if fill_color is None:
            fill_color = self._detect_background_color(image)

        # Fill entire field block interior with uniform color
        normalized[y : y + h, x : x + w] = fill_color

        return normalized

    def _normalize_sample(
        self, image: np.ndarray, labels: dict, normalization_prob: float = 0.85
    ) -> tuple[np.ndarray, dict]:
        """Normalize field blocks in a sample.

        Args:
            image: Input image
            labels: Label dictionary with ROIs
            normalization_prob: Probability of normalizing (0.85 = 85% normalized)

        Returns:
            Tuple of (normalized_image, labels)
        """
        # Random decision: normalize or keep original for variety
        if random.random() > normalization_prob:  # noqa: S311
            return image, labels  # Keep original for variety

        normalized = image.copy()

        if "rois" in labels:
            for roi in labels["rois"]:
                normalized = self._normalize_field_block(normalized, roi)

        return normalized, labels


def main():
    """Run data augmentation."""
    augmenter = DataAugmenter(
        source_images_dir=Path("outputs/training_data/dataset/images"),
        source_labels_dir=Path("outputs/training_data/dataset/labels"),
        output_dir=Path("outputs/training_data/augmented"),
    )

    augmenter.augment_dataset(target_count=200)


if __name__ == "__main__":
    main()
