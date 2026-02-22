"""YOLO format export utilities for OMRChecker training data.

Converts OMRChecker ROI data (bounding boxes, labels) to YOLO annotation format.
Supports hierarchical two-stage detection: field blocks (Stage 1) and bubbles (Stage 2).
"""

import json
import random
import shutil
from pathlib import Path
from typing import ClassVar

from src.utils.logger import logger


class YOLOAnnotationExporter:
    """Converts OMRChecker ROI data to YOLO format.

    YOLO format per line: <class_id> <x_center> <y_center> <width> <height>
    All coordinates normalized to [0, 1] range.

    Supports two dataset types:
    - Bubble detection (Stage 2): Individual bubble bounding boxes
    - Field block detection (Stage 1): Field block bounding boxes
    """

    BUBBLE_CLASSES: ClassVar[dict[str, int]] = {
        "bubble_empty": 0,
        "bubble_filled": 1,
        "bubble_partial": 2,  # For future use
    }

    FIELD_BLOCK_CLASSES: ClassVar[dict[str, int]] = {
        "field_block_mcq": 0,
        "field_block_ocr": 1,
        "field_block_barcode": 2,
    }

    def __init__(self, dataset_dir: Path, dataset_type: str = "bubble") -> None:
        """Initialize exporter.

        Args:
            dataset_dir: Root directory for YOLO dataset
            dataset_type: Type of dataset - "bubble" or "field_block"
        """
        self.dataset_dir = Path(dataset_dir)
        self.dataset_type = dataset_type
        self.yolo_images_dir = self.dataset_dir / "images"
        self.yolo_labels_dir = self.dataset_dir / "labels"
        self.train_split = 0.7  # 70% train, 30% val

        # Select appropriate class mapping
        if dataset_type == "field_block":
            self.class_mapping = self.FIELD_BLOCK_CLASSES
        else:
            self.class_mapping = self.BUBBLE_CLASSES

    def convert_dataset(self, source_images_dir: Path, source_labels_dir: Path) -> None:
        """Convert collected training data to YOLO format.

        Args:
            source_images_dir: Directory containing collected images
            source_labels_dir: Directory containing collected metadata JSON files
        """
        # Get all metadata files
        metadata_files = list(source_labels_dir.glob("*.json"))

        if not metadata_files:
            logger.warning(f"No metadata files found in {source_labels_dir}")
            return

        # Shuffle for train/val split
        random.shuffle(metadata_files)
        split_idx = int(len(metadata_files) * self.train_split)
        train_files = metadata_files[:split_idx]
        val_files = metadata_files[split_idx:]

        logger.info(
            f"Converting {len(metadata_files)} {self.dataset_type} samples to YOLO format "
            f"(train: {len(train_files)}, val: {len(val_files)})"
        )

        # Process train set
        self._convert_split(train_files, source_images_dir, "train")

        # Process val set
        self._convert_split(val_files, source_images_dir, "val")

        # Create data.yaml
        self._create_data_yaml()

        logger.info(f"YOLO {self.dataset_type} dataset created at: {self.dataset_dir}")

    def _convert_split(
        self, metadata_files: list[Path], source_images_dir: Path, split_name: str
    ) -> None:
        """Convert a split (train/val) to YOLO format.

        Args:
            metadata_files: List of metadata JSON files
            source_images_dir: Source directory with images
            split_name: "train" or "val"
        """
        # Create directories
        images_split_dir = self.yolo_images_dir / split_name
        labels_split_dir = self.yolo_labels_dir / split_name
        images_split_dir.mkdir(parents=True, exist_ok=True)
        labels_split_dir.mkdir(parents=True, exist_ok=True)

        for metadata_file in metadata_files:
            metadata = json.loads(metadata_file.read_text())

            file_stem = metadata_file.stem
            image_shape = metadata[
                "image_shape"
            ]  # (height, width, channels) or (height, width)
            img_height = image_shape[0]
            img_width = image_shape[1]

            # Convert annotations based on dataset type
            yolo_annotations = []
            if self.dataset_type == "field_block":
                # Field block dataset: use field_blocks from metadata
                for block in metadata.get("field_blocks", []):
                    yolo_line = self._convert_field_block_to_yolo(
                        block, img_width, img_height
                    )
                    if yolo_line:
                        yolo_annotations.append(yolo_line)
            else:
                # Bubble dataset: use rois from metadata
                for roi in metadata.get("rois", []):
                    yolo_line = self._convert_roi_to_yolo(roi, img_width, img_height)
                    if yolo_line:
                        yolo_annotations.append(yolo_line)

            # Skip if no valid annotations
            if not yolo_annotations:
                logger.debug(f"Skipping {file_stem}: no valid annotations")
                continue

            # Copy image
            source_image = source_images_dir / f"{file_stem}.jpg"
            if source_image.exists():
                dest_image = images_split_dir / f"{file_stem}.jpg"
                shutil.copy2(source_image, dest_image)
            else:
                logger.warning(f"Image not found: {source_image}")
                continue

            # Write YOLO annotation file
            annotation_file = labels_split_dir / f"{file_stem}.txt"
            annotation_file.write_text("\n".join(yolo_annotations))

        logger.info(f"Converted {len(metadata_files)} samples for {split_name} split")

    def _convert_field_block_to_yolo(
        self, block: dict, img_width: int, img_height: int
    ) -> str | None:
        """Convert a field block annotation to YOLO format line.

        Args:
            block: Field block dictionary with bbox_origin, bbox_dimensions, class_id
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            YOLO format line string, or None if invalid
        """
        bbox_origin = block.get("bbox_origin", [0, 0])
        bbox_dimensions = block.get("bbox_dimensions", [0, 0])

        x, y = bbox_origin
        w, h = bbox_dimensions

        # Validate bbox
        if w <= 0 or h <= 0:
            return None

        # Convert to center coordinates
        x_center = x + w / 2
        y_center = y + h / 2

        # Normalize to [0, 1]
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = w / img_width
        height_norm = h / img_height

        # Clamp to valid range
        x_center_norm = max(0.0, min(1.0, x_center_norm))
        y_center_norm = max(0.0, min(1.0, y_center_norm))
        width_norm = max(0.0, min(1.0, width_norm))
        height_norm = max(0.0, min(1.0, height_norm))

        # Get class ID (already computed by data collector)
        class_id = block.get("class_id", 0)

        # Format: <class_id> <x_center> <y_center> <width> <height>
        return f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"

    def _convert_roi_to_yolo(
        self, roi: dict, img_width: int, img_height: int
    ) -> str | None:
        """Convert a single ROI to YOLO format line.

        Args:
            roi: ROI dictionary with bbox and class
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            YOLO format line string, or None if invalid
        """
        bbox = roi.get("bbox", {})
        x = bbox.get("x", 0)
        y = bbox.get("y", 0)
        w = bbox.get("width", 0)
        h = bbox.get("height", 0)

        # Validate bbox
        if w <= 0 or h <= 0:
            return None

        # Convert to center coordinates
        x_center = x + w / 2
        y_center = y + h / 2

        # Normalize to [0, 1]
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = w / img_width
        height_norm = h / img_height

        # Clamp to valid range
        x_center_norm = max(0.0, min(1.0, x_center_norm))
        y_center_norm = max(0.0, min(1.0, y_center_norm))
        width_norm = max(0.0, min(1.0, width_norm))
        height_norm = max(0.0, min(1.0, height_norm))

        # Get class ID
        class_name = roi.get("class", "bubble_empty")
        class_id = self.class_mapping.get(class_name, 0)

        # Format: <class_id> <x_center> <y_center> <width> <height>
        return f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"

    def _create_data_yaml(self) -> None:
        """Create YOLO data.yaml configuration file."""
        # Get absolute paths
        train_path = (self.yolo_images_dir / "train").resolve()
        val_path = (self.yolo_images_dir / "val").resolve()

        # Create class names list based on dataset type
        class_names = [""] * len(self.class_mapping)
        for name, idx in self.class_mapping.items():
            class_names[idx] = name

        # Build names section
        names_section = "\n".join(
            [f"  {idx}: {name}" for idx, name in enumerate(class_names)]
        )

        yaml_content = f"""# YOLO dataset configuration for OMRChecker {self.dataset_type} detection
# Generated automatically by YOLOAnnotationExporter

path: {self.dataset_dir.resolve()}  # dataset root dir
train: {train_path}  # train images
val: {val_path}  # val images

# Classes
names:
{names_section}

# Number of classes
nc: {len(self.class_mapping)}
"""

        data_yaml_path = self.dataset_dir / "data.yaml"
        data_yaml_path.write_text(yaml_content)

        logger.info(f"Created YOLO data.yaml at: {data_yaml_path}")

    def export_field_block(
        self, field_block, _image_shape: tuple, detection_results: dict
    ) -> list[dict]:
        """Export field block to ROI format (for use by DataCollector).

        Args:
            field_block: FieldBlock object
            image_shape: Image shape (height, width)
            detection_results: Detection results for this field block

        Returns:
            List of ROI dictionaries
        """
        rois = []

        for field in field_block.fields:
            field_label = field.field_label

            # Get detection results for this field
            if field_label not in detection_results:
                continue

            field_detection = detection_results[field_label]

            # Extract bubble coordinates from scan_boxes
            for scan_box in field.scan_boxes:
                bubble_origin = scan_box.bubble_origin
                bubble_dimensions = field.bubble_dimensions

                x, y = bubble_origin
                w, h = bubble_dimensions

                # Get bubble state from detection
                bubble_mean = field_detection.get(scan_box.bubble_value, {}).get(
                    "mean_value", 255
                )
                threshold = field_detection.get("threshold", 128)
                is_filled = bubble_mean < threshold

                roi = {
                    "bbox": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                    },
                    "class": "bubble_filled" if is_filled else "bubble_empty",
                    "bubble_value": scan_box.bubble_value,
                    "intensity": float(bubble_mean),
                    "field_label": field_label,
                }

                rois.append(roi)

        return rois
