"""Field block data collection processor for ML training.

Collects field block-level training data (bounding boxes, labels, types) to train
Stage 1 of the hierarchical YOLO detection system.
"""

import json
from pathlib import Path
from typing import ClassVar

import cv2

from src.processors.base import ProcessingContext, Processor
from src.utils.logger import logger


class FieldBlockDataCollector(Processor):
    """Collects field block-level training data for ML.

    This processor extracts field block metadata (bounding boxes, labels, types)
    from high-confidence detections to train the Stage 1 field block detector.
    """

    # Map field detection types to YOLO class IDs
    FIELD_BLOCK_CLASSES: ClassVar[dict[str, int]] = {
        "BUBBLES_THRESHOLD": 0,  # field_block_mcq
        "OCR": 1,  # field_block_ocr
        "BARCODE_QR": 2,  # field_block_barcode
    }

    CLASS_NAMES: ClassVar[list[str]] = [
        "field_block_mcq",
        "field_block_ocr",
        "field_block_barcode",
    ]

    def __init__(
        self, template, confidence_threshold: float = 0.85, export_dir=None
    ) -> None:
        """Initialize the field block data collector.

        Args:
            template: The template object containing field blocks.
            confidence_threshold: Minimum confidence to collect as training data.
            export_dir: Directory to export training data (default: outputs/training_data/field_blocks).
        """
        self.template = template
        self.confidence_threshold = confidence_threshold
        self.export_dir = (
            Path(export_dir)
            if export_dir
            else Path("outputs/training_data/field_blocks")
        )

        # Create directory structure
        self.images_dir = self.export_dir / "images"
        self.labels_dir = self.export_dir / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            "samples_collected": 0,
            "field_blocks_collected": 0,
            "skipped_low_confidence": 0,
        }

        logger.info(
            f"FieldBlockDataCollector initialized with confidence threshold: {confidence_threshold}"
        )

    def get_name(self) -> str:
        """Get the name of this processor."""
        return "FieldBlockDataCollector"

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Collect field block-level training data.

        Args:
            context: Processing context with detection results.

        Returns:
            Updated context (unchanged, only collects data).
        """
        logger.debug(f"Starting {self.get_name()} processor")

        file_path = Path(context.file_path)
        field_id_to_interpretation = context.field_id_to_interpretation

        # Check if all fields have high confidence (same logic as bubble collector)
        all_confident = True
        for field_id, interpretation in field_id_to_interpretation.items():
            confidence_metrics = getattr(
                interpretation, "field_level_confidence_metrics", {}
            )
            confidence_score = confidence_metrics.get("overall_confidence_score", None)

            if confidence_score is None or confidence_score < self.confidence_threshold:
                all_confident = False
                self.stats["skipped_low_confidence"] += 1
                logger.debug(
                    f"Field {field_id} has low/no confidence, skipping field block data collection."
                )
                break

        if all_confident:
            self.stats["samples_collected"] += 1
            logger.info(
                f"Collecting field block training sample {self.stats['samples_collected']} for {file_path.name}"
            )

            # Export field block annotations
            self._export_field_block_sample(context)
        else:
            logger.info(
                f"Skipping {file_path.name} for field block training data due to low confidence."
            )

        logger.debug(f"Completed {self.get_name()} processor")
        return context

    def _export_field_block_sample(self, context: ProcessingContext) -> None:
        """Export a training sample with field block annotations.

        Args:
            context: Processing context with image and template data.
        """
        file_path = Path(context.file_path)
        file_stem = file_path.stem

        # Save image (full OMR sheet for field block detection)
        image_output = self.images_dir / f"{file_stem}.jpg"
        cv2.imwrite(str(image_output), context.colored_image)

        # Extract field block bounding boxes and metadata
        field_block_annotations = self._extract_field_blocks(
            context.template.field_blocks, context.gray_image.shape
        )

        self.stats["field_blocks_collected"] += len(field_block_annotations)

        # Save YOLO format annotations
        yolo_output = self.labels_dir / f"{file_stem}.txt"
        yolo_lines = [
            self._convert_to_yolo_format(block, context.gray_image.shape)
            for block in field_block_annotations
        ]
        yolo_output.write_text("\n".join(yolo_lines))

        # Save metadata (block names, labels, types) as JSON
        metadata = {
            "file_path": str(file_path),
            "image_shape": list(context.gray_image.shape),
            "field_blocks": field_block_annotations,
        }

        metadata_output = self.labels_dir / f"{file_stem}.json"
        metadata_output.write_text(json.dumps(metadata, indent=2))

        logger.debug(
            f"Exported {len(field_block_annotations)} field blocks for {file_path.name}"
        )

    def _extract_field_blocks(
        self,
        field_blocks,
        _image_shape: tuple,
    ) -> list[dict]:
        """Extract field block bounding boxes and metadata.

        Args:
            field_blocks: List of FieldBlock objects from template.
            _image_shape: Shape of the image (unused, for future validation).

        Returns:
            List of field block annotation dictionaries.
        """
        annotations = []

        for field_block in field_blocks:
            # Get bounding box (after alignment shifts have been applied)
            bbox_origin = field_block.get_shifted_origin()
            bbox_dimensions = field_block.bounding_box_dimensions

            # Get block type
            field_detection_type = field_block.field_detection_type.value

            # Map to YOLO class
            class_id = self.FIELD_BLOCK_CLASSES.get(field_detection_type, 0)

            annotation = {
                "block_name": field_block.name,
                "block_type": field_detection_type,
                "class_id": class_id,
                "class_name": self.CLASS_NAMES[class_id],
                "bbox_origin": bbox_origin,  # [x, y]
                "bbox_dimensions": bbox_dimensions,  # [width, height]
                "field_labels": list(field_block.parsed_field_labels),
            }

            annotations.append(annotation)

        return annotations

    def _convert_to_yolo_format(self, block: dict, image_shape: tuple) -> str:
        """Convert field block to YOLO format annotation line.

        YOLO format: <class_id> <x_center> <y_center> <width> <height> (normalized 0-1)

        Args:
            block: Field block annotation dictionary.
            image_shape: Image shape (height, width).

        Returns:
            YOLO format annotation line.
        """
        img_height, img_width = image_shape[:2]

        x, y = block["bbox_origin"]
        w, h = block["bbox_dimensions"]

        # Calculate center and normalize
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width_norm = w / img_width
        height_norm = h / img_height

        # Clamp to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width_norm = max(0.0, min(1.0, width_norm))
        height_norm = max(0.0, min(1.0, height_norm))

        class_id = block["class_id"]

        return f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"

    def finish_processing_directory(self) -> dict:
        """Finish processing and return statistics.

        Returns:
            Statistics dictionary.
        """
        logger.info(f"Field block data collection complete: {self.stats}")

        # Save statistics
        stats_file = self.export_dir / "field_block_collection_stats.json"
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        stats_file.write_text(json.dumps(self.stats, indent=2))

        return self.stats
