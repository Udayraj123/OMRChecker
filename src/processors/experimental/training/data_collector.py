"""Data collection processor for ML training.

Collects high-confidence detections from traditional method to use as training data.
"""

import json
from pathlib import Path

import cv2

from src.processors.base import ProcessingContext, Processor
from src.utils.logger import logger


class TrainingDataCollector(Processor):
    """Collects high-confidence detections for ML training.

    This processor extracts ROI points and detection results from high-confidence
    traditional detections to build a training dataset for YOLO bubble detection.
    """

    def __init__(self, template, confidence_threshold=0.85, export_dir=None) -> None:
        """Initialize the data collector.

        Args:
            template: The template being processed
            confidence_threshold: Minimum confidence score to include sample (0.0-1.0)
            export_dir: Directory to export training data (default: outputs/training_data)
        """
        self.template = template
        self.confidence_threshold = confidence_threshold
        self.export_dir = (
            Path(export_dir) if export_dir else Path("outputs/training_data")
        )

        # Create directory structure
        self.dataset_dir = self.export_dir / "dataset"
        self.images_dir = self.dataset_dir / "images"
        self.labels_dir = self.dataset_dir / "labels"

        # Statistics tracking
        self.stats = {
            "total_processed": 0,
            "high_confidence_collected": 0,
            "low_confidence_skipped": 0,
            "fields_collected": 0,
            "bubbles_collected": 0,
        }

        logger.info(
            f"TrainingDataCollector initialized with confidence threshold: {confidence_threshold}"
        )

    def get_name(self) -> str:
        """Get processor name."""
        return "TrainingDataCollector"

    def finish_processing_directory(self) -> dict:
        """Called after processing all files in a directory.

        Returns:
            Dictionary with collection statistics
        """
        return self.export_statistics()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Process context and collect training data.

        Args:
            context: Processing context with detection results

        Returns:
            Updated context (unchanged, this processor only exports data)
        """
        self.stats["total_processed"] += 1

        # Extract field interpretations from context
        field_id_to_interpretation = context.field_id_to_interpretation

        if not field_id_to_interpretation:
            logger.debug(f"No interpretations found for {context.file_path}, skipping")
            return context

        # Collect data from high-confidence fields
        self._collect_from_interpretations(context, field_id_to_interpretation)

        # Log progress periodically
        if self.stats["total_processed"] % 10 == 0:
            logger.info(
                f"Training data collection progress: "
                f"{self.stats['high_confidence_collected']}/{self.stats['total_processed']} "
                f"high-confidence samples collected"
            )

        return context

    def _collect_from_interpretations(
        self, context: ProcessingContext, field_id_to_interpretation
    ) -> None:
        """Collect training data from field interpretations.

        Args:
            context: Processing context
            field_id_to_interpretation: Map of field IDs to interpretations
        """
        high_confidence_fields = []

        # Filter fields by confidence score
        for field_id, interpretation in field_id_to_interpretation.items():
            # Get confidence score from metrics if available
            confidence_metrics = getattr(
                interpretation, "field_level_confidence_metrics", {}
            )
            confidence_score = confidence_metrics.get("overall_confidence_score", None)

            # If confidence score not calculated (metrics disabled), calculate on-demand
            if confidence_score is None:
                confidence_score = self._estimate_confidence_on_demand(interpretation)

            if confidence_score >= self.confidence_threshold:
                high_confidence_fields.append(
                    (field_id, interpretation, confidence_score)
                )

        if not high_confidence_fields:
            self.stats["low_confidence_skipped"] += 1
            logger.debug(
                f"Skipping {context.file_path}: no fields above confidence threshold {self.confidence_threshold}"
            )
            return

        # Extract ROI data and export
        self._export_training_sample(context, high_confidence_fields)
        self.stats["high_confidence_collected"] += 1
        self.stats["fields_collected"] += len(high_confidence_fields)

    def _estimate_confidence_on_demand(self, interpretation) -> float:
        """Estimate confidence score for interpretations without pre-calculated metrics.

        Args:
            interpretation: Field interpretation object

        Returns:
            Estimated confidence score (0.0-1.0)
        """
        # Simple heuristic: if not multi-marked and has interpretations, use high confidence
        is_multi_marked = getattr(interpretation, "is_multi_marked", False)
        bubble_interpretations = getattr(interpretation, "bubble_interpretations", [])

        if not bubble_interpretations:
            return 0.0

        # Count marked bubbles
        marked_count = sum(1 for b in bubble_interpretations if b.is_attempted)

        # Simple confidence rules:
        if is_multi_marked:
            return 0.5  # Low confidence for multi-marked
        if marked_count == 0:
            return 0.75  # Moderate confidence for no marks (could be intentional)
        if marked_count == 1:
            return 0.95  # High confidence for single mark
        return 0.6  # Moderate-low confidence for multiple marks

    def _export_training_sample(
        self, context: ProcessingContext, high_confidence_fields
    ) -> None:
        """Export a training sample with ROI annotations.

        Args:
            context: Processing context with image data
            high_confidence_fields: List of (field_id, interpretation, confidence_score)
        """
        # Create directories if they don't exist
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # Get file name without extension
        file_path = Path(context.file_path)
        file_stem = file_path.stem

        # Save image
        image_output = self.images_dir / f"{file_stem}.jpg"
        cv2.imwrite(str(image_output), context.gray_image)

        # Collect ROI annotations from all high-confidence fields
        roi_annotations = []
        for _field_id, interpretation, confidence_score in high_confidence_fields:
            field_rois = self._extract_rois_from_field(
                interpretation, context.gray_image.shape, confidence_score
            )
            roi_annotations.extend(field_rois)
            self.stats["bubbles_collected"] += len(field_rois)

        # Save annotations metadata (we'll convert to YOLO format in next phase)
        metadata = {
            "file_path": str(file_path),
            "image_shape": context.gray_image.shape,
            "fields": [
                {
                    "field_id": field_id,
                    "confidence_score": conf,
                    "roi_count": len(
                        self._extract_rois_from_field(
                            interp, context.gray_image.shape, conf
                        )
                    ),
                }
                for field_id, interp, conf in high_confidence_fields
            ],
            "rois": roi_annotations,
        }

        metadata_output = self.labels_dir / f"{file_stem}.json"
        metadata_output.write_text(json.dumps(metadata, indent=2))

        logger.debug(
            f"Exported training sample: {file_stem} with {len(roi_annotations)} ROIs "
            f"from {len(high_confidence_fields)} fields"
        )

    def _extract_rois_from_field(
        self, interpretation, _image_shape, confidence_score
    ) -> list[dict]:
        """Extract ROI data from a field interpretation.

        Args:
            interpretation: Field interpretation object
            image_shape: Shape of the image (height, width)
            confidence_score: Overall confidence score for this field

        Returns:
            List of ROI dictionaries with coordinates and labels
        """
        rois = []

        # Access field and bubble interpretations
        field = interpretation.field
        bubble_interpretations = getattr(interpretation, "bubble_interpretations", [])

        if not bubble_interpretations:
            logger.debug(
                f"No bubble interpretations found for field {field.field_label}"
            )
            return rois

        # Check if field has bubble_dimensions (bubble fields only)
        if not hasattr(field, "bubble_dimensions"):
            logger.debug(f"Field {field.field_label} is not a bubble field, skipping")
            return rois

        bubble_dimensions = field.bubble_dimensions  # [width, height]

        # Extract each bubble's ROI
        for bubble_interp in bubble_interpretations:
            # Get bubble reference (scan box)
            item_ref = bubble_interp.item_reference

            # Get coordinates from scan box (use 'origin' attribute from ScanBox)
            if not hasattr(item_ref, "origin"):
                logger.debug(
                    f"Bubble has no origin attribute in field {field.field_label}"
                )
                continue

            bubble_origin = item_ref.origin  # [x, y]

            # Calculate bounding box
            x, y = bubble_origin
            w, h = bubble_dimensions

            # Validate coordinates
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                logger.warning(
                    f"Invalid bubble coordinates in field {field.field_label}: "
                    f"origin={bubble_origin}, dimensions={bubble_dimensions}"
                )
                continue

            # Determine bubble state (class label)
            is_filled = bubble_interp.is_attempted
            bubble_value = bubble_interp.bubble_value

            roi_data = {
                "bbox": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                },
                "class": "bubble_filled" if is_filled else "bubble_empty",
                "bubble_value": bubble_value,
                "intensity": float(bubble_interp.mean_value),
                "field_label": field.field_label,
                "confidence_score": float(confidence_score),
            }

            rois.append(roi_data)

        if rois:
            logger.debug(f"Extracted {len(rois)} ROIs from field {field.field_label}")

        return rois

    def export_statistics(self) -> dict:
        """Export collection statistics.

        Returns:
            Dictionary with collection statistics
        """
        stats_file = self.export_dir / "collection_stats.json"
        stats_file.parent.mkdir(parents=True, exist_ok=True)

        stats_file.write_text(json.dumps(self.stats, indent=2))

        logger.info(f"Training data collection statistics saved to: {stats_file}")
        logger.info(
            f"Collected {self.stats['high_confidence_collected']} high-confidence samples "
            f"with {self.stats['bubbles_collected']} total bubbles"
        )

        return self.stats
