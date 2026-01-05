"""ML-based bubble detector with field block context (Stage 2 of hierarchical detection).

Detects individual bubbles within field blocks using YOLO,
leveraging spatial context from Stage 1 field block detection.
"""

from pathlib import Path
from typing import ClassVar

from src.processors.base import ProcessingContext, Processor
from src.utils.logger import logger


class MLBubbleDetector(Processor):
    """YOLO-based bubble detector (Stage 2).

    Operates on individual field blocks using context from
    Stage 1 field block detection for improved accuracy.
    """

    # Class names matching training data
    CLASS_NAMES: ClassVar[dict[int, str]] = {
        0: "bubble_empty",
        1: "bubble_filled",
        2: "bubble_partial",
    }

    def __init__(self, model_path: str, confidence_threshold: float = 0.7) -> None:
        """Initialize the bubble detector.

        Args:
            model_path: Path to the trained YOLO model (.pt file).
            confidence_threshold: Minimum confidence for bubble detection.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error(
                "ultralytics package not found. Install ML dependencies with: uv sync --extra ml"
            )
            self.model = None
            return

        self.model = YOLO(model_path) if Path(model_path).exists() else None
        self.confidence_threshold = confidence_threshold

        if self.model:
            logger.info(f"MLBubbleDetector initialized with model: {model_path}")
        else:
            logger.warning(f"Bubble model not found at {model_path}, detector disabled")

    def get_name(self) -> str:
        """Get the name of this processor."""
        return "MLBubbleDetector"

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Run bubble detection on field blocks.

        Args:
            context: Processing context with ML-detected blocks.

        Returns:
            Updated context with ML bubble detections per block.
        """
        if not self.model:
            return context

        logger.debug(f"Starting {self.get_name()} processor")

        # Get ML-detected blocks from context (set by Stage 1)
        ml_blocks = context.metadata.get("ml_detected_blocks", [])

        if not ml_blocks:
            logger.debug("No ML-detected blocks found, skipping ML bubble detection")
            return context

        # Process each field block
        for block_detection in ml_blocks:
            # Crop block region from image
            block_crop = self._crop_block_region(
                context.gray_image, block_detection["bbox_xyxy"]
            )

            if block_crop is None or block_crop.size == 0:
                logger.warning(
                    f"Empty crop for block {block_detection['class_name']}, skipping"
                )
                continue

            # Run bubble detection on crop
            bubble_results = self.model.predict(
                block_crop,
                conf=self.confidence_threshold,
                verbose=False,
                imgsz=640,  # Smaller for cropped blocks
            )

            # Map back to full image coordinates
            bubbles = self._map_to_full_coordinates(
                bubble_results, block_detection["bbox_xyxy"]
            )

            # Store per-block results
            block_detection["ml_bubbles"] = bubbles
            block_detection["ml_bubbles_count"] = len(bubbles)

            logger.debug(
                f"Detected {len(bubbles)} bubbles in block {block_detection['class_name']}"
            )

        logger.info(f"ML bubble detection complete for {len(ml_blocks)} blocks")
        logger.debug(f"Completed {self.get_name()} processor")
        return context

    def _crop_block_region(self, image, bbox_xyxy: list) -> any:
        """Crop field block region from full image.

        Args:
            image: Full grayscale image.
            bbox_xyxy: Bounding box in [x1, y1, x2, y2] format.

        Returns:
            Cropped image region.
        """
        x1, y1, x2, y2 = bbox_xyxy

        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        # Crop and return
        return image[y1:y2, x1:x2]

    def _map_to_full_coordinates(
        self, bubble_results, block_bbox_xyxy: list
    ) -> list[dict]:
        """Map bubble detections from crop coordinates to full image coordinates.

        Args:
            bubble_results: YOLO detection results on cropped block.
            block_bbox_xyxy: Block bounding box in full image [x1, y1, x2, y2].

        Returns:
            List of bubble detections in full image coordinates.
        """
        if not bubble_results or len(bubble_results) == 0:
            return []

        block_x1, block_y1, _, _ = block_bbox_xyxy
        bubbles = []

        for result in bubble_results:
            if not hasattr(result, "boxes") or result.boxes is None:
                continue

            for box in result.boxes:
                # Extract box information
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Get bounding box in xyxy format (crop-relative)
                xyxy_crop = box.xyxy[0].tolist()
                x1_crop, y1_crop, x2_crop, y2_crop = xyxy_crop

                # Map to full image coordinates
                x1_full = int(block_x1 + x1_crop)
                y1_full = int(block_y1 + y1_crop)
                x2_full = int(block_x1 + x2_crop)
                y2_full = int(block_y1 + y2_crop)

                bubble_detection = {
                    "class_id": class_id,
                    "class_name": self.CLASS_NAMES.get(class_id, "unknown"),
                    "confidence": confidence,
                    "bbox_xyxy": [x1_full, y1_full, x2_full, y2_full],
                    "bbox_origin": [x1_full, y1_full],
                    "bbox_dimensions": [x2_full - x1_full, y2_full - y1_full],
                    "state": "filled" if class_id == 1 else "empty",
                }

                bubbles.append(bubble_detection)

        return bubbles
