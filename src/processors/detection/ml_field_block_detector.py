"""ML-based field block detector (Stage 1 of hierarchical YOLO detection).

Detects field blocks and their approximate positions using YOLO,
providing spatial context for bubble detection and alignment refinement.

Optionally uses Spatial Transformer Network (STN) for alignment refinement
before YOLO detection to improve accuracy on misaligned or distorted images.
"""

from pathlib import Path
from typing import ClassVar

import numpy as np

from src.processors.base import ProcessingContext, Processor
from src.utils.geometry import bbox_center, euclidean_distance
from src.utils.logger import logger


class MLFieldBlockDetector(Processor):
    """YOLO-based field block detector (Stage 1).

    Detects field blocks and their approximate positions,
    providing spatial context for Stage 2 bubble detection.

    Supports optional STN preprocessing for improved robustness to
    geometric distortions (rotation, translation, scale).
    """

    # Class names matching training data
    CLASS_NAMES: ClassVar[dict[int, str]] = {
        0: "field_block_mcq",
        1: "field_block_ocr",
        2: "field_block_barcode",
    }

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.7,
        use_stn: bool = False,
        stn_model_path: str | None = None,
    ) -> None:
        """Initialize the field block detector.

        Args:
            model_path: Path to the trained YOLO model (.pt file).
            confidence_threshold: Minimum confidence for field block detection.
            use_stn: Whether to use STN for alignment refinement before detection.
            stn_model_path: Path to trained STN model (.pt file). Required if use_stn=True.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error(
                "ultralytics package not found. Install ML dependencies with: uv sync --extra ml"
            )
            self.model = None
            self.stn = None
            return

        self.model = YOLO(model_path) if Path(model_path).exists() else None
        self.confidence_threshold = confidence_threshold
        self.detected_blocks = []  # Stores detections per image

        # Initialize STN if requested
        self.use_stn = use_stn
        self.stn = None

        if use_stn:
            if not stn_model_path:
                logger.warning(
                    "use_stn=True but no stn_model_path provided. STN disabled."
                )
                self.use_stn = False
            else:
                try:
                    from src.processors.detection.models.stn_utils import (
                        load_stn_model,
                    )

                    self.stn = load_stn_model(
                        stn_model_path,
                        input_channels=1,
                        input_size=(1024, 1024),  # Match YOLO input size
                        device="cpu",
                    )
                    logger.info(f"STN module loaded from {stn_model_path}")
                except Exception as e:
                    logger.error(f"Failed to load STN model: {e}")
                    logger.warning("Continuing without STN preprocessing.")
                    self.use_stn = False
                    self.stn = None

        if self.model:
            stn_status = "with STN" if self.use_stn and self.stn else "without STN"
            logger.info(
                f"MLFieldBlockDetector initialized {stn_status}, model: {model_path}"
            )
        else:
            logger.warning(
                f"Field block model not found at {model_path}, detector disabled"
            )

    def get_name(self) -> str:
        """Get the name of this processor."""
        return "MLFieldBlockDetector"

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Run field block detection and compute alignment adjustments.

        Args:
            context: Processing context with aligned image.

        Returns:
            Updated context with ML-detected blocks in metadata.
        """
        if not self.model:
            return context

        logger.debug(f"Starting {self.get_name()} processor")

        # Apply STN preprocessing if enabled
        image = context.gray_image
        if self.use_stn and self.stn:
            image = self._apply_stn(image)
            logger.debug("Applied STN transformation for alignment refinement")

        # Run YOLO inference on (possibly STN-transformed) image
        results = self.model.predict(
            image,
            conf=self.confidence_threshold,
            verbose=False,
            imgsz=1024,  # Larger for full OMR sheet
        )

        # Parse detections
        detected_blocks = self._parse_block_detections(results, image.shape)

        logger.info(f"ML detected {len(detected_blocks)} field blocks")

        # Compare with template-defined blocks to compute alignment adjustments
        alignment_adjustments = self._compute_alignment_adjustments(
            detected_blocks, context.template.field_blocks
        )

        # Store in context for downstream use
        context.metadata["ml_detected_blocks"] = detected_blocks
        context.metadata["ml_block_alignments"] = alignment_adjustments

        logger.debug(f"Completed {self.get_name()} processor")
        return context

    def _apply_stn(self, image: np.ndarray) -> np.ndarray:
        """Apply STN transformation to refine image alignment.

        Args:
            image: Input grayscale image (H, W)

        Returns:
            STN-transformed image (same shape as input)
        """
        if not self.stn:
            return image

        try:
            from src.processors.detection.models.stn_utils import (
                apply_stn_to_image,
            )

            return apply_stn_to_image(self.stn, image, device="cpu")
        except Exception as e:
            logger.warning(f"STN transformation failed: {e}. Using original image.")
            return image

    def _parse_block_detections(self, results, image_shape: tuple) -> list[dict]:
        """Parse YOLO detection results into structured format.

        Args:
            results: YOLO detection results.
            image_shape: Image shape (height, width).

        Returns:
            List of detected block dictionaries.
        """
        if not results or len(results) == 0:
            return []

        image_height, image_width = image_shape[:2]
        detected_blocks = []

        for result in results:
            if not hasattr(result, "boxes") or result.boxes is None:
                continue

            for box in result.boxes:
                # Extract box information
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Get bounding box in xyxy format (absolute coordinates)
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy

                # Convert to origin + dimensions format (matching OMRChecker convention)
                bbox_origin = [int(x1), int(y1)]
                bbox_dimensions = [int(x2 - x1), int(y2 - y1)]

                block_detection = {
                    "class_id": class_id,
                    "class_name": self.CLASS_NAMES.get(class_id, "unknown"),
                    "confidence": confidence,
                    "bbox_origin": bbox_origin,  # [x, y]
                    "bbox_dimensions": bbox_dimensions,  # [width, height]
                    "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                }

                detected_blocks.append(block_detection)

        # Sort by y-coordinate then x-coordinate (top-to-bottom, left-to-right)
        detected_blocks.sort(key=lambda b: (b["bbox_origin"][1], b["bbox_origin"][0]))

        return detected_blocks

    def _compute_alignment_adjustments(
        self, ml_blocks: list[dict], template_blocks: list
    ) -> dict:
        """Compare ML-detected blocks with template expectations.

        Calculates shift adjustments to refine alignment based on
        spatial proximity matching.

        Args:
            ml_blocks: ML-detected field blocks.
            template_blocks: Template-defined field blocks.

        Returns:
            Dictionary of alignment adjustments per block.
        """
        adjustments = {}

        if not ml_blocks or not template_blocks:
            return adjustments

        # For each template block, find the closest ML detection
        for template_block in template_blocks:
            template_name = template_block.name
            template_origin = template_block.get_shifted_origin()
            template_center = bbox_center(
                template_origin, template_block.bounding_box_dimensions
            )

            # Find closest ML detection by Euclidean distance
            best_match = None
            best_distance = float("inf")

            for ml_block in ml_blocks:
                ml_center = bbox_center(
                    ml_block["bbox_origin"], ml_block["bbox_dimensions"]
                )

                distance = euclidean_distance(template_center, ml_center)

                if distance < best_distance:
                    best_distance = distance
                    best_match = ml_block

            # If a reasonable match is found (within 200 pixels), compute shift
            if best_match and best_distance < 200:
                ml_origin = best_match["bbox_origin"]
                shift_x = ml_origin[0] - template_origin[0]
                shift_y = ml_origin[1] - template_origin[1]

                adjustments[template_name] = {
                    "matched_ml_block": best_match,
                    "shift": [shift_x, shift_y],
                    "distance": best_distance,
                    "confidence": best_match["confidence"],
                }

                logger.debug(
                    f"Block '{template_name}' matched with shift: [{shift_x}, {shift_y}], "
                    f"confidence: {best_match['confidence']:.2f}"
                )

        return adjustments
