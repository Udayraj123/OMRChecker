"""ML-based bubble detection using trained YOLO models.

Provides ML fallback for low-confidence traditional detections.
"""

from pathlib import Path

from src.processors.base import ProcessingContext, Processor
from src.utils.logger import logger

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class MLBubbleDetector(Processor):
    """YOLO-based bubble detection for low-confidence cases.

    This processor uses a trained YOLO model to detect bubbles when
    traditional threshold-based detection has low confidence.
    """

    def __init__(
        self, model_path: str | Path, confidence_threshold: float = 0.7
    ) -> None:
        """Initialize ML detector.

        Args:
            model_path: Path to trained YOLO model (.pt file)
            confidence_threshold: Minimum confidence for ML predictions
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.enabled = False  # Only enabled for flagged low-confidence cases
        self.model = None

        # Lazy load model
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model (lazy loading)."""
        if not self.model_path.exists():
            logger.warning(
                f"ML model not found at {self.model_path}, ML fallback disabled"
            )
            return

        try:
            if YOLO is None:
                raise ImportError

            self.model = YOLO(str(self.model_path))
            logger.info(f"Loaded ML bubble detector from: {self.model_path}")
        except ImportError:
            logger.warning(
                "ultralytics not installed. ML fallback disabled. Install with: uv sync --extra ml"
            )
        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}")

    def get_name(self) -> str:
        """Get processor name."""
        return "MLBubbleDetector"

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Process context with ML detection (if enabled).

        Args:
            context: Processing context

        Returns:
            Updated context with ML detection results
        """
        if not self.enabled or self.model is None:
            return context

        # Run YOLO inference
        try:
            results = self.model.predict(
                context.gray_image, conf=self.confidence_threshold, verbose=False
            )

            # Extract detections
            ml_detections = self._extract_detections(results)

            # Store in context metadata
            context.metadata["ml_detections"] = ml_detections
            context.metadata["ml_fallback_used"] = True

            logger.info(
                f"ML detector found {len(ml_detections)} bubbles "
                f"for {Path(context.file_path).name}"
            )

        except Exception as e:
            logger.error(f"ML detection failed: {e}")

        return context

    def _extract_detections(self, results) -> list[dict]:
        """Extract bubble detections from YOLO results.

        Args:
            results: YOLO prediction results

        Returns:
            List of detection dictionaries
        """
        detections = []

        if not results or len(results) == 0:
            return detections

        # Get first result (single image)
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        # Extract each detection
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Get bounding box in xyxy format
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Convert to center + dimensions
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            detection = {
                "bbox": {
                    "x_center": float(x_center),
                    "y_center": float(y_center),
                    "width": float(width),
                    "height": float(height),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                },
                "class_id": class_id,
                "class_name": self._get_class_name(class_id),
                "confidence": confidence,
            }

            detections.append(detection)

        return detections

    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID.

        Args:
            class_id: YOLO class ID

        Returns:
            Class name string
        """
        class_names = {
            0: "bubble_empty",
            1: "bubble_filled",
            2: "bubble_partial",
        }
        return class_names.get(class_id, "unknown")

    def enable_for_low_confidence(self) -> None:
        """Enable ML detector for processing."""
        self.enabled = True

    def disable(self) -> None:
        """Disable ML detector."""
        self.enabled = False


class HybridDetectionStrategy:
    """Hybrid detection strategy combining traditional and ML methods.

    Uses traditional method first, falls back to ML for low-confidence cases.
    """

    def __init__(
        self,
        ml_detector: MLBubbleDetector | None,
        confidence_threshold: float = 0.75,
    ) -> None:
        """Initialize hybrid strategy.

        Args:
            ml_detector: ML bubble detector (optional)
            confidence_threshold: Threshold for triggering ML fallback
        """
        self.ml_detector = ml_detector
        self.confidence_threshold = confidence_threshold
        self.stats = {
            "total_fields": 0,
            "high_confidence_fields": 0,
            "low_confidence_fields": 0,
            "ml_fallback_used": 0,
        }

    def identify_low_confidence_fields(
        self, field_id_to_interpretation: dict
    ) -> list[tuple[str, float]]:
        """Identify fields with low confidence scores.

        Args:
            field_id_to_interpretation: Map of field IDs to interpretations

        Returns:
            List of (field_id, confidence_score) tuples for low-confidence fields
        """
        low_confidence_fields = []

        for field_id, interpretation in field_id_to_interpretation.items():
            self.stats["total_fields"] += 1

            # Get confidence metrics
            confidence_metrics = getattr(
                interpretation, "field_level_confidence_metrics", {}
            )
            confidence_score = confidence_metrics.get("overall_confidence_score", 1.0)

            if confidence_score < self.confidence_threshold:
                low_confidence_fields.append((field_id, confidence_score))
                self.stats["low_confidence_fields"] += 1
                logger.debug(
                    f"Low confidence field detected: {field_id} "
                    f"(confidence: {confidence_score:.3f})"
                )
            else:
                self.stats["high_confidence_fields"] += 1

        return low_confidence_fields

    def should_use_ml_fallback(self, context: ProcessingContext) -> bool:
        """Determine if ML fallback should be used.

        Args:
            context: Processing context with detection results

        Returns:
            True if ML fallback should be used
        """
        if self.ml_detector is None or self.ml_detector.model is None:
            return False

        # Check for low confidence fields
        field_id_to_interpretation = context.field_id_to_interpretation
        low_confidence = self.identify_low_confidence_fields(field_id_to_interpretation)

        if low_confidence:
            logger.info(
                f"Found {len(low_confidence)} low-confidence fields, "
                f"triggering ML fallback"
            )
            return True

        return False

    def get_statistics(self) -> dict:
        """Get detection statistics.

        Returns:
            Dictionary with statistics
        """
        return self.stats.copy()
