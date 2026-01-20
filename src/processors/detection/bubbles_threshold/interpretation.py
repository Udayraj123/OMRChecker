from src.processors.detection.base.interpretation import FieldInterpretation
from src.processors.detection.bubbles_threshold.interpretation_drawing import (
    BubblesFieldInterpretationDrawing,
)
from src.processors.detection.models.detection_results import (
    BubbleFieldDetectionResult,
)
from src.processors.layout.field.base import Field
from src.processors.threshold.local_threshold import LocalThresholdStrategy
from src.processors.threshold.threshold_result import ThresholdConfig, ThresholdResult
from src.utils.logger import logger


class BubbleInterpretation:
    """Single bubble interpretation result."""

    def __init__(self, bubble_mean, threshold: float) -> None:
        self.bubble_mean = bubble_mean
        self.threshold = threshold
        self.mean_value = bubble_mean.mean_value
        self.is_attempted = bubble_mean.mean_value < threshold
        self.bubble_value = (
            bubble_mean.unit_bubble.bubble_value
            if hasattr(bubble_mean.unit_bubble, "bubble_value")
            else ""
        )
        # item_reference is used by the drawing code
        self.item_reference = bubble_mean.unit_bubble

    def get_value(self) -> str:
        """Get bubble value if marked."""
        return self.bubble_value if self.is_attempted else ""


class BubblesFieldInterpretation(FieldInterpretation):
    def __init__(self, *args, **kwargs) -> None:
        self.bubble_interpretations: list[BubbleInterpretation] = []
        self.is_multi_marked = False
        self.local_threshold_for_field = 0.0
        self.threshold_result: ThresholdResult | None = None
        super().__init__(*args, **kwargs)

    def get_drawing_instance(self):
        """Get drawing instance for visualization."""
        return BubblesFieldInterpretationDrawing(self)

    def get_field_interpretation_string(self) -> str:
        """Get final interpretation string.

        Returns concatenated marked bubble values or empty value.
        Special case: If ALL bubbles are marked, treat as unmarked (likely scanning issue).
        """
        marked_bubbles = [
            interp.bubble_value
            for interp in self.bubble_interpretations
            if interp.is_attempted
        ]

        # If no bubbles marked, return empty value
        if len(marked_bubbles) == 0:
            return self.empty_value

        # If ALL bubbles are marked, treat as unmarked (likely scanning/detection issue)
        total_bubbles = len(self.bubble_interpretations)
        if len(marked_bubbles) == total_bubbles:
            return self.empty_value

        return "".join(marked_bubbles)

    def run_interpretation(
        self,
        field: Field,
        file_level_detection_aggregates,
        file_level_interpretation_aggregates,
    ) -> None:
        """Run interpretation using detection results and threshold strategies.

        Args:
            field: Field to interpret
            file_level_detection_aggregates: Detection results (typed models via bubble_fields)
            file_level_interpretation_aggregates: Interpretation aggregates
        """
        # Step 1: Extract detection result
        detection_result = self._extract_detection_result(
            field, file_level_detection_aggregates
        )

        # Step 2: Calculate thresholds using strategies
        threshold_config = self._create_threshold_config(
            file_level_interpretation_aggregates
        )
        self.threshold_result = self._calculate_threshold(
            detection_result, file_level_interpretation_aggregates, threshold_config
        )

        self.local_threshold_for_field = self.threshold_result.threshold_value

        # Step 3: Interpret bubbles
        self._interpret_bubbles(detection_result)

        # Step 4: Check multi-marking
        self._check_multi_marking()

        # Step 5: Calculate confidence metrics (if enabled)
        if self.tuning_config.outputs.show_confidence_metrics:
            self._calculate_confidence_metrics(
                detection_result, file_level_interpretation_aggregates
            )

    def _extract_detection_result(
        self, field: Field, file_level_detection_aggregates
    ) -> BubbleFieldDetectionResult:
        """Extract detection result from aggregates.

        Uses new typed models pipeline via bubble_fields.
        """
        field_label = field.field_label

        # Get from new typed format
        if "bubble_fields" not in file_level_detection_aggregates:
            msg = f"bubble_fields not found in file_level_detection_aggregates for field {field_label}"
            raise KeyError(msg)

        bubble_fields = file_level_detection_aggregates["bubble_fields"]
        if field_label not in bubble_fields:
            msg = f"Field {field_label} not found in bubble_fields"
            raise KeyError(msg)

        return bubble_fields[field_label]

    def _create_threshold_config(
        self, file_level_interpretation_aggregates
    ) -> ThresholdConfig:
        """Create threshold configuration from tuning config and file-level aggregates."""
        config = self.tuning_config
        return ThresholdConfig(
            min_jump=config.thresholding.MIN_JUMP,
            jump_delta=config.thresholding.JUMP_DELTA,
            min_gap_two_bubbles=config.thresholding.MIN_GAP_TWO_BUBBLES,
            min_jump_surplus_for_global_fallback=config.thresholding.MIN_JUMP_SURPLUS_FOR_GLOBAL_FALLBACK,
            confident_jump_surplus_for_disparity=config.thresholding.CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY,
            global_threshold_margin=config.thresholding.GLOBAL_THRESHOLD_MARGIN,
            outlier_deviation_threshold=file_level_interpretation_aggregates.get(
                "outlier_deviation_threshold_for_file", 5.0
            ),
            default_threshold=config.thresholding.GLOBAL_PAGE_THRESHOLD,
        )

    def _calculate_threshold(
        self,
        detection_result: BubbleFieldDetectionResult,
        file_level_interpretation_aggregates,
        config: ThresholdConfig,
    ) -> ThresholdResult:
        """Calculate threshold using strategies.

        Uses LocalThresholdStrategy with global fallback.
        This replaces 170+ lines of threshold calculation code!
        """
        # Get global fallback threshold
        global_fallback = file_level_interpretation_aggregates.get(
            "file_level_fallback_threshold", config.default_threshold
        )

        # Use local strategy with global fallback
        strategy = LocalThresholdStrategy(global_fallback=global_fallback)

        # Calculate threshold
        return strategy.calculate_threshold(
            detection_result.mean_values,  # Auto-extracted from model!
            config,
        )

    def _interpret_bubbles(self, detection_result: BubbleFieldDetectionResult) -> None:
        """Interpret each bubble using calculated threshold.

        Creates interpretation for each bubble.
        """
        self.bubble_interpretations = [
            BubbleInterpretation(bubble_mean, self.local_threshold_for_field)
            for bubble_mean in detection_result.bubble_means
        ]

    def _check_multi_marking(self) -> None:
        """Check if multiple bubbles are marked."""
        marked_count = sum(
            1 for interp in self.bubble_interpretations if interp.is_attempted
        )
        self.is_multi_marked = marked_count > 1

        if self.is_multi_marked:
            logger.warning(
                f"Multi-marking detected in field: {self.field.field_label}, "
                f"marked bubbles: {marked_count}"
            )

    def _calculate_confidence_metrics(
        self, detection_result: BubbleFieldDetectionResult, file_level_aggregates
    ) -> None:
        """Calculate confidence metrics for the field.

        Simplified version - can be expanded based on threshold result.
        """
        global_threshold = file_level_aggregates.get(
            "file_level_fallback_threshold", self.threshold_result.threshold_value
        )

        # Check for disparity between global and local thresholds
        disparity_bubbles = []
        for bubble_mean in detection_result.bubble_means:
            local_marked = bubble_mean.mean_value < self.local_threshold_for_field
            global_marked = bubble_mean.mean_value < global_threshold

            if local_marked != global_marked:
                disparity_bubbles.append(bubble_mean)

        # Calculate overall confidence score for ML training
        confidence_score = self._calculate_overall_confidence_score(
            detection_result, disparity_bubbles
        )

        # Build confidence metrics
        self.insert_field_level_confidence_metrics(
            {
                "local_threshold": self.local_threshold_for_field,
                "global_threshold": global_threshold,
                "threshold_confidence": self.threshold_result.confidence,
                "threshold_method": self.threshold_result.method_used,
                "max_jump": self.threshold_result.max_jump,
                "bubbles_in_doubt": {
                    "by_disparity": disparity_bubbles,
                },
                "is_local_jump_confident": self.threshold_result.confidence > 0.7,
                "field_label": self.field.field_label,
                "scan_quality": detection_result.scan_quality.value,
                "std_deviation": detection_result.std_deviation,
                "overall_confidence_score": confidence_score,  # NEW: For ML training
            }
        )

        if len(disparity_bubbles) > 0:
            logger.warning(
                f"Threshold disparity in field: {self.field.field_label}, "
                f"bubbles in doubt: {len(disparity_bubbles)}"
            )

    def _calculate_overall_confidence_score(
        self, detection_result: BubbleFieldDetectionResult, disparity_bubbles
    ) -> float:
        """Calculate overall confidence score for this field's detection.

        Score ranges from 0.0 to 1.0 based on:
        - Threshold margin (how far marked bubbles are from threshold)
        - Multi-mark probability
        - Bubble intensity consistency within field
        - Disparity with global threshold

        Returns:
            float: Confidence score between 0.0 (low confidence) and 1.0 (high confidence)
        """
        if not detection_result.bubble_means:
            return 0.0

        # Factor 1: Threshold confidence from strategy (0.0-1.0)
        threshold_confidence = self.threshold_result.confidence

        # Factor 2: Margin from threshold (how clearly marked/unmarked)
        # Calculate average distance from threshold for marked bubbles
        marked_bubbles = [
            b
            for b in detection_result.bubble_means
            if b.mean_value < self.local_threshold_for_field
        ]
        if marked_bubbles:
            avg_margin = sum(
                self.local_threshold_for_field - b.mean_value for b in marked_bubbles
            ) / len(marked_bubbles)
            # Normalize margin confidence (larger margin = higher confidence)
            # Assume 50 intensity units is very confident
            margin_confidence = min(1.0, avg_margin / 50.0)
        else:
            # No bubbles marked - check unmarked confidence
            avg_distance = sum(
                b.mean_value - self.local_threshold_for_field
                for b in detection_result.bubble_means
            ) / len(detection_result.bubble_means)
            margin_confidence = min(1.0, avg_distance / 50.0)

        # Factor 3: Multi-mark penalty
        marked_count = len(marked_bubbles)
        if marked_count > 1:
            # Multi-marking reduces confidence
            multi_mark_penalty = 0.3  # Reduce by 30%
        elif marked_count == 0:
            # No marks - still confident if thresholds agree
            multi_mark_penalty = 0.1  # Slight penalty
        else:
            # Single mark - ideal
            multi_mark_penalty = 0.0

        # Factor 4: Disparity penalty
        disparity_ratio = (
            len(disparity_bubbles) / len(detection_result.bubble_means)
            if detection_result.bubble_means
            else 0
        )
        disparity_penalty = disparity_ratio * 0.4  # Up to 40% penalty

        # Factor 5: Scan quality
        scan_quality_map = {
            "EXCELLENT": 1.0,
            "GOOD": 0.9,
            "MODERATE": 0.7,
            "POOR": 0.5,
        }
        scan_quality_factor = scan_quality_map.get(
            detection_result.scan_quality.value, 0.5
        )

        # Combine factors (weighted average)
        confidence_score = (
            threshold_confidence * 0.35
            + margin_confidence * 0.25
            + scan_quality_factor * 0.20
        ) * (1.0 - multi_mark_penalty - disparity_penalty)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence_score))
