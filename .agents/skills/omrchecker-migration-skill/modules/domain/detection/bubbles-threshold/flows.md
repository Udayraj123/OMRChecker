# Bubble Detection - Complete Flows

**Status**: Detailed Flow Documentation
**Python Reference**: `src/processors/detection/bubbles_threshold/`

---

## Overview

This document covers the **complete end-to-end flow** of bubble detection using threshold strategies, including all edge cases and error handling.

---

## High-Level Flow

```
1. Detection Pass (Computer Vision)
   ├─> Extract mean intensity from each bubble ROI
   ├─> Calculate field-level statistics
   └─> Store in DetectionRepository

2. Interpretation Pass (Threshold Logic)
   ├─> Calculate Global Threshold (file-level)
   ├─> Calculate Local Threshold (field-level)
   ├─> Apply threshold to each bubble
   ├─> Check for multi-marking
   └─> Generate confidence metrics
```

---

## Flow 1: Detection Pass (BubblesThresholdDetectionPass)

### Initialization

```python
# Called once per directory
def initialize_directory_level_aggregates(initial_directory_path):
    aggregates = {
        "file_wise_thresholds": NumberAggregate(),
    }
    # Repository initialized by TemplateFileRunner
```

```python
# Called once per file
def initialize_file_level_aggregates(file_path):
    aggregates = {
        "global_max_jump": None,
        "all_field_bubble_means": [],
        "all_field_bubble_means_std": [],
    }
    # Repository.new_file(file_path) called by TemplateFileRunner
```

### Field Detection Flow

```python
def process_field(field: Field, gray_image, colored_image):
    """Process a single bubble field."""

    # Step 1: Create field detection instance
    field_detection = BubblesFieldDetection(field, gray_image, colored_image)

    # Step 2: Run detection
    field_detection.run_detection(field, gray_image, colored_image)

    # Step 3: Verify result was created
    if field_detection.result is None:
        raise ValueError(f"Detection failed for field {field.id}")

    # Step 4: Save to repository
    repository.save_bubble_field(field.id, field_detection.result)

    # Step 5: Update aggregates
    bubble_means = field_detection.result.bubble_means
    std_deviation = field_detection.result.std_deviation

    # Field-level aggregates
    field_level_aggregates = {
        "field_bubble_means": bubble_means,
        "field_bubble_means_std": std_deviation,
    }

    # File-level aggregates
    file_level_aggregates["all_field_bubble_means"].extend(bubble_means)
    file_level_aggregates["all_field_bubble_means_std"].append(std_deviation)
```

### Bubble Mean Extraction

```python
@staticmethod
def read_bubble_mean_value(unit_bubble: BubblesScanBox, gray_image) -> BubbleMeanValue:
    """Extract mean intensity from a single bubble ROI."""

    # 1. Get bubble dimensions and position
    box_w, box_h = unit_bubble.bubble_dimensions
    x, y = unit_bubble.get_shifted_position()  # Account for preprocessing shifts

    # 2. Extract ROI
    rect = [y, y + box_h, x, x + box_w]
    roi = gray_image[rect[0]:rect[1], rect[2]:rect[3]]

    # 3. Calculate mean intensity (0-255)
    mean_value = cv2.mean(roi, None)[0]

    # 4. Create typed result
    return BubbleMeanValue(
        mean_value=mean_value,
        unit_bubble=unit_bubble,
        position=(x, y)
    )
```

### Edge Cases in Detection Pass

#### Edge Case 1: Empty or Out-of-Bounds ROI

```python
# If bubble position is outside image bounds:
x, y = unit_bubble.get_shifted_position()
if x < 0 or y < 0 or x + box_w > image.width or y + box_h > image.height:
    logger.warning(f"Bubble ROI out of bounds: {unit_bubble}")
    # cv2.mean will return 0 for empty ROI
    mean_value = 0.0  # Or use default value
```

#### Edge Case 2: Zero-Size Bubble

```python
box_w, box_h = unit_bubble.bubble_dimensions
if box_w <= 0 or box_h <= 0:
    logger.error(f"Invalid bubble dimensions: {box_w}x{box_h}")
    # Use default mean value or skip
    mean_value = 127.5
```

#### Edge Case 3: No Bubbles in Field

```python
if len(field.scan_boxes) == 0:
    logger.warning(f"No bubbles defined in field {field.id}")
    result = BubbleFieldDetectionResult(
        field_id=field.id,
        field_label=field.field_label,
        bubble_means=[]  # Empty list
    )
    # std_deviation will be 0.0
    # scan_quality will be POOR
```

---

## Flow 2: Interpretation Pass (BubblesThresholdInterpretationPass)

### Initialization

```python
def initialize_file_level_aggregates(file_path):
    """Initialize interpretation aggregates."""

    # 1. Get all bubble means from repository
    all_bubble_means = repository.get_all_bubble_means_for_current_file()

    # 2. Get std deviations from all fields
    all_outlier_deviations = [
        field_result.std_deviation
        for field_result in repository.get_all_bubble_fields_for_current_file().values()
    ]

    # 3. Calculate outlier deviation threshold (for scan quality)
    outlier_deviation_threshold = get_outlier_deviation_threshold(all_outlier_deviations)

    # 4. Calculate global fallback threshold
    global_threshold, global_max_jump = get_fallback_threshold(all_bubble_means)

    logger.debug(
        f"Thresholding: file_level_fallback_threshold: {round(global_threshold, 2)} "
        f"global_std_THR: {round(outlier_deviation_threshold, 2)} "
        f"{'(Looks like a Xeroxed OMR)' if global_threshold == 255 else ''}"
    )

    # 5. Store aggregates
    aggregates = {
        "file_level_fallback_threshold": global_threshold,
        "global_max_jump": global_max_jump,
        "outlier_deviation_threshold_for_file": outlier_deviation_threshold,
        "field_label_wise_local_thresholds": {},
        "bubble_field_type_wise_thresholds": {},
        "all_fields_local_thresholds": NumberAggregate(),
        "field_wise_confidence_metrics": {},
    }
```

### Global Threshold Calculation

```python
def get_fallback_threshold(field_wise_means_and_refs: list[BubbleMeanValue]) -> tuple[float, float]:
    """Calculate global threshold using GlobalThresholdStrategy."""

    # 1. Create threshold config
    config = ThresholdConfig(
        min_jump=tuning_config.thresholding.min_jump,
        default_threshold=tuning_config.thresholding.global_page_threshold,
    )

    # 2. Extract values from BubbleMeanValue objects
    bubble_values = [item.mean_value for item in field_wise_means_and_refs]

    # 3. Calculate threshold
    strategy = GlobalThresholdStrategy()
    result = strategy.calculate_threshold(bubble_values, config)

    # 4. Return threshold and max jump
    return result.threshold_value, result.max_jump
```

### Outlier Deviation Threshold

```python
def get_outlier_deviation_threshold(all_outlier_deviations: list[float]) -> float:
    """Calculate outlier deviation threshold for scan quality assessment."""

    config = ThresholdConfig(
        min_jump=tuning_config.thresholding.min_jump_std,
        default_threshold=tuning_config.thresholding.global_page_threshold_std,
    )

    strategy = GlobalThresholdStrategy()
    result = strategy.calculate_threshold(all_outlier_deviations, config)

    return result.threshold_value
```

### Field Interpretation Flow

```python
def interpret_field(field: Field, file_level_aggregates):
    """Interpret a single bubble field."""

    # Step 1: Extract detection result from repository
    detection_result = repository.get_bubble_field(field.field_label)

    if detection_result is None:
        raise ValueError(f"No detection result for field {field.field_label}")

    # Step 2: Create threshold config
    threshold_config = ThresholdConfig(
        min_jump=tuning_config.thresholding.min_jump,
        jump_delta=tuning_config.thresholding.jump_delta,
        min_gap_two_bubbles=tuning_config.thresholding.min_gap_two_bubbles,
        min_jump_surplus_for_global_fallback=tuning_config.thresholding.min_jump_surplus_for_global_fallback,
        confident_jump_surplus_for_disparity=tuning_config.thresholding.confident_jump_surplus_for_disparity,
        global_threshold_margin=tuning_config.thresholding.global_threshold_margin,
        outlier_deviation_threshold=file_level_aggregates["outlier_deviation_threshold_for_file"],
        default_threshold=tuning_config.thresholding.global_page_threshold,
    )

    # Step 3: Calculate local threshold
    global_fallback = file_level_aggregates["file_level_fallback_threshold"]
    strategy = LocalThresholdStrategy(global_fallback=global_fallback)
    threshold_result = strategy.calculate_threshold(
        detection_result.mean_values,
        threshold_config
    )

    local_threshold = threshold_result.threshold_value

    # Step 4: Interpret each bubble
    bubble_interpretations = [
        BubbleInterpretation(bubble_mean, local_threshold)
        for bubble_mean in detection_result.bubble_means
    ]

    # Step 5: Check multi-marking
    marked_count = sum(1 for b in bubble_interpretations if b.is_attempted)
    is_multi_marked = marked_count > 1

    if is_multi_marked:
        logger.warning(
            f"Multi-marking detected in field: {field.field_label}, "
            f"marked bubbles: {marked_count}"
        )

    # Step 6: Generate interpretation string
    marked_bubbles = [
        b.bubble_value for b in bubble_interpretations if b.is_attempted
    ]

    # Special case: ALL bubbles marked (likely scanning issue)
    if len(marked_bubbles) == len(bubble_interpretations):
        interpretation_string = ""  # Empty value
    elif len(marked_bubbles) == 0:
        interpretation_string = ""  # Empty value
    else:
        interpretation_string = "".join(marked_bubbles)

    # Step 7: Calculate confidence metrics (if enabled)
    if tuning_config.outputs.show_confidence_metrics:
        confidence_metrics = calculate_confidence_metrics(
            detection_result,
            threshold_result,
            bubble_interpretations,
            global_fallback,
            local_threshold
        )

    return {
        "interpretation_string": interpretation_string,
        "is_multi_marked": is_multi_marked,
        "local_threshold": local_threshold,
        "bubble_interpretations": bubble_interpretations,
        "threshold_result": threshold_result,
        "confidence_metrics": confidence_metrics,
    }
```

### Confidence Metrics Calculation

```python
def calculate_confidence_metrics(
    detection_result: BubbleFieldDetectionResult,
    threshold_result: ThresholdResult,
    bubble_interpretations: list[BubbleInterpretation],
    global_threshold: float,
    local_threshold: float
) -> dict:
    """Calculate comprehensive confidence metrics."""

    # 1. Find bubbles with disparity between global and local thresholds
    disparity_bubbles = []
    for bubble_mean in detection_result.bubble_means:
        local_marked = bubble_mean.mean_value < local_threshold
        global_marked = bubble_mean.mean_value < global_threshold

        if local_marked != global_marked:
            disparity_bubbles.append(bubble_mean)

    # 2. Calculate overall confidence score
    confidence_score = calculate_overall_confidence_score(
        detection_result,
        threshold_result,
        bubble_interpretations,
        disparity_bubbles
    )

    # 3. Build metrics
    metrics = {
        "local_threshold": local_threshold,
        "global_threshold": global_threshold,
        "threshold_confidence": threshold_result.confidence,
        "threshold_method": threshold_result.method_used,
        "max_jump": threshold_result.max_jump,
        "bubbles_in_doubt": {
            "by_disparity": disparity_bubbles,
        },
        "is_local_jump_confident": threshold_result.confidence > 0.7,
        "field_label": field.field_label,
        "scan_quality": detection_result.scan_quality.value,
        "std_deviation": detection_result.std_deviation,
        "overall_confidence_score": confidence_score,
    }

    # 4. Log warnings for low confidence
    if len(disparity_bubbles) > 0:
        logger.warning(
            f"Threshold disparity in field: {field.field_label}, "
            f"bubbles in doubt: {len(disparity_bubbles)}"
        )

    return metrics
```

### Overall Confidence Score

```python
def calculate_overall_confidence_score(
    detection_result: BubbleFieldDetectionResult,
    threshold_result: ThresholdResult,
    bubble_interpretations: list[BubbleInterpretation],
    disparity_bubbles: list[BubbleMeanValue]
) -> float:
    """Calculate 0.0-1.0 confidence score for ML training."""

    if not detection_result.bubble_means:
        return 0.0

    # Factor 1: Threshold confidence (0.0-1.0)
    threshold_confidence = threshold_result.confidence

    # Factor 2: Margin confidence (how far from threshold)
    marked_bubbles = [b for b in bubble_interpretations if b.is_attempted]

    if marked_bubbles:
        # Average distance from threshold for marked bubbles
        avg_margin = sum(
            local_threshold - b.mean_value for b in marked_bubbles
        ) / len(marked_bubbles)
        margin_confidence = min(1.0, avg_margin / 50.0)
    else:
        # No bubbles marked - check unmarked confidence
        avg_distance = sum(
            b.mean_value - local_threshold
            for b in detection_result.bubble_means
        ) / len(detection_result.bubble_means)
        margin_confidence = min(1.0, avg_distance / 50.0)

    # Factor 3: Multi-mark penalty
    marked_count = len(marked_bubbles)
    if marked_count > 1:
        multi_mark_penalty = 0.3  # 30% penalty
    elif marked_count == 0:
        multi_mark_penalty = 0.1  # 10% penalty
    else:
        multi_mark_penalty = 0.0  # No penalty

    # Factor 4: Disparity penalty
    disparity_ratio = (
        len(disparity_bubbles) / len(detection_result.bubble_means)
        if detection_result.bubble_means else 0
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
        threshold_confidence * 0.35 +
        margin_confidence * 0.25 +
        scan_quality_factor * 0.20
    ) * (1.0 - multi_mark_penalty - disparity_penalty)

    # Clamp to [0, 1]
    return max(0.0, min(1.0, confidence_score))
```

---

## Flow 3: Global Threshold Strategy

```python
def calculate_threshold(bubble_mean_values: list[float], config: ThresholdConfig) -> ThresholdResult:
    """Calculate global threshold by finding largest gap in sorted values."""

    # Edge case 1: Empty or single value
    if len(bubble_mean_values) < 2:
        return ThresholdResult(
            threshold_value=config.default_threshold,
            confidence=0.0,
            max_jump=0.0,
            method_used="global_default",
            fallback_used=True,
        )

    # Step 1: Sort values
    sorted_values = sorted(bubble_mean_values)

    # Step 2: Find FIRST LARGE GAP using looseness
    looseness = 1
    ls = (looseness + 1) // 2
    total_bubbles_loose = len(sorted_values) - ls

    max_jump = config.min_jump
    threshold = config.default_threshold

    # Step 3: Iterate through sorted values
    for i in range(ls, total_bubbles_loose):
        jump = sorted_values[i + ls] - sorted_values[i - ls]
        if jump > max_jump:
            max_jump = jump
            threshold = sorted_values[i - ls] + jump / 2

    # Step 4: Calculate confidence
    confidence = min(1.0, max_jump / (config.min_jump * 3))

    # Step 5: Return result
    return ThresholdResult(
        threshold_value=threshold,
        confidence=confidence,
        max_jump=max_jump,
        method_used="global_max_jump",
        fallback_used=max_jump < config.min_jump,
        metadata={
            "num_bubbles": len(bubble_mean_values),
            "min_value": min(bubble_mean_values),
            "max_value": max(bubble_mean_values),
        },
    )
```

---

## Flow 4: Local Threshold Strategy

```python
def calculate_threshold(bubble_mean_values: list[float], config: ThresholdConfig) -> ThresholdResult:
    """Calculate local threshold with global fallback."""

    fallback_threshold = self.global_fallback or config.default_threshold

    # Edge Case 1: Empty or single bubble
    if len(bubble_mean_values) < 2:
        return ThresholdResult(
            threshold_value=fallback_threshold,
            confidence=0.0,
            max_jump=0.0,
            method_used="local_single_bubble_fallback",
            fallback_used=True,
        )

    sorted_values = sorted(bubble_mean_values)

    # Edge Case 2: Exactly 2 bubbles
    if len(sorted_values) == 2:
        gap = sorted_values[1] - sorted_values[0]

        # Small gap: use global fallback
        if gap < config.min_gap_two_bubbles:
            return ThresholdResult(
                threshold_value=fallback_threshold,
                confidence=0.3,
                max_jump=gap,
                method_used="local_two_bubbles_small_gap_fallback",
                fallback_used=True,
            )

        # Good gap: use mean of two values
        return ThresholdResult(
            threshold_value=float(np.mean(sorted_values)),
            confidence=0.7,
            max_jump=gap,
            method_used="local_two_bubbles_mean",
            fallback_used=False,
        )

    # Normal Case: 3+ bubbles, find largest jump
    max_jump = 0.0
    threshold = fallback_threshold

    for i in range(1, len(sorted_values) - 1):
        jump = sorted_values[i + 1] - sorted_values[i - 1]
        if jump > max_jump:
            max_jump = jump
            threshold = sorted_values[i - 1] + jump / 2

    # Check if jump is confident enough
    confident_jump = config.min_jump + config.min_jump_surplus_for_global_fallback

    if max_jump < confident_jump:
        # Low confidence - use global fallback
        return ThresholdResult(
            threshold_value=fallback_threshold,
            confidence=0.4,
            max_jump=max_jump,
            method_used="local_low_confidence_global_fallback",
            fallback_used=True,
            metadata={"local_threshold": threshold},
        )

    # High confidence - use local threshold
    confidence = min(1.0, max_jump / (confident_jump * 2))

    return ThresholdResult(
        threshold_value=threshold,
        confidence=confidence,
        max_jump=max_jump,
        method_used="local_max_jump",
        fallback_used=False,
        metadata={"num_bubbles": len(bubble_mean_values)},
    )
```

---

## Edge Cases Summary

### Detection Edge Cases

| Edge Case | Behavior | Fallback |
|-----------|----------|----------|
| Empty field (no bubbles) | Create result with empty bubble_means | std=0, quality=POOR |
| ROI out of bounds | cv2.mean returns 0 or partial | Log warning, continue |
| Zero-size bubble | Skip or use default mean | Use 127.5 |
| Null/corrupt image | cv2.mean raises error | Catch and log, skip file |

### Interpretation Edge Cases

| Edge Case | Behavior | Fallback |
|-----------|----------|----------|
| Single bubble in field | Use global threshold | confidence=0.0 |
| Two bubbles, small gap | Use global threshold | confidence=0.3 |
| Two bubbles, large gap | Use mean of two values | confidence=0.7 |
| All bubbles same value | max_jump=0, use default | confidence=0.0 |
| All bubbles marked | Return empty string | Treat as unmarked |
| No bubbles marked | Return empty string | Normal case |
| Low global confidence | Use default threshold | 127.5 or 255 |

### Threshold Edge Cases

| Edge Case | Threshold | Confidence |
|-----------|-----------|------------|
| Empty file (no fields) | 127.5 | 0.0 |
| Xeroxed OMR (no contrast) | 255.0 | 0.0 |
| Poor scan quality (low std) | Global fallback | 0.0-0.4 |
| High scan quality (high std) | Local threshold | 0.7-1.0 |

---

## Browser Implementation

### Detection Pass (TypeScript + OpenCV.js)

```typescript
function detectBubbleField(
  field: Field,
  grayImage: cv.Mat
): BubbleFieldDetectionResult {
  const bubbleMeans: BubbleMeanValue[] = [];

  for (const bubble of field.scanBoxes) {
    const [x, y] = bubble.getShiftedPosition();
    const [w, h] = bubble.bubbleDimensions;

    // Extract ROI
    const roi = grayImage.roi(new cv.Rect(x, y, w, h));

    // Calculate mean
    const mean = cv.mean(roi);
    const meanValue = mean[0];

    // Store result
    bubbleMeans.push({
      meanValue,
      unitBubble: bubble,
      position: [x, y]
    });

    // CRITICAL: Free memory
    roi.delete();
  }

  return {
    fieldId: field.id,
    fieldLabel: field.fieldLabel,
    bubbleMeans,

    // Computed properties
    get stdDeviation() {
      const values = this.bubbleMeans.map(b => b.meanValue);
      return calculateStdDev(values);
    },

    get scanQuality() {
      const std = this.stdDeviation;
      if (std > 50) return ScanQuality.EXCELLENT;
      if (std > 30) return ScanQuality.GOOD;
      if (std > 15) return ScanQuality.ACCEPTABLE;
      return ScanQuality.POOR;
    },

    get meanValues() {
      return this.bubbleMeans.map(b => b.meanValue);
    }
  };
}
```

### Interpretation Pass (Pure TypeScript)

```typescript
function interpretBubbleField(
  field: Field,
  detectionResult: BubbleFieldDetectionResult,
  globalThreshold: number,
  config: ThresholdConfig
): BubbleFieldInterpretation {
  // Calculate local threshold
  const strategy = new LocalThresholdStrategy(globalThreshold);
  const thresholdResult = strategy.calculateThreshold(
    detectionResult.meanValues,
    config
  );

  const localThreshold = thresholdResult.thresholdValue;

  // Interpret bubbles
  const interpretations = detectionResult.bubbleMeans.map(bubble =>
    new BubbleInterpretation(bubble, localThreshold)
  );

  // Check multi-marking
  const markedCount = interpretations.filter(b => b.isAttempted).length;
  const isMultiMarked = markedCount > 1;

  // Generate interpretation string
  const markedBubbles = interpretations
    .filter(b => b.isAttempted)
    .map(b => b.bubbleValue);

  let interpretationString = "";
  if (markedBubbles.length > 0 && markedBubbles.length < interpretations.length) {
    interpretationString = markedBubbles.join("");
  }

  return {
    interpretationString,
    isMultiMarked,
    localThreshold,
    bubbleInterpretations: interpretations,
    thresholdResult
  };
}
```

---

## Performance Optimization

### Memory Management (Browser)

```typescript
// Pool ROI processing to avoid memory leaks
class BubbleDetector {
  private tempRoi: cv.Mat | null = null;

  detectField(field: Field, grayImage: cv.Mat): BubbleFieldDetectionResult {
    const bubbleMeans: BubbleMeanValue[] = [];

    for (const bubble of field.scanBoxes) {
      // Reuse temp mat instead of creating new ones
      if (!this.tempRoi) {
        this.tempRoi = new cv.Mat();
      }

      const rect = new cv.Rect(bubble.x, bubble.y, bubble.w, bubble.h);
      grayImage.roi(rect).copyTo(this.tempRoi);

      const mean = cv.mean(this.tempRoi);
      bubbleMeans.push({
        meanValue: mean[0],
        unitBubble: bubble,
        position: [bubble.x, bubble.y]
      });
    }

    return { fieldId: field.id, fieldLabel: field.fieldLabel, bubbleMeans };
  }

  cleanup() {
    if (this.tempRoi) {
      this.tempRoi.delete();
      this.tempRoi = null;
    }
  }
}
```

---

## Related Documentation

- `concept.md` - Architecture and data models
- `decisions.md` - Threshold strategy selection logic
- `constraints.md` - Performance and accuracy constraints
- `integration.md` - Integration with ReadOMR processor
