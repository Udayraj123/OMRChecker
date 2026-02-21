# Bubble Detection - Threshold Strategy Decisions

**Status**: Decision Logic Documentation
**Python Reference**: `src/processors/threshold/`, `src/processors/detection/bubbles_threshold/interpretation_pass.py`

---

## Overview

This document covers the **decision-making logic** for threshold strategy selection, including when to use global vs. local thresholds, when to fall back, and how confidence is calculated.

---

## Decision Hierarchy

```
1. Global Threshold Calculation (File-Level)
   └─> Used as fallback for all fields

2. Local Threshold Calculation (Field-Level)
   ├─> High Confidence → Use local threshold
   └─> Low Confidence → Fall back to global threshold

3. Interpretation
   ├─> is_attempted = mean_value < threshold
   └─> Multi-mark detection
```

---

## Decision 1: Global Threshold Strategy

### When to Use

**Always calculated** at the file level as a fallback for all fields.

### Algorithm Selection

```python
# Uses GlobalThresholdStrategy
strategy = GlobalThresholdStrategy()
result = strategy.calculate_threshold(all_bubble_means, config)
```

### Decision Tree

```
Input: all_bubble_means (from all fields in file)

├─> len(values) < 2?
│   ├─> YES → return default_threshold (127.5), confidence=0.0
│   └─> NO → continue

├─> Sort values: [50, 60, 180, 190, 200]
│
├─> Find max jump using looseness
│   └─> looseness=1, ls=1
│   └─> Iterate from index ls to (len-ls)
│       └─> For each i: jump = values[i+ls] - values[i-ls]
│           └─> Track max_jump and threshold
│
├─> max_jump >= min_jump?
│   ├─> YES → High confidence, use calculated threshold
│   └─> NO → Low confidence, use default_threshold (127.5 or 255)
│
└─> confidence = min(1.0, max_jump / (min_jump * 3))
```

### Example Scenarios

#### Scenario 1: Good Contrast (Clean Scan)

```python
bubble_means = [50, 55, 60, 65, 180, 185, 190, 195]
#               ^^^^^^ marked    ^^^^^^^^ unmarked

sorted = [50, 55, 60, 65, 180, 185, 190, 195]

# Find max jump
for i in range(1, 7):
    jump = sorted[i+1] - sorted[i-1]
    # i=3: jump = 180 - 60 = 120 (MAX JUMP)

threshold = 60 + 120/2 = 120
max_jump = 120
confidence = min(1.0, 120 / (30 * 3)) = 1.0  # High confidence
fallback_used = False
```

#### Scenario 2: Poor Contrast (Xeroxed OMR)

```python
bubble_means = [180, 185, 190, 195, 200, 205]
# All values are close together (no clear marked bubbles)

sorted = [180, 185, 190, 195, 200, 205]

# Find max jump
for i in range(1, 5):
    jump = sorted[i+1] - sorted[i-1]
    # i=2: jump = 195 - 185 = 10 (MAX JUMP)

max_jump = 10  # Less than min_jump (30)
threshold = 255  # Use default (fallback)
confidence = 0.0
fallback_used = True
```

#### Scenario 3: Mixed Scan Quality

```python
bubble_means = [80, 90, 140, 150, 160]
# One marked (80, 90), rest unmarked

sorted = [80, 90, 140, 150, 160]

# Find max jump
# i=1: jump = 140 - 80 = 60 (MAX JUMP)

threshold = 80 + 60/2 = 110
max_jump = 60
confidence = min(1.0, 60 / 90) = 0.67  # Medium confidence
fallback_used = False
```

---

## Decision 2: Local Threshold Strategy

### When to Use

**Always attempted** for each field, but may fall back to global threshold based on confidence.

### Algorithm Selection

```python
# Uses LocalThresholdStrategy with global fallback
strategy = LocalThresholdStrategy(global_fallback=global_threshold)
result = strategy.calculate_threshold(field_bubble_means, config)
```

### Decision Tree

```
Input: field_bubble_means (bubbles in single field)

├─> len(values) < 2?
│   ├─> YES → Use global_fallback, confidence=0.0, fallback_used=True
│   └─> NO → continue

├─> len(values) == 2?
│   ├─> YES → Two-bubble special case
│   │   ├─> gap < min_gap_two_bubbles (20)?
│   │   │   ├─> YES → Use global_fallback, confidence=0.3
│   │   │   └─> NO → Use mean(values), confidence=0.7
│   └─> NO → continue (3+ bubbles)

├─> Find max jump (3+ bubbles)
│   └─> For i in 1..(len-1):
│       └─> jump = values[i+1] - values[i-1]
│           └─> Track max_jump and local_threshold
│
├─> confident_jump = min_jump + min_jump_surplus_for_global_fallback
│   └─> max_jump >= confident_jump?
│       ├─> YES → Use local_threshold, high confidence
│       └─> NO → Use global_fallback, low confidence
│
└─> confidence = min(1.0, max_jump / (confident_jump * 2))
```

### Configuration Parameters

```python
@dataclass
class ThresholdConfig:
    min_jump: float = 30.0
    # Minimum jump to consider significant

    min_gap_two_bubbles: float = 20.0
    # Special case for 2-bubble fields

    min_jump_surplus_for_global_fallback: float = 10.0
    # Extra jump required to avoid global fallback
    # confident_jump = 30 + 10 = 40

    confident_jump_surplus_for_disparity: float = 15.0
    # Extra jump for high confidence despite disparity

    default_threshold: float = 127.5
    # Default fallback threshold
```

### Example Scenarios

#### Scenario 1: High Confidence Local (MCQ with 4 options)

```python
field_bubble_means = [55, 60, 180, 185]  # One marked (A)
global_fallback = 120

sorted = [55, 60, 180, 185]

# Find max jump
# i=1: jump = 180 - 55 = 125 (MAX JUMP)

local_threshold = 55 + 125/2 = 117.5
max_jump = 125
confident_jump = 30 + 10 = 40

# max_jump (125) >= confident_jump (40) → Use local
confidence = min(1.0, 125 / 80) = 1.0
fallback_used = False
method_used = "local_max_jump"
```

#### Scenario 2: Low Confidence Local → Global Fallback

```python
field_bubble_means = [110, 115, 120, 125]  # All close together
global_fallback = 120

sorted = [110, 115, 120, 125]

# Find max jump
# i=1: jump = 120 - 110 = 10 (MAX JUMP)

max_jump = 10
confident_jump = 30 + 10 = 40

# max_jump (10) < confident_jump (40) → Use global fallback
threshold = 120  # global_fallback
confidence = 0.4
fallback_used = True
method_used = "local_low_confidence_global_fallback"
```

#### Scenario 3: Two Bubbles - Large Gap

```python
field_bubble_means = [70, 190]  # Binary choice
global_fallback = 120

gap = 190 - 70 = 120
min_gap_two_bubbles = 20

# gap (120) >= min_gap_two_bubbles (20) → Use mean
threshold = (70 + 190) / 2 = 130
confidence = 0.7
fallback_used = False
method_used = "local_two_bubbles_mean"
```

#### Scenario 4: Two Bubbles - Small Gap

```python
field_bubble_means = [115, 125]  # Ambiguous
global_fallback = 120

gap = 125 - 115 = 10
min_gap_two_bubbles = 20

# gap (10) < min_gap_two_bubbles (20) → Use global fallback
threshold = 120  # global_fallback
confidence = 0.3
fallback_used = True
method_used = "local_two_bubbles_small_gap_fallback"
```

---

## Decision 3: Interpretation Logic

### Bubble Marking Decision

```python
def is_bubble_marked(bubble_mean_value: float, threshold: float) -> bool:
    """Darker values (lower) are marked."""
    return bubble_mean_value < threshold
```

### Decision Tree

```
For each bubble:

├─> mean_value < threshold?
│   ├─> YES → is_attempted = True
│   └─> NO → is_attempted = False

After all bubbles interpreted:

├─> Count marked bubbles
│   ├─> count == 0 → interpretation_string = "" (empty)
│   ├─> count == 1 → interpretation_string = bubble_value (e.g., "A")
│   ├─> count > 1 → is_multi_marked = True, interpretation_string = "AB" (concatenated)
│   └─> count == total_bubbles → interpretation_string = "" (ALL marked = scanning issue)
```

### Example Scenarios

#### Scenario 1: Single Mark (Ideal)

```python
bubble_means = [65, 180, 185, 190]  # A, B, C, D
bubble_values = ['A', 'B', 'C', 'D']
threshold = 120

interpretations = [
    is_attempted(65 < 120) = True → 'A',
    is_attempted(180 < 120) = False → '',
    is_attempted(185 < 120) = False → '',
    is_attempted(190 < 120) = False → '',
]

interpretation_string = 'A'
is_multi_marked = False
```

#### Scenario 2: Multi-Marking

```python
bubble_means = [65, 70, 185, 190]  # A and B marked
bubble_values = ['A', 'B', 'C', 'D']
threshold = 120

interpretations = [
    is_attempted(65 < 120) = True → 'A',
    is_attempted(70 < 120) = True → 'B',
    is_attempted(185 < 120) = False → '',
    is_attempted(190 < 120) = False → '',
]

interpretation_string = 'AB'
is_multi_marked = True
logger.warning("Multi-marking detected")
```

#### Scenario 3: No Marking

```python
bubble_means = [180, 185, 190, 195]  # None marked
bubble_values = ['A', 'B', 'C', 'D']
threshold = 120

interpretations = [all False]

interpretation_string = ''
is_multi_marked = False
```

#### Scenario 4: All Marked (Scanning Issue)

```python
bubble_means = [50, 55, 60, 65]  # ALL marked (likely issue)
bubble_values = ['A', 'B', 'C', 'D']
threshold = 120

interpretations = [all True]

# Special case: treat as unmarked
interpretation_string = ''  # Empty
is_multi_marked = False  # Not flagged as multi-mark
```

---

## Decision 4: Confidence Metrics

### Threshold Disparity Detection

```python
def detect_disparity_bubbles(
    bubble_means: list[BubbleMeanValue],
    local_threshold: float,
    global_threshold: float
) -> list[BubbleMeanValue]:
    """Find bubbles where local and global thresholds disagree."""

    disparity_bubbles = []

    for bubble in bubble_means:
        local_marked = bubble.mean_value < local_threshold
        global_marked = bubble.mean_value < global_threshold

        if local_marked != global_marked:
            disparity_bubbles.append(bubble)

    return disparity_bubbles
```

### Example Scenarios

#### Scenario 1: No Disparity (Thresholds Agree)

```python
bubble_means = [60, 180, 185, 190]
local_threshold = 120
global_threshold = 115

# Bubble 1: 60 < 120 AND 60 < 115 → Both agree (marked)
# Bubble 2: 180 > 120 AND 180 > 115 → Both agree (unmarked)
# Bubble 3: 185 > 120 AND 185 > 115 → Both agree (unmarked)
# Bubble 4: 190 > 120 AND 190 > 115 → Both agree (unmarked)

disparity_bubbles = []  # Empty
```

#### Scenario 2: Disparity (Thresholds Disagree)

```python
bubble_means = [60, 110, 180, 190]
local_threshold = 120
global_threshold = 100

# Bubble 1: 60 < 120 AND 60 < 100 → Both agree (marked)
# Bubble 2: 110 < 120 BUT 110 > 100 → DISAGREE! (local=marked, global=unmarked)
# Bubble 3: 180 > 120 AND 180 > 100 → Both agree (unmarked)
# Bubble 4: 190 > 120 AND 190 > 100 → Both agree (unmarked)

disparity_bubbles = [bubble_2]  # One bubble in doubt
logger.warning("Threshold disparity detected")
```

### Overall Confidence Score Calculation

```python
def calculate_overall_confidence_score(
    detection_result: BubbleFieldDetectionResult,
    threshold_result: ThresholdResult,
    bubble_interpretations: list[BubbleInterpretation],
    disparity_bubbles: list[BubbleMeanValue]
) -> float:
    """Calculate 0.0-1.0 confidence score."""

    # Factor 1: Threshold confidence (35%)
    threshold_confidence = threshold_result.confidence

    # Factor 2: Margin confidence (25%)
    marked_bubbles = [b for b in bubble_interpretations if b.is_attempted]
    if marked_bubbles:
        avg_margin = sum(local_threshold - b.mean_value for b in marked_bubbles) / len(marked_bubbles)
        margin_confidence = min(1.0, avg_margin / 50.0)
    else:
        avg_distance = sum(b.mean_value - local_threshold for b in detection_result.bubble_means) / len(detection_result.bubble_means)
        margin_confidence = min(1.0, avg_distance / 50.0)

    # Factor 3: Scan quality (20%)
    scan_quality_map = {"EXCELLENT": 1.0, "GOOD": 0.9, "ACCEPTABLE": 0.7, "POOR": 0.5}
    scan_quality_factor = scan_quality_map[detection_result.scan_quality.value]

    # Factor 4: Multi-mark penalty
    marked_count = len(marked_bubbles)
    if marked_count > 1:
        multi_mark_penalty = 0.3
    elif marked_count == 0:
        multi_mark_penalty = 0.1
    else:
        multi_mark_penalty = 0.0

    # Factor 5: Disparity penalty
    disparity_ratio = len(disparity_bubbles) / len(detection_result.bubble_means)
    disparity_penalty = disparity_ratio * 0.4

    # Combine
    confidence_score = (
        threshold_confidence * 0.35 +
        margin_confidence * 0.25 +
        scan_quality_factor * 0.20
    ) * (1.0 - multi_mark_penalty - disparity_penalty)

    return max(0.0, min(1.0, confidence_score))
```

### Confidence Score Examples

#### Example 1: High Confidence

```python
threshold_confidence = 1.0  # Large jump
margin_confidence = 1.0     # 60 units from threshold
scan_quality_factor = 1.0   # EXCELLENT (std > 50)
multi_mark_penalty = 0.0    # Single mark
disparity_penalty = 0.0     # No disparity

score = (1.0*0.35 + 1.0*0.25 + 1.0*0.20) * (1.0 - 0.0 - 0.0)
      = 0.80 * 1.0
      = 0.80  # High confidence
```

#### Example 2: Medium Confidence

```python
threshold_confidence = 0.7  # Moderate jump
margin_confidence = 0.6     # 30 units from threshold
scan_quality_factor = 0.9   # GOOD
multi_mark_penalty = 0.0    # Single mark
disparity_penalty = 0.1     # 25% disparity

score = (0.7*0.35 + 0.6*0.25 + 0.9*0.20) * (1.0 - 0.0 - 0.1)
      = 0.575 * 0.9
      = 0.52  # Medium confidence
```

#### Example 3: Low Confidence

```python
threshold_confidence = 0.3  # Small jump
margin_confidence = 0.4     # 20 units from threshold
scan_quality_factor = 0.5   # POOR
multi_mark_penalty = 0.3    # Multi-marking
disparity_penalty = 0.2     # 50% disparity

score = (0.3*0.35 + 0.4*0.25 + 0.5*0.20) * (1.0 - 0.3 - 0.2)
      = 0.305 * 0.5
      = 0.15  # Low confidence
```

---

## Decision 5: ML Fallback Trigger

### When to Use ML Fallback

```python
def should_use_ml_fallback(context: ProcessingContext) -> bool:
    """Determine if ML fallback should be used."""

    confidence_threshold = 0.75

    # Check overall confidence
    if context.metadata.get("overall_confidence_score", 1.0) < confidence_threshold:
        return True

    # Check for multi-marking
    if context.is_multi_marked:
        return True

    # Check for threshold disparity
    disparity_bubbles = context.metadata.get("bubbles_in_doubt", {}).get("by_disparity", [])
    if len(disparity_bubbles) > 0:
        return True

    return False
```

---

## Configuration Tuning Guide

### Adjusting for Different Scan Qualities

#### High-Quality Scans (Good Contrast)

```python
ThresholdConfig(
    min_jump=30.0,  # Standard
    min_gap_two_bubbles=20.0,  # Standard
    min_jump_surplus_for_global_fallback=10.0,  # Standard
    default_threshold=127.5,  # Mid-gray
)
```

#### Medium-Quality Scans (Moderate Contrast)

```python
ThresholdConfig(
    min_jump=25.0,  # Lower threshold
    min_gap_two_bubbles=15.0,  # More lenient
    min_jump_surplus_for_global_fallback=8.0,  # Lower surplus
    default_threshold=127.5,
)
```

#### Low-Quality Scans (Poor Contrast)

```python
ThresholdConfig(
    min_jump=20.0,  # Very lenient
    min_gap_two_bubbles=10.0,  # Very lenient
    min_jump_surplus_for_global_fallback=5.0,  # Low surplus
    default_threshold=100.0,  # Darker default (more bubbles marked)
)
```

#### Xeroxed OMRs (Very Poor Contrast)

```python
ThresholdConfig(
    min_jump=15.0,  # Extremely lenient
    min_gap_two_bubbles=8.0,  # Extremely lenient
    min_jump_surplus_for_global_fallback=3.0,  # Minimal surplus
    default_threshold=255.0,  # Treat all as unmarked
)
```

---

## Browser Implementation

```typescript
class ThresholdDecisionEngine {
  calculateGlobalThreshold(
    allBubbleMeans: number[],
    config: ThresholdConfig
  ): ThresholdResult {
    const strategy = new GlobalThresholdStrategy();
    return strategy.calculateThreshold(allBubbleMeans, config);
  }

  calculateLocalThreshold(
    fieldBubbleMeans: number[],
    globalFallback: number,
    config: ThresholdConfig
  ): ThresholdResult {
    const strategy = new LocalThresholdStrategy(globalFallback);
    return strategy.calculateThreshold(fieldBubbleMeans, config);
  }

  interpretBubbles(
    bubbleMeans: BubbleMeanValue[],
    threshold: number
  ): BubbleInterpretation[] {
    return bubbleMeans.map(bubble => ({
      bubbleMean: bubble,
      threshold,
      isAttempted: bubble.meanValue < threshold,
      bubbleValue: bubble.unitBubble.bubbleValue || ""
    }));
  }

  detectMultiMarking(interpretations: BubbleInterpretation[]): boolean {
    const markedCount = interpretations.filter(b => b.isAttempted).length;
    return markedCount > 1;
  }

  calculateConfidenceScore(
    detectionResult: BubbleFieldDetectionResult,
    thresholdResult: ThresholdResult,
    interpretations: BubbleInterpretation[],
    disparityBubbles: BubbleMeanValue[]
  ): number {
    // Same logic as Python implementation
    // ...
  }
}
```

---

## Decision Summary Table

| Decision Point | Input | Output | Fallback |
|----------------|-------|--------|----------|
| Global Threshold | All bubble means | Threshold + confidence | 127.5 or 255 |
| Local Threshold | Field bubble means | Threshold + confidence | Global threshold |
| Bubble Marking | Mean value, threshold | is_attempted (bool) | N/A |
| Multi-Marking | All interpretations | is_multi_marked (bool) | False |
| Confidence Score | Detection + Threshold | 0.0-1.0 score | 0.0 |
| ML Fallback | Confidence score | Use ML? (bool) | False |

---

## Related Documentation

- `concept.md` - Architecture and data models
- `flows.md` - Complete detection and interpretation flows
- `constraints.md` - Performance and accuracy constraints
- `integration.md` - Integration with ReadOMR processor
