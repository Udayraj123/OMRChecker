# Bubble Interpretation Logic - Core Flows

**Status**: Core Detection System
**Python Reference**: `src/processors/detection/bubbles_threshold/bubble_interpretation.py`, `src/processors/detection/bubbles_threshold/interpretation.py`
**Browser Compatibility**: Pure JavaScript logic (no OpenCV needed)

---

## Overview

Bubble interpretation is the **decision-making layer** that converts raw bubble mean intensity values into actionable results (marked/unmarked). It operates on detection results and applies threshold-based logic to determine which bubbles are marked.

This is **Phase 2** of the bubble detection pipeline:
1. **Phase 1 (Detection)**: Extract mean intensity values → `BubbleMeanValue[]`
2. **Phase 2 (Interpretation)**: Apply thresholds → `BubbleInterpretation[]` → Final answer string

---

## Key Principle: Darkness Calculation

**Core Rule**: Marked bubbles are DARKER (lower mean intensity values)

```
Intensity Scale (0-255):
┌─────────────────────────────────────────────────┐
0                  127.5                     255
(Black)           (Gray)                   (White)
  ↑                                           ↑
Marked           Threshold              Unmarked
Bubbles                                 Bubbles
```

**Detection Logic**:
```typescript
// A bubble is "attempted" (marked) if its mean is BELOW the threshold
const isAttempted = meanValue < threshold;

// Example:
// meanValue = 80  (dark) → 80 < 127.5 → TRUE (marked)
// meanValue = 200 (light) → 200 < 127.5 → FALSE (unmarked)
```

---

## Architecture

### Component Hierarchy

```
BubblesThresholdInterpretationPass
    ├─> Initialize file-level aggregates
    │   ├─> Calculate global threshold (all bubbles in file)
    │   ├─> Calculate outlier deviation threshold
    │   └─> Store in file_level_aggregates
    │
    └─> For each field:
        └─> BubblesFieldInterpretation
            ├─> Extract BubbleFieldDetectionResult from repository
            ├─> Create ThresholdConfig
            ├─> Calculate LocalThreshold (with global fallback)
            ├─> Create BubbleInterpretation[] (one per bubble)
            ├─> Check multi-marking
            └─> Calculate confidence metrics

BubbleInterpretation (single bubble result)
    ├─> bubble_mean: BubbleMeanValue
    ├─> threshold: float
    ├─> is_attempted: bool (mean_value < threshold)
    └─> bubble_value: str ('A', 'B', '1', etc.)
```

---

## Data Flow

### Phase 2A: File-Level Initialization

```python
# Called once per file before processing any fields
def initialize_file_level_aggregates(file_path):
    # Step 1: Get all bubble means across all fields from repository
    all_bubble_means = repository.get_all_bubble_means_for_current_file()
    # Returns: [BubbleMeanValue, BubbleMeanValue, ...]

    # Step 2: Calculate global threshold (file-level fallback)
    global_threshold, global_max_jump = get_fallback_threshold(all_bubble_means)
    # Uses GlobalThresholdStrategy to find max jump across ALL bubbles

    # Step 3: Calculate outlier deviation threshold
    all_std_deviations = [field.std_deviation for field in all_fields]
    outlier_threshold = get_outlier_deviation_threshold(all_std_deviations)

    # Step 4: Store for later use
    file_level_aggregates = {
        "file_level_fallback_threshold": global_threshold,  # e.g., 127.5
        "global_max_jump": global_max_jump,                # e.g., 85.3
        "outlier_deviation_threshold_for_file": outlier_threshold,  # e.g., 5.2
        "all_fields_local_thresholds": NumberAggregate(),
        "field_wise_confidence_metrics": {}
    }
```

**Why file-level threshold?**
- Acts as **safety net** when field-level threshold calculation fails
- Ensures consistent detection across similar fields
- Handles edge cases (single bubble, all same value, etc.)

---

### Phase 2B: Field-Level Interpretation

```python
def run_interpretation(field, file_level_detection_aggregates, file_level_interpretation_aggregates):
    # ========================================
    # Step 1: Extract Detection Result
    # ========================================
    detection_result = extract_detection_result(field, file_level_detection_aggregates)
    # Returns: BubbleFieldDetectionResult
    #   - field_id: "q1"
    #   - field_label: "Q1"
    #   - bubble_means: [BubbleMeanValue(80.5), BubbleMeanValue(210.3), ...]
    #   - std_deviation: 65.4 (auto-calculated)
    #   - scan_quality: "EXCELLENT" (auto-calculated)

    # ========================================
    # Step 2: Create Threshold Configuration
    # ========================================
    threshold_config = ThresholdConfig(
        min_jump=30.0,                             # Min jump to consider significant
        jump_delta=20.0,                           # Delta for two-jump detection
        min_gap_two_bubbles=20.0,                  # Min gap for 2-bubble case
        min_jump_surplus_for_global_fallback=10.0, # Extra needed to avoid global
        confident_jump_surplus_for_disparity=15.0, # Extra for high confidence
        global_threshold_margin=10.0,              # Safety margin
        outlier_deviation_threshold=5.2,           # From file-level init
        default_threshold=127.5                    # Final fallback
    )

    # ========================================
    # Step 3: Calculate Local Threshold
    # ========================================
    global_fallback = file_level_interpretation_aggregates["file_level_fallback_threshold"]
    strategy = LocalThresholdStrategy(global_fallback=global_fallback)

    threshold_result = strategy.calculate_threshold(
        detection_result.mean_values,  # [80.5, 210.3, 215.7, 208.1]
        threshold_config
    )
    # Returns: ThresholdResult
    #   - threshold_value: 145.4 (calculated)
    #   - confidence: 0.85 (high)
    #   - max_jump: 130.8 (difference between marked and unmarked clusters)
    #   - method_used: "local_max_jump"
    #   - fallback_used: False

    local_threshold = threshold_result.threshold_value  # 145.4

    # ========================================
    # Step 4: Interpret Each Bubble
    # ========================================
    bubble_interpretations = [
        BubbleInterpretation(bubble_mean, local_threshold)
        for bubble_mean in detection_result.bubble_means
    ]

    # Example:
    # BubbleInterpretation(mean=80.5, threshold=145.4)
    #   → is_attempted = 80.5 < 145.4 = TRUE
    #   → bubble_value = "A"
    #
    # BubbleInterpretation(mean=210.3, threshold=145.4)
    #   → is_attempted = 210.3 < 145.4 = FALSE
    #   → bubble_value = "B"

    # ========================================
    # Step 5: Check Multi-Marking
    # ========================================
    marked_count = sum(1 for interp in bubble_interpretations if interp.is_attempted)
    is_multi_marked = marked_count > 1

    # ========================================
    # Step 6: Generate Final Answer String
    # ========================================
    final_answer = get_field_interpretation_string(bubble_interpretations)
    # Logic:
    # - 0 marked → ""
    # - 1 marked → "A" (or "B", "C", "1", etc.)
    # - 2+ marked → "AB" (multi-mark, may need review)
    # - ALL marked → "" (treat as scanning issue)
```

---

## Core Algorithm: Local Threshold Calculation

### Algorithm Flow

```
Input: bubble_mean_values = [80.5, 210.3, 215.7, 208.1]
       config = ThresholdConfig(...)
       global_fallback = 127.5

┌─────────────────────────────────────────────────┐
│ 1. Validate Input                               │
│    - Empty? → Return global_fallback            │
│    - Single bubble? → Return global_fallback    │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 2. Sort Values                                  │
│    sorted = [80.5, 208.1, 210.3, 215.7]         │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 3. Handle 2-Bubble Special Case                 │
│    If len == 2:                                 │
│      gap = sorted[1] - sorted[0]                │
│      if gap < min_gap_two_bubbles (20.0):       │
│        → Return global_fallback                 │
│      else:                                      │
│        → Return mean(sorted) = (v1 + v2) / 2    │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 4. Find Maximum Jump (3+ bubbles)               │
│    For each i in range(1, len-1):               │
│      jump = sorted[i+1] - sorted[i-1]           │
│                                                  │
│    Example (our data):                          │
│      i=1: jump = sorted[2] - sorted[0]          │
│           = 210.3 - 80.5 = 129.8                │
│      i=2: jump = sorted[3] - sorted[1]          │
│           = 215.7 - 208.1 = 7.6                 │
│                                                  │
│    max_jump = 129.8 (largest gap)               │
│    threshold = sorted[0] + jump/2               │
│              = 80.5 + 129.8/2 = 145.4           │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 5. Check Confidence                             │
│    confident_jump = min_jump + surplus          │
│                   = 30.0 + 10.0 = 40.0          │
│                                                  │
│    If max_jump < confident_jump:                │
│      confidence = 0.4                           │
│      → Return global_fallback (low confidence)  │
│    Else:                                        │
│      confidence = min(1.0, max_jump / 80.0)     │
│                 = min(1.0, 129.8 / 80.0)        │
│                 = 1.0                           │
│      → Return local threshold (high confidence) │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 6. Return ThresholdResult                       │
│    threshold_value: 145.4                       │
│    confidence: 1.0                              │
│    max_jump: 129.8                              │
│    method_used: "local_max_jump"                │
│    fallback_used: False                         │
└─────────────────────────────────────────────────┘
```

### Why This Works

The algorithm finds the **largest gap** between consecutive sorted values, assuming:
1. **Marked bubbles cluster together** (low values: 50-120)
2. **Unmarked bubbles cluster together** (high values: 200-240)
3. **The gap between clusters is significant** (typically 80-150 units)

**Visual Example**:
```
Intensity Values (sorted):
[80.5] ← Marked (dark)
                        ← GAP = 129.8 (threshold at 145.4)
[208.1, 210.3, 215.7] ← Unmarked (light)
```

---

## BubbleInterpretation Class

### Structure

```python
class BubbleInterpretation:
    """Single bubble interpretation result."""

    def __init__(self, bubble_mean: BubbleMeanValue, threshold: float):
        self.bubble_mean = bubble_mean
        self.threshold = threshold
        self.mean_value = bubble_mean.mean_value
        self.bubble_value = bubble_mean.unit_bubble.bubble_value  # 'A', 'B', '1', etc.
        self.item_reference = bubble_mean.unit_bubble  # For drawing

        # Core decision: is this bubble marked?
        self.is_attempted = bubble_mean.mean_value < threshold

    def get_value(self) -> str:
        """Get bubble value if marked, else empty string."""
        return self.bubble_value if self.is_attempted else ""
```

### Usage Example

```python
# Input: bubble_mean with value 80.5, threshold 145.4
bubble_interp = BubbleInterpretation(bubble_mean, 145.4)

# Results:
bubble_interp.mean_value      # 80.5
bubble_interp.threshold       # 145.4
bubble_interp.is_attempted    # True (80.5 < 145.4)
bubble_interp.bubble_value    # "A"
bubble_interp.get_value()     # "A" (since marked)

# Unmarked example: mean 210.3, threshold 145.4
bubble_interp2 = BubbleInterpretation(bubble_mean2, 145.4)
bubble_interp2.is_attempted   # False (210.3 > 145.4)
bubble_interp2.get_value()    # "" (empty, not marked)
```

---

## Final Answer String Generation

### Logic Flow

```python
def get_field_interpretation_string(bubble_interpretations) -> str:
    """Generate final answer string from interpretations.

    Special cases:
    1. No bubbles marked → empty_value
    2. ALL bubbles marked → empty_value (scanning issue)
    3. 1 bubble marked → bubble_value
    4. 2+ bubbles marked → concatenated values (multi-mark)
    """
    marked_bubbles = [
        interp.bubble_value
        for interp in bubble_interpretations
        if interp.is_attempted
    ]

    # Case 1: No marks
    if len(marked_bubbles) == 0:
        return empty_value  # "" or "-"

    # Case 2: ALL bubbles marked (likely scanning error)
    total_bubbles = len(bubble_interpretations)
    if len(marked_bubbles) == total_bubbles:
        return empty_value  # Treat as unmarked

    # Case 3 & 4: Return concatenated values
    return "".join(marked_bubbles)  # "A" or "AB" or "ACD"
```

### Examples

| Marked Bubbles | Total Bubbles | Result | Reason |
|----------------|---------------|--------|--------|
| [] | 4 | `""` | No marks |
| ['A'] | 4 | `"A"` | Single mark (normal) |
| ['A', 'B'] | 4 | `"AB"` | Multi-mark (needs review) |
| ['A', 'B', 'C', 'D'] | 4 | `""` | All marked (scanning issue) |
| ['1'] | 10 | `"1"` | Single digit |
| ['1', '2', '3'] | 10 | `"123"` | Multi-digit |

**Why treat "all marked" as empty?**
- Likely indicates **scanning/alignment issues** (sheet too dark, poor contrast)
- Better to flag as "unmarked" than return invalid multi-mark
- Can be detected in confidence metrics for manual review

---

## Multi-Marking Detection

### Algorithm

```python
def check_multi_marking(bubble_interpretations) -> bool:
    """Check if multiple bubbles are marked."""
    marked_count = sum(
        1 for interp in bubble_interpretations if interp.is_attempted
    )
    is_multi_marked = marked_count > 1

    if is_multi_marked:
        logger.warning(
            f"Multi-marking detected: {marked_count} bubbles marked"
        )

    return is_multi_marked
```

### Implications

**When multi-marking is detected**:
1. `is_multi_marked = True` flag is set
2. Warning is logged
3. Final answer includes all marked values (e.g., "AB")
4. File may be flagged for manual review
5. Confidence score is reduced (30% penalty)

**Use cases**:
- Student marked multiple answers
- Correction marks not fully erased
- Stray marks near bubbles
- Intentional multi-select fields (e.g., "Select all that apply")

---

## Confidence Metrics Calculation

### Overall Confidence Score

Confidence score (0.0-1.0) combines multiple factors:

```python
def calculate_overall_confidence_score(
    detection_result: BubbleFieldDetectionResult,
    disparity_bubbles: list
) -> float:
    """Calculate 0.0-1.0 confidence score."""

    # ========================================
    # Factor 1: Threshold Confidence (35%)
    # ========================================
    # From ThresholdResult (based on jump size)
    threshold_confidence = threshold_result.confidence  # 0.0-1.0

    # ========================================
    # Factor 2: Margin Confidence (25%)
    # ========================================
    # How far are marked bubbles from threshold?
    marked_bubbles = [b for b in bubbles if b.mean_value < threshold]

    if marked_bubbles:
        # Average distance below threshold
        avg_margin = sum(threshold - b.mean_value for b in marked_bubbles) / len(marked_bubbles)
        # Normalize: 50 units = very confident
        margin_confidence = min(1.0, avg_margin / 50.0)
    else:
        # No marks: check how far unmarked bubbles are above threshold
        avg_distance = sum(b.mean_value - threshold for b in bubbles) / len(bubbles)
        margin_confidence = min(1.0, avg_distance / 50.0)

    # ========================================
    # Factor 3: Scan Quality (20%)
    # ========================================
    scan_quality_map = {
        "EXCELLENT": 1.0,  # std > 50
        "GOOD": 0.9,       # std > 30
        "ACCEPTABLE": 0.7, # std > 15
        "POOR": 0.5        # std <= 15
    }
    scan_quality_factor = scan_quality_map[detection_result.scan_quality]

    # ========================================
    # Factor 4: Multi-Mark Penalty (30%)
    # ========================================
    marked_count = len(marked_bubbles)
    if marked_count > 1:
        multi_mark_penalty = 0.3  # 30% reduction
    elif marked_count == 0:
        multi_mark_penalty = 0.1  # 10% reduction
    else:
        multi_mark_penalty = 0.0  # No penalty

    # ========================================
    # Factor 5: Disparity Penalty (40%)
    # ========================================
    # Disparity = bubbles marked differently by local vs global threshold
    disparity_ratio = len(disparity_bubbles) / len(bubbles)
    disparity_penalty = disparity_ratio * 0.4

    # ========================================
    # Final Score (weighted average)
    # ========================================
    confidence_score = (
        threshold_confidence * 0.35 +
        margin_confidence * 0.25 +
        scan_quality_factor * 0.20
    ) * (1.0 - multi_mark_penalty - disparity_penalty)

    return max(0.0, min(1.0, confidence_score))
```

### Confidence Metrics Output

```python
confidence_metrics = {
    # Threshold info
    "local_threshold": 145.4,
    "global_threshold": 127.5,
    "threshold_confidence": 1.0,
    "threshold_method": "local_max_jump",
    "max_jump": 129.8,

    # Scan quality
    "scan_quality": "EXCELLENT",
    "std_deviation": 65.4,

    # Disparity detection
    "bubbles_in_doubt": {
        "by_disparity": [bubble_mean_1, bubble_mean_2]
    },
    "is_local_jump_confident": True,

    # Overall
    "field_label": "Q1",
    "overall_confidence_score": 0.85  # High confidence
}
```

---

## Browser Migration

### TypeScript Implementation

```typescript
// ========================================
// BubbleInterpretation Class
// ========================================
interface BubbleInterpretation {
  bubbleMean: BubbleMeanValue;
  threshold: number;
  meanValue: number;
  isAttempted: boolean;
  bubbleValue: string;
  itemReference: BubblesScanBox;
}

function createBubbleInterpretation(
  bubbleMean: BubbleMeanValue,
  threshold: number
): BubbleInterpretation {
  const isAttempted = bubbleMean.meanValue < threshold;

  return {
    bubbleMean,
    threshold,
    meanValue: bubbleMean.meanValue,
    isAttempted,
    bubbleValue: bubbleMean.unitBubble.bubbleValue || '',
    itemReference: bubbleMean.unitBubble,
  };
}

// ========================================
// Field Interpretation
// ========================================
interface BubblesFieldInterpretation {
  field: Field;
  bubbleInterpretations: BubbleInterpretation[];
  localThreshold: number;
  thresholdResult: ThresholdResult;
  isMultiMarked: boolean;
  confidenceMetrics?: ConfidenceMetrics;
}

function interpretBubbleField(
  field: Field,
  detectionResult: BubbleFieldDetectionResult,
  fileLevelAggregates: FileLevelAggregates,
  config: ThresholdConfig
): BubblesFieldInterpretation {
  // Step 1: Calculate threshold
  const globalFallback = fileLevelAggregates.fileLevelFallbackThreshold;
  const strategy = new LocalThresholdStrategy(globalFallback);
  const thresholdResult = strategy.calculateThreshold(
    detectionResult.meanValues,
    config
  );

  // Step 2: Interpret bubbles
  const bubbleInterpretations = detectionResult.bubbleMeans.map(
    (bubbleMean) => createBubbleInterpretation(bubbleMean, thresholdResult.thresholdValue)
  );

  // Step 3: Check multi-marking
  const markedCount = bubbleInterpretations.filter(b => b.isAttempted).length;
  const isMultiMarked = markedCount > 1;

  // Step 4: Calculate confidence (if needed)
  const confidenceMetrics = config.showConfidenceMetrics
    ? calculateConfidenceMetrics(detectionResult, thresholdResult, bubbleInterpretations)
    : undefined;

  return {
    field,
    bubbleInterpretations,
    localThreshold: thresholdResult.thresholdValue,
    thresholdResult,
    isMultiMarked,
    confidenceMetrics,
  };
}

// ========================================
// Final Answer String
// ========================================
function getFieldInterpretationString(
  bubbleInterpretations: BubbleInterpretation[],
  emptyValue: string = ''
): string {
  const markedBubbles = bubbleInterpretations
    .filter(b => b.isAttempted)
    .map(b => b.bubbleValue);

  // No marks
  if (markedBubbles.length === 0) {
    return emptyValue;
  }

  // All marked (scanning issue)
  if (markedBubbles.length === bubbleInterpretations.length) {
    return emptyValue;
  }

  // Return concatenated values
  return markedBubbles.join('');
}
```

### Pure JavaScript (No Dependencies)

All bubble interpretation logic is **pure JavaScript** - no external libraries needed:
- Threshold calculation: Basic arithmetic, sorting, Math.min/max
- Interpretation: Simple comparisons (`<` operator)
- Confidence: Arithmetic operations

**No OpenCV.js needed** (interpretation happens after detection).

---

## Integration with Pipeline

### Repository Pattern

```typescript
// Detection Repository stores detection results
class DetectionRepository {
  private bubbleFields: Map<string, BubbleFieldDetectionResult> = new Map();
  private currentFile: string = '';

  saveBubbleField(fieldId: string, result: BubbleFieldDetectionResult): void {
    this.bubbleFields.set(fieldId, result);
  }

  getBubbleField(fieldId: string): BubbleFieldDetectionResult | undefined {
    return this.bubbleFields.get(fieldId);
  }

  getAllBubbleMeansForCurrentFile(): BubbleMeanValue[] {
    return Array.from(this.bubbleFields.values())
      .flatMap(field => field.bubbleMeans);
  }

  getAllBubbleFieldsForCurrentFile(): Map<string, BubbleFieldDetectionResult> {
    return this.bubbleFields;
  }

  clearCurrentFile(): void {
    this.bubbleFields.clear();
  }
}
```

### Pipeline Integration

```typescript
// Main OMR Processing Pipeline
async function processOMRSheet(
  image: cv.Mat,
  template: Template,
  config: TuningConfig
): Promise<ProcessingResult> {
  const repository = new DetectionRepository();

  // ========================================
  // Phase 1: Detection Pass (extract values)
  // ========================================
  for (const field of template.bubbleFields) {
    const detectionResult = await runBubbleDetection(field, image);
    repository.saveBubbleField(field.id, detectionResult);
  }

  // ========================================
  // Phase 2: Interpretation Pass (apply thresholds)
  // ========================================
  const fileLevelAggregates = initializeFileLevelAggregates(repository, config);

  const interpretations: Map<string, BubblesFieldInterpretation> = new Map();

  for (const field of template.bubbleFields) {
    const detectionResult = repository.getBubbleField(field.id)!;
    const interpretation = interpretBubbleField(
      field,
      detectionResult,
      fileLevelAggregates,
      config.thresholding
    );
    interpretations.set(field.id, interpretation);
  }

  // ========================================
  // Phase 3: Generate Output
  // ========================================
  return generateResults(interpretations);
}
```

---

## Related Documentation

- **Detection Pass**: `../detection-pass/flows.md` - ROI extraction and mean calculation
- **Threshold Strategies**: `../../../threshold/` - Global and Local threshold algorithms
- **Confidence Metrics**: `constraints.md` - Performance and edge cases
- **Drawing**: `../drawing/flows.md` - Visual feedback for interpretations

---

## Summary

**Bubble Interpretation** converts detection results into actionable answers:

1. **Input**: `BubbleMeanValue[]` (from detection pass)
2. **Process**: Apply threshold strategies (global + local)
3. **Output**: `BubbleInterpretation[]` → Final answer string

**Key Features**:
- Darkness-based detection (`mean < threshold`)
- Two-level thresholding (file → field)
- Confidence scoring (0.0-1.0)
- Multi-mark detection
- All-marked safety check
- Pure JavaScript (browser-friendly)

**Next**: See `constraints.md` for performance characteristics and edge cases.
