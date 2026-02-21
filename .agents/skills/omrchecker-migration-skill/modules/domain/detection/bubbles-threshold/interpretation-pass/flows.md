# Bubble Detection - Interpretation Pass: Flows

## Overview

The **Interpretation Pass** is the second major stage of bubble detection, following the Detection Pass. It converts raw bubble intensity measurements into meaningful values (marked/unmarked), handles multi-marking, and calculates confidence metrics.

**Python Reference**: `src/processors/detection/bubbles_threshold/interpretation_pass.py`, `interpretation.py`

## Key Components

### 1. BubblesThresholdInterpretationPass (File-level Orchestrator)

Main orchestrator managing file-level and field-level interpretation aggregates.

```python
class BubblesThresholdInterpretationPass(FieldTypeInterpretationPass):
    def __init__(self, repository: DetectionRepository)
    def initialize_file_level_aggregates(file_path)
    def get_field_interpretation(field, file_level_detection_aggregates, file_level_aggregates)
    def update_field_level_aggregates_on_processed_field_interpretation(field, field_interpretation)
    def update_file_level_aggregates_on_processed_field_interpretation(field, field_interpretation, field_level_aggregates)
```

### 2. BubblesFieldInterpretation (Field-level Interpreter)

Processes individual field using threshold strategies to interpret bubbles.

```python
class BubblesFieldInterpretation(FieldInterpretation):
    def run_interpretation(field, file_level_detection_aggregates, file_level_interpretation_aggregates)
    def get_field_interpretation_string() -> str
    def _extract_detection_result(field, file_level_detection_aggregates) -> BubbleFieldDetectionResult
    def _create_threshold_config(file_level_interpretation_aggregates) -> ThresholdConfig
    def _calculate_threshold(detection_result, file_level_interpretation_aggregates, config) -> ThresholdResult
    def _interpret_bubbles(detection_result)
    def _check_multi_marking()
    def _calculate_confidence_metrics(detection_result, file_level_aggregates)
```

### 3. BubbleInterpretation (Single Bubble Result)

Represents interpretation of a single bubble.

```python
class BubbleInterpretation:
    bubble_mean: BubbleMeanValue
    threshold: float
    mean_value: float
    is_attempted: bool  # mean_value < threshold
    bubble_value: str  # "A", "B", "C", etc.
    item_reference: BubblesScanBox
```

---

## Main Flow: File-level Interpretation

### Step 1: Initialize File-level Aggregates

**Purpose**: Calculate global thresholds from detection results across all fields.

```python
def initialize_file_level_aggregates(file_path):
    # 1. Get all bubble means from repository
    all_bubble_means = repository.get_all_bubble_means_for_current_file()

    # 2. Calculate outlier deviation threshold (for scan quality)
    all_outlier_deviations = [
        field_result.std_deviation
        for field_result in repository.get_all_bubble_fields_for_current_file().values()
    ]
    outlier_deviation_threshold = get_outlier_deviation_threshold(all_outlier_deviations)

    # 3. Calculate global fallback threshold
    file_level_fallback_threshold, global_max_jump = get_fallback_threshold(all_bubble_means)

    # 4. Store aggregates
    file_level_aggregates = {
        "file_level_fallback_threshold": file_level_fallback_threshold,
        "global_max_jump": global_max_jump,
        "outlier_deviation_threshold_for_file": outlier_deviation_threshold,
        "field_label_wise_local_thresholds": {},
        "all_fields_local_thresholds": NumberAggregate(),
        "field_wise_confidence_metrics": {}
    }
```

**Key Calculations**:

#### Outlier Deviation Threshold

Uses `GlobalThresholdStrategy` with standard deviations:

```python
def get_outlier_deviation_threshold(all_outlier_deviations):
    config = ThresholdConfig(
        min_jump=config.min_jump_std,  # Default: 15
        default_threshold=config.global_page_threshold_std,  # Default: 10
    )

    strategy = GlobalThresholdStrategy()
    result = strategy.calculate_threshold(all_outlier_deviations, config)
    return result.threshold_value
```

#### Global Fallback Threshold

Uses `GlobalThresholdStrategy` with all bubble means:

```python
def get_fallback_threshold(field_wise_means_and_refs):
    config = ThresholdConfig(
        min_jump=config.min_jump,  # Default: 25
        default_threshold=config.global_page_threshold,  # Default: 200
    )

    # Extract mean values from BubbleMeanValue objects
    bubble_values = [item.mean_value for item in field_wise_means_and_refs]

    strategy = GlobalThresholdStrategy()
    result = strategy.calculate_threshold(bubble_values, config)

    return result.threshold_value, result.max_jump
```

**Special Case**: If `file_level_fallback_threshold == 255`, likely a xeroxed OMR (all bubbles very dark).

---

## Main Flow: Field-level Interpretation

### Step 1: Extract Detection Result

**Purpose**: Retrieve bubble detection data from detection pass.

```python
def _extract_detection_result(field, file_level_detection_aggregates):
    field_label = field.field_label

    # Get from new typed format (BubbleFieldDetectionResult)
    bubble_fields = file_level_detection_aggregates["bubble_fields"]

    if field_label not in bubble_fields:
        raise ValueError(f"No detection result for field '{field_label}'")

    return bubble_fields[field_label]  # BubbleFieldDetectionResult
```

**Data Structure** (`BubbleFieldDetectionResult`):

```python
@dataclass
class BubbleFieldDetectionResult:
    field_id: str
    field_label: str
    bubble_means: list[BubbleMeanValue]  # List of bubble intensities

    # Auto-calculated properties
    @property std_deviation: float
    @property scan_quality: ScanQuality  # EXCELLENT/GOOD/ACCEPTABLE/POOR
    @property mean_values: list[float]
    @property max_jump: float
```

---

### Step 2: Create Threshold Configuration

**Purpose**: Build configuration for local threshold strategy.

```python
def _create_threshold_config(file_level_interpretation_aggregates):
    config = tuning_config

    return ThresholdConfig(
        # Jump-based thresholds
        min_jump=config.thresholding.min_jump,  # 25
        jump_delta=config.thresholding.jump_delta,  # 30
        min_gap_two_bubbles=config.thresholding.min_gap_two_bubbles,  # 30

        # Confidence thresholds
        min_jump_surplus_for_global_fallback=config.thresholding.min_jump_surplus_for_global_fallback,  # 5
        confident_jump_surplus_for_disparity=config.thresholding.confident_jump_surplus_for_disparity,  # 25

        # Global fallback
        global_threshold_margin=config.thresholding.global_threshold_margin,  # 10
        outlier_deviation_threshold=file_level_interpretation_aggregates.get(
            "outlier_deviation_threshold_for_file", 5.0
        ),
        default_threshold=config.thresholding.global_page_threshold,  # 200
    )
```

---

### Step 3: Calculate Threshold (Local with Global Fallback)

**Purpose**: Calculate optimal threshold for this field's bubbles.

```python
def _calculate_threshold(detection_result, file_level_interpretation_aggregates, config):
    # Get global fallback threshold
    global_fallback = file_level_interpretation_aggregates.get(
        "file_level_fallback_threshold", config.default_threshold
    )

    # Use local strategy with global fallback
    strategy = LocalThresholdStrategy(global_fallback=global_fallback)

    # Calculate threshold from bubble means
    threshold_result = strategy.calculate_threshold(
        detection_result.mean_values,  # [180.5, 185.2, 210.3, 240.1, ...]
        config,
    )

    return threshold_result  # ThresholdResult object
```

**Threshold Result**:

```python
@dataclass
class ThresholdResult:
    threshold_value: float  # e.g., 220.5
    confidence: float  # 0.0 to 1.0
    max_jump: float  # Maximum intensity jump
    method_used: str  # "local_max_jump", "local_low_confidence_global_fallback", etc.
    fallback_used: bool  # True if global fallback was used
    metadata: dict  # Additional info
```

**Local Threshold Strategy Logic**:

#### Case 1: Single Bubble (0-1 bubbles)

```python
if len(bubble_mean_values) < 2:
    return ThresholdResult(
        threshold_value=global_fallback,
        confidence=0.0,
        method_used="local_single_bubble_fallback",
        fallback_used=True
    )
```

#### Case 2: Two Bubbles

```python
if len(sorted_values) == 2:
    gap = sorted_values[1] - sorted_values[0]

    if gap < config.min_gap_two_bubbles:  # 30
        # Too close - use global fallback
        return ThresholdResult(
            threshold_value=global_fallback,
            confidence=0.3,
            method_used="local_two_bubbles_small_gap_fallback",
            fallback_used=True
        )

    # Sufficient gap - use mean
    return ThresholdResult(
        threshold_value=np.mean(sorted_values),
        confidence=0.7,
        method_used="local_two_bubbles_mean",
        fallback_used=False
    )
```

#### Case 3: Three or More Bubbles (Normal Case)

```python
# Find largest jump between consecutive bubbles
max_jump = 0.0
threshold = global_fallback

for i in range(1, len(sorted_values) - 1):
    jump = sorted_values[i + 1] - sorted_values[i - 1]
    if jump > max_jump:
        max_jump = jump
        threshold = sorted_values[i - 1] + jump / 2

# Check confidence
confident_jump = config.min_jump + config.min_jump_surplus_for_global_fallback  # 25 + 5 = 30

if max_jump < confident_jump:
    # Low confidence - use global fallback
    return ThresholdResult(
        threshold_value=global_fallback,
        confidence=0.4,
        method_used="local_low_confidence_global_fallback",
        fallback_used=True,
        metadata={"local_threshold": threshold}
    )

# High confidence - use local threshold
confidence = min(1.0, max_jump / (confident_jump * 2))

return ThresholdResult(
    threshold_value=threshold,
    confidence=confidence,
    method_used="local_max_jump",
    fallback_used=False
)
```

**Visual Example**:

```
Bubble Intensities (sorted):
[150, 155, 160, 225, 230, 235]
       ↑              ↑
    Marked      Unmarked

Jumps: [5, 5, 65, 5, 5]
Max Jump: 65 (between 160 and 225)

Threshold: 160 + 65/2 = 192.5
Confidence: min(1.0, 65 / 60) = 1.0  (very confident!)
```

---

### Step 4: Interpret Bubbles

**Purpose**: Apply threshold to classify each bubble as marked/unmarked.

```python
def _interpret_bubbles(detection_result):
    self.bubble_interpretations = [
        BubbleInterpretation(bubble_mean, self.local_threshold_for_field)
        for bubble_mean in detection_result.bubble_means
    ]
```

**BubbleInterpretation Logic**:

```python
class BubbleInterpretation:
    def __init__(self, bubble_mean, threshold):
        self.bubble_mean = bubble_mean
        self.threshold = threshold
        self.mean_value = bubble_mean.mean_value

        # Key decision: is_attempted
        self.is_attempted = bubble_mean.mean_value < threshold

        # Extract bubble value (e.g., "A", "B", "C")
        self.bubble_value = bubble_mean.unit_bubble.bubble_value or ""
```

**Example**:

```python
# Field: Q1, Threshold: 192.5
bubble_means = [
    BubbleMeanValue(mean_value=155, unit_bubble=<A>),  # ✓ Marked (155 < 192.5)
    BubbleMeanValue(mean_value=230, unit_bubble=<B>),  # ✗ Unmarked (230 >= 192.5)
    BubbleMeanValue(mean_value=225, unit_bubble=<C>),  # ✗ Unmarked
    BubbleMeanValue(mean_value=235, unit_bubble=<D>),  # ✗ Unmarked
]

bubble_interpretations = [
    BubbleInterpretation(is_attempted=True, bubble_value="A"),
    BubbleInterpretation(is_attempted=False, bubble_value="B"),
    BubbleInterpretation(is_attempted=False, bubble_value="C"),
    BubbleInterpretation(is_attempted=False, bubble_value="D"),
]
```

---

### Step 5: Check Multi-marking

**Purpose**: Detect if multiple bubbles are marked (typically an error).

```python
def _check_multi_marking():
    marked_count = sum(
        1 for interp in bubble_interpretations if interp.is_attempted
    )

    is_multi_marked = marked_count > 1

    if is_multi_marked:
        logger.warning(
            f"Multi-marking detected in field: {field.field_label}, "
            f"marked bubbles: {marked_count}"
        )
```

**Multi-marking Scenarios**:

1. **Multiple Marked** (marked_count > 1): Student marked multiple answers (error)
2. **None Marked** (marked_count == 0): Student left question blank
3. **All Marked** (marked_count == total_bubbles): Likely scanning issue (treated as unmarked)

---

### Step 6: Generate Field Interpretation String

**Purpose**: Convert bubble interpretations to final string value.

```python
def get_field_interpretation_string():
    marked_bubbles = [
        interp.bubble_value
        for interp in bubble_interpretations
        if interp.is_attempted
    ]

    # Case 1: No bubbles marked
    if len(marked_bubbles) == 0:
        return empty_value  # ""

    # Case 2: ALL bubbles marked (scanning issue)
    total_bubbles = len(bubble_interpretations)
    if len(marked_bubbles) == total_bubbles:
        return empty_value  # Treat as unmarked

    # Case 3: Normal case (concatenate marked values)
    return "".join(marked_bubbles)  # "A" or "AB" (multi-mark) or "ACD", etc.
```

**Examples**:

| Marked Bubbles | Total Bubbles | Result | Reason |
|----------------|---------------|--------|--------|
| [] | 4 | "" | No marks |
| ["A"] | 4 | "A" | Normal |
| ["A", "B"] | 4 | "AB" | Multi-mark |
| ["A", "B", "C", "D"] | 4 | "" | All marked (scanning issue) |

---

### Step 7: Calculate Confidence Metrics (Optional)

**Purpose**: Generate detailed confidence metrics for ML training and validation.

**Enabled when**: `tuning_config.outputs.show_confidence_metrics = True`

```python
def _calculate_confidence_metrics(detection_result, file_level_aggregates):
    global_threshold = file_level_aggregates.get(
        "file_level_fallback_threshold", threshold_result.threshold_value
    )

    # 1. Identify disparity bubbles (different between local and global)
    disparity_bubbles = []
    for bubble_mean in detection_result.bubble_means:
        local_marked = bubble_mean.mean_value < local_threshold_for_field
        global_marked = bubble_mean.mean_value < global_threshold

        if local_marked != global_marked:
            disparity_bubbles.append(bubble_mean)

    # 2. Calculate overall confidence score (0.0 to 1.0)
    confidence_score = _calculate_overall_confidence_score(
        detection_result, disparity_bubbles
    )

    # 3. Build confidence metrics
    field_level_confidence_metrics = {
        "local_threshold": local_threshold_for_field,
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
```

**Overall Confidence Score Calculation**:

Combines 5 factors into a single score (0.0 to 1.0):

```python
def _calculate_overall_confidence_score(detection_result, disparity_bubbles):
    # Factor 1: Threshold confidence from strategy (0.0-1.0)
    threshold_confidence = threshold_result.confidence  # 35% weight

    # Factor 2: Margin from threshold (how clearly marked/unmarked)
    marked_bubbles = [b for b in detection_result.bubble_means if b.mean_value < local_threshold]

    if marked_bubbles:
        avg_margin = sum(local_threshold - b.mean_value for b in marked_bubbles) / len(marked_bubbles)
        margin_confidence = min(1.0, avg_margin / 50.0)  # 25% weight
    else:
        avg_distance = sum(b.mean_value - local_threshold for b in detection_result.bubble_means) / len(detection_result.bubble_means)
        margin_confidence = min(1.0, avg_distance / 50.0)

    # Factor 3: Multi-mark penalty
    if marked_count > 1:
        multi_mark_penalty = 0.3  # Reduce by 30%
    elif marked_count == 0:
        multi_mark_penalty = 0.1  # Slight penalty
    else:
        multi_mark_penalty = 0.0  # Ideal

    # Factor 4: Disparity penalty
    disparity_ratio = len(disparity_bubbles) / len(detection_result.bubble_means)
    disparity_penalty = disparity_ratio * 0.4  # Up to 40% penalty

    # Factor 5: Scan quality (20% weight)
    scan_quality_factor = {
        "EXCELLENT": 1.0,
        "GOOD": 0.9,
        "MODERATE": 0.7,
        "POOR": 0.5,
    }[detection_result.scan_quality.value]

    # Combine factors (weighted average)
    confidence_score = (
        threshold_confidence * 0.35 +
        margin_confidence * 0.25 +
        scan_quality_factor * 0.20
    ) * (1.0 - multi_mark_penalty - disparity_penalty)

    return max(0.0, min(1.0, confidence_score))
```

**Confidence Score Example**:

```
Scenario: Single bubble marked clearly, high threshold confidence
- Threshold confidence: 1.0 (35% → 0.35)
- Margin confidence: 0.8 (25% → 0.20)
- Scan quality: EXCELLENT (20% → 0.20)
- Multi-mark penalty: 0.0
- Disparity penalty: 0.0
Total: (0.35 + 0.20 + 0.20) * 1.0 = 0.75 (high confidence)

Scenario: Multi-marked, low threshold confidence
- Threshold confidence: 0.4 (35% → 0.14)
- Margin confidence: 0.5 (25% → 0.125)
- Scan quality: GOOD (20% → 0.18)
- Multi-mark penalty: 0.3
- Disparity penalty: 0.2
Total: (0.14 + 0.125 + 0.18) * (1.0 - 0.3 - 0.2) = 0.445 * 0.5 = 0.22 (low confidence)
```

---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│ File-level Initialization                                       │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 1. Get all bubble means from detection results             │ │
│ │ 2. Calculate outlier deviation threshold (std devs)        │ │
│ │ 3. Calculate global fallback threshold (all bubble means)  │ │
│ │ 4. Store in file_level_aggregates                          │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Field-level Interpretation (per field)                          │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Step 1: Extract Detection Result                           │ │
│ │   - BubbleFieldDetectionResult (typed model)               │ │
│ │   - Contains bubble_means, std_deviation, scan_quality     │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                           ↓                                     │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Step 2: Create Threshold Configuration                     │ │
│ │   - ThresholdConfig with tuning parameters                 │ │
│ │   - Includes global fallback, min jumps, etc.              │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                           ↓                                     │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Step 3: Calculate Threshold                                │ │
│ │   - LocalThresholdStrategy with global fallback            │ │
│ │   - Returns ThresholdResult (value, confidence, method)    │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                           ↓                                     │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Step 4: Interpret Bubbles                                  │ │
│ │   - Create BubbleInterpretation for each bubble            │ │
│ │   - is_attempted = mean_value < threshold                  │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                           ↓                                     │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Step 5: Check Multi-marking                                │ │
│ │   - Count marked bubbles                                   │ │
│ │   - Set is_multi_marked flag                               │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                           ↓                                     │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Step 6: Generate Interpretation String                     │ │
│ │   - Concatenate marked bubble values                       │ │
│ │   - Handle edge cases (none marked, all marked)            │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                           ↓                                     │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Step 7: Calculate Confidence Metrics (optional)            │ │
│ │   - Overall confidence score (0.0 to 1.0)                  │ │
│ │   - Disparity analysis, scan quality, multi-mark penalty   │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Update Aggregates                                               │
│ - field_level_aggregates: bubble_interpretations, threshold    │
│ - file_level_aggregates: local thresholds, multi-mark flags    │
│ - directory_level_aggregates: stats across all files           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Integration Points

### 1. Input: Detection Results

**From**: `BubblesThresholdDetectionPass` → `DetectionRepository`

```python
# Detection results stored in repository
repository.store_bubble_field_result(
    field_label,
    BubbleFieldDetectionResult(
        field_id=field.id,
        field_label=field.field_label,
        bubble_means=[...],  # List of BubbleMeanValue
    )
)

# Retrieved by interpretation pass
detection_result = repository.get_bubble_field_result(field_label)
```

### 2. Output: Interpretation Results

**To**: Template processing, CSV export, visualization

```python
# Field interpretation
field_interpretation = BubblesFieldInterpretation(...)
field_interpretation.run_interpretation(...)

# Access results
detected_string = field_interpretation.get_field_interpretation_string()  # "A" or "AB"
is_multi_marked = field_interpretation.is_multi_marked  # True/False
local_threshold = field_interpretation.local_threshold_for_field  # 192.5
confidence_metrics = field_interpretation.get_field_level_confidence_metrics()

# Stored in aggregates
file_level_aggregates["field_id_to_interpretation"][field.id] = field_interpretation
```

### 3. Visualization: Drawing

**To**: `BubblesFieldInterpretationDrawing`

```python
drawing = field_interpretation.get_drawing_instance()
drawing.draw_field_interpretation(marked_image, image_type, evaluation_meta, evaluation_config)

# Drawing logic:
# - Marked bubbles: Filled gray box with bubble value text
# - Unmarked bubbles: Hollow box outline
# - With evaluation: Verdicts (correct/incorrect) with colors/symbols
```

---

## Browser Migration

### TypeScript Implementation

```typescript
// 1. BubbleInterpretation
interface BubbleInterpretation {
    bubbleMean: BubbleMeanValue;
    threshold: number;
    meanValue: number;
    isAttempted: boolean;  // meanValue < threshold
    bubbleValue: string;
    itemReference: BubblesScanBox;
}

function createBubbleInterpretation(
    bubbleMean: BubbleMeanValue,
    threshold: number
): BubbleInterpretation {
    return {
        bubbleMean,
        threshold,
        meanValue: bubbleMean.meanValue,
        isAttempted: bubbleMean.meanValue < threshold,
        bubbleValue: bubbleMean.unitBubble.bubbleValue || "",
        itemReference: bubbleMean.unitBubble,
    };
}

// 2. BubblesFieldInterpretation
class BubblesFieldInterpretation {
    bubbleInterpretations: BubbleInterpretation[] = [];
    isMultiMarked = false;
    localThresholdForField = 0.0;
    thresholdResult: ThresholdResult | null = null;

    async runInterpretation(
        field: Field,
        fileLevelDetectionAggregates: any,
        fileLevelInterpretationAggregates: any
    ): Promise<void> {
        // Step 1: Extract detection result
        const detectionResult = this.extractDetectionResult(field, fileLevelDetectionAggregates);

        if (!detectionResult) {
            throw new Error(`No detection result for field '${field.fieldLabel}'`);
        }

        // Step 2: Create threshold config
        const thresholdConfig = this.createThresholdConfig(fileLevelInterpretationAggregates);

        // Step 3: Calculate threshold
        this.thresholdResult = this.calculateThreshold(
            detectionResult,
            fileLevelInterpretationAggregates,
            thresholdConfig
        );

        this.localThresholdForField = this.thresholdResult.thresholdValue;

        // Step 4: Interpret bubbles
        this.interpretBubbles(detectionResult);

        // Step 5: Check multi-marking
        this.checkMultiMarking();

        // Step 6: Calculate confidence metrics (if enabled)
        if (this.tuningConfig.outputs.showConfidenceMetrics) {
            this.calculateConfidenceMetrics(detectionResult, fileLevelInterpretationAggregates);
        }
    }

    getFieldInterpretationString(): string {
        const markedBubbles = this.bubbleInterpretations
            .filter(interp => interp.isAttempted)
            .map(interp => interp.bubbleValue);

        // No bubbles marked
        if (markedBubbles.length === 0) {
            return "";
        }

        // ALL bubbles marked (scanning issue)
        if (markedBubbles.length === this.bubbleInterpretations.length) {
            return "";
        }

        // Normal case
        return markedBubbles.join("");
    }

    private interpretBubbles(detectionResult: BubbleFieldDetectionResult): void {
        this.bubbleInterpretations = detectionResult.bubbleMeans.map(bubbleMean =>
            createBubbleInterpretation(bubbleMean, this.localThresholdForField)
        );
    }

    private checkMultiMarking(): void {
        const markedCount = this.bubbleInterpretations.filter(
            interp => interp.isAttempted
        ).length;

        this.isMultiMarked = markedCount > 1;

        if (this.isMultiMarked) {
            console.warn(
                `Multi-marking detected in field: ${this.field.fieldLabel}, ` +
                `marked bubbles: ${markedCount}`
            );
        }
    }
}

// 3. LocalThresholdStrategy
class LocalThresholdStrategy implements ThresholdStrategy {
    constructor(private globalFallback?: number) {}

    calculateThreshold(
        bubbleMeanValues: number[],
        config: ThresholdConfig
    ): ThresholdResult {
        const fallbackThreshold = this.globalFallback ?? config.defaultThreshold;

        // Base case: empty or single bubble
        if (bubbleMeanValues.length < 2) {
            return {
                thresholdValue: fallbackThreshold,
                confidence: 0.0,
                maxJump: 0.0,
                methodUsed: "local_single_bubble_fallback",
                fallbackUsed: true,
            };
        }

        const sortedValues = [...bubbleMeanValues].sort((a, b) => a - b);

        // Special case: exactly 2 bubbles
        if (sortedValues.length === 2) {
            const gap = sortedValues[1] - sortedValues[0];

            if (gap < config.minGapTwoBubbles) {
                return {
                    thresholdValue: fallbackThreshold,
                    confidence: 0.3,
                    maxJump: gap,
                    methodUsed: "local_two_bubbles_small_gap_fallback",
                    fallbackUsed: true,
                };
            }

            return {
                thresholdValue: (sortedValues[0] + sortedValues[1]) / 2,
                confidence: 0.7,
                maxJump: gap,
                methodUsed: "local_two_bubbles_mean",
                fallbackUsed: false,
            };
        }

        // 3+ bubbles: find largest jump
        let maxJump = 0.0;
        let threshold = fallbackThreshold;

        for (let i = 1; i < sortedValues.length - 1; i++) {
            const jump = sortedValues[i + 1] - sortedValues[i - 1];
            if (jump > maxJump) {
                maxJump = jump;
                threshold = sortedValues[i - 1] + jump / 2;
            }
        }

        // Check if jump is confident
        const confidentJump = config.minJump + config.minJumpSurplusForGlobalFallback;

        if (maxJump < confidentJump) {
            // Low confidence - use global fallback
            return {
                thresholdValue: fallbackThreshold,
                confidence: 0.4,
                maxJump,
                methodUsed: "local_low_confidence_global_fallback",
                fallbackUsed: true,
                metadata: { localThreshold: threshold },
            };
        }

        // High confidence
        const confidence = Math.min(1.0, maxJump / (confidentJump * 2));

        return {
            thresholdValue: threshold,
            confidence,
            maxJump,
            methodUsed: "local_max_jump",
            fallbackUsed: false,
            metadata: { numBubbles: bubbleMeanValues.length },
        };
    }
}
```

### Performance Considerations

1. **Minimal Computation**: Threshold calculation is O(n log n) due to sorting
2. **No Heavy Dependencies**: Pure math operations (sorting, averaging)
3. **Memory Efficient**: Small objects (BubbleInterpretation, ThresholdResult)
4. **Async Ready**: Can wrap in async for Web Worker processing

### Browser-Specific Adaptations

1. **Console Logging**: Use `console.warn()` for multi-mark warnings
2. **Number Precision**: Use `Number.toFixed()` for display precision
3. **Array Operations**: Native JavaScript array methods (map, filter, reduce)
4. **Math Operations**: Native `Math.min()`, `Math.max()`, `Math.round()`

---

## See Also

- **Detection Pass**: `detection-pass/flows.md` - Bubble extraction and intensity calculation
- **Threshold Strategies**: `../../threshold/concept.md` - Global, Local, Adaptive strategies
- **Drawing**: `drawing/flows.md` - Visualization of interpretation results
- **Interpretation Base**: `src/processors/detection/base/interpretation.py` - Base class architecture
