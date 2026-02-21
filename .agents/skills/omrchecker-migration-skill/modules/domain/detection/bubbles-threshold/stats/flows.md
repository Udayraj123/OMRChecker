# Bubble Detection Statistics Flows

**Module**: Domain - Detection - Bubbles Threshold - Statistics
**Python Reference**: `src/processors/detection/bubbles_threshold/stats.py`, `src/processors/detection/models/detection_results.py`
**Last Updated**: 2026-02-21

---

## Overview

The Bubble Detection Statistics module provides **statistical analysis** of bubble intensity values to assess scan quality and calculate metrics for threshold determination. It uses **mean values, standard deviation, percentiles, and jump detection** to evaluate whether bubbles are clearly distinguishable.

**Key Purpose**: Transform raw bubble intensities into actionable quality metrics that inform threshold calculation and confidence scoring.

---

## Core Components

### 1. MeanValueItem Class

**File**: `src/processors/detection/bubbles_threshold/stats.py`

```python
@functools.total_ordering
class MeanValueItem(Generic[ItemReferenceT]):
    """Generic wrapper for mean value with item reference.

    Enables sorting and comparison of items by their mean values.
    Used for maintaining reference to original bubble while sorting.
    """

    def __init__(self, mean_value: float, item_reference: ItemReferenceT) -> None:
        self.mean_value = mean_value
        self.item_reference = item_reference

    def __str__(self) -> str:
        return f"{self.item_reference} : {round(self.mean_value, 2)}"

    def validate_and_extract_value(self, other) -> float | int:
        """Extract comparable value from other item."""
        if hasattr(other, "mean_value") and hasattr(other, "item_reference"):
            return other.mean_value
        if isinstance(other, (float, int)):
            return other
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        value = self.validate_and_extract_value(other)
        return self.mean_value == value

    def __lt__(self, other) -> bool:
        value = self.validate_and_extract_value(other)
        return self.mean_value < value
```

**Purpose**: Generic container that allows sorting while preserving original item reference.

**Key Features**:
- **Type-safe generics**: `ItemReferenceT` can be any bubble type
- **Total ordering**: Implements `__eq__` and `__lt__` with `@functools.total_ordering`
- **Flexible comparison**: Can compare with other `MeanValueItem` or raw numbers
- **Debugging support**: String representation shows both value and reference

---

### 2. BubbleMeanValue Class

**File**: `src/processors/detection/models/detection_results.py`

```python
@dataclass
class BubbleMeanValue:
    """Single bubble mean intensity value with metadata."""

    mean_value: float
    unit_bubble: Any  # BubblesScanBox - avoiding circular import
    position: tuple[int, int] = (0, 0)

    def __lt__(self, other: "BubbleMeanValue") -> bool:
        """Enable sorting by mean value."""
        return self.mean_value < other.mean_value

    def __str__(self) -> str:
        """String representation for debugging."""
        return f"BubbleMean({self.mean_value:.1f})"
```

**Purpose**: Typed container for individual bubble measurements.

**Fields**:
- `mean_value`: Grayscale intensity (0-255, lower = darker)
- `unit_bubble`: Reference to original `BubblesScanBox`
- `position`: (x, y) coordinates in image

---

### 3. BubbleFieldDetectionResult Class

**File**: `src/processors/detection/models/detection_results.py`

The main statistics container with **auto-calculated properties**.

```python
@dataclass
class BubbleFieldDetectionResult:
    """Typed result for bubble field detection.

    Includes auto-calculated statistical properties.
    """

    field_id: str
    field_label: str
    bubble_means: list[BubbleMeanValue]
    timestamp: datetime = field(default_factory=datetime.now)
```

---

## Statistical Property Flows

### Flow 1: Standard Deviation Calculation

**Property**: `std_deviation`

```
START: BubbleFieldDetectionResult.std_deviation
│
├─► STEP 1: Check if bubbles exist
│   │
│   if not self.bubble_means:
│       return 0.0  # No data
│
├─► STEP 2: Extract mean values
│   │
│   values = [bm.mean_value for bm in self.bubble_means]
│   │
│   │ Example:
│   │ bubble_means = [
│   │     BubbleMeanValue(45.0, ...),
│   │     BubbleMeanValue(50.0, ...),
│   │     BubbleMeanValue(52.0, ...),
│   │     BubbleMeanValue(140.0, ...),
│   │     BubbleMeanValue(145.0, ...),
│   │     BubbleMeanValue(150.0, ...)
│   │ ]
│   │
│   │ values = [45.0, 50.0, 52.0, 140.0, 145.0, 150.0]
│
├─► STEP 3: Calculate standard deviation
│   │
│   std = np.std(values)
│   │
│   │ Using NumPy's standard deviation:
│   │ - Mean: (45+50+52+140+145+150)/6 = 97.0
│   │ - Variance: mean of squared differences
│   │ - Std: sqrt(variance)
│   │
│   │ std ≈ 48.5
│
└─► STEP 4: Return as float
    │
    return float(np.std(values))
    │
    │ Returns: 48.5

END
```

**Formula**:
```
std = √(Σ(xi - μ)² / n)

where:
- xi = individual bubble mean value
- μ = mean of all bubble values
- n = number of bubbles
```

**Interpretation**:
```
High std (> 50):   Clear separation between filled/empty bubbles
Medium std (30-50): Good separation, reliable detection
Low std (15-30):   Marginal separation, low confidence
Very low std (< 15): Poor separation, unreliable detection
```

---

### Flow 2: Scan Quality Assessment

**Property**: `scan_quality`

```
START: BubbleFieldDetectionResult.scan_quality
│
├─► STEP 1: Get standard deviation
│   │
│   std = self.std_deviation
│   │
│   │ Example: std = 48.5
│
├─► STEP 2: Apply quality thresholds
│   │
│   if std > 50:
│       return ScanQuality.EXCELLENT
│   │
│   elif std > 30:
│       return ScanQuality.GOOD
│   │
│   elif std > 15:
│       return ScanQuality.ACCEPTABLE
│   │
│   else:
│       return ScanQuality.POOR
│
└─► STEP 3: Return enum value
    │
    │ For std=48.5:
    │ 48.5 > 30 → ScanQuality.GOOD

END
```

**Quality Levels**:

| Quality | Std Range | Meaning | Confidence |
|---------|-----------|---------|------------|
| EXCELLENT | > 50 | Very clear contrast | High |
| GOOD | 30-50 | Acceptable contrast | Medium-High |
| ACCEPTABLE | 15-30 | Marginal contrast | Low-Medium |
| POOR | < 15 | Insufficient contrast | Very Low |

**Usage**:
```python
result = BubbleFieldDetectionResult(...)

if result.scan_quality == ScanQuality.EXCELLENT:
    # Use local threshold with confidence
    use_local_threshold()
elif result.scan_quality == ScanQuality.GOOD:
    # Use local threshold, may fallback to global
    use_local_with_fallback()
elif result.scan_quality == ScanQuality.ACCEPTABLE:
    # Prefer global threshold
    use_global_threshold()
else:  # POOR
    # Use global or ML detection
    use_fallback_detection()
```

---

### Flow 3: Reliability Check

**Property**: `is_reliable`

```
START: BubbleFieldDetectionResult.is_reliable
│
├─► STEP 1: Get scan quality
│   │
│   quality = self.scan_quality
│
├─► STEP 2: Check if reliable
│   │
│   return quality in [ScanQuality.EXCELLENT, ScanQuality.GOOD]
│   │
│   │ Reliable if std > 30 (EXCELLENT or GOOD)
│   │ Not reliable if std <= 30 (ACCEPTABLE or POOR)
│
└─► STEP 3: Return boolean
    │
    │ Example: std=48.5 → GOOD → True

END
```

**Usage**:
```python
if result.is_reliable:
    # Proceed with normal threshold detection
    threshold = local_strategy.calculate_threshold(...)
else:
    # Use fallback or request manual review
    threshold = global_strategy.calculate_threshold(...)
    flag_for_review(field)
```

---

### Flow 4: Jump Detection

**Property**: `jumps`

```
START: BubbleFieldDetectionResult.jumps
│
├─► STEP 1: Get sorted bubble means
│   │
│   sorted_means = self.sorted_bubble_means
│   │
│   │ Example:
│   │ Original: [145, 45, 50, 140, 52, 150]
│   │ Sorted:   [45, 50, 52, 140, 145, 150]
│
├─► STEP 2: Check minimum count
│   │
│   if len(sorted_means) < 2:
│       return []  # Need at least 2 bubbles
│
├─► STEP 3: Calculate jumps between consecutive bubbles
│   │
│   jumps_list = []
│   for i in range(1, len(sorted_means)):
│       jump = sorted_means[i].mean_value - sorted_means[i-1].mean_value
│       jumps_list.append((round(jump, 2), sorted_means[i-1]))
│   │
│   │ ITERATION EXAMPLES:
│   │
│   │ i=1: jump = 50.0 - 45.0 = 5.0
│   │      jumps_list.append((5.0, BubbleMean(45.0)))
│   │
│   │ i=2: jump = 52.0 - 50.0 = 2.0
│   │      jumps_list.append((2.0, BubbleMean(50.0)))
│   │
│   │ i=3: jump = 140.0 - 52.0 = 88.0  ✓ LARGE GAP!
│   │      jumps_list.append((88.0, BubbleMean(52.0)))
│   │
│   │ i=4: jump = 145.0 - 140.0 = 5.0
│   │      jumps_list.append((5.0, BubbleMean(140.0)))
│   │
│   │ i=5: jump = 150.0 - 145.0 = 5.0
│   │      jumps_list.append((5.0, BubbleMean(145.0)))
│
└─► STEP 4: Return jump list
    │
    return jumps_list
    │
    │ Returns: [
    │     (5.0, BubbleMean(45.0)),
    │     (2.0, BubbleMean(50.0)),
    │     (88.0, BubbleMean(52.0)),  ← Maximum jump
    │     (5.0, BubbleMean(140.0)),
    │     (5.0, BubbleMean(145.0))
    │ ]

END
```

**Jump Tuple Format**:
```python
(jump_size: float, bubble_before_jump: BubbleMeanValue)
```

**Purpose**: Identify gaps between filled and empty bubble groups.

**Visualization**:
```
Bubble Values: [45, 50, 52, 140, 145, 150]

45  ■  ┐
50  ■  │ Small jumps (2-5)
52  ■  ┘
       ─── GAP: 88.0 ─── ← Target for threshold
140    ■  ┐
145    ■  │ Small jumps (5)
150    ■  ┘
```

---

### Flow 5: Maximum Jump Detection

**Property**: `max_jump`

```
START: BubbleFieldDetectionResult.max_jump
│
├─► STEP 1: Check bubble count
│   │
│   if len(self.bubble_means) < 2:
│       return 0.0  # No jumps possible
│
├─► STEP 2: Get jumps
│   │
│   jumps = self.jumps
│   │
│   │ jumps = [(5.0, ...), (2.0, ...), (88.0, ...), (5.0, ...), (5.0, ...)]
│
├─► STEP 3: Find maximum
│   │
│   if not jumps:
│       return 0.0
│   │
│   max_jump = max(jump for jump, _ in jumps)
│   │
│   │ Extract just jump values: [5.0, 2.0, 88.0, 5.0, 5.0]
│   │ Maximum: 88.0
│
└─► STEP 4: Return maximum
    │
    return max_jump
    │
    │ Returns: 88.0

END
```

**Usage**:
```python
result = BubbleFieldDetectionResult(...)

if result.max_jump >= 30:  # Significant gap
    # Good separation, high confidence
    confidence = HIGH
elif result.max_jump >= 15:  # Moderate gap
    # Acceptable separation, medium confidence
    confidence = MEDIUM
else:  # Small gap
    # Poor separation, low confidence
    confidence = LOW
```

---

### Flow 6: Min/Max Value Tracking

**Properties**: `min_mean`, `max_mean`

```
START: BubbleFieldDetectionResult.min_mean / max_mean
│
├─► min_mean Property:
│   │
│   if not self.bubble_means:
│       return 0.0  # Default minimum
│   │
│   return min(self.mean_values)
│   │
│   │ Example: min([45.0, 50.0, 52.0, 140.0, 145.0, 150.0]) = 45.0
│
├─► max_mean Property:
│   │
│   if not self.bubble_means:
│       return 255.0  # Default maximum (full grayscale range)
│   │
│   return max(self.mean_values)
│   │
│   │ Example: max([45.0, 50.0, 52.0, 140.0, 145.0, 150.0]) = 150.0
│
└─► Usage:
    │
    value_range = result.max_mean - result.min_mean
    │
    │ Example: 150.0 - 45.0 = 105.0 (wide range, good contrast)

END
```

**Interpretation**:
```
Wide range (> 100):   Good contrast, clear marking
Medium range (50-100): Acceptable contrast
Narrow range (< 50):   Poor contrast, difficult detection
```

---

### Flow 7: Sorted Values Access

**Properties**: `sorted_bubble_means`, `sorted_mean_values`

```
START: BubbleFieldDetectionResult.sorted_bubble_means
│
├─► STEP 1: Sort by mean value
│   │
│   return sorted(self.bubble_means)
│   │
│   │ Uses BubbleMeanValue.__lt__ for comparison
│   │ Original: [BubbleMean(145), BubbleMean(45), BubbleMean(50)]
│   │ Sorted:   [BubbleMean(45), BubbleMean(50), BubbleMean(145)]
│
└─► STEP 2: Extract just values (for sorted_mean_values)
    │
    return [bm.mean_value for bm in self.sorted_bubble_means]
    │
    │ Returns: [45.0, 50.0, 52.0, 140.0, 145.0, 150.0]

END
```

**Usage in Threshold Strategies**:
```python
# Global threshold uses sorted values to find gaps
sorted_values = result.sorted_mean_values
for i in range(1, len(sorted_values)):
    jump = sorted_values[i] - sorted_values[i-1]
    if jump > max_jump:
        max_jump = jump
        threshold = sorted_values[i-1] + jump / 2
```

---

## Complete Statistics Flow Example

### End-to-End Example: Field Detection to Statistics

```
START: Detect bubble field
│
├─► STEP 1: Detection Pass
│   │
│   field = BubbleField(field_label="q1", scan_boxes=[...])
│   gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
│   │
│   bubble_means = []
│   for unit_bubble in field.scan_boxes:
│       x, y = unit_bubble.get_shifted_position()
│       w, h = unit_bubble.bubble_dimensions
│       │
│       # Extract ROI and calculate mean
│       roi = gray_image[y:y+h, x:x+w]
│       mean_value = cv2.mean(roi)[0]
│       │
│       bubble_means.append(BubbleMeanValue(
│           mean_value=mean_value,
│           unit_bubble=unit_bubble,
│           position=(x, y)
│       ))
│   │
│   │ Result: [
│   │     BubbleMeanValue(45.2, bubble_A, (10, 20)),
│   │     BubbleMeanValue(48.7, bubble_B, (50, 20)),
│   │     BubbleMeanValue(51.3, bubble_C, (90, 20)),
│   │     BubbleMeanValue(142.8, bubble_D, (130, 20))
│   │ ]
│
├─► STEP 2: Create Detection Result
│   │
│   result = BubbleFieldDetectionResult(
│       field_id="q1",
│       field_label="q1",
│       bubble_means=bubble_means
│   )
│
├─► STEP 3: Auto-Calculate Statistics (lazy evaluation)
│   │
│   # Standard deviation
│   std = result.std_deviation
│   │   → values = [45.2, 48.7, 51.3, 142.8]
│   │   → np.std(values) = 44.8
│   │
│   # Scan quality
│   quality = result.scan_quality
│   │   → std=44.8 > 30 → ScanQuality.GOOD
│   │
│   # Reliability
│   reliable = result.is_reliable
│   │   → quality in [EXCELLENT, GOOD] → True
│   │
│   # Sorted values
│   sorted_vals = result.sorted_mean_values
│   │   → [45.2, 48.7, 51.3, 142.8]
│   │
│   # Jumps
│   jumps = result.jumps
│   │   → [(3.5, BubbleMean(45.2)),
│   │      (2.6, BubbleMean(48.7)),
│   │      (91.5, BubbleMean(51.3))]  ← Max jump
│   │
│   # Maximum jump
│   max_jump = result.max_jump
│   │   → 91.5
│   │
│   # Value range
│   min_val = result.min_mean  # 45.2
│   max_val = result.max_mean  # 142.8
│   range_val = max_val - min_val  # 97.6
│
└─► STEP 4: Use Statistics for Threshold Calculation
    │
    if result.is_reliable and result.max_jump >= 30:
        # Use local threshold strategy
        threshold = local_strategy.calculate_threshold(
            result.sorted_mean_values,
            config
        )
    else:
        # Use global threshold fallback
        threshold = global_strategy.calculate_threshold(
            all_bubble_means,
            config
        )

    # Expected: threshold ≈ 96.6 (midpoint of 91.5 gap)

END
```

---

## Statistics Usage Patterns

### Pattern 1: Quality-Based Strategy Selection

```python
def select_threshold_strategy(result: BubbleFieldDetectionResult):
    """Select best threshold strategy based on statistics."""

    if result.scan_quality == ScanQuality.EXCELLENT:
        # Very clear contrast, use local threshold
        return LocalThresholdStrategy()

    elif result.scan_quality == ScanQuality.GOOD:
        # Good contrast, try local with global fallback
        return LocalThresholdStrategy(
            global_fallback=get_global_threshold()
        )

    elif result.scan_quality == ScanQuality.ACCEPTABLE:
        # Marginal contrast, prefer global
        return GlobalThresholdStrategy()

    else:  # POOR
        # Poor contrast, may need ML detection
        if ml_detector_available():
            return MLBubbleDetector()
        else:
            return GlobalThresholdStrategy()
```

### Pattern 2: Confidence Scoring

```python
def calculate_confidence(result: BubbleFieldDetectionResult) -> float:
    """Calculate confidence based on statistics."""

    # Base confidence from max jump
    if result.max_jump >= 90:
        confidence = 1.0
    elif result.max_jump >= 60:
        confidence = 0.8
    elif result.max_jump >= 30:
        confidence = 0.5
    else:
        confidence = 0.2

    # Adjust for scan quality
    quality_multiplier = {
        ScanQuality.EXCELLENT: 1.0,
        ScanQuality.GOOD: 0.9,
        ScanQuality.ACCEPTABLE: 0.7,
        ScanQuality.POOR: 0.5
    }

    confidence *= quality_multiplier[result.scan_quality]

    return confidence
```

### Pattern 3: Outlier Detection

```python
def detect_outliers(result: BubbleFieldDetectionResult) -> list[BubbleMeanValue]:
    """Detect outlier bubbles using statistics."""

    mean = np.mean(result.mean_values)
    std = result.std_deviation

    outliers = []
    for bubble_mean in result.bubble_means:
        # Values > 2σ from mean are outliers
        if abs(bubble_mean.mean_value - mean) > 2 * std:
            outliers.append(bubble_mean)

    return outliers
```

---

## Browser Migration

### JavaScript/TypeScript Implementation

```typescript
interface BubbleMeanValue {
  meanValue: number;
  unitBubble: BubblesScanBox;
  position: [number, number];
}

interface BubbleFieldDetectionResult {
  fieldId: string;
  fieldLabel: string;
  bubbleMeans: BubbleMeanValue[];
  timestamp: Date;

  // Computed properties (use getters)
  get stdDeviation(): number;
  get scanQuality(): ScanQuality;
  get isReliable(): boolean;
  get sortedBubbleMeans(): BubbleMeanValue[];
  get meanValues(): number[];
  get sortedMeanValues(): number[];
  get jumps(): Array<[number, BubbleMeanValue]>;
  get maxJump(): number;
  get minMean(): number;
  get maxMean(): number;
}

enum ScanQuality {
  EXCELLENT = "excellent",
  GOOD = "good",
  ACCEPTABLE = "acceptable",
  POOR = "poor"
}

class BubbleFieldDetectionResultImpl implements BubbleFieldDetectionResult {
  constructor(
    public fieldId: string,
    public fieldLabel: string,
    public bubbleMeans: BubbleMeanValue[],
    public timestamp: Date = new Date()
  ) {}

  get stdDeviation(): number {
    if (this.bubbleMeans.length === 0) return 0.0;

    const values = this.bubbleMeans.map(bm => bm.meanValue);
    const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
    const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;

    return Math.sqrt(variance);
  }

  get scanQuality(): ScanQuality {
    const std = this.stdDeviation;

    if (std > 50) return ScanQuality.EXCELLENT;
    if (std > 30) return ScanQuality.GOOD;
    if (std > 15) return ScanQuality.ACCEPTABLE;
    return ScanQuality.POOR;
  }

  get isReliable(): boolean {
    return [ScanQuality.EXCELLENT, ScanQuality.GOOD].includes(this.scanQuality);
  }

  get sortedBubbleMeans(): BubbleMeanValue[] {
    return [...this.bubbleMeans].sort((a, b) => a.meanValue - b.meanValue);
  }

  get meanValues(): number[] {
    return this.bubbleMeans.map(bm => bm.meanValue);
  }

  get sortedMeanValues(): number[] {
    return this.sortedBubbleMeans.map(bm => bm.meanValue);
  }

  get jumps(): Array<[number, BubbleMeanValue]> {
    const sorted = this.sortedBubbleMeans;
    if (sorted.length < 2) return [];

    const jumpsList: Array<[number, BubbleMeanValue]> = [];
    for (let i = 1; i < sorted.length; i++) {
      const jump = sorted[i].meanValue - sorted[i - 1].meanValue;
      jumpsList.push([Math.round(jump * 100) / 100, sorted[i - 1]]);
    }

    return jumpsList;
  }

  get maxJump(): number {
    if (this.bubbleMeans.length < 2) return 0.0;
    const jumps = this.jumps;
    return jumps.length > 0 ? Math.max(...jumps.map(([jump]) => jump)) : 0.0;
  }

  get minMean(): number {
    return this.bubbleMeans.length > 0
      ? Math.min(...this.meanValues)
      : 0.0;
  }

  get maxMean(): number {
    return this.bubbleMeans.length > 0
      ? Math.max(...this.meanValues)
      : 255.0;
  }
}
```

### Key Differences from Python

1. **Standard Deviation**:
   ```javascript
   // Python: np.std(values)

   // JavaScript: Manual calculation
   const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
   const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
   const std = Math.sqrt(variance);
   ```

2. **Sorting**:
   ```javascript
   // Python: sorted() creates new list
   sorted_means = sorted(self.bubble_means)

   // JavaScript: Must explicitly copy
   const sortedMeans = [...this.bubbleMeans].sort((a, b) => a.meanValue - b.meanValue);
   ```

3. **Property Decorators**:
   ```typescript
   // Python: @property
   @property
   def std_deviation(self) -> float:
       ...

   // TypeScript: get keyword
   get stdDeviation(): number {
       ...
   }
   ```

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `std_deviation` | O(n) | Single pass to calculate |
| `scan_quality` | O(n) | Depends on std_deviation |
| `sorted_bubble_means` | O(n log n) | Sorting dominates |
| `jumps` | O(n log n) | Sorting + linear pass |
| `max_jump` | O(n log n) | Depends on jumps |
| `min_mean` / `max_mean` | O(n) | Single pass |

### Space Complexity

| Operation | Space | Notes |
|-----------|-------|-------|
| `sorted_bubble_means` | O(n) | Copy of bubble list |
| `jumps` | O(n) | List of jump tuples |
| Other properties | O(1) | Computed on-demand |

### Typical Performance

```
Bubbles per field: 4-10 (typical)
Time per field: < 1ms (all statistics)
Memory per field: ~1-2 KB
```

---

## Related Documentation

- **Bubble Detection Concept**: `../concept.md`
- **Detection Pass**: `../detection-pass/flows.md`
- **Interpretation Pass**: `../interpretation-pass/flows.md`
- **Global Threshold**: `../../threshold/global/flows.md`

---

## Summary

Bubble Detection Statistics provides:

1. **Standard Deviation**: Measures contrast between filled/empty bubbles
2. **Scan Quality**: Auto-classifies as EXCELLENT/GOOD/ACCEPTABLE/POOR
3. **Reliability Check**: Boolean flag for high-confidence detection
4. **Jump Detection**: Finds gaps between bubble groups
5. **Min/Max Tracking**: Monitors value range for quality assessment
6. **Sorted Access**: Efficient access to ordered bubble values

**Key Insight**: All statistics are **lazily computed properties** that cache internally, enabling efficient multi-access without recalculation.
