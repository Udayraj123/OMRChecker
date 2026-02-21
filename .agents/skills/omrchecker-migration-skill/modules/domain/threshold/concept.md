# Threshold Strategy Pattern

**Module**: Domain - Threshold
**Python Reference**: `src/processors/threshold/threshold_strategy.py`
**Last Updated**: 2026-02-20

---

## Overview

The Threshold Strategy Pattern provides a pluggable system for calculating optimal threshold values used in bubble detection. Different strategies can be applied based on the detection context (file-level vs field-level) to determine whether a bubble is filled or empty.

**Key Principles**:
1. **Strategy Pattern**: Abstract base class with multiple implementations
2. **Confidence-based**: All strategies return confidence scores (0.0-1.0)
3. **Fallback Support**: Graceful degradation when confidence is low
4. **Metadata Tracking**: Each result includes method information and metadata
5. **Composable**: Strategies can be combined (adaptive threshold)

---

## Core Abstractions

### ThresholdStrategy (Abstract Base Class)

```python
class ThresholdStrategy(ABC):
    @abstractmethod
    def calculate_threshold(
        self, bubble_mean_values: list[float], config: ThresholdConfig
    ) -> ThresholdResult:
        """Calculate threshold from bubble mean intensity values."""
```

**Purpose**: Define common interface for all threshold calculation strategies

**Contract**:
- Input: List of bubble mean intensity values (0-255 range)
- Input: Configuration parameters for fine-tuning
- Output: ThresholdResult with threshold value and confidence

---

## Data Structures

### ThresholdConfig

Configuration dataclass controlling threshold calculation behavior.

```python
@dataclass
class ThresholdConfig:
    min_jump: float = 30.0
    jump_delta: float = 20.0
    min_gap_two_bubbles: float = 20.0
    min_jump_surplus_for_global_fallback: float = 10.0
    confident_jump_surplus_for_disparity: float = 15.0
    global_threshold_margin: float = 10.0
    outlier_deviation_threshold: float = 5.0
    default_threshold: float = 127.5
```

**Parameters**:
- `min_jump`: Minimum gap between filled/empty bubbles (default: 30.0)
- `jump_delta`: Delta for detecting two distinct jumps (default: 20.0)
- `min_gap_two_bubbles`: Minimum gap when only 2 bubbles exist (default: 20.0)
- `min_jump_surplus_for_global_fallback`: Extra jump needed to avoid global fallback (default: 10.0)
- `confident_jump_surplus_for_disparity`: Extra jump for high confidence (default: 15.0)
- `global_threshold_margin`: Safety margin for global threshold (default: 10.0)
- `outlier_deviation_threshold`: Std deviation for outlier detection (default: 5.0)
- `default_threshold`: Final fallback value (default: 127.5 - midpoint of 0-255)

### ThresholdResult

Result dataclass returned by all strategies.

```python
@dataclass
class ThresholdResult:
    threshold_value: float       # The calculated threshold
    confidence: float            # 0.0-1.0, higher = more confident
    max_jump: float             # Largest gap found in values
    method_used: str            # Which method/strategy was used
    fallback_used: bool = False # Whether fallback was triggered
    metadata: dict = None       # Strategy-specific metadata
```

**Fields**:
- `threshold_value`: The computed threshold (0-255 range)
- `confidence`: How confident the strategy is (0.0 = no confidence, 1.0 = very confident)
- `max_jump`: Size of largest gap detected in bubble values
- `method_used`: String identifier for debugging ("global_max_jump", "local_two_bubbles_mean", etc.)
- `fallback_used`: True if default/global fallback was used
- `metadata`: Additional context (bubble counts, value ranges, etc.)

---

## Available Strategies

### 1. Global Threshold Strategy

**File**: `src/processors/threshold/global_threshold.py`
**Purpose**: Calculate file-level threshold using all bubble values from entire image

**Algorithm**:
1. Collect all bubble mean values across all fields
2. Sort values in ascending order
3. Find largest gap using "looseness" parameter
4. Place threshold at midpoint of largest gap

**When to Use**:
- As file-level baseline for all fields
- When you have many bubbles across multiple fields
- For images with consistent lighting

**Confidence Calculation**:
```python
confidence = min(1.0, max_jump / (min_jump * 3))
```

### 2. Local Threshold Strategy

**File**: `src/processors/threshold/local_threshold.py`
**Purpose**: Calculate field-level threshold using only bubbles from single field

**Algorithm**:
1. For 0-1 bubbles: Use global fallback
2. For 2 bubbles: Check gap, use mean if gap is sufficient
3. For 3+ bubbles: Find largest jump between adjacent values

**When to Use**:
- For per-field threshold calculation
- When lighting varies across the image
- For higher accuracy on individual fields

**Special Cases**:
- **Single bubble**: Confidence 0.0, uses fallback
- **Two bubbles with small gap**: Confidence 0.3, uses fallback
- **Two bubbles with large gap**: Confidence 0.7, uses mean of values
- **Low confidence jump**: Confidence 0.4, uses global fallback
- **High confidence jump**: Confidence calculated based on jump size

### 3. Adaptive Threshold Strategy

**File**: `src/processors/threshold/adaptive_threshold.py`
**Purpose**: Combine multiple strategies using weighted averaging

**Algorithm**:
1. Run all child strategies
2. Calculate weighted average based on confidence scores
3. Use highest confidence from any strategy
4. Track max jump from all strategies

**When to Use**:
- Default strategy (combines global + local)
- When you want robust threshold across varying conditions
- For production use with best accuracy

**Default Configuration**:
```python
AdaptiveThresholdStrategy(
    strategies=[GlobalThresholdStrategy(), LocalThresholdStrategy()],
    weights=[0.4, 0.6]  # Prefer local threshold
)
```

**Weighted Average Formula**:
```python
weighted_threshold = sum(
    result.threshold * result.confidence * weight
    for result, weight in zip(results, weights)
) / total_weight
```

---

## Usage Flow

### File-Level Detection Flow

```
1. Bubble Detection Pass
   → Extract all bubbles from all fields
   → Calculate mean intensity for each bubble

2. Global Threshold Calculation
   GlobalThresholdStrategy.calculate_threshold(all_bubble_means, config)
   → Returns file-level threshold

3. Store Global Threshold
   → Used as fallback for local thresholds
```

### Field-Level Interpretation Flow

```
1. Per-Field Processing
   For each field:

2. Local Threshold Calculation
   AdaptiveThresholdStrategy.calculate_threshold(field_bubble_means, config)
   → Combines global + local strategies
   → Returns field-specific threshold

3. Bubble Classification
   For each bubble in field:
   if bubble_mean < threshold:
       → Bubble is FILLED (darker)
   else:
       → Bubble is EMPTY (lighter)
```

---

## Strategy Selection Logic

### When Global Strategy Is Preferred

- Very few bubbles per field (< 3)
- Consistent lighting across entire image
- Fast processing required
- Lower accuracy acceptable

### When Local Strategy Is Preferred

- Uneven lighting across image
- Many bubbles per field (3+)
- Higher accuracy required
- Per-field customization needed

### When Adaptive Strategy Is Preferred (Recommended)

- Production use
- Unknown image conditions
- Best accuracy required
- Combination of global baseline + local refinement

---

## Edge Cases & Constraints

### Edge Case: Empty Field (0 bubbles)

**Issue**: No data to calculate threshold
**Handling**: Return global fallback with confidence 0.0
**Impact**: Field will use file-level threshold

### Edge Case: Single Bubble

**Issue**: Cannot determine filled vs empty with single sample
**Handling**: Return global fallback with confidence 0.0
**Impact**: Relies entirely on global threshold

### Edge Case: Two Bubbles with Small Gap

**Issue**: Gap might be noise, not actual filled/empty distinction
**Handling**:
- If gap < `min_gap_two_bubbles`: Use global fallback
- Else: Use mean of two values
**Impact**: Prevents false classification from noise

### Edge Case: All Bubbles Same Intensity

**Issue**: max_jump = 0, no distinction possible
**Handling**: Return global fallback or default threshold
**Impact**: Cannot classify bubbles (likely marking error or scan issue)

### Edge Case: Outlier Bubbles

**Issue**: Single very dark/light bubble skews threshold
**Handling**: Looseness parameter in global strategy ignores extreme values
**Impact**: More robust threshold calculation

### Constraint: Grayscale Range

**Values**: 0 (black) to 255 (white)
**Filled Bubble**: Lower values (darker)
**Empty Bubble**: Higher values (lighter)
**Threshold**: Separates the two groups

### Constraint: Confidence Bounds

**Range**: 0.0 (no confidence) to 1.0 (very confident)
**Interpretation**:
- < 0.3: Low confidence, likely fallback
- 0.3-0.7: Medium confidence
- > 0.7: High confidence

---

## Browser Migration Notes

### JavaScript Strategy Pattern

```typescript
abstract class ThresholdStrategy {
  abstract calculateThreshold(
    bubbleMeanValues: number[],
    config: ThresholdConfig
  ): ThresholdResult;
}

class GlobalThresholdStrategy extends ThresholdStrategy {
  calculateThreshold(
    bubbleMeanValues: number[],
    config: ThresholdConfig
  ): ThresholdResult {
    // Implementation
  }
}
```

### TypeScript Interfaces

```typescript
interface ThresholdConfig {
  minJump: number;
  jumpDelta: number;
  minGapTwoBubbles: number;
  minJumpSurplusForGlobalFallback: number;
  confidentJumpSurplusForDisparity: number;
  globalThresholdMargin: number;
  outlierDeviationThreshold: number;
  defaultThreshold: number;
}

interface ThresholdResult {
  thresholdValue: number;
  confidence: number;
  maxJump: number;
  methodUsed: string;
  fallbackUsed: boolean;
  metadata?: Record<string, unknown>;
}
```

### Array Operations

**Python**:
```python
sorted_values = sorted(bubble_mean_values)
max_jump = max(jumps)
```

**JavaScript**:
```javascript
const sortedValues = [...bubbleMeanValues].sort((a, b) => a - b);
const maxJump = Math.max(...jumps);
```

### NumPy to JavaScript

**Python**:
```python
threshold = float(np.mean(sorted_values))
```

**JavaScript**:
```javascript
const threshold = sortedValues.reduce((a, b) => a + b, 0) / sortedValues.length;
```

### Factory Pattern

**Python**:
```python
def create_default_threshold_calculator(global_threshold=None):
    return AdaptiveThresholdStrategy(
        strategies=[GlobalThresholdStrategy(), LocalThresholdStrategy()],
        weights=[0.4, 0.6]
    )
```

**JavaScript**:
```javascript
function createDefaultThresholdCalculator(globalThreshold?: number): ThresholdStrategy {
  return new AdaptiveThresholdStrategy(
    [new GlobalThresholdStrategy(), new LocalThresholdStrategy(globalThreshold)],
    [0.4, 0.6]
  );
}
```

---

## Performance Considerations

### Time Complexity

- **Global**: O(n log n) - dominated by sorting all bubbles
- **Local**: O(m log m) - sorting bubbles in single field (m << n)
- **Adaptive**: O(k * m log m) - k strategies × local complexity

Where:
- n = total bubbles in file
- m = bubbles per field
- k = number of strategies

### Memory

- **Input**: List of float values (8 bytes × bubble count)
- **Sorted**: Copy of input for sorting
- **Result**: ThresholdResult object (~100 bytes)

### Optimization Tips

1. **Cache Global Threshold**: Calculate once per file, reuse for all fields
2. **Avoid Re-sorting**: Local threshold only needs field bubbles
3. **Lazy Evaluation**: Only calculate strategies when needed
4. **Minimal Metadata**: Only store metadata when debugging

---

## Testing Strategies

### Unit Tests

```python
def test_global_threshold_basic():
    strategy = GlobalThresholdStrategy()
    values = [50, 55, 60, 150, 155, 160]  # Two distinct groups
    config = ThresholdConfig()
    result = strategy.calculate_threshold(values, config)

    assert 60 < result.threshold_value < 150  # Between groups
    assert result.confidence > 0.5
    assert result.max_jump > config.min_jump
```

### Integration Tests

```python
def test_adaptive_strategy_combines_results():
    global_strategy = GlobalThresholdStrategy()
    local_strategy = LocalThresholdStrategy()
    adaptive = AdaptiveThresholdStrategy([global_strategy, local_strategy])

    values = [45, 50, 55, 145, 150, 155]
    config = ThresholdConfig()
    result = adaptive.calculate_threshold(values, config)

    assert "adaptive_weighted" in result.method_used
    assert "strategy_results" in result.metadata
```

### Edge Case Tests

```python
def test_threshold_single_bubble():
    strategy = LocalThresholdStrategy()
    values = [100]
    config = ThresholdConfig()
    result = strategy.calculate_threshold(values, config)

    assert result.fallback_used is True
    assert result.confidence == 0.0
```

---

## Related Documentation

- **Bubble Detection**: `modules/domain/detection/bubbles-threshold/concept.md`
- **Detection Pass**: `modules/domain/detection/bubbles-threshold/detection-pass/flows.md`
- **Interpretation Pass**: `modules/domain/detection/bubbles-threshold/interpretation-pass/flows.md`
- **Math Utils**: `modules/domain/utils/math/flows.md`

---

## Summary

The Threshold Strategy Pattern provides:

1. **Flexibility**: Pluggable algorithms for different scenarios
2. **Robustness**: Multiple fallback levels for edge cases
3. **Accuracy**: Field-level refinement over global baseline
4. **Transparency**: Confidence scores and metadata for debugging
5. **Composability**: Strategies can be combined for best results

**Default Recommendation**: Use `AdaptiveThresholdStrategy` with 40% global + 60% local weighting for production use.
