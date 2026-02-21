# Threshold Strategy Constraints

**Module**: Domain - Threshold
**Python Reference**: `src/processors/threshold/*.py`
**Last Updated**: 2026-02-20

---

## Input Constraints

### Bubble Mean Values

**Type**: `list[float]`
**Range**: 0.0 to 255.0 (grayscale intensity)
**Description**: Mean pixel intensity for each bubble ROI

**Constraints**:
```python
# Valid
bubble_mean_values = [45.5, 50.2, 145.8, 150.1]

# Edge cases
[] # Empty list - valid, triggers fallback
[100.5] # Single value - valid, triggers fallback
[100.5, 100.5] # Duplicate values - valid, small gap
[0.0, 255.0] # Extreme values - valid, large gap
```

**Invariants**:
- All values must be finite (no NaN, no Inf)
- Values typically in range [0, 255] but not strictly enforced
- List can be empty (length 0)
- List can have duplicates

**Physical Meaning**:
- **0**: Pure black (fully filled bubble)
- **255**: Pure white (empty bubble)
- **< 100**: Typically filled/marked bubbles
- **> 100**: Typically empty/unmarked bubbles

**Ordering**:
- Input list is unsorted
- Strategies internally sort for processing
- Original order is not preserved

---

## Configuration Constraints

### ThresholdConfig Parameters

#### min_jump

**Type**: `float`
**Default**: 30.0
**Range**: Recommended 15.0 to 50.0

**Constraints**:
```python
config.min_jump > 0  # Must be positive
config.min_jump < 255  # Less than full grayscale range
```

**Meaning**: Minimum intensity gap to consider significant separation between filled/empty bubbles.

**Impact**:
- **Too small** (< 15): May detect noise as threshold
- **Too large** (> 50): May miss real gaps, overuse fallback
- **Recommended**: 30.0 works well for typical OMR sheets

**Related to**:
- Image contrast
- Bubble darkness variation
- Scan quality

#### jump_delta

**Type**: `float`
**Default**: 20.0
**Range**: Recommended 10.0 to 30.0

**Constraints**:
```python
config.jump_delta > 0
config.jump_delta < config.min_jump  # Typically
```

**Meaning**: Minimum difference between two jumps to distinguish them.

**Impact**:
- Used in two-jump detection scenarios
- Currently less critical (mostly used in advanced threshold logic)

#### min_gap_two_bubbles

**Type**: `float`
**Default**: 20.0
**Range**: Recommended 15.0 to 30.0

**Constraints**:
```python
config.min_gap_two_bubbles > 0
config.min_gap_two_bubbles <= config.min_jump
```

**Meaning**: For exactly 2 bubbles, minimum gap to use their mean as threshold.

**Impact**:
- **Too small** (< 15): May treat noise as real gap
- **Too large** (> 30): May overuse global fallback

**Example**:
```python
# Field with 2 bubbles
bubbles = [50, 70]  # gap = 20

if 20 >= config.min_gap_two_bubbles:  # 20 >= 20 ✓
    threshold = (50 + 70) / 2 = 60
else:
    threshold = global_fallback
```

#### min_jump_surplus_for_global_fallback

**Type**: `float`
**Default**: 10.0
**Range**: Recommended 5.0 to 20.0

**Constraints**:
```python
config.min_jump_surplus_for_global_fallback >= 0
```

**Meaning**: Extra jump required beyond `min_jump` to avoid using global fallback.

**Impact**:
```python
confident_jump = config.min_jump + config.min_jump_surplus_for_global_fallback
# = 30.0 + 10.0 = 40.0

if max_jump < confident_jump:  # e.g., 35 < 40
    use_global_fallback()  # Not confident enough
else:
    use_local_threshold()  # Confident
```

**Purpose**: Provides safety margin for local threshold confidence.

#### global_threshold_margin

**Type**: `float`
**Default**: 10.0
**Range**: Recommended 5.0 to 15.0

**Constraints**:
```python
config.global_threshold_margin >= 0
config.global_threshold_margin < config.min_jump
```

**Meaning**: Safety margin when using global threshold.

**Impact**: Currently used in advanced scenarios; provides buffer around threshold.

#### outlier_deviation_threshold

**Type**: `float`
**Default**: 5.0
**Range**: Recommended 3.0 to 10.0

**Constraints**:
```python
config.outlier_deviation_threshold > 0
```

**Meaning**: Standard deviation threshold for outlier detection.

**Impact**: Used in advanced outlier detection (not in basic strategies).

#### default_threshold

**Type**: `float`
**Default**: 127.5 (midpoint of 0-255)
**Range**: Typically 100.0 to 150.0

**Constraints**:
```python
0 < config.default_threshold < 255
```

**Meaning**: Final fallback when all strategies fail.

**Impact**: Used when:
- No bubbles detected
- All bubbles have same intensity
- All strategies have zero confidence

**Rationale for 127.5**:
- Midpoint of grayscale range
- Neutral assumption (no bias toward filled/empty)
- Works as emergency fallback

---

## Output Constraints

### ThresholdResult

#### threshold_value

**Type**: `float`
**Range**: Typically 0.0 to 255.0

**Constraints**:
```python
# Expected range
0 <= result.threshold_value <= 255

# Typical range (more common)
30 < result.threshold_value < 200
```

**Meaning**: The calculated threshold separating filled/empty bubbles.

**Usage**:
```python
for bubble_mean in field_bubbles:
    if bubble_mean < threshold_value:
        state = "FILLED"  # Darker
    else:
        state = "EMPTY"   # Lighter
```

**Invariant**: Should be between min and max bubble values for best separation.

#### confidence

**Type**: `float`
**Range**: 0.0 to 1.0 (strictly bounded)

**Constraints**:
```python
0.0 <= result.confidence <= 1.0
```

**Meaning**: How confident the strategy is in the threshold.

**Interpretation**:
- **0.0**: No confidence (fallback used)
- **0.0 - 0.3**: Low confidence (likely fallback or edge case)
- **0.3 - 0.7**: Medium confidence (acceptable)
- **0.7 - 1.0**: High confidence (good separation found)
- **1.0**: Very high confidence (capped maximum)

**Calculation Examples**:
```python
# Global strategy
confidence = min(1.0, max_jump / (config.min_jump * 3))
# max_jump=90, min_jump=30 → 90/90 = 1.0 ✓
# max_jump=45, min_jump=30 → 45/90 = 0.5
# max_jump=15, min_jump=30 → 15/90 = 0.17

# Local strategy (high confidence path)
confidence = min(1.0, max_jump / (confident_jump * 2))
# max_jump=80, confident_jump=40 → 80/80 = 1.0 ✓
# max_jump=60, confident_jump=40 → 60/80 = 0.75
```

#### max_jump

**Type**: `float`
**Range**: 0.0 to 255.0

**Constraints**:
```python
0.0 <= result.max_jump <= 255.0
```

**Meaning**: Largest intensity gap found between bubble groups.

**Invariants**:
```python
# For sorted values
max_jump <= max(bubble_values) - min(bubble_values)

# For empty list
max_jump == 0.0
```

**Physical Meaning**:
- **0**: All bubbles same intensity (no separation)
- **< min_jump**: Weak separation (low confidence)
- **>= min_jump**: Good separation (higher confidence)
- **> 100**: Excellent separation (very high confidence)

#### method_used

**Type**: `str`
**Constraints**: Must be one of the documented method names

**Valid Values**:
```python
# Global strategy
"global_default"      # < 2 bubbles
"global_max_jump"     # Normal case

# Local strategy
"local_single_bubble_fallback"
"local_two_bubbles_small_gap_fallback"
"local_two_bubbles_mean"
"local_low_confidence_global_fallback"
"local_max_jump"

# Adaptive strategy
"adaptive_weighted"
"adaptive_all_zero_confidence"
```

**Purpose**: Debugging and monitoring

**Invariant**: Must be non-empty string

#### fallback_used

**Type**: `bool`
**Constraints**: True or False

**Meaning**: Whether fallback threshold was used (vs calculated threshold).

**Correlation with Confidence**:
```python
if fallback_used:
    assert confidence < 0.8  # Typically low confidence

# But not always:
# Two bubbles with small gap:
#   fallback_used = True
#   confidence = 0.3 (medium-low)
```

**Usage**:
```python
if result.fallback_used:
    log.warning(f"Fallback threshold used: {result.method_used}")
```

#### metadata

**Type**: `dict | None`
**Constraints**: Can contain any JSON-serializable data

**Common Fields**:
```python
# Global strategy
{
    "num_bubbles": int,  # >= 0
    "min_value": float,  # 0-255
    "max_value": float   # 0-255
}

# Local strategy
{
    "num_bubbles": int,
    "local_threshold": float  # When fallback used
}

# Adaptive strategy
{
    "strategy_results": [
        {
            "method": str,
            "threshold": float,
            "confidence": float,
            "weight": float
        },
        ...
    ]
}
```

**Invariants**:
- metadata is never None (initialized to {} in __post_init__)
- All keys are strings
- All values are JSON-serializable

---

## Strategy-Specific Constraints

### GlobalThresholdStrategy

**Input Requirements**:
- No minimum bubble count (handles 0, 1, 2+ bubbles)

**Guarantees**:
- Always returns a valid threshold (uses default if needed)
- Never raises exception for valid input

**Confidence Bounds**:
```python
if len(bubbles) < 2:
    confidence = 0.0
else:
    confidence = min(1.0, max_jump / (min_jump * 3))
    # Range: [0.0, 1.0]
```

**Fallback Condition**:
```python
fallback_used = (max_jump < config.min_jump) or (len(bubbles) < 2)
```

### LocalThresholdStrategy

**Input Requirements**:
- Must have global_fallback set (or uses default_threshold)

**Guarantees**:
- Always returns a valid threshold
- Uses global fallback when local confidence is low

**Confidence Bounds**:
```python
if len(bubbles) == 0:
    confidence = 0.0
elif len(bubbles) == 1:
    confidence = 0.0
elif len(bubbles) == 2:
    if gap < min_gap_two_bubbles:
        confidence = 0.3  # Low
    else:
        confidence = 0.7  # Medium-high
else:  # 3+ bubbles
    if max_jump < confident_jump:
        confidence = 0.4  # Medium-low
    else:
        confidence = min(1.0, max_jump / (confident_jump * 2))
        # Range: [0.0, 1.0]
```

**Fallback Conditions**:
```python
fallback_used = (
    len(bubbles) < 2
    or (len(bubbles) == 2 and gap < min_gap_two_bubbles)
    or (len(bubbles) >= 3 and max_jump < confident_jump)
)
```

### AdaptiveThresholdStrategy

**Input Requirements**:
```python
# Must have at least one strategy
len(strategies) >= 1

# Strategies and weights must match
len(strategies) == len(weights)

# Weights must be positive
all(w > 0 for w in weights)
```

**Guarantees**:
- Returns weighted average when any strategy has confidence > 0
- Falls back to default when all strategies have confidence = 0

**Confidence Bounds**:
```python
if all(result.confidence == 0 for result in results):
    confidence = 0.0
else:
    confidence = max(result.confidence for result in results)
    # Range: (0.0, 1.0]
```

**Weighted Threshold Calculation**:
```python
# total_weight = sum(conf * weight for conf, weight in ...)
assert total_weight >= 0

if total_weight == 0:
    threshold = config.default_threshold
else:
    threshold = sum(
        result.threshold * result.confidence * weight
        for result, weight in zip(results, weights)
    ) / total_weight

    # Threshold is bounded by min/max of strategy thresholds
    min_thresh = min(r.threshold_value for r in results)
    max_thresh = max(r.threshold_value for r in results)
    assert min_thresh <= threshold <= max_thresh
```

---

## Performance Constraints

### Time Complexity

**Global Strategy**:
```python
# O(n log n) where n = total bubbles
sorted_values = sorted(bubble_mean_values)  # O(n log n)
for i in range(len(sorted_values)):         # O(n)
    # Find max jump
```

**Local Strategy**:
```python
# O(m log m) where m = bubbles per field (typically << n)
sorted_values = sorted(bubble_mean_values)  # O(m log m)
for i in range(len(sorted_values)):         # O(m)
    # Find max jump
```

**Adaptive Strategy**:
```python
# O(k * m log m) where k = number of strategies
for strategy in strategies:                 # O(k)
    result = strategy.calculate_threshold()  # O(m log m)
# Weighted average calculation                O(k)
```

**Typical Values**:
- n (total bubbles): 100-500
- m (bubbles per field): 4-20
- k (strategies): 2

**Performance Target**: < 1ms per field threshold calculation

### Memory Constraints

**Global Strategy**:
```python
# Memory: O(n)
sorted_values = sorted(bubble_mean_values)  # Copy of input
# ThresholdResult: ~200 bytes
# Total: ~8n + 200 bytes
```

**Local Strategy**:
```python
# Memory: O(m)
sorted_values = sorted(bubble_mean_values)  # Copy of input
# ThresholdResult: ~200 bytes
# Total: ~8m + 200 bytes
```

**Adaptive Strategy**:
```python
# Memory: O(k * m)
results = []  # k ThresholdResult objects
# Each result: ~200 bytes + metadata
# Total: ~k * (8m + 500) bytes
```

**Typical Memory**:
- Local with 10 bubbles: ~280 bytes
- Adaptive with 2 strategies, 10 bubbles: ~1.4 KB
- **Browser Impact**: Negligible (KB range)

---

## Numerical Constraints

### Floating Point Precision

**Concern**: Grayscale values are 0-255 integers, but calculations use floats.

**Constraints**:
```python
# Input: typically integer values from np.mean()
bubble_mean = np.mean(roi)  # Returns float64

# Calculations maintain precision
threshold = sorted_values[i-1] + jump / 2  # Float division

# Comparison uses < operator (strict less than)
if bubble_mean < threshold:  # No epsilon needed for grayscale
```

**Edge Case**: Equal values
```python
# If bubble_mean == threshold exactly:
#   bubble_mean < threshold → False → EMPTY

# This is acceptable: threshold is boundary,
# exact equality is exceedingly rare with float means
```

### Division by Zero

**Prevented**:
```python
# Weighted average
if total_weight == 0:  # Explicit check
    # Use fallback instead of dividing
else:
    threshold = sum(...) / total_weight  # Safe

# Confidence calculation
confidence = min(1.0, max_jump / (min_jump * 3))
# min_jump default is 30.0, never zero
# Division is safe
```

### Overflow/Underflow

**Not a concern**:
- Grayscale values: 0-255 (small numbers)
- Differences: max 255 (small)
- Sums: max n * 255 (Python handles arbitrary precision)
- Float64 has range ±1.7e308 (far exceeds our needs)

---

## Concurrency Constraints

### Thread Safety

**Strategies are stateless**:
```python
# Safe for concurrent use
strategy = GlobalThresholdStrategy()

# These can run in parallel:
result1 = strategy.calculate_threshold(bubbles1, config)
result2 = strategy.calculate_threshold(bubbles2, config)
```

**LocalThresholdStrategy with global_fallback**:
```python
# global_fallback is set in __init__, then immutable
strategy = LocalThresholdStrategy(global_fallback=96.0)

# Safe: no mutation during calculate_threshold
result = strategy.calculate_threshold(bubbles, config)
```

**No shared state**:
- All inputs are passed as parameters
- No class-level mutable state
- Each call is independent

**Browser Implication**: Can use Web Workers for parallel field processing.

---

## Browser Migration Constraints

### JavaScript Number Precision

**Python float64 = JavaScript number**:
- Both are IEEE 754 double precision
- Same precision for our range (0-255)

**Array Operations**:
```javascript
// Python:
sorted_values = sorted(bubble_mean_values)

// JavaScript:
const sortedValues = [...bubbleMeanValues].sort((a, b) => a - b);
//                   ^^^^ Copy to avoid mutating input
```

### Type Constraints

**TypeScript Interfaces**:
```typescript
interface ThresholdConfig {
  readonly minJump: number;
  readonly jumpDelta: number;
  // ... other fields
}

interface ThresholdResult {
  readonly thresholdValue: number;
  readonly confidence: number;
  readonly maxJump: number;
  readonly methodUsed: string;
  readonly fallbackUsed: boolean;
  readonly metadata: Readonly<Record<string, unknown>>;
}
```

**Validation**:
```typescript
function validateConfig(config: ThresholdConfig): void {
  if (config.minJump <= 0) {
    throw new Error("minJump must be positive");
  }
  if (config.confidence < 0 || config.confidence > 1) {
    throw new Error("confidence must be in [0, 1]");
  }
  // ... other validations
}
```

### No NumPy Dependency

**Python**:
```python
threshold = float(np.mean(sorted_values))
```

**JavaScript**:
```javascript
const threshold = sortedValues.reduce((a, b) => a + b, 0) / sortedValues.length;
```

**Both produce same result for our use case.**

---

## Error Handling Constraints

### No Exceptions in Normal Flow

**Strategies never raise exceptions for valid input**:
```python
# Always returns ThresholdResult, never raises
result = strategy.calculate_threshold(bubbles, config)
```

**Invalid input handling**:
```python
# Empty list: OK, returns fallback
result = strategy.calculate_threshold([], config)
assert result.fallback_used is True
assert result.confidence == 0.0

# Invalid config: Raises ValueError in __init__
config = ThresholdConfig(min_jump=-10)  # ❌ Should validate
```

**Recommendation**: Validate config at creation, not during calculation.

---

## Testing Constraints

### Determinism

**Strategies are deterministic**:
```python
# Same input always produces same output
result1 = strategy.calculate_threshold(bubbles, config)
result2 = strategy.calculate_threshold(bubbles, config)
assert result1 == result2
```

**No randomness**:
- No random number generation
- No time-based logic
- Fully reproducible

**Implication**: Easy to unit test, snapshot test results.

### Test Data Constraints

**Valid Test Cases**:
```python
# Empty field
test_case_1 = ([], config)

# Single bubble
test_case_2 = ([100], config)

# Two bubbles - small gap
test_case_3 = ([100, 110], config)

# Two bubbles - large gap
test_case_4 = ([50, 150], config)

# Normal case
test_case_5 = ([45, 50, 55, 145, 150, 155], config)

# All same value
test_case_6 = ([100, 100, 100], config)

# Outliers
test_case_7 = ([10, 50, 55, 60, 250], config)
```

---

## Summary of Critical Constraints

| Constraint | Value | Rationale |
|------------|-------|-----------|
| Bubble values | [0, 255] | Grayscale range |
| Confidence | [0, 1] | Normalized probability |
| min_jump | > 0 | Must be positive gap |
| Strategy count | >= 1 | At least one strategy |
| Weights length | == strategies length | Must match |
| total_weight | > 0 or fallback | Prevent division by zero |
| Sorted array | Non-mutating | Preserve input |
| Time complexity | O(n log n) global, O(m log m) local | Acceptable for typical sizes |
| Memory | O(n) or O(m) | Small overhead |
| Thread safety | Stateless strategies | Safe for concurrency |
| Determinism | Always | Same input → same output |

---

## Related Constraints

- **Bubble Detection Constraints**: `modules/domain/detection/bubbles-threshold/constraints.md`
- **Image Processing Constraints**: `modules/domain/preprocessing/constraints.md` (pending)
- **Configuration Constraints**: `modules/foundation/configuration.md` (pending)
