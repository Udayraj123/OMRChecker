# Global Threshold Strategy Constraints

**Module**: Domain - Threshold - Global
**Python Reference**: `src/processors/threshold/global_threshold.py`
**Last Updated**: 2026-02-20

---

## Input Constraints

### bubble_mean_values

**Type**: `list[float]`
**Source**: All bubble mean intensities from entire OMR sheet
**Range**: 0.0 to 255.0 (grayscale)

**Valid Inputs**:
```python
[]                          # Empty - valid, uses fallback
[100.0]                     # Single value - valid, uses fallback
[50.0, 150.0]               # Two values - valid but limited
[45, 50, 55, 145, 150, 155] # Typical - optimal
```

**Constraints**:
- Minimum length: 0 (empty list allowed)
- Maximum length: Unlimited (typically 50-500 bubbles per sheet)
- Value range: 0.0-255.0 (not strictly enforced, but expected)
- Values must be finite (no NaN, no Inf)
- Duplicates allowed
- Order doesn't matter (internally sorted)

**Typical Size**:
```
Small sheet (20 questions × 4 options): 80 bubbles
Medium sheet (50 questions × 5 options): 250 bubbles
Large sheet (100 questions × 5 options): 500 bubbles
```

---

## Algorithm Constraints

### Looseness Parameter

**Value**: 1 (hardcoded)
**Type**: `int`
**Purpose**: Number of extreme values to ignore when finding gaps

**Impact**:
```python
looseness = 1
ls = (looseness + 1) // 2  # ls = 1

# Range for finding jumps:
for i in range(ls, len(sorted_values) - ls):
    # Skips first and last values

# Minimum bubbles needed for meaningful calculation:
min_bubbles = 2 * ls + 1 = 3
```

**Constraint**:
- With looseness=1, need at least 3 bubbles to calculate jump
- With 0-2 bubbles, loop range is empty → uses default threshold

**Rationale**:
- Protects against single outlier bubbles
- Requires at least small cluster to detect gap
- Balance between robustness and data requirement

### Jump Calculation Window

**Formula**: `jump = sorted_values[i + ls] - sorted_values[i - ls]`

**Window Size**: 3 values (i-1, i, i+1)

**Example**:
```
sorted_values = [45, 50, 52, 55, 140, 145, 150, 155]
                     ^       ^
                   i-1   i  i+1

i=2: jump = sorted[3] - sorted[1] = 55 - 50 = 5
i=3: jump = sorted[4] - sorted[2] = 140 - 52 = 88 ✓
```

**Constraint**:
- Requires i-ls >= 0 and i+ls < len(sorted_values)
- Enforced by range(ls, len - ls)

---

## Output Constraints

### threshold_value

**Type**: `float`
**Range**: Typically 0.0 to 255.0

**Possible Values**:
```python
# Default fallback
threshold = config.default_threshold  # Usually 127.5

# Calculated threshold
threshold = sorted_values[i-ls] + jump/2
# Range: min(bubbles) + min_jump/2  to  max(bubbles) - min_jump/2

# Example:
# sorted = [45, ..., 155]
# threshold ≈ 45 + (155-45)/2 = 100
```

**Constraint**:
- Always returns a valid float
- Never None or NaN
- If calculated, falls within [min_bubble + jump/2, max_bubble - jump/2]
- If fallback, equals config.default_threshold

### confidence

**Type**: `float`
**Range**: 0.0 to 1.0 (strictly bounded)

**Formula**:
```python
if len(bubbles) < 2:
    confidence = 0.0
else:
    confidence = min(1.0, max_jump / (config.min_jump * 3))
```

**Possible Values**:
```
max_jump | confidence
---------|------------
0        | 0.00
15       | 0.17 (15 / 90)
30       | 0.33 (at min_jump)
45       | 0.50
60       | 0.67
90+      | 1.00 (capped)
```

**Constraint**:
- Always in [0.0, 1.0]
- 0.0 if < 2 bubbles
- ≥ 0.33 if max_jump ≥ min_jump (30)
- Capped at 1.0 (even if formula > 1.0)

### max_jump

**Type**: `float`
**Range**: config.min_jump to 255.0

**Initialization**: `max_jump = config.min_jump`

**Update Rule**:
```python
for i in range(...):
    jump = sorted_values[i+ls] - sorted_values[i-ls]
    if jump > max_jump:
        max_jump = jump
```

**Constraint**:
- Minimum value: config.min_jump (never goes below initial value)
- Maximum value: max(bubbles) - min(bubbles) (full range)
- If no jump exceeds config.min_jump, stays at config.min_jump

**Meaning**:
```
max_jump < config.min_jump: No significant gap found
max_jump >= config.min_jump: Good separation detected
```

### method_used

**Type**: `str`
**Possible Values**: Exactly 2 values

```python
"global_default"  # When len(bubbles) < 2
"global_max_jump" # Normal case
```

**Constraint**:
- Always one of these two strings
- Never None or empty
- Used for debugging/monitoring

### fallback_used

**Type**: `bool`
**Values**: True or False

**Logic**:
```python
if len(bubbles) < 2:
    fallback_used = True  # Used default, no calculation possible
else:
    fallback_used = (max_jump < config.min_jump)
    # True if no significant gap found
```

**Correlation with Confidence**:
```python
if fallback_used and len(bubbles) < 2:
    confidence == 0.0

if fallback_used and len(bubbles) >= 2:
    confidence == 0.33  # Because max_jump = min_jump
```

### metadata

**Type**: `dict`
**Keys**: Always present when len(bubbles) >= 2

```python
{
    "num_bubbles": int,  # >= 0
    "min_value": float,  # Minimum bubble value
    "max_value": float   # Maximum bubble value
}
```

**Constraints**:
```python
# When bubbles is non-empty:
assert metadata["num_bubbles"] == len(bubble_mean_values)
assert metadata["min_value"] == min(bubble_mean_values)
assert metadata["max_value"] == max(bubble_mean_values)
assert metadata["min_value"] <= metadata["max_value"]

# When bubbles is empty:
metadata = {}  # Empty dict (from __post_init__)
```

---

## Configuration Constraints

### min_jump

**Type**: `float`
**Default**: 30.0
**Valid Range**: 1.0 to 100.0

**Constraints**:
```python
config.min_jump > 0  # Must be positive
config.min_jump < 255  # Less than full grayscale range
```

**Recommended Range**: 20.0 to 50.0

**Impact on Results**:
```
min_jump=20:
- More sensitive, detects smaller gaps
- Higher confidence for same gap
- Less likely to use fallback

min_jump=30 (default):
- Balanced sensitivity
- Good for typical OMR sheets

min_jump=50:
- Less sensitive, requires larger gaps
- Lower confidence for same gap
- More likely to use fallback
```

### default_threshold

**Type**: `float`
**Default**: 127.5
**Valid Range**: 0.0 to 255.0

**Constraints**:
```python
0 < config.default_threshold < 255
```

**Recommended Value**: 127.5 (midpoint)

**When Used**:
- len(bubbles) < 2
- max_jump < min_jump (though threshold may still be calculated value)

---

## Edge Case Constraints

### Edge Case 1: Empty List

**Input**: `bubble_mean_values = []`

**Constraints**:
```python
threshold_value == config.default_threshold
confidence == 0.0
max_jump == 0.0
method_used == "global_default"
fallback_used == True
metadata == {}  # Empty, no min/max to report
```

### Edge Case 2: Single Bubble

**Input**: `bubble_mean_values = [100.0]`

**Constraints**:
```python
# Same as empty list
threshold_value == config.default_threshold
confidence == 0.0
max_jump == 0.0
method_used == "global_default"
fallback_used == True
metadata == {}
```

### Edge Case 3: Two Bubbles

**Input**: `bubble_mean_values = [50, 150]`

**Constraints**:
```python
# sorted = [50, 150]
# ls = 1
# range(1, 1) is empty → no loop iterations

threshold_value == config.default_threshold  # Not updated
confidence == 0.33  # min(1.0, 30/90)
max_jump == 30.0  # Not updated, stays at min_jump
method_used == "global_max_jump"
fallback_used == False  # Technically not fallback, but no jump found
metadata == {
    "num_bubbles": 2,
    "min_value": 50,
    "max_value": 150
}
```

**Note**: Two bubbles with looseness=1 don't provide enough data for jump calculation. This is a known limitation.

### Edge Case 4: All Same Value

**Input**: `bubble_mean_values = [100, 100, 100, 100]`

**Constraints**:
```python
# All jumps = 0
# max_jump never exceeds min_jump

threshold_value == config.default_threshold
confidence == 0.33
max_jump == 30.0  # Stays at initial min_jump
method_used == "global_max_jump"
fallback_used == False  # Loop ran, just found no jump
metadata == {
    "num_bubbles": 4,
    "min_value": 100,
    "max_value": 100
}
```

### Edge Case 5: Extreme Values (Outliers)

**Input**: `bubble_mean_values = [10, 50, 55, 60, 250]`

**Constraints**:
```python
# sorted = [10, 50, 55, 60, 250]
# ls = 1, skip first (10) and last (250)

# Jump calculations:
# i=1: jump = sorted[2] - sorted[0] = 55 - 10 = 45
# i=2: jump = sorted[3] - sorted[1] = 60 - 50 = 10
# i=3: jump = sorted[4] - sorted[2] = 250 - 55 = 195 ✓

# Looseness DOES NOT fully protect against outliers!
# The extreme value (250) still affects jump calculation at i=3

# Result:
max_jump == 195  # Influenced by outlier
threshold ≈ 55 + 195/2 = 152.5  # Skewed by outlier
confidence == 1.0  # Very confident, but threshold may be wrong!
```

**Limitation**: Looseness parameter only skips extremes from loop bounds, but they're still in the sorted array and affect jump calculations.

**Better Outlier Handling**: Would require statistical outlier removal (e.g., remove values > 2σ from mean) before sorting. Not currently implemented in GlobalThresholdStrategy.

---

## Performance Constraints

### Time Complexity

**Complexity**: O(n log n)

**Breakdown**:
```python
sorted_values = sorted(bubble_mean_values)  # O(n log n)
for i in range(ls, total_bubbles_loose):    # O(n)
    # Constant time operations
```

**Dominant Factor**: Sorting

**Time Estimates**:
```
n=50:   ~0.1ms
n=100:  ~0.2ms
n=500:  ~1ms
n=1000: ~2ms
```

**Constraint**: Must complete in < 10ms for typical sheet (500 bubbles)

### Space Complexity

**Complexity**: O(n)

**Memory Usage**:
```python
sorted_values = sorted(bubble_mean_values)  # Copy of input: n floats
# Python float = 8 bytes
# sorted_values ≈ 8n bytes

metadata = {
    "num_bubbles": ...,  # int: ~28 bytes
    "min_value": ...,    # float: ~24 bytes
    "max_value": ...     # float: ~24 bytes
}
# metadata ≈ 100 bytes

ThresholdResult object ≈ 150 bytes

# Total: ~8n + 250 bytes
```

**Examples**:
```
n=50:   ~650 bytes
n=100:  ~1.0 KB
n=500:  ~4.3 KB
n=1000: ~8.3 KB
```

**Constraint**: Must use < 1MB memory (plenty of headroom)

### Browser Constraint

**JavaScript Number**: IEEE 754 double precision
- Same as Python float
- Range: ±1.7e308
- Precision: ~15 decimal digits

**Constraint**: Grayscale values (0-255) are well within precision range. No precision issues expected.

---

## Concurrency Constraints

### Thread Safety

**Status**: Fully thread-safe

**Reasoning**:
```python
class GlobalThresholdStrategy:
    # No instance variables
    # No shared state

    def calculate_threshold(self, bubble_mean_values, config):
        # All data is local to this function call
        # No mutation of input parameters
        # Creates new sorted list
        pass
```

**Constraint**: Multiple threads can call `calculate_threshold()` concurrently on the same instance without issues.

**Browser Implication**: Can use in Web Workers for parallel processing of multiple images.

---

## Validation Constraints

### Input Validation

**Required Checks**:
```python
# Type check (not enforced in Python, but should be in TypeScript)
assert isinstance(bubble_mean_values, list)
assert all(isinstance(x, (int, float)) for x in bubble_mean_values)

# Finite check
assert all(math.isfinite(x) for x in bubble_mean_values)

# Range check (recommended but not enforced)
assert all(0 <= x <= 255 for x in bubble_mean_values)
```

**Current Implementation**: No explicit validation (relies on Python duck typing)

**Recommendation for Browser**: Add TypeScript type guards

### Config Validation

**Required Checks**:
```python
assert config.min_jump > 0
assert config.default_threshold > 0
assert config.default_threshold < 255
```

**Current Implementation**: No validation in strategy (assumes config is valid)

**Recommendation**: Validate config at creation time, not at every strategy call

---

## Error Handling Constraints

### No Exceptions

**Guarantee**: `calculate_threshold()` never raises exceptions for valid input

**Handling**:
```python
# Invalid input length: Returns fallback
if len(bubble_mean_values) < 2:
    return ThresholdResult(...)  # No exception

# No good threshold: Returns default
if max_jump < config.min_jump:
    # Still returns valid result with fallback flag
    return ThresholdResult(threshold_value=default, fallback_used=True)
```

**Constraint**: Always returns a valid ThresholdResult, never None or exception.

### Invalid Input

**Undefined Behavior**:
```python
# NaN in input
bubble_mean_values = [50, float('nan'), 150]
# Sorting behavior with NaN is undefined
# May place NaN at beginning or end
# Jump calculation may produce NaN

# Infinity in input
bubble_mean_values = [50, float('inf'), 150]
# Sorting works, but jump may be infinite
# Confidence may be infinite (then capped to 1.0)
```

**Recommendation**: Validate input to exclude NaN/Inf before calling strategy.

---

## Determinism Constraints

### Fully Deterministic

**Guarantee**: Same input always produces same output

**No Randomness**:
- No random number generation
- No time-based logic
- No external state

**Sorting Stability**:
```python
# Python's sorted() is stable (preserves relative order of equal elements)
# But for our use case, stability doesn't matter
# (we don't care about order of equal values)

bubble_mean_values = [100, 100, 50]
# Could sort to [50, 100, 100] or [50, 100, 100]
# Both are valid, both produce same result
```

**Constraint**: Perfect reproducibility for testing

---

## Summary of Critical Constraints

| Constraint | Value/Rule | Impact |
|------------|-----------|---------|
| Min input length | 0 (allowed) | < 2 bubbles → fallback |
| Looseness | 1 (hardcoded) | Skip first/last values |
| Min bubbles for jump | 3 | Need at least 3 for meaningful calculation |
| Confidence range | [0.0, 1.0] | Strictly bounded |
| max_jump initialization | config.min_jump | Never goes below |
| Time complexity | O(n log n) | Dominated by sorting |
| Space complexity | O(n) | Linear in bubble count |
| Thread safety | Yes | No shared state |
| Determinism | Yes | No randomness |
| Exception handling | Never throws | Always returns valid result |

---

## Related Constraints

- **Threshold Strategy Pattern**: `../constraints.md`
- **Local Threshold**: `../local/constraints.md`
- **Adaptive Threshold**: `../adaptive/constraints.md`
