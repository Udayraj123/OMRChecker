# Local Threshold Strategy Constraints

**Module**: Domain - Threshold - Local
**Python Reference**: `src/processors/threshold/local_threshold.py`
**Last Updated**: 2026-02-21

---

## Input Constraints

### bubble_mean_values

**Type**: `list[float]`
**Source**: Bubble mean intensities from a single field
**Range**: 0.0 to 255.0 (grayscale)

**Valid Inputs**:
```python
[]                          # Empty - valid, uses fallback
[100.0]                     # Single value - valid, uses fallback
[50.0, 150.0]               # Two values - valid, special handling
[45, 50, 55, 145, 150]      # Typical - optimal (3+ bubbles)
```

**Constraints**:
- Minimum length: 0 (empty list allowed)
- Maximum length: Typically 2-20 bubbles per field
- Value range: 0.0-255.0 (not strictly enforced, but expected)
- Values must be finite (no NaN, no Inf)
- Duplicates allowed
- Order doesn't matter (internally sorted)

**Typical Size**:
```
Single-select field (4 options): 4 bubbles
Multiple-choice field (5 options): 5 bubbles
Grid field (10 rows × 2 cols): 20 bubbles
```

**Critical Difference from Global Strategy**:
- Local uses bubbles from ONE field only
- Global uses bubbles from ENTIRE sheet (50-500 bubbles)
- Local threshold is field-specific, adapts to local lighting

---

## Algorithm Constraints

### global_fallback Parameter

**Type**: `float | None`
**Purpose**: Fallback threshold when local calculation has low confidence
**Default**: `None` (uses config.default_threshold)

**Initialization**:
```python
fallback_threshold = self.global_fallback or config.default_threshold

# Examples:
# If global_fallback = 96.0 → use 96.0
# If global_fallback = None → use 127.5 (default_threshold)
```

**Constraint**:
- If provided, must be in range [0.0, 255.0]
- Typically set to the global threshold result
- Used as safety net when field has insufficient contrast

**Recommended Usage**:
```python
# Step 1: Calculate global threshold for entire sheet
global_strategy = GlobalThresholdStrategy()
global_result = global_strategy.calculate_threshold(all_bubbles, config)

# Step 2: Use global threshold as fallback for local calculations
local_strategy = LocalThresholdStrategy(global_fallback=global_result.threshold_value)
```

### Two-Bubble Gap Threshold

**Parameter**: `config.min_gap_two_bubbles`
**Type**: `float`
**Default**: 20.0

**Purpose**: Minimum gap required between two bubbles to consider them distinct groups

**Logic**:
```python
if len(bubbles) == 2:
    gap = sorted_values[1] - sorted_values[0]
    if gap < config.min_gap_two_bubbles:
        # Gap too small, use fallback
        return fallback_threshold
    else:
        # Gap significant, use mean
        return mean(sorted_values)
```

**Constraint**:
- Must be > 0
- Typical range: 15.0 to 30.0
- Higher values = more conservative (fewer local thresholds)
- Lower values = more aggressive (more local thresholds)

**Edge Cases**:
```python
# Example 1: Small gap (< 20)
bubbles = [100, 110]  # gap = 10
result → use fallback_threshold (global)

# Example 2: Large gap (≥ 20)
bubbles = [50, 150]  # gap = 100
result → use mean = 100.0
```

### Confidence Jump Threshold

**Formula**: `confident_jump = config.min_jump + config.min_jump_surplus_for_global_fallback`

**Components**:
- `config.min_jump`: Minimum significant jump (default 30.0)
- `config.min_jump_surplus_for_global_fallback`: Additional buffer (default 10.0)
- `confident_jump`: Combined threshold (default 40.0)

**Purpose**: Require local jump to be significantly larger than min_jump to override global

**Logic**:
```python
if max_jump < confident_jump:
    # Not confident enough in local threshold
    return fallback_threshold
else:
    # Confident in local threshold
    return calculated_threshold
```

**Constraint**:
- confident_jump = 30.0 + 10.0 = 40.0 (typical)
- More conservative than global strategy (which uses min_jump * 3 = 90.0)
- Local strategy requires smaller jump because field has fewer bubbles

**Comparison**:
```
Global Strategy:
- Requires max_jump ≥ 90 for high confidence
- Uses all bubbles from sheet

Local Strategy:
- Requires max_jump ≥ 40 for high confidence
- Uses only bubbles from single field
- Lower threshold compensates for smaller sample size
```

### Jump Calculation Window

**Formula**: `jump = sorted_values[i + 1] - sorted_values[i - 1]`

**Window Size**: 3 values (i-1, i, i+1)

**Range**: `for i in range(1, len(sorted_values) - 1)`

**Example**:
```
sorted_values = [45, 50, 55, 145, 150]
                     ^   ^   ^
                   i-1  i  i+1

i=1: jump = sorted[2] - sorted[0] = 55 - 45 = 10
i=2: jump = sorted[3] - sorted[1] = 145 - 50 = 95 ✓ MAX
i=3: jump = sorted[4] - sorted[2] = 150 - 55 = 95
```

**Constraint**:
- Requires at least 3 bubbles for meaningful calculation
- Skips first and last indices (need i-1 and i+1)
- No looseness parameter (unlike global strategy)

**Critical Difference from Global**:
```
Global Strategy:
- Applies looseness parameter (skip extremes)
- ls = 1, skips first and last values from loop bounds

Local Strategy:
- No looseness parameter
- Uses ALL values in field
- Simpler logic, no outlier protection
```

---

## Output Constraints

### threshold_value

**Type**: `float`
**Range**: Typically 0.0 to 255.0

**Possible Values**:
```python
# Fallback (multiple cases)
threshold = fallback_threshold  # global_fallback or config.default_threshold

# Two bubbles with large gap
threshold = np.mean([bubble1, bubble2])

# 3+ bubbles with confident jump
threshold = sorted_values[i - 1] + max_jump / 2
```

**Constraint**:
- Always returns a valid float
- Never None or NaN
- If calculated from local data, falls within [min_bubble + jump/2, max_bubble - jump/2]
- If fallback, equals fallback_threshold

**Examples**:
```python
# Fallback cases
empty field → fallback_threshold
single bubble → fallback_threshold
two bubbles, small gap → fallback_threshold
3+ bubbles, low confidence → fallback_threshold

# Calculated cases
two bubbles, large gap → mean(bubbles)
3+ bubbles, high confidence → local threshold
```

### confidence

**Type**: `float`
**Range**: 0.0 to 1.0 (strictly bounded)

**Formula by Case**:
```python
# Case 1: < 2 bubbles
confidence = 0.0

# Case 2: Exactly 2 bubbles, small gap
confidence = 0.3

# Case 3: Exactly 2 bubbles, large gap
confidence = 0.7

# Case 4: 3+ bubbles, low confidence jump
confidence = 0.4

# Case 5: 3+ bubbles, high confidence jump
confidence = min(1.0, max_jump / (confident_jump * 2))
#          = min(1.0, max_jump / 80.0)
```

**Confidence Levels**:
```
Confidence | Meaning               | Use Case
-----------|----------------------|---------------------------
0.0        | No data              | Empty or single bubble
0.3        | Low                  | Two bubbles, small gap
0.4        | Low-medium           | Low confidence jump
0.7        | Medium-high          | Two bubbles, large gap
0.8-1.0    | High                 | Confident local threshold
```

**Confidence Examples**:
```python
max_jump=80  → 80/80 = 1.0 ✓ capped
max_jump=60  → 60/80 = 0.75
max_jump=40  → 40/80 = 0.50 (at confident_jump)
max_jump=30  → 30/80 = 0.38 (below confident_jump, uses fallback with conf=0.4)
```

**Constraint**:
- Always in [0.0, 1.0]
- Distinct confidence values for each case (see formula)
- Higher confidence = more trust in local threshold
- Lower confidence = relied on fallback

### max_jump

**Type**: `float`
**Range**: 0.0 to 255.0

**Initialization**: `max_jump = 0.0` (unlike global strategy!)

**Update Rule**:
```python
for i in range(1, len(sorted_values) - 1):
    jump = sorted_values[i + 1] - sorted_values[i - 1]
    if jump > max_jump:
        max_jump = jump
```

**Constraint**:
- Minimum value: 0.0 (can stay at initial value)
- Maximum value: max(bubbles) - min(bubbles) (full range)
- Reports actual maximum jump found (no artificial minimum like global)

**Critical Difference from Global**:
```
Global Strategy:
- max_jump = config.min_jump (initial value 30.0)
- Never goes below min_jump

Local Strategy:
- max_jump = 0.0 (initial value)
- Can stay at 0.0 if no jumps found
- Reports true maximum
```

**Edge Cases**:
```python
# All same value
bubbles = [100, 100, 100]
max_jump = 0.0  # True maximum

# Two bubbles (loop doesn't run)
bubbles = [50, 150]
max_jump = 150 - 50 = 100  # Reported in metadata, but handled separately

# Empty field
bubbles = []
max_jump = 0.0  # No data
```

### method_used

**Type**: `str`
**Possible Values**: Exactly 5 values

```python
"local_single_bubble_fallback"              # < 2 bubbles
"local_two_bubbles_small_gap_fallback"      # 2 bubbles, gap < min_gap
"local_two_bubbles_mean"                    # 2 bubbles, gap ≥ min_gap
"local_low_confidence_global_fallback"      # 3+ bubbles, jump < confident_jump
"local_max_jump"                            # 3+ bubbles, jump ≥ confident_jump
```

**Constraint**:
- Always one of these five strings
- Never None or empty
- Used for debugging/monitoring
- Indicates which code path was taken

**Decision Tree**:
```
< 2 bubbles → "local_single_bubble_fallback"
2 bubbles:
  gap < 20 → "local_two_bubbles_small_gap_fallback"
  gap ≥ 20 → "local_two_bubbles_mean"
3+ bubbles:
  max_jump < 40 → "local_low_confidence_global_fallback"
  max_jump ≥ 40 → "local_max_jump"
```

### fallback_used

**Type**: `bool`
**Values**: True or False

**Logic by Case**:
```python
# Case 1: < 2 bubbles
fallback_used = True

# Case 2: 2 bubbles, small gap
fallback_used = True

# Case 3: 2 bubbles, large gap
fallback_used = False  # Used mean, not fallback

# Case 4: 3+ bubbles, low confidence
fallback_used = True

# Case 5: 3+ bubbles, high confidence
fallback_used = False
```

**Constraint**:
- True when using fallback_threshold (global or default)
- False when using locally calculated threshold
- Correlates with confidence levels

**Correlation Table**:
```
fallback_used | method_used                              | confidence
--------------|------------------------------------------|------------
True          | local_single_bubble_fallback             | 0.0
True          | local_two_bubbles_small_gap_fallback     | 0.3
False         | local_two_bubbles_mean                   | 0.7
True          | local_low_confidence_global_fallback     | 0.4
False         | local_max_jump                           | 0.5-1.0
```

### metadata

**Type**: `dict`
**Keys**: Varies by case

**Case 1: < 2 bubbles**:
```python
metadata = {}  # Empty
```

**Case 2: 2 bubbles (either path)**:
```python
metadata = {}  # Empty
```

**Case 3: 3+ bubbles, low confidence**:
```python
metadata = {
    "local_threshold": float  # What we would have used
}
```

**Case 4: 3+ bubbles, high confidence**:
```python
metadata = {
    "num_bubbles": int  # Number of bubbles in field
}
```

**Constraint**:
- Always a dict (never None)
- Keys are optional, dict may be empty
- `local_threshold` only present when fallback_used=True for 3+ bubbles
- `num_bubbles` only present when fallback_used=False for 3+ bubbles

---

## Configuration Constraints

### min_gap_two_bubbles

**Type**: `float`
**Default**: 20.0
**Valid Range**: 5.0 to 50.0

**Constraints**:
```python
config.min_gap_two_bubbles > 0  # Must be positive
config.min_gap_two_bubbles < 255  # Less than full grayscale range
```

**Recommended Range**: 15.0 to 30.0

**Impact on Results**:
```
min_gap_two_bubbles=15:
- More sensitive, accepts smaller gaps
- More likely to use local threshold for 2-bubble fields
- Risk: May incorrectly classify noise as distinct groups

min_gap_two_bubbles=20 (default):
- Balanced sensitivity
- Good for typical OMR sheets

min_gap_two_bubbles=30:
- Less sensitive, requires larger gaps
- More likely to use fallback for 2-bubble fields
- Safer but may miss legitimate distinctions
```

### min_jump

**Type**: `float`
**Default**: 30.0
**Valid Range**: 10.0 to 100.0

**Usage**: Part of confident_jump calculation

**Constraint**:
```python
config.min_jump > 0
```

**Impact**: Shared with global strategy, affects both

### min_jump_surplus_for_global_fallback

**Type**: `float`
**Default**: 10.0
**Valid Range**: 0.0 to 50.0

**Purpose**: Extra buffer required for local threshold to override global

**Usage**: `confident_jump = min_jump + min_jump_surplus_for_global_fallback`

**Constraint**:
```python
config.min_jump_surplus_for_global_fallback >= 0
```

**Impact**:
```
min_jump_surplus_for_global_fallback=0:
- confident_jump = 30.0
- Easier for local to override global
- More local thresholds used

min_jump_surplus_for_global_fallback=10 (default):
- confident_jump = 40.0
- Balanced approach

min_jump_surplus_for_global_fallback=20:
- confident_jump = 50.0
- Harder for local to override global
- More global fallbacks used
```

### default_threshold

**Type**: `float`
**Default**: 127.5
**Valid Range**: 0.0 to 255.0

**Usage**: Fallback when no global_fallback provided

**Constraint**:
```python
0 < config.default_threshold < 255
```

**Recommended Value**: 127.5 (midpoint)

**When Used**:
```python
# If global_fallback is None
fallback_threshold = config.default_threshold

# If global_fallback is provided
fallback_threshold = self.global_fallback  # Ignore default_threshold
```

---

## Edge Case Constraints

### Edge Case 1: Empty Field

**Input**: `bubble_mean_values = []`

**Constraints**:
```python
threshold_value == fallback_threshold
confidence == 0.0
max_jump == 0.0
method_used == "local_single_bubble_fallback"
fallback_used == True
metadata == {}
```

**Reason**: Cannot determine filled vs empty with no data

### Edge Case 2: Single Bubble

**Input**: `bubble_mean_values = [100.0]`

**Constraints**:
```python
# Same as empty field
threshold_value == fallback_threshold
confidence == 0.0
max_jump == 0.0
method_used == "local_single_bubble_fallback"
fallback_used == True
metadata == {}
```

**Reason**: Cannot determine filled vs empty with single sample

### Edge Case 3: Two Bubbles - Small Gap

**Input**:
```python
bubble_mean_values = [100, 110]
config.min_gap_two_bubbles = 20.0
```

**Constraints**:
```python
gap = 110 - 100 = 10
gap < 20 → True

threshold_value == fallback_threshold
confidence == 0.3
max_jump == 10.0  # Actual gap, not 0
method_used == "local_two_bubbles_small_gap_fallback"
fallback_used == True
metadata == {}
```

**Reason**: Gap too small, likely just variation within same group (both empty or both filled)

### Edge Case 4: Two Bubbles - Large Gap

**Input**:
```python
bubble_mean_values = [50, 150]
config.min_gap_two_bubbles = 20.0
```

**Constraints**:
```python
gap = 150 - 50 = 100
gap >= 20 → True

threshold_value == np.mean([50, 150]) == 100.0
confidence == 0.7
max_jump == 100.0
method_used == "local_two_bubbles_mean"
fallback_used == False
metadata == {}
```

**Reason**: Large gap indicates real filled/empty distinction

**Special Note**: This is the ONLY case where local strategy uses mean without checking confident_jump

### Edge Case 5: Three Bubbles - Insufficient Jump

**Input**:
```python
bubble_mean_values = [45, 50, 55]
config.min_jump = 30.0
config.min_jump_surplus_for_global_fallback = 10.0
```

**Constraints**:
```python
sorted = [45, 50, 55]
i=1: jump = 55 - 45 = 10
max_jump = 10

confident_jump = 40
max_jump < confident_jump → 10 < 40 → True

threshold_value == fallback_threshold
confidence == 0.4
max_jump == 10.0
method_used == "local_low_confidence_global_fallback"
fallback_used == True
metadata == {"local_threshold": ~50.0}  # What would have been used
```

**Reason**: All bubbles appear similar (all empty or all filled), no clear distinction

### Edge Case 6: All Same Value

**Input**: `bubble_mean_values = [100, 100, 100, 100]`

**Constraints**:
```python
sorted = [100, 100, 100, 100]
All jumps = 0
max_jump = 0

confident_jump = 40
max_jump < confident_jump → 0 < 40 → True

threshold_value == fallback_threshold
confidence == 0.4
max_jump == 0.0
method_used == "local_low_confidence_global_fallback"
fallback_used == True
metadata == {"local_threshold": fallback_threshold}  # Not updated
```

**Reason**: No variance in bubble intensities

### Edge Case 7: Field with One Filled, Rest Empty

**Input**:
```python
bubble_mean_values = [45, 50, 52, 55, 150]  # 4 empty, 1 filled
config.min_jump = 30.0
config.min_jump_surplus_for_global_fallback = 10.0
```

**Constraints**:
```python
sorted = [45, 50, 52, 55, 150]
i=1: jump = 52 - 45 = 7
i=2: jump = 55 - 50 = 5
i=3: jump = 150 - 52 = 98 ✓ MAX

max_jump = 98
confident_jump = 40
max_jump >= confident_jump → 98 >= 40 → True

threshold = 52 + 98/2 = 101.0
confidence = min(1.0, 98/80) = 1.0

threshold_value == 101.0
confidence == 1.0
max_jump == 98.0
method_used == "local_max_jump"
fallback_used == False
metadata == {"num_bubbles": 5}
```

**Reason**: Clear distinction between filled and empty groups

**Critical Success Case**: This is the ideal scenario for local threshold

---

## Performance Constraints

### Time Complexity

**Complexity**: O(m log m) where m = bubbles in field

**Breakdown**:
```python
sorted_values = sorted(bubble_mean_values)  # O(m log m)

# Two bubbles: O(1)
if len(sorted_values) == 2:
    gap = sorted_values[1] - sorted_values[0]  # O(1)

# 3+ bubbles: O(m)
for i in range(1, len(sorted_values) - 1):    # O(m)
    jump = sorted_values[i + 1] - sorted_values[i - 1]  # O(1)
```

**Dominant Factor**: Sorting (O(m log m))

**Time Estimates**:
```
m=2:    ~0.01ms (special case, no sort needed)
m=4:    ~0.02ms
m=10:   ~0.05ms
m=20:   ~0.10ms
```

**Constraint**: Must complete in < 1ms per field

**Comparison to Global**:
```
Global Strategy: O(n log n) where n = 50-500 bubbles
Local Strategy:  O(m log m) where m = 2-20 bubbles

For typical sheet:
- Global: sort 200 bubbles → ~0.5ms
- Local: sort 5 bubbles × 40 fields → 40 × 0.03ms = 1.2ms total

Trade-off: Local is slower overall but more accurate
```

### Space Complexity

**Complexity**: O(m)

**Memory Usage**:
```python
sorted_values = sorted(bubble_mean_values)  # m floats = 8m bytes
ThresholdResult object ≈ 200 bytes
metadata dict ≈ 50 bytes

# Total: ~8m + 250 bytes
```

**Examples**:
```
m=2:   ~270 bytes
m=4:   ~280 bytes
m=10:  ~330 bytes
m=20:  ~410 bytes
```

**Constraint**: Must use < 1KB per field

**Total Memory for Full Sheet**:
```
40 fields × 20 bubbles/field × 8 bytes = 6.4 KB (sorted arrays)
40 fields × 250 bytes = 10 KB (results)
Total: ~16.4 KB for entire sheet

Much smaller than global strategy (single 8n array for all bubbles)
```

### Browser Constraint

**JavaScript Number**: IEEE 754 double precision
- Same as Python float
- Range: ±1.7e308
- Precision: ~15 decimal digits

**Constraint**: Grayscale values (0-255) are well within precision range. No precision issues expected.

**NumPy Mean**:
```python
# Python
threshold = float(np.mean(sorted_values))

# JavaScript (two bubbles only)
const threshold = (sortedValues[0] + sortedValues[1]) / 2;
```

**Constraint**: For two-bubble case, manual mean calculation is trivial and exact

---

## Concurrency Constraints

### Thread Safety

**Status**: Fully thread-safe

**Reasoning**:
```python
class LocalThresholdStrategy:
    def __init__(self, global_fallback=None):
        self.global_fallback = global_fallback  # Immutable after construction

    def calculate_threshold(self, bubble_mean_values, config):
        # All data is local to this function call
        # No mutation of instance variables
        # No mutation of input parameters
        # Creates new sorted list
        pass
```

**Constraint**: Multiple threads can call `calculate_threshold()` concurrently on the same instance without issues.

**Caveat**: If `self.global_fallback` is mutated after construction, not thread-safe. Recommendation: Treat instance as immutable after construction.

**Browser Implication**: Can use in Web Workers for parallel field processing

**Parallel Field Processing Pattern**:
```javascript
// Worker 1 processes fields 0-9
// Worker 2 processes fields 10-19
// Worker 3 processes fields 20-29
// Worker 4 processes fields 30-39

// Each worker has its own LocalThresholdStrategy instance
// All share same global_fallback value (passed via message)
```

---

## Validation Constraints

### Input Validation

**Required Checks**:
```python
# Type check
assert isinstance(bubble_mean_values, list)
assert all(isinstance(x, (int, float)) for x in bubble_mean_values)

# Finite check
assert all(math.isfinite(x) for x in bubble_mean_values)

# Range check (recommended)
assert all(0 <= x <= 255 for x in bubble_mean_values)
```

**Current Implementation**: No explicit validation (relies on Python duck typing)

**Recommendation for Browser**: Add TypeScript type guards

**TypeScript Interface**:
```typescript
interface LocalThresholdStrategy {
  calculateThreshold(
    bubbleMeanValues: number[],  // Enforced at compile time
    config: ThresholdConfig
  ): ThresholdResult;
}

// Runtime validation
function validateBubbleValues(values: number[]): void {
  if (!values.every(x => Number.isFinite(x))) {
    throw new Error('Bubble values must be finite');
  }
  if (!values.every(x => x >= 0 && x <= 255)) {
    console.warn('Bubble values outside expected range [0, 255]');
  }
}
```

### Config Validation

**Required Checks**:
```python
assert config.min_jump > 0
assert config.min_gap_two_bubbles > 0
assert config.min_jump_surplus_for_global_fallback >= 0
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

# Low confidence: Returns fallback with flag
if max_jump < confident_jump:
    return ThresholdResult(threshold_value=fallback, fallback_used=True)
```

**Constraint**: Always returns a valid ThresholdResult, never None or exception

### Invalid Input

**Undefined Behavior**:
```python
# NaN in input
bubble_mean_values = [50, float('nan'), 150]
# Sorting behavior with NaN is undefined
# np.mean() returns NaN
# threshold_value may be NaN (violates constraint!)

# Infinity in input
bubble_mean_values = [50, float('inf'), 150]
# Sorting works, but jump may be infinite
# np.mean() returns Inf
# confidence may be infinite (then capped to 1.0)
```

**Recommendation**: Validate input to exclude NaN/Inf before calling strategy

**Defensive Browser Implementation**:
```javascript
calculateThreshold(bubbleMeanValues, config) {
  // Filter out NaN/Inf
  const validValues = bubbleMeanValues.filter(x => Number.isFinite(x));

  if (validValues.length !== bubbleMeanValues.length) {
    console.warn(`Filtered ${bubbleMeanValues.length - validValues.length} invalid bubble values`);
  }

  // Proceed with validValues
  // ...
}
```

---

## Determinism Constraints

### Fully Deterministic

**Guarantee**: Same input always produces same output

**No Randomness**:
- No random number generation
- No time-based logic
- No external state (except immutable global_fallback)

**Sorting Stability**:
```python
# Python's sorted() is stable (preserves relative order of equal elements)
# But for our use case, stability doesn't matter

bubble_mean_values = [100, 100, 50]
# Could sort to [50, 100, 100] (order of 100s preserved)
# Both produce same result (jumps are same)
```

**Constraint**: Perfect reproducibility for testing

**NumPy Mean Determinism**:
```python
# np.mean is deterministic for same input
np.mean([50, 150]) → 100.0 (always)
```

**Constraint**: No floating-point precision issues for typical grayscale values

---

## Field-Level vs File-Level Constraints

### Scope Constraint

**Local Strategy**:
- Operates on single field only
- Bubbles from one field block → one field → one threshold

**Global Strategy**:
- Operates on entire file
- All bubbles from all fields → one threshold

**Constraint**: Local strategy requires field boundaries to be known

**Usage Pattern**:
```python
# Step 1: Global threshold for file
global_result = global_strategy.calculate_threshold(all_file_bubbles, config)

# Step 2: Local threshold for each field
local_strategy = LocalThresholdStrategy(global_fallback=global_result.threshold_value)

for field in fields:
    field_bubbles = extract_bubbles_from_field(field, image)
    field_result = local_strategy.calculate_threshold(field_bubbles, config)
    # Use field_result.threshold_value for this field only
```

**Constraint**: Cannot use local strategy before field detection

### Independence Constraint

**Guarantee**: Each field's threshold is independent

**Implication**:
```python
# Field 1: [45, 50, 145, 150] → threshold=97.5
# Field 2: [80, 85, 90, 95]   → threshold=fallback (low confidence)
# Field 3: [40, 150]          → threshold=95.0

# Each field's result doesn't affect others
# (except they all share same global_fallback)
```

**Constraint**: Changing one field's bubbles doesn't affect other fields' thresholds

**Trade-off**:
- Pro: More adaptive to per-field lighting
- Con: Each field needs sufficient bubble diversity

---

## Summary of Critical Constraints

| Constraint | Value/Rule | Impact |
|------------|-----------|---------|
| Min input length | 0 (allowed) | < 2 bubbles → fallback |
| Two-bubble gap threshold | 20.0 (default) | Filters noise in 2-bubble fields |
| Confident jump | min_jump + surplus (40.0) | Local must beat this to override global |
| Confidence range | [0.0, 1.0] | 5 distinct levels by case |
| max_jump initialization | 0.0 | Unlike global (min_jump) |
| Fallback source | global_fallback or default | Prefers global over default |
| Time complexity | O(m log m) | m = bubbles per field (2-20) |
| Space complexity | O(m) | Linear in field bubble count |
| Thread safety | Yes | Immutable global_fallback |
| Determinism | Yes | No randomness |
| Field independence | Yes | Each field calculated separately |
| Exception handling | Never throws | Always returns valid result |

---

## Comparison: Local vs Global Constraints

| Aspect | Global Strategy | Local Strategy |
|--------|----------------|----------------|
| Input size | 50-500 bubbles (entire file) | 2-20 bubbles (single field) |
| Looseness | Yes (ls=1, skip extremes) | No (use all bubbles) |
| Two-bubble handling | No special case (uses loop) | Special case with gap check |
| Confident threshold | min_jump * 3 (90) | min_jump + surplus (40) |
| max_jump init | min_jump (30) | 0.0 |
| Fallback source | default_threshold | global_fallback or default |
| Scope | File-level | Field-level |
| Confidence levels | 2 main levels | 5 distinct levels |
| method_used values | 2 values | 5 values |
| Outlier protection | Looseness (weak) | None (uses all data) |

---

## Browser Migration Constraints

### TypeScript Type Constraints

```typescript
class LocalThresholdStrategy extends ThresholdStrategy {
  private readonly globalFallback: number | null;

  constructor(globalFallback?: number) {
    super();
    this.globalFallback = globalFallback ?? null;
  }

  calculateThreshold(
    bubbleMeanValues: number[],
    config: ThresholdConfig
  ): ThresholdResult {
    // Implementation
  }
}

// Constraint: globalFallback is immutable after construction
// Ensures thread safety in Web Workers
```

### Memory Constraint

**Browser Limit**: Varies by browser and device
- Desktop Chrome: ~2GB per tab
- Mobile Safari: ~500MB per tab
- Web Worker: Shares tab's memory limit

**Local Strategy Memory**:
```
Per field: ~400 bytes (8m + 250 bytes for m=20)
40 fields: ~16 KB
100 sheets: ~1.6 MB

Well within browser limits
```

**Constraint**: No special memory management needed

### Immutability Constraint

**Recommendation**: Use immutable patterns in browser

```javascript
// Anti-pattern (mutable)
const strategy = new LocalThresholdStrategy(96.0);
strategy.globalFallback = 100.0;  // BAD: Mutation after construction

// Good pattern (immutable)
const strategy = new LocalThresholdStrategy(96.0);
// Never mutate strategy.globalFallback
// To change, create new instance
const newStrategy = new LocalThresholdStrategy(100.0);
```

**Constraint**: Treat instances as immutable for Web Worker safety

---

## Related Constraints

- **Threshold Strategy Pattern**: `../constraints.md`
- **Global Threshold**: `../global/constraints.md`
- **Adaptive Threshold**: `../adaptive/constraints.md`
- **Local Threshold Flows**: `./flows.md`
