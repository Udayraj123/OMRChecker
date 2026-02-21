# Bubble Detection Statistics Constraints

**Module**: Domain - Detection - Bubbles Threshold - Statistics
**Python Reference**: `src/processors/detection/bubbles_threshold/stats.py`, `src/processors/detection/models/detection_results.py`
**Last Updated**: 2026-02-21

---

## Input Constraints

### BubbleMeanValue

**Type**: `@dataclass`
**Fields**:

| Field | Type | Range | Constraints |
|-------|------|-------|-------------|
| `mean_value` | `float` | 0.0 - 255.0 | Grayscale intensity, lower = darker |
| `unit_bubble` | `BubblesScanBox` | N/A | Reference to bubble definition |
| `position` | `tuple[int, int]` | (0, 0) to (width, height) | (x, y) in image coordinates |

**Valid Inputs**:
```python
BubbleMeanValue(45.2, bubble_A, (10, 20))   # Typical filled bubble
BubbleMeanValue(150.8, bubble_B, (50, 20))  # Typical empty bubble
BubbleMeanValue(0.0, bubble_C, (90, 20))    # Completely black (rare)
BubbleMeanValue(255.0, bubble_D, (130, 20)) # Completely white
```

**Constraints**:
- `mean_value` must be finite (no NaN, no Inf)
- `mean_value` expected in range [0.0, 255.0] but not strictly enforced
- `unit_bubble` must not be None
- `position` must be tuple of exactly 2 integers

---

### BubbleFieldDetectionResult

**Type**: `@dataclass`
**Fields**:

| Field | Type | Constraints |
|-------|------|-------------|
| `field_id` | `str` | Non-empty string, unique identifier |
| `field_label` | `str` | Non-empty string, human-readable label |
| `bubble_means` | `list[BubbleMeanValue]` | 0 to 100+ items (typically 4-10) |
| `timestamp` | `datetime` | Auto-generated, UTC recommended |

**Valid Inputs**:
```python
# Empty field (edge case)
BubbleFieldDetectionResult("q1", "q1", [])

# Single bubble (edge case)
BubbleFieldDetectionResult("q1", "q1", [BubbleMeanValue(100, ...)])

# Typical field (4 options)
BubbleFieldDetectionResult("q1", "q1", [
    BubbleMeanValue(45, bubble_A, ...),
    BubbleMeanValue(50, bubble_B, ...),
    BubbleMeanValue(145, bubble_C, ...),
    BubbleMeanValue(150, bubble_D, ...)
])

# Large field (10 options)
BubbleFieldDetectionResult("q1", "q1", [BubbleMeanValue(...) for _ in range(10)])
```

**Constraints**:
- `bubble_means` can be empty list (edge case)
- No maximum length enforced (memory permitting)
- Typical length: 2-10 bubbles per field
- All `BubbleMeanValue` items must be valid

---

## Statistical Property Constraints

### std_deviation

**Type**: `float`
**Range**: 0.0 to ~90.0 (theoretical max ~127.5)

**Formula**:
```
std = √(Σ(xi - μ)² / n)
```

**Constraints**:
```python
# Empty list
if len(bubble_means) == 0:
    std_deviation == 0.0  # Always

# Single bubble
if len(bubble_means) == 1:
    std_deviation == 0.0  # No variance

# Two identical values
if all values are equal:
    std_deviation == 0.0  # No variance

# Maximum possible std (theoretical)
# Values: [0, 255] → std ≈ 127.5
# But typical filled/empty bubbles: [50, 150] → std ≈ 50
```

**Typical Values**:
```
Excellent scan:  50-90  (clear filled/empty separation)
Good scan:       30-50  (acceptable separation)
Acceptable scan: 15-30  (marginal separation)
Poor scan:       0-15   (no clear separation)
```

**Edge Cases**:
```python
# All empty bubbles (no marking)
bubble_means = [200, 205, 210, 208]
std ≈ 4.3  # Very low, POOR quality

# All filled bubbles (over-marking)
bubble_means = [45, 48, 50, 52]
std ≈ 2.9  # Very low, POOR quality

# Mixed (normal case)
bubble_means = [45, 50, 145, 150]
std ≈ 54.0  # High, EXCELLENT quality
```

**Browser Consideration**:
- JavaScript `Math.sqrt()` has same precision as Python
- Variance calculation may have minor floating-point differences
- Difference typically < 0.001 (negligible for quality thresholds)

---

### scan_quality

**Type**: `ScanQuality` enum
**Values**: Exactly 4 possible values

```python
class ScanQuality(str, Enum):
    EXCELLENT = "excellent"   # std > 50
    GOOD = "good"             # std > 30
    ACCEPTABLE = "acceptable" # std > 15
    POOR = "poor"             # std <= 15
```

**Mapping**:

| Std Range | Quality | Confidence Level |
|-----------|---------|------------------|
| > 50 | EXCELLENT | Very High (> 0.9) |
| 30-50 | GOOD | High (0.7-0.9) |
| 15-30 | ACCEPTABLE | Medium (0.5-0.7) |
| <= 15 | POOR | Low (< 0.5) |

**Constraints**:
- Always returns a valid enum value (never None)
- Deterministic (same std → same quality)
- Boundary cases use strict inequality:
  ```python
  std = 50.0 → GOOD (not EXCELLENT, since not > 50)
  std = 50.1 → EXCELLENT
  std = 30.0 → ACCEPTABLE (not GOOD, since not > 30)
  std = 30.1 → GOOD
  ```

**Usage Constraint**: Only use for **relative quality assessment**, not absolute accuracy. A "POOR" scan may still be readable, and an "EXCELLENT" scan may still have errors.

---

### is_reliable

**Type**: `bool`
**Values**: `True` or `False`

**Logic**:
```python
is_reliable = scan_quality in [ScanQuality.EXCELLENT, ScanQuality.GOOD]

# Equivalent to:
is_reliable = std_deviation > 30
```

**Mapping**:

| Std | Scan Quality | is_reliable |
|-----|--------------|-------------|
| 60 | EXCELLENT | True |
| 45 | GOOD | True |
| 30.1 | GOOD | True |
| 30.0 | ACCEPTABLE | False |
| 20 | ACCEPTABLE | False |
| 10 | POOR | False |

**Constraints**:
- Binary decision boundary at std = 30
- Use for **strategy selection**, not error guarantee
- `is_reliable = True` means "likely to work with local threshold"
- `is_reliable = False` means "prefer global threshold or ML detection"

**Not a Guarantee**: Even `is_reliable = True` can fail in edge cases (e.g., all bubbles marked, stray marks).

---

### sorted_bubble_means

**Type**: `list[BubbleMeanValue]`
**Length**: Same as `bubble_means`
**Order**: Ascending by `mean_value`

**Constraints**:
```python
# Creates NEW list (does not mutate original)
assert sorted_bubble_means is not bubble_means

# Sorted order
for i in range(1, len(sorted_bubble_means)):
    assert sorted_bubble_means[i].mean_value >= sorted_bubble_means[i-1].mean_value

# Same elements (just reordered)
assert set(sorted_bubble_means) == set(bubble_means)
```

**Stability**: Python's `sorted()` is stable (preserves relative order of equal values), but this doesn't matter for our use case.

**Performance**: O(n log n) - computed on-demand, not cached by default.

---

### jumps

**Type**: `list[tuple[float, BubbleMeanValue]]`
**Length**: `len(bubble_means) - 1` or 0

**Tuple Format**: `(jump_size: float, bubble_before_jump: BubbleMeanValue)`

**Constraints**:
```python
# Empty list case
if len(bubble_means) < 2:
    jumps == []  # No jumps possible

# Single bubble case
if len(bubble_means) == 1:
    jumps == []  # No jumps possible

# Two bubbles case
if len(bubble_means) == 2:
    jumps == [(jump, bubble_means[0])]  # Exactly one jump

# General case
if len(bubble_means) == n:
    len(jumps) == n - 1  # One jump between each consecutive pair
```

**Jump Size**:
```python
# Jump is always non-negative (since list is sorted)
for jump_size, _ in jumps:
    assert jump_size >= 0.0

# Jump is difference between consecutive sorted values
sorted_values = [45.0, 50.0, 140.0, 145.0]
jumps = [
    (5.0, BubbleMean(45.0)),    # 50 - 45
    (90.0, BubbleMean(50.0)),   # 140 - 50
    (5.0, BubbleMean(140.0))    # 145 - 140
]
```

**Precision**: Rounded to 2 decimal places
```python
jump = round(sorted_means[i].mean_value - sorted_means[i-1].mean_value, 2)
```

**Edge Case - All Same Value**:
```python
bubble_means = [100, 100, 100, 100]
sorted_means = [100, 100, 100, 100]
jumps = [
    (0.0, BubbleMean(100)),
    (0.0, BubbleMean(100)),
    (0.0, BubbleMean(100))
]
```

---

### max_jump

**Type**: `float`
**Range**: 0.0 to 255.0 (theoretical max)

**Constraints**:
```python
# Empty or single bubble
if len(bubble_means) < 2:
    max_jump == 0.0

# General case
if len(bubble_means) >= 2:
    max_jump == max(jump for jump, _ in jumps)

    # Bounded by value range
    max_jump <= (max_mean - min_mean)

    # Example:
    # values = [45, 50, 145, 150]
    # jumps = [5, 95, 5]
    # max_jump = 95
```

**Typical Values**:
```
Excellent separation:  > 80   (clear gap)
Good separation:       50-80  (acceptable gap)
Marginal separation:   30-50  (small gap)
Poor separation:       < 30   (no clear gap)
```

**Usage in Threshold Calculation**:
```python
# Threshold strategies use max_jump to assess confidence
if max_jump >= min_jump:  # e.g., 30
    # Found significant gap, use calculated threshold
    threshold = sorted_values[i] + max_jump / 2
else:
    # No significant gap, use default threshold
    threshold = default_threshold
```

---

### min_mean / max_mean

**Type**: `float`
**Range**: 0.0 to 255.0

**Constraints**:
```python
# Empty list defaults
if len(bubble_means) == 0:
    min_mean == 0.0    # Arbitrary minimum
    max_mean == 255.0  # Full grayscale range

# Non-empty list
if len(bubble_means) > 0:
    min_mean == min(bm.mean_value for bm in bubble_means)
    max_mean == max(bm.mean_value for bm in bubble_means)

    # Ordering constraint
    assert min_mean <= max_mean

    # Range constraint
    assert 0.0 <= min_mean <= 255.0
    assert 0.0 <= max_mean <= 255.0
```

**Value Range**:
```python
value_range = max_mean - min_mean

# Typical ranges:
# Excellent: 100-200 (wide range)
# Good:      60-100  (medium range)
# Poor:      0-60    (narrow range)
```

**Edge Cases**:
```python
# Single value
bubble_means = [100.0]
min_mean == max_mean == 100.0
value_range == 0.0

# All same
bubble_means = [100, 100, 100]
min_mean == max_mean == 100.0
value_range == 0.0
```

---

## Performance Constraints

### Time Complexity

| Property | Complexity | Computation | Caching |
|----------|------------|-------------|---------|
| `std_deviation` | O(n) | NumPy std calculation | No |
| `scan_quality` | O(n) | Depends on std_deviation | No |
| `is_reliable` | O(n) | Depends on scan_quality | No |
| `sorted_bubble_means` | O(n log n) | Sorting | No |
| `sorted_mean_values` | O(n log n) | Depends on sorted_bubble_means | No |
| `jumps` | O(n log n) | Sorting + iteration | No |
| `max_jump` | O(n log n) | Depends on jumps | No |
| `min_mean` | O(n) | Min over list | No |
| `max_mean` | O(n) | Max over list | No |

**Note**: Properties are computed **on-demand** each time accessed. No automatic caching (though implementation could add it).

**Typical Performance** (n = 4-10 bubbles):
```
All statistics combined: < 0.1ms
std_deviation alone:     < 0.01ms
sorted_bubble_means:     < 0.01ms
```

### Space Complexity

| Property | Space | Notes |
|----------|-------|-------|
| `std_deviation` | O(1) | Single float value |
| `sorted_bubble_means` | O(n) | New sorted list |
| `jumps` | O(n) | List of tuples |
| Other properties | O(1) | Computed values |

**Total Memory** (n = 10 bubbles):
```
BubbleFieldDetectionResult: ~100 bytes
bubble_means list: ~800 bytes (10 × 80 bytes)
sorted_bubble_means: ~800 bytes (if accessed)
jumps: ~900 bytes (9 tuples, if accessed)
Total: ~2.6 KB per field
```

### Browser Constraints

**JavaScript Performance**:
- Array operations slower than NumPy (no SIMD)
- Sorting performance similar (both O(n log n))
- No built-in std function, must implement manually

**Memory**:
- JavaScript numbers: 8 bytes (same as Python float)
- Objects have higher overhead than Python dataclasses
- Expect ~1.5-2x memory usage vs Python

**Optimization**:
```typescript
// Cache frequently accessed properties
class BubbleFieldDetectionResultImpl {
  private _stdDeviation?: number;
  private _sortedBubbleMeans?: BubbleMeanValue[];

  get stdDeviation(): number {
    if (this._stdDeviation === undefined) {
      this._stdDeviation = this.computeStdDeviation();
    }
    return this._stdDeviation;
  }

  private computeStdDeviation(): number {
    // Expensive calculation here
  }
}
```

---

## Edge Case Constraints

### Edge Case 1: Empty Field

**Input**: `bubble_means = []`

**Constraints**:
```python
std_deviation == 0.0
scan_quality == ScanQuality.POOR
is_reliable == False
sorted_bubble_means == []
sorted_mean_values == []
jumps == []
max_jump == 0.0
min_mean == 0.0    # Default
max_mean == 255.0  # Default
```

**Handling**: All properties return safe defaults, no exceptions.

---

### Edge Case 2: Single Bubble

**Input**: `bubble_means = [BubbleMeanValue(100, ...)]`

**Constraints**:
```python
std_deviation == 0.0  # No variance
scan_quality == ScanQuality.POOR
is_reliable == False
sorted_bubble_means == [BubbleMeanValue(100, ...)]
sorted_mean_values == [100.0]
jumps == []  # No consecutive pairs
max_jump == 0.0
min_mean == 100.0
max_mean == 100.0
```

**Usage**: Cannot calculate meaningful threshold with 1 bubble. Must use default or global fallback.

---

### Edge Case 3: All Same Value

**Input**: `bubble_means = [BubbleMeanValue(100, ...) for _ in range(4)]`

**Constraints**:
```python
std_deviation == 0.0  # No variance
scan_quality == ScanQuality.POOR
is_reliable == False
sorted_bubble_means == [all same]
jumps == [(0.0, ...), (0.0, ...), (0.0, ...)]
max_jump == 0.0
min_mean == max_mean == 100.0
```

**Meaning**: Either all bubbles filled or all empty. Cannot determine threshold. Flag for manual review.

---

### Edge Case 4: Extreme Values

**Input**: `bubble_means = [0.0, 10.0, 240.0, 255.0]`

**Constraints**:
```python
std_deviation ≈ 126.5  # Very high
scan_quality == ScanQuality.EXCELLENT  # Based on std
is_reliable == True
min_mean == 0.0
max_mean == 255.0
max_jump == 230.0  # 240 - 10

# BUT: May not represent actual filled/empty bubbles!
# Extreme values could be outliers/noise
```

**Warning**: High std doesn't guarantee correct detection. Outliers can inflate std without providing useful information.

**Recommendation**: Add outlier detection before statistics:
```python
# Remove values > 2σ from median
median = np.median(mean_values)
std = np.std(mean_values)
filtered = [v for v in mean_values if abs(v - median) <= 2 * std]
```

---

### Edge Case 5: Two Bubbles

**Input**: `bubble_means = [BubbleMeanValue(50, ...), BubbleMeanValue(150, ...)]`

**Constraints**:
```python
std_deviation ≈ 50.0
scan_quality == ScanQuality.GOOD  # std > 30 but not > 50
is_reliable == True
sorted_bubble_means == [BubbleMean(50), BubbleMean(150)]
jumps == [(100.0, BubbleMean(50))]
max_jump == 100.0
```

**Usage**: Can calculate threshold (midpoint = 100), but low confidence. Prefer global threshold with multiple bubbles.

---

## Validation Constraints

### Input Validation

**Required Checks** (TypeScript):
```typescript
function validateBubbleMeanValue(bm: BubbleMeanValue): void {
  if (!Number.isFinite(bm.meanValue)) {
    throw new Error("mean_value must be finite");
  }
  if (bm.meanValue < 0 || bm.meanValue > 255) {
    console.warn(`mean_value ${bm.meanValue} out of expected range [0, 255]`);
  }
  if (!bm.unitBubble) {
    throw new Error("unit_bubble must not be null");
  }
  if (!Array.isArray(bm.position) || bm.position.length !== 2) {
    throw new Error("position must be [x, y]");
  }
}
```

**Current Python Implementation**: No explicit validation (duck typing). Assumes input from trusted detection pass.

---

### Output Validation

**Guaranteed Properties**:
```python
# std_deviation
assert std_deviation >= 0.0
assert std_deviation < 255.0  # Theoretical max ~127.5

# scan_quality
assert scan_quality in [
    ScanQuality.EXCELLENT,
    ScanQuality.GOOD,
    ScanQuality.ACCEPTABLE,
    ScanQuality.POOR
]

# jumps
assert all(jump >= 0.0 for jump, _ in jumps)
assert len(jumps) == max(0, len(bubble_means) - 1)

# min/max
assert min_mean <= max_mean
```

---

## Determinism Constraints

### Fully Deterministic

**Guarantee**: Same input always produces same output.

**No Randomness**:
- All operations are deterministic
- No random number generation
- No time-based logic (except timestamp, which doesn't affect statistics)
- No external dependencies

**Floating-Point Precision**:
```python
# Minor differences possible between Python/JavaScript
# Due to different floating-point implementations

# Python std:      48.52345678901234
# JavaScript std:  48.52345678901236  # Difference < 1e-12

# Impact on scan_quality: NONE (thresholds have large margins)
```

**Testing**: Can use exact value assertions for unit tests (with small epsilon for float comparison).

---

## Concurrency Constraints

### Thread Safety

**Status**: Fully thread-safe

**Reasoning**:
```python
@dataclass
class BubbleFieldDetectionResult:
    # Fields are immutable after construction
    field_id: str
    field_label: str
    bubble_means: list[BubbleMeanValue]
    timestamp: datetime

    # Properties are read-only
    @property
    def std_deviation(self) -> float:
        # Computes from immutable data
        # No shared state mutation
```

**Constraint**: Safe to access properties from multiple threads concurrently, as long as `bubble_means` list is not mutated.

**Browser Implication**: Safe to use in Web Workers for parallel field processing.

---

## Browser Compatibility Constraints

### Feature Requirements

| Feature | Requirement | Compatibility |
|---------|-------------|---------------|
| JavaScript Arrays | ES5+ | All browsers |
| Array.sort() | ES5+ | All browsers |
| Array.map/reduce | ES5+ | All browsers |
| Math.sqrt() | ES5+ | All browsers |
| Getters (get keyword) | ES5+ | All browsers |
| Enums (TypeScript) | TS 2.4+ | Compile-time only |

**No Dependencies**: Statistics module has zero external dependencies (no NumPy in browser).

**Polyfills**: None required.

---

## Summary of Critical Constraints

| Constraint | Value/Rule | Impact |
|------------|-----------|---------|
| Min bubble count | 0 (allowed) | Returns safe defaults |
| Std calculation | Manual in JS | ~10% slower than NumPy |
| Quality thresholds | 50, 30, 15 | Fixed boundaries |
| Reliability threshold | std > 30 | Binary decision point |
| Jump precision | 2 decimals | Sufficient for threshold |
| Time complexity | O(n log n) | Dominated by sorting |
| Space complexity | O(n) | Linear in bubble count |
| Thread safety | Yes | No shared mutable state |
| Determinism | Yes | No randomness |
| Browser support | ES5+ | Universal compatibility |

---

## Related Constraints

- **Detection Pass**: `../detection-pass/constraints.md`
- **Interpretation Pass**: `../interpretation-pass/constraints.md`
- **Global Threshold**: `../../threshold/global/constraints.md`
- **Bubble Detection Concept**: `../concept.md`
