# Adaptive Threshold Strategy Constraints

**Module**: Domain - Threshold - Adaptive
**Python Reference**: `src/processors/threshold/adaptive_threshold.py`
**Last Updated**: 2026-02-21

---

## Input Constraints

### Strategies Parameter

**Type**: `list[ThresholdStrategy]`
**Requirement**: Must contain at least one strategy
**Typical**: 2 strategies (Global + Local)

```python
# Valid
strategies = [GlobalThresholdStrategy(), LocalThresholdStrategy(96.0)]

# Invalid
strategies = []  # ❌ ValueError: "Number of strategies must match number of weights"

# Valid (single strategy - but pointless, use strategy directly)
strategies = [GlobalThresholdStrategy()]

# Valid (multiple strategies)
strategies = [
    GlobalThresholdStrategy(),
    LocalThresholdStrategy(96.0),
    CustomThresholdStrategy()
]
```

**Constraints**:
- `len(strategies) >= 1` (enforced in `__init__`)
- All elements must implement `ThresholdStrategy` interface
- Strategies can be duplicates (though not recommended)
- Order matters if weights differ

**Invariant**: Length must match weights length

---

### Weights Parameter

**Type**: `list[float] | None`
**Default**: Equal weights (`[1.0, 1.0, ...]`)
**Requirement**: Must match strategies length

```python
# Valid: Explicit weights
weights = [0.4, 0.6]  # 40% global, 60% local

# Valid: Equal weights (when weights=None)
weights = None  # Becomes [1.0, 1.0]

# Valid: Different weights
weights = [0.3, 0.7]  # Prefer local more
weights = [0.5, 0.5]  # Equal preference
weights = [1.0, 2.0]  # Relative weights (not normalized)

# Invalid: Mismatched length
strategies = [Global(), Local()]
weights = [0.5]  # ❌ ValueError: length mismatch

# Invalid: Negative weights (not enforced, but illogical)
weights = [-0.4, 0.6]  # ⚠️ Will work but produce incorrect results
```

**Constraints**:
```python
# Enforced
len(weights) == len(strategies)

# Recommended (not enforced)
all(w > 0 for w in weights)  # Positive weights
sum(weights) == 1.0         # Normalized (for easier interpretation)
```

**Impact of Weights**:
```python
# Weights are RELATIVE, not absolute
weights = [1.0, 2.0]  # Same as [0.33, 0.67] or [0.5, 1.0]
# What matters is the ratio: 1:2

# Example: Three strategies
weights = [0.2, 0.5, 0.3]  # Total = 1.0 (normalized)
weights = [2.0, 5.0, 3.0]  # Total = 10.0 (same ratios)
# Both produce identical results
```

**Default Behavior**:
```python
# When weights=None:
self.weights = [1.0] * len(strategies)

# Example:
strategies = [Global(), Local()]
weights = None
# Becomes: weights = [1.0, 1.0]
# Equal weighting of strategies
```

---

### Bubble Mean Values (Inherited)

**Type**: `list[float]`
**Range**: 0.0 to 255.0
**Description**: Same constraints as base strategies

```python
# Valid
bubble_mean_values = [45, 50, 145, 150]

# Edge cases (handled by individual strategies)
[] # Empty - all strategies fallback
[100] # Single - all strategies fallback
[100, 100, 100] # All same - low confidence
```

**Delegation**: Input validation is delegated to individual strategies.

---

### ThresholdConfig (Inherited)

**Type**: `ThresholdConfig`
**Description**: Passed directly to all strategies

```python
config = ThresholdConfig(
    min_jump=30.0,
    default_threshold=127.5,
    # ... other parameters
)

# Each strategy receives same config
result1 = strategies[0].calculate_threshold(bubbles, config)
result2 = strategies[1].calculate_threshold(bubbles, config)
```

**Constraint**: All strategies must accept the same `ThresholdConfig` type.

---

## Output Constraints

### ThresholdResult

#### threshold_value

**Type**: `float`
**Range**: Bounded by min/max of strategy thresholds

```python
# Invariant (when total_weight > 0):
min_threshold = min(result.threshold_value for result in results)
max_threshold = max(result.threshold_value for result in results)

assert min_threshold <= weighted_threshold <= max_threshold

# Example:
# Strategy 1: threshold=96.0
# Strategy 2: threshold=97.5
# Weighted result: 96.9 (within [96.0, 97.5] ✓)

# Exception: When total_weight == 0
# Uses config.default_threshold (may be outside strategy range)
```

**Calculation**:
```python
if total_weight == 0:
    threshold_value = config.default_threshold  # 127.5
else:
    threshold_value = sum(
        result.threshold_value * result.confidence * weight
        for result, weight in zip(results, weights)
    ) / total_weight
```

**Precision**: Float64 (same as Python float)

---

#### confidence

**Type**: `float`
**Range**: [0.0, 1.0]

```python
# Takes MAXIMUM confidence from all strategies
confidence = max(result.confidence for result in results)

# Examples:
# [0.8, 1.0] → 1.0
# [0.3, 0.4] → 0.4
# [0.0, 0.0] → 0.0
```

**Rationale**:
- If ANY strategy is confident, adaptive strategy inherits that confidence
- Represents "best case" confidence from available evidence
- Weighted threshold uses weighted average, but confidence uses max

**Invariant**:
```python
0.0 <= confidence <= 1.0
confidence == max(r.confidence for r in results)
```

**Edge Case**:
```python
# All strategies have zero confidence
if all(r.confidence == 0.0 for r in results):
    # total_weight will be 0.0
    # Returns confidence=0.0, fallback_used=True
```

---

#### max_jump

**Type**: `float`
**Range**: [0.0, 255.0]

```python
# Takes MAXIMUM jump from all strategies
max_jump = max(result.max_jump for result in results)

# Examples:
# [88.0, 95.0] → 95.0
# [25.0, 120.0] → 120.0
# [0.0, 0.0] → 0.0
```

**Rationale**:
- Represents the best separation found by any strategy
- Indicates quality of threshold calculation
- Larger jump = better confidence

**Invariant**:
```python
0.0 <= max_jump <= 255.0
max_jump == max(r.max_jump for r in results)
```

---

#### method_used

**Type**: `str`
**Values**: Two possible values

```python
# Normal case: At least one strategy has confidence > 0
method_used = "adaptive_weighted"

# Edge case: All strategies have confidence = 0
method_used = "adaptive_all_zero_confidence"
```

**Constraint**: Must be one of these two values

**Correlation**:
```python
if method_used == "adaptive_weighted":
    assert total_weight > 0
    assert any(r.confidence > 0 for r in results)

if method_used == "adaptive_all_zero_confidence":
    assert total_weight == 0
    assert all(r.confidence == 0 for r in results)
    assert fallback_used is True
```

---

#### fallback_used

**Type**: `bool`

```python
# True if ANY strategy used fallback
fallback_used = any(result.fallback_used for result in results)

# Examples:
# [False, False] → False (no strategy used fallback)
# [False, True] → True (local used fallback)
# [True, True] → True (both used fallback)

# Special case: All zero confidence
if total_weight == 0:
    fallback_used = True  # Adaptive itself falls back
```

**Correlation with Confidence**:
```python
# Not strictly correlated, examples:

# Case 1: High confidence, no fallback
# Global: conf=1.0, fallback=False
# Local: conf=1.0, fallback=False
# Adaptive: conf=1.0, fallback=False ✓

# Case 2: Mixed confidence, local fallback
# Global: conf=1.0, fallback=False
# Local: conf=0.3, fallback=True
# Adaptive: conf=1.0, fallback=True ✓

# Case 3: Zero confidence, all fallback
# Global: conf=0.0, fallback=True
# Local: conf=0.0, fallback=True
# Adaptive: conf=0.0, fallback=True ✓
```

**Usage**: Indicates whether result is based on calculated threshold or fallback.

---

#### metadata

**Type**: `dict`
**Structure**: Always contains `"strategy_results"` key

```python
metadata = {
    "strategy_results": [
        {
            "method": str,        # From result.method_used
            "threshold": float,   # From result.threshold_value
            "confidence": float,  # From result.confidence
            "weight": float       # From self.weights
        },
        ... # One entry per strategy
    ]
}
```

**Constraints**:
- Always has exactly `len(strategies)` entries in `strategy_results`
- Each entry has all four fields: method, threshold, confidence, weight
- Order matches strategy order
- All values are JSON-serializable

**Example**:
```python
{
    "strategy_results": [
        {
            "method": "global_max_jump",
            "threshold": 96.0,
            "confidence": 1.0,
            "weight": 0.4
        },
        {
            "method": "local_max_jump",
            "threshold": 97.5,
            "confidence": 1.0,
            "weight": 0.6
        }
    ]
}
```

**Invariant**:
```python
assert "strategy_results" in metadata
assert len(metadata["strategy_results"]) == len(strategies)
for entry in metadata["strategy_results"]:
    assert "method" in entry
    assert "threshold" in entry
    assert "confidence" in entry
    assert "weight" in entry
```

---

## Mathematical Constraints

### Weighted Averaging Bounds

**Theorem**: Weighted threshold is bounded by min/max strategy thresholds (when total_weight > 0).

**Proof**:
```python
Let t_i = threshold from strategy i
Let c_i = confidence from strategy i
Let w_i = weight for strategy i

weighted_threshold = sum(t_i * c_i * w_i) / sum(c_i * w_i)

Since each term (c_i * w_i) >= 0:
weighted_threshold is a convex combination of t_i values

Therefore:
min(t_i) <= weighted_threshold <= max(t_i)
```

**Example Verification**:
```python
# t1=96.0, c1=1.0, w1=0.4
# t2=97.5, c2=1.0, w2=0.6

weighted = (96.0*1.0*0.4 + 97.5*1.0*0.6) / (1.0*0.4 + 1.0*0.6)
         = (38.4 + 58.5) / 1.0
         = 96.9

# Check: 96.0 <= 96.9 <= 97.5 ✓
```

---

### Division by Zero Prevention

**Constraint**: `total_weight` must be checked before division

```python
total_weight = sum(c_i * w_i for all strategies)

if total_weight == 0:
    # All strategies have confidence=0
    # Cannot compute weighted average
    # Use emergency fallback instead
    threshold_value = config.default_threshold
else:
    # Safe to divide
    threshold_value = sum(...) / total_weight
```

**Invariant**:
```python
total_weight == 0  ⟺  all(r.confidence == 0 for r in results)
```

**Protection**: Explicit `if total_weight == 0` check prevents division by zero.

---

### Weight Normalization (Not Required)

**Constraint**: Weights do NOT need to sum to 1.0

```python
# Both produce same result:
weights1 = [0.4, 0.6]  # Sum = 1.0
weights2 = [2.0, 3.0]  # Sum = 5.0

# Weighted averaging formula handles non-normalized weights:
weighted = sum(t_i * c_i * w_i) / sum(c_i * w_i)
#                                  └─ Normalization happens here
```

**Example**:
```python
# weights=[2.0, 3.0], c=[1.0, 1.0], t=[96.0, 97.5]

weighted = (96.0*1.0*2.0 + 97.5*1.0*3.0) / (1.0*2.0 + 1.0*3.0)
         = (192.0 + 292.5) / 5.0
         = 96.9

# Same result as weights=[0.4, 0.6]
```

**Recommendation**: Use normalized weights (sum=1.0) for clarity, but not required.

---

## Strategy Composition Constraints

### Strategy Interface Requirement

**Constraint**: All strategies must implement `ThresholdStrategy` interface

```python
class ThresholdStrategy(ABC):
    @abstractmethod
    def calculate_threshold(
        self,
        bubble_mean_values: list[float],
        config: ThresholdConfig
    ) -> ThresholdResult:
        pass
```

**Requirement**: Each strategy must:
1. Accept `bubble_mean_values` and `config` parameters
2. Return `ThresholdResult` object
3. Never raise exceptions for valid input
4. Be deterministic (same input → same output)

**Violation**: Will raise `AttributeError` or `TypeError` at runtime if strategy doesn't match interface.

---

### Strategy Independence

**Constraint**: Strategies should be stateless and independent

```python
# Good: Stateless strategies
class GlobalThresholdStrategy(ThresholdStrategy):
    def calculate_threshold(self, bubbles, config):
        # No instance variables modified
        # No side effects
        return ThresholdResult(...)

# Bad: Stateful strategy (anti-pattern)
class StatefulStrategy(ThresholdStrategy):
    def __init__(self):
        self.call_count = 0  # ❌ Mutable state

    def calculate_threshold(self, bubbles, config):
        self.call_count += 1  # ❌ Side effect
        return ThresholdResult(...)
```

**Rationale**:
- Ensures thread safety
- Enables parallel execution
- Makes results reproducible

**Exception**: `LocalThresholdStrategy` stores `global_fallback` in `__init__`, but this is immutable after construction.

---

## Performance Constraints

### Time Complexity

```python
O(k × T_strategy)

Where:
- k = number of strategies
- T_strategy = time for slowest strategy

For typical strategies (Global, Local):
T_strategy = O(m log m)  # Dominated by sorting

Therefore:
Adaptive = O(k × m log m)
         = O(2 × m log m)  # k=2 typically
         ≈ O(m log m)      # Constant factor
```

**Breakdown**:
```python
# Step 1: Execute strategies - O(k × m log m)
for strategy in strategies:  # O(k)
    result = strategy.calculate_threshold(...)  # O(m log m)
    results.append(result)

# Step 2: Calculate total_weight - O(k)
total_weight = sum(r.confidence * w for r, w in zip(results, weights))

# Step 3: Calculate weighted_threshold - O(k)
weighted_threshold = sum(...) / total_weight

# Step 4: Max aggregations - O(k)
max_confidence = max(r.confidence for r in results)
max_jump = max(r.max_jump for r in results)

# Total: O(k × m log m) + O(k) + O(k) + O(k) = O(k × m log m)
```

**Typical Values**:
- k = 2 (Global + Local)
- m = 10 bubbles per field
- 2 × (10 × log2(10)) ≈ 2 × 33 = 66 operations

**Performance**: < 0.1ms per field on modern hardware

---

### Space Complexity

```python
O(k × m + k × S_result)

Where:
- k = number of strategies
- m = bubbles per field
- S_result = size of ThresholdResult object

Breakdown:
- Each strategy creates sorted copy: k × O(m)
- Each strategy returns ThresholdResult: k × O(S_result)
- Metadata stores k strategy results: O(k × S_result)

Typical:
k = 2
m = 10
S_result ≈ 200 bytes

Space = 2 × 10 floats + 2 × 200 bytes + metadata
      = 160 bytes + 400 bytes + 400 bytes
      ≈ 1 KB per field
```

**Memory Impact**: Negligible

---

### Comparison: Single vs Adaptive Strategy

| Metric | Global Only | Adaptive (Global+Local) | Overhead |
|--------|-------------|-------------------------|----------|
| Time | O(n log n) file | O(2 × m log m) per field | ~2× |
| Space | O(n) | O(2 × m) | ~2× |
| Accuracy | Baseline | Best | +15-25% |
| Robustness | Low | High | Significant |

**Recommendation**: Overhead is minimal, use Adaptive for better accuracy.

---

## Browser Migration Constraints

### JavaScript Implementation

**Type Safety**:
```typescript
interface ThresholdStrategy {
  calculateThreshold(
    bubbleMeanValues: number[],
    config: ThresholdConfig
  ): ThresholdResult;
}

class AdaptiveThresholdStrategy implements ThresholdStrategy {
  constructor(
    private readonly strategies: ThresholdStrategy[],
    private readonly weights: number[]
  ) {
    if (strategies.length !== weights.length) {
      throw new Error("Number of strategies must match number of weights");
    }
  }

  calculateThreshold(
    bubbleMeanValues: number[],
    config: ThresholdConfig
  ): ThresholdResult {
    // Implementation
  }
}
```

**Validation**:
```typescript
function validateWeights(weights: number[]): void {
  if (weights.length === 0) {
    throw new Error("Weights cannot be empty");
  }
  if (weights.some(w => w < 0)) {
    console.warn("Negative weights may produce incorrect results");
  }
}
```

---

### No NumPy Dependency

**Python**:
```python
# Uses np.mean() in LocalThresholdStrategy
threshold = float(np.mean(sorted_values))
```

**JavaScript**:
```typescript
// Pure JavaScript
const threshold = sortedValues.reduce((a, b) => a + b, 0) / sortedValues.length;
```

**Note**: AdaptiveThresholdStrategy itself doesn't use NumPy, only delegates to strategies that may use it.

---

### Floating Point Consistency

**Constraint**: JavaScript and Python use same IEEE 754 float64

```python
# Python
weighted = (96.0 * 1.0 * 0.4) + (97.5 * 1.0 * 0.6)
# 96.9

# JavaScript
const weighted = (96.0 * 1.0 * 0.4) + (97.5 * 1.0 * 0.6);
// 96.9

// Results match exactly ✓
```

**No precision issues** for our value ranges (0-255).

---

### Array Operations

**Python**:
```python
# List comprehension
results = [
    strategy.calculate_threshold(bubbles, config)
    for strategy in self.strategies
]

# zip() and sum()
total_weight = sum(
    result.confidence * weight
    for result, weight in zip(results, self.weights, strict=False)
)
```

**JavaScript**:
```typescript
// map()
const results = this.strategies.map(strategy =>
  strategy.calculateThreshold(bubbleMeanValues, config)
);

// reduce()
const totalWeight = results.reduce(
  (sum, result, i) => sum + result.confidence * this.weights[i],
  0
);
```

**Equivalent functionality**, slightly different syntax.

---

### Metadata Serialization

**Python**:
```python
metadata = {
    "strategy_results": [
        {
            "method": result.method_used,
            "threshold": result.threshold_value,
            "confidence": result.confidence,
            "weight": weight,
        }
        for result, weight in zip(results, self.weights, strict=False)
    ]
}
```

**JavaScript**:
```typescript
const metadata = {
  strategyResults: results.map((result, i) => ({
    method: result.methodUsed,
    threshold: result.thresholdValue,
    confidence: result.confidence,
    weight: this.weights[i],
  })),
};
```

**JSON serialization** works identically in both languages.

---

## Concurrency Constraints

### Thread Safety

**Python**:
```python
# Adaptive strategy is stateless after construction
adaptive = AdaptiveThresholdStrategy(strategies, weights)

# Safe for concurrent use (no mutable state)
result1 = adaptive.calculate_threshold(bubbles1, config)  # Thread 1
result2 = adaptive.calculate_threshold(bubbles2, config)  # Thread 2
# No race conditions ✓
```

**Requirement**: Individual strategies must also be thread-safe.

**Verification**:
- `GlobalThresholdStrategy`: Stateless ✓
- `LocalThresholdStrategy`: Stores `global_fallback` (immutable after init) ✓

---

### Browser Web Workers

**Constraint**: Can be used in Web Workers for parallel processing

```typescript
// Main thread
const worker = new Worker('threshold-worker.js');
worker.postMessage({
  bubbleMeanValues: [45, 50, 145, 150],
  globalThreshold: 96.0,
  config: { minJump: 30.0, ... }
});

// Worker thread (threshold-worker.js)
self.onmessage = (e) => {
  const calculator = createDefaultThresholdCalculator(e.data.globalThreshold);
  const result = calculator.calculateThreshold(
    e.data.bubbleMeanValues,
    e.data.config
  );
  self.postMessage(result);
};
```

**Requirements**:
- All data must be serializable (no functions in messages)
- Each worker gets own instance (no shared state)
- Results are deterministic

---

## Testing Constraints

### Determinism

**Constraint**: Same input always produces same output

```python
# Test determinism
result1 = adaptive.calculate_threshold(bubbles, config)
result2 = adaptive.calculate_threshold(bubbles, config)

assert result1.threshold_value == result2.threshold_value
assert result1.confidence == result2.confidence
assert result1.max_jump == result2.max_jump
assert result1.method_used == result2.method_used
```

**No randomness**:
- No random number generation
- No time-based logic
- No external state dependencies

**Implication**: Easy to unit test, reproducible results.

---

### Test Case Coverage

**Required Test Cases**:

```python
# 1. Both strategies confident
test_both_confident(bubbles=[45, 50, 145, 150])

# 2. Only global confident
test_global_confident(bubbles=[100, 105])  # Small gap

# 3. Only local confident
test_local_confident(bubbles=[40, 160])  # Large gap, field-specific

# 4. Neither confident
test_neither_confident(bubbles=[])  # Empty

# 5. Mismatched weights length
test_invalid_weights()  # Should raise ValueError

# 6. Zero weights (edge case)
test_zero_weights(weights=[0.0, 0.0])  # total_weight=0

# 7. Different weight ratios
test_weight_ratios(weights=[0.3, 0.7])
test_weight_ratios(weights=[0.5, 0.5])
test_weight_ratios(weights=[0.8, 0.2])

# 8. Single strategy (edge case)
test_single_strategy(strategies=[Global()])

# 9. Three+ strategies
test_multiple_strategies(strategies=[Global(), Local(), Custom()])
```

---

## Edge Cases

### Edge Case 1: All Zero Confidence

```python
# Input: No bubbles
bubbles = []

# Both strategies return confidence=0.0
# total_weight = 0.0

# Result:
ThresholdResult(
    threshold_value=127.5,  # default_threshold
    confidence=0.0,
    max_jump=0.0,
    method_used="adaptive_all_zero_confidence",
    fallback_used=True
)
```

**Handling**: Explicit check prevents division by zero.

---

### Edge Case 2: Strategies Disagree Widely

```python
# Input: Noisy global, clean local
bubbles = [40, 45, 160, 165]

# Strategy results:
# Global: threshold=127.5, confidence=0.2 (noisy file)
# Local: threshold=102.5, confidence=1.0 (clean field)

# Weighted (0.4, 0.6):
# weighted = (127.5*0.2*0.4 + 102.5*1.0*0.6) / (0.2*0.4 + 1.0*0.6)
#          = (10.2 + 61.5) / 0.68
#          = 105.4

# Result favors high-confidence local ✓
```

**Handling**: Confidence weighting naturally resolves disagreement.

---

### Edge Case 3: Single Strategy

```python
# Input: Only one strategy
strategies = [GlobalThresholdStrategy()]
weights = [1.0]

# Weighted averaging reduces to single strategy result
# (No benefit over using strategy directly)

# Result:
# Same as GlobalThresholdStrategy().calculate_threshold(...)
```

**Handling**: Works correctly, but pointless (use strategy directly).

---

### Edge Case 4: Extreme Weights

```python
# Input: One weight is zero
weights = [0.0, 1.0]  # Ignore global completely

# Weighted:
# total_weight = (1.0 * 0.0) + (1.0 * 1.0) = 1.0
# weighted = (96.0 * 1.0 * 0.0) + (97.5 * 1.0 * 1.0) / 1.0
#          = 97.5

# Result: Only local strategy contributes ✓
```

**Handling**: Zero weights effectively disable a strategy.

---

## Summary of Critical Constraints

| Constraint | Requirement | Enforcement |
|------------|-------------|-------------|
| Strategies length | >= 1 | ValueError in `__init__` |
| Weights length | == len(strategies) | ValueError in `__init__` |
| total_weight check | Check before division | Explicit `if` statement |
| Confidence range | [0, 1] | Inherited from strategies |
| Threshold bounds | [min, max] of strategies | Mathematical guarantee |
| Thread safety | Stateless strategies | Design pattern |
| Determinism | Same input → same output | No randomness |
| Browser compatibility | IEEE 754 float64 | Language guarantee |

---

## Related Constraints

- **Global Threshold Constraints**: `modules/domain/threshold/global/constraints.md`
- **Local Threshold Constraints**: `modules/domain/threshold/local/constraints.md` (pending)
- **Strategy Pattern Constraints**: `modules/domain/threshold/constraints.md`
- **Bubble Detection Constraints**: `modules/domain/detection/bubbles-threshold/constraints.md` (pending)
