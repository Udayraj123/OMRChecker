# Threshold Strategy Decisions

**Module**: Domain - Threshold
**Python Reference**: `src/processors/threshold/*.py`
**Last Updated**: 2026-02-20

---

## Decision 1: Strategy Pattern vs Hardcoded Algorithm

### Context

Need flexible threshold calculation that works across varying lighting conditions, bubble counts, and image quality.

### Options Considered

**Option A: Single Hardcoded Algorithm**
- Pros: Simple, fast, fewer abstractions
- Cons: Can't adapt to different scenarios, difficult to extend

**Option B: Strategy Pattern with Multiple Implementations**
- Pros: Pluggable algorithms, easy to test, composable
- Cons: More abstraction, slightly more code

### Decision

**Chosen: Option B - Strategy Pattern**

### Rationale

1. **Different contexts need different strategies**:
   - File-level: Need global view of all bubbles
   - Field-level: Need local adaptation to specific field

2. **Research and experimentation**:
   - Easy to add new strategies without modifying existing code
   - Can A/B test different algorithms

3. **Composition**:
   - Adaptive strategy can combine multiple strategies
   - Weights can be tuned without code changes

4. **Testing**:
   - Each strategy can be unit tested independently
   - Easy to mock for integration tests

### Code Impact

```python
# Instead of:
def calculate_threshold(values):
    # Hardcoded algorithm
    pass

# We have:
class ThresholdStrategy(ABC):
    @abstractmethod
    def calculate_threshold(values, config):
        pass

class GlobalThresholdStrategy(ThresholdStrategy):
    def calculate_threshold(values, config):
        # Global algorithm
        pass

class LocalThresholdStrategy(ThresholdStrategy):
    def calculate_threshold(values, config):
        # Local algorithm
        pass
```

---

## Decision 2: Confidence Scores vs Binary Success/Failure

### Context

Threshold calculation may be uncertain (e.g., few bubbles, similar intensities). Need to communicate quality of result.

### Options Considered

**Option A: Binary Success/Failure**
- Return threshold or None/exception
- Simple boolean: success = True/False
- Pros: Simple, clear
- Cons: No gradation, difficult to choose between strategies

**Option B: Confidence Score (0.0-1.0)**
- Return threshold + confidence value
- Pros: Gradual quality metric, enables weighted averaging
- Cons: Slightly more complex

### Decision

**Chosen: Option B - Confidence Scores**

### Rationale

1. **Adaptive strategy needs weighted averaging**:
   - Can weight strategies by their confidence
   - Better result than simple majority vote

2. **Debugging and monitoring**:
   - Can track low-confidence detections
   - Alerts for systematic issues

3. **Graceful degradation**:
   - Can use low-confidence threshold with warning
   - Better than complete failure

4. **Future ML integration**:
   - Confidence scores align with ML model outputs
   - Can combine classical + ML approaches

### Code Impact

```python
@dataclass
class ThresholdResult:
    threshold_value: float
    confidence: float  # 0.0 to 1.0
    # ...

# Usage in adaptive strategy:
weighted_threshold = sum(
    result.threshold_value * result.confidence * weight
    for result, weight in zip(results, weights)
) / total_weight
```

---

## Decision 3: Global Fallback in Local Strategy

### Context

Local threshold might fail (0-1 bubbles, all same intensity). Need fallback mechanism.

### Options Considered

**Option A: Hard Default (127.5)**
- Always fallback to midpoint of 0-255
- Pros: Simple, predictable
- Cons: Ignores file-level information

**Option B: Global Threshold Fallback**
- Use file-level threshold as fallback
- Pros: Uses available information, more accurate
- Cons: Requires passing global threshold

**Option C: Raise Exception**
- Fail loudly when threshold can't be calculated
- Pros: Forces handling of edge cases
- Cons: Too strict, many legitimate edge cases

### Decision

**Chosen: Option B - Global Threshold Fallback**

### Rationale

1. **Best available information**:
   - Global threshold based on all bubbles in file
   - Better than arbitrary default

2. **Graceful degradation**:
   - Still provides reasonable result
   - Marked with low confidence + fallback flag

3. **Real-world robustness**:
   - OMR sheets may have optional fields (0-1 bubbles)
   - Better to use global than fail

4. **Transparency**:
   - Fallback flag indicates when it was used
   - Can monitor fallback rate

### Code Impact

```python
class LocalThresholdStrategy:
    def __init__(self, global_fallback: float | None = None):
        self.global_fallback = global_fallback

    def calculate_threshold(self, values, config):
        fallback = self.global_fallback or config.default_threshold

        if len(values) < 2:
            return ThresholdResult(
                threshold_value=fallback,
                confidence=0.0,
                fallback_used=True
            )
```

---

## Decision 4: Looseness Parameter in Global Strategy

### Context

Outlier bubbles (very dark/light) can skew threshold. Need to handle extreme values.

### Options Considered

**Option A: No Outlier Handling**
- Use all values as-is
- Pros: Simple
- Cons: Sensitive to outliers

**Option B: Statistical Outlier Removal**
- Remove values outside 2σ or 3σ
- Pros: Statistically sound
- Cons: May remove legitimate values, complex

**Option C: Looseness Parameter**
- Skip extreme indices when finding jumps
- Pros: Simple, configurable, preserves all data
- Cons: Less precise than statistical methods

### Decision

**Chosen: Option C - Looseness Parameter**

### Rationale

1. **Simplicity**:
   - Easy to understand and implement
   - Configurable without complex statistics

2. **Preserves data**:
   - All bubbles still considered
   - Just excludes extremes from jump calculation

3. **Historical compatibility**:
   - Matches existing OMRChecker behavior
   - Proven in production

4. **Performance**:
   - No statistical calculations needed
   - Fast execution

### Code Impact

```python
looseness = 1
ls = (looseness + 1) // 2  # ls = 1
total_bubbles_loose = len(sorted_values) - ls

# Skip first/last 'ls' values when finding jumps
for i in range(ls, total_bubbles_loose):
    jump = sorted_values[i + ls] - sorted_values[i - ls]
    # ...
```

### Example

```
Values: [10, 45, 50, 55, 145, 150, 250]
         ^                           ^
         Skip (outliers)         Skip

With looseness=1:
- Consider jumps from index 1 to len-2
- Ignores extreme values (10, 250)
```

---

## Decision 5: Two-Bubble Special Case

### Context

With exactly 2 bubbles, can't find "gap between groups". Need special handling.

### Options Considered

**Option A: Always Use Mean**
- threshold = (bubble1 + bubble2) / 2
- Pros: Simple, symmetric
- Cons: Doesn't check if gap is significant

**Option B: Always Use Global Fallback**
- Ignore local bubbles, use global
- Pros: Safe
- Cons: Ignores available data

**Option C: Conditional on Gap Size**
- If gap >= min_gap, use mean
- Else use global fallback
- Pros: Adapts to situation
- Cons: More complex

### Decision

**Chosen: Option C - Conditional on Gap Size**

### Rationale

1. **Noise handling**:
   - Small gap (< 20) might just be variation
   - Prevents false distinction

2. **Real bubble handling**:
   - Large gap likely indicates real filled/empty distinction
   - Can confidently use mean

3. **Confidence scoring**:
   - Small gap: confidence 0.3, use fallback
   - Large gap: confidence 0.7, use mean
   - Reflects actual certainty

### Code Impact

```python
if len(sorted_values) == 2:
    gap = sorted_values[1] - sorted_values[0]

    if gap < config.min_gap_two_bubbles:  # 20.0
        # Likely noise
        return ThresholdResult(
            threshold_value=fallback_threshold,
            confidence=0.3,
            method_used="local_two_bubbles_small_gap_fallback",
            fallback_used=True
        )

    # Real distinction
    return ThresholdResult(
        threshold_value=float(np.mean(sorted_values)),
        confidence=0.7,
        method_used="local_two_bubbles_mean",
        fallback_used=False
    )
```

---

## Decision 6: Adaptive Strategy Weighting

### Context

When combining global + local strategies, how much weight to give each?

### Options Considered

**Option A: Equal Weight (50/50)**
- Pros: Fair, balanced
- Cons: Doesn't reflect that local is usually better

**Option B: Fixed Unequal (40/60)**
- Global 40%, Local 60%
- Pros: Reflects reality (local usually more accurate)
- Cons: Not configurable

**Option C: Dynamic Based on Bubble Count**
- More local weight when more bubbles
- Pros: Adaptive
- Cons: Complex, unpredictable

**Option D: Configurable Weights**
- Pass weights as parameters
- Pros: Flexible, testable
- Cons: User must choose

### Decision

**Chosen: Hybrid - Option D (Configurable) with Option B (Default)**

### Rationale

1. **Best default**:
   - 40% global, 60% local works well in practice
   - Local adapts better to field-specific lighting

2. **Flexibility**:
   - Users can override for special cases
   - Research can experiment with different weights

3. **Transparency**:
   - Clear in code what weights are used
   - Metadata tracks contribution of each strategy

### Code Impact

```python
def create_default_threshold_calculator(global_threshold=None):
    return AdaptiveThresholdStrategy(
        strategies=[
            GlobalThresholdStrategy(),
            LocalThresholdStrategy(global_fallback=global_threshold)
        ],
        weights=[0.4, 0.6]  # Default: prefer local
    )

# But users can override:
custom = AdaptiveThresholdStrategy(
    strategies=[global_strat, local_strat],
    weights=[0.3, 0.7]  # Even more local preference
)
```

---

## Decision 7: Metadata Structure

### Context

Need to store additional information about threshold calculation for debugging.

### Options Considered

**Option A: No Metadata**
- Just return threshold value
- Pros: Simple
- Cons: Hard to debug

**Option B: Fixed Metadata Fields**
- Pre-defined fields in ThresholdResult
- Pros: Type-safe, documented
- Cons: Inflexible, can't extend

**Option C: Flexible Dictionary**
- metadata: dict field
- Pros: Extensible, strategy-specific data
- Cons: No type checking

### Decision

**Chosen: Option C - Flexible Dictionary**

### Rationale

1. **Strategy-specific needs**:
   - Global strategy: num_bubbles, min/max values
   - Local strategy: num_bubbles
   - Adaptive strategy: strategy_results list
   - Different needs for different strategies

2. **Future-proof**:
   - New strategies can add custom metadata
   - No need to modify base ThresholdResult

3. **Debugging**:
   - Can store arbitrary diagnostic info
   - Useful for investigating edge cases

4. **Optional**:
   - metadata defaults to {}
   - No overhead if not needed

### Code Impact

```python
@dataclass
class ThresholdResult:
    threshold_value: float
    confidence: float
    max_jump: float
    method_used: str
    fallback_used: bool = False
    metadata: dict = None  # Flexible dictionary

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

# Usage:
return ThresholdResult(
    ...,
    metadata={
        "num_bubbles": 8,
        "min_value": 45,
        "max_value": 155,
        "custom_field": "any_value"
    }
)
```

---

## Decision 8: Jump Calculation Direction

### Context

When finding the largest gap, iterate forward or backward through sorted values?

### Options Considered

**Option A: Forward Iteration (i → i+1)**
- Check adjacent pairs: sorted[i+1] - sorted[i]
- Pros: Simple
- Cons: Sensitive to single outliers

**Option B: Bidirectional Jump (i-1 → i+1)**
- Check jump across value: sorted[i+1] - sorted[i-1]
- Pros: More robust, smooths over single outliers
- Cons: Slightly more complex

### Decision

**Chosen: Option B - Bidirectional Jump**

### Rationale

1. **Robustness**:
   - Single misclassified bubble doesn't create false gap
   - Jump considers 3 values instead of 2

2. **Historical success**:
   - Original OMRChecker uses this approach
   - Proven in production

3. **Example**:
   ```
   Values: [45, 50, 150, 155]
              ^    ^

   Forward: max_jump = 150 - 50 = 100 ✓
   Bidirectional: max_jump = sorted[3] - sorted[1]
                             = 155 - 50 = 105 ✓ (even larger!)
   ```

### Code Impact

```python
# Local strategy:
for i in range(1, len(sorted_values) - 1):
    jump = sorted_values[i + 1] - sorted_values[i - 1]  # Bidirectional
    if jump > max_jump:
        max_jump = jump
        threshold = sorted_values[i - 1] + jump / 2
```

---

## Decision 9: Threshold Placement in Gap

### Context

When gap is found, where exactly to place threshold?

### Options Considered

**Option A: At Lower Edge**
- threshold = sorted[i-1]
- Pros: Conservative
- Cons: Biased toward filled

**Option B: At Upper Edge**
- threshold = sorted[i+1]
- Pros: Conservative other direction
- Cons: Biased toward empty

**Option C: At Midpoint**
- threshold = sorted[i-1] + jump/2
- Pros: Balanced, fair
- Cons: None

### Decision

**Chosen: Option C - Midpoint**

### Rationale

1. **Symmetry**:
   - Equal margin on both sides
   - Fair to filled and empty bubbles

2. **Robustness**:
   - Maximum separation from both groups
   - Tolerates variation in either direction

3. **Example**:
   ```
   Empty bubbles: [50, 52, 55]
   Filled bubbles: [145, 150, 155]

   Gap: 145 - 55 = 90
   Threshold: 55 + 90/2 = 100

   Margin from empty: 100 - 55 = 45
   Margin from filled: 145 - 100 = 45
   (Equal margins!)
   ```

### Code Impact

```python
if jump > max_jump:
    max_jump = jump
    threshold = sorted_values[i - 1] + jump / 2  # Midpoint
```

---

## Decision 10: Configuration as Dataclass vs Dictionary

### Context

Need to pass configuration parameters to strategies.

### Options Considered

**Option A: Dictionary**
- config = {"min_jump": 30.0, ...}
- Pros: Flexible, easy to serialize
- Cons: No type checking, typos possible

**Option B: Dataclass**
- @dataclass class ThresholdConfig
- Pros: Type-safe, autocomplete, documented defaults
- Cons: More rigid

### Decision

**Chosen: Option B - Dataclass**

### Rationale

1. **Type safety**:
   - IDE autocomplete
   - Catches typos at development time

2. **Documentation**:
   - Defaults are explicit
   - Docstrings for each parameter

3. **Validation**:
   - Can add __post_init__ checks
   - Ensure valid ranges

4. **Serialization**:
   - Can still convert to/from dict when needed
   - dataclasses.asdict() for JSON export

### Code Impact

```python
@dataclass
class ThresholdConfig:
    min_jump: float = 30.0
    """Minimum jump to consider significant."""

    jump_delta: float = 20.0
    """Delta between jumps for two-jump detection."""

    # ... more fields with defaults and docstrings

# Usage:
config = ThresholdConfig()  # Use defaults
config = ThresholdConfig(min_jump=40.0)  # Override specific
```

---

## Summary of Key Decisions

| Decision | Chosen Approach | Primary Rationale |
|----------|----------------|-------------------|
| Architecture | Strategy Pattern | Flexibility, composability |
| Quality Metric | Confidence Scores | Enables weighted averaging |
| Fallback | Global Threshold | Best available information |
| Outliers | Looseness Parameter | Simple, proven in production |
| Two Bubbles | Conditional on Gap | Handles noise vs real gaps |
| Weights | Configurable (40/60 default) | Flexibility with good default |
| Metadata | Flexible Dictionary | Strategy-specific extensibility |
| Jump Calculation | Bidirectional | Robustness to outliers |
| Threshold Placement | Midpoint of Gap | Symmetry, maximum margin |
| Configuration | Dataclass | Type safety, documentation |

---

## Impact on Browser Migration

### TypeScript Strategy Pattern

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

### Configuration Type Safety

```typescript
interface ThresholdConfig {
  minJump: number;
  jumpDelta: number;
  minGapTwoBubbles: number;
  // ... with JSDoc comments for documentation
}

const DEFAULT_CONFIG: ThresholdConfig = {
  minJump: 30.0,
  jumpDelta: 20.0,
  minGapTwoBubbles: 20.0,
  // ... defaults
};
```

### Metadata Flexibility

```typescript
interface ThresholdResult {
  thresholdValue: number;
  confidence: number;
  maxJump: number;
  methodUsed: string;
  fallbackUsed: boolean;
  metadata?: Record<string, unknown>; // Flexible dictionary
}
```

---

## Related Decisions

- **Bubble Detection Architecture**: `modules/domain/detection/bubbles-threshold/decisions.md`
- **Detection Pass Design**: `modules/domain/detection/bubbles-threshold/detection-pass/constraints.md`
- **Configuration Management**: `modules/foundation/configuration.md` (pending)

---

## Future Considerations

### Potential Decision Points

1. **Machine Learning Integration**:
   - Should ML-based threshold calculation be a strategy?
   - How to combine ML confidence with classical confidence?

2. **Per-Bubble Adaptive Threshold**:
   - Currently field-level, could be bubble-level
   - Trade-off: accuracy vs complexity

3. **Dynamic Weight Adjustment**:
   - Adjust adaptive weights based on bubble count?
   - Based on confidence disparity?

4. **Configuration Auto-Tuning**:
   - Learn optimal config params from historical data?
   - Would require training data and evaluation metrics

These decisions can be revisited as the system evolves and new requirements emerge.
