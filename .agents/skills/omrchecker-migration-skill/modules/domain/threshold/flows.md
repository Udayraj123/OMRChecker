# Threshold Strategy Flows

**Module**: Domain - Threshold
**Python Reference**: `src/processors/threshold/*.py`
**Last Updated**: 2026-02-20

---

## Flow 1: Global Threshold Calculation

**File**: `src/processors/threshold/global_threshold.py`
**Purpose**: Calculate file-level threshold from all bubble values

### Input

```python
bubble_mean_values: list[float]  # All bubbles from all fields [50, 55, 145, 150, ...]
config: ThresholdConfig          # Configuration parameters
```

### Processing Steps

```
START: GlobalThresholdStrategy.calculate_threshold()
│
├─► Step 1: Validate Input
│   ├─ Check bubble count
│   └─ If < 2 bubbles:
│       └─ Return ThresholdResult(
│           threshold_value=config.default_threshold,  # 127.5
│           confidence=0.0,
│           method_used="global_default",
│           fallback_used=True
│         )
│
├─► Step 2: Sort Values
│   sorted_values = sorted(bubble_mean_values)
│   Example: [45, 50, 52, 55, 140, 145, 150, 155]
│                ↑_____________↑
│              Empty bubbles  Filled bubbles
│
├─► Step 3: Calculate Looseness Parameters
│   looseness = 1                    # Configurable smoothing
│   ls = (looseness + 1) // 2        # ls = 1
│   total_bubbles_loose = len - ls   # Skip extreme values
│
├─► Step 4: Find Maximum Jump
│   max_jump = config.min_jump       # Start with minimum
│   threshold = config.default_threshold
│
│   For i in range(ls, total_bubbles_loose):
│       jump = sorted_values[i+ls] - sorted_values[i-ls]
│       │
│       │ Example iteration (i=2):
│       │ jump = sorted_values[3] - sorted_values[1]
│       │      = 55 - 50 = 5 (small jump)
│       │
│       │ Example iteration (i=3):
│       │ jump = sorted_values[4] - sorted_values[2]
│       │      = 140 - 52 = 88 (LARGE jump - filled/empty boundary!)
│       │
│       If jump > max_jump:
│           max_jump = jump
│           threshold = sorted_values[i-ls] + jump/2
│           │
│           └─ Place threshold at MIDPOINT of gap
│              threshold = 52 + 88/2 = 96
│
├─► Step 5: Calculate Confidence
│   confidence = min(1.0, max_jump / (config.min_jump * 3))
│   │
│   │ Examples:
│   │ max_jump=90, min_jump=30 → 90/(30*3) = 1.0 (capped)
│   │ max_jump=60, min_jump=30 → 60/90 = 0.67
│   │ max_jump=30, min_jump=30 → 30/90 = 0.33
│   │ max_jump=15, min_jump=30 → 15/90 = 0.17
│
└─► Step 6: Return Result
    return ThresholdResult(
        threshold_value=96.0,
        confidence=1.0,
        max_jump=88.0,
        method_used="global_max_jump",
        fallback_used=False,
        metadata={
            "num_bubbles": 8,
            "min_value": 45.0,
            "max_value": 155.0
        }
    )

END
```

### Output

```python
ThresholdResult(
    threshold_value=96.0,    # Midpoint of largest gap
    confidence=1.0,          # High confidence (large jump)
    max_jump=88.0,          # Size of filled/empty gap
    method_used="global_max_jump",
    fallback_used=False,
    metadata={"num_bubbles": 8, "min_value": 45, "max_value": 155}
)
```

---

## Flow 2: Local Threshold Calculation

**File**: `src/processors/threshold/local_threshold.py`
**Purpose**: Calculate field-level threshold from single field's bubbles

### Input

```python
bubble_mean_values: list[float]  # Bubbles from ONE field [45, 50, 145, 150]
config: ThresholdConfig          # Configuration parameters
global_fallback: float           # 96.0 from global threshold
```

### Processing Steps

```
START: LocalThresholdStrategy.calculate_threshold()
│
├─► Step 1: Initialize Fallback
│   fallback_threshold = self.global_fallback or config.default_threshold
│   Example: fallback_threshold = 96.0
│
├─► Step 2: Handle Empty/Single Bubble
│   If len(bubble_mean_values) < 2:
│       return ThresholdResult(
│           threshold_value=fallback_threshold,
│           confidence=0.0,
│           method_used="local_single_bubble_fallback",
│           fallback_used=True
│       )
│
├─► Step 3: Sort Values
│   sorted_values = sorted(bubble_mean_values)
│   Example: [45, 50, 145, 150]
│
├─► Step 4: Handle Two Bubbles (Special Case)
│   If len(sorted_values) == 2:
│       │
│       ├─ Calculate gap
│       │  gap = sorted_values[1] - sorted_values[0]
│       │  Example: gap = 150 - 45 = 105
│       │
│       ├─ Check if gap is significant
│       │  If gap < config.min_gap_two_bubbles (20.0):
│       │      │ Gap too small - might be noise
│       │      └─ return ThresholdResult(
│       │          threshold_value=fallback_threshold,
│       │          confidence=0.3,
│       │          method_used="local_two_bubbles_small_gap_fallback",
│       │          fallback_used=True
│       │        )
│       │
│       └─ Gap is significant
│          return ThresholdResult(
│              threshold_value=mean(sorted_values),  # (45+150)/2 = 97.5
│              confidence=0.7,
│              method_used="local_two_bubbles_mean",
│              fallback_used=False
│          )
│
├─► Step 5: Handle Three+ Bubbles
│   │ Example values: [45, 50, 55, 145, 150]
│   │
│   max_jump = 0.0
│   threshold = fallback_threshold
│
│   For i in range(1, len(sorted_values) - 1):
│       jump = sorted_values[i+1] - sorted_values[i-1]
│       │
│       │ i=1: jump = sorted_values[2] - sorted_values[0]
│       │           = 55 - 45 = 10
│       │
│       │ i=2: jump = sorted_values[3] - sorted_values[1]
│       │           = 145 - 50 = 95 (MAXIMUM!)
│       │
│       │ i=3: jump = sorted_values[4] - sorted_values[2]
│       │           = 150 - 55 = 95
│       │
│       If jump > max_jump:
│           max_jump = jump
│           threshold = sorted_values[i-1] + jump/2
│           │
│           └─ For i=2: threshold = 50 + 95/2 = 97.5
│
├─► Step 6: Check Confidence
│   confident_jump = config.min_jump + config.min_jump_surplus_for_global_fallback
│   confident_jump = 30.0 + 10.0 = 40.0
│
│   If max_jump < confident_jump:
│       │ Jump not large enough - use global fallback
│       └─ return ThresholdResult(
│           threshold_value=fallback_threshold,
│           confidence=0.4,
│           method_used="local_low_confidence_global_fallback",
│           fallback_used=True,
│           metadata={"local_threshold": threshold}
│         )
│
│   max_jump >= confident_jump: ✓ (95 > 40)
│
├─► Step 7: Calculate Confidence
│   confidence = min(1.0, max_jump / (confident_jump * 2))
│   confidence = min(1.0, 95 / 80) = 1.0
│
└─► Step 8: Return High-Confidence Result
    return ThresholdResult(
        threshold_value=97.5,
        confidence=1.0,
        max_jump=95.0,
        method_used="local_max_jump",
        fallback_used=False,
        metadata={"num_bubbles": 5}
    )

END
```

### Output

```python
ThresholdResult(
    threshold_value=97.5,    # Field-specific threshold
    confidence=1.0,          # High confidence
    max_jump=95.0,          # Large gap detected
    method_used="local_max_jump",
    fallback_used=False,
    metadata={"num_bubbles": 5}
)
```

---

## Flow 3: Adaptive Threshold Calculation

**File**: `src/processors/threshold/adaptive_threshold.py`
**Purpose**: Combine multiple strategies using weighted averaging

### Input

```python
bubble_mean_values: list[float]  # [45, 50, 55, 145, 150]
config: ThresholdConfig
strategies: [GlobalThresholdStrategy, LocalThresholdStrategy]
weights: [0.4, 0.6]             # Prefer local
```

### Processing Steps

```
START: AdaptiveThresholdStrategy.calculate_threshold()
│
├─► Step 1: Execute All Strategies
│   results = []
│
│   ┌─ Execute Strategy 1: GlobalThresholdStrategy
│   │  result_global = {
│   │      threshold_value: 96.0,
│   │      confidence: 1.0,
│   │      max_jump: 88.0,
│   │      method_used: "global_max_jump"
│   │  }
│   │  results.append(result_global)
│   │
│   └─ Execute Strategy 2: LocalThresholdStrategy
│      result_local = {
│          threshold_value: 97.5,
│          confidence: 1.0,
│          max_jump: 95.0,
│          method_used: "local_max_jump"
│      }
│      results.append(result_local)
│
├─► Step 2: Calculate Total Weight
│   total_weight = sum(result.confidence * weight for result, weight)
│
│   total_weight = (1.0 * 0.4) + (1.0 * 0.6)
│                = 0.4 + 0.6
│                = 1.0
│
│   If total_weight == 0:
│       └─ All strategies have zero confidence
│          return ThresholdResult(
│              threshold_value=config.default_threshold,
│              confidence=0.0,
│              method_used="adaptive_all_zero_confidence",
│              fallback_used=True
│          )
│
├─► Step 3: Calculate Weighted Threshold
│   weighted_threshold = sum(
│       result.threshold_value * result.confidence * weight
│       for result, weight in zip(results, weights)
│   ) / total_weight
│
│   weighted_threshold = (96.0 * 1.0 * 0.4) + (97.5 * 1.0 * 0.6)
│                        ───────────────────────────────────
│                                    1.0
│
│                      = 38.4 + 58.5
│                        ───────────
│                            1.0
│
│                      = 96.9
│
│   └─ Threshold is weighted average favoring local (0.6 weight)
│
├─► Step 4: Aggregate Metadata
│   max_confidence = max(result.confidence for result in results)
│                  = max(1.0, 1.0) = 1.0
│
│   max_jump_value = max(result.max_jump for result in results)
│                  = max(88.0, 95.0) = 95.0
│
│   any_fallback = any(result.fallback_used for result in results)
│                = False
│
├─► Step 5: Build Metadata
│   metadata = {
│       "strategy_results": [
│           {
│               "method": "global_max_jump",
│               "threshold": 96.0,
│               "confidence": 1.0,
│               "weight": 0.4
│           },
│           {
│               "method": "local_max_jump",
│               "threshold": 97.5,
│               "confidence": 1.0,
│               "weight": 0.6
│           }
│       ]
│   }
│
└─► Step 6: Return Combined Result
    return ThresholdResult(
        threshold_value=96.9,    # Weighted average
        confidence=1.0,          # Best confidence from any strategy
        max_jump=95.0,          # Best jump from any strategy
        method_used="adaptive_weighted",
        fallback_used=False,
        metadata={strategy_results: [...]}
    )

END
```

### Output

```python
ThresholdResult(
    threshold_value=96.9,    # Weighted: 40% global + 60% local
    confidence=1.0,          # max(global.conf, local.conf)
    max_jump=95.0,          # max(global.jump, local.jump)
    method_used="adaptive_weighted",
    fallback_used=False,
    metadata={
        "strategy_results": [
            {"method": "global_max_jump", "threshold": 96.0, ...},
            {"method": "local_max_jump", "threshold": 97.5, ...}
        ]
    }
)
```

---

## Flow 4: End-to-End Threshold Usage

**Purpose**: Show how thresholds are used in bubble detection pipeline

### Complete Flow

```
┌─────────────────────────────────────────────────────────┐
│ 1. DETECTION PASS (File-Level)                          │
└─────────────────────────────────────────────────────────┘
│
├─► Load Image & Template
│   image: numpy.ndarray (H×W grayscale)
│   template: field definitions
│
├─► Extract All Bubbles
│   For each field in template:
│       For each bubble position:
│           bubble_roi = image[y:y+h, x:x+w]
│           bubble_mean = np.mean(bubble_roi)
│           all_bubbles.append(bubble_mean)
│
│   all_bubbles = [45, 50, 52, 55, 58, 140, 145, 150, 155, 160, ...]
│                  └─────Empty bubbles─────┘  └──Filled bubbles──┘
│
├─► Calculate Global Threshold
│   global_strategy = GlobalThresholdStrategy()
│   global_result = global_strategy.calculate_threshold(
│       all_bubbles,
│       ThresholdConfig()
│   )
│
│   global_threshold = 96.0
│
│   └─ Store for use in interpretation pass
│
┌─────────────────────────────────────────────────────────┐
│ 2. INTERPRETATION PASS (Field-Level)                    │
└─────────────────────────────────────────────────────────┘
│
For each field in template:
│
├─► Get Field's Bubbles
│   field_bubbles = [45, 50, 55, 145]  # Only this field's bubbles
│
├─► Calculate Local Threshold
│   adaptive_strategy = AdaptiveThresholdStrategy(
│       strategies=[
│           GlobalThresholdStrategy(),
│           LocalThresholdStrategy(global_fallback=96.0)
│       ],
│       weights=[0.4, 0.6]
│   )
│
│   local_result = adaptive_strategy.calculate_threshold(
│       field_bubbles,
│       ThresholdConfig()
│   )
│
│   field_threshold = 96.9
│
├─► Classify Each Bubble
│   For each bubble in field:
│       if bubble_mean < field_threshold:
│           bubble_state = "FILLED" (darker = marked)
│       else:
│           bubble_state = "EMPTY" (lighter = unmarked)
│
│   Example:
│   bubble_mean=45  < 96.9 → FILLED ✓
│   bubble_mean=50  < 96.9 → FILLED ✓
│   bubble_mean=55  < 96.9 → FILLED ✓
│   bubble_mean=145 > 96.9 → EMPTY
│
├─► Generate Field Answer
│   field_value = "A,B,C"  # Bubbles 0, 1, 2 were filled
│
└─► Return Field Results
    {
        "field_label": "Q1",
        "field_value": "A,B,C",
        "threshold_used": 96.9,
        "threshold_confidence": 1.0,
        "threshold_method": "adaptive_weighted"
    }

END
```

---

## Flow 5: Factory Pattern Usage

**File**: `src/processors/threshold/adaptive_threshold.py`
**Function**: `create_default_threshold_calculator()`

### Flow

```
START: create_default_threshold_calculator(global_threshold=96.0)
│
├─► Create Strategy Instances
│   global_strategy = GlobalThresholdStrategy()
│   local_strategy = LocalThresholdStrategy(global_fallback=96.0)
│
├─► Create Adaptive Strategy
│   adaptive = AdaptiveThresholdStrategy(
│       strategies=[global_strategy, local_strategy],
│       weights=[0.4, 0.6]  # Prefer local 60%, global 40%
│   )
│
└─► Return Strategy
    return adaptive

END

USAGE:
threshold_calculator = create_default_threshold_calculator(global_threshold=96.0)
result = threshold_calculator.calculate_threshold(field_bubbles, config)
```

---

## Decision Points

### Decision 1: Use Global vs Local Strategy

```
Is this file-level or field-level threshold calculation?
│
├─► File-Level (all bubbles from image)
│   └─ Use GlobalThresholdStrategy
│       └─ Provides baseline for entire image
│
└─► Field-Level (bubbles from single field)
    └─ Use AdaptiveThresholdStrategy
        └─ Combines global baseline + local refinement
```

### Decision 2: Confidence-Based Fallback

```
Is local threshold confident?
│
├─► max_jump < confident_jump_threshold
│   └─ Low confidence: Use global fallback
│       └─ fallback_used = True
│
└─► max_jump >= confident_jump_threshold
    └─ High confidence: Use local threshold
        └─ fallback_used = False
```

### Decision 3: Two Bubbles Special Case

```
Field has exactly 2 bubbles?
│
├─► gap < min_gap_two_bubbles
│   └─ Gap too small (might be noise)
│       └─ Use global fallback (confidence=0.3)
│
└─► gap >= min_gap_two_bubbles
    └─ Gap is significant
        └─ Use mean of two values (confidence=0.7)
```

---

## Error Handling

### No Bubbles (Empty Field)

```python
if len(bubble_mean_values) == 0:
    return ThresholdResult(
        threshold_value=fallback_threshold,
        confidence=0.0,
        method_used="no_bubbles_fallback",
        fallback_used=True
    )
```

### All Strategies Fail

```python
if total_weight == 0:  # All strategies have zero confidence
    return ThresholdResult(
        threshold_value=config.default_threshold,  # 127.5
        confidence=0.0,
        method_used="adaptive_all_zero_confidence",
        fallback_used=True
    )
```

### Invalid Configuration

```python
if len(strategies) != len(weights):
    raise ValueError("Number of strategies must match number of weights")
```

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Global Strategy | O(n log n) | Dominated by sorting |
| Local Strategy | O(m log m) | m = bubbles per field |
| Adaptive Strategy | O(k × m log m) | k = number of strategies |

### Typical Values

- n (total bubbles): 100-500
- m (bubbles per field): 5-20
- k (strategies): 2

### Memory Usage

- Sorted array: O(m) for local, O(n) for global
- Result object: ~100 bytes
- Metadata: Variable (typically < 1KB)

---

## Related Flows

- **Bubble Detection**: `modules/domain/detection/bubbles-threshold/detection-pass/flows.md`
- **Bubble Interpretation**: `modules/domain/detection/bubbles-threshold/interpretation-pass/flows.md`
- **Statistics Calculation**: `modules/domain/detection/bubbles-threshold/stats/flows.md`

---

## Summary

The threshold calculation flows provide:

1. **File-Level Baseline**: Global strategy establishes overall threshold
2. **Field-Level Refinement**: Local strategy adapts to per-field conditions
3. **Adaptive Combination**: Weighted average of multiple strategies
4. **Confidence Tracking**: Each result includes confidence score
5. **Graceful Fallback**: Multiple levels of fallback for edge cases

**Recommended Flow**: Use global threshold as baseline, then adaptive strategy per field for best accuracy.
