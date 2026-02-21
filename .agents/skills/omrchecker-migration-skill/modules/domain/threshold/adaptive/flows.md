# Adaptive Threshold Strategy Flows

**Module**: Domain - Threshold - Adaptive
**Python Reference**: `src/processors/threshold/adaptive_threshold.py`
**Last Updated**: 2026-02-21

---

## Overview

Adaptive Threshold Strategy combines multiple threshold strategies (Global and Local) using weighted averaging based on confidence scores. This provides robust threshold calculation that adapts to both file-level and field-level characteristics.

**Key Concept**: Not OpenCV's `cv2.adaptiveThreshold` (image-level), but a **meta-strategy** that adaptively combines bubble detection threshold strategies.

**Use Case**: Default threshold calculator for bubble detection that balances global baseline with local field-specific refinement.

---

## Algorithm Flow

### Main Calculation Flow

```
START: AdaptiveThresholdStrategy.calculate_threshold(bubble_mean_values, config)
│
├─► STEP 1: Validate Strategy Configuration
│   │
│   │ Performed in __init__, not during calculation:
│   │
│   if len(self.strategies) != len(self.weights):
│       raise ValueError("Number of strategies must match number of weights")
│   │
│   │ Example valid configuration:
│   │ strategies: [GlobalThresholdStrategy, LocalThresholdStrategy]
│   │ weights: [0.4, 0.6]  # Prefer local (60%) over global (40%)
│
├─► STEP 2: Execute All Strategies
│   │
│   results = []
│   │
│   ┌─► FOR EACH strategy in self.strategies:
│   │   │
│   │   └─► Execute strategy.calculate_threshold(bubble_mean_values, config)
│   │       │
│   │       │ STRATEGY 1: GlobalThresholdStrategy
│   │       ├─► Input: [45, 50, 55, 145, 150, 155]  # All bubbles from field
│   │       │   Output: ThresholdResult(
│   │       │     threshold_value: 96.0,
│   │       │     confidence: 1.0,
│   │       │     max_jump: 88.0,
│   │       │     method_used: "global_max_jump",
│   │       │     fallback_used: False
│   │       │   )
│   │       │   results.append(result_global)
│   │       │
│   │       │ STRATEGY 2: LocalThresholdStrategy (global_fallback=96.0)
│   │       └─► Input: [45, 50, 55, 145, 150, 155]  # Same field bubbles
│   │           Output: ThresholdResult(
│   │             threshold_value: 97.5,
│   │             confidence: 1.0,
│   │             max_jump: 95.0,
│   │             method_used: "local_max_jump",
│   │             fallback_used: False
│   │           )
│   │           results.append(result_local)
│   │
│   │ Now: results = [result_global, result_local]
│
├─► STEP 3: Calculate Total Weighted Confidence
│   │
│   total_weight = sum(
│       result.confidence * weight
│       for result, weight in zip(results, self.weights)
│   )
│   │
│   │ Example calculation:
│   │ total_weight = (result_global.confidence * 0.4) + (result_local.confidence * 0.6)
│   │              = (1.0 * 0.4) + (1.0 * 0.6)
│   │              = 0.4 + 0.6
│   │              = 1.0
│   │
│   ┌─► CHECK: All strategies have zero confidence?
│   │   │
│   │   if total_weight == 0:
│   │       │
│   │       └─► RETURN ThresholdResult(
│   │             threshold_value: config.default_threshold,  # 127.5
│   │             confidence: 0.0,
│   │             max_jump: 0.0,
│   │             method_used: "adaptive_all_zero_confidence",
│   │             fallback_used: True
│   │           )
│   │           END (early exit - all strategies failed)
│   │
│   └─ Else: total_weight > 0, continue to weighted averaging
│
├─► STEP 4: Calculate Weighted Threshold
│   │
│   weighted_threshold = sum(
│       result.threshold_value * result.confidence * weight
│       for result, weight in zip(results, self.weights)
│   ) / total_weight
│   │
│   │ Example calculation:
│   │
│   │ Numerator:
│   │   (96.0 * 1.0 * 0.4) + (97.5 * 1.0 * 0.6)
│   │   = 38.4 + 58.5
│   │   = 96.9
│   │
│   │ Denominator:
│   │   total_weight = 1.0
│   │
│   │ weighted_threshold = 96.9 / 1.0 = 96.9
│   │
│   │ Interpretation:
│   │ - Global suggests 96.0
│   │ - Local suggests 97.5
│   │ - Weighted average (40%/60%) = 96.9
│   │ - Result is closer to local (preferred with 0.6 weight)
│
├─► STEP 5: Aggregate Metadata from All Strategies
│   │
│   ├─► max_confidence = max(result.confidence for result in results)
│   │   │
│   │   │ Example: max(1.0, 1.0) = 1.0
│   │   │ Use the BEST confidence from any strategy
│   │
│   ├─► max_jump_value = max(result.max_jump for result in results)
│   │   │
│   │   │ Example: max(88.0, 95.0) = 95.0
│   │   │ Use the LARGEST jump found by any strategy
│   │
│   └─► any_fallback = any(result.fallback_used for result in results)
│       │
│       │ Example: any([False, False]) = False
│       │ True if ANY strategy used fallback
│
├─► STEP 6: Build Strategy Results Metadata
│   │
│   metadata = {
│       "strategy_results": [
│           {
│               "method": result.method_used,
│               "threshold": result.threshold_value,
│               "confidence": result.confidence,
│               "weight": weight
│           }
│           for result, weight in zip(results, self.weights)
│       ]
│   }
│   │
│   │ Example output:
│   │ {
│   │   "strategy_results": [
│   │     {
│   │       "method": "global_max_jump",
│   │       "threshold": 96.0,
│   │       "confidence": 1.0,
│   │       "weight": 0.4
│   │     },
│   │     {
│   │       "method": "local_max_jump",
│   │       "threshold": 97.5,
│   │       "confidence": 1.0,
│   │       "weight": 0.6
│   │     }
│   │   ]
│   │ }
│
└─► STEP 7: Return Combined Result
    │
    return ThresholdResult(
        threshold_value: weighted_threshold,  # 96.9
        confidence: max_confidence,           # 1.0
        max_jump: max_jump_value,            # 95.0
        method_used: "adaptive_weighted",
        fallback_used: any_fallback,         # False
        metadata: {"strategy_results": [...]}
    )
    │
    END

```

---

## Weighted Averaging Examples

### Example 1: Both Strategies Confident

**Scenario**: Clear filled/empty separation in both global and local view

```python
# Input
bubble_mean_values = [45, 50, 55, 145, 150, 155]
strategies = [GlobalThresholdStrategy(), LocalThresholdStrategy(96.0)]
weights = [0.4, 0.6]

# Strategy Results
result_global = ThresholdResult(
    threshold_value=96.0,
    confidence=1.0,
    max_jump=88.0,
    method_used="global_max_jump"
)

result_local = ThresholdResult(
    threshold_value=97.5,
    confidence=1.0,
    max_jump=95.0,
    method_used="local_max_jump"
)

# Weighted Calculation
total_weight = (1.0 * 0.4) + (1.0 * 0.6) = 1.0

weighted_threshold = (96.0 * 1.0 * 0.4) + (97.5 * 1.0 * 0.6)
                     ─────────────────────────────────────
                                  1.0
                   = 96.9

# Final Result
ThresholdResult(
    threshold_value=96.9,    # Balanced between global and local
    confidence=1.0,           # max(1.0, 1.0)
    max_jump=95.0,           # max(88.0, 95.0)
    method_used="adaptive_weighted"
)
```

**Interpretation**: Both strategies agree (96.0 vs 97.5 is close), high confidence result.

---

### Example 2: Local Not Confident, Global Confident

**Scenario**: Field has too few bubbles for local confidence, but global is confident

```python
# Input
bubble_mean_values = [50, 55]  # Only 2 bubbles in field
global_threshold = 96.0

# Strategy Results
result_global = ThresholdResult(
    threshold_value=96.0,
    confidence=1.0,      # Global saw many bubbles across file
    max_jump=88.0,
    method_used="global_max_jump"
)

result_local = ThresholdResult(
    threshold_value=96.0,     # Using global fallback
    confidence=0.3,           # Low confidence (only 2 bubbles, small gap)
    max_jump=5.0,
    method_used="local_two_bubbles_small_gap_fallback",
    fallback_used=True
)

# Weighted Calculation
total_weight = (1.0 * 0.4) + (0.3 * 0.6) = 0.4 + 0.18 = 0.58

weighted_threshold = (96.0 * 1.0 * 0.4) + (96.0 * 0.3 * 0.6)
                     ─────────────────────────────────────
                                  0.58
                   = (38.4 + 17.28) / 0.58
                   = 55.68 / 0.58
                   = 96.0

# Final Result
ThresholdResult(
    threshold_value=96.0,    # Dominated by global (high confidence)
    confidence=1.0,           # max(1.0, 0.3) = 1.0
    max_jump=88.0,           # max(88.0, 5.0) = 88.0
    method_used="adaptive_weighted"
)
```

**Interpretation**: Global strategy dominates due to higher confidence, adaptive strategy gracefully handles weak local evidence.

---

### Example 3: Global Not Confident, Local Confident

**Scenario**: File has noisy overall bubbles, but field has clear separation

```python
# Input
bubble_mean_values = [40, 45, 160, 165]  # Clear field-level separation
global_threshold = 127.5  # Global used default (noisy file)

# Strategy Results
result_global = ThresholdResult(
    threshold_value=127.5,   # Using default fallback
    confidence=0.2,          # Low confidence (noisy file)
    max_jump=25.0,          # Below min_jump threshold
    method_used="global_max_jump",
    fallback_used=True
)

result_local = ThresholdResult(
    threshold_value=102.5,   # Clear local separation
    confidence=1.0,          # High confidence
    max_jump=115.0,         # Large jump: 160 - 45 = 115
    method_used="local_max_jump"
)

# Weighted Calculation
total_weight = (0.2 * 0.4) + (1.0 * 0.6) = 0.08 + 0.6 = 0.68

weighted_threshold = (127.5 * 0.2 * 0.4) + (102.5 * 1.0 * 0.6)
                     ──────────────────────────────────────
                                   0.68
                   = (10.2 + 61.5) / 0.68
                   = 71.7 / 0.68
                   = 105.4

# Final Result
ThresholdResult(
    threshold_value=105.4,   # Dominated by local (high confidence)
    confidence=1.0,          # max(0.2, 1.0) = 1.0
    max_jump=115.0,         # max(25.0, 115.0) = 115.0
    method_used="adaptive_weighted"
)
```

**Interpretation**: Local strategy dominates when it has higher confidence, adaptive strategy trusts field-specific evidence over global noise.

---

### Example 4: All Strategies Have Zero Confidence

**Scenario**: Extremely poor image quality or no bubbles detected

```python
# Input
bubble_mean_values = []  # No bubbles detected

# Strategy Results
result_global = ThresholdResult(
    threshold_value=127.5,
    confidence=0.0,      # No bubbles
    max_jump=0.0,
    method_used="global_default",
    fallback_used=True
)

result_local = ThresholdResult(
    threshold_value=127.5,
    confidence=0.0,      # No bubbles
    max_jump=0.0,
    method_used="local_single_bubble_fallback",
    fallback_used=True
)

# Weighted Calculation
total_weight = (0.0 * 0.4) + (0.0 * 0.6) = 0.0

# SPECIAL CASE: total_weight == 0
# Skip weighted averaging, use emergency fallback

# Final Result
ThresholdResult(
    threshold_value=127.5,   # Default fallback
    confidence=0.0,
    max_jump=0.0,
    method_used="adaptive_all_zero_confidence",
    fallback_used=True
)
```

**Interpretation**: Safety fallback when all strategies fail, prevents division by zero.

---

## Factory Pattern Flow

### create_default_threshold_calculator()

**Purpose**: Convenient factory function to create the recommended adaptive strategy configuration.

```
START: create_default_threshold_calculator(global_threshold=96.0)
│
├─► Create Strategy Instances
│   │
│   global_strategy = GlobalThresholdStrategy()
│   │
│   │ - No parameters needed
│   │ - Calculates file-level threshold from all bubbles
│   │
│   local_strategy = LocalThresholdStrategy(global_fallback=96.0)
│   │
│   │ - global_fallback: Threshold to use when local confidence is low
│   │ - Calculates field-level threshold with fallback
│
├─► Create Adaptive Strategy with Recommended Weights
│   │
│   adaptive = AdaptiveThresholdStrategy(
│       strategies=[global_strategy, local_strategy],
│       weights=[0.4, 0.6]  # Prefer local (60%) over global (40%)
│   )
│   │
│   │ Weight Rationale:
│   │ - Local (0.6): Field-specific conditions often more accurate
│   │ - Global (0.4): Provides stable baseline, prevents overfitting
│   │
│   │ Total weight = 1.0 (normalized weights for easier interpretation)
│
└─► Return Configured Strategy
    │
    return adaptive
    │
    END

USAGE EXAMPLE:

# Step 1: Calculate global threshold from all bubbles
global_result = GlobalThresholdStrategy().calculate_threshold(
    all_bubbles_in_file,
    ThresholdConfig()
)
global_threshold = global_result.threshold_value  # e.g., 96.0

# Step 2: Create adaptive calculator with global baseline
calculator = create_default_threshold_calculator(global_threshold=96.0)

# Step 3: Calculate field-specific thresholds
for field in fields:
    result = calculator.calculate_threshold(
        field.bubble_mean_values,
        ThresholdConfig()
    )
    field_threshold = result.threshold_value
    field_confidence = result.confidence
```

---

## Integration with Bubble Detection Pipeline

### End-to-End Flow

```
┌──────────────────────────────────────────────────────────────┐
│ PHASE 1: DETECTION PASS (File-Level)                         │
└──────────────────────────────────────────────────────────────┘
│
├─► Extract All Bubbles from Image
│   │
│   all_bubbles = []
│   for field in template.fields:
│       for bubble_position in field.bubble_positions:
│           bubble_roi = image[y:y+h, x:x+w]
│           bubble_mean = np.mean(bubble_roi)
│           all_bubbles.append(bubble_mean)
│   │
│   all_bubbles = [45, 50, 52, 55, 58, 140, 145, 150, ...]
│                  └──Empty bubbles──┘  └─Filled bubbles─┘
│
├─► Calculate Global Threshold
│   │
│   global_strategy = GlobalThresholdStrategy()
│   global_result = global_strategy.calculate_threshold(
│       all_bubbles,
│       ThresholdConfig()
│   )
│   │
│   global_threshold = 96.0
│   global_confidence = 1.0
│
└─► Store for Interpretation Pass
    │
    context.global_threshold = global_threshold

┌──────────────────────────────────────────────────────────────┐
│ PHASE 2: INTERPRETATION PASS (Field-Level)                   │
└──────────────────────────────────────────────────────────────┘
│
├─► Create Adaptive Calculator (Once)
│   │
│   calculator = create_default_threshold_calculator(
│       global_threshold=context.global_threshold  # 96.0
│   )
│
└─► For Each Field in Template:
    │
    ├─► Get Field's Bubbles
    │   │
    │   field_bubbles = [45, 50, 55, 145]
    │
    ├─► Calculate Adaptive Threshold
    │   │
    │   result = calculator.calculate_threshold(
    │       field_bubbles,
    │       ThresholdConfig()
    │   )
    │   │
    │   field_threshold = 96.9
    │   field_confidence = 1.0
    │   method_used = "adaptive_weighted"
    │
    ├─► Classify Each Bubble
    │   │
    │   for bubble_mean in field_bubbles:
    │       if bubble_mean < field_threshold:
    │           bubble_state = "FILLED"  # Darker = marked
    │       else:
    │           bubble_state = "EMPTY"   # Lighter = unmarked
    │   │
    │   Example:
    │   45  < 96.9 → FILLED ✓
    │   50  < 96.9 → FILLED ✓
    │   55  < 96.9 → FILLED ✓
    │   145 > 96.9 → EMPTY
    │
    ├─► Generate Field Answer
    │   │
    │   field_value = "A,B,C"  # Bubbles 0, 1, 2 were filled
    │
    └─► Return Field Results
        │
        {
            "field_label": "Q1",
            "field_value": "A,B,C",
            "threshold_used": 96.9,
            "threshold_confidence": 1.0,
            "threshold_method": "adaptive_weighted",
            "strategy_breakdown": [
                {"method": "global_max_jump", "threshold": 96.0, "weight": 0.4},
                {"method": "local_max_jump", "threshold": 97.5, "weight": 0.6}
            ]
        }
```

---

## Decision Points

### Decision 1: When to Use Adaptive Strategy

```
Do you need different thresholds for different fields?
│
├─► YES: Use AdaptiveThresholdStrategy
│   │    (Combines global baseline + local refinement)
│   │
│   └─► Scenarios:
│       - Multi-field forms with varying lighting per field
│       - Fields with different bubble counts
│       - Want robust threshold that adapts to field conditions
│
└─► NO: Use GlobalThresholdStrategy alone
    │   (Single threshold for entire image)
    │
    └─► Scenarios:
        - Uniform lighting across entire form
        - Simple forms with consistent bubble density
        - Performance-critical (fewer calculations)
```

### Decision 2: How to Configure Weights

```
How confident is local threshold typically?
│
├─► High Confidence: Increase local weight
│   │
│   │ weights = [0.3, 0.7]  # 70% local, 30% global
│   │
│   └─► Scenarios:
│       - Fields have many bubbles (10+)
│       - Consistent within-field lighting
│       - Trust field-specific evidence
│
├─► Medium Confidence: Balanced weights
│   │
│   │ weights = [0.4, 0.6]  # 60% local, 40% global (DEFAULT)
│   │
│   └─► Scenarios:
│       - Typical OMR forms
│       - Mix of field sizes
│       - Balance global baseline and local refinement
│
└─► Low Confidence: Increase global weight
    │
    │ weights = [0.6, 0.4]  # 60% global, 40% local
    │
    └─► Scenarios:
        - Fields have few bubbles (2-5)
        - Inconsistent within-field lighting
        - Trust global baseline more
```

### Decision 3: Handling Zero Confidence

```
What if all strategies have zero confidence?
│
└─► Use Emergency Fallback
    │
    │ ThresholdResult(
    │     threshold_value=127.5,  # Grayscale midpoint
    │     confidence=0.0,
    │     method_used="adaptive_all_zero_confidence",
    │     fallback_used=True
    │ )
    │
    └─► Implications:
        - Image quality is extremely poor
        - No bubbles detected
        - Threshold will likely misclassify
        - User should be warned about low confidence
```

---

## Performance Characteristics

### Time Complexity

```
O(k × m log m)

Where:
- k = number of strategies (typically 2)
- m = bubbles per field (typically 4-20)

Breakdown:
- For each strategy (k times):
    - Sort bubble values: O(m log m)
    - Find max jump: O(m)
- Weighted averaging: O(k)

Typical: 2 × (10 log 10) ≈ 2 × 33 = 66 operations per field
Fast: < 0.1ms per field on modern hardware
```

### Space Complexity

```
O(k × m)

Where:
- k = number of strategies
- m = bubbles per field

Memory:
- Each strategy creates sorted copy: O(m)
- Each strategy returns ThresholdResult: ~200 bytes
- Metadata for strategy_results: k × ~100 bytes

Typical: 2 × 10 floats + 2 × 200 bytes ≈ 560 bytes per field
Negligible memory impact
```

### Comparison with Single Strategy

| Metric | Global Only | Local Only | Adaptive (Both) |
|--------|-------------|------------|-----------------|
| Time | O(n log n) file-level | O(m log m) per field | O(k × m log m) per field |
| Accuracy | Baseline | High (when confident) | Best (robust) |
| Fallback | Default | Global → Default | Multi-level |
| Overhead | None | None | ~2× local (minimal) |

**Recommendation**: Use Adaptive strategy for production (best accuracy with minimal overhead).

---

## Related Flows

- **Global Threshold**: `modules/domain/threshold/global/flows.md`
- **Local Threshold**: `modules/domain/threshold/local/flows.md` (pending)
- **Threshold Strategy Pattern**: `modules/domain/threshold/flows.md`
- **Bubble Detection**: `modules/domain/detection/bubbles-threshold/flows.md` (pending)

---

## Summary

The Adaptive Threshold Strategy provides:

1. **Robustness**: Combines multiple strategies for reliable thresholding
2. **Confidence-Weighted**: Automatically favors more confident strategies
3. **Graceful Fallback**: Multiple levels of fallback for edge cases
4. **Configurable**: Adjustable weights for different use cases
5. **Efficient**: Minimal overhead (< 2× single strategy)

**Recommended Usage**: Default threshold calculator for bubble detection with `create_default_threshold_calculator()`.

**Key Insight**: "Adaptive" refers to combining strategies adaptively based on confidence, NOT OpenCV's image-level adaptive thresholding (cv2.adaptiveThreshold).
