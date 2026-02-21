# Local Threshold Strategy Flows

**Module**: Domain - Threshold - Local
**Python Reference**: `src/processors/threshold/local_threshold.py`
**Last Updated**: 2026-02-20

---

## Overview

Local Threshold Strategy calculates a field-specific threshold using only the bubbles from a single field. It provides fine-grained adaptation to per-field lighting conditions while using the global threshold as a fallback for low-confidence scenarios.

**Use Case**: Per-field threshold when lighting varies across the OMR sheet or for highest accuracy.

---

## Algorithm Flow

### Main Calculation Flow

```
START: LocalThresholdStrategy.calculate_threshold(bubble_mean_values, config)
│
├─► STEP 1: Initialize Fallback
│   │
│   fallback_threshold = self.global_fallback or config.default_threshold
│   │
│   │ Examples:
│   │ - If global_fallback was set to 96.0 → use 96.0
│   │ - If global_fallback is None → use config.default_threshold (127.5)
│   │
│   │ Purpose: Have a known-good threshold to fall back to
│
├─► STEP 2: Handle Empty or Single Bubble
│   │
│   If len(bubble_mean_values) < 2:
│       │
│       └─► RETURN ThresholdResult(
│               threshold_value: fallback_threshold,
│               confidence: 0.0,
│               max_jump: 0.0,
│               method_used: "local_single_bubble_fallback",
│               fallback_used: True
│           )
│           END (early exit)
│
├─► STEP 3: Sort Bubble Values
│   │
│   sorted_values = sorted(bubble_mean_values)
│   │
│   │ Example:
│   │ Input:  [150, 45, 145, 50, 55]
│   │ Sorted: [45, 50, 55, 145, 150]
│   │          └Empty┘  └Filled┘
│
├─► STEP 4: Handle Two Bubbles (Special Case)
│   │
│   If len(sorted_values) == 2:
│       │
│       ├─ Calculate gap
│       gap = sorted_values[1] - sorted_values[0]
│       │
│       │ Example: sorted_values = [50, 150]
│       │ gap = 150 - 50 = 100
│       │
│       ├─ Check if gap is significant
│       If gap < config.min_gap_two_bubbles (20.0):
│           │
│           │ Gap too small - might be noise, not real distinction
│           │
│           └─► RETURN ThresholdResult(
│                   threshold_value: fallback_threshold,
│                   confidence: 0.3,  # Low-medium confidence
│                   max_jump: gap,
│                   method_used: "local_two_bubbles_small_gap_fallback",
│                   fallback_used: True
│               )
│               END
│       │
│       Else:  # gap >= min_gap_two_bubbles
│           │
│           │ Gap is significant - real filled/empty distinction
│           │
│           └─► RETURN ThresholdResult(
│                   threshold_value: np.mean(sorted_values),  # Midpoint
│                   confidence: 0.7,  # Medium-high confidence
│                   max_jump: gap,
│                   method_used: "local_two_bubbles_mean",
│                   fallback_used: False
│               )
│               END
│
├─► STEP 5: Handle Three+ Bubbles (Normal Case)
│   │
│   │ Initialize tracking variables
│   max_jump = 0.0  # Start at zero (unlike global strategy!)
│   threshold = fallback_threshold
│   │
│   │ Find largest jump using bidirectional window
│   For i in range(1, len(sorted_values) - 1):
│   │   # i goes from 1 to len-2
│   │   # We need i-1 and i+1, so skip first and last indices
│   │
│   │   ┌─ Calculate jump across 3-value window
│   │   jump = sorted_values[i + 1] - sorted_values[i - 1]
│   │
│   │   ┌─ ITERATION EXAMPLES (sorted_values = [45, 50, 55, 145, 150]):
│   │   │
│   │   ├─ i=1: jump = sorted[2] - sorted[0] = 55 - 45 = 10
│   │   │       (small jump, within empty group)
│   │   │
│   │   ├─ i=2: jump = sorted[3] - sorted[1] = 145 - 50 = 95 ✓ MAXIMUM!
│   │   │       (large jump, between groups)
│   │   │
│   │   └─ i=3: jump = sorted[4] - sorted[2] = 150 - 55 = 95
│   │           (also large jump)
│   │
│   │   If jump > max_jump:
│   │       max_jump = jump
│   │       threshold = sorted_values[i - 1] + jump / 2
│   │       │
│   │       │ For i=2: threshold = sorted[1] + 95/2
│   │       │                    = 50 + 47.5
│   │       │                    = 97.5
│
├─► STEP 6: Check Confidence
│   │
│   │ Calculate confidence threshold
│   confident_jump = config.min_jump + config.min_jump_surplus_for_global_fallback
│   confident_jump = 30.0 + 10.0 = 40.0
│   │
│   │ Purpose: Require jump to be "significantly" larger than min_jump
│   │
│   If max_jump < confident_jump:
│       │
│       │ Jump not large enough - low confidence in local threshold
│       │
│       └─► RETURN ThresholdResult(
│               threshold_value: fallback_threshold,  # Use global!
│               confidence: 0.4,  # Low-medium confidence
│               max_jump: max_jump,
│               method_used: "local_low_confidence_global_fallback",
│               fallback_used: True,
│               metadata: {
│                   "local_threshold": threshold  # What we would have used
│               }
│           )
│           END
│
├─► STEP 7: Calculate High-Confidence Result
│   │
│   │ Jump is confident, use local threshold
│   │
│   confidence = min(1.0, max_jump / (confident_jump * 2))
│   │
│   │ Formula explanation:
│   │ - Denominator: confident_jump * 2 = 40 * 2 = 80
│   │ - If max_jump >= 80: confidence = 1.0 (capped)
│   │ - If max_jump < 80: confidence = max_jump / 80
│   │
│   │ Examples:
│   │ max_jump=95  → 95/80 = 1.0 (capped) ✓
│   │ max_jump=60  → 60/80 = 0.75
│   │ max_jump=40  → 40/80 = 0.50 (at confident_jump)
│   │
│   │ Our example: max_jump=95 → confidence=1.0
│
└─► STEP 8: Return High-Confidence Result
    │
    RETURN ThresholdResult(
        threshold_value: 97.5,
        confidence: 1.0,
        max_jump: 95.0,
        method_used: "local_max_jump",
        fallback_used: False,
        metadata: {
            "num_bubbles": 5
        }
    )

END
```

---

## Edge Case Flows

### Edge Case 1: Empty Field (0 Bubbles)

```
Input: bubble_mean_values = []
       global_fallback = 96.0

Flow:
├─► STEP 1: fallback_threshold = 96.0
├─► STEP 2: len([]) < 2 → True
└─► RETURN ThresholdResult(
        threshold_value: 96.0,  # Global fallback
        confidence: 0.0,
        max_jump: 0.0,
        method_used: "local_single_bubble_fallback",
        fallback_used: True
    )

Reason: Cannot determine filled vs empty with no data
```

### Edge Case 2: Single Bubble

```
Input: bubble_mean_values = [100.0]
       global_fallback = 96.0

Flow:
├─► STEP 1: fallback_threshold = 96.0
├─► STEP 2: len([100.0]) < 2 → True
└─► RETURN ThresholdResult(
        threshold_value: 96.0,
        confidence: 0.0,
        max_jump: 0.0,
        method_used: "local_single_bubble_fallback",
        fallback_used: True
    )

Reason: Cannot determine filled vs empty with single sample
```

### Edge Case 3: Two Bubbles - Small Gap

```
Input: bubble_mean_values = [100, 110]
       global_fallback = 96.0
       config.min_gap_two_bubbles = 20.0

Flow:
├─► STEP 1: fallback_threshold = 96.0
├─► STEP 2: len([100, 110]) >= 2 → Continue
├─► STEP 3: sorted_values = [100, 110]
├─► STEP 4: len(sorted_values) == 2 → True
│   ├─ gap = 110 - 100 = 10
│   ├─ gap < 20.0 → True (10 < 20)
│   └─► RETURN ThresholdResult(
│           threshold_value: 96.0,
│           confidence: 0.3,
│           max_jump: 10.0,
│           method_used: "local_two_bubbles_small_gap_fallback",
│           fallback_used: True
│       )

Reason: Gap too small, likely just variation within same group
```

### Edge Case 4: Two Bubbles - Large Gap

```
Input: bubble_mean_values = [50, 150]
       global_fallback = 96.0
       config.min_gap_two_bubbles = 20.0

Flow:
├─► STEP 1: fallback_threshold = 96.0
├─► STEP 2: len([50, 150]) >= 2 → Continue
├─► STEP 3: sorted_values = [50, 150]
├─► STEP 4: len(sorted_values) == 2 → True
│   ├─ gap = 150 - 50 = 100
│   ├─ gap >= 20.0 → True (100 >= 20)
│   └─► RETURN ThresholdResult(
│           threshold_value: np.mean([50, 150]) = 100.0,
│           confidence: 0.7,
│           max_jump: 100.0,
│           method_used: "local_two_bubbles_mean",
│           fallback_used: False
│       )

Reason: Large gap indicates real filled/empty distinction
```

### Edge Case 5: Low Confidence Jump

```
Input: bubble_mean_values = [45, 50, 55, 70, 75]
       global_fallback = 96.0
       config.min_jump = 30.0
       config.min_jump_surplus_for_global_fallback = 10.0

Flow:
├─► STEP 1-3: Sort → [45, 50, 55, 70, 75]
├─► STEP 5: For i in [1, 2, 3]:
│   ├─ i=1: jump = 55 - 45 = 10
│   ├─ i=2: jump = 70 - 50 = 20
│   └─ i=3: jump = 75 - 55 = 20
│   max_jump = 20, threshold ≈ 60
│
├─► STEP 6: confident_jump = 30 + 10 = 40
│   max_jump < confident_jump? → 20 < 40 → True
│   └─► RETURN ThresholdResult(
│           threshold_value: 96.0,  # Global fallback!
│           confidence: 0.4,
│           max_jump: 20.0,
│           method_used: "local_low_confidence_global_fallback",
│           fallback_used: True,
│           metadata: {"local_threshold": 60.0}
│       )

Reason: Local jump too small, not confident enough to override global
```

### Edge Case 6: All Same Value

```
Input: bubble_mean_values = [100, 100, 100]
       global_fallback = 96.0

Flow:
├─► STEP 1-3: Sort → [100, 100, 100]
├─► STEP 5: For i in [1]:
│   └─ i=1: jump = 100 - 100 = 0
│   max_jump = 0, threshold = 96.0 (not updated)
│
├─► STEP 6: confident_jump = 40
│   max_jump < confident_jump? → 0 < 40 → True
│   └─► RETURN ThresholdResult(
│           threshold_value: 96.0,
│           confidence: 0.4,
│           max_jump: 0.0,
│           method_used: "local_low_confidence_global_fallback",
│           fallback_used: True,
│           metadata: {"local_threshold": 96.0}
│       )

Reason: No distinction between bubbles, use global
```

---

## Comparison: Local vs Global

### Key Differences

| Aspect | Global Strategy | Local Strategy |
|--------|----------------|----------------|
| Input | All bubbles from file | Bubbles from single field |
| Looseness | Yes (skip extremes) | No (use all values) |
| Two bubbles | Not handled specially | Special case with gap check |
| Fallback | Default threshold (127.5) | Global threshold (96.0) |
| Confidence threshold | min_jump * 3 (90) | (min_jump + surplus) * 2 (80) |
| max_jump init | config.min_jump | 0.0 |

### Example: Same Bubbles, Different Results

```
Bubbles: [45, 50, 55, 145, 150]

Global Strategy:
- Processes as part of larger set (ignores field boundaries)
- May include bubbles from other fields
- Returns file-level threshold

Local Strategy:
- Only these 5 bubbles
- Calculates field-specific threshold
- max_jump = 145 - 50 = 95
- threshold = 50 + 95/2 = 97.5
- confidence = 1.0
```

---

## Decision Tree

```
┌─────────────────────────────────────┐
│ Local Threshold Calculation         │
└─────────────────────────────────────┘
             │
             ├─► < 2 bubbles?
             │   └─► YES: Return global fallback (conf=0.0)
             │
             ├─► Exactly 2 bubbles?
             │   ├─► gap < min_gap_two_bubbles?
             │   │   └─► YES: Return global fallback (conf=0.3)
             │   └─► NO: Return mean of bubbles (conf=0.7)
             │
             └─► 3+ bubbles
                 ├─► Find max_jump
                 │
                 ├─► max_jump < confident_jump?
                 │   └─► YES: Return global fallback (conf=0.4)
                 │
                 └─► NO: Return local threshold
                         confidence = min(1.0, max_jump/80)
```

---

## Visual Examples

### Example 1: High Confidence Local

```
Field Bubbles: [45, 48, 52, 145, 150, 155]

Histogram:
45  ■
48  ■
52  ■
    ─── GAP (93) ───
145         ■
150         ■
155         ■

max_jump = 145 - 52 = 93
confident_jump = 40
93 >= 40 ✓ Use local threshold
threshold = 52 + 93/2 = 98.5
confidence = min(1.0, 93/80) = 1.0
```

### Example 2: Low Confidence → Global Fallback

```
Field Bubbles: [80, 85, 90, 95, 100]

Histogram:
80  ■
85  ■
90  ■
95  ■
100 ■

max_jump = 100 - 85 = 15
confident_jump = 40
15 < 40 ✗ Too small, use global fallback
threshold = global_fallback (e.g., 96.0)
confidence = 0.4
```

### Example 3: Two Bubbles - Use Mean

```
Field Bubbles: [50, 150]

Gap = 100 ≥ 20 ✓ Significant
threshold = mean([50, 150]) = 100
confidence = 0.7
```

---

## Performance Characteristics

### Time Complexity

```
O(m log m) where m = bubbles in field

Breakdown:
- Sorting: O(m log m)
- Two bubbles: O(1)
- 3+ bubbles loop: O(m)
- Total: O(m log m) dominated by sorting

Typical values:
- m (bubbles per field) = 4-20
- Much smaller than n (total bubbles) for global strategy
```

### Space Complexity

```
O(m)

Memory:
- sorted_values: copy of field bubbles (m floats = 8m bytes)
- ThresholdResult: ~200 bytes
- Total: ~8m + 200 bytes

Example:
- 4 bubbles: ~240 bytes
- 10 bubbles: ~280 bytes
- 20 bubbles: ~360 bytes
```

---

## Browser Migration

### JavaScript Implementation

```javascript
class LocalThresholdStrategy extends ThresholdStrategy {
  constructor(globalFallback) {
    super();
    this.globalFallback = globalFallback;
  }

  calculateThreshold(bubbleMeanValues, config) {
    // Step 1: Initialize fallback
    const fallbackThreshold = this.globalFallback ?? config.defaultThreshold;

    // Step 2: Handle < 2 bubbles
    if (bubbleMeanValues.length < 2) {
      return {
        thresholdValue: fallbackThreshold,
        confidence: 0.0,
        maxJump: 0.0,
        methodUsed: 'local_single_bubble_fallback',
        fallbackUsed: true,
        metadata: {}
      };
    }

    // Step 3: Sort
    const sortedValues = [...bubbleMeanValues].sort((a, b) => a - b);

    // Step 4: Handle two bubbles
    if (sortedValues.length === 2) {
      const gap = sortedValues[1] - sortedValues[0];

      if (gap < config.minGapTwoBubbles) {
        return {
          thresholdValue: fallbackThreshold,
          confidence: 0.3,
          maxJump: gap,
          methodUsed: 'local_two_bubbles_small_gap_fallback',
          fallbackUsed: true,
          metadata: {}
        };
      }

      return {
        thresholdValue: (sortedValues[0] + sortedValues[1]) / 2,
        confidence: 0.7,
        maxJump: gap,
        methodUsed: 'local_two_bubbles_mean',
        fallbackUsed: false,
        metadata: {}
      };
    }

    // Step 5: Handle 3+ bubbles
    let maxJump = 0.0;  // Start at zero
    let threshold = fallbackThreshold;

    for (let i = 1; i < sortedValues.length - 1; i++) {
      const jump = sortedValues[i + 1] - sortedValues[i - 1];
      if (jump > maxJump) {
        maxJump = jump;
        threshold = sortedValues[i - 1] + jump / 2;
      }
    }

    // Step 6: Check confidence
    const confidentJump = config.minJump + config.minJumpSurplusForGlobalFallback;

    if (maxJump < confidentJump) {
      return {
        thresholdValue: fallbackThreshold,
        confidence: 0.4,
        maxJump: maxJump,
        methodUsed: 'local_low_confidence_global_fallback',
        fallbackUsed: true,
        metadata: { localThreshold: threshold }
      };
    }

    // Step 7: Calculate high-confidence result
    const confidence = Math.min(1.0, maxJump / (confidentJump * 2));

    return {
      thresholdValue: threshold,
      confidence: confidence,
      maxJump: maxJump,
      methodUsed: 'local_max_jump',
      fallbackUsed: false,
      metadata: { numBubbles: bubbleMeanValues.length }
    };
  }
}
```

### Key Differences from Python

1. **Nullish Coalescing**:
   ```javascript
   // Python: or
   fallback = self.global_fallback or config.default_threshold

   // JavaScript: ?? (nullish coalescing)
   const fallback = this.globalFallback ?? config.defaultThreshold;
   ```

2. **NumPy Mean**:
   ```javascript
   // Python: np.mean()
   threshold = float(np.mean(sorted_values))

   // JavaScript: manual calculation
   const threshold = (sortedValues[0] + sortedValues[1]) / 2;
   ```

3. **Constructor**:
   ```javascript
   // Python: __init__
   def __init__(self, global_fallback=None):
       self.global_fallback = global_fallback

   // JavaScript: constructor
   constructor(globalFallback) {
     super();
     this.globalFallback = globalFallback;
   }
   ```

---

## Related Documentation

- **Threshold Strategy Pattern**: `../concept.md`
- **All Strategy Flows**: `../flows.md`
- **Global Threshold**: `../global/flows.md`
- **Adaptive Threshold**: `../adaptive/flows.md`

---

## Summary

Local Threshold Strategy:

1. **Uses global threshold as fallback** for low-confidence scenarios
2. **Sorts field bubbles** from darkest to lightest
3. **Special handling for 2 bubbles** with gap check
4. **Finds largest jump for 3+ bubbles** using bidirectional window
5. **Confidence-based fallback** requires jump > confident_jump threshold
6. **Returns field-specific threshold** when confident

**Best For**: Per-field adaptation, varying lighting conditions, highest accuracy
**Limitations**: Requires at least 2 bubbles per field, needs good global fallback
**Typical Confidence**: 0.7-1.0 for well-marked fields, 0.0-0.4 for edge cases
