# Global Threshold Strategy Flows

**Module**: Domain - Threshold - Global
**Python Reference**: `src/processors/threshold/global_threshold.py`
**Last Updated**: 2026-02-20

---

## Overview

Global Threshold Strategy calculates a file-level threshold by analyzing ALL bubbles across ALL fields in an image. It finds the largest intensity gap between filled and empty bubbles to determine the optimal separation point.

**Use Case**: Baseline threshold for entire OMR sheet when lighting is relatively uniform.

---

## Algorithm Flow

### Main Calculation Flow

```
START: GlobalThresholdStrategy.calculate_threshold(bubble_mean_values, config)
│
├─► STEP 1: Validate Input
│   │
│   └─► Check bubble count
│       │
│       ├─ If len(bubble_mean_values) < 2:
│       │  │
│       │  └─► RETURN ThresholdResult(
│       │        threshold_value: config.default_threshold,  # 127.5
│       │        confidence: 0.0,
│       │        max_jump: 0.0,
│       │        method_used: "global_default",
│       │        fallback_used: True
│       │      )
│       │      END (early exit)
│       │
│       └─ Else: Continue to step 2
│
├─► STEP 2: Sort Bubble Values
│   │
│   sorted_values = sorted(bubble_mean_values)
│   │
│   │ Example:
│   │ Input:  [55, 150, 45, 145, 50, 155, 52, 140]
│   │ Sorted: [45, 50, 52, 55, 140, 145, 150, 155]
│   │          └─Empty group─┘  └──Filled group──┘
│
├─► STEP 3: Initialize Looseness Parameters
│   │
│   │ Looseness = 1 (hardcoded smoothing factor)
│   │ Purpose: Ignore extreme values when finding gaps
│   │
│   ls = (looseness + 1) // 2
│   ls = (1 + 1) // 2 = 1
│   │
│   total_bubbles_loose = len(sorted_values) - ls
│   total_bubbles_loose = 8 - 1 = 7
│   │
│   │ Meaning: When finding gaps, we'll:
│   │ - Start from index 'ls' (1)
│   │ - End at index 'total_bubbles_loose' (7)
│   │ - Skip first and last values as potential outliers
│
├─► STEP 4: Find Maximum Jump
│   │
│   max_jump = config.min_jump  # Initialize with minimum threshold (30.0)
│   threshold = config.default_threshold  # Initialize with default (127.5)
│   │
│   For i in range(ls, total_bubbles_loose):
│   │   # i goes from 1 to 7
│   │
│   │   ┌─ Calculate jump using "looseness window"
│   │   jump = sorted_values[i + ls] - sorted_values[i - ls]
│   │   #      sorted_values[i + 1] - sorted_values[i - 1]
│   │   #      Look at 3-value window: [i-1, i, i+1]
│   │
│   │   ┌─ ITERATION EXAMPLES:
│   │   │
│   │   ├─ i=1: jump = sorted[2] - sorted[0] = 52 - 45 = 7
│   │   │       (small jump, within empty group)
│   │   │
│   │   ├─ i=2: jump = sorted[3] - sorted[1] = 55 - 50 = 5
│   │   │       (small jump, within empty group)
│   │   │
│   │   ├─ i=3: jump = sorted[4] - sorted[2] = 140 - 52 = 88 ✓ MAXIMUM!
│   │   │       (large jump, between empty and filled groups)
│   │   │       This is the gap we're looking for!
│   │   │
│   │   ├─ i=4: jump = sorted[5] - sorted[3] = 145 - 55 = 90 ✓ ALSO LARGE!
│   │   │       (but we already found 88, so we take first large gap)
│   │   │       Actually this one is larger, so max_jump updates to 90
│   │   │
│   │   ├─ i=5: jump = sorted[6] - sorted[4] = 150 - 140 = 10
│   │   │       (small jump, within filled group)
│   │   │
│   │   └─ i=6: jump = sorted[7] - sorted[5] = 155 - 145 = 10
│   │           (small jump, within filled group)
│   │
│   │   If jump > max_jump:
│   │       max_jump = jump
│   │       threshold = sorted_values[i - ls] + jump / 2
│   │       │
│   │       │ For i=4 (max_jump = 90):
│   │       │ threshold = sorted[4 - 1] + 90 / 2
│   │       │           = sorted[3] + 45
│   │       │           = 55 + 45
│   │       │           = 100.0
│   │       │
│   │       │ Threshold placed at MIDPOINT of gap:
│   │       │ Empty bubbles: [45, 50, 52, 55] all < 100 ✓
│   │       │ Filled bubbles: [140, 145, 150, 155] all > 100 ✓
│   │       │ Perfect separation!
│
├─► STEP 5: Calculate Confidence
│   │
│   │ Confidence based on jump size relative to min_jump
│   │
│   confidence = min(1.0, max_jump / (config.min_jump * 3))
│   │
│   │ Formula explanation:
│   │ - Denominator: min_jump * 3 = 30 * 3 = 90
│   │ - If max_jump >= 90: confidence = 1.0 (capped)
│   │ - If max_jump < 90: confidence = max_jump / 90
│   │
│   │ Examples:
│   │ max_jump=90  → 90/90 = 1.0 (capped) ✓
│   │ max_jump=60  → 60/90 = 0.67
│   │ max_jump=45  → 45/90 = 0.50
│   │ max_jump=30  → 30/90 = 0.33 (at min_jump threshold)
│   │ max_jump=15  → 15/90 = 0.17 (low confidence)
│   │
│   │ Our example: max_jump=90 → confidence=1.0
│
├─► STEP 6: Check Fallback Condition
│   │
│   fallback_used = (max_jump < config.min_jump)
│   │
│   │ Our example: 90 < 30 → False
│   │ We did NOT use fallback, we found a good threshold!
│
└─► STEP 7: Return Result
    │
    RETURN ThresholdResult(
        threshold_value: 100.0,
        confidence: 1.0,
        max_jump: 90.0,
        method_used: "global_max_jump",
        fallback_used: False,
        metadata: {
            "num_bubbles": 8,
            "min_value": 45.0,
            "max_value": 155.0
        }
    )

END
```

---

## Edge Case Flows

### Edge Case 1: Empty List

```
Input: bubble_mean_values = []

Flow:
├─► STEP 1: len([]) < 2 → True
└─► RETURN ThresholdResult(
        threshold_value: 127.5,  # default_threshold
        confidence: 0.0,
        max_jump: 0.0,
        method_used: "global_default",
        fallback_used: True
    )
```

### Edge Case 2: Single Bubble

```
Input: bubble_mean_values = [100.0]

Flow:
├─► STEP 1: len([100.0]) < 2 → True
└─► RETURN ThresholdResult(
        threshold_value: 127.5,  # default_threshold
        confidence: 0.0,
        max_jump: 0.0,
        method_used: "global_default",
        fallback_used: True
    )
```

### Edge Case 3: All Same Value

```
Input: bubble_mean_values = [100, 100, 100, 100]

Flow:
├─► STEP 1: len([...]) >= 2 → Continue
├─► STEP 2: sorted_values = [100, 100, 100, 100]
├─► STEP 3: ls=1, total_bubbles_loose=3
├─► STEP 4: For i in [1, 2, 3]:
│   │
│   ├─ i=1: jump = sorted[2] - sorted[0] = 100 - 100 = 0
│   ├─ i=2: jump = sorted[3] - sorted[1] = 100 - 100 = 0
│   └─ i=3: Would be out of bounds
│   │
│   max_jump = 30.0 (never updated, stays at config.min_jump)
│   threshold = 127.5 (never updated, stays at default)
│
├─► STEP 5: confidence = min(1.0, 30/90) = 0.33
├─► STEP 6: fallback_used = (30 < 30) → False (edge case!)
└─► RETURN ThresholdResult(
        threshold_value: 127.5,
        confidence: 0.33,
        max_jump: 30.0,
        method_used: "global_max_jump",
        fallback_used: False,  # But threshold is still default!
        metadata: {...}
    )

Note: Fallback flag is False but we still used default threshold.
This is technically correct: we ran the algorithm, it just didn't
find a better threshold than the default.
```

### Edge Case 4: Two Bubbles

```
Input: bubble_mean_values = [50, 150]

Flow:
├─► STEP 1: len([50, 150]) >= 2 → Continue
├─► STEP 2: sorted_values = [50, 150]
├─► STEP 3: ls=1, total_bubbles_loose=1
├─► STEP 4: For i in range(1, 1):  # Empty range!
│   │       Loop doesn't execute
│   │
│   max_jump = 30.0 (never updated)
│   threshold = 127.5 (never updated)
│
├─► STEP 5: confidence = min(1.0, 30/90) = 0.33
├─► STEP 6: fallback_used = (30 < 30) → False
└─► RETURN ThresholdResult(
        threshold_value: 127.5,
        confidence: 0.33,
        max_jump: 30.0,
        method_used: "global_max_jump",
        fallback_used: False,
        metadata: {...}
    )

Note: Two bubbles with looseness=1 don't have enough data points
to calculate a meaningful jump. Falls back to default threshold.
This is why LocalThresholdStrategy has special two-bubble handling.
```

### Edge Case 5: Small Gap

```
Input: bubble_mean_values = [100, 105, 110, 115]

Flow:
├─► STEP 1: len([...]) >= 2 → Continue
├─► STEP 2: sorted_values = [100, 105, 110, 115]
├─► STEP 3: ls=1, total_bubbles_loose=3
├─► STEP 4: For i in [1, 2, 3]:
│   │
│   ├─ i=1: jump = sorted[2] - sorted[0] = 110 - 100 = 10
│   ├─ i=2: jump = sorted[3] - sorted[1] = 115 - 105 = 10
│   └─ i=3: Out of bounds
│   │
│   max_jump = 30.0 (no jump > 30, stays at min_jump)
│   threshold = 127.5 (never updated)
│
├─► STEP 5: confidence = min(1.0, 30/90) = 0.33
├─► STEP 6: fallback_used = (30 < 30) → False
└─► RETURN ThresholdResult(
        threshold_value: 127.5,
        confidence: 0.33,
        max_jump: 30.0,  # Actually max was 10, but initialized to 30
        method_used: "global_max_jump",
        fallback_used: False
    )

Note: All bubbles have similar intensity (no clear filled/empty groups).
Uses default threshold with low confidence.
```

---

## Visual Examples

### Example 1: Clear Separation

```
Bubble Values: [45, 48, 52, 55, 140, 145, 150, 155]
Histogram:

45 ■■
48 ■■
52 ■■
55 ■■
   ──── GAP (90 units) ────
140     ■■
145     ■■
150     ■■
155     ■■

Threshold: 100 (midpoint of gap)
Confidence: 1.0 (large gap)
```

### Example 2: Overlapping Groups

```
Bubble Values: [50, 60, 70, 80, 90, 100, 110, 120]
Histogram:

50  ■
60  ■
70  ■
80  ■
90  ■
100 ■
110 ■
120 ■

Max Jump: ~20 (between consecutive values)
Jump < min_jump (30) → Use default threshold
Confidence: 0.33 (low)
```

### Example 3: Three Groups (Rare)

```
Bubble Values: [40, 45, 100, 105, 160, 165]
Histogram:

40  ■■
45  ■■
    ─── GAP 1 (55) ───
100     ■■
105     ■■
    ─── GAP 2 (55) ───
160         ■■
165         ■■

Max Jump: 110 (100 - 45 = 55, or 165 - 105 = 60, actually depends on i)
Algorithm finds FIRST large gap: between 45 and 100
Threshold: 72.5 (midpoint of first gap)
Confidence: 1.0

Note: This creates two groups: [40,45] vs [100,105,160,165]
If you have three actual groups (empty, partial, filled),
global threshold may not be optimal. Use local threshold instead.
```

---

## Parameter Tuning

### min_jump

**Current**: 30.0
**Effect**: Minimum gap size to consider significant

**Tuning Guide**:
```
Increase min_jump (e.g., 40):
- More conservative
- Only accepts very clear gaps
- More likely to use fallback
- Good for: high-quality scans with clear marking

Decrease min_jump (e.g., 20):
- Less conservative
- Accepts smaller gaps as valid
- Less likely to use fallback
- Good for: lower-quality scans, faint marking
```

### looseness

**Current**: 1 (hardcoded)
**Effect**: Number of extreme values to skip

**Potential Values**:
```
looseness=0:
- ls=0, no skipping
- Consider all values including extremes
- Sensitive to outliers

looseness=1:
- ls=1, skip first/last
- Balance robustness and data usage
- Current default

looseness=2:
- ls=1, skip first/last (same as looseness=1)
- Formula: (looseness+1)//2

looseness=3:
- ls=2, skip first/last 2 values
- More robust to outliers
- Requires more bubbles
```

### default_threshold

**Current**: 127.5
**Effect**: Fallback when no good gap found

**Tuning Guide**:
```
If bubbles are typically darker overall:
- Lower default (e.g., 100)
- Bias toward classifying as filled

If bubbles are typically lighter overall:
- Higher default (e.g., 150)
- Bias toward classifying as empty

Neutral (recommended):
- Keep at 127.5 (midpoint)
- No bias either direction
```

---

## Performance Characteristics

### Time Complexity

```
O(n log n) where n = total number of bubbles

Breakdown:
- Sorting: O(n log n)
- Loop: O(n)
- Total: O(n log n) dominated by sorting
```

### Space Complexity

```
O(n)

Memory:
- sorted_values: copy of input (n floats = 8n bytes)
- ThresholdResult: ~200 bytes
- Total: ~8n + 200 bytes

Example:
- 100 bubbles: ~1 KB
- 500 bubbles: ~4 KB
```

### Typical Performance

```
Bubble Count | Time (estimated)
-------------|------------------
50           | < 0.1 ms
100          | < 0.2 ms
500          | < 1 ms
1000         | < 2 ms
```

---

## Browser Migration

### JavaScript Implementation

```javascript
class GlobalThresholdStrategy extends ThresholdStrategy {
  calculateThreshold(bubbleMeanValues, config) {
    // Step 1: Validate
    if (bubbleMeanValues.length < 2) {
      return {
        thresholdValue: config.defaultThreshold,
        confidence: 0.0,
        maxJump: 0.0,
        methodUsed: 'global_default',
        fallbackUsed: true,
        metadata: {}
      };
    }

    // Step 2: Sort (create copy to avoid mutation)
    const sortedValues = [...bubbleMeanValues].sort((a, b) => a - b);

    // Step 3: Looseness parameters
    const looseness = 1;
    const ls = Math.floor((looseness + 1) / 2);
    const totalBubblesLoose = sortedValues.length - ls;

    // Step 4: Find max jump
    let maxJump = config.minJump;
    let threshold = config.defaultThreshold;

    for (let i = ls; i < totalBubblesLoose; i++) {
      const jump = sortedValues[i + ls] - sortedValues[i - ls];
      if (jump > maxJump) {
        maxJump = jump;
        threshold = sortedValues[i - ls] + jump / 2;
      }
    }

    // Step 5: Calculate confidence
    const confidence = Math.min(1.0, maxJump / (config.minJump * 3));

    // Step 6: Return result
    return {
      thresholdValue: threshold,
      confidence: confidence,
      maxJump: maxJump,
      methodUsed: 'global_max_jump',
      fallbackUsed: maxJump < config.minJump,
      metadata: {
        numBubbles: bubbleMeanValues.length,
        minValue: Math.min(...bubbleMeanValues),
        maxValue: Math.max(...bubbleMeanValues)
      }
    };
  }
}
```

### Key Differences from Python

1. **Array Sorting**:
   ```javascript
   // Python: sorted() creates new list
   sorted_values = sorted(bubble_mean_values)

   // JavaScript: must explicitly copy
   const sortedValues = [...bubbleMeanValues].sort((a, b) => a - b);
   ```

2. **Integer Division**:
   ```javascript
   // Python: // operator
   ls = (looseness + 1) // 2

   // JavaScript: Math.floor
   const ls = Math.floor((looseness + 1) / 2);
   ```

3. **Min/Max**:
   ```javascript
   // Python: built-in
   min_value = min(bubble_mean_values)

   // JavaScript: Math.min with spread
   const minValue = Math.min(...bubbleMeanValues);
   ```

---

## Related Documentation

- **Threshold Strategy Pattern**: `../concept.md`
- **All Strategy Flows**: `../flows.md`
- **Local Threshold**: `../local/flows.md`
- **Adaptive Threshold**: `../adaptive/flows.md`

---

## Summary

Global Threshold Strategy:

1. **Sorts all bubble values** from darkest to lightest
2. **Finds largest gap** between consecutive groups
3. **Places threshold at midpoint** of gap
4. **Returns confidence** based on gap size
5. **Falls back to default** if no significant gap found

**Best For**: Uniform lighting, baseline threshold for file
**Limitations**: Doesn't adapt to per-field variations
**Typical Confidence**: 0.8-1.0 for good scans, 0.3-0.5 for poor scans
