# Shift Detection - Flows

## Overview

This document details the step-by-step flows for shift detection, validation, and confidence adjustment. The system implements a dual-detection strategy with result comparison.

## Main Flow: Shift Detection Process

```
┌──────────────────────────────────────────────────────────────┐
│ 1. Check Preconditions                                      │
│    - ml_block_alignments present in context?               │
│    - shift_config.enabled == true?                          │
│    └─→ No → Return context unchanged                        │
│    └─→ Yes → Continue to step 2                            │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ 2. Validate Shifts                                          │
│    For each block in ml_block_alignments:                   │
│      a. Extract dx, dy from shift data                      │
│      b. Get max_shift (per-block override or global)        │
│      c. Calculate magnitude = sqrt(dx² + dy²)               │
│      d. If magnitude <= max_shift:                          │
│           - Add to validated_shifts                         │
│           - Increment stats.shifts_applied                  │
│         Else:                                               │
│           - Log warning                                     │
│           - Increment stats.shifts_rejected                 │
│    └─→ validated_shifts dictionary                          │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ 3. Check Validated Shifts                                   │
│    - validated_shifts not empty?                            │
│    └─→ No → Return context unchanged                        │
│    └─→ Yes → Continue to step 4                            │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ 4. Run Detection With Shifts                                │
│    a. Apply shifts to field blocks:                         │
│         block.shifts = [shift["dx"], shift["dy"]]           │
│    b. Run traditional detection:                            │
│         omr_response, is_multi_marked, interpretations      │
│    c. Store shifted_results                                 │
│    d. Reset all shifts:                                     │
│         block.reset_all_shifts()                            │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ 5. Run Detection Without Shifts (Baseline)                  │
│    a. Ensure all shifts are reset (already done in step 4d) │
│    b. Run traditional detection:                            │
│         omr_response, is_multi_marked, interpretations      │
│    c. Store baseline_results                                │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ 6. Compare Results and Adjust Confidence                    │
│    a. For each field in shifted_results:                    │
│         - Compare bubble values                             │
│         - Compare field responses                           │
│         - Calculate mismatch severity                       │
│         - Adjust confidence if mismatches found             │
│    b. Collect comparison metadata                           │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ 7. Update Context                                           │
│    context.field_id_to_interpretation = final_results       │
│    context.omr_response = shifted_results.omr_response      │
│    context.is_multi_marked = shifted_results.is_multi_marked│
│    context.metadata["shift_detection"] = {                  │
│      "applied_shifts": validated_shifts,                    │
│      "comparison": comparison_meta,                         │
│      "confidence_adjustments": ...                          │
│    }                                                        │
│    └─→ Return updated context                               │
└──────────────────────────────────────────────────────────────┘
```

## Flow 1: Shift Validation

**Purpose**: Filter out shifts that exceed safety margins.

### Step-by-Step

```python
def _validate_shifts(ml_alignments: dict) -> dict:
    """
    Input:  ml_alignments = {
              "MCQBlock1a1": {"shift": [20, 30], "confidence": 0.95},
              "MCQBlock1a2": {"shift": [40, 40], "confidence": 0.85}
            }
    Output: validated_shifts = {
              "MCQBlock1a1": {"dx": 20, "dy": 30}
            }
    """
    validated = {}

    for block_name, shift_data in ml_alignments.items():
        # Step 1: Extract shift components
        dx, dy = shift_data["shift"]

        # Step 2: Get max allowed shift
        max_shift = config.per_block_max_shift_pixels.get(
            block_name,
            config.global_max_shift_pixels  # Default: 50 pixels
        )

        # Step 3: Calculate Euclidean magnitude
        shift_magnitude = vector_magnitude([dx, dy])
        # = sqrt(dx^2 + dy^2)

        # Step 4: Validate against margin
        if shift_magnitude <= max_shift:
            # Accept shift
            validated[block_name] = {"dx": dx, "dy": dy}
            stats["shifts_applied"] += 1
            logger.debug(f"Shift for {block_name} validated: ({dx:.1f}, {dy:.1f})px")
        else:
            # Reject shift
            logger.warning(
                f"Shift for {block_name} rejected: "
                f"{shift_magnitude:.1f}px > {max_shift}px"
            )
            stats["shifts_rejected"] += 1

    return validated
```

### Example Scenarios

**Scenario 1: All shifts valid**
```python
ml_alignments = {
    "MCQBlock1a1": {"shift": [20, 30]},  # magnitude = 36.1
    "MCQBlock1a2": {"shift": [-15, 25]}  # magnitude = 29.2
}
global_max_shift_pixels = 50

# Result: Both accepted
validated_shifts = {
    "MCQBlock1a1": {"dx": 20, "dy": 30},
    "MCQBlock1a2": {"dx": -15, "dy": 25}
}
```

**Scenario 2: Mixed validation**
```python
ml_alignments = {
    "MCQBlock1a1": {"shift": [20, 20]},  # magnitude = 28.3
    "MCQBlock1a2": {"shift": [40, 40]}   # magnitude = 56.6
}
global_max_shift_pixels = 30

# Result: First accepted, second rejected
validated_shifts = {
    "MCQBlock1a1": {"dx": 20, "dy": 20}
}
```

**Scenario 3: Per-block override**
```python
ml_alignments = {
    "MCQBlock1a1": {"shift": [15, 15]},  # magnitude = 21.2
    "MCQBlock1a2": {"shift": [30, 30]},  # magnitude = 42.4
    "Booklet_No": {"shift": [20, 20]}    # magnitude = 28.3
}
global_max_shift_pixels = 30
per_block_max_shift_pixels = {
    "MCQBlock1a1": 20,   # More restrictive
    "MCQBlock1a2": 50    # More permissive
}

# Result:
validated_shifts = {
    # MCQBlock1a1: REJECTED (21.2 > 20)
    "MCQBlock1a2": {"dx": 30, "dy": 30},   # 42.4 < 50 ✓
    "Booklet_No": {"dx": 20, "dy": 20}     # 28.3 < 30 ✓
}
```

## Flow 2: Detection with Shifts

**Purpose**: Apply validated shifts and run traditional detection.

### Step-by-Step

```python
def _run_detection_with_shifts(context: ProcessingContext, shifts: dict) -> dict:
    # Step 1: Apply shifts to template blocks
    for block_name, shift in shifts.items():
        block = _find_block_by_name(block_name)
        if block:
            block.shifts = [shift["dx"], shift["dy"]]
            logger.debug(f"Applied shift to {block_name}: ({shift['dx']}, {shift['dy']})")

    # Step 2: Run traditional detection with shifted positions
    # This uses TemplateFileRunner which processes all field blocks
    results = _run_traditional_detection(context)
    # results = {
    #   "interpretations": { "q1": {...}, "q2": {...} },
    #   "omr_response": { "q1": "A", "q2": "B" },
    #   "is_multi_marked": { "q1": False, "q2": True }
    # }

    # Step 3: Reset shifts to prevent contamination
    for block in template.field_blocks:
        block.reset_all_shifts()

    return results
```

### Shift Application Example

```python
# Before shift
block = FieldBlock("MCQBlock1a1")
block.origin = [100, 200]
block.shifts = [0, 0]
block.get_shifted_origin()  # → [100, 200]

# Apply shift
block.shifts = [20, 30]
block.get_shifted_origin()  # → [120, 230]

# All bubble positions are recalculated relative to shifted origin
field = block.fields[0]
bubble = field.scan_boxes[0]
bubble.shifted_position  # Updated automatically
```

### Detection Workflow

```
Template with shifted blocks
        ↓
TemplateFileRunner.read_omr_and_update_metrics()
        ↓
TemplateDetectionPass (coordinates all field types)
        ↓
BubblesThresholdFileRunner.detect_and_interpret()
        ↓
├─ DetectionPass: Extract bubble ROIs using shifted positions
└─ InterpretationPass: Threshold and interpret bubbles
        ↓
Return (omr_response, is_multi_marked, interpretations)
```

## Flow 3: Detection without Shifts (Baseline)

**Purpose**: Run traditional detection with original positions for comparison.

### Step-by-Step

```python
def _run_detection_without_shifts(context: ProcessingContext) -> dict:
    # Step 1: Ensure all shifts are reset (defensive)
    # (Already done in previous step, but we double-check)
    for block in template.field_blocks:
        block.shifts = [0, 0]

    # Step 2: Run traditional detection with original positions
    results = _run_traditional_detection(context)

    return results
```

### Why Baseline Detection?

**Purpose**: Establish a reference for validating shift effectiveness.

**Scenarios**:
1. **Shift improves results**: Shifted and baseline differ, shifted has higher confidence
2. **Shift has no effect**: Shifted and baseline identical (shift too small or already aligned)
3. **Shift worsens results**: Shifted and baseline differ significantly (ML error)

**Example**:
```python
# Scenario: ML shift improves detection
baseline_response = {"q1": "A", "q2": "B", "q3": ""}  # Empty due to misalignment
shifted_response = {"q1": "A", "q2": "B", "q3": "C"}  # Fixed with shift

# Comparison: q3 differs, but shifted has higher confidence
# Result: Use shifted, adjust confidence slightly for q3
```

## Flow 4: Result Comparison and Confidence Adjustment

**Purpose**: Compare shifted vs baseline results and adjust confidence based on discrepancies.

### Step-by-Step

```python
def _compare_and_adjust_confidence(
    shifted_results: dict,
    baseline_results: dict,
    shifts: dict
) -> tuple[dict, dict]:
    """
    Input:
      shifted_results = {
        "interpretations": {
          "q1": {"response": "A", "bubble_values": [0,1,0,0], "confidence": 0.95},
          "q2": {"response": "B", "bubble_values": [0,0,1,0], "confidence": 0.90}
        }
      }
      baseline_results = {
        "interpretations": {
          "q1": {"response": "A", "bubble_values": [0,1,0,0], "confidence": 0.95},
          "q2": {"response": "A", "bubble_values": [1,0,0,0], "confidence": 0.85}
        }
      }

    Output:
      final_results = shifted_results (with adjusted confidence)
      comparison_meta = {...}
    """

    comparison = {
        "bubble_mismatches": [],
        "field_mismatches": [],
        "confidence_reductions": {}
    }

    shifted_interpretations = shifted_results["interpretations"]
    baseline_interpretations = baseline_results["interpretations"]

    # Step 1: Iterate through all fields
    for field_id in shifted_interpretations:
        shifted_interp = shifted_interpretations[field_id]
        baseline_interp = baseline_interpretations.get(field_id)

        if not baseline_interp:
            continue  # Skip if field missing in baseline

        # Step 2: Compare bubble values
        bubble_diffs = _compare_bubbles(shifted_interp, baseline_interp)
        # bubble_diffs = [
        #   {"index": 2, "shifted": 1, "baseline": 0},
        #   {"index": 0, "shifted": 0, "baseline": 1}
        # ]

        # Step 3: Compare field responses
        field_diff = _compare_field_responses(shifted_interp, baseline_interp)
        # field_diff = {"shifted": "B", "baseline": "A"} or None

        # Step 4: Calculate mismatch severity
        total_bubbles = len(shifted_interp.get("bubble_values", []))
        severity = (
            len(bubble_diffs) / max(total_bubbles, 1)
            if total_bubbles > 0
            else 0.0
        )
        # severity = 2/4 = 0.5 (50% of bubbles differ)

        # Step 5: Record mismatches
        if bubble_diffs:
            comparison["bubble_mismatches"].append({
                "field_id": field_id,
                "count": len(bubble_diffs),
                "bubbles": bubble_diffs
            })

        if field_diff:
            comparison["field_mismatches"].append({
                "field_id": field_id,
                "shifted_response": shifted_interp.get("response"),
                "baseline_response": baseline_interp.get("response")
            })

        # Step 6: Adjust confidence if mismatches found
        if bubble_diffs or field_diff:
            reduction = _calculate_confidence_reduction(severity)
            # reduction = 0.1 + 0.5 * (0.5 - 0.1) = 0.1 + 0.2 = 0.3

            original_conf = shifted_interp.get("confidence", 1.0)
            adjusted_conf = max(0.0, original_conf - reduction)
            # adjusted_conf = 0.95 - 0.3 = 0.65

            # Update confidence in place
            shifted_interp["confidence"] = adjusted_conf

            comparison["confidence_reductions"][field_id] = {
                "original": original_conf,
                "reduction": reduction,
                "final": adjusted_conf,
                "reason": "shift_mismatch"
            }

            stats["mismatches_detected"] += 1
            stats["confidence_reductions"].append(reduction)

    # Step 7: Return adjusted results
    return shifted_results, comparison
```

### Bubble Comparison Algorithm

```python
def _compare_bubbles(shifted: dict, baseline: dict) -> list:
    """Compare individual bubble states."""
    diffs = []
    shifted_bubbles = shifted.get("bubble_values", [])
    baseline_bubbles = baseline.get("bubble_values", [])

    # Zip and compare element-wise
    for i, (s_val, b_val) in enumerate(zip(shifted_bubbles, baseline_bubbles, strict=True)):
        if s_val != b_val:
            diffs.append({
                "index": i,
                "shifted": s_val,
                "baseline": b_val
            })

    return diffs
```

**Example**:
```python
shifted_bubbles  = [0, 1, 0, 1, 0]
baseline_bubbles = [0, 0, 0, 1, 1]
# Differences at indices 1 and 4

diffs = [
    {"index": 1, "shifted": 1, "baseline": 0},
    {"index": 4, "shifted": 0, "baseline": 1}
]
```

### Field Response Comparison

```python
def _compare_field_responses(shifted: dict, baseline: dict) -> dict | None:
    """Compare field-level responses."""
    shifted_resp = shifted.get("response")
    baseline_resp = baseline.get("response")

    if shifted_resp != baseline_resp:
        return {"shifted": shifted_resp, "baseline": baseline_resp}
    return None
```

**Example**:
```python
shifted_resp = "B"
baseline_resp = "A"
# Result: {"shifted": "B", "baseline": "A"}

shifted_resp = "A"
baseline_resp = "A"
# Result: None (no difference)
```

### Confidence Reduction Calculation

```python
def _calculate_confidence_reduction(severity: float) -> float:
    """Linear interpolation between min and max reduction."""
    min_reduction = 0.1  # config.confidence_reduction_min
    max_reduction = 0.5  # config.confidence_reduction_max

    # Linear interpolation
    return min_reduction + (severity * (max_reduction - min_reduction))
```

**Examples**:
```python
severity = 0.0   → reduction = 0.1 + 0.0 * 0.4 = 0.10 (10%)
severity = 0.25  → reduction = 0.1 + 0.25 * 0.4 = 0.20 (20%)
severity = 0.5   → reduction = 0.1 + 0.5 * 0.4 = 0.30 (30%)
severity = 0.75  → reduction = 0.1 + 0.75 * 0.4 = 0.40 (40%)
severity = 1.0   → reduction = 0.1 + 1.0 * 0.4 = 0.50 (50%)
```

## Flow 5: ML-Based Shift Prediction (MLFieldBlockDetector)

**Purpose**: Generate shift predictions that feed into ShiftDetectionProcessor.

### Step-by-Step in MLFieldBlockDetector

```python
def process(context: ProcessingContext) -> ProcessingContext:
    # Step 1: Run YOLO field block detection
    results = model.predict(context.gray_image, conf=confidence_threshold)

    # Step 2: Match detected blocks to template blocks
    ml_alignments = {}
    for detection in results:
        bbox = detection.bbox  # [x, y, w, h]
        class_id = detection.class_id
        confidence = detection.confidence

        # Find corresponding template block
        template_block = find_closest_template_block(bbox, class_id)

        # Step 3: Calculate shift from expected position
        detected_center = bbox_center(bbox)
        expected_center = bbox_center(template_block.bounding_box)

        dx = detected_center[0] - expected_center[0]
        dy = detected_center[1] - expected_center[1]

        # Step 4: Store alignment data
        ml_alignments[template_block.name] = {
            "shift": [dx, dy],
            "confidence": confidence,
            "bbox": bbox,
            "expected_bbox": template_block.bounding_box
        }

    # Step 5: Store in context for ShiftDetectionProcessor
    context.metadata["ml_block_alignments"] = ml_alignments

    return context
```

### Integration Flow

```
MLFieldBlockDetector (Pipeline Stage 3)
        ↓
  context.metadata["ml_block_alignments"] = {
    "MCQBlock1a1": {"shift": [20, 30], ...}
  }
        ↓
ShiftDetectionProcessor (Pipeline Stage 4)
        ↓
  Validate shifts → Apply shifts → Compare results
        ↓
  context.field_id_to_interpretation (with adjusted confidence)
```

## Complete Example: End-to-End Flow

### Input State

```python
# From MLFieldBlockDetector
context.metadata["ml_block_alignments"] = {
    "MCQBlock1a1": {"shift": [20, 30], "confidence": 0.95},
    "MCQBlock1a2": {"shift": [-15, 25], "confidence": 0.90}
}

# Configuration
shift_config = ShiftDetectionConfig(
    enabled=True,
    global_max_shift_pixels=50,
    confidence_reduction_min=0.1,
    confidence_reduction_max=0.5
)
```

### Processing Steps

**1. Validation**
```python
validated_shifts = {
    "MCQBlock1a1": {"dx": 20, "dy": 30},  # magnitude = 36.1 < 50 ✓
    "MCQBlock1a2": {"dx": -15, "dy": 25}  # magnitude = 29.2 < 50 ✓
}
```

**2. Shifted Detection**
```python
# Apply shifts
MCQBlock1a1.shifts = [20, 30]
MCQBlock1a2.shifts = [-15, 25]

# Run detection
shifted_results = {
    "interpretations": {
        "q1": {"response": "A", "bubble_values": [0,1,0,0], "confidence": 0.95},
        "q2": {"response": "B", "bubble_values": [0,0,1,0], "confidence": 0.90}
    },
    "omr_response": {"q1": "A", "q2": "B"},
    "is_multi_marked": {"q1": False, "q2": False}
}

# Reset shifts
MCQBlock1a1.reset_all_shifts()
MCQBlock1a2.reset_all_shifts()
```

**3. Baseline Detection**
```python
baseline_results = {
    "interpretations": {
        "q1": {"response": "A", "bubble_values": [0,1,0,0], "confidence": 0.95},
        "q2": {"response": "A", "bubble_values": [1,0,0,0], "confidence": 0.85}
    },
    "omr_response": {"q1": "A", "q2": "A"},
    "is_multi_marked": {"q1": False, "q2": False}
}
```

**4. Comparison**
```python
# q1: Identical → No adjustment
# q2: Different → Adjust confidence

# Bubble comparison for q2
bubble_diffs = [
    {"index": 0, "shifted": 0, "baseline": 1},
    {"index": 2, "shifted": 1, "baseline": 0}
]
severity = 2/4 = 0.5

# Field comparison for q2
field_diff = {"shifted": "B", "baseline": "A"}

# Confidence adjustment
reduction = 0.1 + 0.5 * (0.5 - 0.1) = 0.3
adjusted_conf = 0.90 - 0.3 = 0.60
```

**5. Final Results**
```python
context.field_id_to_interpretation = {
    "q1": {"response": "A", "bubble_values": [0,1,0,0], "confidence": 0.95},
    "q2": {"response": "B", "bubble_values": [0,0,1,0], "confidence": 0.60}  # Adjusted
}

context.metadata["shift_detection"] = {
    "applied_shifts": {
        "MCQBlock1a1": {"dx": 20, "dy": 30},
        "MCQBlock1a2": {"dx": -15, "dy": 25}
    },
    "comparison": {
        "bubble_mismatches": [
            {"field_id": "q2", "count": 2, "bubbles": [...]}
        ],
        "field_mismatches": [
            {"field_id": "q2", "shifted_response": "B", "baseline_response": "A"}
        ],
        "confidence_reductions": {
            "q2": {"original": 0.90, "reduction": 0.3, "final": 0.60, "reason": "shift_mismatch"}
        }
    }
}
```

## Browser Migration Flow

### TypeScript Implementation

```typescript
class ShiftDetectionProcessor implements Processor {
  async process(context: ProcessingContext): Promise<ProcessingContext> {
    // 1. Check preconditions
    const mlAlignments = context.metadata.mlBlockAlignments;
    if (!mlAlignments || !this.config.enabled) {
      return context;
    }

    // 2. Validate shifts
    const validatedShifts = this.validateShifts(mlAlignments);
    if (Object.keys(validatedShifts).length === 0) {
      return context;
    }

    // 3 & 4. Run detections in parallel using Web Workers
    const [shiftedResults, baselineResults] = await Promise.all([
      this.runDetectionWithShifts(context, validatedShifts),
      this.runDetectionWithoutShifts(context)
    ]);

    // 5. Compare and adjust
    const [finalResults, comparison] = this.compareAndAdjustConfidence(
      shiftedResults,
      baselineResults,
      validatedShifts
    );

    // 6. Update context
    context.fieldIdToInterpretation = finalResults.interpretations;
    context.omrResponse = finalResults.omrResponse;
    context.isMultiMarked = finalResults.isMultiMarked;
    context.metadata.shiftDetection = {
      appliedShifts: validatedShifts,
      comparison,
      confidenceAdjustments: comparison.confidenceReductions
    };

    return context;
  }
}
```

### Performance Optimization

```typescript
// Use Web Workers for parallel detection
async runParallelDetection(
  context: ProcessingContext,
  shifts: ValidatedShifts
): Promise<[DetectionResults, DetectionResults]> {
  const worker1 = new Worker('detection-worker.js');
  const worker2 = new Worker('detection-worker.js');

  const shiftedPromise = new Promise((resolve) => {
    worker1.postMessage({ context, shifts });
    worker1.onmessage = (e) => resolve(e.data);
  });

  const baselinePromise = new Promise((resolve) => {
    worker2.postMessage({ context, shifts: {} });
    worker2.onmessage = (e) => resolve(e.data);
  });

  const results = await Promise.all([shiftedPromise, baselinePromise]);

  worker1.terminate();
  worker2.terminate();

  return results;
}
```
