# Shift Detection - Decisions

## Overview

This document explains the key architectural and design decisions made in the Shift Detection system, along with their rationale, trade-offs, and alternatives considered.

## Decision 1: Dual Detection Strategy

### Decision

Run traditional detection twice (with and without shifts) and compare results, rather than blindly trusting ML-detected shifts.

### Rationale

**Problem**: ML field block detectors can produce incorrect shifts due to:
- Low-quality or degraded images
- Partial occlusions or artifacts
- Similar-looking regions (false positives)
- Distribution shift between training and real-world data

**Risks of Blind Trust**:
- Applying incorrect shifts can worsen detection accuracy
- No validation mechanism for ML predictions
- Silent failures when ML is wrong

**Benefits of Dual Detection**:
1. **Validation**: Confirms that shifts actually improve results
2. **Confidence Metric**: Quantifies result stability
3. **Fallback Safety**: Can revert to baseline if shifted results are worse
4. **Debugging**: Provides comparison data for tuning

### Trade-offs

**Advantages**:
- Robust to ML errors
- Self-validating system
- Confidence scores reflect uncertainty
- No silent failures

**Disadvantages**:
- 2× detection time (performance cost)
- 2× memory usage during comparison
- Additional complexity in pipeline
- May be overkill for high-confidence ML predictions

### Alternatives Considered

**Alternative 1: Trust ML blindly**
```python
# Simply apply shifts and proceed
for block_name, shift in ml_alignments.items():
    block.shifts = shift["shift"]
results = run_detection(context)
```
- ✅ Faster (1× detection)
- ✅ Simpler code
- ❌ No validation
- ❌ Silent failures
- **Rejected**: Too risky for production

**Alternative 2: ML confidence thresholding**
```python
if ml_confidence > 0.95:
    apply_shifts()
else:
    skip_shifts()
```
- ✅ Faster for high-confidence cases
- ✅ Some safety mechanism
- ❌ Binary decision loses information
- ❌ No validation even for high confidence
- **Rejected**: Still no validation, and threshold is arbitrary

**Alternative 3: Conditional dual detection**
```python
if ml_confidence < threshold:
    run_dual_detection()  # Validate low-confidence shifts
else:
    apply_shifts()  # Trust high-confidence shifts
```
- ✅ Balances speed and safety
- ❌ Complex logic with multiple code paths
- ❌ Still no validation for "high confidence" cases
- **Considered for future optimization**

### Current Implementation

```python
# Always run dual detection for validation
shifted_results = run_detection_with_shifts(context, validated_shifts)
baseline_results = run_detection_without_shifts(context)
final_results, comparison = compare_and_adjust_confidence(shifted_results, baseline_results)
```

### Future Optimizations

- **Skip baseline for very high ML confidence** (>0.98)
- **Parallel execution** of shifted and baseline detection
- **Early exit** if initial field comparisons show identity
- **Adaptive strategy** based on historical accuracy

---

## Decision 2: Linear Confidence Reduction

### Decision

Use linear interpolation between min and max confidence reduction based on mismatch severity, rather than binary accept/reject or exponential scaling.

### Rationale

**Problem**: Need to quantify uncertainty when shifted and baseline results differ.

**Options**:
1. **Binary**: Accept or reject shift entirely
2. **Linear**: Proportional reduction based on severity
3. **Exponential**: Aggressive reduction for even minor mismatches
4. **Logarithmic**: Gentle reduction with diminishing returns

**Why Linear**:
- Simple and predictable
- Easy to reason about and tune
- Balanced response to severity
- Preserves useful information even with discrepancies

### Formula

```python
reduction = min_reduction + (severity * (max_reduction - min_reduction))

# Where:
severity = mismatched_bubbles / total_bubbles  # 0.0 to 1.0
min_reduction = 0.1  # 10% reduction for minimal mismatch
max_reduction = 0.5  # 50% reduction for complete mismatch
```

### Examples

```python
# Scenario 1: 1 bubble differs out of 10 (10% severity)
severity = 0.1
reduction = 0.1 + 0.1 * (0.5 - 0.1) = 0.14
confidence = 0.95 - 0.14 = 0.81

# Scenario 2: 5 bubbles differ out of 10 (50% severity)
severity = 0.5
reduction = 0.1 + 0.5 * (0.5 - 0.1) = 0.30
confidence = 0.95 - 0.30 = 0.65

# Scenario 3: All bubbles differ (100% severity)
severity = 1.0
reduction = 0.1 + 1.0 * (0.5 - 0.1) = 0.50
confidence = 0.95 - 0.50 = 0.45
```

### Trade-offs

**Advantages**:
- Preserves results with minor discrepancies
- Gradual degradation proportional to mismatch
- Configurable min/max bounds
- Intuitive behavior

**Disadvantages**:
- May be too lenient for critical applications
- Linear may not match real-world confidence degradation
- Doesn't account for mismatch location (early vs late bubbles)

### Alternatives Considered

**Alternative 1: Binary accept/reject**
```python
if severity > threshold:
    reject_shift()  # Use baseline
else:
    accept_shift()  # Use shifted
```
- ✅ Simple
- ❌ Loses information
- ❌ Threshold selection is arbitrary
- **Rejected**: Too coarse-grained

**Alternative 2: Exponential reduction**
```python
reduction = max_reduction * (1 - e^(-k * severity))
```
- ✅ Aggressive for any mismatch
- ❌ Too punitive for minor differences
- ❌ Less intuitive
- **Rejected**: Overly conservative

**Alternative 3: Step function**
```python
if severity < 0.1: reduction = 0.1
elif severity < 0.3: reduction = 0.2
elif severity < 0.5: reduction = 0.3
else: reduction = 0.5
```
- ✅ Predictable buckets
- ❌ Discontinuous
- ❌ Arbitrary thresholds
- **Rejected**: Less flexible than linear

### Current Implementation

```python
def _calculate_confidence_reduction(severity: float) -> float:
    """Linear interpolation between min and max reduction."""
    min_reduction = self.shift_config.confidence_reduction_min  # 0.1
    max_reduction = self.shift_config.confidence_reduction_max  # 0.5
    return min_reduction + (severity * (max_reduction - min_reduction))
```

### Tuning Recommendations

**Conservative** (strict):
```python
confidence_reduction_min = 0.2  # 20% reduction even for 1 bubble diff
confidence_reduction_max = 0.8  # 80% reduction for major mismatches
```

**Lenient** (permissive):
```python
confidence_reduction_min = 0.05  # 5% reduction for minor diffs
confidence_reduction_max = 0.3   # 30% reduction for major mismatches
```

**Default** (balanced):
```python
confidence_reduction_min = 0.1  # 10%
confidence_reduction_max = 0.5  # 50%
```

---

## Decision 3: Per-Block Shift Margins

### Decision

Allow per-block override of global max shift pixels, enabling fine-grained control over shift validation.

### Rationale

**Problem**: Different field blocks have different reliability characteristics:
- Some blocks are stable (e.g., fixed-position headers)
- Some blocks are variable (e.g., handwritten sections that move)
- Some blocks are critical (e.g., student ID) and require strict validation
- Some blocks are less critical (e.g., optional fields)

**Why Per-Block Overrides**:
- Flexibility to tune per block type or location
- Accommodate templates with mixed reliability
- Stricter validation for critical blocks
- More permissive for known-variable blocks

### Implementation

```python
@dataclass
class ShiftDetectionConfig:
    global_max_shift_pixels: int = 50  # Default for all blocks
    per_block_max_shift_pixels: dict[str, int] = field(default_factory=dict)

# Configuration example
shift_config = ShiftDetectionConfig(
    global_max_shift_pixels=50,
    per_block_max_shift_pixels={
        "StudentID": 20,       # Critical: strict (more restrictive)
        "MCQBlock1": 50,       # Standard: use global
        "Handwritten": 80      # Variable: permissive (more lenient)
    }
)

# Validation logic
max_shift = config.per_block_max_shift_pixels.get(
    block_name,
    config.global_max_shift_pixels  # Fallback to global
)
```

### Trade-offs

**Advantages**:
- Fine-grained control
- Adapts to template characteristics
- Can encode domain knowledge
- No code changes needed for tuning

**Disadvantages**:
- More configuration complexity
- Requires knowledge of block reliability
- Potential for misconfiguration
- Harder to maintain across templates

### Alternatives Considered

**Alternative 1: Global margin only**
```python
global_max_shift_pixels: int = 50
# All blocks use same threshold
```
- ✅ Simple
- ✅ Consistent
- ❌ Inflexible
- ❌ Can't tune per block
- **Rejected**: Too rigid for real-world templates

**Alternative 2: Block type-based margins**
```python
max_shift_by_type = {
    "MCQ": 50,
    "OCR": 30,
    "Barcode": 70
}
```
- ✅ Automatic based on field type
- ❌ Doesn't account for position or criticality
- ❌ Same type may have different needs
- **Rejected**: Not granular enough

**Alternative 3: Position-based margins**
```python
# More permissive for blocks far from edges
max_shift = base_margin + (distance_from_edge * factor)
```
- ✅ Automatic based on geometry
- ❌ Complex logic
- ❌ May not match actual reliability
- **Rejected**: Overengineered

### Current Implementation

```python
def _validate_shifts(ml_alignments: dict) -> dict:
    for block_name, shift_data in ml_alignments.items():
        # Per-block override or global fallback
        max_shift = self.shift_config.per_block_max_shift_pixels.get(
            block_name,
            self.shift_config.global_max_shift_pixels
        )

        if shift_magnitude <= max_shift:
            validated[block_name] = {"dx": dx, "dy": dy}
        else:
            # Reject shift
            stats["shifts_rejected"] += 1
```

---

## Decision 4: Store Final Results as Shifted Version

### Decision

Store the shifted detection results (not baseline) as the final output, with adjusted confidence scores.

### Rationale

**Problem**: After running dual detection, must decide which results to use.

**Options**:
1. Always use shifted results
2. Always use baseline results
3. Choose better results based on some metric
4. Blend results

**Why Shifted Results**:
- ML shifts are intentional corrections for misalignment
- If shifts were validated, they likely improve accuracy
- Confidence adjustment handles uncertainty
- Consistent with trusting ML (with validation)

**Confidence Adjustment Compensates**:
- If shifted results differ from baseline, confidence is reduced
- Downstream systems can use confidence to make decisions
- Severely mismatched results will have very low confidence

### Example

```python
# Scenario: Shifted fixes a misalignment issue
baseline_response = {"q1": "A", "q2": "", "q3": "C"}  # q2 missed due to misalignment
shifted_response = {"q1": "A", "q2": "B", "q3": "C"}  # q2 detected after shift

# Comparison
# q1: identical → confidence unchanged
# q2: differs (empty vs B) → confidence reduced
# q3: identical → confidence unchanged

# Final output
final_results = shifted_response  # Use shifted
final_confidence = {
    "q1": 0.95,  # No change
    "q2": 0.60,  # Reduced due to mismatch
    "q3": 0.95   # No change
}
```

### Trade-offs

**Advantages**:
- Leverages ML improvements
- Confidence scores reflect uncertainty
- Single output (simpler downstream)

**Disadvantages**:
- May use incorrect results if ML is wrong
- No automatic fallback to baseline
- Requires downstream to check confidence

### Alternatives Considered

**Alternative 1: Choose better results per field**
```python
for field_id in all_fields:
    if shifted_conf > baseline_conf:
        use shifted
    else:
        use baseline
```
- ✅ Adaptive per field
- ❌ Mixing results from different detections
- ❌ Inconsistent (some shifted, some baseline)
- **Rejected**: Inconsistent state

**Alternative 2: Always use baseline**
```python
final_results = baseline_results
# Store shifted in metadata for debugging
```
- ✅ Conservative
- ❌ Wastes ML shift information
- ❌ Doesn't benefit from alignment corrections
- **Rejected**: Defeats purpose of shift detection

**Alternative 3: Weighted blend**
```python
final_response = (
    ml_confidence * shifted_response +
    (1 - ml_confidence) * baseline_response
)
```
- ✅ Gradual transition
- ❌ Doesn't work for discrete responses (A, B, C)
- ❌ Complex interpretation
- **Rejected**: Not applicable to discrete values

### Current Implementation

```python
# Store shifted results with adjusted confidence
context.field_id_to_interpretation = shifted_results["interpretations"]
context.omr_response = shifted_results["omr_response"]
context.is_multi_marked = shifted_results["is_multi_marked"]

# Store comparison for debugging/audit
context.metadata["shift_detection"] = {
    "applied_shifts": validated_shifts,
    "comparison": comparison_meta,  # Includes baseline for reference
    "confidence_adjustments": {...}
}
```

---

## Decision 5: Euclidean Distance for Shift Magnitude

### Decision

Use Euclidean distance (L2 norm) to calculate shift magnitude, rather than Manhattan distance or max component.

### Rationale

**Problem**: Shifts are 2D vectors (dx, dy). Need a scalar magnitude for validation.

**Options**:
1. **Euclidean**: `sqrt(dx² + dy²)`
2. **Manhattan**: `|dx| + |dy|`
3. **Max component**: `max(|dx|, |dy|)`
4. **Separate thresholds**: Check dx and dy independently

**Why Euclidean**:
- Natural metric for 2D distance
- Treats diagonal and cardinal shifts equally
- Standard in geometry and ML
- Matches physical intuition

### Examples

```python
shift = [30, 40]

# Euclidean distance
magnitude = sqrt(30² + 40²) = sqrt(900 + 1600) = 50

# Manhattan distance
magnitude = |30| + |40| = 70

# Max component
magnitude = max(|30|, |40|) = 40
```

### Trade-offs

**Euclidean**:
- ✅ Natural 2D metric
- ✅ Rotation-invariant
- ✅ Standard in ML/CV
- ❌ Slightly more expensive (sqrt)

**Manhattan**:
- ✅ Fast (no sqrt)
- ✅ Conservative (larger values)
- ❌ Treats diagonal differently
- ❌ Not rotation-invariant

**Max Component**:
- ✅ Very fast
- ❌ Ignores one dimension
- ❌ Can miss large diagonal shifts

### Current Implementation

```python
from src.utils.geometry import vector_magnitude

shift_magnitude = vector_magnitude([dx, dy])
# = sqrt(dx^2 + dy^2)

if shift_magnitude <= max_shift:
    validated[block_name] = {"dx": dx, "dy": dy}
```

---

## Decision 6: Reset Shifts After Detection

### Decision

Explicitly reset all field block shifts after running shifted detection, before running baseline detection.

### Rationale

**Problem**: Template field blocks are mutable. Shifts applied for one detection persist unless reset.

**Risks of Not Resetting**:
- Baseline detection would use shifted positions (contamination)
- Comparison would be invalid (comparing shifted vs shifted)
- Future detections would inherit shifts (state leakage)

**Why Explicit Reset**:
- Defensive programming
- Clear separation of shifted and baseline runs
- Prevents state leakage across images
- Easier to debug

### Implementation

```python
def _run_detection_with_shifts(context, shifts):
    # Apply shifts
    for block_name, shift in shifts.items():
        block = find_block_by_name(block_name)
        block.shifts = [shift["dx"], shift["dy"]]

    # Run detection
    results = run_traditional_detection(context)

    # CRITICAL: Reset shifts to prevent contamination
    for block in template.field_blocks:
        block.reset_all_shifts()

    return results

def _run_detection_without_shifts(context):
    # Defensive: Ensure shifts are reset (redundant but safe)
    for block in template.field_blocks:
        block.shifts = [0, 0]

    results = run_traditional_detection(context)
    return results
```

### Trade-offs

**Advantages**:
- Prevents state contamination
- Clear intent
- Safe for concurrent processing

**Disadvantages**:
- Redundant resets (defensive)
- Relies on mutable state

### Alternatives Considered

**Alternative 1: Clone template**
```python
shifted_template = deepcopy(template)
apply_shifts(shifted_template)
shifted_results = run_detection(context, shifted_template)

baseline_results = run_detection(context, template)
```
- ✅ No mutation
- ❌ Expensive (deep copy of template)
- ❌ Memory overhead
- **Rejected**: Performance cost too high

**Alternative 2: Functional shifts**
```python
def get_shifted_position(block, shift):
    return [block.origin[0] + shift[0], block.origin[1] + shift[1]]

# Pass shifts as parameter, don't mutate
results = run_detection(context, shifts=validated_shifts)
```
- ✅ No mutation
- ✅ Functional style
- ❌ Requires refactoring detection pipeline
- ❌ More complex API
- **Rejected**: Too invasive a change

### Current Implementation

**Mutation with explicit reset** is the pragmatic choice given existing architecture.

---

## Decision 7: Enable/Disable Flag

### Decision

Provide `enabled` flag in ShiftDetectionConfig to allow disabling shift detection entirely.

### Rationale

**Problem**: Shift detection adds overhead. Not all use cases need it.

**Use Cases for Disabling**:
- Templates with high alignment accuracy (shifts not needed)
- ML field block detector disabled
- Performance-critical applications
- Debugging (isolate shift detection impact)

**Why Flag Instead of Conditional Instantiation**:
- Simpler pipeline configuration
- No code changes needed
- Easy to toggle via config file
- Consistent interface

### Implementation

```python
@dataclass
class ShiftDetectionConfig:
    enabled: bool = False  # Default: disabled

def process(context: ProcessingContext):
    if not self.shift_config.enabled:
        return context  # Skip processing
```

### Configuration

```json
{
  "ml": {
    "shiftDetection": {
      "enabled": true,
      "globalMaxShiftPixels": 50
    }
  }
}
```

### Trade-offs

**Advantages**:
- Simple on/off switch
- No code changes
- Clear intent

**Disadvantages**:
- Processor still instantiated (minor overhead)
- Another config option to document

### Alternative: Conditional Pipeline

```python
# In pipeline setup
processors = [
    PreprocessingCoordinator(...),
    AlignmentProcessor(...),
]

if config.ml.shift_detection.enabled:
    processors.append(MLFieldBlockDetector(...))
    processors.append(ShiftDetectionProcessor(...))

processors.append(DetectionProcessor(...))
```

**Current approach is simpler**: Check enabled flag in process() method.

---

## Summary of Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Validation Strategy | Dual detection | Robust validation, confidence metric |
| Confidence Adjustment | Linear interpolation | Balanced, intuitive, configurable |
| Shift Margins | Per-block overrides | Fine-grained control, flexibility |
| Final Results | Shifted (with adjusted confidence) | Leverages ML improvements |
| Shift Magnitude | Euclidean distance | Natural 2D metric, standard |
| State Management | Explicit reset | Prevents contamination |
| Enablement | Config flag | Simple on/off, no code changes |

## Future Considerations

### Potential Improvements

1. **Adaptive Dual Detection**: Skip baseline for very high ML confidence
2. **Parallel Execution**: Run shifted and baseline in separate threads
3. **Confidence Blending**: Combine ML confidence with comparison confidence
4. **Historical Learning**: Track shift accuracy over time, adjust thresholds
5. **Field-Level Shifts**: Apply shifts per field instead of per block
6. **Non-Linear Confidence**: Experiment with exponential or logarithmic reduction
7. **Automatic Margin Tuning**: Learn optimal margins from validation data

### Open Questions

1. Should we ever prefer baseline over shifted results?
2. How to handle partial shifts (some blocks shifted, some not)?
3. Should confidence reduction account for ML confidence?
4. How to balance speed vs validation accuracy?
5. Should we cache shift patterns per template?
