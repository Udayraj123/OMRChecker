# Shift Detection - Concept

## Overview

The Shift Detection system is a validation and correction layer that applies ML-detected field block position shifts to the template, validates the results against baseline detection, and adjusts confidence based on discrepancies. It acts as a quality assurance mechanism between ML-based field block detection and traditional OMR detection.

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                  ShiftDetectionProcessor                    │
│                                                             │
│  1. Reads ML shifts from context                           │
│  2. Validates shifts against configured margins            │
│  3. Runs detection with shifts (shifted version)           │
│  4. Runs detection without shifts (baseline)               │
│  5. Compares results and adjusts confidence                │
│  6. Stores final results in context                        │
└─────────────────────────────────────────────────────────────┘
         ↑                                    ↓
         │ ml_block_alignments                │ shift_detection metadata
         │                                    │
┌────────┴──────────┐              ┌─────────┴──────────┐
│ MLFieldBlock      │              │  ProcessingContext │
│ Detector          │              │  + final results   │
└───────────────────┘              └────────────────────┘
```

### Data Flow

```
Input Context (from MLFieldBlockDetector)
  ├── metadata.ml_block_alignments: {
  │     "MCQBlock1a1": { "shift": [dx, dy] },
  │     "MCQBlock1a2": { "shift": [dx, dy] }
  │   }
  │
  ↓
Shift Validation
  ├── Check against global_max_shift_pixels
  ├── Check per-block overrides
  └── Filter out invalid shifts
  │
  ↓
Parallel Detection
  ├── Apply shifts → Run detection → Shifted results
  └── No shifts   → Run detection → Baseline results
  │
  ↓
Result Comparison
  ├── Bubble-level comparison (bubble_values)
  ├── Field-level comparison (response)
  └── Calculate mismatch severity (0.0 to 1.0)
  │
  ↓
Confidence Adjustment
  ├── severity = mismatches / total_bubbles
  ├── reduction = min + severity * (max - min)
  └── adjusted_conf = original_conf - reduction
  │
  ↓
Output Context
  ├── field_id_to_interpretation (with adjusted confidence)
  ├── omr_response
  ├── is_multi_marked
  └── metadata.shift_detection: {
        "applied_shifts": {...},
        "comparison": {...},
        "confidence_adjustments": {...}
      }
```

## Key Entities

### 1. ShiftDetectionProcessor

**Purpose**: Orchestrates the shift detection and validation workflow.

**Responsibilities**:
- Read ML shifts from context
- Validate shifts against configured constraints
- Run detection with and without shifts
- Compare results and compute confidence adjustments
- Store final results in context

**Dependencies**:
- `Template`: For accessing field blocks and their configurations
- `ShiftDetectionConfig`: Configuration for shift validation and confidence adjustment
- `TemplateFileRunner`: For running traditional OMR detection

### 2. ShiftDetectionConfig

**Purpose**: Configuration for shift detection behavior.

**Fields**:
```python
@dataclass
class ShiftDetectionConfig:
    enabled: bool = False
    global_max_shift_pixels: int = 50
    per_block_max_shift_pixels: dict[str, int] = field(default_factory=dict)

    # Confidence adjustment
    confidence_reduction_min: float = 0.1
    confidence_reduction_max: float = 0.5

    # Comparison thresholds (for logging/flagging)
    bubble_mismatch_threshold: int = 3
    field_mismatch_threshold: int = 1
```

**Configuration Levels**:
1. Global max shift: Default limit for all field blocks
2. Per-block overrides: More restrictive or permissive limits for specific blocks
3. Confidence reduction: Linear interpolation between min and max based on severity

### 3. Shift Data Structure

**ML Block Alignments** (input):
```python
ml_block_alignments = {
    "MCQBlock1a1": {
        "shift": [dx, dy],          # Pixel shift from expected position
        "confidence": 0.95,          # ML detection confidence
        "bbox": [x, y, w, h]         # Detected bounding box
    }
}
```

**Validated Shifts** (internal):
```python
validated_shifts = {
    "MCQBlock1a1": {"dx": 20, "dy": 30},
    "MCQBlock1a2": {"dx": -15, "dy": 25}
}
```

**Shift Detection Metadata** (output):
```python
metadata["shift_detection"] = {
    "applied_shifts": {
        "MCQBlock1a1": {"dx": 20, "dy": 30}
    },
    "comparison": {
        "bubble_mismatches": [
            {
                "field_id": "q1",
                "count": 2,
                "bubbles": [
                    {"index": 1, "shifted": 1, "baseline": 0},
                    {"index": 4, "shifted": 0, "baseline": 1}
                ]
            }
        ],
        "field_mismatches": [
            {"field_id": "q1", "shifted_response": "B", "baseline_response": "A"}
        ],
        "confidence_reductions": {
            "q1": {
                "original": 0.95,
                "reduction": 0.2,
                "final": 0.75,
                "reason": "shift_mismatch"
            }
        }
    },
    "confidence_adjustments": {...}
}
```

## Core Algorithms

### 1. Shift Validation

**Purpose**: Filter out shifts that exceed configured safety margins.

```python
def _validate_shifts(ml_alignments: dict) -> dict:
    validated = {}

    for block_name, shift_data in ml_alignments.items():
        dx, dy = shift_data["shift"]

        # Get max allowed shift (per-block override or global)
        max_shift = per_block_max_shift_pixels.get(
            block_name,
            global_max_shift_pixels
        )

        # Check magnitude using Euclidean distance
        shift_magnitude = sqrt(dx^2 + dy^2)

        if shift_magnitude <= max_shift:
            validated[block_name] = {"dx": dx, "dy": dy}

    return validated
```

**Rationale**:
- Prevents extreme shifts that might indicate ML detection errors
- Per-block overrides allow fine-tuning for blocks with known issues
- Euclidean distance accounts for diagonal shifts

### 2. Dual Detection Strategy

**Purpose**: Run detection twice to compare shifted vs non-shifted results.

```python
def process(context: ProcessingContext):
    # Apply shifts to template
    for block_name, shift in validated_shifts.items():
        block = find_block_by_name(block_name)
        block.shifts = [shift["dx"], shift["dy"]]

    # Run detection with shifts
    shifted_results = run_traditional_detection(context)

    # Reset shifts
    for block in template.field_blocks:
        block.reset_all_shifts()

    # Run detection without shifts (baseline)
    baseline_results = run_traditional_detection(context)

    # Compare and adjust
    return compare_and_adjust_confidence(shifted_results, baseline_results)
```

**Why Dual Detection**:
- Validates that ML shifts actually improve detection accuracy
- Detects cases where shifts might worsen results (ML false positives)
- Provides confidence metric based on result stability

### 3. Confidence Adjustment Algorithm

**Purpose**: Reduce confidence proportionally to mismatch severity.

```python
def _calculate_confidence_reduction(severity: float) -> float:
    """Linear interpolation between min and max reduction."""
    min_reduction = 0.1  # 10% reduction for minor mismatches
    max_reduction = 0.5  # 50% reduction for severe mismatches

    return min_reduction + (severity * (max_reduction - min_reduction))

# Severity calculation
total_bubbles = len(bubble_values)
mismatched_bubbles = count_different_bubbles(shifted, baseline)
severity = mismatched_bubbles / total_bubbles  # 0.0 to 1.0

# Apply reduction
reduction = _calculate_confidence_reduction(severity)
adjusted_conf = max(0.0, original_conf - reduction)
```

**Examples**:
- 0% mismatch → 10% reduction (minimum uncertainty)
- 25% mismatch → 20% reduction
- 50% mismatch → 30% reduction
- 100% mismatch → 50% reduction (maximum uncertainty)

## Integration Points

### 1. Pipeline Position

```
ProcessingPipeline:
  1. Preprocessing (rotate, crop, filters)
  2. Alignment (SIFT/phase correlation)
  3. MLFieldBlockDetector ← Populates ml_block_alignments
  4. ShiftDetectionProcessor ← Validates and applies shifts
  5. Detection/Interpretation ← Uses adjusted positions
  6. Training Data Collection
```

### 2. Context Flow

```python
# MLFieldBlockDetector output
context.metadata["ml_block_alignments"] = {
    "MCQBlock1a1": {"shift": [20, 30], "confidence": 0.95}
}

# ShiftDetectionProcessor input/output
result_context = ShiftDetectionProcessor.process(context)
result_context.field_id_to_interpretation  # Final interpretations
result_context.metadata["shift_detection"]  # Validation metadata
```

### 3. Template Mutation

The processor temporarily mutates the template's field blocks:

```python
# Before detection
block.shifts = [0, 0]

# During shifted detection
block.shifts = [20, 30]
run_detection()

# Reset after detection
block.reset_all_shifts()

# During baseline detection
block.shifts = [0, 0]
run_detection()
```

## Design Decisions

### Why Validate Shifts?

**Problem**: ML detectors can produce spurious shifts due to:
- Low-quality images
- Partial occlusions
- Similar-looking regions
- Training data distribution mismatch

**Solution**: Configurable safety margins prevent extreme shifts from corrupting results.

### Why Dual Detection?

**Problem**: Trusting ML shifts blindly can worsen accuracy if:
- ML model is wrong
- Shift is correct but causes alignment issues
- Template positions are already accurate

**Solution**: Compare shifted vs baseline results to validate improvement.

### Why Linear Confidence Reduction?

**Problem**: Binary decisions (accept/reject shift) lose information.

**Solution**: Proportional confidence adjustment:
- Preserves results with minor discrepancies
- Flags severe discrepancies with low confidence
- Allows downstream systems to make informed decisions

### Why Not Always Use Shifts?

**Problem**: Shifts add complexity and processing time.

**Solution**:
- `enabled` flag allows disabling when not needed
- Skip processing if no ML alignments present
- Skip if all shifts rejected during validation

## Edge Cases

### 1. No ML Alignments

```python
if not ml_alignments or not shift_config.enabled:
    return context  # Skip processing
```

**Scenario**: ML field block detector disabled or no blocks detected.

### 2. All Shifts Rejected

```python
validated_shifts = _validate_shifts(ml_alignments)
if not validated_shifts:
    return context  # No valid shifts to apply
```

**Scenario**: All shifts exceed configured margins.

### 3. Missing Field Block

```python
block = _find_block_by_name(block_name)
if block:
    block.shifts = [shift["dx"], shift["dy"]]
```

**Scenario**: ML detector finds block not in template (possible with flexible templates).

### 4. Identical Results

```python
if not bubble_diffs and not field_diff:
    # No confidence adjustment needed
    pass
```

**Scenario**: Shifted and baseline results are identical (shift had no effect).

### 5. Partial Block Coverage

```python
# Only some blocks have ML alignments
ml_alignments = {
    "MCQBlock1a1": {"shift": [20, 30]}
    # MCQBlock1a2 not detected by ML
}
# Only MCQBlock1a1 gets shifted, others use original positions
```

## Statistics Tracking

```python
self.stats = {
    "shifts_applied": 0,       # Number of validated shifts applied
    "shifts_rejected": 0,      # Number of shifts rejected (exceeds margin)
    "mismatches_detected": 0,  # Number of fields with discrepancies
    "confidence_reductions": []  # List of reduction values
}
```

**Usage**:
- Debugging: Identify problematic blocks or images
- Tuning: Adjust margins based on rejection rates
- Monitoring: Track confidence adjustment distribution

## Browser Migration Considerations

### Challenges

1. **No Python**: Port ShiftDetectionProcessor to TypeScript
2. **No NumPy**: Use TypedArrays for shift calculations
3. **Memory**: Store only final results, not both shifted and baseline
4. **Performance**: Dual detection doubles processing time

### Migration Strategy

```typescript
interface ShiftDetectionConfig {
  enabled: boolean;
  globalMaxShiftPixels: number;
  perBlockMaxShiftPixels: Record<string, number>;
  confidenceReductionMin: number;
  confidenceReductionMax: number;
}

class ShiftDetectionProcessor implements Processor {
  private validateShifts(mlAlignments: MLBlockAlignments): ValidatedShifts {
    // Use Math.sqrt for magnitude calculation
    const magnitude = Math.sqrt(dx * dx + dy * dy);
    return magnitude <= maxShift;
  }

  private async runDetectionWithShifts(
    context: ProcessingContext,
    shifts: ValidatedShifts
  ): Promise<DetectionResults> {
    // Apply shifts to field blocks
    // Run traditional detection
    // Reset shifts
  }

  private calculateConfidenceReduction(severity: number): number {
    const { confidenceReductionMin, confidenceReductionMax } = this.config;
    return confidenceReductionMin +
           severity * (confidenceReductionMax - confidenceReductionMin);
  }
}
```

### Optimization Opportunities

1. **Skip Baseline**: If ML confidence is very high, skip baseline detection
2. **Parallel Detection**: Run shifted and baseline in Web Workers
3. **Incremental Comparison**: Compare results bubble-by-bubble to early-exit
4. **Confidence Caching**: Cache confidence adjustments per template

### Browser-Specific Adaptations

```typescript
// Use performance.now() for timing
const start = performance.now();
const results = await processDetection();
const duration = performance.now() - start;

// Store metadata for debugging in IndexedDB
await db.put('shift_metadata', {
  fileId: context.fileId,
  appliedShifts: validatedShifts,
  comparison: comparisonMeta,
  timestamp: Date.now()
});

// Emit progress events
this.emit('shift-detection-progress', {
  stage: 'validation',
  shiftsValidated: validatedShifts.length,
  shiftsRejected: rejectedCount
});
```

## Performance Characteristics

### Time Complexity

- **Shift Validation**: O(B) where B = number of blocks
- **Dual Detection**: 2 × O(D) where D = detection complexity
- **Comparison**: O(F × N) where F = fields, N = bubbles per field

**Total**: O(B + 2D + F×N)

### Memory Complexity

- **Shifted Results**: O(F × N) for all field interpretations
- **Baseline Results**: O(F × N) for all field interpretations
- **Comparison Metadata**: O(F × M) where M = average mismatches

**Peak Memory**: 2 × O(F × N) during comparison

### Optimization Strategies

1. **Early Exit**: Skip baseline if shifted confidence is very low
2. **Lazy Comparison**: Only compare fields with low confidence
3. **Streaming**: Process fields one at a time instead of batch
4. **Selective Shifts**: Only apply shifts to blocks with low alignment confidence

## Related Systems

1. **MLFieldBlockDetector**: Produces ml_block_alignments input
2. **TemplateFileRunner**: Executes traditional detection
3. **Alignment System**: Alternative to shift detection for global alignment
4. **Field Blocks**: Targets of shift application
5. **ProcessingContext**: Container for shift metadata
