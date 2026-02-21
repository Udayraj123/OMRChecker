# Shift Detection - Constraints

## Overview

This document outlines the constraints, limitations, and boundary conditions of the Shift Detection system. Understanding these constraints is critical for successful migration to browser environments and production deployment.

## Performance Constraints

### Time Complexity

**Dual Detection Overhead**:
```
Total Time = T_validation + 2 × T_detection + T_comparison

Where:
- T_validation ≈ O(B) where B = number of blocks (~1ms)
- T_detection ≈ O(F × N) where F = fields, N = bubbles (~100-500ms)
- T_comparison ≈ O(F × N) (~10-50ms)

Total ≈ 2 × T_detection (approximately doubles processing time)
```

**Example Timing** (Python, typical OMR sheet):
```
Validation:        1ms
Shifted Detection: 250ms
Baseline Detection: 250ms
Comparison:        20ms
---
Total:            521ms (vs 250ms without shift detection)
```

**Browser Impact**:
- Main thread blocking: ~500ms per image
- Use Web Workers to avoid UI freeze
- Consider progress indicators for batch processing

### Memory Constraints

**Peak Memory Usage**:
```
Memory = M_image + M_template + 2 × M_results + M_metadata

Where:
- M_image ≈ width × height × 3 (colored) + width × height (gray) (~5-10MB)
- M_template ≈ Field blocks + fields metadata (~1MB)
- M_results ≈ F × (bubble_values + metadata) (~100KB per result)
- M_metadata ≈ Comparison data (~50KB)

Peak ≈ M_image + M_template + 200KB
```

**Browser Limits**:
```javascript
// Typical browser memory limits
Mobile (Chrome): ~500MB per tab
Desktop (Chrome): ~2GB per tab
Safari (iOS): ~300MB (aggressive cleanup)

// With shift detection
Max concurrent images ≈ 5-10 on desktop, 2-3 on mobile
```

**Memory Management Strategy**:
```typescript
class ShiftDetectionProcessor {
  async process(context: ProcessingContext): Promise<ProcessingContext> {
    // Run shifted detection
    const shiftedResults = await this.runDetectionWithShifts(context, shifts);

    // Run baseline detection
    const baselineResults = await this.runDetectionWithoutShifts(context);

    // Compare and store final results
    const [finalResults, comparison] = this.compareAndAdjustConfidence(
      shiftedResults,
      baselineResults,
      shifts
    );

    // IMPORTANT: Free baseline results immediately after comparison
    // Keep only final results and comparison metadata
    // Don't retain both full result sets

    return this.updateContext(context, finalResults, comparison);
  }
}
```

### Computational Constraints

**CPU Intensive Operations**:
1. **Shift Magnitude Calculation**: O(B) square roots
2. **Traditional Detection**: 2× O(F × N) bubble thresholding
3. **Result Comparison**: O(F × N) element-wise comparison

**Browser Optimization**:
```javascript
// Use SIMD when available
if (typeof SharedArrayBuffer !== 'undefined') {
  // Use SharedArrayBuffer for parallel processing
}

// Use Web Workers for parallel detection
const workerPool = new WorkerPool(navigator.hardwareConcurrency || 4);
const [shifted, baseline] = await Promise.all([
  workerPool.execute('detection', { context, shifts }),
  workerPool.execute('detection', { context, shifts: {} })
]);
```

### Latency Constraints

**Real-Time Processing**:
```
Target: < 1 second per image (including all pipeline stages)

Shift Detection Budget: ~500ms
- Validation: 1ms
- Shifted Detection: 200ms
- Baseline Detection: 200ms
- Comparison: 50ms
- Overhead: 49ms

Critical: Can't exceed 500ms without impacting UX
```

**Optimization Strategies**:
1. **Skip baseline for high ML confidence** (>0.98)
2. **Early exit on identical results** (compare incrementally)
3. **Parallel detection** (Web Workers)
4. **Cache template processing** (reuse across images)

---

## Accuracy Constraints

### Shift Validation Margins

**Physical Limits**:
```
Global max shift: 50 pixels (default)

Rationale:
- Typical bubble spacing: 20-40 pixels
- Shift > 50px likely indicates:
  - ML detection error
  - Wrong template
  - Severe image distortion

Risk:
- Too strict → Reject valid shifts (false negatives)
- Too lenient → Accept spurious shifts (false positives)
```

**Recommended Margins** (based on bubble size):
```python
# Small bubbles (10-15px diameter)
global_max_shift_pixels = 30

# Medium bubbles (15-25px diameter)
global_max_shift_pixels = 50  # Default

# Large bubbles (25-40px diameter)
global_max_shift_pixels = 80
```

### Confidence Reduction Bounds

**Configuration Limits**:
```python
confidence_reduction_min: float = 0.1  # 10% minimum
confidence_reduction_max: float = 0.5  # 50% maximum

Rationale:
- Min > 0: Always reduce confidence for any mismatch (uncertainty exists)
- Max < 1.0: Don't completely zero confidence (shifted results might still be valid)
```

**Sensitivity Analysis**:
```python
# Too aggressive (not recommended)
confidence_reduction_min = 0.3  # 30%
confidence_reduction_max = 0.9  # 90%
# Problem: Even minor mismatches lead to very low confidence

# Too lenient (not recommended)
confidence_reduction_min = 0.01  # 1%
confidence_reduction_max = 0.1   # 10%
# Problem: Doesn't reflect actual uncertainty

# Balanced (default)
confidence_reduction_min = 0.1  # 10%
confidence_reduction_max = 0.5  # 50%
# Good for most use cases
```

### Comparison Granularity

**Bubble-Level Comparison**:
```python
# Constraint: Can only compare if bubble arrays are same length
if len(shifted_bubbles) != len(baseline_bubbles):
    # Skip comparison (template mismatch)
    logger.warning(f"Bubble count mismatch for {field_id}")
    continue
```

**Field-Level Comparison**:
```python
# Constraint: Responses must be comparable (same type)
shifted_resp = "A"  # String
baseline_resp = "A"  # String
# ✓ Comparable

shifted_resp = ["A", "B"]  # Multi-select
baseline_resp = "A"  # Single-select
# ✗ Not directly comparable (type mismatch)
```

---

## System Constraints

### Template Requirements

**Field Block Structure**:
```python
# REQUIRED: Field blocks must have unique names
template.field_blocks = [
    FieldBlock(name="MCQBlock1a1"),  # ✓ Unique
    FieldBlock(name="MCQBlock1a2"),  # ✓ Unique
    # FieldBlock(name="MCQBlock1a1"),  # ✗ Duplicate (causes issues)
]

# REQUIRED: Blocks must be findable by name
block = find_block_by_name("MCQBlock1a1")
assert block is not None
```

**Shift Application**:
```python
# REQUIRED: Blocks must support shift mutation
class FieldBlock:
    shifts: list[float, float]  # [dx, dy]

    def reset_all_shifts(self):
        self.shifts = [0, 0]

    def get_shifted_origin(self):
        return [self.origin[0] + self.shifts[0], self.origin[1] + self.shifts[1]]
```

### ML Integration Requirements

**Input Format**:
```python
# REQUIRED: MLFieldBlockDetector must populate ml_block_alignments
context.metadata["ml_block_alignments"] = {
    "MCQBlock1a1": {
        "shift": [dx, dy],        # REQUIRED
        "confidence": 0.95,        # Optional
        "bbox": [x, y, w, h]       # Optional
    }
}

# Constraint: Block names must match template field block names
# Constraint: Shifts must be numeric (int or float)
```

**Missing Data Handling**:
```python
# Case 1: No ml_block_alignments at all
if "ml_block_alignments" not in context.metadata:
    return context  # Skip shift detection

# Case 2: Empty alignments
if not context.metadata["ml_block_alignments"]:
    return context  # Skip shift detection

# Case 3: Partial alignments (some blocks detected, some not)
# ✓ Valid: Apply shifts only to detected blocks
```

### Detection Pipeline Requirements

**Processor Dependencies**:
```
Pipeline Order (REQUIRED):
1. Preprocessing
2. Alignment
3. MLFieldBlockDetector  ← Populates ml_block_alignments
4. ShiftDetectionProcessor  ← Consumes ml_block_alignments
5. (Optional) Other detection processors
```

**Context State**:
```python
# REQUIRED: Context must have gray_image for detection
assert context.gray_image is not None

# REQUIRED: Context must have colored_image for visualization
assert context.colored_image is not None

# REQUIRED: Template must be available
assert context.template is not None
```

---

## Browser-Specific Constraints

### JavaScript/TypeScript Limitations

**No Native Vector Operations**:
```typescript
// Python:
from src.utils.geometry import vector_magnitude
magnitude = vector_magnitude([dx, dy])

// Browser: Must implement manually
function vectorMagnitude(vec: [number, number]): number {
  const [dx, dy] = vec;
  return Math.sqrt(dx * dx + dy * dy);
}
```

**No Concurrent Execution** (without Web Workers):
```typescript
// Python: ThreadPoolExecutor provides true parallelism
with ThreadPoolExecutor() as executor:
    shifted_future = executor.submit(run_detection, context, True)
    baseline_future = executor.submit(run_detection, context, False)
    shifted, baseline = shifted_future.result(), baseline_future.result()

// Browser: Single-threaded unless using Web Workers
const shifted = await runDetection(context, true);   // Sequential
const baseline = await runDetection(context, false); // Sequential

// With Web Workers (parallel):
const [shifted, baseline] = await Promise.all([
  runDetectionInWorker(context, true),
  runDetectionInWorker(context, false)
]);
```

### Memory Management

**No Manual Garbage Collection**:
```typescript
// Python: Explicit deletion
del baseline_results  // Free memory immediately

// Browser: Rely on GC
// Best practice: Null references and let GC run
baselineResults = null;

// Force GC (not guaranteed, browser-dependent)
if (performance?.memory && performance.memory.usedJSHeapSize > threshold) {
  // Trigger GC by creating/releasing large objects
}
```

**ArrayBuffer Constraints**:
```typescript
// Max ArrayBuffer size (browser-dependent)
Chrome: ~2GB (desktop), ~500MB (mobile)
Firefox: ~2GB
Safari: ~1GB (desktop), ~300MB (mobile)

// Constraint: Large images may hit limits
const imageSize = width * height * 4; // RGBA
if (imageSize > maxArrayBufferSize) {
  // Downscale or tile processing
}
```

### Web Worker Constraints

**Data Transfer Overhead**:
```typescript
// Problem: Structured cloning for message passing
worker.postMessage({ context }); // Deep copy (expensive)

// Solution: Use Transferable objects
const imageBuffer = context.grayImage.buffer;
worker.postMessage({ context, imageBuffer }, [imageBuffer]);
// imageBuffer is transferred (zero-copy), but now unusable in main thread
```

**Worker Pool Limits**:
```typescript
// Constraint: Limited by CPU cores
const maxWorkers = navigator.hardwareConcurrency || 4;

// Don't create unlimited workers
const workerPool = new WorkerPool(Math.min(maxWorkers, 8));
```

### WASM Constraints

**OpenCV.js Limitations**:
```typescript
// Constraint: OpenCV.js is ~8MB (initial load)
// Mitigation: Lazy load only when needed

// Constraint: WASM memory limits
// Default: 16MB stack, 2GB heap (can be increased)
```

### Browser Storage Constraints

**IndexedDB Limits**:
```typescript
// Quota limits (browser-dependent)
Chrome: ~60% of free disk space
Firefox: ~50% of free disk space
Safari: ~1GB (may prompt user)

// Constraint: Can't store unlimited shift metadata
// Solution: Implement LRU cache, expire old data
```

---

## Edge Cases and Boundary Conditions

### Edge Case 1: Zero Shifts

```python
ml_alignments = {
    "MCQBlock1a1": {"shift": [0, 0]}  # No shift
}

# Behavior:
- Validation: ✓ Passes (magnitude = 0 <= max_shift)
- Detection: Both shifted and baseline produce identical results
- Comparison: No mismatches
- Confidence: No reduction
- Performance: Wasted computation (could optimize)

# Optimization:
if all(shift_magnitude == 0 for shift in validated_shifts.values()):
    return context  # Skip dual detection
```

### Edge Case 2: All Shifts Rejected

```python
ml_alignments = {
    "MCQBlock1a1": {"shift": [100, 100]},  # magnitude = 141 > 50
    "MCQBlock1a2": {"shift": [80, 80]}     # magnitude = 113 > 50
}

# Behavior:
- Validation: All rejected
- validated_shifts = {}
- Processing: Returns context unchanged
- Result: No shift detection performed

# Logging:
logger.warning("Shift for MCQBlock1a1 rejected: 141.4px > 50px")
logger.info("No valid shifts to apply")
```

### Edge Case 3: Block Not in Template

```python
ml_alignments = {
    "MCQBlock1a1": {"shift": [20, 30]},  # ✓ Exists
    "UnknownBlock": {"shift": [10, 10]}  # ✗ Not in template
}

# Behavior:
block = find_block_by_name("UnknownBlock")
# block = None

if block:
    block.shifts = [dx, dy]  # Skipped for UnknownBlock
else:
    logger.warning(f"Block UnknownBlock not found in template")

# Result: Apply shift only to MCQBlock1a1
```

### Edge Case 4: Empty Bubble Values

```python
shifted_interp = {"response": "", "bubble_values": []}
baseline_interp = {"response": "", "bubble_values": []}

# Behavior:
bubble_diffs = _compare_bubbles(shifted_interp, baseline_interp)
# bubble_diffs = [] (empty comparison)

severity = len(bubble_diffs) / max(total_bubbles, 1)
# severity = 0 / max(0, 1) = 0 / 1 = 0

reduction = 0.1 + 0 * (0.5 - 0.1) = 0.1
# Minimum reduction applied (defensive)
```

### Edge Case 5: Identical Results

```python
shifted_results = {
    "interpretations": {
        "q1": {"response": "A", "bubble_values": [0,1,0,0], "confidence": 0.95}
    }
}
baseline_results = {
    "interpretations": {
        "q1": {"response": "A", "bubble_values": [0,1,0,0], "confidence": 0.95}
    }
}

# Behavior:
bubble_diffs = []
field_diff = None

if not bubble_diffs and not field_diff:
    # No confidence adjustment
    pass

# Result: Confidence unchanged (shift had no effect)
```

### Edge Case 6: Complete Mismatch

```python
shifted_bubbles  = [1, 1, 1, 1]
baseline_bubbles = [0, 0, 0, 0]

# Behavior:
bubble_diffs = [
    {"index": 0, "shifted": 1, "baseline": 0},
    {"index": 1, "shifted": 1, "baseline": 0},
    {"index": 2, "shifted": 1, "baseline": 0},
    {"index": 3, "shifted": 1, "baseline": 0}
]
severity = 4/4 = 1.0

reduction = 0.1 + 1.0 * (0.5 - 0.1) = 0.5  # Maximum reduction
adjusted_conf = 0.95 - 0.5 = 0.45

# Result: Very low confidence (signals likely error)
```

### Edge Case 7: Negative Confidence

```python
original_conf = 0.3
reduction = 0.5
adjusted_conf = original_conf - reduction  # 0.3 - 0.5 = -0.2

# Behavior:
adjusted_conf = max(0.0, original_conf - reduction)  # Clamp to 0.0

# Result: Confidence = 0.0 (minimum)
```

---

## Configuration Constraints

### Valid Ranges

```python
@dataclass
class ShiftDetectionConfig:
    enabled: bool = False
    # ✓ Valid: Any boolean

    global_max_shift_pixels: int = 50
    # ✓ Valid: 0 to 200 (practical)
    # ✗ Invalid: Negative (doesn't make sense)
    # ⚠ Warning: > 200 (likely indicates ML error)

    per_block_max_shift_pixels: dict[str, int] = field(default_factory=dict)
    # ✓ Valid: Any block name with int value
    # ✗ Invalid: Negative values

    confidence_reduction_min: float = 0.1
    # ✓ Valid: 0.0 to 1.0
    # ⚠ Warning: 0.0 (no minimum reduction, overly optimistic)
    # ✗ Invalid: Negative or > 1.0

    confidence_reduction_max: float = 0.5
    # ✓ Valid: 0.0 to 1.0
    # ✓ Valid: >= confidence_reduction_min
    # ⚠ Warning: 1.0 (completely zeros confidence)
    # ✗ Invalid: < confidence_reduction_min
```

### Validation Logic

```python
def validate_config(config: ShiftDetectionConfig) -> list[str]:
    """Validate configuration and return list of errors."""
    errors = []

    if config.global_max_shift_pixels < 0:
        errors.append("global_max_shift_pixels must be non-negative")

    if config.global_max_shift_pixels > 200:
        errors.append("global_max_shift_pixels > 200 is unusually large")

    if config.confidence_reduction_min < 0 or config.confidence_reduction_min > 1:
        errors.append("confidence_reduction_min must be in [0, 1]")

    if config.confidence_reduction_max < 0 or config.confidence_reduction_max > 1:
        errors.append("confidence_reduction_max must be in [0, 1]")

    if config.confidence_reduction_max < config.confidence_reduction_min:
        errors.append("confidence_reduction_max must be >= confidence_reduction_min")

    for block_name, max_shift in config.per_block_max_shift_pixels.items():
        if max_shift < 0:
            errors.append(f"per_block_max_shift_pixels[{block_name}] must be non-negative")

    return errors
```

---

## Deployment Constraints

### Production Requirements

**Minimum Requirements**:
- Node.js 18+ (for native TypedArray support)
- Modern browser (Chrome 90+, Firefox 88+, Safari 14+)
- 512MB available RAM
- Web Worker support

**Recommended**:
- Node.js 20+
- Chrome 120+, Firefox 115+, Safari 16+
- 2GB available RAM
- SharedArrayBuffer support (for parallel processing)

### Scaling Constraints

**Concurrent Processing**:
```typescript
// Constraint: Memory limits concurrent images
const maxConcurrent = Math.floor(availableMemory / memoryPerImage);

// Typical values:
Desktop (2GB available): ~10 concurrent images
Mobile (500MB available): ~2-3 concurrent images
```

**Batch Processing**:
```typescript
// Process images in batches to avoid memory exhaustion
async function processImageBatch(images: File[], batchSize: number) {
  for (let i = 0; i < images.length; i += batchSize) {
    const batch = images.slice(i, i + batchSize);
    await Promise.all(batch.map(img => processImage(img)));

    // Allow GC to run between batches
    await new Promise(resolve => setTimeout(resolve, 100));
  }
}
```

---

## Testing Constraints

### Test Coverage Requirements

**Unit Tests**:
- ✓ Shift validation (valid/invalid/edge cases)
- ✓ Bubble comparison (identical/different/empty)
- ✓ Field comparison (identical/different)
- ✓ Confidence calculation (min/max/intermediate)
- ✓ Block lookup (found/not found)

**Integration Tests**:
- ✓ End-to-end shift detection flow
- ✓ Pipeline integration
- ✓ ML alignment data flow
- ✓ Context mutation correctness

**Performance Tests**:
- ✓ Processing time < 500ms (target)
- ✓ Memory usage < 100MB per image
- ✓ No memory leaks (repeated processing)

### Test Data Requirements

**Synthetic Test Cases**:
```python
# Case 1: Perfect alignment (no shifts needed)
# Case 2: Small shifts (5-10px)
# Case 3: Medium shifts (20-30px)
# Case 4: Large shifts (50-80px)
# Case 5: Extreme shifts (>100px, should be rejected)
# Case 6: Mixed shifts (some valid, some invalid)
```

---

## Summary

| Constraint Category | Key Limits |
|---------------------|-----------|
| Performance | 2× detection time, ~500ms total |
| Memory | ~100MB per image, 2× results during comparison |
| Accuracy | 50px max shift (default), 10-50% confidence reduction |
| System | Unique block names, ML integration required |
| Browser | Single-threaded (use Web Workers), GC-dependent memory |
| Configuration | Valid ranges: shifts [0, 200], confidence [0, 1] |
| Deployment | Modern browser, 512MB+ RAM, Web Worker support |
