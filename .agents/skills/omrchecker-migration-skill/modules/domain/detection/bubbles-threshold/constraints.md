# Bubble Detection - Constraints & Limitations

**Status**: Constraints Documentation
**Python Reference**: `src/processors/detection/bubbles_threshold/`

---

## Overview

This document covers the **performance constraints, accuracy limitations, and edge cases** of the threshold-based bubble detection system.

---

## Performance Constraints

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Single bubble detection | O(w × h) | cv2.mean over bubble ROI |
| Field detection (n bubbles) | O(n × w × h) | Linear in number of bubbles |
| Global threshold | O(m log m) | Sort all bubble means (m = total bubbles) |
| Local threshold | O(k log k) | Sort field bubble means (k = bubbles per field) |
| Interpretation | O(k) | Linear pass over bubbles |
| **Total per file** | **O(m × w × h + m log m)** | Dominated by detection |

### Space Complexity

| Structure | Space | Notes |
|-----------|-------|-------|
| BubbleMeanValue | O(1) per bubble | float + reference + tuple |
| BubbleFieldDetectionResult | O(k) | k bubbles per field |
| DetectionRepository | O(f × k) | f fields, k bubbles per field |
| Interpretation results | O(f × k) | Same as detection |
| **Total per file** | **O(f × k)** | Linear in total bubbles |

### Real-World Performance

#### Python (with OpenCV and NumPy)

```python
# Typical OMR sheet: 100 fields, 4 bubbles each = 400 bubbles
# Bubble size: 20×20 pixels

# Detection time
single_bubble = 0.1ms  # cv2.mean on 20×20 ROI
field_detection = 0.4ms  # 4 bubbles
all_fields = 40ms  # 100 fields

# Threshold calculation
global_threshold = 1ms  # Sort 400 values
local_threshold = 0.01ms × 100 fields = 1ms

# Total per file: ~42ms
# Throughput: ~24 files/second
```

#### Browser (with OpenCV.js)

```typescript
// Same OMR sheet: 100 fields, 4 bubbles each = 400 bubbles

// Detection time (slower than native)
single_bubble = 0.5ms  // cv.mean on 20×20 ROI
field_detection = 2ms  // 4 bubbles
all_fields = 200ms  // 100 fields

// Threshold calculation (JavaScript, no WASM)
global_threshold = 2ms  // Sort 400 values
local_threshold = 0.05ms × 100 fields = 5ms

// Total per file: ~207ms
// Throughput: ~5 files/second
```

### Memory Constraints

#### Python

```python
# Per file
bubbles = 400 × 32 bytes (BubbleMeanValue) = 12.8 KB
results = 100 × 200 bytes (BubbleFieldDetectionResult) = 20 KB
interpretations = 400 × 100 bytes = 40 KB
total_per_file ≈ 73 KB

# For 1000 files: ~73 MB (manageable)
```

#### Browser

```typescript
// Per file
bubbles = 400 × 48 bytes (JS objects) = 19.2 KB
results = 100 × 300 bytes = 30 KB
interpretations = 400 × 150 bytes = 60 KB
opencv_mats = 400 × (20×20) bytes = 160 KB  // ROI memory
total_per_file ≈ 269 KB

// For 100 files: ~27 MB
// Memory limit: ~100 files before GC pressure
// Recommendation: Process in batches of 50
```

---

## Accuracy Constraints

### Assumptions

The threshold strategy makes several **critical assumptions**:

1. **Marked bubbles are darker** than unmarked bubbles
2. **Clear contrast** exists between marked and unmarked
3. **Uniform lighting** across the scan
4. **Bubbles are properly aligned** (preprocessing successful)
5. **No stray marks** outside bubble regions

### Accuracy vs. Scan Quality

| Scan Quality | Std Dev | Accuracy | Notes |
|--------------|---------|----------|-------|
| Excellent | > 50 | 98-100% | Clear contrast, high confidence |
| Good | 30-50 | 95-98% | Acceptable contrast |
| Acceptable | 15-30 | 85-95% | Marginal contrast, some errors |
| Poor | < 15 | 60-85% | Low contrast, high error rate |

### Error Modes

#### False Positive (Unmarked → Marked)

```python
# Causes:
# 1. Stray marks near bubble
# 2. Dust/dirt on scan
# 3. Threshold too high (too lenient)
# 4. Shadows or uneven lighting

# Example:
bubble_mean = 125  # Should be unmarked (180+)
threshold = 130    # Too lenient
is_marked = 125 < 130  # FALSE POSITIVE
```

#### False Negative (Marked → Unmarked)

```python
# Causes:
# 1. Light marking (pencil not dark enough)
# 2. Faded scan
# 3. Threshold too low (too strict)
# 4. Overexposure

# Example:
bubble_mean = 145  # Should be marked (< 120)
threshold = 140    # Too strict
is_marked = 145 < 140  # FALSE NEGATIVE
```

### Threshold Selection Trade-offs

```
Stricter Threshold (lower value, e.g., 100)
    ├─> Fewer false positives
    └─> More false negatives (missed marks)

Lenient Threshold (higher value, e.g., 140)
    ├─> Fewer false negatives
    └─> More false positives (stray marks)

Optimal Threshold (max gap)
    ├─> Balances both error types
    └─> Depends on scan quality
```

---

## Edge Cases

### Edge Case 1: Empty Field (No Bubbles)

```python
# Scenario: Field definition has no scan_boxes
field.scan_boxes = []

# Detection behavior:
bubble_means = []
result = BubbleFieldDetectionResult(
    field_id=field.id,
    field_label=field.field_label,
    bubble_means=[]
)
result.std_deviation = 0.0
result.scan_quality = ScanQuality.POOR

# Interpretation behavior:
threshold = global_fallback  # Use global
interpretations = []  # Empty
interpretation_string = ""
```

### Edge Case 2: Single Bubble Field

```python
# Scenario: Field has only one bubble
field.scan_boxes = [bubble_A]

# Detection:
bubble_means = [BubbleMeanValue(mean_value=65, ...)]

# Interpretation:
# Cannot calculate local threshold (need 2+ values)
threshold = global_fallback
confidence = 0.0
fallback_used = True
method_used = "local_single_bubble_fallback"

# Interpretation:
is_marked = 65 < global_fallback
```

### Edge Case 3: Two Bubbles - Ambiguous Gap

```python
# Scenario: Two bubbles with small gap
bubble_means = [115, 125]
min_gap_two_bubbles = 20

gap = 125 - 115 = 10  # Less than threshold

# Threshold decision:
# Gap too small to be confident
threshold = global_fallback  # Fall back
confidence = 0.3
fallback_used = True
method_used = "local_two_bubbles_small_gap_fallback"
```

### Edge Case 4: All Bubbles Same Value

```python
# Scenario: All bubbles have identical mean
bubble_means = [120, 120, 120, 120]

# Threshold calculation:
sorted = [120, 120, 120, 120]
max_jump = 0  # No gap

# Result:
threshold = default_threshold  # 127.5
confidence = 0.0
fallback_used = True

# Interpretation:
# All bubbles are unmarked (120 < 127.5 is False)
interpretation_string = ""
```

### Edge Case 5: All Bubbles Marked

```python
# Scenario: All bubbles marked (scanning issue)
bubble_means = [50, 55, 60, 65]
bubble_values = ['A', 'B', 'C', 'D']
threshold = 120

# Interpretation:
interpretations = [all is_attempted=True]
marked_count = 4
total_count = 4

# Special handling:
if marked_count == total_count:
    interpretation_string = ""  # Treat as unmarked
    # Likely scanning issue (sheet is too dark)
```

### Edge Case 6: No Marked Bubbles

```python
# Scenario: All bubbles unmarked
bubble_means = [180, 185, 190, 195]
threshold = 120

# Interpretation:
interpretations = [all is_attempted=False]
marked_count = 0

# Result:
interpretation_string = ""  # Empty value
is_multi_marked = False
```

### Edge Case 7: Multi-Marking

```python
# Scenario: Multiple bubbles marked
bubble_means = [60, 65, 180, 190]
bubble_values = ['A', 'B', 'C', 'D']
threshold = 120

# Interpretation:
interpretations = [True, True, False, False]
marked_count = 2

# Result:
interpretation_string = "AB"  # Concatenated
is_multi_marked = True
logger.warning("Multi-marking detected")
```

### Edge Case 8: ROI Out of Bounds

```python
# Scenario: Bubble position outside image
image_shape = (1000, 1000)
bubble_position = (950, 950)
bubble_size = (100, 100)

# ROI extraction:
x, y = 950, 950
w, h = 100, 100
roi = image[y:y+h, x:x+w]  # [950:1050, 950:1050]

# Result:
# ROI is partially outside image (only 50×50 pixels)
# cv2.mean will calculate mean of valid region only
mean_value = cv2.mean(roi)[0]  # May be 0 if completely outside
```

### Edge Case 9: Xeroxed OMR (No Contrast)

```python
# Scenario: Poor quality scan with no contrast
bubble_means = [180, 185, 190, 195, 200, 205]
# All values are close together

# Global threshold:
sorted = [180, 185, 190, 195, 200, 205]
max_jump = 10  # Less than min_jump (30)

# Result:
threshold = 255.0  # Default for Xeroxed OMRs
confidence = 0.0
fallback_used = True

# Interpretation:
# All bubbles unmarked (all values < 255)
# interpretation_string = "" for all fields
logger.debug("Looks like a Xeroxed OMR")
```

### Edge Case 10: Threshold Disparity

```python
# Scenario: Local and global thresholds disagree
bubble_means = [60, 110, 180, 190]
local_threshold = 120
global_threshold = 100

# Bubble 2 has disparity:
# local: 110 < 120 → marked
# global: 110 > 100 → unmarked

# Detection:
disparity_bubbles = [bubble_2]
confidence_penalty = 0.4 × (1/4) = 0.1

# Result:
# Lower confidence score
# May trigger ML fallback
```

---

## Browser-Specific Constraints

### Memory Management

```typescript
// OpenCV.js mats must be manually deleted
function detectBubbleField(field: Field, grayImage: cv.Mat): BubbleFieldDetectionResult {
  const bubbleMeans: BubbleMeanValue[] = [];

  for (const bubble of field.scanBoxes) {
    const roi = grayImage.roi(new cv.Rect(bubble.x, bubble.y, bubble.w, bubble.h));

    try {
      const mean = cv.mean(roi);
      bubbleMeans.push({
        meanValue: mean[0],
        unitBubble: bubble,
        position: [bubble.x, bubble.y]
      });
    } finally {
      roi.delete();  // CRITICAL: Prevent memory leak
    }
  }

  return { fieldId: field.id, fieldLabel: field.fieldLabel, bubbleMeans };
}
```

### Performance Optimization

```typescript
// Use Web Worker for parallel processing
class BubbleDetectionWorker {
  async detectFields(
    fields: Field[],
    grayImage: cv.Mat
  ): Promise<BubbleFieldDetectionResult[]> {
    // Process in batches to avoid blocking UI
    const batchSize = 10;
    const results: BubbleFieldDetectionResult[] = [];

    for (let i = 0; i < fields.length; i += batchSize) {
      const batch = fields.slice(i, i + batchSize);

      // Process batch
      const batchResults = batch.map(field =>
        this.detectField(field, grayImage)
      );

      results.push(...batchResults);

      // Yield to main thread
      await new Promise(resolve => setTimeout(resolve, 0));
    }

    return results;
  }
}
```

### Memory Limits

```typescript
// Monitor memory usage
class MemoryMonitor {
  checkMemoryPressure(): boolean {
    if (performance.memory) {
      const used = performance.memory.usedJSHeapSize;
      const limit = performance.memory.jsHeapSizeLimit;
      return (used / limit) > 0.8;  // 80% threshold
    }
    return false;
  }

  async processWithMemoryManagement(
    files: File[]
  ): Promise<void> {
    for (const file of files) {
      if (this.checkMemoryPressure()) {
        // Force GC by releasing references
        await this.clearCache();
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      await this.processFile(file);
    }
  }
}
```

---

## Configuration Constraints

### Parameter Ranges

```python
@dataclass
class ThresholdConfig:
    # min_jump: Minimum gap to consider significant
    min_jump: float = 30.0  # Range: 15-50
    # Too low: Noise triggers false jumps
    # Too high: Misses real gaps

    # min_gap_two_bubbles: Special case for 2-bubble fields
    min_gap_two_bubbles: float = 20.0  # Range: 10-40
    # Too low: Ambiguous cases use local
    # Too high: Always falls back to global

    # min_jump_surplus_for_global_fallback: Extra confidence required
    min_jump_surplus_for_global_fallback: float = 10.0  # Range: 5-20
    # Too low: Always use local (risky)
    # Too high: Always fall back to global (conservative)

    # default_threshold: Fallback value
    default_threshold: float = 127.5  # Range: 100-255
    # 127.5: Mid-gray (balanced)
    # 255: Treat all as unmarked (Xeroxed OMRs)
    # 100: More lenient (low-quality scans)
```

### Tuning Guidelines

```python
# High-quality scans (good contrast)
config = ThresholdConfig(
    min_jump=30.0,
    min_gap_two_bubbles=20.0,
    min_jump_surplus_for_global_fallback=10.0,
    default_threshold=127.5
)

# Medium-quality scans
config = ThresholdConfig(
    min_jump=25.0,
    min_gap_two_bubbles=15.0,
    min_jump_surplus_for_global_fallback=8.0,
    default_threshold=127.5
)

# Low-quality scans (poor contrast)
config = ThresholdConfig(
    min_jump=20.0,
    min_gap_two_bubbles=10.0,
    min_jump_surplus_for_global_fallback=5.0,
    default_threshold=100.0
)

# Xeroxed OMRs (no contrast)
config = ThresholdConfig(
    min_jump=15.0,
    min_gap_two_bubbles=8.0,
    min_jump_surplus_for_global_fallback=3.0,
    default_threshold=255.0
)
```

---

## Limitations Summary

### When Threshold Strategy Works Well

1. **Good scan quality** (high contrast, clear marks)
2. **Uniform lighting** (no shadows or gradients)
3. **Proper alignment** (bubbles in expected positions)
4. **Clean bubbles** (no stray marks, dust, or noise)
5. **Consistent marking** (all marked bubbles similar darkness)

### When Threshold Strategy Struggles

1. **Poor scan quality** (low contrast, faded marks)
2. **Uneven lighting** (shadows, gradients)
3. **Misaligned bubbles** (preprocessing failed)
4. **Stray marks** (dust, smudges, annotations)
5. **Inconsistent marking** (some light, some dark)
6. **Xeroxed OMRs** (all bubbles similar darkness)

### Mitigation Strategies

```python
# 1. Preprocessing
# - Auto-rotation for alignment
# - CropOnMarkers for perspective correction
# - Gaussian blur for noise reduction
# - Contrast enhancement

# 2. Threshold strategies
# - Local threshold for per-field adaptation
# - Global fallback for low confidence
# - Confidence-based ML fallback

# 3. Post-processing
# - Multi-mark detection
# - Disparity detection
# - Confidence scoring for review
```

---

## Testing Recommendations

### Test Cases

```python
# 1. Perfect scan
bubble_means = [50, 55, 60, 180, 185, 190]
expected_threshold ≈ 120
expected_confidence > 0.9

# 2. Poor scan
bubble_means = [110, 115, 120, 125, 130, 135]
expected_threshold = global_fallback
expected_confidence < 0.5

# 3. Single mark
bubble_means = [60, 180, 185, 190]
expected_interpretation = "A"
expected_multi_mark = False

# 4. Multi-mark
bubble_means = [60, 65, 180, 190]
expected_interpretation = "AB"
expected_multi_mark = True

# 5. No marks
bubble_means = [180, 185, 190, 195]
expected_interpretation = ""
expected_multi_mark = False

# 6. All marks (scanning issue)
bubble_means = [50, 55, 60, 65]
expected_interpretation = ""
expected_multi_mark = False  # Special case

# 7. Threshold disparity
local_threshold = 120
global_threshold = 100
bubble_mean = 110
expected_disparity = True
```

---

## Related Documentation

- `concept.md` - Architecture and data models
- `flows.md` - Complete detection and interpretation flows
- `decisions.md` - Threshold strategy selection logic
- `integration.md` - Integration with ReadOMR processor
