# Bubble Detection Pass - Constraints & Edge Cases

**Status**: Core Detection Component
**Python Reference**: `src/processors/detection/bubbles_threshold/detection_pass.py`
**Browser Compatibility**: OpenCV.js with memory management

---

## Performance Constraints

### Time Complexity

| Operation | Complexity | Typical Time | Notes |
|-----------|-----------|--------------|-------|
| Single bubble extraction | O(w × h) | < 1ms | 30×30 = 900 pixels |
| Field detection (10 bubbles) | O(10 × w × h) | < 5ms | 9,000 pixels |
| File detection (50 fields) | O(50 × 10 × w × h) | < 200ms | 450,000 pixels |
| Standard deviation calc | O(n) | < 1ms | n = bubbles per field |
| Sorting for jumps | O(n log n) | < 1ms | n typically 4-10 |

**Python Performance:**
- Desktop: 100-200 files/second (single thread)
- Server: 500-1000 files/second (multi-threaded)

**Browser Performance:**
- Modern browser: 20-50 files/second (single thread)
- Web Worker: 100-200 files/second (multi-threaded)
- **Bottleneck**: Memory allocation/deallocation in OpenCV.js

### Space Complexity

| Component | Space | Typical Size | Notes |
|-----------|-------|--------------|-------|
| BubbleMeanValue | O(1) | 40 bytes | float + 2 references + tuple |
| BubbleFieldDetectionResult | O(n) | 400-1000 bytes | n = 4-10 bubbles |
| File results | O(f × n) | 20-50 KB | f = 50 fields |
| Repository (100 files) | O(100 × f × n) | 2-5 MB | In-memory cache |

**Browser Constraints:**
- Chrome: ~2-4 GB heap limit
- Firefox: ~2 GB heap limit
- Safari: ~1-2 GB heap limit
- Mobile: 500 MB - 1 GB heap limit

**Recommendation**: Process files in batches of 100-200 to avoid memory pressure

---

## OpenCV.js Memory Management

### Critical Memory Constraints

**Problem**: OpenCV.js uses WASM heap, NOT JavaScript heap

```typescript
// WRONG: Memory leak
function detectBubbles(field: Field, image: cv.Mat) {
  for (const bubble of field.scanBoxes) {
    const roi = image.roi(new cv.Rect(...));  // Allocates WASM memory
    const mean = cv.mean(roi);
    // ❌ LEAK: roi never deleted
  }
}

// CORRECT: Explicit cleanup
function detectBubbles(field: Field, image: cv.Mat) {
  for (const bubble of field.scanBoxes) {
    const roi = image.roi(new cv.Rect(...));
    const mean = cv.mean(roi);
    roi.delete();  // ✅ Free WASM memory
  }
}
```

### Memory Management Pattern

```typescript
class BubblesFieldDetection {
  runDetection(field: Field, grayImage: cv.Mat): void {
    const bubbleMeans: BubbleMeanValue[] = [];

    try {
      for (const bubble of field.scanBoxes) {
        const [boxW, boxH] = bubble.bubbleDimensions;
        const [x, y] = bubble.getShiftedPosition();

        // Allocate ROI
        const roi = grayImage.roi(new cv.Rect(x, y, boxW, boxH));

        try {
          // Use ROI
          const mean = cv.mean(roi);
          bubbleMeans.push({
            meanValue: mean[0],
            unitBubble: bubble,
            position: [x, y]
          });
        } finally {
          // ALWAYS cleanup
          roi.delete();
        }
      }

      this.result = new BubbleFieldDetectionResultImpl(
        field.id,
        field.fieldLabel,
        bubbleMeans
      );
    } catch (error) {
      // Cleanup partial results if needed
      console.error('Detection failed:', error);
      throw error;
    }
  }
}
```

### WASM Memory Limits

| Browser | Default Heap | Maximum Heap | Notes |
|---------|-------------|--------------|-------|
| Chrome Desktop | 2 GB | 4 GB | With `--js-flags="--max-old-space-size=4096"` |
| Chrome Android | 512 MB | 1 GB | Limited by system memory |
| Firefox | 2 GB | 4 GB | `about:config` tuning |
| Safari Desktop | 1 GB | 2 GB | No user control |
| Safari iOS | 256 MB | 512 MB | Aggressive memory management |

**Mitigation Strategies:**
1. Delete all cv.Mat objects immediately after use
2. Process files in batches
3. Use Web Workers to isolate memory
4. Implement garbage collection hints

---

## Edge Cases

### 1. Empty or Missing Bubbles

**Scenario**: Field has no scan boxes

```python
# Python handling
def run_detection(self, field: Field, gray_image, _colored_image):
    bubble_means = []

    if not field.scan_boxes:
        # Empty field - create result with no bubbles
        self.result = BubbleFieldDetectionResult(
            field_id=field.id,
            field_label=field.field_label,
            bubble_means=[]  # Empty list
        )
        return

    # Normal processing...
```

**Impact on Properties:**
- `std_deviation`: Returns 0.0
- `scan_quality`: Returns `ScanQuality.POOR`
- `jumps`: Returns empty list
- `max_jump`: Returns 0.0

**Browser Handling:**
```typescript
if (field.scanBoxes.length === 0) {
  this.result = {
    fieldId: field.id,
    fieldLabel: field.fieldLabel,
    bubbleMeans: [],
    stdDeviation: 0,
    scanQuality: ScanQuality.POOR
  };
  return;
}
```

---

### 2. Single Bubble Field

**Scenario**: Field has only one bubble

```python
# Example: Roll number field with single digit
field.scan_boxes = [BubblesScanBox(...)]  # Only one bubble
```

**Impact:**
- `std_deviation`: 0.0 (no variance)
- `scan_quality`: `POOR` (std ≤ 15)
- `jumps`: Empty list (need 2+ bubbles)
- `max_jump`: 0.0

**Threshold Implications:**
- Cannot calculate local threshold (need 2+ bubbles)
- Must use global threshold fallback
- May trigger multi-mark warning in interpretation

**Handling:**
```python
def calculate_threshold(self, bubble_means: list[float]) -> ThresholdResult:
    if len(bubble_means) < 2:
        return ThresholdResult(
            threshold_value=self.config.default_threshold,
            confidence=0.0,
            method_used="default_fallback",
            fallback_used=True
        )
```

---

### 3. All Bubbles Same Intensity

**Scenario**: All bubbles have identical or very similar mean values

**Causes:**
- All bubbles unmarked (mean ≈ 220-240)
- All bubbles marked (mean ≈ 50-80) - rare
- Poor scan quality (low contrast)
- Completely blank/white image

**Example:**
```python
bubble_means = [218.5, 219.1, 218.8, 219.3]  # All ~219
std_deviation = 0.3  # Very low
scan_quality = ScanQuality.POOR
```

**Detection:**
```python
if result.std_deviation < 15:
    logger.warning(
        f"Poor scan quality for {field.field_label}: "
        f"std={result.std_deviation:.1f}"
    )
```

**Mitigation:**
- Use global threshold from other fields
- Flag for manual review
- Check if image is blank

---

### 4. Out-of-Bounds ROI

**Scenario**: Shifted position places bubble outside image bounds

**Causes:**
- Excessive alignment shift
- Template mismatch
- Incorrect preprocessing
- Manual shift errors

**Example:**
```python
# Bubble position after shifts
x, y = bubble.get_shifted_position()  # (1890, 2650)
image_shape = gray_image.shape  # (2200, 1700)
# x=1890 > 1700 → Out of bounds!
```

**Python Handling:**
```python
def read_bubble_mean_value(unit_bubble, gray_image):
    box_w, box_h = unit_bubble.bubble_dimensions
    x, y = unit_bubble.get_shifted_position()

    # Validate bounds
    h, w = gray_image.shape[:2]
    if x < 0 or y < 0 or x + box_w > w or y + box_h > h:
        logger.error(
            f"Bubble {unit_bubble.name} out of bounds: "
            f"({x},{y}) {box_w}x{box_h} in {w}x{h} image"
        )
        # Return default "unmarked" value
        return BubbleMeanValue(
            mean_value=220.0,  # Default unmarked
            unit_bubble=unit_bubble,
            position=(x, y)
        )

    # Normal processing
    rect = [y, y + box_h, x, x + box_w]
    mean_value = cv2.mean(gray_image[rect[0]:rect[1], rect[2]:rect[3]], None)[0]
    return BubbleMeanValue(mean_value, unit_bubble, (x, y))
```

**Browser Handling:**
```typescript
function readBubbleMeanValue(
  bubble: BubblesScanBox,
  image: cv.Mat
): BubbleMeanValue {
  const [boxW, boxH] = bubble.bubbleDimensions;
  const [x, y] = bubble.getShiftedPosition();

  // Validate bounds
  if (x < 0 || y < 0 || x + boxW > image.cols || y + boxH > image.rows) {
    console.error(
      `Bubble ${bubble.name} out of bounds: ` +
      `(${x},${y}) ${boxW}x${boxH} in ${image.cols}x${image.rows}`
    );

    // Return default unmarked value
    return {
      meanValue: 220.0,
      unitBubble: bubble,
      position: [x, y]
    };
  }

  // Normal processing
  const roi = image.roi(new cv.Rect(x, y, boxW, boxH));
  const mean = cv.mean(roi);
  const meanValue = mean[0];
  roi.delete();

  return { meanValue, unitBubble: bubble, position: [x, y] };
}
```

---

### 5. Negative or Zero Dimensions

**Scenario**: Bubble has invalid dimensions

**Example:**
```python
bubble.bubble_dimensions = [0, 30]  # Width = 0
bubble.bubble_dimensions = [-5, 30]  # Negative width
```

**Validation:**
```python
def read_bubble_mean_value(unit_bubble, gray_image):
    box_w, box_h = unit_bubble.bubble_dimensions

    if box_w <= 0 or box_h <= 0:
        raise ValueError(
            f"Invalid bubble dimensions for {unit_bubble.name}: "
            f"{box_w}x{box_h}"
        )

    # Normal processing...
```

---

### 6. Grayscale Conversion Issues

**Scenario**: Colored image not properly converted to grayscale

**Problem:**
```python
# Multi-channel image passed as grayscale
gray_image.shape = (2200, 1700, 3)  # Should be (2200, 1700)
mean = cv2.mean(roi, None)[0]  # Returns mean of first channel only
```

**Validation:**
```python
def run_detection(self, field, gray_image, colored_image):
    # Validate grayscale
    if len(gray_image.shape) != 2:
        logger.warning(
            f"Expected grayscale image, got shape {gray_image.shape}. "
            f"Converting to grayscale."
        )
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

    # Normal processing...
```

**Browser Validation:**
```typescript
function ensureGrayscale(image: cv.Mat): cv.Mat {
  if (image.channels() !== 1) {
    console.warn('Converting multi-channel image to grayscale');
    const gray = new cv.Mat();
    cv.cvtColor(image, gray, cv.COLOR_RGBA2GRAY);
    return gray;
  }
  return image;
}
```

---

### 7. None/Null Detection Result

**Scenario**: Detection fails to create result

**Current Implementation:**
```python
def update_field_level_aggregates_on_processed_field_detection(
    self, field, field_detection
):
    # Explicit check for None
    if field_detection.result is None:
        logger.error(f"field_detection.result is None for field {field.id}")
        raise ValueError(f"field_detection.result is None for field {field.id}")

    # Safe to use result
    self.repository.save_bubble_field(field.id, field_detection.result)
```

**Prevention:**
```python
def run_detection(self, field, gray_image, colored_image):
    bubble_means = []

    try:
        for unit_bubble in field.scan_boxes:
            bubble_mean = self.read_bubble_mean_value(unit_bubble, gray_image)
            bubble_means.append(bubble_mean)

        # ALWAYS create result
        self.result = BubbleFieldDetectionResult(
            field_id=field.id,
            field_label=field.field_label,
            bubble_means=bubble_means
        )
    except Exception as e:
        logger.error(f"Detection failed for {field.id}: {e}")
        # Create empty result instead of None
        self.result = BubbleFieldDetectionResult(
            field_id=field.id,
            field_label=field.field_label,
            bubble_means=[]
        )
        raise
```

---

### 8. Very Large Bubbles

**Scenario**: Bubble dimensions exceed reasonable size

**Example:**
```python
bubble.bubble_dimensions = [500, 500]  # 250,000 pixels per bubble
# 50 fields × 10 bubbles × 250,000 = 125 million pixel operations
```

**Performance Impact:**
- Slower mean calculation
- Higher memory usage
- Potential timeout in browser

**Validation:**
```python
MAX_BUBBLE_DIMENSION = 200  # pixels

def read_bubble_mean_value(unit_bubble, gray_image):
    box_w, box_h = unit_bubble.bubble_dimensions

    if box_w > MAX_BUBBLE_DIMENSION or box_h > MAX_BUBBLE_DIMENSION:
        logger.warning(
            f"Unusually large bubble {unit_bubble.name}: {box_w}x{box_h}"
        )

    # Normal processing (still allow, just warn)
```

---

### 9. Floating Point Precision

**Scenario**: Shifted positions result in non-integer coordinates

**Example:**
```python
x, y = bubble.get_shifted_position()  # (125.7, 450.3)
# Array indexing requires integers
```

**Current Handling:**
```python
x, y = unit_bubble.get_shifted_position()  # Already rounds
x, y = round(x), round(y)  # Explicit rounding

# Safe indexing
rect = [y, y + box_h, x, x + box_w]
mean = cv2.mean(gray_image[rect[0]:rect[1], rect[2]:rect[3]], None)[0]
```

---

### 10. Repository Race Conditions

**Scenario**: Multiple threads/workers accessing repository simultaneously

**Problem:**
```python
# Thread 1
repository.save_bubble_field("field1", result1)

# Thread 2 (simultaneous)
repository.save_bubble_field("field1", result2)  # Overwrites result1
```

**Mitigation (Python):**
```python
# Repository uses file-scoped isolation
class DetectionRepository:
    def __init__(self):
        self._current_file_path = None
        self._current_file_results = None
        self._lock = threading.Lock()  # If multi-threaded

    def save_bubble_field(self, field_id, result):
        with self._lock:  # Thread-safe
            current_file = self._get_current_file()
            current_file.bubble_fields[field_id] = result
```

**Browser (Single-threaded):**
```typescript
// No race conditions in single-threaded context
// But be careful with Web Workers

// Worker 1
postMessage({ type: 'save', fieldId: 'field1', result: result1 });

// Worker 2
postMessage({ type: 'save', fieldId: 'field1', result: result2 });

// Main thread: Last message wins
```

---

## Browser-Specific Constraints

### 1. Safari WASM Limits

**Constraint**: Safari has stricter WASM memory limits

```typescript
// Detect Safari
const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);

// Adjust batch size
const BATCH_SIZE = isSafari ? 50 : 100;

// Process in smaller batches on Safari
function processBatch(files: File[]) {
  const batchSize = isSafari ? 50 : 100;
  for (let i = 0; i < files.length; i += batchSize) {
    const batch = files.slice(i, i + batchSize);
    processBatchFiles(batch);
  }
}
```

### 2. Mobile Memory Constraints

**Constraint**: Mobile devices have 4-10x less memory

```typescript
// Detect mobile
const isMobile = /Android|webOS|iPhone|iPad|iPod/i.test(navigator.userAgent);

// Reduce concurrency
const MAX_WORKERS = isMobile ? 1 : navigator.hardwareConcurrency || 4;

// Smaller batch size
const BATCH_SIZE = isMobile ? 10 : 100;

// Progressive loading
async function processMobile(files: File[]) {
  for (const file of files) {
    await processFile(file);  // Sequential, not parallel
    // Allow GC between files
    await new Promise(resolve => setTimeout(resolve, 100));
  }
}
```

### 3. Firefox WASM Memory Bug

**Issue**: Firefox may not properly free WASM memory immediately

```typescript
// Explicit GC hint (non-standard, Firefox only)
if ('gc' in window) {
  (window as any).gc();
}

// Workaround: Process in batches with delays
async function processWithGC(files: File[]) {
  for (let i = 0; i < files.length; i++) {
    await processFile(files[i]);

    if (i % 10 === 0) {
      // Force GC on Firefox
      if ('gc' in window) {
        (window as any).gc();
      }
      // Allow event loop to clear
      await new Promise(resolve => setTimeout(resolve, 50));
    }
  }
}
```

### 4. Chrome DevTools Memory Profiling

**Tool**: Use Chrome Memory Profiler to detect leaks

```typescript
// Add memory tracking
class MemoryMonitor {
  private baseline: number = 0;

  start() {
    if (performance.memory) {
      this.baseline = performance.memory.usedJSHeapSize;
    }
  }

  check(label: string) {
    if (performance.memory) {
      const current = performance.memory.usedJSHeapSize;
      const delta = current - this.baseline;
      console.log(`${label}: ${(delta / 1024 / 1024).toFixed(2)} MB`);
    }
  }
}

// Usage
const monitor = new MemoryMonitor();
monitor.start();

for (const file of files) {
  processFile(file);
  monitor.check(`After file ${file.name}`);
}
```

---

## Accuracy Constraints

### 1. Bubble Size Requirements

**Minimum Size**: 10×10 pixels
**Recommended**: 20×20 to 40×40 pixels
**Maximum**: 200×200 pixels

**Rationale:**
- Too small: Noise dominates signal
- Too large: Slower processing, no accuracy gain

### 2. Image Resolution

**Minimum**: 150 DPI
**Recommended**: 200-300 DPI
**Maximum**: 600 DPI (diminishing returns)

**Impact:**
- < 150 DPI: Poor bubble detection accuracy
- > 600 DPI: Excessive processing time, no accuracy improvement

### 3. Contrast Requirements

**Minimum std**: 15 (POOR quality)
**Recommended std**: 30+ (GOOD quality)
**Excellent std**: 50+ (EXCELLENT quality)

**Calculation:**
```python
# For 4 bubbles: 1 marked (mean=80), 3 unmarked (mean=220)
values = [80, 220, 220, 220]
std = np.std(values)  # ~60.6 → EXCELLENT
```

---

## Validation Checklist

```typescript
interface ValidationResult {
  valid: boolean;
  warnings: string[];
  errors: string[];
}

function validateDetection(
  result: BubbleFieldDetectionResult
): ValidationResult {
  const warnings: string[] = [];
  const errors: string[] = [];

  // 1. Check for empty result
  if (result.bubbleMeans.length === 0) {
    warnings.push('Field has no bubbles');
  }

  // 2. Check scan quality
  if (result.scanQuality === ScanQuality.POOR) {
    warnings.push(`Poor scan quality (std=${result.stdDeviation.toFixed(1)})`);
  }

  // 3. Check for out-of-range values
  for (const bm of result.bubbleMeans) {
    if (bm.meanValue < 0 || bm.meanValue > 255) {
      errors.push(`Invalid mean value: ${bm.meanValue}`);
    }
  }

  // 4. Check for suspicious patterns
  if (result.bubbleMeans.length > 1 && result.stdDeviation < 5) {
    warnings.push('All bubbles have identical values');
  }

  // 5. Check for outliers
  const sorted = result.sortedMeanValues;
  if (sorted.length >= 3) {
    const median = sorted[Math.floor(sorted.length / 2)];
    for (const value of sorted) {
      if (Math.abs(value - median) > 100) {
        warnings.push(`Outlier detected: ${value} vs median ${median}`);
      }
    }
  }

  return {
    valid: errors.length === 0,
    warnings,
    errors
  };
}
```

---

## Performance Optimization Strategies

### 1. Batch ROI Extraction

```typescript
// Instead of extracting ROIs one-by-one, batch them
function batchExtractROIs(
  bubbles: BubblesScanBox[],
  image: cv.Mat
): BubbleMeanValue[] {
  const rois: cv.Mat[] = [];
  const results: BubbleMeanValue[] = [];

  try {
    // Extract all ROIs first
    for (const bubble of bubbles) {
      const [w, h] = bubble.bubbleDimensions;
      const [x, y] = bubble.getShiftedPosition();
      const roi = image.roi(new cv.Rect(x, y, w, h));
      rois.push(roi);
    }

    // Calculate means (can be parallelized)
    for (let i = 0; i < rois.length; i++) {
      const mean = cv.mean(rois[i]);
      results.push({
        meanValue: mean[0],
        unitBubble: bubbles[i],
        position: bubbles[i].getShiftedPosition()
      });
    }
  } finally {
    // Cleanup all ROIs
    for (const roi of rois) {
      roi.delete();
    }
  }

  return results;
}
```

### 2. Web Worker Pool

```typescript
class DetectionWorkerPool {
  private workers: Worker[] = [];
  private queue: Array<{ file: File; resolve: Function }> = [];

  constructor(workerCount: number = navigator.hardwareConcurrency || 4) {
    for (let i = 0; i < workerCount; i++) {
      const worker = new Worker('/detection-worker.js');
      worker.onmessage = (e) => this.handleResult(e.data);
      this.workers.push(worker);
    }
  }

  async detectFile(file: File): Promise<BubbleFieldDetectionResult[]> {
    return new Promise((resolve) => {
      this.queue.push({ file, resolve });
      this.processQueue();
    });
  }

  private processQueue() {
    // Assign work to available workers
  }
}
```

### 3. Caching Computed Properties

```typescript
class BubbleFieldDetectionResultImpl {
  private _stdDeviation?: number;
  private _scanQuality?: ScanQuality;

  get stdDeviation(): number {
    if (this._stdDeviation === undefined) {
      this._stdDeviation = this.calculateStdDeviation();
    }
    return this._stdDeviation;
  }

  get scanQuality(): ScanQuality {
    if (this._scanQuality === undefined) {
      const std = this.stdDeviation;
      if (std > 50) this._scanQuality = ScanQuality.EXCELLENT;
      else if (std > 30) this._scanQuality = ScanQuality.GOOD;
      else if (std > 15) this._scanQuality = ScanQuality.ACCEPTABLE;
      else this._scanQuality = ScanQuality.POOR;
    }
    return this._scanQuality;
  }
}
```

---

## Related Documentation

- **Parent**: `/bubbles-threshold/concept.md` - Overall architecture
- **Flows**: `flows.md` - Detection pass flow
- **Next**: `/bubbles-threshold/interpretation-pass/constraints.md` - Interpretation constraints
- **Technical**: `/technical/opencv/opencv-operations.md` - OpenCV.js patterns
- **Performance**: `/migration/performance.md` - Browser performance guide
