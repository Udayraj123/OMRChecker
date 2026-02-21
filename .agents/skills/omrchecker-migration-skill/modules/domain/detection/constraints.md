# ReadOMR Processor - Constraints and Limitations

## Performance Constraints

### 1. Processing Time Limits

#### Per-Image Processing Time

| Component | Target | Maximum | Exceeded Impact |
|-----------|--------|---------|-----------------|
| **Resize + Normalize** | 10ms | 50ms | Minimal impact |
| **Bubble Detection** | 50ms | 200ms | User-noticeable delay |
| **OCR Detection** | 500ms | 2000ms | Significant delay |
| **Barcode Detection** | 50ms | 200ms | Minor delay |
| **ML Fallback** | 200ms | 1000ms | Major delay |
| **Total (no ML)** | 150ms | 500ms | Batch processing slows |
| **Total (with ML)** | 350ms | 2000ms | Severe batch slowdown |

**Constraint**:
```python
# For 1000 images:
# No ML:  150ms × 1000 = 150 seconds (~2.5 minutes)
# With ML: 2000ms × 1000 = 2000 seconds (~33 minutes)

# Practical limit: ~5 minutes for 1000 images
# Exceeding limit: Users may abandon processing
```

**Mitigation**:
- Use ML fallback sparingly (confidence-based)
- Process in parallel (ThreadPoolExecutor)
- Skip OCR if not needed

---

### 2. Memory Constraints

#### Per-Image Memory Usage

```python
# Image storage (2000×3000 pixels):
gray_image: 2000 × 3000 × 1 byte = 6 MB
colored_image: 2000 × 3000 × 3 bytes = 18 MB
normalized_images: 2 copies × (6 + 18) MB = 48 MB

# Detection aggregates:
field_detections: ~1 KB per field × 100 fields = 100 KB
interpretations: ~500 bytes × 100 fields = 50 KB

# ML model (if loaded):
yolo_model: ~50 MB (model weights)
ml_inference: ~20 MB (intermediate tensors)

Total per image (no ML): ~50 MB
Total per image (with ML): ~120 MB
```

**Batch Processing Constraint**:
```python
# With 8 GB RAM available:
# No ML: 8000 MB / 50 MB = 160 images in parallel
# With ML: 8000 MB / 120 MB = 66 images in parallel

# Practical limit: 4-8 images in parallel (leave room for OS)
```

**Code**: Memory released after each image processed
```python
# After processing:
context = None  # Allow garbage collection
gray_image = None
colored_image = None
```

**Browser Constraint**:
```
Chrome: ~2 GB per tab
Firefox: ~1.5 GB per tab
Safari: ~1 GB per tab

Practical limit: Process 10-20 images at once in browser
```

---

### 3. CPU/GPU Constraints

#### CPU-Bound Operations

| Operation | CPU Usage | Duration | Parallelizable? |
|-----------|-----------|----------|-----------------|
| Image resize | 1 core | 10ms | Yes (per image) |
| Normalization | 1 core | 5ms | Yes |
| Bubble detection | 1 core | 50ms | Yes (per field) |
| OCR (EasyOCR) | 4 cores | 500ms | Limited (GIL) |
| Barcode decode | 1 core | 50ms | Yes |

**Python GIL Limitation**:
```python
# ThreadPoolExecutor doesn't help for CPU-bound tasks (GIL)
# Must use ProcessPoolExecutor:
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
    results = executor.map(process_image, images)

# But: Requires pickling ProcessingContext (may fail for complex objects)
```

#### GPU Constraints (ML Models)

```python
# YOLO inference:
GPU memory: ~2 GB for model + batch
Batch size: Limited by GPU memory (typically 1-8 images)

# If no GPU:
CPU inference: 10x slower (500ms → 5000ms per image)
```

**Constraint**: ML fallback not practical on CPU-only systems for real-time processing

---

### 4. Dependency Availability Constraints

#### Optional Dependencies

```python
# ML fallback requires ultralytics:
try:
    from ultralytics import YOLO
except ImportError:
    logger.warning("ML disabled: uv sync --extra ml")
    # Constraint: No ML fallback available

# OCR requires EasyOCR or Tesseract:
try:
    from easyocr import Reader
except ImportError:
    logger.warning("OCR disabled: uv sync --extra ocr")
    # Constraint: OCR fields return empty strings

# Barcode requires PyZbar:
try:
    import pyzbar
except ImportError:
    logger.warning("Barcode disabled: install pyzbar")
    # Constraint: Barcode fields return empty strings
```

**Impact**:
- Missing dependencies = feature disabled
- No error, just silent degradation
- Users must explicitly install extras

---

## Accuracy Constraints

### 1. Image Quality Requirements

#### Minimum Image Quality

| Metric | Minimum | Recommended | Notes |
|--------|---------|-------------|-------|
| **Resolution** | 150 DPI | 300 DPI | Below 150 DPI: bubble detection fails |
| **Contrast** | 30% | 60% | Low contrast: threshold detection unreliable |
| **Noise level** | SNR > 20 dB | SNR > 40 dB | High noise: false detections |
| **Skew angle** | < 5° | < 2° | Large skew: alignment may fail |
| **Blur** | < 2 pixels | < 1 pixel | Heavy blur: OCR/barcode fails |

**Constraint**: Images below minimum quality may produce incorrect results

**Code**: No explicit quality checks (assumes preprocessing fixed issues)

---

### 2. Field Detection Accuracy

#### Bubble Detection Accuracy

```python
# Threshold-based detection accuracy:
High confidence (score > 0.90): ~99% accurate
Medium confidence (0.75-0.90): ~95% accurate
Low confidence (< 0.75): ~85% accurate (triggers ML fallback)

# ML fallback accuracy:
With ML: ~98% accurate (on low-confidence cases)
Without ML: ~85% accurate

# Constraint: 1-2% error rate even with ML
```

**Failure Modes**:
1. **Faint marks**: Below detection threshold
2. **Erasure marks**: Residual darkness above threshold
3. **Multi-marks**: Ambiguous when both bubbles dark
4. **Stray marks**: False positives from scratches

#### OCR Accuracy

```python
# EasyOCR accuracy (English, printed text):
Clean scans: ~98% character accuracy
Low quality: ~85% character accuracy
Handwriting: ~70% character accuracy (not recommended)

# Constraint: OCR unreliable for handwritten text
```

**Failure Modes**:
1. **Similar characters**: 0/O, 1/I, S/5 confusion
2. **Cursive writing**: Not supported well
3. **Multi-line text**: Layout detection issues

#### Barcode Accuracy

```python
# PyZbar accuracy:
High contrast QR: ~99.9% decode rate
Low contrast: ~95% decode rate
Damaged codes: ~70% decode rate

# Constraint: Damaged or low-quality barcodes may not decode
```

---

### 3. Alignment Dependency

**Constraint**: ReadOMR assumes alignment already applied

```python
# If AlignmentProcessor failed or skipped:
# Detection uses unshifted coordinates → misalignment

shift_detection_meta = context.metadata.get("shift_detection")
if not shift_detection_meta:
    # No alignment metadata → may be misaligned
    # ReadOMR proceeds anyway (no validation)
```

**Impact**:
- Misaligned images → wrong bubbles detected
- No built-in alignment validation
- Assumes pipeline ran AlignmentProcessor correctly

---

### 4. Template Configuration Accuracy

**Constraint**: Detection accuracy depends on template quality

```python
# Template must specify:
1. Correct bubble positions (within 5 pixels)
2. Correct field detection types
3. Correct bubble values/labels

# If template wrong:
# Detection runs on wrong regions → incorrect results
```

**Example Failure**:
```json
// Template says bubble at [100, 200]
// Actual bubble at [105, 205] (5 pixel offset)

// If alignment fails:
// Detection extracts wrong region → wrong darkness value
```

---

## Integration Constraints

### 1. Pipeline Dependency Order

**Strict Constraint**: ReadOMR must run AFTER preprocessing and alignment

```python
# Required pipeline order:
1. PreprocessingCoordinator  # Rotation, cropping, filtering
2. AlignmentProcessor        # Shift detection, coordinate warping
3. ReadOMRProcessor          # Detection (THIS)

# If order violated:
# - Skip preprocessing → wrong image dimensions
# - Skip alignment → misaligned detections
```

**Code**: No validation of pipeline order (assumes correct usage)

---

### 2. ShiftDetectionProcessor Conflict

**Constraint**: ReadOMR and ShiftDetectionProcessor cannot both run detection

```python
# Conflict scenario:
Pipeline = [
    AlignmentProcessor,
    ShiftDetectionProcessor,  # Runs detection + applies shifts
    ReadOMRProcessor,         # Would run detection again (redundant)
]

# Solution: Early exit
if context.metadata.get("shift_detection"):
    return context  # Skip ReadOMR
```

**Impact**: If early exit logic removed, detection runs twice (wasted time)

---

### 3. ProcessingContext Mutation

**Constraint**: ReadOMR mutates context in-place

```python
def process(self, context):
    # Mutates context:
    context.omr_response = {...}
    context.is_multi_marked = True
    context.gray_image = normalized_gray  # Replaces original
    context.colored_image = normalized_colored

    return context  # Same object, mutated
```

**Impact**:
- Cannot revert changes after ReadOMR runs
- Original images lost (replaced with normalized)
- Must copy context if original needed

---

### 4. Template Mutability

**Constraint**: TemplateFileRunner does NOT mutate template

```python
# Template is read-only during processing
# Shifts applied to scan_box.shifts by AlignmentProcessor

# If template accidentally mutated:
# - May affect subsequent images (unintended side effects)
```

**Code**: Template passed by reference, but not modified by ReadOMR

---

## Scale Constraints

### 1. Number of Fields

**Practical Limit**: ~1000 fields per template

```python
# Detection time scales linearly with field count:
Time = n_fields × (field_detection_time + field_interpretation_time)

# Example:
100 fields:   100 × 1ms = 100ms (acceptable)
1000 fields:  1000 × 1ms = 1000ms (slow)
10000 fields: 10000 × 1ms = 10000ms (very slow)

# Constraint: Detection becomes impractical beyond 1000 fields
```

**Mitigation**: Split large templates into multiple pages

---

### 2. Number of Bubbles per Field

**Practical Limit**: ~100 bubbles per field

```python
# Bubble detection time:
Time = n_bubbles × bubble_extraction_time

# Example:
4 bubbles (A-D):    4 × 0.1ms = 0.4ms (fast)
26 bubbles (A-Z):   26 × 0.1ms = 2.6ms (acceptable)
100 bubbles:        100 × 0.1ms = 10ms (slow)

# Constraint: Many-bubble fields slow down detection
```

---

### 3. Batch Processing Size

**Practical Limit**: ~10,000 images per batch

```python
# Sequential processing:
Time = n_images × 150ms = 1500 seconds (25 minutes) for 10,000 images

# Parallel processing (8 workers):
Time = n_images × 150ms / 8 = 187 seconds (~3 minutes)

# Constraint: Beyond 10,000 images, consider distributed processing
```

---

### 4. Concurrent Processing

**Python Threading Constraint**:
```python
# GIL limits CPU concurrency
# Max effective workers: ~2-4 (even with 16 cores)

with ThreadPoolExecutor(max_workers=8) as executor:
    # Only 2-4 workers actually run concurrently (GIL)
```

**Browser Constraint**:
```javascript
// Web Worker limit: typically 4-8 workers
// Beyond this: browser may throttle or crash

const maxWorkers = navigator.hardwareConcurrency || 4;  // Typically 4-8
```

---

## Browser-Specific Constraints

### 1. File API Limitations

**Constraint**: Cannot access file system directly

```python
# Python: Direct file access
with open("image.jpg", "rb") as f:
    image = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)

# Browser: Must use File API
const file = await fileInput.files[0];
const arrayBuffer = await file.arrayBuffer();
const imageData = new Uint8Array(arrayBuffer);
```

**Impact**: All file I/O must go through user interaction (file picker, drag-drop)

---

### 2. Model Size Constraints

**Browser Model Limits**:
```
Recommended max model size: 50 MB
Hard limit (mobile): 100 MB

YOLO model: ~50 MB (OK)
Tesseract.js: ~10 MB language files (OK)
@zxing/library: ~500 KB (OK)

Constraint: Models > 100 MB may fail to load on mobile
```

**Mitigation**:
- Use quantized models (INT8 instead of FP32)
- Split models into chunks
- Progressive loading with Web Workers

---

### 3. Memory Constraints (Browser)

**Per-Tab Memory Limits**:
```
Chrome: ~2 GB per tab
Firefox: ~1.5 GB per tab
Safari: ~1 GB per tab
Mobile: ~500 MB per page

Constraint: Process 10-20 images max in browser
```

**Impact**: Cannot process 1000+ images in browser (batch too large)

---

### 4. Synchronous API Unavailable

**Constraint**: All operations must be async in browser

```python
# Python: Synchronous
result = self.template_file_runner.read_omr(...)

# Browser: Must be async
const result = await this.templateFileRunner.readOMR(...);
```

**Impact**:
- Cannot use synchronous libraries
- All detection code must be async
- UI remains responsive (benefit)

---

### 5. WASM Performance

**OpenCV.js Performance**:
```
Python OpenCV: ~10ms for resize
OpenCV.js (WASM): ~20ms for resize (2x slower)

OCR:
Python EasyOCR: ~500ms
Tesseract.js (WASM): ~2000ms (4x slower)

Constraint: Browser detection 2-4x slower than Python
```

---

## Edge Cases and Error Conditions

### 1. Empty Fields

**Constraint**: No bubbles marked

```python
# Bubble field with all bubbles empty:
raw_detections = {"A": 0.05, "B": 0.06, "C": 0.04, "D": 0.05}

# All below threshold (0.50):
interpretation = ""  # Empty string

# Is this correct?
# - Student didn't answer → correct
# - Faint mark missed → incorrect

# Constraint: Cannot distinguish intentional empty vs. missed mark
```

---

### 2. Multi-Marked Fields

**Constraint**: Multiple bubbles above threshold

```python
# Bubble field:
raw_detections = {"A": 0.85, "B": 0.80, "C": 0.10, "D": 0.12}

# Both A and B above threshold (0.50):
# Options:
# 1. Return "A,B" (concatenated)
# 2. Return "MULTI_MARKED" (error flag)
# 3. Return darkest only ("A")

# Current: Depends on template config
is_multi_marked = True  # Flag set
```

---

### 3. OCR Non-Printable Characters

**Constraint**: Handwritten text poorly recognized

```python
# Student writes: "Jöhn Døe" (with diacritics)
# OCR result: "J?hn D?e" (� for unknown chars)

# Constraint: OCR limited to printable ASCII/UTF-8
```

---

### 4. Barcode Partial Damage

**Constraint**: Damaged QR codes may not decode

```python
# 30% of QR code damaged:
result = pyzbar.decode(image)

if len(result) == 0:
    # Failed to decode
    interpretation = ""  # Empty

# Constraint: No partial decoding (all or nothing)
```

---

### 5. Template Field Type Mismatch

**Constraint**: Wrong field detection type specified

```python
# Template says: "field_detection_type": "BUBBLES_THRESHOLD"
# But image actually has QR code at that location

# Detection runs bubble detection on QR code:
# - Treats QR code pixels as bubble darkness
# - Returns garbage result

# Constraint: No automatic field type detection
```

---

## Validation Constraints

### 1. No Pre-Detection Validation

**Constraint**: ReadOMR does not validate inputs

```python
# No checks for:
# - Image quality (resolution, contrast)
# - Template correctness (field positions)
# - Alignment success (shift validation)

# Runs detection regardless of input quality
# Assumes preprocessing/alignment already validated
```

**Impact**: Garbage in, garbage out (no early warnings)

---

### 2. No Post-Detection Validation

**Constraint**: No confidence threshold enforcement

```python
# Even if confidence very low (0.20):
# - Still returns answer
# - No automatic rejection

# User must check:
# - context.field_id_to_interpretation[field].confidence
# - context.is_multi_marked

# Constraint: No built-in quality gates
```

---

### 3. No Cross-Field Validation

**Constraint**: Fields processed independently

```python
# Example: Student ID must be 5 digits
# Field 1: "1"
# Field 2: "2"
# Field 3: "3"
# Field 4: "A"  ← Invalid (should be digit)
# Field 5: "5"

# ReadOMR returns: "123A5"
# No validation that A is invalid for ID field

# Constraint: No semantic validation
```

---

## Summary of Critical Constraints

| Category | Constraint | Impact | Mitigation |
|----------|-----------|--------|------------|
| **Performance** | 150ms/image target | Batch processing slow | Parallel processing, ML only for low confidence |
| **Memory** | 50 MB/image (no ML) | Limited parallel batch size | Process in chunks, release memory |
| **Accuracy** | 95-99% (depends on quality) | 1-5% error rate | ML fallback, quality checks |
| **Dependencies** | Optional libs (ultralytics, OCR) | Features disabled if missing | Graceful degradation |
| **Pipeline Order** | Must run after alignment | Wrong order = incorrect results | Document pipeline requirements |
| **Browser Memory** | 2 GB/tab (desktop) | 10-20 images max | Process in batches, use Web Workers |
| **Browser Performance** | 2-4x slower than Python | Longer processing time | WASM optimization, GPU acceleration |
| **Validation** | No input/output validation | Garbage in, garbage out | Add validation layer upstream |

These constraints define the operational boundaries of the ReadOMR Processor and guide decisions about batch size, parallel processing, and quality thresholds.
