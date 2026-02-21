# OCR Detection Constraints

**Module**: Domain - Detection - OCR
**Python Reference**: `src/processors/detection/ocr/`
**Last Updated**: 2026-02-21

---

## Overview

This document outlines the constraints, limitations, edge cases, and browser migration challenges for the OCR detection system.

---

## Performance Constraints

### Python (EasyOCR) Performance

#### Initialization

**Cold Start (First Run)**:
```
Model Download: 2-5 seconds (if not cached)
Model Loading: 3-5 seconds
GPU Initialization: 1-2 seconds
Total: 6-12 seconds
```

**Warm Start (Subsequent Runs)**:
```
Singleton Pattern: <100ms (reader already loaded)
```

**Memory Usage**:
```
EasyOCR Reader: ~500MB (model weights in RAM)
Per-Detection: ~1-5MB (image processing buffers)
Peak Memory: ~600MB for OCR subsystem
```

#### Per-Field Detection Time

| Zone Size | GPU Time | CPU Time | Description |
|-----------|----------|----------|-------------|
| Small (200×50px) | 200-500ms | 1-2s | Single line (name, ID) |
| Medium (400×100px) | 500ms-1s | 2-4s | Multi-word field |
| Large (800×200px) | 1-2s | 4-8s | Paragraph text |

**GPU Requirements**:
- CUDA-compatible GPU (NVIDIA)
- CUDA Toolkit 11.0+
- 2GB+ VRAM recommended

**CPU Fallback**:
- Automatically used if GPU unavailable
- 3-5x slower than GPU
- Works on all systems

#### Scalability

**Fields Per File**:
```
1-5 OCR fields: <3s total detection time (GPU)
10-20 OCR fields: <10s total detection time (GPU)
50+ OCR fields: 30-60s total detection time (GPU)
```

**Batch Processing**:
- No parallelization within single file (singleton reader)
- Can process multiple files in parallel (reader is thread-safe)
- Optimal: 2-4 worker threads on 4-core CPU

---

### Browser (Tesseract.js) Performance

#### Initialization

**Cold Start**:
```
Language Model Download: 2-5 seconds (~2-4MB)
Worker Initialization: 1-2 seconds
WASM Loading: 500ms-1s
Total: 4-8 seconds
```

**Warm Start**:
```
Cached Worker: <200ms
Cached Language Model: No download needed
```

**Memory Usage**:
```
Tesseract.js Worker: ~50-100MB
Language Model: ~2-4MB (cached)
Per-Detection: ~5-10MB (canvas processing)
Peak Memory: ~150MB for OCR subsystem
```

#### Per-Field Detection Time

| Zone Size | Browser Time | Notes |
|-----------|--------------|-------|
| Small (200×50px) | 1-2s | Single line text |
| Medium (400×100px) | 2-4s | Multi-word field |
| Large (800×200px) | 4-8s | Paragraph text |

**No GPU Acceleration**:
- Browser OCR is CPU-only
- WebGL cannot be used for Tesseract
- **3-5x slower** than Python EasyOCR (GPU)
- **Similar speed** to Python EasyOCR (CPU)

#### Browser Limits

**Maximum Image Size**:
```
Canvas Max Size: 32767×32767px (varies by browser)
Practical Limit: 8000×8000px (memory constraints)
OCR Zone Max: 2000×2000px recommended
```

**Memory Pressure**:
```
Mobile: 100-300MB available
Desktop: 500MB-2GB available
Tesseract.js: Uses ~150MB
Leaves room for: Image processing, UI, other operations
```

**Concurrent Workers**:
```
Desktop (8+ cores): 4 workers max
Mobile (4-6 cores): 2 workers max
Low-end devices: 1 worker only
```

---

## OCR Accuracy Constraints

### EasyOCR Accuracy

**Text Type Performance**:

| Text Type | Accuracy | Confidence | Notes |
|-----------|----------|------------|-------|
| Printed (clear) | 95-99% | 0.90-0.99 | Excellent |
| Printed (low quality) | 85-95% | 0.75-0.90 | Good |
| Handwritten (neat) | 80-90% | 0.70-0.85 | Acceptable |
| Handwritten (messy) | 60-75% | 0.50-0.70 | Poor |
| Handwritten (illegible) | <50% | <0.50 | Fails threshold |

**Character-Specific Challenges**:

```
Easily Confused:
  O vs 0 (letter O vs zero)
  I vs l vs 1 (capital I, lowercase L, one)
  S vs 5
  Z vs 2
  B vs 8

Requires Context:
  "0RCA" vs "ORCA"
  "B0B" vs "BOB"
  "5ARAH" vs "SARAH"
```

**Language Support**:
- **English**: Excellent (primary use case)
- **Latin Script**: Good (French, Spanish, German)
- **Non-Latin**: Medium (requires language model change)
- **Mixed Scripts**: Poor (English + Arabic)

### Tesseract.js Accuracy

**Compared to EasyOCR**:
```
Printed Text: Similar accuracy (90-95%)
Handwritten Text: Lower accuracy (60-75% vs 80-90%)
Overall: 10-15% less accurate than EasyOCR
```

**Configuration Impact**:
```typescript
// PSM (Page Segmentation Mode)
PSM_SINGLE_LINE: Best for single-line OCR fields
PSM_SINGLE_WORD: Better for isolated words
PSM_AUTO: Slower, less accurate for small zones

// OEM (OCR Engine Mode)
OEM_LSTM_ONLY: Best accuracy, slower
OEM_TESSERACT_ONLY: Faster, less accurate
OEM_DEFAULT: Balanced
```

---

## Edge Cases

### Edge Case 1: Empty Field (No Text)

**Scenario**: Student didn't fill the field

**Python Behavior**:
```python
text_results = EasyOCR.reader.readtext(image_zone)
# Returns: []

text_detection = get_single_text_detection(image_zone)
# Returns: None

self.detections = []
confidence = 0.0
```

**Interpretation**:
```python
marked_interpretations = []
result = self.empty_value  # "" by default
```

**CSV Output**:
```csv
studentName,rollNumber
"",123
```

**Constraint**: Cannot distinguish between:
- Truly empty (no text written)
- Illegible text (OCR failed)
- Very low confidence text (below threshold)

---

### Edge Case 2: Text Outside Scan Zone

**Scenario**: Student wrote outside the designated box

**Behavior**:
```python
# Only scan_zone_rectangle is processed
image_zone = extract_image_from_zone_rectangle(image, scan_zone_rectangle)

# Text outside zone is not included in image_zone
# Result: No detection
```

**Mitigation**:
```python
# Add margins to scan zone
"scanZone": {
  "dimensions": [200, 50],
  "margins": {
    "top": 10,    # Extra 10px above
    "right": 10,  # Extra 10px to right
    "bottom": 10, # Extra 10px below
    "left": 10    # Extra 10px to left
  }
}
# Effective zone: 220×70px
```

**Constraint**: Margins must be balanced with:
- Risk of detecting adjacent fields
- Risk of detecting noise/artifacts

---

### Edge Case 3: Multiple Words in Single Field

**Scenario**: Name field with first and last name

**Current Behavior** (Single Detection):
```python
# EasyOCR detects: ["John", "Doe"] (two separate detections)
# get_single_text_detection() returns highest confidence only
# Result: "John" (if confidence higher than "Doe")
```

**Issue**: Loses "Doe"

**Future Fix** (Multiple Detections):
```python
# get_all_text_detections() returns all detections
# Concatenate: "John" + " " + "Doe" = "John Doe"
```

**Current Workaround**:
- Use larger scan zone to capture both words as single detection
- Rely on EasyOCR to cluster nearby words

**Constraint**: EasyOCR may or may not cluster words depending on:
- Distance between words
- Font size
- Handwriting style

---

### Edge Case 4: Tilted/Rotated Text

**Scenario**: Student wrote at an angle

**EasyOCR Behavior**:
```python
# EasyOCR handles rotation well
# Returns rotated_rectangle (4 points)
# Example: [[10,5], [45,10], [43,22], [8,17]]  (not axis-aligned)
```

**Current Implementation**:
```python
# rotated_rectangle is stored but not used
# bounding_box is used for drawing (axis-aligned)
# Result: Bounding box is larger than actual text
```

**Drawing**:
```
Actual Text (rotated):
  /John/

Bounding Box (axis-aligned):
┌──────┐
│/John/│
└──────┘
```

**Constraint**: Some whitespace in bounding box is unavoidable for rotated text.

---

### Edge Case 5: Low Contrast / Faded Text

**Scenario**: Low-quality scan, light pen, faded photocopy

**Impact on Confidence**:
```
Clear Text: 0.90-0.99 confidence
Medium Contrast: 0.70-0.85 confidence
Low Contrast: 0.50-0.70 confidence (below threshold)
Very Faded: <0.30 confidence (definitely missed)
```

**Mitigation Strategies**:

**1. Preprocessing** (before OCR):
```python
# Increase contrast
enhanced_image = cv2.convertScaleAbs(image, alpha=1.5, beta=10)

# Adaptive thresholding
binary_image = cv2.adaptiveThreshold(
    gray_image,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)
```

**2. Lower Confidence Threshold**:
```python
# Default: 0.8
# For low-quality scans: 0.6-0.7
text_detection = get_single_text_detection(image, confidence_threshold=0.6)
```

**3. Multiple OCR Passes**:
```python
# Try original image first
detection = ocr_with_threshold(image, 0.8)

# If failed, try enhanced image
if detection is None:
    enhanced = enhance_contrast(image)
    detection = ocr_with_threshold(enhanced, 0.7)

# If still failed, try binary image
if detection is None:
    binary = binarize(image)
    detection = ocr_with_threshold(binary, 0.6)
```

**Constraint**: Preprocessing may help or hurt depending on specific image characteristics.

---

### Edge Case 6: Noise / Stray Marks

**Scenario**: Pen marks, smudges, fingerprints in scan zone

**EasyOCR Behavior**:
```python
# May detect noise as text
# Example: Smudge detected as "1" or "l"
# Confidence: Usually low (0.3-0.5)
```

**Current Protection**:
```python
# Confidence threshold filters most noise
if score <= 0.8:  # Noise usually <0.6
    return None
```

**Remaining Risk**:
```
High-confidence noise is rare but possible:
- Stamp/signature in wrong zone
- Adjacent field bleed-through
- Printing artifacts
```

**Mitigation**:
```python
# Character set filtering
"ocrConfig": {
  "charset": "alphanumeric",  # Ignores punctuation/symbols
  "maxLength": 50  # Ignores very long noise
}
```

---

### Edge Case 7: Unicode / Non-ASCII Characters

**Scenario**: Student writes accented characters (José, François)

**Post-Processing Behavior**:
```python
# cleanup_text removes non-ASCII
input: "José García"
output: "Jos Garca"  # é and í removed
```

**Reason**: OpenCV text drawing fails on non-ASCII characters.

**Alternatives**:

**1. Use PIL for Drawing** (instead of OpenCV):
```python
from PIL import Image, ImageDraw, ImageFont

# PIL supports Unicode
draw.text((x, y), "José García", font=font)
```

**2. Transliteration**:
```python
from unidecode import unidecode

text = unidecode("José García")
# Output: "Jose Garcia"
```

**3. No Drawing** (keep original text):
```python
# Store full Unicode text in CSV
# Skip drawing text on image (only draw bounding box)
```

**Current Constraint**: Non-ASCII names will have characters removed.

**Browser**: No constraint (Canvas API supports Unicode natively)

---

### Edge Case 8: Very Long Text

**Scenario**: Student writes paragraph instead of single word

**Current Behavior**:
```python
# get_single_text_detection() returns highest confidence detection
# If paragraph detected as single block: Returns full text
# If paragraph detected as multiple words: Returns first word only
```

**Length Constraint**:
```python
# No hard limit in current implementation
# Future: Add maxLength config
if max_length and len(cleaned_text) > max_length:
    cleaned_text = cleaned_text[:max_length]
```

**Performance Impact**:
```
Larger text zones = slower OCR
800×200px zone with paragraph: 2-3 seconds
800×200px zone with single word: 1-2 seconds
```

---

## Browser-Specific Constraints

### Tesseract.js Migration Challenges

#### Challenge 1: No GPU Acceleration

**Impact**:
- 3-5x slower than Python EasyOCR (GPU)
- Similar speed to Python EasyOCR (CPU)

**Mitigation**:
```typescript
// Use Web Workers to avoid blocking UI
const ocrWorker = new Worker('./ocr-worker.js');
ocrWorker.postMessage({ imageData, threshold: 0.8 });

// Show progress indicator
ocrWorker.onmessage = (e) => {
  if (e.data.status === 'recognizing text') {
    updateProgress(e.data.progress);
  }
};
```

#### Challenge 2: Model Download Size

**Constraint**:
```
English Language Model: ~2-4MB
First Load: 2-5 seconds download
Subsequent Loads: Cached (IndexedDB)
```

**Mitigation**:
```typescript
// Pre-load model on app start
async function preloadOCRModel() {
  const worker = await Tesseract.createWorker('eng', undefined, {
    cacheMethod: 'write',  // Cache to IndexedDB
  });
  await worker.terminate();  // Keep model cached
}

// Call on app initialization
preloadOCRModel();
```

#### Challenge 3: Memory Limits (Mobile)

**Constraint**:
```
Mobile Browsers: 100-300MB available memory
Tesseract.js: ~150MB usage
Risk: Out of memory on low-end devices
```

**Mitigation**:
```typescript
// Process fields sequentially on mobile
const isMobile = /Android|iPhone|iPad/i.test(navigator.userAgent);

if (isMobile) {
  // Sequential processing
  for (const field of ocrFields) {
    const result = await processOCRField(field);
    results.push(result);
  }
} else {
  // Parallel processing (desktop)
  const results = await Promise.all(
    ocrFields.map(field => processOCRField(field))
  );
}
```

#### Challenge 4: Canvas Size Limits

**Constraint**:
```
Canvas Maximum: 32767×32767px (Chrome)
Canvas Maximum: 11180×11180px (Safari)
Practical Limit: 8000×8000px
```

**Mitigation**:
```typescript
// Downscale large images
function preprocessImageForOCR(image: ImageData): ImageData {
  const MAX_DIMENSION = 4000;

  if (image.width > MAX_DIMENSION || image.height > MAX_DIMENSION) {
    const scale = Math.min(
      MAX_DIMENSION / image.width,
      MAX_DIMENSION / image.height
    );
    return downscaleImage(image, scale);
  }

  return image;
}
```

---

## Tesseract.js Browser Migration

### Configuration for Browser

**Optimal Tesseract.js Settings**:
```typescript
await worker.setParameters({
  tessedit_pageseg_mode: Tesseract.PSM.SINGLE_LINE,  // For single-line fields
  tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ',
  tessedit_ocr_engine_mode: Tesseract.OEM.LSTM_ONLY,  // Best accuracy
});
```

**PSM (Page Segmentation Mode) Selection**:
```typescript
// Single line (name, ID)
PSM.SINGLE_LINE  // Use this for most OCR fields

// Multiple words
PSM.AUTO  // Auto-detect layout

// Single word only
PSM.SINGLE_WORD  // Slightly faster for single words
```

**Character Whitelist**:
```typescript
// Alphanumeric only
tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

// Digits only (roll number)
tessedit_char_whitelist: '0123456789'

// Letters only (name)
tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
```

### Performance Optimization

**Worker Pool Pattern**:
```typescript
class TesseractWorkerPool {
  private workers: Tesseract.Worker[] = [];
  private queue: Array<() => void> = [];

  async initialize(poolSize: number) {
    for (let i = 0; i < poolSize; i++) {
      const worker = await Tesseract.createWorker('eng');
      this.workers.push(worker);
    }
  }

  async recognize(imageData: ImageData): Promise<OCRResult> {
    // Get available worker or wait
    const worker = await this.getAvailableWorker();

    try {
      const result = await worker.recognize(imageData);
      return this.parseResult(result);
    } finally {
      this.releaseWorker(worker);
    }
  }
}

// Usage
const pool = new TesseractWorkerPool();
await pool.initialize(2);  // 2 workers for parallel processing

const results = await Promise.all(
  ocrFields.map(field => pool.recognize(field.imageData))
);
```

**Caching Strategy**:
```typescript
// Cache OCR results per file
const ocrCache = new Map<string, OCRFieldDetectionResult>();

async function getOCRResult(
  fieldId: string,
  imageData: ImageData
): Promise<OCRFieldDetectionResult> {
  // Check cache first
  if (ocrCache.has(fieldId)) {
    return ocrCache.get(fieldId)!;
  }

  // Perform OCR
  const result = await performOCR(imageData);

  // Cache result
  ocrCache.set(fieldId, result);

  return result;
}
```

---

## System Resource Constraints

### Python System Requirements

**Minimum**:
- CPU: Dual-core 2.0GHz
- RAM: 2GB (4GB recommended)
- Storage: 1GB (for EasyOCR models)

**Recommended**:
- CPU: Quad-core 2.5GHz+
- RAM: 8GB+
- GPU: NVIDIA with 2GB+ VRAM
- Storage: 2GB (for multiple language models)

### Browser System Requirements

**Minimum**:
- Modern browser (Chrome 90+, Firefox 88+, Safari 14+)
- RAM: 2GB (mobile), 4GB (desktop)
- CPU: Dual-core 1.5GHz

**Recommended**:
- Chrome 100+ or Firefox 100+
- RAM: 4GB+ (mobile), 8GB+ (desktop)
- CPU: Quad-core 2.0GHz+

### Network Constraints (Browser)

**First Load**:
```
Tesseract.js Core: ~500KB
Tesseract.js Worker: ~2MB
English Language Model: ~2-4MB
Total: ~5-7MB
```

**Subsequent Loads**:
```
Cached (IndexedDB/Service Worker)
No network required
```

**Offline Support**:
```typescript
// Service worker caches OCR assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open('ocr-assets').then((cache) => {
      return cache.addAll([
        '/tesseract.js',
        '/tesseract-worker.js',
        '/eng.traineddata.gz'
      ]);
    })
  );
});
```

---

## Error Recovery

### Python Error Handling

**EasyOCR Initialization Failure**:
```python
try:
    EasyOCR.initialize()
except Exception as e:
    logger.error(f"Failed to initialize EasyOCR: {e}")
    # Fallback to Tesseract (future)
    # Or return empty detections
    return []
```

**Detection Runtime Error**:
```python
try:
    text_detection = EasyOCR.get_single_text_detection(image_zone)
except Exception as e:
    logger.warning(f"OCR detection failed for field {field.id}: {e}")
    # Return empty detection
    return []
```

### Browser Error Handling

**Worker Initialization Failure**:
```typescript
try {
  worker = await Tesseract.createWorker('eng');
} catch (error) {
  console.error('Failed to initialize Tesseract worker:', error);
  // Show user error message
  // Disable OCR fields
  return null;
}
```

**Recognition Timeout**:
```typescript
async function recognizeWithTimeout(
  worker: Tesseract.Worker,
  imageData: ImageData,
  timeout: number = 30000  // 30 seconds
): Promise<OCRResult | null> {
  const timeoutPromise = new Promise<null>((resolve) =>
    setTimeout(() => resolve(null), timeout)
  );

  const recognizePromise = worker.recognize(imageData);

  const result = await Promise.race([recognizePromise, timeoutPromise]);

  if (result === null) {
    console.warn('OCR recognition timed out');
  }

  return result;
}
```

---

## Related Documentation

- **Concept**: `modules/domain/detection/ocr/concept.md`
- **Flows**: `modules/domain/detection/ocr/flows.md`
- **Decisions**: `modules/domain/detection/ocr/decisions.md`
- **Integration**: `modules/domain/detection/ocr/integration.md`
- **Performance**: `modules/migration/performance.md`

---

## Summary

Key constraints to remember:

1. **Performance**: EasyOCR GPU ~200-500ms per field, Tesseract.js ~1-2s per field
2. **Accuracy**: 80-90% for neat handwriting, 60-75% for messy handwriting
3. **Memory**: ~500MB for Python, ~150MB for browser
4. **Character Limits**: Non-ASCII removed in Python (OpenCV constraint)
5. **Edge Cases**: Empty fields, tilted text, multiple words, noise
6. **Browser**: No GPU, slower processing, memory limits on mobile
7. **Network**: 5-7MB first load, cached thereafter

**Migration Impact**: Expect 3-5x slower OCR in browser, but acceptable UX with progress indicators and Web Workers.
