# ML Field Block Detector - Constraints & Edge Cases

## System Constraints

### 1. Model File Size Constraint

**Constraint**: YOLO models range from 6 MB (nano) to 100 MB (extra-large)

**Implications**:
```python
# Model variants
YOLOv8n (nano):      ~6 MB,  80-85% accuracy, ~50ms inference
YOLOv8s (small):     ~22 MB, 85-88% accuracy, ~80ms inference
YOLOv8m (medium):    ~50 MB, 88-90% accuracy, ~120ms inference
YOLOv8l (large):     ~87 MB, 90-92% accuracy, ~180ms inference
YOLOv8x (extra):     ~136 MB, 92-94% accuracy, ~250ms inference

Recommended: YOLOv8s or YOLOv8m (balance size vs. accuracy)
```

**Handling**:
```python
# Check model file exists and size is reasonable
model_path = Path(model_path)

if not model_path.exists():
    logger.error(f"Field block model not found at {model_path}")
    self.model = None
    return

file_size_mb = model_path.stat().st_size / (1024 * 1024)

if file_size_mb > 150:
    logger.warning(f"Large model file ({file_size_mb:.1f} MB), may cause memory issues")
```

**Browser Constraint**:
```javascript
// Model download limits
const MAX_MODEL_SIZE = 100 * 1024 * 1024;  // 100 MB

if (modelSize > MAX_MODEL_SIZE) {
  throw new Error(`Model too large: ${modelSize} bytes (max: ${MAX_MODEL_SIZE})`);
}

// Show download progress for large models
fetch(modelUrl)
  .then(response => {
    const reader = response.body.getReader();
    const contentLength = response.headers.get('Content-Length');

    let receivedLength = 0;

    return new ReadableStream({
      async start(controller) {
        while (true) {
          const { done, value } = await reader.read();

          if (done) break;

          receivedLength += value.length;
          const progress = (receivedLength / contentLength) * 100;
          updateProgressBar(progress);  // UI feedback

          controller.enqueue(value);
        }
        controller.close();
      }
    });
  });
```

---

### 2. GPU Memory Constraint

**Constraint**: YOLO inference requires 800-1200 MB GPU memory

**Implications**:
```python
# Memory usage per batch
Single image (1024×1024):
  - Input tensor: 1 MB
  - Model weights: 6-100 MB (cached)
  - Activations: 800 MB (temporary)
  - Total: ~900 MB GPU memory

Batch of 10 images (if batching):
  - Input tensors: 10 MB
  - Model weights: 6-100 MB (cached)
  - Activations: 1.5 GB (temporary)
  - Total: ~1.6 GB GPU memory
```

**Handling**:
```python
try:
    # Try GPU inference
    results = self.model.predict(image, device="cuda")
except RuntimeError as e:
    if "out of memory" in str(e):
        logger.warning("GPU out of memory, falling back to CPU")
        torch.cuda.empty_cache()
        results = self.model.predict(image, device="cpu")
    else:
        raise
```

**Browser Constraint**:
```javascript
// WebGL texture size limits
const gl = canvas.getContext('webgl2');
const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);

console.log(`Max texture size: ${maxTextureSize}`);  // Typically 4096-16384

if (imageWidth > maxTextureSize || imageHeight > maxTextureSize) {
  // Downsample image to fit WebGL limits
  const scale = maxTextureSize / Math.max(imageWidth, imageHeight);
  imageWidth *= scale;
  imageHeight *= scale;
}
```

---

### 3. Inference Time Constraint

**Constraint**: Real-time processing requires <200ms per image

**Performance Targets**:
```
Desktop CPU (Intel i7):       100-150ms  ✓ Acceptable
Laptop CPU (Intel i5):        150-250ms  ⚠️ Marginal
Mobile CPU (ARM Cortex-A):    500-1000ms ✗ Too slow
GPU (NVIDIA RTX 3060):        20-50ms    ✓ Excellent
```

**Handling**:
```python
import time

def process(self, context):
    start_time = time.time()

    # Run inference
    results = self.model.predict(context.gray_image, ...)

    elapsed_ms = (time.time() - start_time) * 1000

    if elapsed_ms > 500:
        logger.warning(f"Slow inference: {elapsed_ms:.0f}ms (target: <200ms)")
        # Consider: Downsample images, use smaller model, enable GPU

    return context
```

**Optimization Strategies**:
1. **Model Quantization**: INT8 reduces inference time by 30-50%
2. **Input Downsampling**: 1024→640 reduces time by 50% (accuracy trade-off)
3. **GPU Acceleration**: CUDA inference 5-10× faster than CPU
4. **Batch Processing**: Process multiple images simultaneously (GPU only)

---

### 4. Confidence Threshold Constraint

**Constraint**: Threshold affects precision-recall trade-off

**Edge Cases**:

#### Case 1: Threshold Too Low (< 0.5)
```python
# confidence_threshold = 0.3
results = model.predict(image, conf=0.3)

# Problem: Many false positives
detected_blocks = [
    {"class_name": "field_block_mcq", "confidence": 0.45, ...},  # Actual: table border
    {"class_name": "field_block_ocr", "confidence": 0.38, ...},  # Actual: background noise
    {"class_name": "field_block_barcode", "confidence": 0.32, ...},  # Actual: page number
]

# Result: Corrupted alignment adjustments, wasted processing
```

#### Case 2: Threshold Too High (> 0.9)
```python
# confidence_threshold = 0.95
results = model.predict(image, conf=0.95)

# Problem: Misses valid field blocks
detected_blocks = []  # Empty! Model found blocks with conf 0.85-0.90, but rejected

# Result: Falls back to template-only (defeats purpose of ML detection)
```

#### Case 3: Optimal Threshold (0.6-0.8)
```python
# confidence_threshold = 0.7
results = model.predict(image, conf=0.7)

detected_blocks = [
    {"class_name": "field_block_mcq", "confidence": 0.92, ...},  # ✓ Valid
    {"class_name": "field_block_mcq", "confidence": 0.88, ...},  # ✓ Valid
    {"class_name": "field_block_ocr", "confidence": 0.75, ...},  # ✓ Valid
]

# Result: Good balance, reliable detections
```

**Recommended Validation**:
```python
if not (0.0 <= confidence_threshold <= 1.0):
    raise ValueError(f"Invalid confidence threshold: {confidence_threshold} (must be 0.0-1.0)")

if confidence_threshold < 0.5:
    logger.warning(f"Low confidence threshold ({confidence_threshold}), expect false positives")

if confidence_threshold > 0.9:
    logger.warning(f"High confidence threshold ({confidence_threshold}), may miss valid blocks")
```

---

### 5. STN Model Availability Constraint

**Constraint**: STN model is optional; detector must work without it

**Edge Cases**:

#### Case 1: `use_stn=True` but `stn_model_path=None`
```python
detector = MLFieldBlockDetector(
    model_path="models/yolo.pt",
    use_stn=True,       # Requested
    stn_model_path=None # Missing!
)

# Handling:
if use_stn:
    if not stn_model_path:
        logger.warning("use_stn=True but no stn_model_path provided. STN disabled.")
        self.use_stn = False  # Disable gracefully
```

#### Case 2: `stn_model_path` exists but file not found
```python
detector = MLFieldBlockDetector(
    model_path="models/yolo.pt",
    use_stn=True,
    stn_model_path="models/stn_missing.pt"  # File doesn't exist
)

# Handling:
try:
    from src.processors.detection.models.stn_utils import load_stn_model
    self.stn = load_stn_model(stn_model_path, ...)
except FileNotFoundError as e:
    logger.error(f"STN model not found: {e}")
    logger.warning("Continuing without STN preprocessing.")
    self.use_stn = False
    self.stn = None
```

#### Case 3: STN transformation fails at runtime
```python
def _apply_stn(self, image):
    if not self.stn:
        return image

    try:
        return apply_stn_to_image(self.stn, image, device="cpu")
    except Exception as e:
        logger.warning(f"STN transformation failed: {e}. Using original image.")
        return image  # Fallback to original
```

---

### 6. Image Dimension Constraint

**Constraint**: YOLO expects specific input sizes (multiples of 32)

**Valid Sizes**: 640, 1024, 1280, 1600 (multiples of 32 for YOLOv8)

**Edge Cases**:

#### Case 1: Odd Dimensions
```python
# Input image: 1023×1025 pixels (not multiples of 32)
results = model.predict(image, imgsz=1024)

# YOLO auto-pads to nearest multiple:
# 1023 → 1024, 1025 → 1024 (slight distortion)
```

#### Case 2: Very Small Images
```python
# Input image: 200×300 pixels
results = model.predict(image, imgsz=1024)

# YOLO upsamples to 1024×1024 (severe distortion, low accuracy)
# Recommendation: Use imgsz=640 for small images
```

#### Case 3: Very Large Images
```python
# Input image: 4000×6000 pixels
results = model.predict(image, imgsz=1024)

# YOLO downsamples to 1024×1024 (loses detail)
# Recommendation: Use imgsz=1600 or process in tiles
```

**Handling**:
```python
def process(self, context):
    image = context.gray_image
    h, w = image.shape[:2]

    # Determine optimal input size
    if max(h, w) < 800:
        imgsz = 640  # Small images
    elif max(h, w) < 1500:
        imgsz = 1024  # Medium images (default)
    else:
        imgsz = 1280  # Large images (more detail)

    results = self.model.predict(image, imgsz=imgsz, ...)
```

---

### 7. Detection Count Constraint

**Constraint**: YOLO may detect 0 to 100+ objects

**Edge Cases**:

#### Case 1: Zero Detections
```python
# Poor quality image or wrong model
results = model.predict(image, conf=0.7)
detected_blocks = []  # Empty!

# Handling:
if len(detected_blocks) == 0:
    logger.warning("No field blocks detected by ML, using template-only")
    context.metadata["ml_detected_blocks"] = []
    context.metadata["ml_block_alignments"] = {}
    return context
```

#### Case 2: Too Many Detections (>50)
```python
# Very low confidence threshold or repeated patterns
results = model.predict(image, conf=0.3)
detected_blocks = [...]  # 150 blocks!

# Handling:
if len(detected_blocks) > 50:
    logger.warning(f"Unusually high detection count: {len(detected_blocks)}")
    # Filter by higher confidence
    detected_blocks = [b for b in detected_blocks if b["confidence"] > 0.8]
    logger.info(f"Filtered to {len(detected_blocks)} high-confidence blocks")
```

#### Case 3: Duplicate Detections (Overlapping Boxes)
```python
# Multiple detections for same field block
detected_blocks = [
    {"bbox_xyxy": [100, 200, 400, 600], "confidence": 0.92},
    {"bbox_xyxy": [105, 198, 405, 602], "confidence": 0.88},  # ~90% overlap
]

# Handling: Non-Maximum Suppression (NMS)
# YOLO applies NMS automatically, but can configure:
results = model.predict(image, iou=0.5)  # IoU threshold for NMS
# iou=0.5: Boxes with >50% overlap are suppressed (keep highest confidence)
```

---

### 8. Class Imbalance Constraint

**Constraint**: Training data may have unequal class distribution

**Impact**:
```
Training Data:
  field_block_mcq:     1000 samples (80%)
  field_block_ocr:     200 samples (16%)
  field_block_barcode: 50 samples (4%)

Result:
  MCQ detection: 92% accuracy ✓
  OCR detection: 78% accuracy ⚠️
  Barcode detection: 65% accuracy ✗
```

**Mitigation** (during training):
```python
# Class weights to balance loss
class_weights = {
    0: 1.0,   # MCQ (baseline)
    1: 5.0,   # OCR (5× weight)
    2: 20.0   # Barcode (20× weight)
}

# Data augmentation for minority classes
# - Rotate barcode samples (0°, 90°, 180°, 270°)
# - Add noise to OCR samples
# - Oversample rare classes
```

**Runtime Handling**:
```python
# Adjust confidence thresholds per class
class_thresholds = {
    "field_block_mcq": 0.7,      # Strict (many samples)
    "field_block_ocr": 0.6,      # Lenient (fewer samples)
    "field_block_barcode": 0.5   # Very lenient (rare class)
}

# Filter detections
filtered_blocks = []
for block in detected_blocks:
    threshold = class_thresholds.get(block["class_name"], 0.7)
    if block["confidence"] >= threshold:
        filtered_blocks.append(block)
```

---

### 9. Template-ML Mismatch Constraint

**Constraint**: Template may define blocks that ML doesn't detect (or vice versa)

**Edge Cases**:

#### Case 1: Template Has More Blocks
```python
# Template: 10 field blocks
# ML Detected: 7 field blocks (missed 3)

adjustments = compute_alignment_adjustments(ml_blocks, template_blocks)
# adjustments contains only 7 entries (3 template blocks unmatchged)

# Handling:
unmatched_blocks = [
    block.name for block in template_blocks
    if block.name not in adjustments
]

if unmatched_blocks:
    logger.warning(f"Unmatched template blocks: {unmatched_blocks}")
    # Use template coordinates for unmatched blocks (no ML adjustment)
```

#### Case 2: ML Detects More Blocks
```python
# Template: 5 field blocks
# ML Detected: 8 field blocks (3 extra)

adjustments = compute_alignment_adjustments(ml_blocks, template_blocks)
# adjustments contains 5 entries (3 ML detections unused)

# Handling:
matched_ml_indices = set(
    adjustments[name]["matched_ml_block"]["index"]
    for name in adjustments
)

unmatched_ml_blocks = [
    block for i, block in enumerate(ml_blocks)
    if i not in matched_ml_indices
]

if unmatched_ml_blocks:
    logger.info(f"Unmatched ML detections: {len(unmatched_ml_blocks)}")
    # Could add to template dynamically (future feature)
```

#### Case 3: Complete Mismatch
```python
# Template: Field blocks at [100, 200], [450, 200]
# ML Detected: Field blocks at [800, 900], [1000, 1100] (completely different)

adjustments = compute_alignment_adjustments(ml_blocks, template_blocks)
# adjustments = {} (empty, all distances > 200px threshold)

# Handling:
if len(adjustments) == 0:
    logger.critical("No ML-template matches found, check template or scan orientation")
    # Fallback to template-only, or alert user
```

---

### 10. Coordinate System Constraint

**Constraint**: Bounding boxes must align with image coordinate system

**YOLO Output Format**:
```
Normalized coordinates (0.0-1.0):
  x_center, y_center, width, height

Absolute coordinates (pixels):
  x1, y1, x2, y2 (xyxy format)
```

**Conversion**:
```python
# YOLO internal format: normalized
box_normalized = [0.3, 0.4, 0.25, 0.35]  # [x_center, y_center, w, h]

# Convert to absolute xyxy
image_width = 1024
image_height = 1024

x_center = box_normalized[0] * image_width   # 0.3 * 1024 = 307.2
y_center = box_normalized[1] * image_height  # 0.4 * 1024 = 409.6
box_w = box_normalized[2] * image_width      # 0.25 * 1024 = 256
box_h = box_normalized[3] * image_height     # 0.35 * 1024 = 358.4

x1 = x_center - box_w / 2  # 307.2 - 128 = 179.2
y1 = y_center - box_h / 2  # 409.6 - 179.2 = 230.4
x2 = x_center + box_w / 2  # 307.2 + 128 = 435.2
y2 = y_center + box_h / 2  # 409.6 + 179.2 = 588.8

bbox_xyxy = [int(x1), int(y1), int(x2), int(y2)]  # [179, 230, 435, 588]
```

**Edge Case**: Out-of-Bounds Coordinates
```python
# YOLO may predict boxes slightly outside image bounds
x1, y1, x2, y2 = -5, -10, 1030, 1040  # Outside [0, 0, 1024, 1024]

# Clipping:
x1 = max(0, x1)  # 0
y1 = max(0, y1)  # 0
x2 = min(image_width, x2)  # 1024
y2 = min(image_height, y2)  # 1024

bbox_xyxy = [0, 0, 1024, 1024]  # Clipped to image bounds
```

---

## Browser-Specific Constraints

### 1. ONNX Runtime Web Constraints

**Memory Limit**: Browser tabs typically have 1-4 GB memory limit

**Handling**:
```javascript
// Monitor memory usage
if (performance.memory) {
  const usedMB = performance.memory.usedJSHeapSize / (1024 * 1024);
  console.log(`Memory usage: ${usedMB.toFixed(2)} MB`);

  if (usedMB > 1500) {
    console.warn('High memory usage, consider downsampling images');
  }
}

// Explicitly delete tensors
async function runInference(session, inputTensor) {
  const outputs = await session.run({ images: inputTensor });

  // Process outputs...

  // CRITICAL: Clean up
  inputTensor.dispose();
  Object.values(outputs).forEach(tensor => tensor.dispose());
}
```

### 2. WebGL Texture Size Limit

**Constraint**: Most browsers limit WebGL textures to 4096×4096 or 16384×16384

**Handling**:
```javascript
const gl = canvas.getContext('webgl2');
const maxSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);

console.log(`Max WebGL texture size: ${maxSize}x${maxSize}`);

if (imageWidth > maxSize || imageHeight > maxSize) {
  // Downsample to fit
  const scale = maxSize / Math.max(imageWidth, imageHeight);
  const newWidth = Math.floor(imageWidth * scale);
  const newHeight = Math.floor(imageHeight * scale);

  console.warn(`Downsampling ${imageWidth}x${imageHeight} → ${newWidth}x${newHeight}`);
  image = cv.resize(image, new cv.Size(newWidth, newHeight));
}
```

### 3. Model Loading Time

**Constraint**: Large models (50-100 MB) take 5-30 seconds to download

**Handling**:
```javascript
async function loadModelWithProgress(modelUrl) {
  const response = await fetch(modelUrl);
  const contentLength = response.headers.get('Content-Length');

  let receivedLength = 0;
  const chunks = [];

  const reader = response.body.getReader();

  while (true) {
    const { done, value } = await reader.read();

    if (done) break;

    chunks.push(value);
    receivedLength += value.length;

    const progress = (receivedLength / contentLength) * 100;
    updateProgressBar(progress);  // UI feedback
  }

  const blob = new Blob(chunks);
  return await blob.arrayBuffer();
}
```

### 4. WASM Thread Limit

**Constraint**: SharedArrayBuffer requires specific headers (COOP, COEP)

**Headers Required**:
```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

**Fallback** (if headers not set):
```javascript
// Check if SharedArrayBuffer is available (multi-threading)
const wasmThreadsAvailable = typeof SharedArrayBuffer !== 'undefined';

if (wasmThreadsAvailable) {
  console.log('Using multi-threaded WASM (faster)');
  session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all'
  });
} else {
  console.warn('SharedArrayBuffer unavailable, using single-threaded WASM (slower)');
  session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ['wasm'],
    intraOpNumThreads: 1  // Single-threaded fallback
  });
}
```

### 5. TensorFlow.js vs. ONNX Runtime Web

**Performance Comparison** (1024×1024 image, YOLOv8s):

| Runtime | Load Time | Inference Time | Memory Usage |
|---------|-----------|----------------|--------------|
| ONNX Runtime Web (WASM) | 2-5s | 150-300ms | 800 MB |
| ONNX Runtime Web (WebGL) | 2-5s | 80-150ms | 1.2 GB |
| TensorFlow.js (WASM) | 5-10s | 300-500ms | 1 GB |
| TensorFlow.js (WebGL) | 5-10s | 150-250ms | 1.5 GB |

**Recommendation**: ONNX Runtime Web (WebGL) for best performance

---

## Edge Case Summary Table

| Edge Case | Current Behavior | Recommended Handling |
|-----------|------------------|----------------------|
| Model file not found | `self.model = None`, skip detection | ✓ Graceful degradation |
| STN model missing | Disable STN, continue without | ✓ Fallback to YOLO-only |
| Zero detections | Return empty list | ✓ Log warning, use template |
| Too many detections (>50) | Process all | ⚠️ Filter by higher confidence |
| Confidence < 0.5 | Many false positives | ⚠️ Log warning |
| Confidence > 0.9 | May miss blocks | ⚠️ Log warning |
| Template-ML mismatch | Partial alignments | ✓ Use template for unmatched |
| Out-of-bounds coords | Clipping (implicit) | ✓ Explicit validation |
| GPU out of memory | Crash | ⚠️ Fallback to CPU |
| Image too small (<500px) | Upsampling artifacts | ⚠️ Use smaller imgsz |
| Image too large (>4000px) | Downsampling loss of detail | ⚠️ Use larger imgsz or tiling |

---

## Validation Checklist

### Before Processing

1. ✓ Model file exists and is valid YOLO format
2. ✓ Confidence threshold in valid range [0.0, 1.0]
3. ✓ If `use_stn=True`, STN model path provided and exists
4. ✓ Input image is valid numpy array (H, W) or (H, W, 3)
5. ✓ Image dimensions reasonable (100 < W,H < 10000)

### After Processing

1. ✓ Detections are sorted by position (top-to-bottom, left-to-right)
2. ✓ All bounding boxes within image bounds
3. ✓ All confidence scores in [0.0, 1.0]
4. ✓ All class IDs in [0, 1, 2]
5. ✓ Alignment adjustments have reasonable shifts (< 500 pixels)

---

## Related Constraints

- **YOLO Training Constraints**: See `scripts/ai-generated/train_stn_yolo.py`
- **STN Constraints**: See `modules/domain/detection/stn/constraints.md`
- **Memory Constraints**: See `modules/migration/performance.md`
- **Browser Compatibility**: See `modules/migration/compatibility.md`
