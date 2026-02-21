# ML Bubble Detector - Constraints

**Module**: Domain - Detection - ML Bubble
**Python Reference**: `src/processors/detection/ml_bubble_detector.py`
**Last Updated**: 2026-02-21

---

## Overview

This document outlines the technical constraints, performance limits, and migration challenges for the ML Bubble Detector system.

---

## Performance Constraints

### 1. Inference Time Limits

**Python (CPU)**:
```
Per-block inference: 50-100ms
5 blocks per image: 250-500ms total
Acceptable for batch processing
```

**Python (GPU - CUDA)**:
```
Per-block inference: 10-20ms
5 blocks per image: 50-100ms total
Recommended for real-time processing
```

**Browser (TensorFlow.js WebGL)**:
```
Model load: 1-2s (first time), < 100ms (cached)
Per-block inference: 30-50ms
5 blocks per image: 150-250ms
Acceptable for interactive use
```

**Browser (TensorFlow.js CPU/WASM)**:
```
Per-block inference: 200-300ms
5 blocks per image: 1000-1500ms (1-1.5s)
Too slow for interactive use
⚠️ Requires Web Worker to avoid blocking UI
```

**Browser (ONNX Runtime Web)**:
```
Model load: 500ms-1s
Per-block inference (WebGL): 20-40ms
Per-block inference (WASM): 100-200ms
5 blocks: 100-200ms (WebGL), 500-1000ms (WASM)
Better than TensorFlow.js
```

**Performance Targets**:
- **Batch processing**: < 500ms per image acceptable
- **Interactive use**: < 200ms per image required
- **Real-time**: < 100ms per image (GPU only)

### 2. Memory Constraints

**Python**:
```
YOLO model (YOLOv8n): ~6MB
Model in RAM: ~50MB (loaded)
Per-block crop: ~0.5MB (300x150x1 grayscale)
Peak memory (5 blocks): ~55MB
Acceptable for server/desktop
```

**Browser**:
```
Model (ONNX): ~6-10MB
Model (TensorFlow.js): ~10-15MB
Model in RAM: ~100MB (GPU textures)
Per-block crop: ~0.5MB
Peak memory: ~110MB

Browser limits:
- Desktop Chrome: ~4GB
- Mobile Safari: ~1-2GB
- Total OMRChecker app: ~500MB budget
- ML detector allocation: ~150MB max
✅ Within budget
```

**Memory Management**:
```python
# Python: Automatic garbage collection
block_crop = image[y1:y2, x1:x2]  # View, not copy
# GC handles cleanup

# Browser: Manual cleanup required
const tensor = tf.browser.fromPixels(blockCrop);
const predictions = model.predict(tensor);
// CRITICAL: Dispose tensors
tensor.dispose();
predictions.dispose();
```

### 3. Model Size Constraints

**Uncompressed**:
- YOLOv8n: ~6MB (.pt file)
- YOLOv8s: ~25MB
- YOLOv8m: ~50MB
- YOLOv8l: ~80MB
- YOLOv8x: ~130MB

**Browser-Optimized** (with quantization):
- FP16 (half precision): ~3MB (50% reduction)
- INT8 (8-bit integer): ~1.5MB (75% reduction)
- **Recommended**: FP16 for browser

**Network Transfer**:
```
FP32 (full): 6MB × 8 bits = 48 Mbits ≈ 6s @ 10 Mbps
FP16 (half): 3MB × 8 bits = 24 Mbits ≈ 3s @ 10 Mbps
INT8: 1.5MB × 8 bits = 12 Mbits ≈ 1.5s @ 10 Mbps

Target: < 3s download on average connection
✅ FP16 or INT8 quantization required
```

**Caching Strategy**:
```typescript
// Cache model in IndexedDB
const cachedModel = await loadFromIndexedDB('ml-bubble-model');
if (cachedModel) {
  model = cachedModel;  // Instant load
} else {
  model = await loadModelFromNetwork();
  await saveToIndexedDB('ml-bubble-model', model);
}
```

---

## Browser Migration Constraints

### 1. TensorFlow.js Limitations

**Challenge 1: No Direct YOLO Support**
```
Problem: TensorFlow.js doesn't natively support YOLO format
Solution: Convert YOLO → ONNX → TensorFlow → TensorFlow.js

Conversion Pipeline:
yolo export format=onnx
↓
onnx-tf convert -i model.onnx -o model_tf
↓
tensorflowjs_converter --input_format=tf_saved_model model_tf model_tfjs
↓
TensorFlow.js GraphModel (.json + .bin files)
```

**Challenge 2: Post-Processing**
```
Problem: YOLO post-processing (NMS) not built-in
Solution: Implement custom NMS in JavaScript

// Non-Maximum Suppression
function nms(boxes: Box[], iouThreshold: number): Box[] {
  boxes.sort((a, b) => b.confidence - a.confidence);
  const keep: Box[] = [];

  while (boxes.length > 0) {
    const current = boxes.shift()!;
    keep.push(current);

    boxes = boxes.filter(box => {
      const iou = calculateIoU(current, box);
      return iou < iouThreshold;
    });
  }

  return keep;
}
```

**Challenge 3: Memory Leaks**
```typescript
// BAD: Memory leak
const tensor = tf.browser.fromPixels(image);
const predictions = model.predict(tensor);
// ❌ Tensors not disposed → memory leak

// GOOD: Proper cleanup
const tensor = tf.browser.fromPixels(image);
const predictions = model.predict(tensor) as tf.Tensor;
const data = await predictions.data();

// Cleanup
tensor.dispose();
predictions.dispose();
```

**Challenge 4: WebGL Context Limits**
```
Problem: Browsers limit WebGL contexts (~16 per page)
Solution: Reuse single model instance, avoid multiple tf.loadGraphModel() calls

// SINGLETON pattern
class MLBubbleDetector {
  private static instance: MLBubbleDetector;
  private static model: tf.GraphModel;

  static async getInstance(): Promise<MLBubbleDetector> {
    if (!MLBubbleDetector.instance) {
      MLBubbleDetector.model = await tf.loadGraphModel('model.json');
      MLBubbleDetector.instance = new MLBubbleDetector();
    }
    return MLBubbleDetector.instance;
  }
}
```

### 2. ONNX Runtime Web Limitations

**Challenge 1: WebGL Backend Initialization**
```
Problem: WebGL backend may fail to initialize on some browsers
Solution: Fallback to WASM backend

import * as ort from 'onnxruntime-web';

try {
  ort.env.wasm.numThreads = 4;
  ort.env.wasm.simd = true;
  const session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ['webgl', 'wasm']  // Fallback order
  });
} catch (e) {
  // Fallback to CPU-only
  const session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ['wasm']
  });
}
```

**Challenge 2: Preprocessing Differences**
```
Problem: ONNX expects specific input format (NCHW vs NHWC)
Solution: Manual tensor layout conversion

// TensorFlow.js uses NHWC (batch, height, width, channels)
// ONNX expects NCHW (batch, channels, height, width)

function preprocessForONNX(imageData: ImageData): Float32Array {
  const { width, height, data } = imageData;
  const pixels = new Float32Array(3 * height * width);

  // Convert RGBA to RGB and transpose to NCHW
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const j = y * width + x;

      pixels[0 * height * width + j] = data[i] / 255;     // R
      pixels[1 * height * width + j] = data[i + 1] / 255; // G
      pixels[2 * height * width + j] = data[i + 2] / 255; // B
    }
  }

  return pixels;
}
```

**Challenge 3: Output Parsing**
```
Problem: YOLO output format varies by export method
Solution: Detect output shape and parse accordingly

const results = await session.run(feeds);
const output = results.output0;

// YOLOv8 output: [1, 84, 8400] (for 80 classes)
// For 3 classes: [1, 8, 8400]
// Format: [x_center, y_center, width, height, conf_class0, conf_class1, conf_class2]

const shape = output.dims;  // [1, 8, 8400]
const data = output.data as Float32Array;

const numDetections = shape[2];  // 8400
const numValues = shape[1];      // 8

for (let i = 0; i < numDetections; i++) {
  const offset = i * numValues;
  const xCenter = data[offset];
  const yCenter = data[offset + 1];
  const width = data[offset + 2];
  const height = data[offset + 3];
  const confEmpty = data[offset + 4];
  const confFilled = data[offset + 5];
  const confPartial = data[offset + 6];

  const maxConf = Math.max(confEmpty, confFilled, confPartial);
  if (maxConf < threshold) continue;

  // ... process detection
}
```

### 3. Mobile Browser Constraints

**Challenge 1: Limited RAM**
```
Problem: Mobile Safari limits RAM to 1-2GB
Solution: Aggressive memory management + model quantization

// Strategies:
1. Use INT8 quantized model (~1.5MB vs 6MB)
2. Process one image at a time (no batch processing)
3. Clear tensors immediately after use
4. Limit concurrent Web Workers to 2
5. Use OffscreenCanvas to reduce main thread pressure
```

**Challenge 2: No WebGL on Some Devices**
```
Problem: Older mobile devices lack WebGL support
Solution: Feature detection + graceful degradation

async function getOptimalBackend(): Promise<string> {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');

  if (gl) {
    return 'webgl';  // GPU acceleration
  } else {
    return 'wasm';   // CPU fallback
  }
}

const backend = await getOptimalBackend();
if (backend === 'wasm') {
  logger.warn('WebGL not supported, using WASM (slower)');
}
```

**Challenge 3: Thermal Throttling**
```
Problem: Mobile devices throttle after extended ML inference
Solution: Batch processing with cooldown periods

async function processImages(images: ImageData[]): Promise<void> {
  const BATCH_SIZE = 5;
  const COOLDOWN_MS = 2000;

  for (let i = 0; i < images.length; i += BATCH_SIZE) {
    const batch = images.slice(i, i + BATCH_SIZE);

    // Process batch
    for (const image of batch) {
      await detectBubbles(image);
    }

    // Cooldown to prevent throttling
    if (i + BATCH_SIZE < images.length) {
      await sleep(COOLDOWN_MS);
    }
  }
}
```

---

## Training Constraints

### 1. Dataset Requirements

**Minimum Dataset Size**:
```
bubble_empty: 500+ samples
bubble_filled: 500+ samples
bubble_partial: 300+ samples
Total: 1300+ annotated bubbles

Recommended: 2000+ samples per class
Ideal: 5000+ samples per class (for production)
```

**Image Quality**:
```
✅ Required:
- Resolution: 300+ DPI scans
- Format: PNG or JPEG (high quality)
- Grayscale or RGB
- Clear bubble boundaries

❌ Avoid:
- Low resolution (< 150 DPI)
- Heavy compression artifacts
- Blurry images
- Partial bubbles cut off by crop
```

**Augmentation Requirements**:
```python
# Essential augmentations for robustness
- Rotation: ±5 degrees (account for scan skew)
- Brightness: ±20% (lighting variations)
- Contrast: ±15% (scan quality variations)
- Gaussian blur: σ=0-2 (simulate focus issues)
- Gaussian noise: σ=0-10 (scanner artifacts)

# Not recommended:
- Heavy rotation (> 15 degrees): Unrealistic for OMR
- Flipping: Breaks left-to-right reading order
- Extreme crops: May cut off bubble context
```

### 2. Training Time Constraints

**Hardware Requirements**:
```
CPU training: ~2-4 hours for 100 epochs (slow, not recommended)
GPU training (GTX 1080): ~30-60 minutes for 100 epochs
GPU training (RTX 3090): ~15-30 minutes for 100 epochs

Recommended: GPU with 8GB+ VRAM
```

**Hyperparameter Constraints**:
```yaml
# YOLOv8 training config
epochs: 100-200  # More for small datasets
batch_size: 16   # Limited by GPU memory
imgsz: 640       # Standard for YOLOv8
patience: 50     # Early stopping patience

# Learning rate
lr0: 0.01        # Initial learning rate
lrf: 0.01        # Final learning rate

# Augmentation
degrees: 5       # Max rotation
translate: 0.1   # Translation (10%)
scale: 0.1       # Scale variation
```

### 3. Validation Constraints

**Minimum Validation Set**:
```
20% of total dataset
At least 200 validation samples
Stratified sampling (maintain class distribution)

Metrics to track:
- mAP@0.5: > 0.85 (good)
- Precision: > 0.90 (minimize false positives)
- Recall: > 0.85 (minimize false negatives)
- Inference time: < 50ms per block (on target hardware)
```

**Cross-Validation**:
```
K-fold cross-validation not typical for YOLO
Instead: Fixed validation set + test set

Training: 70%
Validation: 15%
Test (held-out): 15%
```

---

## Deployment Constraints

### 1. Model Serving

**Python (Server)**:
```python
# Constraint: Single model instance per process
# Reason: YOLO models are large, loading multiple instances wastes memory

# GOOD: Singleton pattern
_model_cache = {}

def get_model(model_path: str) -> YOLO:
    if model_path not in _model_cache:
        _model_cache[model_path] = YOLO(model_path)
    return _model_cache[model_path]

# BAD: Multiple instances
def detect(image):
    model = YOLO("bubble_detector.pt")  # ❌ Reloads every call
    return model.predict(image)
```

**Browser (Client)**:
```typescript
// Constraint: Model cached in IndexedDB
// Reason: Avoid re-downloading on every page load

class ModelCache {
  private static db: IDBDatabase;

  static async cacheModel(modelUrl: string): Promise<void> {
    const response = await fetch(modelUrl);
    const blob = await response.blob();

    const transaction = this.db.transaction(['models'], 'readwrite');
    const store = transaction.objectStore('models');
    await store.put({ url: modelUrl, blob, timestamp: Date.now() });
  }

  static async getCachedModel(modelUrl: string): Promise<Blob | null> {
    const transaction = this.db.transaction(['models'], 'readonly');
    const store = transaction.objectStore('models');
    const result = await store.get(modelUrl);

    // Expire cache after 7 days
    if (result && Date.now() - result.timestamp < 7 * 24 * 60 * 60 * 1000) {
      return result.blob;
    }

    return null;
  }
}
```

### 2. Dependency Constraints

**Python**:
```toml
# pyproject.toml
[project.optional-dependencies]
ml = [
    "ultralytics>=8.0.0",  # YOLOv8
    "torch>=2.0.0",        # PyTorch (required by ultralytics)
    "torchvision>=0.15.0", # Vision utilities
]

# Install: uv sync --extra ml
# Size: ~2GB (PyTorch + dependencies)
```

**Browser**:
```json
// package.json
{
  "dependencies": {
    "@tensorflow/tfjs": "^4.11.0",        // ~15MB
    "onnxruntime-web": "^1.16.0",         // ~10MB
    "@tensorflow/tfjs-backend-webgl": "^4.11.0"  // GPU acceleration
  }
}

// Total bundle impact: ~25MB (before tree-shaking)
```

**Constraint**: ML dependencies are optional
```
Graceful degradation if ML not available:
1. Check for model file
2. Check for ML library
3. Fall back to traditional detection if missing
```

### 3. Version Compatibility

**Python**:
```
Python: 3.10+
ultralytics: 8.0.0+
torch: 2.0.0+
CUDA (optional): 11.8+

Known issues:
- ultralytics < 8.0.0: Different export API
- torch < 2.0.0: Slower inference
- CUDA < 11.8: Limited GPU features
```

**Browser**:
```
TensorFlow.js: 4.11.0+
ONNX Runtime Web: 1.16.0+

Browser requirements:
- Chrome 90+ (WebGL 2.0)
- Safari 15+ (WebGL 2.0)
- Firefox 88+ (WebGL 2.0)

Fallback for older browsers:
- TensorFlow.js WASM backend
- CPU-only inference (slow)
```

---

## Edge Cases & Limitations

### 1. Field Block Detection Failures

**Constraint**: ML Bubble Detector depends on Stage 1 field block detection

**Edge Case 1: No field blocks detected**
```python
ml_blocks = context.metadata.get("ml_detected_blocks", [])

if not ml_blocks:
    logger.debug("No ML-detected blocks, skipping ML bubble detection")
    return context  # Graceful skip
```

**Edge Case 2: Incorrect field block bounding boxes**
```
Problem: Stage 1 detector outputs wrong bounding box
Impact: Bubble detector crops wrong region
Solution: Validate bounding boxes before cropping

def _crop_block_region(self, image, bbox_xyxy: list) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy

    # Bounds checking
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    # Validate: x2 > x1 and y2 > y1
    if x2 <= x1 or y2 <= y1:
        logger.warning("Invalid bounding box, skipping")
        return None

    return image[y1:y2, x1:x2]
```

### 2. Model Inference Failures

**Edge Case 1: Model file not found**
```python
if not Path(model_path).exists():
    logger.warning(f"ML model not found at {model_path}")
    self.model = None  # Disable detector
```

**Edge Case 2: CUDA out of memory**
```python
try:
    bubble_results = self.model.predict(block_crop, ...)
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        logger.error("GPU out of memory, falling back to CPU")
        # Retry on CPU
        self.model.to('cpu')
        bubble_results = self.model.predict(block_crop, ...)
```

**Edge Case 3: Invalid model format**
```python
try:
    self.model = YOLO(model_path)
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    self.model = None
```

### 3. Coordinate Mapping Issues

**Edge Case 1: Bubble detected outside field block**
```
Problem: YOLO may detect bubbles slightly outside crop region
Solution: Clamp coordinates to field block bounds

x1_full = max(block_x1, min(block_x2, block_x1 + x1_crop))
y1_full = max(block_y1, min(block_y2, block_y1 + y1_crop))
```

**Edge Case 2: Overlapping bubbles**
```
Problem: YOLO may detect multiple overlapping boxes for same bubble
Solution: Non-Maximum Suppression (NMS) - built into YOLO

# NMS is automatic in YOLO predict()
# Uses IoU threshold (default: 0.45)
# Keeps highest confidence box, removes overlaps
```

### 4. Performance Degradation

**Edge Case 1: Large images**
```
Problem: Full image > 4K resolution → slow inference
Solution: Hierarchical detection already crops to smaller regions

# Field block crops are typically 300x150 → 640x640 resize
# Much faster than full image inference
```

**Edge Case 2: Many field blocks**
```
Problem: Image with 20+ field blocks → slow total inference time
Solution: Batch inference (future enhancement)

# Current: Sequential processing
for block in ml_blocks:
    bubbles = detect(block)

# Future: Batch processing
all_crops = [crop_block(block) for block in ml_blocks]
all_bubbles = model.predict_batch(all_crops)
```

---

## Summary

Key constraints:

**Performance**:
- Python: 50-100ms/block (CPU), 10-20ms (GPU)
- Browser: 30-50ms/block (WebGL), 200-300ms (WASM)
- Memory: ~6MB model, ~50MB RAM (Python), ~100MB (browser)

**Browser Migration**:
- TensorFlow.js: Requires YOLO→ONNX→TF→TFJS conversion + custom NMS
- ONNX Runtime Web: Direct YOLO export, better performance
- Mobile: RAM limits, thermal throttling, no WebGL fallback

**Training**:
- Dataset: 1300+ samples minimum, 5000+ recommended
- Hardware: GPU with 8GB+ VRAM
- Metrics: mAP@0.5 > 0.85, Precision > 0.90, Recall > 0.85

**Deployment**:
- Model caching (IndexedDB for browser)
- Singleton pattern (avoid multiple model loads)
- Graceful degradation (optional ML dependencies)

**Edge Cases**:
- No field blocks detected → skip gracefully
- Model failures → disable detector, use traditional
- Coordinate issues → bounds checking and clamping

**Next Steps**: See `integration.md` for pipeline integration and ReadOMR coordination.
