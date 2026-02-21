# ML Model Migration for Browser

**Module**: modules/migration/
**Created**: 2026-02-20

---

## Model Format Conversions

### YOLO → TensorFlow.js

**Step 1: Convert to TensorFlow**
```bash
# Using yolov5
python export.py --weights yolov5s.pt --include saved_model
```

**Step 2: Convert to TensorFlow.js**
```bash
tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_format=tfjs_graph_model \
  --signature_name=serving_default \
  yolov5_saved_model \
  web_model/
```

**Step 3: Load in Browser**
```javascript
import * as tf from '@tensorflow/tfjs';

const model = await tf.loadGraphModel('/models/yolo/model.json');

// Inference
const imageTensor = tf.browser.fromPixels(imageElement);
const resized = tf.image.resizeBilinear(imageTensor, [640, 640]);
const normalized = resized.div(255.0);
const batched = normalized.expandDims(0);

const predictions = await model.predict(batched);
```

---

## PyTorch → ONNX → ONNX Runtime Web

**Step 1: Export to ONNX**
```python
import torch

model = YourModel()
model.eval()

dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=14,
    input_names=['input'],
    output_names=['output']
)
```

**Step 2: Use in Browser**
```javascript
import * as ort from 'onnxruntime-web';

const session = await ort.InferenceSession.create('/models/model.onnx');

// Prepare input
const inputTensor = new ort.Tensor('float32', imageData, [1, 3, 640, 640]);

// Run inference
const outputs = await session.run({ input: inputTensor });
const predictions = outputs.output.data;
```

---

## Barcode Detection: PyZbar → @zxing/library

**Python (PyZbar)**:
```python
from pyzbar.pyzbar import decode

barcodes = decode(image)
for barcode in barcodes:
    data = barcode.data.decode('utf-8')
```

**Browser (@zxing/library)**:
```javascript
import { BrowserMultiFormatReader } from '@zxing/library';

const codeReader = new BrowserMultiFormatReader();

// From image element
const result = await codeReader.decodeFromImageElement(imgElement);
console.log(result.text);

// From canvas
const result = await codeReader.decodeFromCanvas(canvas);
```

---

## OCR: EasyOCR/Tesseract → Tesseract.js

**Python (EasyOCR)**:
```python
import easyocr

reader = easyocr.Reader(['en'])
result = reader.readtext(image)
```

**Browser (Tesseract.js)**:
```javascript
import { createWorker } from 'tesseract.js';

const worker = await createWorker('eng');

const { data: { text } } = await worker.recognize(image);
console.log(text);

await worker.terminate();
```

**With Progress**:
```javascript
const worker = await createWorker('eng', 1, {
  logger: (m) => {
    if (m.status === 'recognizing text') {
      console.log(`Progress: ${Math.round(m.progress * 100)}%`);
    }
  }
});
```

---

## Model Optimization for Browser

### 1. Quantization

**TensorFlow.js**:
```bash
tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_format=tfjs_graph_model \
  --quantize_uint8 \
  saved_model \
  quantized_model/
```

**Benefits**: 4x smaller model size, faster loading

### 2. Model Pruning

```python
import tensorflow_model_optimization as tfmot

# Prune model
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
pruned_model = prune_low_magnitude(model)

# Train and export
# ... training code ...
```

### 3. WebAssembly Backend

```javascript
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';

// Use WASM backend for better performance
await tf.setBackend('wasm');
```

---

## Browser ML Model Serving

### Option 1: CDN Hosting

```javascript
// Load from CDN
const model = await tf.loadGraphModel(
  'https://cdn.example.com/models/yolo/model.json'
);
```

### Option 2: Self-Hosted

```javascript
// Load from same origin
const model = await tf.loadGraphModel('/static/models/yolo/model.json');
```

### Option 3: IndexedDB Cache

```javascript
async function loadModelWithCache(modelUrl) {
  const db = await openDB('ml-models', 1);

  // Try cache first
  const cached = await db.get('models', modelUrl);
  if (cached) {
    return tf.loadGraphModel(tf.io.fromMemory(cached));
  }

  // Load from network
  const model = await tf.loadGraphModel(modelUrl);

  // Cache for next time
  const modelArtifacts = await model.save(tf.io.withSaveHandler(
    async (artifacts) => artifacts
  ));
  await db.put('models', modelArtifacts, modelUrl);

  return model;
}
```

---

## Performance Optimization

### 1. WebGL Backend (GPU)

```javascript
await tf.setBackend('webgl');

// Check if WebGL is available
const backend = tf.getBackend();
console.log(`Using backend: ${backend}`);
```

### 2. Model Warmup

```javascript
// Warm up model with dummy input
async function warmupModel(model, inputShape) {
  const dummy = tf.zeros(inputShape);
  await model.predict(dummy);
  dummy.dispose();
}

await warmupModel(model, [1, 640, 640, 3]);
```

### 3. Batch Processing

```javascript
// Process multiple images at once
async function batchPredict(model, images) {
  const tensors = images.map(img => tf.browser.fromPixels(img));
  const batched = tf.stack(tensors);
  const predictions = await model.predict(batched);

  // Clean up
  tensors.forEach(t => t.dispose());
  batched.dispose();

  return predictions;
}
```

---

## Fallback Strategies

### Hybrid Approach

```javascript
async function detectBubbles(image, config) {
  // Try ML if available and enabled
  if (config.useML && await isMLModelLoaded()) {
    try {
      return await mlBubbleDetection(image);
    } catch (error) {
      console.warn('ML detection failed, falling back to threshold', error);
    }
  }

  // Fallback to classical method
  return thresholdBubbleDetection(image);
}
```

### Progressive Enhancement

```javascript
// Start with classical methods
let detectionMethod = 'threshold';

// Upgrade to ML when model loads
async function initializeML() {
  try {
    await loadMLModel();
    detectionMethod = 'ml';
    console.log('ML model loaded, upgraded to ML detection');
  } catch (error) {
    console.warn('Failed to load ML model, using threshold detection');
  }
}

// Load ML in background
initializeML();
```

---

## Model Size Guidelines

| Model Type | Size | Loading Time (4G) | Recommendation |
|------------|------|-------------------|----------------|
| Tiny YOLO | 5-10 MB | 2-3s | Good for mobile |
| Standard YOLO | 20-50 MB | 5-10s | Desktop only |
| Large YOLO | 100+ MB | 20+ s | Avoid for web |
| Tesseract.js | 2-3 MB | 1-2s | Acceptable |
| Custom Bubble Detector | 1-5 MB | 1-2s | Ideal |

**Recommendation**: Keep total model size under 10 MB for good UX

---

## Browser ML Libraries Comparison

| Library | Best For | Size | GPU Support |
|---------|----------|------|-------------|
| **TensorFlow.js** | General ML, YOLO | Medium | ✅ WebGL |
| **ONNX Runtime Web** | PyTorch models | Small | ✅ WebGL |
| **Tesseract.js** | OCR | Small | ❌ |
| **@zxing/library** | Barcodes | Small | ❌ |
| **ml5.js** | Beginner-friendly | Large | ✅ |

---

## Summary

**Conversion Path**:
1. YOLO: PyTorch → ONNX → ONNX Runtime Web (or → TensorFlow → TF.js)
2. Barcode: PyZbar → @zxing/library
3. OCR: EasyOCR/Tesseract → Tesseract.js

**Best Practices**:
- Quantize models for smaller size
- Use WebGL backend for GPU acceleration
- Cache models in IndexedDB
- Implement fallbacks to classical methods
- Keep model size under 10 MB
- Warm up models before first use
- Monitor performance with tf.profile()

**Performance Priority**:
1. Model size (affects loading time)
2. Inference speed (affects UX)
3. Accuracy (critical for OMR)
