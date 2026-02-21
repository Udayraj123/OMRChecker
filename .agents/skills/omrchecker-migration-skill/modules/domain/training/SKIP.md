# Training System - SKIP for Browser Migration

**Status**: SKIP
**Reason**: Training is a server-side operation not applicable to browser environment
**Created**: 2026-02-20

---

## Overview

The OMRChecker training system (Tasks 9.1-9.2) is designed for **server-side model training** and is not applicable to browser-based JavaScript migration. This document explains why these components should be skipped and what alternatives exist.

---

## Components Being Skipped

### Task 9.1: Training Data Collector

**Python Reference**: `src/processors/training/data_collector.py`

**Purpose**: Collects high-confidence detections from traditional threshold-based methods to build training datasets for YOLO bubble detection models.

**Key Functionality**:
- Extracts ROI points and detection results from high-confidence samples
- Filters detections by confidence threshold (default: 0.85)
- Exports training data with bounding box annotations
- Tracks collection statistics (total processed, high-confidence collected, bubbles collected)
- Saves images and metadata JSON files for downstream YOLO export

**Why Skip for Browser**:
1. **Data Collection is Server-Side**: Collecting training data requires processing large batches of images offline, not a real-time browser operation
2. **Training is Offline**: Model training happens on servers with GPU resources, not in browser
3. **File System Operations**: Requires creating directory structures, saving files, which is limited in browser environment
4. **No User Need**: Browser users consume pre-trained models; they don't train new ones

### Task 9.2: YOLO Exporter

**Python Reference**: `src/processors/training/yolo_exporter.py`

**Purpose**: Converts OMRChecker ROI data (bounding boxes, labels) to YOLO annotation format for training YOLO object detection models.

**Key Functionality**:
- Converts ROI data to YOLO format (normalized coordinates)
- Supports two dataset types:
  - Bubble detection (Stage 2): Individual bubble bounding boxes
  - Field block detection (Stage 1): Field block bounding boxes
- Splits data into train/val sets (70/30 split)
- Creates YOLO data.yaml configuration file
- Exports annotations in YOLO format: `<class_id> <x_center> <y_center> <width> <height>`

**Class Mappings**:
```python
BUBBLE_CLASSES = {
    "bubble_empty": 0,
    "bubble_filled": 1,
    "bubble_partial": 2,
}

FIELD_BLOCK_CLASSES = {
    "field_block_mcq": 0,
    "field_block_ocr": 1,
    "field_block_barcode": 2,
}
```

**Why Skip for Browser**:
1. **Training Format Export**: YOLO format export is only needed for training, not inference
2. **Server-Side Tool**: This is a data preparation tool for offline training pipelines
3. **No Browser Use Case**: Browser applications use pre-trained models (TensorFlow.js, ONNX.js)
4. **File System Heavy**: Requires creating directory structures, copying images, writing annotation files

---

## What Browser Implementation Should Do Instead

### 1. Use Pre-Trained Models

Browser implementation should use **pre-trained models** exported in web-compatible formats:

- **YOLO Models**: Convert to TensorFlow.js format
  ```bash
  # Server-side conversion
  python -m tensorflowjs_converter \
    --input_format keras \
    --output_format tfjs_graph_model \
    yolo_model.h5 \
    web/models/yolo/
  ```

- **ONNX Models**: Use ONNX Runtime Web
  ```javascript
  import * as ort from 'onnxruntime-web';
  const session = await ort.InferenceSession.create('model.onnx');
  ```

### 2. Model Inference Only

Browser should focus on **inference** (running pre-trained models), not training:

```javascript
// Example: TensorFlow.js inference
import * as tf from '@tensorflow/tfjs';

async function detectBubbles(imageData) {
  const model = await tf.loadGraphModel('/models/yolo/model.json');
  const tensor = tf.browser.fromPixels(imageData);
  const predictions = await model.predict(tensor.expandDims(0));
  return predictions;
}
```

### 3. Model Download & Caching

Browser implementation should:
- Download pre-trained models from CDN/server
- Cache models in IndexedDB or Cache API
- Load models on-demand based on template configuration

```javascript
// Example: Model caching
async function loadModel(modelUrl) {
  const cache = await caches.open('omr-models');
  const cached = await cache.match(modelUrl);

  if (cached) {
    return cached.json();
  }

  const response = await fetch(modelUrl);
  await cache.put(modelUrl, response.clone());
  return response.json();
}
```

### 4. Fallback to Traditional Methods

When ML models are not available or fail:
- Use threshold-based bubble detection (classical approach)
- Provide configuration to disable ML features
- Progressive enhancement: start with classical, enhance with ML if available

```javascript
// Example: Fallback strategy
async function detectBubbles(image, config) {
  if (config.useML && await isModelAvailable('bubble-detector')) {
    return mlBubbleDetection(image);
  }

  // Fallback to classical threshold method
  return thresholdBubbleDetection(image);
}
```

---

## Migration Notes

### Server-Side Training Pipeline

If you need to train models for browser deployment:

1. **Train on Server**: Use Python OMRChecker to collect data and train models
2. **Export for Web**: Convert trained models to TensorFlow.js or ONNX format
3. **Deploy Models**: Host models on CDN or bundle with application
4. **Version Models**: Use versioned URLs for model updates

### Alternative: Transfer Learning in Browser

While not recommended for production, it's technically possible to do transfer learning in browser using TensorFlow.js:

```javascript
// Example: Transfer learning (for advanced use cases only)
import * as tf from '@tensorflow/tfjs';

async function fineTuneModel(baseModel, trainingData) {
  // Freeze base layers
  for (const layer of baseModel.layers.slice(0, -3)) {
    layer.trainable = false;
  }

  // Add custom layers
  const model = tf.sequential({
    layers: [
      ...baseModel.layers,
      tf.layers.dense({ units: 128, activation: 'relu' }),
      tf.layers.dense({ units: 2, activation: 'softmax' })
    ]
  });

  // Compile and train
  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  await model.fit(trainingData.x, trainingData.y, {
    epochs: 10,
    batchSize: 32
  });

  return model;
}
```

**Warning**: This approach is:
- Memory intensive
- Slow compared to server training
- Not recommended for production use
- Only suitable for small datasets and simple models

---

## Related Documentation

When migrating detection components, see:
- **modules/domain/detection/ml-bubble/**: ML bubble detector (inference only)
- **modules/domain/detection/ml-field-block/**: ML field block detector (inference only)
- **modules/migration/ml-model-migration.md**: Comprehensive ML migration guide
- **modules/technical/concurrency/**: For Web Workers usage in ML inference

---

## Summary

**Skip Rationale**: Training system components are server-side tools for building ML models. Browser implementation should focus on **inference** using pre-trained models in web-compatible formats (TensorFlow.js, ONNX).

**Browser Strategy**:
1. Use pre-trained models exported from server
2. Implement model loading and caching
3. Provide fallback to classical threshold methods
4. Do NOT attempt browser-based training

**Decision**: These components (9.1-9.2) are intentionally excluded from browser migration scope.
