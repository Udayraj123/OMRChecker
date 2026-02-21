# ML Bubble Detector - Flows

**Module**: Domain - Detection - ML Bubble
**Python Reference**: `src/processors/detection/ml_bubble_detector.py`
**Last Updated**: 2026-02-21

---

## Overview

This document details the complete data flows for ML-based bubble detection, including YOLO integration, coordinate mapping, and fallback logic.

---

## Flow 1: Hierarchical Detection (Stage 2)

### Context

ML Bubble Detector operates as **Stage 2** of hierarchical detection, processing crops from Stage 1 field blocks.

### Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: ML Field Block Detector                           │
│ - Detects field blocks on full image                       │
│ - Stores in context.metadata["ml_detected_blocks"]         │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: ML Bubble Detector (this module)                  │
│                                                             │
│ For each field block:                                      │
│   1. Crop block region from full image                     │
│   2. Run YOLO bubble detection on crop                     │
│   3. Map bubble coordinates to full image                  │
│   4. Store results in block metadata                       │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Shift Detection or ReadOMR                        │
│ - Uses bubble positions to calculate field shifts          │
│ - Or: Interprets bubble states for final response          │
└─────────────────────────────────────────────────────────────┘
```

### Detailed Steps

```python
def process(self, context: ProcessingContext) -> ProcessingContext:
    """Run bubble detection on field blocks."""

    # ────────────────────────────────────────────────────────
    # Step 1: Validate Model and Dependencies
    # ────────────────────────────────────────────────────────
    if not self.model:
        logger.debug("ML bubble detector disabled (no model)")
        return context

    # ────────────────────────────────────────────────────────
    # Step 2: Get ML-detected Blocks from Context
    # ────────────────────────────────────────────────────────
    ml_blocks = context.metadata.get("ml_detected_blocks", [])

    if not ml_blocks:
        logger.debug("No ML-detected blocks, skipping ML bubble detection")
        return context

    # ────────────────────────────────────────────────────────
    # Step 3: Process Each Field Block
    # ────────────────────────────────────────────────────────
    for block_detection in ml_blocks:
        # 3a. Crop block region
        block_crop = self._crop_block_region(
            context.gray_image,
            block_detection["bbox_xyxy"]
        )

        # 3b. Validate crop
        if block_crop is None or block_crop.size == 0:
            logger.warning(f"Empty crop for block {block_detection['class_name']}")
            continue

        # 3c. Run YOLO inference
        bubble_results = self.model.predict(
            block_crop,
            conf=self.confidence_threshold,
            verbose=False,
            imgsz=640
        )

        # 3d. Map to full coordinates
        bubbles = self._map_to_full_coordinates(
            bubble_results,
            block_detection["bbox_xyxy"]
        )

        # 3e. Store results
        block_detection["ml_bubbles"] = bubbles
        block_detection["ml_bubbles_count"] = len(bubbles)

        logger.debug(
            f"Detected {len(bubbles)} bubbles in block "
            f"{block_detection['class_name']}"
        )

    # ────────────────────────────────────────────────────────
    # Step 4: Log Summary and Return
    # ────────────────────────────────────────────────────────
    logger.info(f"ML bubble detection complete for {len(ml_blocks)} blocks")
    return context
```

### Data Structures

**Input** (from Stage 1):
```python
context.metadata["ml_detected_blocks"] = [
    {
        "class_id": 0,
        "class_name": "mcq_block_4",
        "confidence": 0.92,
        "bbox_xyxy": [100, 200, 400, 350],
        "bbox_origin": [100, 200],
        "bbox_dimensions": [300, 150]
    }
]
```

**Output** (updated blocks):
```python
context.metadata["ml_detected_blocks"] = [
    {
        # Original block fields...
        "class_id": 0,
        "class_name": "mcq_block_4",
        "confidence": 0.92,
        "bbox_xyxy": [100, 200, 400, 350],

        # NEW: ML bubble detections
        "ml_bubbles": [
            {
                "class_id": 1,
                "class_name": "bubble_filled",
                "confidence": 0.89,
                "bbox_xyxy": [150, 230, 175, 255],
                "bbox_origin": [150, 230],
                "bbox_dimensions": [25, 25],
                "state": "filled"
            },
            # ... more bubbles
        ],
        "ml_bubbles_count": 4
    }
]
```

---

## Flow 2: Image Cropping

### Purpose

Extract field block region from full image for focused bubble detection.

### Algorithm

```python
def _crop_block_region(self, image, bbox_xyxy: list) -> np.ndarray:
    """Crop field block region from full image."""

    x1, y1, x2, y2 = bbox_xyxy

    # ────────────────────────────────────────────────────────
    # Step 1: Bounds Checking
    # ────────────────────────────────────────────────────────
    # Ensure coordinates are within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)  # width
    y2 = min(image.shape[0], y2)  # height

    # ────────────────────────────────────────────────────────
    # Step 2: Crop (NumPy Slicing)
    # ────────────────────────────────────────────────────────
    # Creates a view, not a copy (memory efficient)
    return image[y1:y2, x1:x2]
```

### Coordinate System

```
Full Image (1920x1080):
┌──────────────────────────────────────┐
│                                      │
│     Field Block                      │
│     ┌─────────────┐                  │
│     │ (100, 200)  │ ← bbox_xyxy[0:2] │
│     │             │                  │
│     │             │                  │
│     │ (400, 350)  │ ← bbox_xyxy[2:4] │
│     └─────────────┘                  │
│                                      │
└──────────────────────────────────────┘

Cropped Block (300x150):
┌─────────────┐
│ (0, 0)      │ ← Origin shifts to top-left
│             │
│             │
│ (300, 150)  │
└─────────────┘
```

### Edge Cases

**Case 1: Block extends beyond image bounds**
```python
# Input: bbox_xyxy = [1800, 950, 2000, 1150]
# Image size: 1920x1080

# After bounds checking:
x1 = max(0, 1800) = 1800
y1 = max(0, 950) = 950
x2 = min(1920, 2000) = 1920  # Clipped
y2 = min(1080, 1150) = 1080  # Clipped

# Result: Crop is [1800:1920, 950:1080] = 120x130 (valid)
```

**Case 2: Invalid bbox (x1 > x2 or y1 > y2)**
```python
# Input: bbox_xyxy = [400, 350, 100, 200]  # INVALID

# After bounds checking: x1=400, y1=350, x2=100, y2=200
# Crop: image[350:200, 400:100] = empty array

# Handled by validation:
if block_crop is None or block_crop.size == 0:
    logger.warning("Empty crop, skipping")
    continue
```

---

## Flow 3: YOLO Inference

### Purpose

Detect bubbles within cropped field block using trained YOLO model.

### YOLO Prediction Flow

```python
bubble_results = self.model.predict(
    block_crop,
    conf=self.confidence_threshold,  # 0.7 default
    verbose=False,                    # Suppress logs
    imgsz=640,                        # Input size
)
```

### YOLO Internal Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Input: block_crop (300x150 grayscale image)                │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Preprocessing                                            │
│    - Resize to 640x640                                      │
│    - Normalize to [0, 1]                                    │
│    - Convert to RGB (duplicate gray channel)                │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Model Inference                                          │
│    - Forward pass through YOLOv8 network                    │
│    - Outputs: boxes, scores, classes                        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Post-processing (built-in)                               │
│    - Non-Maximum Suppression (NMS)                          │
│    - Filter by confidence threshold (0.7)                   │
│    - Scale boxes back to 300x150 (crop size)                │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Output: Results object                                      │
│ - results[0].boxes: List of detections                      │
│ - Each box: xyxy, conf, cls                                 │
└─────────────────────────────────────────────────────────────┘
```

### Result Structure

```python
# YOLO Results Object
bubble_results = [
    Results(
        boxes=Boxes(
            xyxy=tensor([
                [50, 30, 75, 55],   # Box 1 (in crop coords)
                [50, 60, 75, 85],   # Box 2
                # ...
            ]),
            conf=tensor([0.89, 0.93, ...]),
            cls=tensor([1, 0, ...])  # 1=filled, 0=empty
        ),
        # ... other metadata
    )
]
```

### Confidence Filtering

YOLO automatically filters detections below `conf` threshold:

```python
# Only boxes with confidence >= 0.7 are returned
bubble_results = self.model.predict(block_crop, conf=0.7)

# Example:
# Detected: bubble_1 (conf=0.89) ✓ Kept
# Detected: bubble_2 (conf=0.65) ✗ Filtered out
# Detected: bubble_3 (conf=0.93) ✓ Kept
```

---

## Flow 4: Coordinate Mapping

### Purpose

Map bubble detections from crop-relative coordinates to full image coordinates.

### Algorithm

```python
def _map_to_full_coordinates(
    self, bubble_results, block_bbox_xyxy: list
) -> list[dict]:
    """Map bubble detections from crop to full image coordinates."""

    if not bubble_results or len(bubble_results) == 0:
        return []

    block_x1, block_y1, _, _ = block_bbox_xyxy
    bubbles = []

    for result in bubble_results:
        if not hasattr(result, "boxes") or result.boxes is None:
            continue

        for box in result.boxes:
            # ────────────────────────────────────────────────
            # Step 1: Extract Box Information (crop coords)
            # ────────────────────────────────────────────────
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            xyxy_crop = box.xyxy[0].tolist()
            x1_crop, y1_crop, x2_crop, y2_crop = xyxy_crop

            # ────────────────────────────────────────────────
            # Step 2: Map to Full Image Coordinates
            # ────────────────────────────────────────────────
            x1_full = int(block_x1 + x1_crop)
            y1_full = int(block_y1 + y1_crop)
            x2_full = int(block_x1 + x2_crop)
            y2_full = int(block_y1 + y2_crop)

            # ────────────────────────────────────────────────
            # Step 3: Build Detection Dictionary
            # ────────────────────────────────────────────────
            bubble_detection = {
                "class_id": class_id,
                "class_name": self.CLASS_NAMES.get(class_id, "unknown"),
                "confidence": confidence,
                "bbox_xyxy": [x1_full, y1_full, x2_full, y2_full],
                "bbox_origin": [x1_full, y1_full],
                "bbox_dimensions": [x2_full - x1_full, y2_full - y1_full],
                "state": "filled" if class_id == 1 else "empty",
            }

            bubbles.append(bubble_detection)

    return bubbles
```

### Coordinate Transformation

```
Field Block in Full Image:
    bbox_xyxy = [100, 200, 400, 350]
    block_x1 = 100
    block_y1 = 200

YOLO detects bubble in crop coords:
    [x1_crop, y1_crop, x2_crop, y2_crop] = [50, 30, 75, 55]

Transformation:
    x1_full = 100 + 50 = 150
    y1_full = 200 + 30 = 230
    x2_full = 100 + 75 = 175
    y2_full = 200 + 55 = 255

Full Image Coordinates:
    [150, 230, 175, 255]
```

### Visual Example

```
Full Image (1920x1080):
┌──────────────────────────────────────────────────────┐
│                                                      │
│     Field Block (100, 200) to (400, 350)             │
│     ┌────────────────────────────────┐               │
│     │ (100,200)                      │               │
│     │                                │               │
│     │     Bubble (crop: 50,30-75,55) │               │
│     │     ┌──────┐                   │               │
│     │     │      │ ← Full: (150,230) │               │
│     │     │      │    to (175,255)   │               │
│     │     └──────┘                   │               │
│     │                                │               │
│     │                    (400,350)   │               │
│     └────────────────────────────────┘               │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### Class Mapping

```python
CLASS_NAMES = {
    0: "bubble_empty",
    1: "bubble_filled",
    2: "bubble_partial",
}

# State interpretation
state = "filled" if class_id == 1 else "empty"
# Note: bubble_partial (class_id=2) is treated as "empty"
```

---

## Flow 5: Hybrid Fallback Detection

### Purpose

Use ML bubble detector as fallback when traditional detection has low confidence.

### Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Traditional Bubble Detection (Threshold-based)             │
│ - Detects bubbles, calculates confidence per field         │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ HybridDetectionStrategy.should_use_ml_fallback()           │
│ - Check field confidence scores                            │
│ - Identify low-confidence fields (< 0.75)                  │
└────────────────┬────────────────────────────────────────────┘
                 │
         ┌───────┴───────┐
         │               │
    Low Conf         High Conf
         │               │
         ▼               ▼
┌────────────────┐  ┌────────────────┐
│ Enable ML      │  │ Use Traditional│
│ Fallback       │  │ Results        │
└────────┬───────┘  └────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ ML Bubble Detector                                          │
│ - Run YOLO on full image or field blocks                   │
│ - Extract bubble states                                    │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Use ML Results for Low-Confidence Fields                   │
│ - Replace traditional results with ML results              │
│ - Keep traditional results for high-confidence fields      │
└─────────────────────────────────────────────────────────────┘
```

### Implementation

```python
# In ReadOMR Processor
def process(self, context: ProcessingContext) -> ProcessingContext:
    # ... traditional detection ...

    # Check for low-confidence fields
    if self.hybrid_strategy and self.hybrid_strategy.should_use_ml_fallback(context):
        logger.info("Using ML fallback for low-confidence fields")

        # Enable ML detector
        self.ml_detector.enable_for_low_confidence()

        # Run ML detection
        context = self.ml_detector.process(context)

        # Disable ML detector
        self.ml_detector.disable()

        # Update statistics
        self.hybrid_strategy.stats["ml_fallback_used"] += 1

    return context
```

### Confidence Evaluation

```python
# In HybridDetectionStrategy
def identify_low_confidence_fields(
    self, field_id_to_interpretation: dict
) -> list[tuple[str, float]]:
    """Identify fields with low confidence scores."""

    low_confidence_fields = []

    for field_id, interpretation in field_id_to_interpretation.items():
        # Get confidence metrics
        confidence_metrics = getattr(
            interpretation, "field_level_confidence_metrics", {}
        )
        confidence_score = confidence_metrics.get("overall_confidence_score", 1.0)

        # Check threshold (default: 0.75)
        if confidence_score < self.confidence_threshold:
            low_confidence_fields.append((field_id, confidence_score))
            logger.debug(
                f"Low confidence field: {field_id} "
                f"(confidence: {confidence_score:.3f})"
            )

    return low_confidence_fields
```

### Statistics Tracking

```python
self.stats = {
    "total_fields": 0,
    "high_confidence_fields": 0,
    "low_confidence_fields": 0,
    "ml_fallback_used": 0,  # Number of images where ML was triggered
}

# Log at end of processing
logger.info("=" * 60)
logger.info("ML Fallback Statistics")
logger.info("=" * 60)
logger.info(f"Total fields processed: {stats['total_fields']}")
logger.info(f"High confidence fields: {stats['high_confidence_fields']}")
logger.info(f"Low confidence fields: {stats['low_confidence_fields']}")
logger.info(f"ML fallback used: {stats['ml_fallback_used']} times")
logger.info("=" * 60)
```

---

## Flow 6: Browser Migration Flow

### TensorFlow.js Flow

```typescript
class MLBubbleDetector {
  async process(context: ProcessingContext): Promise<ProcessingContext> {
    // ────────────────────────────────────────────────────────
    // Step 1: Get ML Detected Blocks
    // ────────────────────────────────────────────────────────
    const mlBlocks = context.metadata.mlDetectedBlocks || [];
    if (!mlBlocks.length || !this.model) {
      return context;
    }

    // ────────────────────────────────────────────────────────
    // Step 2: Process Each Block
    // ────────────────────────────────────────────────────────
    for (const block of mlBlocks) {
      // 2a. Crop block region (using canvas)
      const blockCrop = this.cropBlockRegion(
        context.grayImage,
        block.bboxXyxy
      );

      // 2b. Run inference
      const bubbles = await this.detectBubbles(blockCrop, block.bboxXyxy);

      // 2c. Store results
      block.mlBubbles = bubbles;
      block.mlBubblesCount = bubbles.length;
    }

    return context;
  }

  private cropBlockRegion(
    image: ImageData,
    bbox: [number, number, number, number]
  ): ImageData {
    const [x1, y1, x2, y2] = bbox;

    // Bounds checking
    const clampedX1 = Math.max(0, x1);
    const clampedY1 = Math.max(0, y1);
    const clampedX2 = Math.min(image.width, x2);
    const clampedY2 = Math.min(image.height, y2);

    const width = clampedX2 - clampedX1;
    const height = clampedY2 - clampedY1;

    // Create canvas for cropping
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d')!;

    // Create temporary canvas for source image
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = image.width;
    tempCanvas.height = image.height;
    const tempCtx = tempCanvas.getContext('2d')!;
    tempCtx.putImageData(image, 0, 0);

    // Draw cropped region
    ctx.drawImage(
      tempCanvas,
      clampedX1, clampedY1, width, height,
      0, 0, width, height
    );

    return ctx.getImageData(0, 0, width, height);
  }

  private async detectBubbles(
    blockCrop: ImageData,
    blockBbox: [number, number, number, number]
  ): Promise<BubbleDetection[]> {
    // ────────────────────────────────────────────────────────
    // Step 1: Preprocess Image
    // ────────────────────────────────────────────────────────
    const tensor = tf.browser.fromPixels(blockCrop)
      .resizeBilinear([640, 640])
      .div(255.0)
      .expandDims(0);

    // ────────────────────────────────────────────────────────
    // Step 2: Run Inference
    // ────────────────────────────────────────────────────────
    const predictions = this.model!.predict(tensor) as tf.Tensor;

    // ────────────────────────────────────────────────────────
    // Step 3: Post-process (NMS, coordinate mapping)
    // ────────────────────────────────────────────────────────
    const bubbles = await this.postProcess(predictions, blockBbox);

    // ────────────────────────────────────────────────────────
    // Step 4: Cleanup
    // ────────────────────────────────────────────────────────
    tensor.dispose();
    predictions.dispose();

    return bubbles;
  }

  private async postProcess(
    predictions: tf.Tensor,
    blockBbox: [number, number, number, number]
  ): Promise<BubbleDetection[]> {
    const [blockX1, blockY1] = blockBbox;
    const data = await predictions.data();

    const bubbles: BubbleDetection[] = [];

    // Parse YOLO output (format depends on model export)
    // Assuming: [x1, y1, x2, y2, confidence, classId]
    for (let i = 0; i < data.length; i += 6) {
      const confidence = data[i + 4];
      if (confidence < this.confidenceThreshold) continue;

      const [x1Crop, y1Crop, x2Crop, y2Crop] = [
        data[i], data[i + 1], data[i + 2], data[i + 3]
      ];
      const classId = Math.round(data[i + 5]);

      // Map to full image coordinates
      bubbles.push({
        classId,
        className: this.getClassName(classId),
        confidence,
        bboxXyxy: [
          blockX1 + x1Crop,
          blockY1 + y1Crop,
          blockX1 + x2Crop,
          blockY1 + y2Crop
        ],
        state: classId === 1 ? 'filled' : 'empty'
      });
    }

    return bubbles;
  }
}
```

### Web Worker Integration

```typescript
// main.ts
const worker = new Worker('ml-bubble-detector.worker.js');

worker.postMessage({
  type: 'detect',
  blockCrop: blockCropImageData,
  blockBbox: [100, 200, 400, 350]
});

worker.onmessage = (event) => {
  const { bubbles } = event.data;
  // Process results
};

// ml-bubble-detector.worker.js
import * as tf from '@tensorflow/tfjs';

let model: tf.GraphModel | null = null;

self.onmessage = async (event) => {
  const { type, blockCrop, blockBbox } = event.data;

  if (type === 'detect') {
    const bubbles = await detectBubbles(blockCrop, blockBbox);
    self.postMessage({ bubbles });
  }
};

async function detectBubbles(
  blockCrop: ImageData,
  blockBbox: number[]
): Promise<BubbleDetection[]> {
  // ... TensorFlow.js inference logic ...
}
```

---

## Performance Metrics

### Typical Values

**Per-block Processing**:
- Crop: < 5ms
- YOLO inference (CPU): 50-100ms
- YOLO inference (GPU): 10-20ms
- Coordinate mapping: < 1ms
- **Total per block**: 60-110ms (CPU), 20-30ms (GPU)

**Per-image Processing** (5 blocks):
- Total: 300-550ms (CPU), 100-150ms (GPU)

**Browser (TensorFlow.js)**:
- Model load: 1-2s (cached: < 100ms)
- Inference (WebGL): 30-50ms per block
- Inference (CPU): 200-300ms per block

---

## Error Handling

### Error Recovery Flow

```python
try:
    bubble_results = self.model.predict(block_crop, ...)
except Exception as e:
    logger.error(f"ML bubble detection failed for block: {e}")
    # Continue with next block (don't fail entire pipeline)
    continue
```

### Graceful Degradation

```python
# If model not loaded
if not self.model:
    logger.debug("ML bubble detector disabled")
    return context  # Skip ML detection, use traditional

# If no blocks detected
if not ml_blocks:
    logger.debug("No ML-detected blocks, skipping")
    return context

# If crop is empty
if block_crop is None or block_crop.size == 0:
    logger.warning("Empty crop, skipping block")
    continue  # Skip this block, continue with others
```

---

## Summary

The ML Bubble Detector flows include:

1. **Hierarchical Detection**: 2-stage pipeline (field blocks → bubbles)
2. **Image Cropping**: Extract field block regions with bounds checking
3. **YOLO Inference**: Detect bubbles with confidence filtering
4. **Coordinate Mapping**: Transform crop coords to full image coords
5. **Hybrid Fallback**: ML-based fallback for low-confidence traditional detections
6. **Browser Migration**: TensorFlow.js with Web Workers for performance

**Next Steps**: See `decisions.md`, `constraints.md`, `integration.md` for additional details.
