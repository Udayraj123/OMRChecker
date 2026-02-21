# ML Bubble Detector

**Module**: Domain - Detection - ML Bubble
**Python Reference**: `src/processors/detection/ml_bubble_detector.py`
**Status**: Stage 2 Hierarchical Detection
**Browser Migration**: TensorFlow.js / ONNX Runtime Web

---

## Overview

The ML Bubble Detector is a **YOLO-based machine learning system** for detecting individual bubbles within field blocks. It operates as **Stage 2 of hierarchical detection**, leveraging spatial context from Stage 1 field block detection for improved accuracy.

This detector serves two primary purposes:
1. **Hierarchical Detection**: Operates on cropped field blocks from ML Field Block Detector (Stage 1)
2. **Fallback Detection**: Provides ML-based fallback for low-confidence traditional threshold detections

**Key Principles**:
- **Context-aware**: Uses field block boundaries to focus bubble detection
- **Three-class detection**: Detects `bubble_empty`, `bubble_filled`, `bubble_partial`
- **Confidence-based**: Returns confidence scores for each detection
- **Coordinate mapping**: Maps detections from crop coordinates to full image coordinates
- **Optional/On-demand**: Only enabled when needed (not part of default pipeline)

---

## Architecture

### Hierarchical Detection Pipeline (2-Stage)

```
Stage 1: ML Field Block Detector
    └─> Detects field blocks (question groups) on full image
        └─> Outputs: Bounding boxes + confidence scores

Stage 2: ML Bubble Detector (this module)
    └─> For each field block:
        └─> Crop block region from full image
        └─> Run YOLO bubble detection on crop
        └─> Map bubble coordinates back to full image
        └─> Store results per block
```

**Benefits of 2-Stage Approach**:
1. **Reduced search space**: Bubble detection only runs on small crops
2. **Better accuracy**: Smaller images → better YOLO performance
3. **Contextual awareness**: Knows which bubbles belong to which field block
4. **Faster inference**: Smaller input size (640px vs 1920px+)

### Class Structure

```python
class MLBubbleDetector(Processor):
    """YOLO-based bubble detector (Stage 2)."""

    CLASS_NAMES: ClassVar[dict[int, str]] = {
        0: "bubble_empty",   # Unmarked bubble
        1: "bubble_filled",  # Marked bubble
        2: "bubble_partial", # Partially marked bubble
    }

    def __init__(self, model_path: str, confidence_threshold: float = 0.7):
        """Initialize with YOLO model path."""

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Run bubble detection on field blocks."""

    def _crop_block_region(self, image, bbox_xyxy: list) -> np.ndarray:
        """Crop field block region from full image."""

    def _map_to_full_coordinates(
        self, bubble_results, block_bbox_xyxy: list
    ) -> list[dict]:
        """Map bubble detections to full image coordinates."""
```

---

## Detection Classes

### Bubble States

The detector classifies each bubble into one of three states:

```python
CLASS_NAMES = {
    0: "bubble_empty",    # Clean, unmarked bubble
    1: "bubble_filled",   # Clearly marked bubble
    2: "bubble_partial",  # Partially marked (light marks, erasures)
}
```

**Training Data Characteristics**:
- **bubble_empty**: Mean intensity > 200 (white/light gray)
- **bubble_filled**: Mean intensity < 100 (dark/black)
- **bubble_partial**: Mean intensity 100-200 (gray)

**Interpretation Logic**:
```python
state = "filled" if class_id == 1 else "empty"
# Note: bubble_partial is treated as empty (conservative approach)
```

---

## Data Flow

### Input (from Stage 1)

```python
# Set by ML Field Block Detector (Stage 1)
ml_detected_blocks = context.metadata.get("ml_detected_blocks", [])

# Structure:
[
    {
        "class_id": 0,
        "class_name": "mcq_block_4",
        "confidence": 0.92,
        "bbox_xyxy": [100, 200, 400, 350],  # Field block bounding box
        "bbox_origin": [100, 200],
        "bbox_dimensions": [300, 150]
    },
    # ... more blocks
]
```

### Processing Flow

```python
for block_detection in ml_detected_blocks:
    # 1. Crop field block region
    block_crop = self._crop_block_region(
        context.gray_image,
        block_detection["bbox_xyxy"]
    )

    # 2. Run YOLO on cropped region
    bubble_results = self.model.predict(
        block_crop,
        conf=self.confidence_threshold,
        verbose=False,
        imgsz=640  # Smaller for cropped blocks
    )

    # 3. Map to full image coordinates
    bubbles = self._map_to_full_coordinates(
        bubble_results,
        block_detection["bbox_xyxy"]
    )

    # 4. Store per-block results
    block_detection["ml_bubbles"] = bubbles
    block_detection["ml_bubbles_count"] = len(bubbles)
```

### Output (stored in block metadata)

```python
# Added to each block_detection:
{
    # ... original block detection fields
    "ml_bubbles": [
        {
            "class_id": 1,
            "class_name": "bubble_filled",
            "confidence": 0.89,
            "bbox_xyxy": [150, 230, 175, 255],  # Full image coords
            "bbox_origin": [150, 230],
            "bbox_dimensions": [25, 25],
            "state": "filled"
        },
        {
            "class_id": 0,
            "class_name": "bubble_empty",
            "confidence": 0.93,
            "bbox_xyxy": [150, 260, 175, 285],
            "bbox_origin": [150, 260],
            "bbox_dimensions": [25, 25],
            "state": "empty"
        },
        # ... more bubbles
    ],
    "ml_bubbles_count": 4
}
```

---

## Coordinate Mapping

### Problem: Crop Coordinates → Full Image Coordinates

When YOLO runs on a cropped image, it returns coordinates relative to the crop origin (0,0). We need to map these back to full image coordinates.

### Algorithm

```python
def _map_to_full_coordinates(
    self, bubble_results, block_bbox_xyxy: list
) -> list[dict]:
    """Map bubble detections from crop to full image coordinates."""

    block_x1, block_y1, _, _ = block_bbox_xyxy
    bubbles = []

    for result in bubble_results:
        for box in result.boxes:
            # Get bounding box in xyxy format (crop-relative)
            x1_crop, y1_crop, x2_crop, y2_crop = box.xyxy[0].tolist()

            # Map to full image coordinates
            x1_full = int(block_x1 + x1_crop)
            y1_full = int(block_y1 + y1_crop)
            x2_full = int(block_x1 + x2_crop)
            y2_full = int(block_y1 + y2_crop)

            bubble_detection = {
                "class_id": int(box.cls[0]),
                "class_name": CLASS_NAMES.get(class_id, "unknown"),
                "confidence": float(box.conf[0]),
                "bbox_xyxy": [x1_full, y1_full, x2_full, y2_full],
                "bbox_origin": [x1_full, y1_full],
                "bbox_dimensions": [x2_full - x1_full, y2_full - y1_full],
                "state": "filled" if class_id == 1 else "empty"
            }

            bubbles.append(bubble_detection)

    return bubbles
```

**Example**:
```
Field block: bbox_xyxy = [100, 200, 400, 350]
YOLO detects bubble at crop coords: [50, 30, 75, 55]

Full image coords:
    x1_full = 100 + 50 = 150
    y1_full = 200 + 30 = 230
    x2_full = 100 + 75 = 175
    y2_full = 200 + 55 = 255

Result: [150, 230, 175, 255]
```

---

## YOLO Integration

### Model Loading

```python
def __init__(self, model_path: str, confidence_threshold: float = 0.7):
    """Initialize the bubble detector."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error(
            "ultralytics package not found. "
            "Install ML dependencies with: uv sync --extra ml"
        )
        self.model = None
        return

    self.model = YOLO(model_path) if Path(model_path).exists() else None
    self.confidence_threshold = confidence_threshold
```

**Lazy Loading**: Model is only loaded if file exists and ultralytics is installed.

**Graceful Degradation**: If model fails to load, detector is disabled (returns context unchanged).

### Inference

```python
bubble_results = self.model.predict(
    block_crop,
    conf=self.confidence_threshold,  # 0.7 default
    verbose=False,                    # Suppress YOLO logs
    imgsz=640,                        # Input size for YOLO
)
```

**Parameters**:
- `conf`: Confidence threshold (0.7 = 70%)
- `verbose`: Set to False to avoid cluttering logs
- `imgsz`: Input image size (640px is standard for YOLOv8)

### Model Format

**Expected**: YOLOv8 `.pt` file trained on bubble dataset

**Training Data Requirements**:
- **Classes**: 3 (bubble_empty, bubble_filled, bubble_partial)
- **Image size**: 640x640 (or similar)
- **Format**: YOLO format (normalized xywh)
- **Augmentations**: Rotation, blur, brightness (to simulate scan variations)

---

## Integration with Pipeline

### Context Flow

```
1. ML Field Block Detector (Stage 1)
    └─> Sets: context.metadata["ml_detected_blocks"]

2. ML Bubble Detector (Stage 2) [this module]
    └─> Reads: context.metadata["ml_detected_blocks"]
    └─> Adds: block["ml_bubbles"] and block["ml_bubbles_count"]
    └─> Returns: Updated context

3. Shift Detection Processor
    └─> Reads: block["ml_bubbles"]
    └─> Uses bubbles to calculate field shifts
```

### Processor Registration

```python
# In pipeline.py
from src.processors.detection.ml_bubble_detector import MLBubbleDetector

# Add to pipeline stages
ml_bubble_detector = MLBubbleDetector(
    model_path="models/bubble_detector.pt",
    confidence_threshold=0.7
)

stages.append(ml_bubble_detector)
```

---

## Fallback Detection (Hybrid Strategy)

### HybridDetectionStrategy Integration

The ML Bubble Detector can be used as a **fallback** for low-confidence traditional detections.

```python
# In ml_detector.py (separate file)
class HybridDetectionStrategy:
    """Hybrid detection combining traditional and ML methods."""

    def should_use_ml_fallback(self, context: ProcessingContext) -> bool:
        """Determine if ML fallback should be used."""
        field_id_to_interpretation = context.field_id_to_interpretation
        low_confidence = self.identify_low_confidence_fields(
            field_id_to_interpretation
        )

        if low_confidence:
            logger.info(
                f"Found {len(low_confidence)} low-confidence fields, "
                f"triggering ML fallback"
            )
            return True

        return False
```

**Usage in ReadOMR Processor**:
```python
# Check for low-confidence fields and use ML fallback if needed
if self.hybrid_strategy and self.hybrid_strategy.should_use_ml_fallback(context):
    logger.info("Using ML fallback for low-confidence fields")
    self.ml_detector.enable_for_low_confidence()
    context = self.ml_detector.process(context)
    self.ml_detector.disable()
    self.hybrid_strategy.stats["ml_fallback_used"] += 1
```

**Key Point**: ML detector is **disabled by default** and only enabled on-demand for flagged low-confidence cases.

---

## Memory Management

### Image Cropping

```python
def _crop_block_region(self, image, bbox_xyxy: list) -> np.ndarray:
    """Crop field block region from full image."""
    x1, y1, x2, y2 = bbox_xyxy

    # Ensure coordinates are within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    # Crop and return (creates a view, not a copy)
    return image[y1:y2, x1:x2]
```

**Bounds Checking**: Prevents out-of-bounds errors if field block extends beyond image.

**NumPy Views**: `image[y1:y2, x1:x2]` creates a view, not a copy (memory efficient).

### Error Handling

```python
if block_crop is None or block_crop.size == 0:
    logger.warning(
        f"Empty crop for block {block_detection['class_name']}, skipping"
    )
    continue
```

**Defensive Programming**: Skip empty crops to avoid YOLO errors.

---

## Performance Characteristics

### Time Complexity

- **Per-block crop**: O(1) - NumPy view operation
- **YOLO inference**: O(w × h) where w×h is crop size (typically 300×150)
- **Coordinate mapping**: O(n) where n = number of detected bubbles
- **Total per image**: O(b × (w × h + n)) where b = number of blocks

**Typical Values**:
- Blocks per image: 5-15
- Crop size: 300×150 pixels
- Bubbles per block: 4-10
- **Inference time**: 50-100ms per block on CPU, 10-20ms on GPU

### Space Complexity

- **Cropped images**: O(b × w × h) - temporary, released after inference
- **Detection results**: O(b × n) where n = bubbles per block
- **YOLO model**: ~50MB for YOLOv8n, ~100MB for YOLOv8s

### Optimization Strategies

1. **Batch inference**: Process multiple crops in single YOLO call
2. **GPU acceleration**: Use CUDA for faster inference
3. **Model quantization**: Reduce model size with INT8 quantization
4. **Early exit**: Skip ML detection if Stage 1 has no blocks

---

## Browser Migration

### Challenge: YOLO in Browser

**Python YOLO** (ultralytics) is not directly available in browser. We need to convert the model.

### Migration Options

#### Option 1: TensorFlow.js (Recommended)

```typescript
import * as tf from '@tensorflow/tfjs';

class MLBubbleDetector {
  private model: tf.GraphModel | null = null;
  private confidenceThreshold: number = 0.7;

  async loadModel(modelUrl: string): Promise<void> {
    // Load converted TensorFlow.js model
    this.model = await tf.loadGraphModel(modelUrl);
  }

  async detectBubbles(
    blockCrop: ImageData,
    blockBbox: [number, number, number, number]
  ): Promise<BubbleDetection[]> {
    if (!this.model) return [];

    // 1. Preprocess image
    const tensor = tf.browser.fromPixels(blockCrop)
      .resizeBilinear([640, 640])
      .div(255.0)
      .expandDims(0);

    // 2. Run inference
    const predictions = await this.model.predict(tensor) as tf.Tensor;

    // 3. Post-process (NMS, coordinate mapping)
    const bubbles = this.postProcess(predictions, blockBbox);

    // 4. Cleanup
    tensor.dispose();
    predictions.dispose();

    return bubbles;
  }

  private postProcess(
    predictions: tf.Tensor,
    blockBbox: [number, number, number, number]
  ): BubbleDetection[] {
    // Extract boxes, scores, classes from YOLO output
    const data = predictions.dataSync();
    const bubbles: BubbleDetection[] = [];

    const [blockX1, blockY1] = blockBbox;

    // Parse YOLO output format (depends on conversion)
    // Apply NMS, filter by confidence, map coordinates
    for (let i = 0; i < data.length; i += 6) {
      const confidence = data[i + 4];
      if (confidence < this.confidenceThreshold) continue;

      const classId = Math.round(data[i + 5]);
      const [x1Crop, y1Crop, x2Crop, y2Crop] = [
        data[i], data[i + 1], data[i + 2], data[i + 3]
      ];

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

**Conversion Process**:
```bash
# 1. Export YOLO to ONNX
yolo export model=bubble_detector.pt format=onnx

# 2. Convert ONNX to TensorFlow.js
pip install onnx-tf tensorflowjs
onnx-tf convert -i bubble_detector.onnx -o bubble_detector_tf
tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_format=tfjs_graph_model \
  bubble_detector_tf \
  bubble_detector_tfjs
```

#### Option 2: ONNX Runtime Web

```typescript
import * as ort from 'onnxruntime-web';

class MLBubbleDetector {
  private session: ort.InferenceSession | null = null;

  async loadModel(modelPath: string): Promise<void> {
    this.session = await ort.InferenceSession.create(modelPath);
  }

  async detectBubbles(
    blockCrop: ImageData,
    blockBbox: [number, number, number, number]
  ): Promise<BubbleDetection[]> {
    if (!this.session) return [];

    // 1. Preprocess to tensor
    const inputTensor = this.preprocessImage(blockCrop);

    // 2. Run inference
    const feeds = { images: inputTensor };
    const results = await this.session.run(feeds);

    // 3. Post-process
    const bubbles = this.postProcess(results.output0, blockBbox);

    return bubbles;
  }

  private preprocessImage(image: ImageData): ort.Tensor {
    // Convert ImageData to Float32Array
    const { width, height, data } = image;
    const pixels = new Float32Array(3 * 640 * 640);

    // Resize and normalize (simplified)
    // ... resize logic ...

    // Create tensor [1, 3, 640, 640]
    return new ort.Tensor('float32', pixels, [1, 3, 640, 640]);
  }
}
```

**Advantages of ONNX**:
- Direct YOLO export (no intermediate conversion)
- Better performance (optimized runtime)
- Smaller model size

**Conversion**:
```bash
yolo export model=bubble_detector.pt format=onnx
```

### Performance Considerations

**Browser Constraints**:
- **Model Size**: Keep < 10MB (quantize to INT8 or FP16)
- **Inference Time**: Target < 200ms per block on modern hardware
- **Memory**: Browser has limited RAM (use Web Workers)

**Optimization Techniques**:
1. **Model Quantization**: Reduce precision (FP32 → FP16 or INT8)
2. **WebGL Backend**: Use GPU acceleration (TensorFlow.js WebGL)
3. **Web Workers**: Run inference off main thread
4. **Caching**: Cache model in browser (IndexedDB)

---

## Edge Cases & Error Handling

### 1. Model Not Found

```python
if not Path(model_path).exists():
    logger.warning(
        f"ML model not found at {model_path}, ML fallback disabled"
    )
    return
```

**Behavior**: Detector is disabled, returns context unchanged.

### 2. Ultralytics Not Installed

```python
try:
    from ultralytics import YOLO
except ImportError:
    logger.error(
        "ultralytics package not found. "
        "Install ML dependencies with: uv sync --extra ml"
    )
    self.model = None
```

**Behavior**: Graceful degradation, detector disabled.

### 3. No ML Detected Blocks

```python
ml_blocks = context.metadata.get("ml_detected_blocks", [])

if not ml_blocks:
    logger.debug("No ML-detected blocks found, skipping ML bubble detection")
    return context
```

**Behavior**: Skip processing, return unchanged context.

### 4. Empty Crop

```python
if block_crop is None or block_crop.size == 0:
    logger.warning(
        f"Empty crop for block {block_detection['class_name']}, skipping"
    )
    continue
```

**Behavior**: Skip this block, continue with next block.

### 5. YOLO Inference Error

```python
try:
    bubble_results = self.model.predict(...)
except Exception as e:
    logger.error(f"ML bubble detection failed for block: {e}")
    continue
```

**Behavior**: Log error, skip block, continue processing.

---

## Testing Strategy

### Unit Tests

```python
def test_coordinate_mapping():
    """Test crop to full image coordinate mapping."""
    detector = MLBubbleDetector("models/test.pt")

    block_bbox = [100, 200, 400, 350]
    mock_box = MockYOLOBox(
        xyxy=[50, 30, 75, 55],  # Crop coords
        cls=1,  # bubble_filled
        conf=0.89
    )

    bubbles = detector._map_to_full_coordinates([mock_box], block_bbox)

    assert bubbles[0]["bbox_xyxy"] == [150, 230, 175, 255]
    assert bubbles[0]["class_name"] == "bubble_filled"
    assert bubbles[0]["state"] == "filled"
```

### Integration Tests

```python
def test_ml_bubble_detector_pipeline():
    """Test ML bubble detector in pipeline context."""
    context = ProcessingContext(
        gray_image=test_image,
        metadata={
            "ml_detected_blocks": [
                {
                    "class_name": "mcq_block_4",
                    "bbox_xyxy": [100, 200, 400, 350],
                    "confidence": 0.92
                }
            ]
        }
    )

    detector = MLBubbleDetector("models/bubble_detector.pt")
    result_context = detector.process(context)

    blocks = result_context.metadata["ml_detected_blocks"]
    assert "ml_bubbles" in blocks[0]
    assert blocks[0]["ml_bubbles_count"] > 0
```

---

## Related Components

- **ML Field Block Detector**: Stage 1 hierarchical detection
- **Shift Detection Processor**: Uses bubble positions to calculate shifts
- **ReadOMR Processor**: Orchestrates ML fallback logic
- **HybridDetectionStrategy**: Manages confidence-based fallback
- **Detection Fusion**: Merges ML and traditional results

---

## Configuration

### Model Path

```python
# Default location
model_path = "models/bubble_detector.pt"

# Environment variable
import os
model_path = os.getenv("ML_BUBBLE_MODEL_PATH", "models/bubble_detector.pt")
```

### Confidence Threshold

```python
# Default: 0.7 (70%)
detector = MLBubbleDetector(
    model_path="models/bubble_detector.pt",
    confidence_threshold=0.7  # Adjust for precision/recall tradeoff
)
```

**Tuning Guidelines**:
- **High precision** (fewer false positives): 0.8-0.9
- **Balanced**: 0.7
- **High recall** (fewer false negatives): 0.5-0.6

---

## Future Enhancements

1. **Batch Inference**: Process multiple crops in single YOLO call
2. **Model Ensemble**: Combine multiple models for better accuracy
3. **Active Learning**: Collect misclassified samples for retraining
4. **Multi-scale Detection**: Run detection at multiple crop sizes
5. **Attention Mechanism**: Focus on high-uncertainty regions

---

## Summary

The ML Bubble Detector provides:

- **YOLO-based bubble detection** within field blocks (Stage 2)
- **Three-class detection**: empty, filled, partial bubbles
- **Hierarchical context**: Operates on crops from Stage 1 field blocks
- **Coordinate mapping**: Maps crop coordinates to full image
- **Fallback detection**: Optional ML-based fallback for low-confidence cases
- **Browser migration**: TensorFlow.js or ONNX Runtime Web
- **Graceful degradation**: Disabled if model/dependencies unavailable

**Next Steps**: See `flows.md`, `decisions.md`, `constraints.md`, `integration.md` for detailed documentation.
