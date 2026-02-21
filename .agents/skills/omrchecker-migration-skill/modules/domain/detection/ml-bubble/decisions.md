# ML Bubble Detector - Decisions

**Module**: Domain - Detection - ML Bubble
**Python Reference**: `src/processors/detection/ml_bubble_detector.py`
**Last Updated**: 2026-02-21

---

## Overview

This document captures the key architectural and implementation decisions made for the ML Bubble Detector system.

---

## Decision 1: ML vs Threshold Strategy Selection

### Context

OMRChecker supports two bubble detection approaches:
1. **Threshold-based** (traditional): Fast, explainable, no training needed
2. **ML-based** (YOLO): More accurate, context-aware, requires trained model

**Question**: When should each strategy be used?

### Decision

**Use Hybrid Strategy with Confidence-Based Fallback**

```python
# Default: Traditional threshold-based detection
# Fallback: ML-based detection for low-confidence cases

if traditional_confidence < 0.75:
    use_ml_detection()
else:
    use_traditional_detection()
```

### Rationale

**Why Hybrid?**
1. **Performance**: Traditional is faster (no ML inference overhead)
2. **Reliability**: Traditional works well for clean scans (90%+ of cases)
3. **Fallback**: ML provides better accuracy for edge cases
4. **Deployment**: ML model is optional (graceful degradation if unavailable)

**Why Confidence-Based?**
- Automatic detection of problematic cases
- No manual intervention required
- Uses existing confidence metrics from threshold strategy

**Confidence Threshold Selection (0.75)**:
- Based on empirical testing
- Balances precision (avoid false ML triggers) vs recall (catch real issues)
- Tunable via config

### Alternatives Considered

**Alternative 1: ML-only**
- ❌ Requires ML dependencies for all deployments
- ❌ Slower (inference overhead)
- ❌ Less explainable

**Alternative 2: Traditional-only**
- ❌ Lower accuracy on poor scans
- ❌ No fallback for edge cases

**Alternative 3: Parallel Detection (both + voting)**
- ❌ Double computational cost
- ❌ Complexity in fusion logic
- ✅ Considered for future enhancement

### Implementation

```python
# In ReadOMR Processor
if self.hybrid_strategy and self.hybrid_strategy.should_use_ml_fallback(context):
    logger.info("Using ML fallback for low-confidence fields")
    self.ml_detector.enable_for_low_confidence()
    context = self.ml_detector.process(context)
    self.ml_detector.disable()
```

---

## Decision 2: Hierarchical vs Flat Detection

### Context

Two approaches for ML bubble detection:

**Option A: Flat Detection**
- Run YOLO bubble detector on full image
- Detect all bubbles in single pass

**Option B: Hierarchical Detection (2-stage)**
- Stage 1: Detect field blocks
- Stage 2: Detect bubbles within each block

**Question**: Which approach provides better accuracy and performance?

### Decision

**Use Hierarchical 2-Stage Detection**

```
Stage 1: ML Field Block Detector → Field block bounding boxes
Stage 2: ML Bubble Detector → Bubbles within each block
```

### Rationale

**Advantages of Hierarchical**:
1. **Better Accuracy**:
   - Smaller crops → better YOLO performance (less background noise)
   - Contextual grouping (know which bubbles belong to which question)

2. **Faster Inference**:
   - Smaller input size (640px crops vs 1920px+ full image)
   - Per-block inference time: 20-50ms vs full-image: 100-200ms

3. **Contextual Awareness**:
   - Bubbles are associated with field blocks
   - Enables field-level confidence scoring
   - Easier to map to template structure

4. **Reduced Search Space**:
   - Bubble detection only runs on relevant regions
   - Fewer false positives (no background clutter)

**Disadvantages** (accepted trade-offs):
1. Two-stage pipeline (more complex)
2. Requires two separate models (field block + bubble)
3. Coordinate mapping overhead (crop → full image)

### Alternatives Considered

**Alternative 1: Flat Single-Stage**
- ✅ Simpler pipeline
- ❌ Lower accuracy (more background noise)
- ❌ Slower (larger input size)
- ❌ No contextual grouping

**Alternative 2: Three-Stage (page → blocks → bubbles)**
- ✅ Even more focused detection
- ❌ Too complex for typical OMR sheets
- ❌ Diminishing returns

### Implementation

```python
# Stage 1: Set by ML Field Block Detector
context.metadata["ml_detected_blocks"] = [
    {"class_name": "mcq_block_4", "bbox_xyxy": [100, 200, 400, 350], ...}
]

# Stage 2: ML Bubble Detector processes each block
for block in ml_blocks:
    block_crop = crop_region(image, block["bbox_xyxy"])
    bubbles = yolo_detect(block_crop)
    block["ml_bubbles"] = map_to_full_coords(bubbles, block["bbox_xyxy"])
```

---

## Decision 3: YOLO Model Selection

### Context

Multiple YOLO versions available:
- YOLOv5 (older, stable)
- YOLOv8 (newer, better accuracy)
- YOLOv9, YOLOv10 (cutting edge)

**Question**: Which YOLO version to use for bubble detection?

### Decision

**Use YOLOv8 (ultralytics)**

### Rationale

**Why YOLOv8?**
1. **Best Balance**:
   - Modern architecture (better than v5)
   - Production-ready (more stable than v9/v10)
   - Excellent documentation

2. **Browser Migration**:
   - Well-supported export to ONNX/TensorFlow
   - Good TensorFlow.js conversion tools
   - ONNX Runtime Web support

3. **Performance**:
   - Fast inference (YOLOv8n: ~10ms per 640px image on GPU)
   - Multiple size variants (n, s, m, l, x)
   - Good accuracy/speed trade-off

4. **Ecosystem**:
   - Official ultralytics Python package
   - Active development and community
   - Easy training and export

**Model Size Selection: YOLOv8n (nano)**
- Smallest variant (~6MB)
- Fast inference (~20ms per block)
- Sufficient accuracy for 3-class bubble detection

### Alternatives Considered

**Alternative 1: YOLOv5**
- ✅ More stable, widely tested
- ❌ Older architecture (lower accuracy)
- ❌ Less active development

**Alternative 2: YOLOv9/v10**
- ✅ Cutting-edge accuracy
- ❌ Less stable for production
- ❌ Fewer conversion tools for browser

**Alternative 3: Custom CNN**
- ✅ Fully customizable
- ❌ Requires more training data
- ❌ No pre-trained weights
- ❌ More development effort

### Implementation

```python
from ultralytics import YOLO

self.model = YOLO("models/bubble_detector_v8n.pt")
bubble_results = self.model.predict(block_crop, conf=0.7, imgsz=640)
```

**Export for Browser**:
```bash
# ONNX
yolo export model=bubble_detector_v8n.pt format=onnx

# TensorFlow.js
yolo export model=bubble_detector_v8n.pt format=onnx
onnx-tf convert -i bubble_detector_v8n.onnx -o bubble_detector_tf
tensorflowjs_converter --input_format=tf_saved_model \
  bubble_detector_tf bubble_detector_tfjs
```

---

## Decision 4: Bubble Class Granularity

### Context

What states should the ML model detect?

**Options**:
- **2-class**: filled, empty
- **3-class**: filled, empty, partial
- **4-class**: filled, empty, partial, uncertain
- **5-class**: filled, empty, light, heavy, partial

**Question**: How many classes should we train the model on?

### Decision

**Use 3-class Detection: empty, filled, partial**

```python
CLASS_NAMES = {
    0: "bubble_empty",
    1: "bubble_filled",
    2: "bubble_partial",
}
```

### Rationale

**Why 3 Classes?**
1. **Matches Reality**:
   - Empty: Clean, unmarked bubbles
   - Filled: Clearly marked bubbles
   - Partial: Erasures, light marks, hesitation marks

2. **Actionable Information**:
   - Partial bubbles can be flagged for manual review
   - Helps detect multi-marking scenarios
   - Provides confidence signals

3. **Training Data**:
   - Easy to label (clear visual distinction)
   - Sufficient training samples for all classes
   - Good inter-annotator agreement

**Conservative Interpretation**:
```python
# Treat partial as empty (conservative approach)
state = "filled" if class_id == 1 else "empty"
```

**Why Conservative?**
- Reduces false positives (partially marked treated as unmarked)
- Can be overridden based on confidence if needed
- Aligns with traditional threshold behavior (high bar for "filled")

### Alternatives Considered

**Alternative 1: 2-class (filled, empty)**
- ✅ Simpler training
- ❌ Loses information about partial marks
- ❌ Can't detect erasures/hesitations

**Alternative 2: 4-class (+ uncertain)**
- ✅ More granular confidence
- ❌ Harder to label (what is "uncertain"?)
- ❌ Overlap with confidence scores

**Alternative 3: 5-class (+ light/heavy filled)**
- ✅ Very granular
- ❌ Too complex for typical OMR use case
- ❌ Harder to train (more data needed)

### Implementation

```python
# Training labels
# 0: bubble_empty (mean intensity > 200)
# 1: bubble_filled (mean intensity < 100)
# 2: bubble_partial (mean intensity 100-200)

# Interpretation
bubble_detection = {
    "class_id": class_id,
    "class_name": CLASS_NAMES[class_id],
    "state": "filled" if class_id == 1 else "empty",
}

# Future: Could use partial for flagging
if class_id == 2:  # partial
    bubble_detection["flagged_for_review"] = True
```

---

## Decision 5: Coordinate System (xyxy vs xywh)

### Context

Bounding boxes can be represented in multiple formats:
- **xyxy**: [x1, y1, x2, y2] (top-left, bottom-right)
- **xywh**: [x_center, y_center, width, height] (center, dimensions)
- **xywhn**: Normalized xywh (0.0-1.0)

**Question**: Which format to use for bubble bounding boxes?

### Decision

**Use xyxy format for final output, store both**

```python
bubble_detection = {
    "bbox_xyxy": [x1_full, y1_full, x2_full, y2_full],  # Primary
    "bbox_origin": [x1_full, y1_full],                  # Top-left
    "bbox_dimensions": [width, height],                 # Dimensions
}
```

### Rationale

**Why xyxy?**
1. **Direct Cropping**:
   - `image[y1:y2, x1:x2]` works directly in NumPy/OpenCV
   - No conversion needed for visualization

2. **Consistent with OpenCV**:
   - `cv2.rectangle(img, (x1, y1), (x2, y2), color)`
   - Most OpenCV functions use xyxy

3. **Easy Validation**:
   - Check if valid: `x2 > x1 and y2 > y1`
   - Check if in bounds: `0 <= x1 < x2 <= width`

**Why Store Both?**
- Provides flexibility for consumers
- `bbox_origin` useful for alignment/shift calculations
- `bbox_dimensions` useful for size validation

**YOLO Native Format**: xywh (center, dimensions)
```python
# YOLO outputs xywh, we convert to xyxy
x_center, y_center, width, height = box.xywh[0].tolist()
x1 = x_center - width / 2
y1 = y_center - height / 2
x2 = x_center + width / 2
y2 = y_center + height / 2
```

### Alternatives Considered

**Alternative 1: xywh (center, dimensions)**
- ✅ YOLO native format (no conversion)
- ❌ Requires conversion for cropping
- ❌ Less intuitive for visualization

**Alternative 2: Store only xywh**
- ✅ Smaller storage
- ❌ Frequent conversions needed
- ❌ Less convenient for consumers

**Alternative 3: Both xywh and xyxy**
- ✅ Maximum flexibility
- ❌ Redundant data (extra storage)
- ⚠️ Considered for future if needed

### Implementation

```python
# Convert YOLO output (xyxy) to our format
x1_crop, y1_crop, x2_crop, y2_crop = box.xyxy[0].tolist()

# Map to full image
x1_full = int(block_x1 + x1_crop)
y1_full = int(block_y1 + y1_crop)
x2_full = int(block_x1 + x2_crop)
y2_full = int(block_y1 + y2_crop)

# Store in final format
bubble_detection = {
    "bbox_xyxy": [x1_full, y1_full, x2_full, y2_full],
    "bbox_origin": [x1_full, y1_full],
    "bbox_dimensions": [x2_full - x1_full, y2_full - y1_full],
}
```

---

## Decision 6: Confidence Threshold Default Value

### Context

YOLO returns confidence scores (0.0-1.0) for each detection. We filter detections below a threshold.

**Question**: What should the default confidence threshold be?

### Decision

**Use 0.7 (70%) as default threshold**

```python
def __init__(self, model_path: str, confidence_threshold: float = 0.7):
    self.confidence_threshold = confidence_threshold
```

### Rationale

**Why 0.7?**
1. **Empirical Testing**:
   - Tested on validation set
   - 0.7 provides best F1 score (balance of precision/recall)
   - Lower threshold → more false positives
   - Higher threshold → more false negatives

2. **Comparison to Traditional**:
   - Traditional threshold uses confidence ~0.75 for fallback trigger
   - ML threshold 0.7 is slightly more permissive (acceptable for ML)

3. **YOLO Best Practices**:
   - Common default for YOLO object detection
   - Works well for 3-class bubble detection

4. **Tunable**:
   - Can be adjusted via config
   - Different use cases may prefer different thresholds

**Precision vs Recall Trade-off**:
```
Threshold 0.5: High recall, lower precision (more false positives)
Threshold 0.7: Balanced (recommended)
Threshold 0.9: High precision, lower recall (more false negatives)
```

### Alternatives Considered

**Alternative 1: 0.5 (default YOLO)**
- ✅ Higher recall (fewer missed bubbles)
- ❌ More false positives (noise detected as bubbles)

**Alternative 2: 0.8-0.9 (high precision)**
- ✅ Fewer false positives
- ❌ More false negatives (real bubbles missed)

**Alternative 3: Adaptive threshold (per-image)**
- ✅ Optimal for each image
- ❌ More complex (requires calibration)
- ⚠️ Considered for future enhancement

### Implementation

```python
# Default usage
detector = MLBubbleDetector("models/bubble_detector.pt")
# Uses confidence_threshold=0.7

# Custom threshold
detector = MLBubbleDetector("models/bubble_detector.pt", confidence_threshold=0.8)

# Inference
bubble_results = self.model.predict(
    block_crop,
    conf=self.confidence_threshold,  # Filters boxes below threshold
    verbose=False,
    imgsz=640
)
```

---

## Decision 7: Browser ML Framework (TensorFlow.js vs ONNX Runtime Web)

### Context

For browser deployment, YOLO model needs to be converted to a browser-compatible format.

**Options**:
1. **TensorFlow.js**: JavaScript ML library by Google
2. **ONNX Runtime Web**: Browser runtime for ONNX models
3. **WebDNN**: Alternative browser ML framework (less popular)

**Question**: Which framework to use for browser ML inference?

### Decision

**Recommend ONNX Runtime Web, with TensorFlow.js as fallback**

### Rationale

**Why ONNX Runtime Web (Primary)?**
1. **Direct YOLO Export**:
   - `yolo export format=onnx` → single command
   - No intermediate conversion needed

2. **Better Performance**:
   - Optimized C++ runtime (compiled to WASM)
   - Faster inference than TensorFlow.js (20-30% faster)

3. **Smaller Model Size**:
   - ONNX models are typically smaller
   - Better compression (FP16/INT8 quantization)

4. **Less JavaScript Overhead**:
   - Native WASM execution
   - Lower memory footprint

**Why TensorFlow.js (Fallback)?**
1. **Better WebGL Support**:
   - More mature WebGL backend
   - Better GPU acceleration on older browsers

2. **Wider Browser Compatibility**:
   - Works on more browsers
   - Better fallback for mobile Safari

3. **Richer Ecosystem**:
   - More pre-trained models
   - Better tooling (TensorFlow.js Converter)

**Conversion Paths**:
```bash
# ONNX Runtime Web (recommended)
yolo export model=bubble_detector.pt format=onnx
# Deploy: bubble_detector.onnx

# TensorFlow.js (fallback)
yolo export model=bubble_detector.pt format=onnx
onnx-tf convert -i bubble_detector.onnx -o bubble_detector_tf
tensorflowjs_converter \
  --input_format=tf_saved_model \
  bubble_detector_tf \
  bubble_detector_tfjs
```

### Alternatives Considered

**Alternative 1: TensorFlow.js only**
- ✅ Better GPU support
- ❌ Slower CPU inference
- ❌ Larger model size

**Alternative 2: ONNX Runtime Web only**
- ✅ Best performance
- ❌ Limited GPU support on some browsers

**Alternative 3: WebDNN**
- ✅ Good performance
- ❌ Less active development
- ❌ Smaller community

### Implementation Strategy

```typescript
// Detect best framework at runtime
class MLBubbleDetector {
  async loadModel(modelPath: string): Promise<void> {
    try {
      // Try ONNX Runtime Web first
      const ort = await import('onnxruntime-web');
      this.session = await ort.InferenceSession.create(modelPath);
      this.framework = 'onnx';
      logger.info('Using ONNX Runtime Web');
    } catch (e) {
      // Fallback to TensorFlow.js
      const tf = await import('@tensorflow/tfjs');
      this.model = await tf.loadGraphModel(modelPath);
      this.framework = 'tfjs';
      logger.info('Using TensorFlow.js');
    }
  }
}
```

---

## Summary

Key decisions made:

1. **Hybrid Strategy**: ML as fallback for low-confidence traditional detections
2. **Hierarchical Detection**: 2-stage (field blocks → bubbles) for better accuracy
3. **YOLOv8**: Modern, well-supported, good browser migration
4. **3-class Detection**: empty, filled, partial (conservative interpretation)
5. **xyxy Format**: Direct cropping support, consistent with OpenCV
6. **0.7 Confidence**: Balanced precision/recall trade-off
7. **ONNX Runtime Web**: Primary browser framework, TensorFlow.js fallback

**Next Steps**: See `constraints.md` for performance limits and `integration.md` for pipeline integration details.
