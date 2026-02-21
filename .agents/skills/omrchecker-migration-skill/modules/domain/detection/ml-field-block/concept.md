# ML Field Block Detector - Core Concepts

## Overview

The **ML Field Block Detector** is a machine learning-based processor that detects field blocks (groups of related fields like MCQ sections, OCR zones, or barcode areas) using YOLO (You Only Look Once) object detection. It serves as **Stage 1** in a hierarchical two-stage detection pipeline, providing spatial context for downstream bubble detection and alignment refinement.

## What is ML Field Block Detection?

ML Field Block Detection is the process of automatically identifying and localizing field blocks on OMR sheets using deep learning, eliminating the need for manual template definition or traditional computer vision techniques.

**Key Capabilities**:
- Detects field blocks with bounding boxes and confidence scores
- Classifies field block types (MCQ, OCR, Barcode)
- Computes alignment adjustments by comparing ML detections with template expectations
- Optional STN (Spatial Transformer Network) preprocessing for improved robustness

## Architecture

### Entry Point

**File**: `src/processors/detection/ml_field_block_detector.py`

```python
class MLFieldBlockDetector(Processor):
    """YOLO-based field block detector (Stage 1).

    Detects field blocks and their approximate positions,
    providing spatial context for Stage 2 bubble detection.
    """
```

### Two-Stage Hierarchical Detection

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: ML Field Block Detector                           │
│ - Detects field blocks (MCQ sections, OCR zones, barcodes) │
│ - Provides bounding boxes and confidence scores            │
│ - Computes alignment adjustments                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Bubble/OCR/Barcode Detection                      │
│ - Uses field block spatial context                         │
│ - Refines alignment within detected blocks                 │
│ - Extracts actual values (bubble darkness, text, codes)    │
└─────────────────────────────────────────────────────────────┘
```

**Why Hierarchical?**
- **Robustness**: Coarse-to-fine approach handles large variations
- **Efficiency**: Focus computational resources on detected regions
- **Accuracy**: Spatial context improves downstream detection

### Components

```
MLFieldBlockDetector
├── YOLO Model (Ultralytics)
│   ├── Input: 1024x1024 grayscale/color image
│   ├── Output: Bounding boxes, class IDs, confidence scores
│   └── Classes: field_block_mcq, field_block_ocr, field_block_barcode
│
├── Optional: STN (Spatial Transformer Network)
│   ├── Input: Grayscale image
│   ├── Learns: Affine transformation (rotation, translation, scale, shear)
│   └── Output: Aligned image (improves YOLO accuracy)
│
└── Alignment Adjustment Calculator
    ├── Input: ML detections + Template field blocks
    ├── Matching: Euclidean distance (< 200px threshold)
    └── Output: Per-block shift adjustments [dx, dy]
```

## Field Block Classes

The detector recognizes three field block types:

| Class ID | Class Name | Description | Example Use Case |
|----------|-----------|-------------|------------------|
| 0 | `field_block_mcq` | Multiple-choice question sections | Answer grids with bubbles |
| 1 | `field_block_ocr` | Optical character recognition zones | Student name, roll number |
| 2 | `field_block_barcode` | Barcode/QR code regions | Unique sheet identifiers |

**Training Data Format**:
```yaml
# YOLO format: class_id x_center y_center width height (normalized 0-1)
0 0.5 0.3 0.4 0.2  # MCQ block at center-left
1 0.8 0.1 0.15 0.1  # OCR block at top-right
2 0.5 0.9 0.1 0.05  # Barcode at bottom-center
```

## Configuration Schema

### Initialization Parameters

```python
MLFieldBlockDetector(
    model_path: str,                    # Path to trained YOLO model (.pt file)
    confidence_threshold: float = 0.7,  # Minimum detection confidence
    use_stn: bool = False,              # Enable STN preprocessing
    stn_model_path: str | None = None,  # Path to STN model (required if use_stn=True)
)
```

### Example Configuration

**Minimal (YOLO only)**:
```python
detector = MLFieldBlockDetector(
    model_path="models/field_block_yolo.pt",
    confidence_threshold=0.7
)
```

**With STN (Alignment Refinement)**:
```python
detector = MLFieldBlockDetector(
    model_path="models/field_block_yolo.pt",
    confidence_threshold=0.7,
    use_stn=True,
    stn_model_path="models/stn_alignment.pt"
)
```

## Spatial Transformer Network (STN)

### What is STN?

STN is a learnable module that predicts and applies geometric transformations to correct residual misalignments BEFORE YOLO detection.

**Reference**: Jaderberg et al., "Spatial Transformer Networks" (2015)
https://arxiv.org/abs/1506.02025

### STN Architecture

```
Input Image (1024x1024)
    ↓
Localization Network (Lightweight CNN)
    ├── Conv2d(1→8, 7x7, stride=2) + BN + ReLU + MaxPool
    ├── Conv2d(8→16, 5x5, stride=2) + BN + ReLU + MaxPool
    ├── Conv2d(16→32, 3x3, stride=1) + BN + ReLU
    └── AdaptiveAvgPool2d(4x4)
    ↓
FC Layers (Regression)
    ├── Linear(32*4*4 → 64) + ReLU + Dropout(0.1)
    └── Linear(64 → 6)  # Predict 6 affine parameters
    ↓
Affine Matrix (2x3)
    [[θ11, θ12, θ13],   # [scale_x*cos(θ), -sin(θ), tx]
     [θ21, θ22, θ23]]   # [sin(θ), scale_y*cos(θ), ty]
    ↓
Grid Sampler (Bilinear Interpolation)
    ↓
Aligned Image (1024x1024)
    ↓
YOLO Detection
```

### STN Variants

**File**: `src/processors/detection/models/stn_module.py`

| Variant | Parameters | Use Case | Performance |
|---------|-----------|----------|-------------|
| **SpatialTransformerNetwork** | 6 (full affine) | Rotation, translation, scale, shear | ~10K params, slower |
| **TranslationOnlySTN** | 2 (tx, ty) | Simple shifts only | ~8K params, faster |
| **STNWithRegularization** | 6 + regularization | Prevent extreme transformations | More stable training |
| **TranslationOnlySTNWithRegularization** | 2 + regularization | Constrained shifts | Fastest + stable |

### When to Use STN?

**Enable STN (`use_stn=True`) if**:
- Mobile camera photos (rotation, perspective distortion)
- Low-quality scans (skew, warping)
- Bent or curled sheets
- Inconsistent scanning setup

**Skip STN (`use_stn=False`) if**:
- High-quality flatbed scans
- Consistent scanning environment
- Already preprocessed images (AutoRotate, CropOnMarkers applied)
- Inference speed critical (STN adds ~50-100ms overhead)

## Data Flow

### Input

```python
context.gray_image           # Preprocessed scanned image (H, W)
context.template.field_blocks  # Template-defined field blocks (for alignment)
```

### Processing Steps

```python
# 1. Optional: Apply STN transformation
if use_stn and stn:
    image = stn(gray_image)  # Correct residual misalignments

# 2. Run YOLO inference
results = yolo.predict(
    image,
    conf=confidence_threshold,  # Default: 0.7
    imgsz=1024                  # Larger size for full OMR sheet
)

# 3. Parse detections
detected_blocks = [
    {
        "class_id": 0,
        "class_name": "field_block_mcq",
        "confidence": 0.92,
        "bbox_origin": [100, 200],       # [x, y]
        "bbox_dimensions": [300, 400],   # [width, height]
        "bbox_xyxy": [100, 200, 400, 600]
    },
    # ... more detections
]

# 4. Compare with template to compute alignment adjustments
alignment_adjustments = {
    "Q1-30": {
        "matched_ml_block": detected_blocks[0],
        "shift": [5, -3],      # ML detected at [105, 197], template expects [100, 200]
        "distance": 5.83,      # Euclidean distance
        "confidence": 0.92
    }
}
```

### Output

```python
context.metadata["ml_detected_blocks"] = detected_blocks
context.metadata["ml_block_alignments"] = alignment_adjustments
```

**Downstream Usage**:
- **Shift Detection Processor**: Uses `ml_block_alignments` to refine template coordinates
- **Bubble Detection**: Operates within detected field block regions
- **Visualization**: Draws detected bounding boxes for debugging

## Alignment Adjustment Algorithm

### Matching Strategy

**Goal**: Match ML-detected blocks with template-defined blocks to compute shift adjustments.

**Algorithm**: Spatial proximity matching with Euclidean distance

```python
def compute_alignment_adjustments(ml_blocks, template_blocks):
    adjustments = {}

    for template_block in template_blocks:
        template_center = bbox_center(template_block.origin, template_block.dimensions)

        # Find closest ML detection
        best_match = None
        best_distance = float("inf")

        for ml_block in ml_blocks:
            ml_center = bbox_center(ml_block["bbox_origin"], ml_block["bbox_dimensions"])
            distance = euclidean_distance(template_center, ml_center)

            if distance < best_distance:
                best_distance = distance
                best_match = ml_block

        # Accept match if within threshold
        if best_match and best_distance < 200:  # 200 pixels max
            shift = [
                ml_block["bbox_origin"][0] - template_block.origin[0],  # dx
                ml_block["bbox_origin"][1] - template_block.origin[1]   # dy
            ]

            adjustments[template_block.name] = {
                "matched_ml_block": best_match,
                "shift": shift,
                "distance": best_distance,
                "confidence": best_match["confidence"]
            }

    return adjustments
```

### Distance Threshold

**Why 200 pixels?**
- Typical field block: 300-500 pixels wide
- 200px = ~40-65% of block width (reasonable shift tolerance)
- Larger threshold risks false matches (wrong block)
- Smaller threshold may miss valid matches on poorly aligned scans

**Trade-off**:
- **Higher threshold** (300px): More matches, but risk pairing wrong blocks
- **Lower threshold** (100px): Fewer false positives, but may miss valid matches
- **Adaptive** (future): Scale threshold based on field block size

### Example Matching

```
Template Field Blocks:
  Q1-30:  origin=[100, 200], dimensions=[300, 400]  → center=[250, 400]
  Q31-60: origin=[450, 200], dimensions=[300, 400]  → center=[600, 400]

ML Detections:
  Block A: bbox=[105, 197, 405, 597]  → center=[255, 397]
  Block B: bbox=[448, 203, 748, 603]  → center=[598, 403]

Matching:
  Q1-30 ↔ Block A:
    distance = √((255-250)² + (397-400)²) = √34 ≈ 5.83 pixels ✓
    shift = [105-100, 197-200] = [5, -3]

  Q31-60 ↔ Block B:
    distance = √((598-600)² + (403-400)²) = √13 ≈ 3.61 pixels ✓
    shift = [448-450, 203-200] = [-2, 3]
```

## Key Design Decisions

### Decision 1: YOLO for Field Block Detection

**Choice**: Ultralytics YOLO (YOLOv8)

**Rationale**:
- **Fast**: Real-time inference (~50-200ms per image)
- **Accurate**: State-of-the-art object detection
- **Robust**: Handles scale, rotation, partial occlusion
- **Easy to train**: Transfer learning from COCO pretrained models

**Alternatives Considered**:
| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Template matching | No training needed | Fails on rotation/distortion | ❌ Rejected |
| Edge detection + contours | Fast, simple | Unreliable (noise, shadows) | ❌ Rejected |
| **YOLO** | **Fast + accurate** | **Requires training data** | ✅ **Selected** |
| Faster R-CNN | Higher accuracy | Slower (~500ms) | ❌ Slower |

### Decision 2: Confidence Threshold = 0.7

**Choice**: Default confidence threshold of 0.7

**Rationale**:
- **0.5-0.6**: Too low, many false positives (background noise detected as blocks)
- **0.7-0.8**: Good balance (recommended for production)
- **0.9+**: Very strict, may miss valid blocks on low-quality scans

**Recommended Values**:
- **High-quality scans**: 0.8-0.9 (strict filtering)
- **Mobile camera photos**: 0.6-0.7 (more lenient)
- **Mixed quality**: 0.7 (default)

### Decision 3: Image Size = 1024x1024

**Choice**: YOLO inference at 1024x1024 resolution

**Rationale**:
- Standard OMR sheet: A4 at 200 DPI ≈ 1654×2339 pixels
- Downsampling to 1024×1024 maintains sufficient detail for field blocks
- YOLO models typically trained on 640×640 or 1024×1024
- Larger input = better accuracy but slower inference

**Performance Impact**:
| Input Size | Inference Time | Accuracy | Memory Usage |
|------------|----------------|----------|--------------|
| 640×640 | ~30ms | Good | ~500 MB |
| **1024×1024** | **~50-100ms** | **Better** | **~800 MB** |
| 1280×1280 | ~150ms | Best | ~1.2 GB |

### Decision 4: Optional STN Preprocessing

**Choice**: STN is optional (`use_stn=False` by default)

**Rationale**:
- **Not always needed**: Well-preprocessed images don't benefit
- **Overhead**: Adds ~50-100ms latency
- **Trade-off**: Accuracy improvement vs. speed
- **User choice**: Enable for challenging scenarios

**When STN Helps**:
- Mobile camera scans (rotation, perspective)
- Misaligned scans (translation, skew)
- Bent/curled sheets (local distortions)

**When STN Doesn't Help**:
- Perfect scans (already aligned)
- Heavy preprocessing applied (AutoRotate, CropOnMarkers)
- Speed-critical applications

## Dependencies

### ML Framework

**Ultralytics YOLO**:
```python
from ultralytics import YOLO

model = YOLO(model_path)
results = model.predict(image, conf=0.7, imgsz=1024)
```

**Installation**:
```bash
uv sync --extra ml  # Installs ultralytics, torch, torchvision
```

**Browser Alternative**: ONNX Runtime Web (see `constraints.md`)

### STN Framework

**PyTorch**:
```python
import torch
import torch.nn.functional as F

# STN components
stn = SpatialTransformerNetwork(input_channels=1, input_size=(1024, 1024))
aligned_image = stn(image_tensor)
```

**Browser Alternative**: TensorFlow.js or ONNX Runtime Web

### Geometry Utilities

**File**: `src/utils/geometry.py`

```python
from src.utils.geometry import bbox_center, euclidean_distance

center = bbox_center(origin, dimensions)  # [x + w/2, y + h/2]
dist = euclidean_distance(point1, point2)  # √((x2-x1)² + (y2-y1)²)
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Typical Time |
|-----------|-----------|--------------|
| STN forward pass | O(HW) | ~50ms (1024×1024) |
| YOLO inference | O(HW) | ~100ms (1024×1024) |
| Bounding box parsing | O(N) | <1ms (N ≈ 5-20 detections) |
| Alignment matching | O(M×N) | <1ms (M ≈ 5 template blocks, N ≈ 5 ML blocks) |
| **Total** | **O(HW)** | **~150ms with STN, ~100ms without** |

### Memory Usage

```
YOLO Model:
  - Weights: ~6 MB (YOLOv8n) to ~100 MB (YOLOv8x)
  - Inference: ~800 MB GPU memory

STN Model:
  - Weights: <1 MB (~10K parameters)
  - Inference: ~50 MB

Per-Image Processing:
  - Input image: 1024×1024×1 byte = 1 MB
  - Detections: ~20 blocks × 100 bytes = 2 KB
  - Total: ~2-3 MB per image
```

### Optimization Strategies

1. **Model Quantization**: INT8 quantization reduces model size by 4× (6 MB → 1.5 MB)
2. **Batch Processing**: Process multiple images in single YOLO call (GPU utilization)
3. **Early Exit**: Skip STN if confidence already high
4. **Caching**: Cache model in memory (avoid reload overhead)

## Browser Migration Notes

### Critical Challenges

#### 1. YOLO Model Format

**Python**: Ultralytics `.pt` format (PyTorch)

**Browser Options**:

| Format | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| **ONNX Runtime Web** | Fast, well-supported | Requires ONNX export | ✅ **Recommended** |
| TensorFlow.js | Native browser support | Slower than ONNX | ⚠️ Fallback |
| Custom WASM | Maximum speed | Complex implementation | ❌ Too complex |

**Export to ONNX**:
```python
from ultralytics import YOLO

model = YOLO("field_block_yolo.pt")
model.export(format="onnx")  # Creates field_block_yolo.onnx
```

**Load in Browser**:
```javascript
import * as ort from 'onnxruntime-web';

const session = await ort.InferenceSession.create('field_block_yolo.onnx');
const input = new ort.Tensor('float32', imageData, [1, 1, 1024, 1024]);
const outputs = await session.run({ images: input });
```

#### 2. STN Model Migration

**PyTorch → Browser**:

**Option A: ONNX**:
```python
# Export STN to ONNX
torch.onnx.export(
    stn_model,
    dummy_input,
    "stn_alignment.onnx",
    input_names=['input'],
    output_names=['output']
)
```

**Option B: TensorFlow.js**:
```bash
# Convert PyTorch → ONNX → TensorFlow → TensorFlow.js
pip install onnx-tf
onnx-tf convert -i stn_alignment.onnx -o stn_tf/
tensorflowjs_converter --input_format=tf_saved_model stn_tf/ stn_tfjs/
```

#### 3. NumPy → TypedArrays

**Python**:
```python
detected_blocks.sort(key=lambda b: (b["bbox_origin"][1], b["bbox_origin"][0]))
```

**JavaScript**:
```javascript
detectedBlocks.sort((a, b) => {
  if (a.bbox_origin[1] !== b.bbox_origin[1]) {
    return a.bbox_origin[1] - b.bbox_origin[1];  // Sort by y
  }
  return a.bbox_origin[0] - b.bbox_origin[0];    // Then by x
});
```

#### 4. Model Loading

**Challenge**: Large model files (6-100 MB)

**Solutions**:
1. **Lazy Loading**: Load model on demand, show progress
2. **IndexedDB Caching**: Cache downloaded model locally
3. **CDN Hosting**: Serve models from CDN for faster download
4. **Model Compression**: Use quantized models (INT8)

```javascript
// Cache model in IndexedDB
async function loadModel(modelUrl) {
  const cache = await caches.open('ml-models');
  let response = await cache.match(modelUrl);

  if (!response) {
    console.log('Downloading model...');
    response = await fetch(modelUrl);
    await cache.put(modelUrl, response.clone());
  }

  const modelBuffer = await response.arrayBuffer();
  return await ort.InferenceSession.create(modelBuffer);
}
```

### Browser Implementation Example

```javascript
class MLFieldBlockDetector {
  constructor(modelPath, confidenceThreshold = 0.7) {
    this.modelPath = modelPath;
    this.confidenceThreshold = confidenceThreshold;
    this.session = null;
  }

  async initialize() {
    this.session = await ort.InferenceSession.create(this.modelPath);
  }

  async process(context) {
    if (!this.session) await this.initialize();

    // Prepare input tensor
    const inputTensor = this.prepareInput(context.grayImage);

    // Run inference
    const outputs = await this.session.run({ images: inputTensor });

    // Parse detections
    const detectedBlocks = this.parseDetections(outputs, context.grayImage.shape);

    // Compute alignment adjustments
    const alignmentAdjustments = this.computeAlignmentAdjustments(
      detectedBlocks,
      context.template.fieldBlocks
    );

    // Store in metadata
    context.metadata.ml_detected_blocks = detectedBlocks;
    context.metadata.ml_block_alignments = alignmentAdjustments;

    return context;
  }

  prepareInput(image) {
    // Resize to 1024x1024, normalize, convert to tensor
    const resized = cv.resize(image, new cv.Size(1024, 1024));
    const normalized = new Float32Array(1024 * 1024);

    for (let i = 0; i < resized.data.length; i++) {
      normalized[i] = resized.data[i] / 255.0;
    }

    return new ort.Tensor('float32', normalized, [1, 1, 1024, 1024]);
  }

  parseDetections(outputs, imageShape) {
    // Parse YOLO output format
    const boxes = outputs.boxes.data;
    const scores = outputs.scores.data;
    const classes = outputs.classes.data;

    const detections = [];

    for (let i = 0; i < scores.length; i++) {
      if (scores[i] >= this.confidenceThreshold) {
        detections.push({
          class_id: classes[i],
          class_name: this.CLASS_NAMES[classes[i]],
          confidence: scores[i],
          bbox_origin: [boxes[i * 4], boxes[i * 4 + 1]],
          bbox_dimensions: [
            boxes[i * 4 + 2] - boxes[i * 4],
            boxes[i * 4 + 3] - boxes[i * 4 + 1]
          ],
          bbox_xyxy: [
            boxes[i * 4], boxes[i * 4 + 1],
            boxes[i * 4 + 2], boxes[i * 4 + 3]
          ]
        });
      }
    }

    // Sort by position (top-to-bottom, left-to-right)
    return detections.sort((a, b) => {
      if (a.bbox_origin[1] !== b.bbox_origin[1]) {
        return a.bbox_origin[1] - b.bbox_origin[1];
      }
      return a.bbox_origin[0] - b.bbox_origin[0];
    });
  }
}
```

## Related Modules

- **ML Bubble Detector** (`ml-bubble/`) - Stage 2 bubble detection with YOLO
- **Shift Detection** (`shift-detection/`) - Uses ML block alignments for coordinate refinement
- **STN Module** (`stn/`) - Spatial transformer network details
- **Bubble Detection** (`bubbles-threshold/`) - Traditional threshold-based detection
- **Alignment System** (`modules/domain/alignment/`) - Feature-based alignment

## Next Steps

1. Read `flows.md` for detailed execution flow
2. Read `decisions.md` for design rationale and trade-offs
3. Read `constraints.md` for performance limits and browser migration challenges
4. Read `integration.md` for pipeline integration and context management
