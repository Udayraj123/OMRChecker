# ML Field Block Detector - Execution Flow

## High-Level Flow

```
Pipeline
  ↓
Preprocessing (AutoRotate, CropOnMarkers, etc.)
  ↓
Alignment (SIFT-based or Phase Correlation)
  ↓
[ML FIELD BLOCK DETECTOR] ← YOU ARE HERE
  ↓
Shift Detection (uses ML block alignments)
  ↓
Bubble/OCR/Barcode Detection (uses detected regions)
  ↓
Evaluation
  ↓
Output (CSV)
```

## Detailed Step-by-Step Flow

### Step 1: Processor Entry Point

**File**: `src/processors/detection/ml_field_block_detector.py:109-152`

```python
def process(self, context: ProcessingContext) -> ProcessingContext:
    """Run field block detection and compute alignment adjustments."""

    # 1.1: Guard clause - Skip if model not loaded
    if not self.model:
        return context

    logger.debug(f"Starting {self.get_name()} processor")

    # 1.2: Get input image
    image = context.gray_image

    # 1.3: Optional STN preprocessing
    if self.use_stn and self.stn:
        image = self._apply_stn(image)
        logger.debug("Applied STN transformation for alignment refinement")

    # 1.4: Run YOLO inference
    results = self.model.predict(
        image,
        conf=self.confidence_threshold,  # Default: 0.7
        verbose=False,
        imgsz=1024,  # Larger for full OMR sheet
    )

    # 1.5: Parse detections
    detected_blocks = self._parse_block_detections(results, image.shape)
    logger.info(f"ML detected {len(detected_blocks)} field blocks")

    # 1.6: Compare with template
    alignment_adjustments = self._compute_alignment_adjustments(
        detected_blocks, context.template.field_blocks
    )

    # 1.7: Store in context metadata
    context.metadata["ml_detected_blocks"] = detected_blocks
    context.metadata["ml_block_alignments"] = alignment_adjustments

    logger.debug(f"Completed {self.get_name()} processor")
    return context
```

**Key Points**:
- Returns early if model not loaded (graceful degradation)
- Optional STN preprocessing improves robustness
- Stores results in `context.metadata` (not directly modifying template)
- Downstream processors (Shift Detection) use these results

---

### Step 2: STN Preprocessing (Optional)

**File**: `src/processors/detection/ml_field_block_detector.py:154-174`

**Sub-Flow**:

```python
def _apply_stn(self, image: np.ndarray) -> np.ndarray:
    """Apply STN transformation to refine image alignment."""

    # 2.1: Guard clause
    if not self.stn:
        return image

    try:
        # 2.2: Import STN utilities
        from src.processors.detection.models.stn_utils import apply_stn_to_image

        # 2.3: Apply transformation
        # - Converts numpy → torch tensor
        # - Runs STN forward pass
        # - Converts back to numpy
        return apply_stn_to_image(self.stn, image, device="cpu")

    except Exception as e:
        # 2.4: Fallback on error
        logger.warning(f"STN transformation failed: {e}. Using original image.")
        return image
```

**STN Application Details** (`stn_utils.py:132-188`):

```python
def apply_stn_to_image(model, image, device="cpu"):
    # Step 2.3.1: Prepare input tensor
    if len(image.shape) == 2:
        # Grayscale: (H, W) → (1, 1, H, W)
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    else:
        # Color: (H, W, C) → (1, C, H, W)
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)

    # Step 2.3.2: Normalize to [0, 1]
    if image_tensor.max() > 1:
        image_tensor = image_tensor / 255.0

    # Step 2.3.3: Move to device
    image_tensor = image_tensor.to(device)

    # Step 2.3.4: Apply STN (no gradient tracking)
    with torch.no_grad():
        transformed_tensor = model(image_tensor)

    # Step 2.3.5: Convert back to numpy
    transformed = transformed_tensor.cpu().squeeze(0)

    if transformed.shape[0] == 1:
        # Grayscale: (1, H, W) → (H, W)
        transformed = transformed.squeeze(0).numpy()
    else:
        # Color: (C, H, W) → (H, W, C)
        transformed = transformed.permute(1, 2, 0).numpy()

    # Step 2.3.6: Denormalize to [0, 255]
    if transformed.max() <= 1:
        transformed = (transformed * 255).clip(0, 255).astype(np.uint8)

    return transformed
```

**STN Forward Pass** (`stn_module.py:83-106`):

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply spatial transformation to input image."""

    # Step 2.3.4.1: Localization network (predict transformation)
    xs = self.localization(x)      # CNN feature extraction
    xs = xs.view(xs.size(0), -1)   # Flatten: (batch, 32*4*4)
    theta = self.fc_loc(xs)        # Predict 6 parameters: (batch, 6)

    # Step 2.3.4.2: Reshape to 2x3 affine matrix
    theta = theta.view(-1, 2, 3)   # (batch, 2, 3)

    # Step 2.3.4.3: Generate sampling grid
    grid = F.affine_grid(theta, x.size(), align_corners=False)

    # Step 2.3.4.4: Sample from input using bilinear interpolation
    return F.grid_sample(x, grid, align_corners=False)
```

**Transformation Matrix Format**:
```
theta = [[θ11, θ12, θ13],    # [scale_x*cos(θ), -sin(θ), tx]
         [θ21, θ22, θ23]]    # [sin(θ), scale_y*cos(θ), ty]

Where:
  θ11, θ22: Scale + rotation
  θ12, θ21: Rotation + shear
  θ13, θ23: Translation (tx, ty)

Example (5° rotation, 10px right shift):
  [[0.996, -0.087, 0.0098],   # cos(5°) ≈ 0.996, sin(5°) ≈ 0.087
   [0.087,  0.996, 0.0000]]   # 10px / 1024px = 0.0098 (normalized)
```

**Why STN Helps**:
```
Without STN:                    With STN:
┌─────────────┐                ┌─────────────┐
│   ╱━━━━━╲   │                │   ┌─────┐   │
│  ╱       ╲  │ → YOLO →       │   │     │   │ → YOLO →
│ ╱  Skewed╲ │   Misses        │   │Aligned  │   Detects
│╱__________╲│   boxes         │   └─────┘   │   accurately
└─────────────┘                └─────────────┘
```

---

### Step 3: YOLO Inference

**File**: `src/processors/detection/ml_field_block_detector.py:129-135`

```python
results = self.model.predict(
    image,
    conf=self.confidence_threshold,  # Minimum confidence: 0.7
    verbose=False,                   # Suppress YOLO logging
    imgsz=1024,                      # Input size (larger for full OMR sheet)
)
```

**YOLO Model Architecture** (YOLOv8):

```
Input: 1024×1024×1 (grayscale) or 1024×1024×3 (color)
  ↓
Backbone: CSPDarknet
  ├── Conv + BN + SiLU layers
  ├── C2f modules (split-concatenate)
  └── SPPF (Spatial Pyramid Pooling - Fast)
  ↓
Neck: PAN (Path Aggregation Network)
  ├── Feature pyramid (multi-scale)
  └── Bottom-up + top-down fusion
  ↓
Head: Detection head (3 scales)
  ├── Large objects: 32×32 grid
  ├── Medium objects: 64×64 grid
  └── Small objects: 128×128 grid
  ↓
Output: [boxes, scores, classes]
  ├── boxes: [x1, y1, x2, y2] in pixels
  ├── scores: confidence [0.0-1.0]
  └── classes: [0=MCQ, 1=OCR, 2=Barcode]
```

**Inference Parameters**:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `conf` | 0.7 | Filter low-confidence detections |
| `imgsz` | 1024 | Input resolution (higher = more accurate, slower) |
| `verbose` | False | Suppress logging (batch processing) |
| `device` | "cpu" | CPU inference (GPU if available) |

**Performance**:
```
CPU (Intel i7):    ~150ms per image
GPU (NVIDIA RTX):  ~20ms per image
Mobile (ARM):      ~500ms per image (ONNX optimized)
```

---

### Step 4: Parse YOLO Detections

**File**: `src/processors/detection/ml_field_block_detector.py:176-223`

```python
def _parse_block_detections(self, results, image_shape: tuple) -> list[dict]:
    """Parse YOLO detection results into structured format."""

    # 4.1: Guard clause
    if not results or len(results) == 0:
        return []

    image_height, image_width = image_shape[:2]
    detected_blocks = []

    # 4.2: Iterate over results
    for result in results:
        if not hasattr(result, "boxes") or result.boxes is None:
            continue

        # 4.3: Iterate over detected boxes
        for box in result.boxes:
            # 4.3.1: Extract box information
            class_id = int(box.cls[0])         # 0, 1, or 2
            confidence = float(box.conf[0])    # 0.0-1.0

            # 4.3.2: Get bounding box in xyxy format (absolute coordinates)
            xyxy = box.xyxy[0].tolist()        # [x1, y1, x2, y2]
            x1, y1, x2, y2 = xyxy

            # 4.3.3: Convert to origin + dimensions format (OMRChecker convention)
            bbox_origin = [int(x1), int(y1)]
            bbox_dimensions = [int(x2 - x1), int(y2 - y1)]

            # 4.3.4: Create detection dictionary
            block_detection = {
                "class_id": class_id,
                "class_name": self.CLASS_NAMES.get(class_id, "unknown"),
                "confidence": confidence,
                "bbox_origin": bbox_origin,        # [x, y]
                "bbox_dimensions": bbox_dimensions,  # [width, height]
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            }

            detected_blocks.append(block_detection)

    # 4.4: Sort by position (top-to-bottom, left-to-right)
    detected_blocks.sort(key=lambda b: (b["bbox_origin"][1], b["bbox_origin"][0]))

    return detected_blocks
```

**Example Parsed Output**:
```python
detected_blocks = [
    {
        "class_id": 0,
        "class_name": "field_block_mcq",
        "confidence": 0.92,
        "bbox_origin": [100, 200],         # Top-left corner
        "bbox_dimensions": [300, 400],     # Width × height
        "bbox_xyxy": [100, 200, 400, 600]  # [x1, y1, x2, y2]
    },
    {
        "class_id": 0,
        "class_name": "field_block_mcq",
        "confidence": 0.88,
        "bbox_origin": [450, 200],
        "bbox_dimensions": [300, 400],
        "bbox_xyxy": [450, 200, 750, 600]
    },
    {
        "class_id": 1,
        "class_name": "field_block_ocr",
        "confidence": 0.75,
        "bbox_origin": [100, 50],
        "bbox_dimensions": [150, 100],
        "bbox_xyxy": [100, 50, 250, 150]
    }
]
```

**Coordinate Format**:
```
YOLO Output (xyxy):        OMRChecker Format:
  x1, y1, x2, y2             origin + dimensions

  (x1,y1)                    (origin.x, origin.y)
    ┌────────┐                 ┌────────┐
    │        │                 │        │
    │        │                 │ w × h  │
    └────────┘                 └────────┘
         (x2,y2)

Example:
  xyxy: [100, 200, 400, 600]
  origin: [100, 200]
  dimensions: [400-100, 600-200] = [300, 400]
```

**Sorting Logic**:
```python
# Sort by (y, x): top-to-bottom, then left-to-right
detected_blocks.sort(key=lambda b: (b["bbox_origin"][1], b["bbox_origin"][0]))

# Example:
#   Block A: origin=[100, 200]  → sort key = (200, 100)
#   Block B: origin=[450, 200]  → sort key = (200, 450)
#   Block C: origin=[100, 50]   → sort key = (50, 100)
#
# Sorted order: C (y=50), A (y=200, x=100), B (y=200, x=450)
```

---

### Step 5: Compute Alignment Adjustments

**File**: `src/processors/detection/ml_field_block_detector.py:225-286`

```python
def _compute_alignment_adjustments(
    self, ml_blocks: list[dict], template_blocks: list
) -> dict:
    """Compare ML-detected blocks with template expectations."""

    adjustments = {}

    # 5.1: Guard clause
    if not ml_blocks or not template_blocks:
        return adjustments

    # 5.2: For each template block, find closest ML detection
    for template_block in template_blocks:
        # 5.2.1: Get template block info
        template_name = template_block.name
        template_origin = template_block.get_shifted_origin()
        template_center = bbox_center(
            template_origin, template_block.bounding_box_dimensions
        )

        # 5.2.2: Find closest ML detection by Euclidean distance
        best_match = None
        best_distance = float("inf")

        for ml_block in ml_blocks:
            ml_center = bbox_center(
                ml_block["bbox_origin"], ml_block["bbox_dimensions"]
            )

            distance = euclidean_distance(template_center, ml_center)

            if distance < best_distance:
                best_distance = distance
                best_match = ml_block

        # 5.2.3: Accept match if within threshold (200 pixels)
        if best_match and best_distance < 200:
            # 5.2.3.1: Compute shift
            ml_origin = best_match["bbox_origin"]
            shift_x = ml_origin[0] - template_origin[0]
            shift_y = ml_origin[1] - template_origin[1]

            # 5.2.3.2: Store adjustment
            adjustments[template_name] = {
                "matched_ml_block": best_match,
                "shift": [shift_x, shift_y],
                "distance": best_distance,
                "confidence": best_match["confidence"],
            }

            logger.debug(
                f"Block '{template_name}' matched with shift: [{shift_x}, {shift_y}], "
                f"confidence: {best_match['confidence']:.2f}"
            )

    return adjustments
```

**Matching Algorithm Details**:

#### 5.2.1: Compute Centers

**Helper Function** (`src/utils/geometry.py`):
```python
def bbox_center(origin: list[int], dimensions: list[int]) -> list[float]:
    """Calculate center point of bounding box.

    Args:
        origin: [x, y] top-left corner
        dimensions: [width, height]

    Returns:
        [center_x, center_y]
    """
    return [
        origin[0] + dimensions[0] / 2.0,  # x + width/2
        origin[1] + dimensions[1] / 2.0   # y + height/2
    ]
```

**Example**:
```python
template_origin = [100, 200]
template_dimensions = [300, 400]
template_center = bbox_center(template_origin, template_dimensions)
# = [100 + 300/2, 200 + 400/2] = [250, 400]
```

#### 5.2.2: Compute Distances

**Helper Function** (`src/utils/geometry.py`):
```python
def euclidean_distance(point1: list[float], point2: list[float]) -> float:
    """Calculate Euclidean distance between two points.

    Args:
        point1: [x1, y1]
        point2: [x2, y2]

    Returns:
        Distance in pixels
    """
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return (dx ** 2 + dy ** 2) ** 0.5
```

**Example**:
```python
template_center = [250, 400]
ml_center = [255, 397]
distance = euclidean_distance(template_center, ml_center)
# = √((255-250)² + (397-400)²) = √(25 + 9) = √34 ≈ 5.83 pixels
```

#### 5.2.3: Threshold Filtering

**Why 200 pixels?**
```
Typical OMR sheet field blocks:
  - Width: 200-500 pixels
  - Height: 300-600 pixels

200 pixels = ~40-100% of width
  - Handles reasonable scan variations
  - Filters out wrong block matches
  - Prevents matching distant blocks

Examples:
  Good match:  distance = 5.83 pixels   ✓ ACCEPT (< 200)
  OK match:    distance = 150 pixels    ✓ ACCEPT (< 200)
  Bad match:   distance = 250 pixels    ✗ REJECT (> 200)
```

#### 5.2.3.1: Compute Shift

```python
shift_x = ml_origin[0] - template_origin[0]
shift_y = ml_origin[1] - template_origin[1]

# Interpretation:
#   Positive shift_x: ML block is RIGHT of template
#   Negative shift_x: ML block is LEFT of template
#   Positive shift_y: ML block is BELOW template
#   Negative shift_y: ML block is ABOVE template

# Example:
template_origin = [100, 200]
ml_origin = [105, 197]
shift = [105 - 100, 197 - 200] = [5, -3]
# ML block is 5 pixels RIGHT and 3 pixels ABOVE template
```

**Complete Matching Example**:
```
Template Blocks:
  Q1-30:  origin=[100, 200], dimensions=[300, 400]  → center=[250, 400]
  Q31-60: origin=[450, 200], dimensions=[300, 400]  → center=[600, 400]
  Name:   origin=[100, 50],  dimensions=[150, 100]  → center=[175, 100]

ML Detections (sorted):
  [0] class=OCR,  origin=[102, 48],  dimensions=[148, 102] → center=[176, 99]
  [1] class=MCQ,  origin=[105, 197], dimensions=[298, 403] → center=[254, 398.5]
  [2] class=MCQ,  origin=[448, 203], dimensions=[302, 397] → center=[599, 401.5]

Matching:
  Q1-30:
    - ML[0] distance = √((176-250)² + (99-400)²) = √(5476+90601) = 309.95 ✗ > 200
    - ML[1] distance = √((254-250)² + (398.5-400)²) = √(16+2.25) = 4.27 ✓ < 200
    - ML[2] distance = √((599-250)² + (401.5-400)²) = √(121801+2.25) = 349.0 ✗ > 200
    → Best match: ML[1], shift = [105-100, 197-200] = [5, -3]

  Q31-60:
    - ML[0] distance = √((176-600)² + (99-400)²) = √(179776+90601) = 520.0 ✗ > 200
    - ML[1] distance = √((254-600)² + (398.5-400)²) = √(119716+2.25) = 346.0 ✗ > 200
    - ML[2] distance = √((599-600)² + (401.5-400)²) = √(1+2.25) = 1.80 ✓ < 200
    → Best match: ML[2], shift = [448-450, 203-200] = [-2, 3]

  Name:
    - ML[0] distance = √((176-175)² + (99-100)²) = √(1+1) = 1.41 ✓ < 200
    - ML[1] distance = √((254-175)² + (398.5-100)²) = √(6241+89100.25) = 308.75 ✗ > 200
    - ML[2] distance = √((599-175)² + (401.5-100)²) = √(179776+90902.25) = 520.36 ✗ > 200
    → Best match: ML[0], shift = [102-100, 48-50] = [2, -2]

Result:
  adjustments = {
    "Q1-30": {
      "matched_ml_block": ML[1],
      "shift": [5, -3],
      "distance": 4.27,
      "confidence": 0.92
    },
    "Q31-60": {
      "matched_ml_block": ML[2],
      "shift": [-2, 3],
      "distance": 1.80,
      "confidence": 0.88
    },
    "Name": {
      "matched_ml_block": ML[0],
      "shift": [2, -2],
      "distance": 1.41,
      "confidence": 0.75
    }
  }
```

---

### Step 6: Store Results in Context

**File**: `src/processors/detection/ml_field_block_detector.py:148-150`

```python
# Store in context for downstream use
context.metadata["ml_detected_blocks"] = detected_blocks
context.metadata["ml_block_alignments"] = alignment_adjustments
```

**Metadata Structure**:
```python
context.metadata = {
    "ml_detected_blocks": [
        {
            "class_id": 0,
            "class_name": "field_block_mcq",
            "confidence": 0.92,
            "bbox_origin": [105, 197],
            "bbox_dimensions": [298, 403],
            "bbox_xyxy": [105, 197, 403, 600]
        },
        # ... more detections
    ],
    "ml_block_alignments": {
        "Q1-30": {
            "matched_ml_block": {...},
            "shift": [5, -3],
            "distance": 4.27,
            "confidence": 0.92
        },
        # ... more adjustments
    }
}
```

**Downstream Usage**:
```python
# Shift Detection Processor:
def process(self, context):
    alignments = context.metadata.get("ml_block_alignments", {})

    for block_name, adjustment in alignments.items():
        if adjustment["confidence"] > 0.8:
            # Apply shift to template coordinates
            field_block = context.template.get_field_block(block_name)
            field_block.apply_shift(adjustment["shift"])

# Visualization:
def draw_detections(context):
    ml_blocks = context.metadata.get("ml_detected_blocks", [])

    for block in ml_blocks:
        cv2.rectangle(
            image,
            tuple(block["bbox_origin"]),
            tuple([block["bbox_origin"][0] + block["bbox_dimensions"][0],
                   block["bbox_origin"][1] + block["bbox_dimensions"][1]]),
            color=(0, 255, 0),
            thickness=2
        )
        cv2.putText(
            image,
            f"{block['class_name']} {block['confidence']:.2f}",
            tuple(block["bbox_origin"]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
```

---

## Data Structures

### Detection Result

```python
{
    "class_id": int,           # 0=MCQ, 1=OCR, 2=Barcode
    "class_name": str,         # "field_block_mcq", "field_block_ocr", "field_block_barcode"
    "confidence": float,       # 0.0-1.0
    "bbox_origin": [int, int], # [x, y] top-left corner
    "bbox_dimensions": [int, int],  # [width, height]
    "bbox_xyxy": [int, int, int, int]  # [x1, y1, x2, y2]
}
```

### Alignment Adjustment

```python
{
    "<block_name>": {
        "matched_ml_block": {...},  # Full detection dict
        "shift": [int, int],        # [dx, dy] pixel displacement
        "distance": float,          # Euclidean distance in pixels
        "confidence": float         # ML detection confidence
    }
}
```

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Typical Time |
|-----------|-----------|--------------|
| STN forward pass | O(HW) | ~50ms (1024×1024) |
| YOLO inference | O(HW) | ~100ms (1024×1024, CPU) |
| Parse detections | O(N) | <1ms (N ≈ 5-20) |
| Bbox center calculation | O(1) | <0.01ms per block |
| Distance calculation | O(1) | <0.01ms per pair |
| Alignment matching | O(M×N) | <1ms (M ≈ 5 template, N ≈ 5 ML) |
| **Total** | **O(HW)** | **~150ms with STN, ~100ms without** |

### Memory Usage

```
Peak Memory per Image:
  - Input image: 1024×1024×1 byte = 1 MB
  - STN transformed: 1024×1024×1 byte = 1 MB (temporary)
  - YOLO model: ~800 MB (cached)
  - STN model: ~50 MB (cached)
  - Detection results: ~2 KB
  - Total: ~900 MB (model cached), ~2 MB per image
```

---

## Error Handling

### Model Not Loaded

```python
# Guard clause at entry
if not self.model:
    return context  # Skip detection, proceed without ML blocks
```

### STN Failure

```python
except Exception as e:
    logger.warning(f"STN transformation failed: {e}. Using original image.")
    return image  # Fallback to original image
```

### No Detections

```python
if not results or len(results) == 0:
    return []  # Return empty list, downstream handles gracefully
```

### No Matches Found

```python
if not ml_blocks or not template_blocks:
    return adjustments  # Return empty dict, no adjustments applied
```

---

## Browser Migration Notes

### Async Processing

```javascript
async process(context) {
  // Step 1: Optional STN (async)
  let image = context.grayImage;
  if (this.useStn && this.stnSession) {
    image = await this.applyStn(image);
  }

  // Step 2: YOLO inference (async)
  const inputTensor = this.prepareInput(image);
  const outputs = await this.yoloSession.run({ images: inputTensor });

  // Step 3: Parse (sync)
  const detectedBlocks = this.parseDetections(outputs, image.shape);

  // Step 4: Alignment matching (sync)
  const alignmentAdjustments = this.computeAlignmentAdjustments(
    detectedBlocks,
    context.template.fieldBlocks
  );

  // Step 5: Store in metadata
  context.metadata.ml_detected_blocks = detectedBlocks;
  context.metadata.ml_block_alignments = alignmentAdjustments;

  return context;
}
```

### Web Worker Offloading

```javascript
// Main thread:
const worker = new Worker('ml-field-block-worker.js');

worker.postMessage({
  type: 'detect',
  image: imageData,
  template: templateJSON,
  confidenceThreshold: 0.7
});

worker.onmessage = (event) => {
  const { detectedBlocks, alignmentAdjustments } = event.data;
  context.metadata.ml_detected_blocks = detectedBlocks;
  context.metadata.ml_block_alignments = alignmentAdjustments;
  continueProcessing(context);
};

// Worker:
self.onmessage = async (event) => {
  const { image, template, confidenceThreshold } = event.data;
  const detector = new MLFieldBlockDetector(modelPath, confidenceThreshold);
  const detectedBlocks = await detector.detect(image);
  const alignmentAdjustments = detector.computeAlignments(detectedBlocks, template);
  self.postMessage({ detectedBlocks, alignmentAdjustments });
};
```

---

## Related Flows

- **STN Forward Pass** (`stn/flows.md`) - Detailed STN transformation
- **YOLO Training** (see training scripts in `scripts/ai-generated/`)
- **Shift Detection** (`shift-detection/flows.md`) - Uses ML block alignments
- **Bubble Detection** (`bubbles-threshold/flows.md`) - Uses detected regions
- **Alignment Flow** (`modules/domain/alignment/flows.md`) - Feature-based alignment

---

## Testing Considerations

### Test Cases

1. **Perfect Detection**: All blocks detected with high confidence (>0.9)
2. **Partial Detection**: Some blocks missed (confidence <0.7)
3. **No Detections**: YOLO returns empty results
4. **Wrong Class**: Block detected with incorrect class
5. **Overlapping Blocks**: Multiple detections for same region
6. **STN Enabled vs Disabled**: Compare accuracy
7. **Different Confidence Thresholds**: 0.5, 0.7, 0.9

### Edge Cases

- Empty image (all black/white)
- Rotated image (90°, 180°, 270°)
- Cropped image (partial field blocks)
- Multiple pages in single image
- Very low resolution (<500×500)
- Very high resolution (>4000×4000)
