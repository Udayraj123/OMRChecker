# ML Field Block Detector - Integration Points

## Pipeline Integration

### Position in Processing Pipeline

```
Entry Point
  ↓
Pre-processors (AutoRotate, CropOnMarkers, GaussianBlur, etc.)
  ↓
Alignment Processor (SIFT-based coordinate warping)
  ↓
[ML FIELD BLOCK DETECTOR] ← YOU ARE HERE
  ↓
Shift Detection Processor (uses ML block alignments)
  ↓
Detection Processor (Bubble/OCR/Barcode within detected regions)
  ↓
Evaluation Processor
  ↓
Output (CSV)
```

**Key Position**: After alignment, before shift detection

**Why This Order?**
1. **After Pre-processors**: Image normalized (rotation, cropping, filtering)
2. **After Alignment**: Coarse-grain coordinate adjustments applied
3. **Before Shift Detection**: ML provides fine-grain alignment hints
4. **Before Detection**: ML regions guide where to look for bubbles

---

## Integration with ProcessingContext

### Context Flow

**Input Context**:
```python
class ProcessingContext:
    gray_image: np.ndarray          # From pre-processors + alignment
    colored_image: np.ndarray       # Color version (optional)
    template: Template              # Contains field block definitions
    metadata: dict                  # Empty or from previous processors
    # ... other fields
```

**ML Field Block Detector Processing**:
```python
def process(self, context: ProcessingContext) -> ProcessingContext:
    # Read from context
    gray_image = context.gray_image
    template = context.template

    # Optional STN preprocessing
    if self.use_stn and self.stn:
        image = self._apply_stn(gray_image)
    else:
        image = gray_image

    # Run YOLO inference
    results = self.model.predict(image, conf=self.confidence_threshold, imgsz=1024)

    # Parse detections
    detected_blocks = self._parse_block_detections(results, image.shape)

    # Compute alignment adjustments
    alignment_adjustments = self._compute_alignment_adjustments(
        detected_blocks, template.field_blocks
    )

    # Write to context metadata
    context.metadata["ml_detected_blocks"] = detected_blocks
    context.metadata["ml_block_alignments"] = alignment_adjustments

    return context
```

**Output Context**:
```python
# Updated context with ML metadata
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
        # ... more alignments
    }
}
```

---

## Integration with Template System

### Template Loading

**File**: `src/processors/template/template.py` (inferred)

**Template Structure** (relevant to ML detection):
```python
class Template:
    field_blocks: list[FieldBlock]
    template_dimensions: tuple[int, int]  # [width, height]
    # ... other fields

class FieldBlock:
    name: str
    origin: list[int]                   # [x, y]
    bounding_box_origin: list[int]      # [x, y]
    bounding_box_dimensions: list[int]  # [width, height]
    fields: list[Field]
    # ... other fields
```

**Template JSON Example**:
```json
{
  "templateDimensions": [1200, 1600],
  "fieldBlocks": {
    "Q1-30": {
      "origin": [100, 200],
      "dimensions": [300, 400],
      "fieldType": "BUBBLES_THRESHOLD",
      "fieldLabels": ["Q1", "Q2", ..., "Q30"]
    },
    "Q31-60": {
      "origin": [450, 200],
      "dimensions": [300, 400],
      "fieldType": "BUBBLES_THRESHOLD",
      "fieldLabels": ["Q31", "Q32", ..., "Q60"]
    },
    "StudentName": {
      "origin": [100, 50],
      "dimensions": [150, 100],
      "fieldType": "OCR"
    }
  }
}
```

**Integration Point**: Template field blocks compared with ML detections

```python
# Template field blocks (from JSON)
template_blocks = [
    FieldBlock(name="Q1-30", origin=[100, 200], dimensions=[300, 400]),
    FieldBlock(name="Q31-60", origin=[450, 200], dimensions=[300, 400]),
    FieldBlock(name="StudentName", origin=[100, 50], dimensions=[150, 100])
]

# ML detections (from YOLO)
ml_blocks = [
    {"class_name": "field_block_ocr", "bbox_origin": [102, 48], ...},
    {"class_name": "field_block_mcq", "bbox_origin": [105, 197], ...},
    {"class_name": "field_block_mcq", "bbox_origin": [448, 203], ...}
]

# Alignment matching
alignments = compute_alignment_adjustments(ml_blocks, template_blocks)
# {
#   "Q1-30": {"shift": [5, -3], ...},
#   "Q31-60": {"shift": [-2, 3], ...},
#   "StudentName": {"shift": [2, -2], ...}
# }
```

---

## Integration with Shift Detection Processor

**Downstream Consumer**: Shift Detection Processor uses ML alignments

**File**: `src/processors/detection/shift_detection_processor.py` (inferred)

```python
class ShiftDetectionProcessor(Processor):
    """Refines field block positions using ML detections."""

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # Read ML alignments from metadata
        ml_alignments = context.metadata.get("ml_block_alignments", {})

        if not ml_alignments:
            logger.info("No ML alignments available, using template coordinates")
            return context

        # Apply shifts to field blocks
        for field_block in context.template.field_blocks:
            block_name = field_block.name

            if block_name in ml_alignments:
                adjustment = ml_alignments[block_name]

                # Only apply high-confidence adjustments
                if adjustment["confidence"] > 0.8:
                    shift = adjustment["shift"]

                    # Update field block shifts
                    field_block.shifts = shift

                    # Propagate to all scan boxes
                    for field in field_block.fields:
                        for scan_box in field.scan_boxes:
                            scan_box.shifts = shift

                    logger.debug(
                        f"Applied ML shift to '{block_name}': {shift}, "
                        f"confidence: {adjustment['confidence']:.2f}"
                    )

        return context
```

**Data Flow**:
```
ML Field Block Detector:
  context.metadata["ml_block_alignments"] = {
    "Q1-30": {"shift": [5, -3], "confidence": 0.92}
  }
  ↓
Shift Detection Processor:
  Reads ml_block_alignments
  Applies to template.field_blocks[i].shifts
  ↓
Bubble Detection:
  Uses field_block.shifts for accurate positioning
```

---

## Integration with Detection Processors

### Bubble Detection Integration

**File**: `src/processors/detection/bubbles_threshold/` (inferred)

**Usage**: ML-detected regions guide where to detect bubbles

```python
class BubbleDetectionProcessor(Processor):
    def process(self, context: ProcessingContext) -> ProcessingContext:
        # Read ML detected blocks (optional optimization)
        ml_blocks = context.metadata.get("ml_detected_blocks", [])

        for field_block in context.template.field_blocks:
            # Use shift (potentially from ML alignments)
            shift = field_block.shifts  # [dx, dy] from Shift Detection

            for field in field_block.fields:
                for scan_box in field.scan_boxes:
                    # Apply shift to position
                    x = scan_box.position[0] + shift[0]
                    y = scan_box.position[1] + shift[1]

                    # Extract ROI
                    roi = context.gray_image[
                        y - radius : y + radius,
                        x - radius : x + radius
                    ]

                    # Detect bubble darkness
                    darkness = calculate_darkness(roi)
                    scan_box.darkness = darkness

        return context
```

### OCR Detection Integration

**File**: `src/processors/detection/ocr/` (inferred)

**Usage**: ML class labels filter which blocks to process with OCR

```python
class OCRDetectionProcessor(Processor):
    def process(self, context: ProcessingContext) -> ProcessingContext:
        ml_blocks = context.metadata.get("ml_detected_blocks", [])

        # Filter OCR blocks
        ocr_blocks = [
            block for block in ml_blocks
            if block["class_name"] == "field_block_ocr"
        ]

        for ocr_block in ocr_blocks:
            # Extract ROI
            x, y = ocr_block["bbox_origin"]
            w, h = ocr_block["bbox_dimensions"]

            roi = context.gray_image[y:y+h, x:x+w]

            # Run OCR
            text = self.ocr_engine.extract_text(roi)

            # Store result
            context.metadata["ocr_results"] = context.metadata.get("ocr_results", {})
            context.metadata["ocr_results"][ocr_block["bbox_origin"]] = text

        return context
```

### Barcode Detection Integration

**File**: `src/processors/detection/barcode/` (inferred)

**Usage**: ML class labels identify barcode regions

```python
class BarcodeDetectionProcessor(Processor):
    def process(self, context: ProcessingContext) -> ProcessingContext:
        ml_blocks = context.metadata.get("ml_detected_blocks", [])

        # Filter barcode blocks
        barcode_blocks = [
            block for block in ml_blocks
            if block["class_name"] == "field_block_barcode"
        ]

        for barcode_block in barcode_blocks:
            # Extract ROI
            x, y = barcode_block["bbox_origin"]
            w, h = barcode_block["bbox_dimensions"]

            roi = context.gray_image[y:y+h, x:x+w]

            # Decode barcode
            codes = self.barcode_decoder.decode(roi)

            if codes:
                context.metadata["barcode_results"] = codes[0].data.decode()

        return context
```

---

## Integration with Visualization System

### Drawing ML Detections

**File**: `src/processors/visualization/` (inferred)

```python
def draw_ml_detections(context: ProcessingContext, output_image: np.ndarray):
    """Draw ML-detected field blocks for debugging."""
    ml_blocks = context.metadata.get("ml_detected_blocks", [])

    for block in ml_blocks:
        # Unpack bounding box
        x1, y1, x2, y2 = block["bbox_xyxy"]
        class_name = block["class_name"]
        confidence = block["confidence"]

        # Color by class
        color_map = {
            "field_block_mcq": (0, 255, 0),      # Green
            "field_block_ocr": (255, 0, 0),      # Blue
            "field_block_barcode": (0, 0, 255)   # Red
        }
        color = color_map.get(class_name, (255, 255, 255))

        # Draw rectangle
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{class_name.replace('field_block_', '')}: {confidence:.2f}"
        cv2.putText(
            output_image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    return output_image
```

**Usage**:
```python
# In visualization processor
output_image = context.colored_image.copy()
output_image = draw_ml_detections(context, output_image)

cv2.imwrite("debug_ml_detections.png", output_image)
```

---

## Integration with Configuration System

### TuningConfig Integration

**File**: `src/schemas/models/config.py` (inferred)

```python
@dataclass
class MLConfig:
    """Configuration for ML-based detection."""

    enable_field_block_detection: bool = False
    field_block_model_path: str = "models/field_block_yolo.pt"
    field_block_confidence_threshold: float = 0.7
    use_stn: bool = False
    stn_model_path: str | None = None

@dataclass
class TuningConfig:
    ml: MLConfig = field(default_factory=MLConfig)
    # ... other configs
```

**Template JSON**:
```json
{
  "tuningOptions": {
    "ml": {
      "enableFieldBlockDetection": true,
      "fieldBlockModelPath": "models/field_block_yolo_v8s.pt",
      "fieldBlockConfidenceThreshold": 0.7,
      "useStn": true,
      "stnModelPath": "models/stn_alignment.pt"
    }
  }
}
```

**Pipeline Setup**:
```python
# Read config
config = TuningConfig.from_dict(config_json)

# Conditionally add ML Field Block Detector
processors = [
    PreprocessingCoordinator(template),
    AlignmentProcessor(template),
]

if config.ml.enable_field_block_detection:
    ml_detector = MLFieldBlockDetector(
        model_path=config.ml.field_block_model_path,
        confidence_threshold=config.ml.field_block_confidence_threshold,
        use_stn=config.ml.use_stn,
        stn_model_path=config.ml.stn_model_path
    )
    processors.append(ml_detector)

processors.extend([
    ShiftDetectionProcessor(template),
    DetectionProcessor(template),
    EvaluationProcessor(template)
])

# Run pipeline
for processor in processors:
    context = processor.process(context)
```

---

## Integration with Geometry Utilities

### Dependencies

**File**: `src/utils/geometry.py`

**Used Functions**:

1. **bbox_center**:
```python
from src.utils.geometry import bbox_center

# In _compute_alignment_adjustments:
template_center = bbox_center(
    template_block.origin,
    template_block.bounding_box_dimensions
)
ml_center = bbox_center(
    ml_block["bbox_origin"],
    ml_block["bbox_dimensions"]
)
```

2. **euclidean_distance**:
```python
from src.utils.geometry import euclidean_distance

# In _compute_alignment_adjustments:
distance = euclidean_distance(template_center, ml_center)

if distance < 200:  # Threshold
    # Accept match
```

**Function Signatures**:
```python
# src/utils/geometry.py
def bbox_center(origin: list[int], dimensions: list[int]) -> list[float]:
    """Calculate bounding box center.

    Args:
        origin: [x, y] top-left corner
        dimensions: [width, height]

    Returns:
        [center_x, center_y]
    """
    return [
        origin[0] + dimensions[0] / 2.0,
        origin[1] + dimensions[1] / 2.0
    ]

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

---

## Integration with Logging System

### Logger Integration

**File**: `src/utils/logger.py`

**Usage Patterns**:

1. **Debug Level** (verbose details):
```python
logger.debug(f"Starting {self.get_name()} processor")
logger.debug("Applied STN transformation for alignment refinement")
logger.debug(
    f"Block '{template_name}' matched with shift: [{shift_x}, {shift_y}], "
    f"confidence: {confidence:.2f}"
)
```

2. **Info Level** (important events):
```python
logger.info(f"ML detected {len(detected_blocks)} field blocks")
logger.info(f"MLFieldBlockDetector initialized with STN, model: {model_path}")
```

3. **Warning Level** (non-critical issues):
```python
logger.warning(f"Field block model not found at {model_path}, detector disabled")
logger.warning("use_stn=True but no stn_model_path provided. STN disabled.")
logger.warning(f"STN transformation failed: {e}. Using original image.")
```

4. **Error Level** (failures):
```python
logger.error("ultralytics package not found. Install ML dependencies with: uv sync --extra ml")
logger.error(f"Failed to load STN model: {e}")
```

---

## Integration with STN Module

### STN Utilities Dependencies

**File**: `src/processors/detection/models/stn_utils.py`

**Used Functions**:

1. **load_stn_model**:
```python
from src.processors.detection.models.stn_utils import load_stn_model

# In __init__:
if use_stn and stn_model_path:
    self.stn = load_stn_model(
        stn_model_path,
        input_channels=1,
        input_size=(1024, 1024),
        device="cpu"
    )
```

2. **apply_stn_to_image**:
```python
from src.processors.detection.models.stn_utils import apply_stn_to_image

# In _apply_stn:
return apply_stn_to_image(self.stn, image, device="cpu")
```

**Function Signatures**:
```python
# src/processors/detection/models/stn_utils.py
def load_stn_model(
    model_path: str | Path,
    input_channels: int = 1,
    input_size: tuple[int, int] = (640, 640),
    device: str = "cpu"
) -> SpatialTransformerNetwork | TranslationOnlySTN:
    """Load trained STN model from disk."""

def apply_stn_to_image(
    model: SpatialTransformerNetwork | TranslationOnlySTN,
    image: np.ndarray,
    device: str = "cpu"
) -> np.ndarray:
    """Apply STN transformation to numpy image."""

def get_transformation_matrix(
    model: SpatialTransformerNetwork | TranslationOnlySTN,
    image: np.ndarray,
    device: str = "cpu"
) -> np.ndarray:
    """Get predicted transformation matrix (2, 3)."""
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Template JSON                                                │
│ - fieldBlocks definitions                                    │
│ - tuningOptions.ml config                                    │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ Template Parser                                              │
│ - Parse field blocks                                         │
│ - Parse ML config                                            │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ ProcessingContext                                            │
│ - gray_image (from pre-processors + alignment)               │
│ - template (with field blocks)                               │
│ - metadata (empty initially)                                 │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ MLFieldBlockDetector.__init__()                              │
│ - Load YOLO model (cached)                                   │
│ - Optional: Load STN model                                   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ MLFieldBlockDetector.process()                               │
│ 1. Optional STN preprocessing (_apply_stn)                   │
│ 2. YOLO inference (model.predict)                            │
│ 3. Parse detections (_parse_block_detections)                │
│ 4. Compute alignments (_compute_alignment_adjustments)       │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ Updated ProcessingContext                                    │
│ - metadata["ml_detected_blocks"] = [...]                     │
│ - metadata["ml_block_alignments"] = {...}                    │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ ShiftDetectionProcessor                                      │
│ - Reads ml_block_alignments                                  │
│ - Applies shifts to template.field_blocks                    │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ Detection Processors (Bubble/OCR/Barcode)                    │
│ - Use field_block.shifts for accurate positioning            │
│ - Optional: Filter by ML class labels                        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ Evaluation & Output                                          │
│ - CSV with detected values                                   │
│ - Debug visualization (optional)                             │
└─────────────────────────────────────────────────────────────┘
```

---

## File Dependencies Map

```
MLFieldBlockDetector (ml_field_block_detector.py)
├── YOLO Model (ultralytics)
│   ├── torch
│   └── torchvision
│
├── Optional: STN Model (stn_module.py, stn_utils.py)
│   ├── SpatialTransformerNetwork
│   ├── TranslationOnlySTN
│   └── torch.nn.functional
│
├── ProcessingContext (src/processors/base.py)
│
├── Template & FieldBlock (src/processors/template/, src/processors/layout/field_block/)
│
├── Geometry Utils (src/utils/geometry.py)
│   ├── bbox_center()
│   └── euclidean_distance()
│
└── Logger (src/utils/logger.py)
```

---

## Browser Integration Considerations

### ONNX Runtime Web Integration

```javascript
import * as ort from 'onnxruntime-web';

class MLFieldBlockDetector {
  constructor(modelPath, confidenceThreshold = 0.7) {
    this.modelPath = modelPath;
    this.confidenceThreshold = confidenceThreshold;
    this.session = null;
  }

  async initialize() {
    // Load model from IndexedDB cache or URL
    const modelBuffer = await this.loadModelCached(this.modelPath);
    this.session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['webgl', 'wasm'],  // Prefer WebGL, fallback to WASM
      graphOptimizationLevel: 'all'
    });
  }

  async loadModelCached(modelUrl) {
    const cache = await caches.open('ml-models');
    let response = await cache.match(modelUrl);

    if (!response) {
      response = await fetch(modelUrl);
      await cache.put(modelUrl, response.clone());
    }

    return await response.arrayBuffer();
  }

  async process(context) {
    if (!this.session) await this.initialize();

    // Prepare input
    const inputTensor = this.prepareInput(context.grayImage);

    // Run inference
    const outputs = await this.session.run({ images: inputTensor });

    // Parse detections
    const detectedBlocks = this.parseDetections(outputs, context.grayImage.shape);

    // Compute alignments
    const alignmentAdjustments = this.computeAlignmentAdjustments(
      detectedBlocks,
      context.template.fieldBlocks
    );

    // Store in metadata
    context.metadata.ml_detected_blocks = detectedBlocks;
    context.metadata.ml_block_alignments = alignmentAdjustments;

    return context;
  }
}
```

### Web Worker Integration

```javascript
// Main thread
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

// Worker thread (ml-field-block-worker.js)
importScripts('onnxruntime-web.min.js');

let session = null;

self.onmessage = async (event) => {
  const { type, image, template, confidenceThreshold } = event.data;

  if (type === 'detect') {
    if (!session) {
      const modelBuffer = await fetchModel('models/field_block_yolo.onnx');
      session = await ort.InferenceSession.create(modelBuffer);
    }

    const detectedBlocks = await detectBlocks(session, image, confidenceThreshold);
    const alignmentAdjustments = computeAlignments(detectedBlocks, template);

    self.postMessage({ detectedBlocks, alignmentAdjustments });
  }
};
```

---

## API Contract

### Input Requirements

```python
# MLFieldBlockDetector.__init__() expects:
model_path: str                    # Path to YOLO .pt file (must exist)
confidence_threshold: float = 0.7  # 0.0-1.0
use_stn: bool = False              # Optional STN preprocessing
stn_model_path: str | None = None  # Required if use_stn=True

# MLFieldBlockDetector.process() expects:
context.gray_image: np.ndarray          # Shape: (H, W), dtype: uint8
context.template.field_blocks: list     # List of FieldBlock objects
context.metadata: dict                  # Existing metadata (may be empty)
```

### Output Guarantees

```python
# MLFieldBlockDetector.process() returns:
context.metadata["ml_detected_blocks"]: list[dict]  # Sorted by position
context.metadata["ml_block_alignments"]: dict       # Template name → adjustment

# Invariants:
# - detected_blocks sorted by (y, x) position
# - All bbox coordinates within image bounds (0 ≤ x < W, 0 ≤ y < H)
# - All confidences in range [0.0, 1.0]
# - All class_ids in [0, 1, 2]
# - alignment shifts reasonable (typically < 200 pixels)
# - If ML fails, returns empty lists/dicts (no exceptions raised)
```

---

## Testing Integration Points

### Unit Test Interfaces

```python
# Test MLFieldBlockDetector
def test_ml_field_block_detector():
    detector = MLFieldBlockDetector(
        model_path="tests/fixtures/test_yolo_model.pt",
        confidence_threshold=0.7
    )

    context = ProcessingContext(
        gray_image=load_test_image("test_omr_sheet.png"),
        template=load_test_template("test_template.json"),
        metadata={}
    )

    result_context = detector.process(context)

    # Assert detections stored
    assert "ml_detected_blocks" in result_context.metadata
    assert len(result_context.metadata["ml_detected_blocks"]) > 0

    # Assert alignments stored
    assert "ml_block_alignments" in result_context.metadata
```

### Integration Test Points

1. **Pre-processor → ML Detector**: Verify pre-processed images compatible
2. **ML Detector → Shift Detection**: Verify alignments applied correctly
3. **ML Detector → Bubble Detection**: Verify shifts improve detection accuracy
4. **Template Loading → ML Detector**: Verify field blocks parsed correctly
5. **Config → ML Detector**: Verify ML config enables/disables detector

---

## Related Integration Docs

- **Pipeline Integration**: See `modules/domain/pipeline/integration.md`
- **Shift Detection Integration**: See `modules/domain/detection/shift-detection/integration.md`
- **STN Integration**: See `modules/domain/detection/stn/integration.md`
- **Context Management**: See `modules/domain/processing-context/concept.md`
- **Template System**: See `modules/domain/template/concept.md`
