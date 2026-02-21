# ReadOMR Processor - Core Concepts

## Overview

The **ReadOMR Processor** is the central orchestrator for OMR detection and interpretation in the OMRChecker pipeline. It coordinates multi-format field detection (bubbles, OCR, barcodes), runs interpretation logic, and optionally applies ML-based fallback for low-confidence results. This is the processor that transforms aligned images into structured OMR response data.

## What is ReadOMR?

ReadOMR is the processor that:
- Creates and manages a `TemplateFileRunner` for template-based detection
- Resizes and normalizes images to template dimensions
- Dispatches detection tasks to field-type-specific runners (bubbles, OCR, barcode)
- Runs interpretation logic to convert raw detections into answers
- Optionally uses ML fallback (YOLO-based) for low-confidence detections
- Stores results in ProcessingContext for downstream consumers

**Key Insight**: ReadOMR is a **coordinator**, not a detector. It delegates actual detection to specialized runners based on field type.

## Why is ReadOMR Needed?

### Multi-Format Support
OMR sheets contain different types of fields:
- **Bubble fields**: Multiple-choice questions (BUBBLES_THRESHOLD)
- **Text fields**: Student names, IDs (OCR)
- **Code fields**: QR codes, barcodes (BARCODE_QR)

Each field type requires different detection algorithms. ReadOMR dispatches to the right detector.

### Unified Interface
```python
# Without ReadOMR (manual dispatch):
for field in fields:
    if field.type == "BUBBLES_THRESHOLD":
        run_bubble_detection(field)
    elif field.type == "OCR":
        run_ocr_detection(field)
    elif field.type == "BARCODE_QR":
        run_barcode_detection(field)

# With ReadOMR (automatic dispatch):
context = read_omr_processor.process(context)
# Returns: context.omr_response = {"Q1": "A", "Q2": "B", "Name": "John"}
```

### Detection + Interpretation Pipeline
```
Detection Pass:
  Raw Detection → Field-level metrics → Aggregation

Interpretation Pass:
  Raw metrics → Threshold logic → Answers + Confidence
```

## Core Architecture

### Entry Point
```python
# src/processors/detection/processor.py
class ReadOMRProcessor(Processor):
    """Main OMR detection and interpretation orchestrator"""

    def __init__(self, template, ml_model_path=None):
        self.template = template
        self.template_file_runner = TemplateFileRunner(template)
        self.ml_detector = None  # Optional ML fallback
        self.hybrid_strategy = None
```

### Primary Flow
```
ReadOMRProcessor
    ↓
TemplateFileRunner
    ↓
For each field:
    1. Dispatch to field-type runner (bubbles/OCR/barcode)
    2. Run detection pass (extract raw data)
    3. Run interpretation pass (convert to answers)
    ↓
Optional: ML Fallback
    ↓
Return context with omr_response
```

### Key Components

#### 1. TemplateFileRunner
**File**: `src/processors/detection/template_file_runner.py`

The template-level coordinator that:
- Maintains field-type-specific runners (BubblesThresholdFileRunner, OCRFileRunner, BarcodeFileRunner)
- Orchestrates two-pass detection (detection → interpretation)
- Manages directory-level and file-level aggregates
- Uses DetectionRepository for typed storage

```python
class TemplateFileRunner(FileLevelRunner):
    field_detection_type_to_runner = {
        FieldDetectionType.BUBBLES_THRESHOLD: BubblesThresholdFileRunner,
        FieldDetectionType.OCR: OCRFileRunner,
        FieldDetectionType.BARCODE_QR: BarcodeFileRunner,
    }
```

#### 2. Field-Type Runners
Each field type has its own file-level runner:

| Runner | Detection Strategy | Example Use Case |
|--------|-------------------|------------------|
| **BubblesThresholdFileRunner** | Threshold-based darkness detection | Multiple-choice questions |
| **OCRFileRunner** | EasyOCR/Tesseract text recognition | Student name, ID fields |
| **BarcodeFileRunner** | PyZbar decoding | QR codes, barcodes |

#### 3. Detection Repository
**File**: `src/processors/repositories/detection_repository.py`

Type-safe storage for detection results:
```python
self.repository = DetectionRepository()
self.repository.initialize_file(file_path)
# Stores: bubble_fields, ocr_fields, barcode_fields
```

#### 4. ML Fallback System
**File**: `src/processors/detection/ml_detector.py`

Optional YOLO-based bubble detector for low-confidence cases:
```python
class MLBubbleDetector(Processor):
    """YOLO-based bubble detection for low-confidence fallback"""

class HybridDetectionStrategy:
    """Combines traditional + ML detection"""
    confidence_threshold = 0.75  # Trigger ML if below this
```

## Configuration Schema

### Processor Initialization
```python
# Basic initialization
processor = ReadOMRProcessor(template)

# With ML fallback
processor = ReadOMRProcessor(
    template,
    ml_model_path="models/bubble_detector.pt"
)
```

### ML Configuration (in template config)
```json
{
  "ml": {
    "confidence_threshold": 0.75,
    "enable_ml_fallback": true
  }
}
```

## Key Concepts

### 1. Two-Pass Architecture

#### Detection Pass
- Extracts raw data from images (bubble darkness, OCR text, barcode values)
- Computes field-level metrics (mean darkness, std deviation)
- Aggregates statistics (global threshold for bubbles)

#### Interpretation Pass
- Applies threshold logic to raw detections
- Converts to structured answers (Q1 → "A", Name → "John Doe")
- Computes confidence scores
- Detects multi-marked fields

```python
# Detection Pass
raw_detections = {
    "Q1": {"A": 0.85, "B": 0.12, "C": 0.09, "D": 0.11}  # Darkness values
}

# Interpretation Pass (after threshold)
omr_response = {
    "Q1": "A"  # Selected answer
}
```

### 2. Field Detection Types

```python
FieldDetectionType = {
    "BUBBLES_THRESHOLD": "BUBBLES_THRESHOLD",  # Threshold-based bubbles
    "OCR": "OCR",                               # Text recognition
    "BARCODE_QR": "BARCODE_QR",                 # Barcode/QR decoding
}
```

Each field in template specifies its detection type:
```json
{
  "field_label": "Q1",
  "field_detection_type": "BUBBLES_THRESHOLD",
  "bubbleValues": ["A", "B", "C", "D"]
}
```

### 3. Shift Detection Integration

**Special Case**: If ShiftDetectionProcessor already ran and validated shifts:
```python
if context.metadata.get("shift_detection"):
    # Results already populated by ShiftDetectionProcessor
    return context
```

This avoids redundant detection when ML-based shift detection is enabled.

### 4. Image Normalization

```python
# Resize to template dimensions
gray_image, colored_image = ImageUtils.resize_to_dimensions(
    template.template_dimensions,
    input_gray_image,
    colored_image
)

# Normalize images (contrast, brightness)
gray_image, colored_image = ImageUtils.normalize(gray_image, colored_image)
```

**Why Normalize?**
- Ensures consistent detection across different scanners
- Reduces impact of lighting variations
- Matches template dimensions exactly

### 5. Concatenated Response

Template may define custom output labels:
```python
# Raw response: {"Q1": "A", "Q2": "B"}
# Template config: {"Q1": {"outputFormat": "q1_answer"}}
# Concatenated response: {"q1_answer": "A", "Q2": "B"}

concatenated_omr_response = template.get_concatenated_omr_response(
    raw_omr_response
)
```

### 6. ML Fallback Strategy

**HybridDetectionStrategy** decides when to use ML:

```python
def should_use_ml_fallback(context):
    # Check field-level confidence scores
    for field_id, interpretation in context.field_id_to_interpretation.items():
        confidence = interpretation.field_level_confidence_metrics.get(
            "overall_confidence_score", 1.0
        )
        if confidence < 0.75:
            return True  # Trigger ML fallback
    return False
```

**Workflow**:
1. Traditional detection runs first (fast)
2. Check confidence scores
3. If any field has low confidence → run ML detector
4. ML results stored in `context.metadata["ml_detections"]`

## Data Flow

### Input (from Pipeline)
```python
context.gray_image           # Preprocessed + aligned gray image
context.colored_image        # Preprocessed + aligned color image
context.template             # Template with field definitions
context.file_path            # Current file being processed
```

### Processing Steps

1. **Check for Existing Results**
```python
if context.metadata.get("shift_detection"):
    return context  # ShiftDetectionProcessor already populated results
```

2. **Resize Images**
```python
gray_image, colored_image = ImageUtils.resize_to_dimensions(
    template.template_dimensions, input_gray_image, colored_image
)
```

3. **Normalize Images**
```python
gray_image, colored_image = ImageUtils.normalize(gray_image, colored_image)
```

4. **Run Detection + Interpretation**
```python
raw_omr_response = self.template_file_runner.read_omr_and_update_metrics(
    file_path, gray_image, colored_image
)
```

5. **Apply Custom Labels**
```python
concatenated_omr_response = template.get_concatenated_omr_response(
    raw_omr_response
)
```

6. **Extract Metrics**
```python
directory_level_aggregates = self.template_file_runner.get_directory_level_interpretation_aggregates()
file_level_aggregates = directory_level_aggregates["file_wise_aggregates"][file_path]
is_multi_marked = file_level_aggregates["read_response_flags"]["is_multi_marked"]
field_id_to_interpretation = file_level_aggregates["field_id_to_interpretation"]
```

7. **Optional ML Fallback**
```python
if self.hybrid_strategy.should_use_ml_fallback(context):
    self.ml_detector.enable_for_low_confidence()
    context = self.ml_detector.process(context)
    self.ml_detector.disable()
```

### Output (to Context)
```python
context.omr_response                    # {"Q1": "A", "Q2": "B", "Name": "John"}
context.is_multi_marked                 # True if any multi-marked fields
context.field_id_to_interpretation      # Detailed interpretation per field
context.gray_image                      # Normalized gray image
context.colored_image                   # Normalized colored image
context.metadata["raw_omr_response"]    # Before custom labels applied
context.metadata["ml_detections"]       # If ML fallback was used
```

## Edge Cases & Constraints

### 1. Shift Detection Already Ran
```python
shift_detection_meta = context.metadata.get("shift_detection")
if shift_detection_meta:
    # Results already in context, skip ReadOMR
    return context
```

### 2. ML Model Not Available
```python
if ml_model_path and not ml_model_path.exists():
    logger.warning("ML model not found, fallback disabled")
    # Continues with traditional detection only
```

### 3. ML Dependencies Missing
```python
try:
    from ultralytics import YOLO
except ImportError:
    logger.warning("ultralytics not installed, ML disabled")
    # Install with: uv sync --extra ml
```

### 4. No Fields to Process
```python
# TemplateFileRunner handles empty field list gracefully
if len(self.all_fields) == 0:
    return {}  # Empty response
```

### 5. Multi-Marked Detection
```python
# If student marks multiple answers for same question:
is_multi_marked = True  # Flag set in interpretation pass
# Response may contain: "A,B" or "INVALID" based on config
```

## Dependencies

### Core Dependencies
- **Template**: Field definitions, dimensions, configuration
- **TemplateFileRunner**: Orchestrates field-type runners
- **ImageUtils**: Resize, normalize, save operations
- **ProcessingContext**: Unified data container

### Field-Type Dependencies
- **BubblesThresholdFileRunner**: Threshold-based bubble detection
- **OCRFileRunner**: EasyOCR/Tesseract integration
- **BarcodeFileRunner**: PyZbar barcode decoding

### Optional ML Dependencies
- **ultralytics**: YOLO model inference
- **MLBubbleDetector**: ML-based bubble detection
- **HybridDetectionStrategy**: Traditional + ML fusion

## Performance Considerations

### Computational Cost

| Stage | Complexity | Time (typical) |
|-------|-----------|----------------|
| Image resize | O(w × h) | ~10ms |
| Normalization | O(w × h) | ~5ms |
| Bubble detection | O(n_fields × n_bubbles) | ~50-200ms |
| OCR detection | O(n_fields × OCR_time) | ~500-2000ms |
| Barcode detection | O(n_fields) | ~50-100ms |
| ML fallback | O(YOLO_inference) | ~200-500ms |

### Optimization Strategies

1. **Early Exit for Shift Detection**
```python
if context.metadata.get("shift_detection"):
    return context  # Skip redundant detection
```

2. **Lazy ML Model Loading**
```python
self.model = None  # Not loaded until first use
self._load_model()  # Deferred initialization
```

3. **Field-Type Batching**
```python
# TemplateFileRunner groups fields by type
# Processes all bubble fields together → better cache locality
```

4. **Repository Pattern**
```python
# DetectionRepository provides typed storage
# Avoids repeated dictionary lookups
```

## Browser Migration Notes

### Critical Challenges

#### 1. YOLO Model Migration
**Python**: ultralytics YOLO (.pt files)
**Browser**:
- **TensorFlow.js**: Convert with `tensorflowjs_converter`
- **ONNX Runtime Web**: Convert to ONNX format
- **Model size**: 5-50 MB (use quantization)

```javascript
// TensorFlow.js approach
const model = await tf.loadGraphModel('model.json');
const predictions = model.predict(imageTensor);

// ONNX Runtime Web
const session = await ort.InferenceSession.create('model.onnx');
const results = await session.run({input: imageTensor});
```

#### 2. OCR Libraries
**Python**: EasyOCR, Tesseract
**Browser**:
- **Tesseract.js**: Browser-compatible OCR
- **Model download**: ~2-10 MB language files
- **Performance**: Slower than Python (WASM-based)

```javascript
const worker = await Tesseract.createWorker();
await worker.loadLanguage('eng');
await worker.initialize('eng');
const { data: { text } } = await worker.recognize(image);
```

#### 3. Barcode Detection
**Python**: PyZbar (C library wrapper)
**Browser**:
- **@zxing/library**: Pure JavaScript barcode decoder
- **Supports**: QR, Code128, Code39, EAN, UPC, etc.

```javascript
import { BrowserMultiFormatReader } from '@zxing/library';
const codeReader = new BrowserMultiFormatReader();
const result = await codeReader.decodeFromImageElement(imageElement);
```

#### 4. Image Normalization
**Python**: OpenCV `cv2.normalize()`
**Browser**:
- **OpenCV.js**: `cv.normalize()`
- **Canvas API**: Manual histogram adjustment

```javascript
// OpenCV.js
const normalized = new cv.Mat();
cv.normalize(src, normalized, 0, 255, cv.NORM_MINMAX);

// Or Canvas API
const imageData = ctx.getImageData(0, 0, width, height);
// Apply normalization to imageData.data
```

#### 5. Async Processing
**Python**: Synchronous processing
**Browser**: Must be async for UI responsiveness

```javascript
async function processOMR(context) {
  // Resize (fast, can be sync)
  const resized = await resizeImage(context.grayImage);

  // Detection (slow, must be async)
  const response = await runDetection(resized);

  // ML fallback (very slow, use Web Worker)
  if (needsMLFallback(response)) {
    const mlResults = await runMLInWorker(resized);
  }

  return context;
}
```

### Recommended Browser Strategy

```javascript
class ReadOMRProcessor {
  constructor(template, config) {
    this.template = template;
    this.fileRunner = new TemplateFileRunner(template);

    // Lazy load ML model
    this.mlModel = null;
    this.mlReady = false;

    // Load ML model in background
    if (config.enableML) {
      this.loadMLModel(config.mlModelPath);
    }
  }

  async loadMLModel(modelPath) {
    this.mlModel = await tf.loadGraphModel(modelPath);
    this.mlReady = true;
  }

  async process(context) {
    // 1. Resize and normalize
    const { grayImage, coloredImage } = await this.normalizeImages(context);

    // 2. Run detection (field-type dispatch)
    const rawResponse = await this.fileRunner.readOMR(
      context.filePath,
      grayImage,
      coloredImage
    );

    // 3. Check confidence
    const needsML = this.checkConfidence(rawResponse);

    // 4. ML fallback if needed
    if (needsML && this.mlReady) {
      const mlResults = await this.runMLFallback(grayImage);
      this.mergeMLResults(rawResponse, mlResults);
    }

    return context;
  }
}
```

## Related Modules

### Detection Systems
- **Bubble Detection** (`bubbles-threshold/`) - Threshold-based bubble detection
- **OCR Detection** (`ocr/`) - Text recognition
- **Barcode Detection** (`barcode/`) - QR/barcode decoding
- **ML Bubble Detector** (`ml-bubble/`) - YOLO-based fallback
- **Shift Detection** (`shift-detection/`) - ML-based shift correction

### Supporting Modules
- **Template** (`../template/`) - Field definitions and layout
- **Field** (`../field/`) - Individual field types
- **FieldBlock** (`../field-block/`) - Field grouping
- **ProcessingContext** (`../processing-context/`) - Data flow container

## Next Steps

1. Read `flows.md` for detailed execution flow
2. Read `decisions.md` for multi-format dispatch strategy
3. Read `constraints.md` for performance and accuracy limits
4. Read `integration.md` for pipeline integration patterns
