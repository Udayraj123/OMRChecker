# Barcode Detection - Core Concepts

**Status**: Production Detection System
**Python Reference**: `src/processors/detection/barcode/`
**Browser Compatibility**: @zxing/library (ZXing for JavaScript)

---

## Overview

The **Barcode Detection System** provides text extraction from 1D/2D barcodes and QR codes in OMR sheets. It uses **PyZBar** (Python wrapper for ZBar library) to decode various barcode formats, extracting text data for fields like student IDs, roll numbers, serial numbers, or any other encoded information.

This is one of three primary detection methods in OMRChecker:
- **BUBBLES_THRESHOLD**: Classical bubble detection (most common)
- **OCR**: Text recognition from images
- **BARCODE_QR**: Barcode/QR code decoding (this module)

---

## Architecture

### Component Hierarchy

```
BarcodeFileRunner (Orchestrator)
    ├─> BarcodeDetectionPass (Detection Phase)
    │   └─> BarcodeFieldDetection (per field)
    │       └─> PyZBar.get_single_text_detection()
    │           └─> pyzbar.decode()
    │               └─> BarcodeDetection (result)
    │
    └─> BarcodeInterpretationPass (Interpretation Phase)
        └─> BarcodeFieldInterpretation (per field)
            └─> BarcodeInterpretation (per detection)
                └─> detected_text (string output)
```

### Key Components

#### 1. **BarcodeDetection** (`detection.py`)
Core detection class that wraps PyZBar results with absolute coordinates.

```python
class BarcodeDetection(TextDetection):
    """Single barcode detection result with position and confidence."""

    def __init__(self, scan_zone_rectangle, detected_text,
                 bounding_box, rotated_rectangle, confident_score):
        self.library = BARCODE_LIBS.PYZBAR
        self.scan_zone_rectangle = scan_zone_rectangle
        self.detected_text = detected_text
        self.bounding_box = bounding_box           # 4 corner points
        self.rotated_rectangle = rotated_rectangle  # Actual barcode polygon
        self.confident_score = confident_score      # 0.0 - 1.0
```

**Key Method**: `from_scan_zone_detection()`
- Converts relative coordinates (within scan zone) to absolute coordinates (on full image)
- Shifts bounding box and rotated rectangle by zone origin

#### 2. **BarcodeFieldDetection** (`detection.py`)
Performs detection for a single barcode field.

```python
class BarcodeFieldDetection(FieldDetection):
    """Detects barcode within field's scan zone."""

    def run_detection(self, field, gray_image, colored_image):
        # 1. Extract scan zone from field definition
        scan_box = field.scan_boxes[0]
        scan_zone_rectangle = scan_box.scan_zone_rectangle

        # 2. Crop image to scan zone
        image_zone = extract_image_from_zone_rectangle(
            gray_image, scan_zone_rectangle
        )

        # 3. Decode barcode with PyZBar
        text_detection = PyZBar.get_single_text_detection(
            image_zone, confidence_threshold=0.8
        )

        # 4. Convert to absolute coordinates
        if text_detection:
            self.detections = [
                BarcodeDetection.from_scan_zone_detection(
                    scan_zone_rectangle, text_detection
                )
            ]

        # 5. Create typed result for repository
        self.result = BarcodeFieldDetectionResult(
            field_id=field.id,
            field_label=field.field_label,
            detections=self.detections
        )
```

#### 3. **PyZBar Library Wrapper** (`lib/pyzbar.py`)
Lazy-loaded wrapper around PyZBar C library.

```python
class PyZBar(TextBarcode):
    """Singleton wrapper for PyZBar library."""

    decode_barcode = None  # Lazy-loaded function

    @staticmethod
    def initialize():
        """Lazy load PyZBar to avoid import errors if not installed."""
        from pyzbar.pyzbar import decode as decode_barcode
        PyZBar.decode_barcode = decode_barcode

    @staticmethod
    def get_single_text_detection(image, confidence_threshold=0.8):
        """Decode highest-confidence barcode in image."""
        # Returns TextDetection or None
```

**Supported Operations**:
- `get_single_text_detection()`: Returns best barcode (sorted by score)
- `get_all_text_detections()`: Returns all barcodes found (for multi-barcode fields)

#### 4. **BarcodeFieldInterpretation** (`interpretation.py`)
Interprets detected barcode text into final field value.

```python
class BarcodeFieldInterpretation(FieldInterpretation):
    """Interprets barcode detection results."""

    def get_field_interpretation_string(self):
        # Concatenate all detected texts
        marked_interpretations = [
            interpretation.get_value()
            for interpretation in self.interpretations
            if interpretation.is_attempted
        ]

        if len(marked_interpretations) == 0:
            return self.empty_value  # "" or configured value

        return "".join(marked_interpretations)
```

**Key Properties**:
- `is_attempted`: True if any barcode detected
- `is_multi_marked`: True if multiple barcodes detected (usually invalid)
- `empty_value`: Return value when no barcode found

#### 5. **Detection Result Models** (`models/detection_results.py`)
Strongly-typed result container.

```python
@dataclass
class BarcodeFieldDetectionResult:
    """Typed result for barcode field detection."""

    field_id: str
    field_label: str
    detections: list[BarcodeDetection]
    timestamp: datetime = field(default_factory=datetime.now)

    def __repr__(self):
        return (
            f"BarcodeFieldDetectionResult("
            f"field={self.field_label}, "
            f"detections={len(self.detections)})"
        )
```

**Benefits over dictionaries**:
- Type safety with IDE autocomplete
- Auto-generated timestamps
- Readable debug output
- Integration with DetectionRepository

---

## Data Flow

### End-to-End Pipeline

```
Template Definition (JSON)
    ↓
    fieldBlocks: {
        "RollNumber": {
            "fieldDetectionType": "BARCODE_QR",
            "scanZone": {
                "origin": [100, 50],
                "dimensions": [300, 80],
                "margins": [5, 5, 5, 5]
            }
        }
    }
    ↓
Field Object Creation
    ↓
    BarcodeField
        ├─> scan_boxes: [BarcodeScanBox]
        └─> scan_zone_rectangle: [(100,50), (400,50), (400,130), (100,130)]
    ↓
Detection Phase (BarcodeDetectionPass)
    ↓
    For each BarcodeField:
        1. Extract scan zone image
        2. PyZBar.decode(image)
        3. Parse results (text, bbox, polygon, score)
        4. Convert to absolute coordinates
        5. Save to DetectionRepository
    ↓
Interpretation Phase (BarcodeInterpretationPass)
    ↓
    For each BarcodeField:
        1. Load detections from repository
        2. Create BarcodeInterpretation objects
        3. Check multi-marking
        4. Generate final string value
    ↓
Output
    ↓
    {
        "RollNumber": "STU-2024-12345",
        "is_multi_marked": false
    }
```

---

## Field Definition Schema

### Template JSON Structure

```json
{
    "fieldBlocks": {
        "RollNumber": {
            "fieldDetectionType": "BARCODE_QR",
            "emptyValue": "",
            "scanZone": {
                "origin": [100, 50],
                "dimensions": [300, 80],
                "margins": [5, 5, 5, 5]
            }
        },
        "SerialQR": {
            "fieldDetectionType": "BARCODE_QR",
            "emptyValue": "NOT_FOUND",
            "scanZone": {
                "origin": [500, 50],
                "dimensions": [150, 150],
                "margins": [10, 10, 10, 10]
            }
        }
    }
}
```

### Scan Zone Anatomy

```
┌─────────────────────────────────────┐
│ Margins (5px padding)               │
│  ┌───────────────────────────────┐  │
│  │                               │  │
│  │   Barcode Scan Area           │  │
│  │   (300 x 80 px)               │  │
│  │                               │  │
│  │   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓             │  │
│  │   ▓ Barcode Data ▓             │  │
│  │   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓             │  │
│  │                               │  │
│  └───────────────────────────────┘  │
│                                     │
└─────────────────────────────────────┘
     Origin: (100, 50)
     Total dimensions: 310 x 90 (with margins)
```

**Why margins?**
- Handles alignment errors (small shifts)
- Ensures full barcode visible even with slight rotation
- Provides buffer for PyZBar edge detection

---

## Integration Points

### 1. Template Layout (`src/processors/layout/field_block/base.py`)

```python
FIELD_CLASS_FOR_DETECTION_TYPE = {
    FieldDetectionType.BUBBLES_THRESHOLD: BubbleField,
    FieldDetectionType.OCR: OCRField,
    FieldDetectionType.BARCODE_QR: BarcodeField,  # ← Registered here
}
```

### 2. Detection Repository (`src/processors/repositories/detection_repository.py`)

```python
class DetectionRepository:
    """Centralized storage for all detection results."""

    def save_barcode_field(self, field_id: str,
                          result: BarcodeFieldDetectionResult):
        """Save barcode detection result."""
        self.barcode_fields[field_id] = result

    def get_barcode_field(self, field_id: str) -> BarcodeFieldDetectionResult:
        """Retrieve barcode detection result."""
        return self.barcode_fields[field_id]
```

**Benefits**:
- Decouples detection from interpretation
- Enables result caching
- Simplifies testing (mock repository)

### 3. Template File Runner (`src/processors/detection/template_file_runner.py`)

```python
RUNNER_CLASS_FOR_FIELD_TYPE = {
    FieldDetectionType.BUBBLES_THRESHOLD: BubblesFileRunner,
    FieldDetectionType.OCR: OCRFileRunner,
    FieldDetectionType.BARCODE_QR: BarcodeFileRunner,  # ← Orchestrator
}
```

### 4. ReadOMR Processor (`src/processors/detection/processor.py`)

```python
class ReadOMRProcessor(Processor):
    """Main OMR detection orchestrator."""

    def process(self, context):
        # Runs detection and interpretation for ALL field types:
        # - Bubbles (threshold-based)
        # - OCR (text recognition)
        # - Barcodes (decoding)  ← Includes this module
```

---

## Barcode Field Entity

### BarcodeField Class (`src/processors/layout/field/barcode_field.py`)

```python
class BarcodeField(Field):
    """Field that contains a barcode or QR code."""

    def setup_scan_boxes(self, field_block):
        """Create single scan zone for barcode detection."""
        scan_zone = field_block.scan_zone
        origin = field_block.origin

        # Single scan box per barcode field
        scan_box = BarcodeScanBox(0, self, origin, scan_zone)
        self.scan_boxes = [scan_box]
```

**Key Difference from BubbleField**:
- BubbleField: Multiple scan boxes (grid of bubbles)
- BarcodeField: **Single scan box** (entire barcode region)

### BarcodeScanBox Class

```python
class BarcodeScanBox(ScanBox):
    """Single scan zone for barcode detection."""

    def __init__(self, field_index, field, origin, scan_zone):
        dimensions = scan_zone["dimensions"]
        margins = scan_zone["margins"]
        super().__init__(field_index, field, origin, dimensions, margins)

        # Pre-compute scan zone rectangle for reuse
        self.scan_zone_rectangle = ShapeUtils.compute_scan_zone_rectangle(
            self.zone_description, include_margins=True
        )
```

**Pre-computed Properties**:
- `scan_zone_rectangle`: 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
- `zone_description`: Full zone metadata (origin, dimensions, margins)

---

## PyZBar Formats Supported

### 1D Barcodes
- **CODE-128**: Variable length, high density (common for roll numbers)
- **CODE-39**: Alphanumeric, widely used
- **EAN-13/UPC-A**: Product barcodes (13 digits)
- **EAN-8/UPC-E**: Short product barcodes (8 digits)
- **Interleaved 2 of 5**: Numeric only

### 2D Barcodes
- **QR Code**: Most versatile, error correction (Level L/M/Q/H)
- **Data Matrix**: Compact, high data density
- **PDF417**: Multi-row barcode, ID cards

### Detection Quality Metadata

```python
# PyZBar result structure
result.data       # bytes: b'STU-2024-12345'
result.type       # str: 'QRCODE', 'CODE128', etc.
result.quality    # int: 0-100 (confidence score)
result.rect       # Rectangle: (x, y, width, height)
result.polygon    # List[Point]: Actual barcode corners (for rotation)
```

**Quality Score Interpretation**:
- `100`: Perfect decode, high contrast
- `50-99`: Good decode, acceptable
- `< 50`: Low confidence, may be incorrect

**Browser Migration Note**: PyZBar quality mapping differs from @zxing/library confidence scores. ZXing returns boolean `successful` rather than numeric quality.

---

## Configuration

### Tuning Config (Optional)

```python
# Currently using hardcoded threshold
confidence_threshold = 0.8  # 80% minimum quality

# Future enhancement: Make configurable
{
    "detection": {
        "barcode": {
            "confidence_threshold": 0.8,
            "retry_with_preprocessing": True,
            "max_barcodes_per_field": 1
        }
    }
}
```

### Empty Value Handling

```python
# Field-level configuration
field_block["emptyValue"] = ""         # Default: empty string
field_block["emptyValue"] = "ABSENT"   # Custom value for missing barcode
field_block["emptyValue"] = "0000"     # Numeric default
```

**Use Cases**:
- **""** (empty): Optional barcode fields
- **"ABSENT"**: Tracking sheet completeness
- **"0000"**: Numeric fields requiring default value

---

## Error Handling

### No Barcode Detected

```python
# Detection Phase
text_detection = PyZBar.get_single_text_detection(image_zone)
if text_detection is None:
    self.detections = []  # Empty list

# Interpretation Phase
if len(self.interpretations) == 0:
    logger.warning(f"No Barcode detection for field: {field.id}")
    return self.empty_value
```

### Multiple Barcodes (Multi-Marking)

```python
# Currently only returns single best barcode
# Multi-barcode support exists but not enabled

# Future: Enable multi-barcode fields
text_detections = PyZBar.get_all_text_detections(image_zone)
self.is_multi_marked = len(text_detections) > 1
```

### Low Confidence

```python
# Current: Uses hardcoded threshold
if result.quality < confidence_threshold:
    return None  # Treated as no detection

# Future: Return with warning flag
return {
    "value": detected_text,
    "confidence": result.quality,
    "needs_review": result.quality < 0.9
}
```

---

## Drawing & Visualization

### Debug Output (`interpretation_drawing.py`)

```python
class BarcodeFieldInterpretationDrawing(FieldInterpretationDrawing):
    """Draws barcode detection results on output image."""

    def draw_field_interpretation(self, marked_image, ...):
        # 1. Draw individual bounding boxes (per detection)
        for interpretation in field_interpretation.interpretations:
            bounding_box = interpretation.text_detection.bounding_box
            DrawingUtils.draw_contour(marked_image, bounding_box,
                                     color=CLR_BLACK)

        # 2. Draw combined bounding box (all detections)
        combined_bounding_box = get_bounding_box_of_points(
            all_bounding_box_points
        )
        DrawingUtils.draw_contour(marked_image, combined_bounding_box,
                                 color=verdict_color)  # Green/Red based on evaluation

        # 3. Draw interpreted text above bounding box
        interpreted_text = field_interpretation.get_field_interpretation_string()
        DrawingUtils.draw_text_responsive(
            marked_image, interpreted_text, text_position,
            color=CLR_BLACK, thickness=3
        )
```

**Visual Output Example**:
```
┌─────────────────────┐
│  STU-2024-12345 ←── Interpreted text (above box)
│  ┌───────────────┐  │
│  │ ▓▓▓▓▓▓▓▓▓▓▓▓  │  │ ← Bounding box
│  │ ▓ Barcode ▓   │  │
│  │ ▓▓▓▓▓▓▓▓▓▓▓▓  │  │
│  └───────────────┘  │
└─────────────────────┘
```

**Color Coding** (with evaluation):
- **Green**: Correct barcode (matches answer key)
- **Red**: Incorrect barcode
- **Black**: No evaluation (detection only mode)

---

## Browser Migration Strategy

### Python (Current)

```python
from pyzbar.pyzbar import decode

# Decode barcode
results = decode(image)  # Returns list of Decoded objects
for result in results:
    print(result.data.decode('utf-8'))
```

### JavaScript (Browser Target)

```typescript
import { BrowserQRCodeReader, BrowserBarcodeReader } from '@zxing/library';

// QR Code Reader
const qrReader = new BrowserQRCodeReader();
const result = await qrReader.decodeFromImageElement(imageElement);

// Multi-format Barcode Reader
const barcodeReader = new BrowserBarcodeReader();
const result = await barcodeReader.decodeFromImageElement(imageElement);

// Result structure
{
    text: "STU-2024-12345",           // Decoded text
    format: BarcodeFormat.QR_CODE,    // Format enum
    resultPoints: [...],               // Corner points
    timestamp: 1234567890
}
```

### Migration Mapping

| Python (PyZBar) | JavaScript (@zxing/library) | Notes |
|-----------------|----------------------------|-------|
| `decode(image)` | `decodeFromImageElement()` | Async in browser |
| `result.data` | `result.text` | Different property name |
| `result.quality` | N/A | ZXing doesn't provide quality score |
| `result.type` | `result.format` | Enum-based in ZXing |
| `result.polygon` | `result.resultPoints` | Different structure |

### Browser-Specific Adaptations

#### 1. Image Source Handling

```typescript
// Option 1: From Canvas
const canvas = document.getElementById('scanCanvas') as HTMLCanvasElement;
const result = await reader.decodeFromCanvas(canvas);

// Option 2: From ImageData
const imageData = ctx.getImageData(0, 0, width, height);
const result = await reader.decodeFromImageData(imageData);

// Option 3: From File input
const file = fileInput.files[0];
const result = await reader.decodeFromImageUrl(URL.createObjectURL(file));
```

#### 2. No Quality Score

```typescript
// PyZBar provides quality score (0-100)
// ZXing only returns success/failure

// Workaround: Use try-catch for confidence
try {
    const result = await reader.decode(image);
    return { text: result.text, confidence: 1.0 };  // Success = 100%
} catch (e) {
    return { text: null, confidence: 0.0 };  // Failure = 0%
}
```

#### 3. Format Selection

```typescript
import { BarcodeFormat, DecodeHintType } from '@zxing/library';

// Limit to specific formats (performance optimization)
const hints = new Map();
hints.set(DecodeHintType.POSSIBLE_FORMATS, [
    BarcodeFormat.QR_CODE,
    BarcodeFormat.CODE_128,
    BarcodeFormat.EAN_13
]);

const reader = new BrowserMultiFormatReader(hints);
```

#### 4. Coordinate Conversion

```typescript
// PyZBar polygon → ZXing resultPoints
// PyZBar: List[Point(x, y)]
// ZXing: ResultPoint[] with getX()/getY() methods

function convertToPolygon(resultPoints: ResultPoint[]): [number, number][] {
    return resultPoints.map(pt => [pt.getX(), pt.getY()]);
}
```

---

## Performance Considerations

### Python (PyZBar)
- **Speed**: Very fast (C library), ~10-50ms per barcode
- **Memory**: Low overhead, processes single image region
- **Threading**: Thread-safe, can parallelize across fields

### Browser (@zxing/library)
- **Speed**: Slower (JavaScript), ~50-200ms per barcode
- **Memory**: Higher overhead (WASM/JS), monitor heap usage
- **Threading**: Use Web Workers for parallel processing

### Optimization Strategies

```typescript
// 1. Web Worker Pool
const workerPool = new BarcodeWorkerPool(4);  // 4 workers
const results = await workerPool.decodeFields(barcodeFields);

// 2. Canvas optimization
const offscreenCanvas = new OffscreenCanvas(width, height);
const ctx = offscreenCanvas.getContext('2d');
// Decode from offscreen canvas (faster)

// 3. Format hints (reduce search space)
const hints = new Map();
hints.set(DecodeHintType.POSSIBLE_FORMATS, [BarcodeFormat.QR_CODE]);
hints.set(DecodeHintType.TRY_HARDER, false);  // Faster, less thorough

// 4. Image preprocessing (improve detection rate)
// Resize scan zone to optimal size (300-500px width)
// Apply sharpening filter
// Increase contrast
```

---

## Testing Strategy

### Unit Tests

```python
# test_barcode_detection.py
def test_barcode_detection_success():
    field = create_barcode_field("RollNumber")
    image = load_test_image("barcode_sample.png")

    detection = BarcodeFieldDetection(field, image, image)

    assert len(detection.detections) == 1
    assert detection.detections[0].detected_text == "STU-2024-12345"
    assert detection.detections[0].confident_score >= 0.8

def test_barcode_detection_no_barcode():
    field = create_barcode_field("RollNumber")
    image = load_test_image("blank_image.png")

    detection = BarcodeFieldDetection(field, image, image)

    assert len(detection.detections) == 0
```

### Integration Tests

```python
# test_barcode_integration.py
def test_end_to_end_barcode_flow():
    template = load_template("barcode_template.json")
    scan = load_scan("student_sheet.png")

    runner = TemplateFileRunner(template)
    response = runner.read_omr_and_update_metrics(
        "test.png", scan, scan
    )

    assert response["RollNumber"] == "STU-2024-12345"
    assert response["SerialQR"] == "BATCH-A-001"
```

---

## Summary

**Barcode Detection System** provides:
1. **PyZBar Integration**: Wrapper for ZBar C library with lazy loading
2. **Coordinate Management**: Automatic conversion from scan-zone to absolute coordinates
3. **Typed Results**: Strongly-typed models with DetectionRepository integration
4. **Visualization**: Debug drawing with bounding boxes and interpreted text
5. **Browser Migration**: Clear path to @zxing/library with documented adaptations
6. **Single Responsibility**: Each class has focused purpose (detection, interpretation, drawing)

**Key Design Patterns**:
- **Singleton**: PyZBar lazy-loaded once per process
- **Strategy Pattern**: TextBarcode base class allows future library alternatives
- **Repository Pattern**: Centralized detection storage
- **Typed Models**: BarcodeFieldDetectionResult for type safety

**Next Steps**:
- See `flows.md` for detailed detection algorithms
- See `decisions.md` for format selection logic
- See `constraints.md` for performance limits and edge cases
- See `integration.md` for ReadOMR pipeline integration
