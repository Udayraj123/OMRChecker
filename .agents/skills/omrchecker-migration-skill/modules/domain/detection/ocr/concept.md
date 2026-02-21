# OCR Detection System

**Module**: Domain - Detection - OCR
**Python Reference**: `src/processors/detection/ocr/`
**Last Updated**: 2026-02-21

---

## Overview

The OCR Detection System enables detection and extraction of handwritten or printed text from OMR sheets using Optical Character Recognition. It supports multiple OCR engines (EasyOCR, Tesseract) and provides text extraction from defined scan zones with confidence scoring.

**Key Principles**:
1. **Multi-Engine Support**: Pluggable OCR libraries (EasyOCR primary, Tesseract backup)
2. **Zone-Based Detection**: Extract text from specific scan boxes/regions
3. **Confidence Scoring**: All detections include confidence scores (0.0-1.0)
4. **Text Post-Processing**: Cleanup, filtering, and normalization of detected text
5. **Typed Results**: Strongly-typed detection results for reliability

---

## Architecture Overview

### Component Hierarchy

```
OCRDetectionPass (Orchestrator)
    │
    ├── OCRFieldDetection (Per-Field)
    │       │
    │       ├── OCRField (Field Definition)
    │       │       └── OCRScanBox (Scan Zone)
    │       │
    │       └── EasyOCR / TesseractOCR (Engine)
    │               └── TextOCR (Base Class)
    │
    └── OCRFieldDetectionResult (Typed Result)
            └── OCRDetection[] (Text Detections)
```

### Processing Flow

```
1. Template Definition
   → OCRField with scan_zone defined

2. Detection Pass (OCRDetectionPass)
   → For each OCR field:
      → Extract scan zone from image
      → Run OCR engine (EasyOCR)
      → Post-process detected text
      → Create OCRDetection with confidence
      → Store in OCRFieldDetectionResult

3. Interpretation Pass (OCRInterpretationPass)
   → Map detections to interpretations
   → Concatenate multiple detections if needed
   → Apply empty value logic

4. Drawing Pass (Optional)
   → Draw bounding boxes around detected text
   → Display interpreted text on output image
```

---

## Core Classes

### 1. OCRDetection (TextDetection)

**File**: `src/processors/detection/ocr/detection.py`

Represents a single text detection within an OCR field.

```python
class OCRDetection(TextDetection):
    def __init__(
        self,
        scan_zone_rectangle,
        detected_text,
        bounding_box,
        rotated_rectangle,
        confident_score,
    ) -> None:
        self.library = OCR_LIBS.EASY_OCR  # Which OCR engine was used
        self.scan_zone_rectangle = scan_zone_rectangle
        super().__init__(
            detected_text, bounding_box, rotated_rectangle, confident_score
        )
```

**Fields**:
- `detected_text`: The extracted text string (post-processed)
- `bounding_box`: 4-point polygon surrounding text (absolute coordinates)
- `rotated_rectangle`: Rotated rectangle for tilted text (absolute coordinates)
- `confident_score`: OCR confidence (0.0-1.0)
- `library`: Which OCR library was used (EASY_OCR)
- `scan_zone_rectangle`: The scan zone where text was found

**Coordinate Transformation**:
```python
@staticmethod
def from_scan_zone_detection(scan_zone_rectangle, text_detection: TextDetection):
    # Convert zone-relative coordinates to absolute image coordinates
    zone_start = scan_zone_rectangle[0]
    absolute_bounding_box = MathUtils.shift_points_from_origin(
        zone_start, text_detection.bounding_box
    )
    absolute_rotated_rectangle = MathUtils.shift_points_from_origin(
        zone_start, text_detection.rotated_rectangle
    )
    return OCRDetection(...)
```

**Purpose**: Stores zone-relative detections as absolute coordinates for drawing and interpretation.

---

### 2. OCRFieldDetection (FieldDetection)

**File**: `src/processors/detection/ocr/detection.py`

Performs OCR detection for a single field at runtime.

```python
class OCRFieldDetection(FieldDetection):
    def __init__(self, field: Field, gray_image, colored_image) -> None:
        self.detections: list[OCRDetection] | None = None
        super().__init__(field, gray_image, colored_image)

    def run_detection(self, field: OCRField, gray_image, _colored_image):
        # 1. Get scan zone
        scan_box = field.scan_boxes[0]
        scan_zone_rectangle = scan_box.scan_zone_rectangle

        # 2. Extract image zone
        image_zone = ShapeUtils.extract_image_from_zone_rectangle(
            gray_image, zone_label, scan_zone_rectangle
        )

        # 3. Run OCR
        text_detection = EasyOCR.get_single_text_detection(
            image_zone, confidence_threshold=0.8
        )

        # 4. Convert to absolute coordinates
        self.detections = [
            OCRDetection.from_scan_zone_detection(
                scan_zone_rectangle, text_detection
            )
        ] if text_detection else []

        # 5. Create typed result
        confidence = self.detections[0].confident_score if self.detections else 0.0
        self.result = OCRFieldDetectionResult(
            field_id=field.id,
            field_label=field.field_label,
            detections=self.detections,
            confidence=confidence,
        )
```

**Key Methods**:
- `run_detection()`: Execute OCR on field's scan zone
- Result stored in `self.result` (OCRFieldDetectionResult)
- Result also stored in `self.detections` for backward compatibility

---

### 3. OCRFieldDetectionResult (Typed Model)

**File**: `src/processors/detection/models/detection_results.py`

Strongly-typed detection result replacing dictionary-based aggregates.

```python
@dataclass
class OCRFieldDetectionResult:
    field_id: str
    field_label: str
    detections: list[Any]  # List of OCRDetection
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        return (
            f"OCRFieldDetectionResult(field={self.field_label}, "
            f"detections={len(self.detections)}, confidence={self.confidence:.2f})"
        )
```

**Fields**:
- `field_id`: Unique field identifier
- `field_label`: Human-readable field name
- `detections`: List of OCRDetection objects
- `confidence`: Highest confidence from all detections
- `timestamp`: When detection was performed

**Storage**: Stored in `DetectionRepository` for access during interpretation pass.

---

### 4. OCRField (Field)

**File**: `src/processors/layout/field/ocr_field.py`

Field definition for OCR text fields in template.

```python
class OCRField(Field):
    def __init__(
        self,
        direction,
        empty_value,
        field_block,
        field_detection_type,
        field_label,
        origin,
    ) -> None:
        super().__init__(
            direction,
            empty_value,
            field_block,
            field_detection_type,
            field_label,
            origin,
        )

    def setup_scan_boxes(self, field_block) -> None:
        scan_zone = field_block.scan_zone
        origin = field_block.origin
        field_index = 0
        scan_box = OCRScanBox(field_index, self, origin, scan_zone)
        self.scan_boxes: list[OCRScanBox] = [scan_box]
```

**OCRScanBox**:
```python
class OCRScanBox(ScanBox):
    def __init__(self, field_index, field: OCRField, origin, scan_zone) -> None:
        dimensions = scan_zone["dimensions"]
        margins = scan_zone["margins"]
        super().__init__(field_index, field, origin, dimensions, margins)

        # Zone metadata
        self.zone_description = {"origin": origin, "label": self.name, **scan_zone}

        # Pre-computed scan zone rectangle (with margins)
        self.scan_zone_rectangle = ShapeUtils.compute_scan_zone_rectangle(
            self.zone_description, include_margins=True
        )
```

**Purpose**: Defines the scan zone where OCR will be performed.

---

## OCR Engine Architecture

### Base Class: TextOCR

**File**: `src/processors/detection/ocr/lib/text_ocr.py`

Abstract base class for all OCR engines.

```python
class TextOCR:
    # Character sets for filtering
    digits_set: ClassVar = set(string.digits)
    letters_set: ClassVar = set(string.ascii_letters)
    alphanumeric_set: ClassVar = set(string.ascii_letters + string.digits)

    @staticmethod
    def get_single_text_detection(image, confidence_threshold=0.8):
        raise NotImplementedError

    @staticmethod
    def cleanup_text(text: str) -> str:
        # Strip non-ASCII characters for OpenCV compatibility
        return "".join([c for c in text if ord(c) < 128])

    @staticmethod
    def postprocess_text(
        text, clear_whitespace=False, max_length=None, charset=None
    ) -> str:
        stripped_text = text.strip()
        printable_text = TextOCR.cleanup_text(stripped_text)

        if clear_whitespace:
            cleaned_text = re.sub("\\s{2,}", " ", printable_text)

        if charset is not None:
            cleaned_text = TextOCR.filter_text(cleaned_text, charset)

        if max_length is not None and len(cleaned_text) > max_length:
            cleaned_text = cleaned_text[:max_length]

        return cleaned_text
```

**Character Set Support**:
- `digits_set`: 0-9
- `letters_set`: a-zA-Z
- `lowercase_letters_set`: a-z
- `uppercase_letters_set`: A-Z
- `alphanumeric_set`: 0-9a-zA-Z
- `url_symbols_set`: @$-_.+!*'(),

---

### EasyOCR Engine (Primary)

**File**: `src/processors/detection/ocr/lib/easyocr.py`

Default OCR engine using EasyOCR library.

```python
class EasyOCR(TextOCR):
    reader = None  # Singleton reader instance

    @staticmethod
    def initialize() -> None:
        import easyocr
        # Load model once (GPU-accelerated if available)
        EasyOCR.reader = easyocr.Reader(["en"], gpu=True)

    @staticmethod
    def get_single_text_detection(image, confidence_threshold=0.8, clear_whitespace=True):
        # Get all detections sorted by score
        filtered_texts_with_boxes = EasyOCR.read_texts_with_boxes(
            image, confidence_threshold, sort_by_score=True
        )

        if len(filtered_texts_with_boxes) == 0:
            return None

        # Return highest confidence detection
        box, text, score = filtered_texts_with_boxes[0]

        if score <= confidence_threshold:
            return None

        return EasyOCR.convert_to_text_detection(box, text, score, clear_whitespace)

    @staticmethod
    def read_texts_with_boxes(image, confidence_threshold=0.8, sort_by_score=True):
        # Lazy load reader
        if EasyOCR.reader is None:
            EasyOCR.initialize()

        # Run OCR
        text_results = EasyOCR.reader.readtext(image)

        # Filter by confidence
        filtered = [
            (box, text, score)
            for (box, text, score) in text_results
            if score >= confidence_threshold
        ]

        # Sort by score (descending)
        if sort_by_score:
            return sorted(filtered, key=operator.itemgetter(2), reverse=True)
        return filtered
```

**Key Features**:
- **Lazy Loading**: Reader initialized on first use
- **GPU Acceleration**: Uses GPU if available
- **Confidence Filtering**: Only returns detections above threshold (default 0.8)
- **Sorting**: Returns highest confidence detection first
- **Text Post-Processing**: Cleanup and whitespace handling

---

### Tesseract Engine (Future)

**File**: `src/processors/detection/ocr/lib/tesseract.py`

Placeholder for Tesseract OCR integration.

```python
class TesseractOCR(TextOCR):
    @staticmethod
    def get_single_text_detection(image, confidence_threshold=0.8):
        # TODO: pytesseract.image_to_string
        raise NotImplementedError
```

**Status**: Not yet implemented
**Planned Use**: Alternative OCR engine for fallback or specific use cases

---

## Detection Pass

### OCRDetectionPass

**File**: `src/processors/detection/ocr/detection_pass.py`

Orchestrates OCR detection across all OCR fields in a template.

```python
class OCRDetectionPass(FieldTypeDetectionPass):
    def __init__(self, *args, repository: DetectionRepository, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.repository = repository

    def get_field_detection(
        self, field: Field, gray_image, colored_image
    ) -> OCRFieldDetection:
        return OCRFieldDetection(field, gray_image, colored_image)

    def update_field_level_aggregates_on_processed_field_detection(
        self, field: Field, field_detection: OCRFieldDetection
    ) -> None:
        super().update_field_level_aggregates_on_processed_field_detection(
            field, field_detection
        )

        # Save to repository for interpretation pass
        self.repository.save_ocr_field(field.id, field_detection.result)

        self.insert_field_level_aggregates({"detections": field_detection.detections})
```

**Responsibilities**:
1. Create `OCRFieldDetection` for each OCR field
2. Save results to `DetectionRepository`
3. Track field-level statistics
4. Aggregate file-level metadata

---

## Interpretation Pass

### OCRInterpretation

**File**: `src/processors/detection/ocr/interpretation.py`

Interprets a single OCR detection.

```python
class OCRInterpretation(BaseInterpretation):
    def __init__(self, detection) -> None:
        super().__init__(detection)
```

**Inherits From**: `BaseInterpretation`
- `is_attempted`: Whether text was detected
- `get_value()`: Returns detected text

---

### OCRFieldInterpretation

**File**: `src/processors/detection/ocr/interpretation.py`

Interprets all detections for a field and combines them.

```python
class OCRFieldInterpretation(FieldInterpretation):
    def __init__(self, *args, **kwargs) -> None:
        self.interpretations: list[OCRInterpretation] = None
        super().__init__(*args, **kwargs)

    def get_field_interpretation_string(self):
        marked_interpretations = [
            interpretation.get_value()
            for interpretation in self.interpretations
            if interpretation.is_attempted
        ]

        # Empty value logic
        if len(marked_interpretations) == 0:
            return self.empty_value

        # Concatenate all detected texts
        return "".join(marked_interpretations)

    def run_interpretation(
        self,
        field: Field,
        file_level_detection_aggregates,
        file_level_interpretation_aggregates,
    ) -> None:
        self.initialize_from_file_level_aggregates(...)
        self.update_common_interpretations()

    def initialize_from_file_level_aggregates(
        self, field, file_level_detection_aggregates, _
    ) -> None:
        # Retrieve from repository
        ocr_fields = file_level_detection_aggregates["ocr_fields"]
        ocr_result = ocr_fields[field.field_label]
        detections = ocr_result.detections

        # Map detections to interpretations
        self.interpretations = [
            OCRInterpretation(detection) for detection in detections
        ]
```

**Multi-Mark Detection**:
```python
def update_common_interpretations(self) -> None:
    marked_interpretations = [
        interpretation.get_value()
        for interpretation in self.interpretations
        if interpretation.is_attempted
    ]
    self.is_attempted = len(marked_interpretations) > 0
    self.is_multi_marked = len(marked_interpretations) > 1
```

**Concatenation Logic**:
- If 0 detections: Return `empty_value`
- If 1+ detections: Concatenate all detected texts (no spaces)
- Future: Support configurable concatenation (space-separated, custom delimiter)

---

## Visualization

### OCRFieldInterpretationDrawing

**File**: `src/processors/detection/ocr/interpretation_drawing.py`

Draws OCR detections and interpretations on output images.

```python
class OCRFieldInterpretationDrawing(FieldInterpretationDrawing):
    def draw_field_interpretation(
        self, marked_image, image_type, evaluation_meta, evaluation_config
    ) -> None:
        if len(field_interpretation.interpretations) == 0:
            return

        # Draw all individual bounding boxes
        all_bounding_box_points = []
        for interpretation in field_interpretation.interpretations:
            bounding_box = interpretation.text_detection.bounding_box
            DrawingUtils.draw_contour(marked_image, bounding_box, color=CLR_BLACK)
            all_bounding_box_points.extend(bounding_box)

        # Draw combined bounding box
        combined_bounding_box, _ = MathUtils.get_bounding_box_of_points(
            all_bounding_box_points
        )
        DrawingUtils.draw_contour(
            marked_image, combined_bounding_box, color=combined_bounding_box_color
        )

        # Draw interpreted text above bounding box
        interpreted_text = field_interpretation.get_field_interpretation_string()
        DrawingUtils.draw_text_responsive(
            marked_image, interpreted_text, text_position, color=CLR_BLACK, thickness=3
        )
```

**Drawing Elements**:
1. Individual bounding boxes (black) for each detection
2. Combined bounding box encompassing all detections
3. Interpreted text label above bounding box
4. Color-coded based on evaluation result (if evaluation enabled)

---

## Configuration

### Default OCR Configuration

**Current Implementation**:
```python
# Hardcoded in detection.py
confidence_threshold = 0.8
```

**Future Configuration** (TODO):
```python
# Field-level config in template
{
  "fieldType": "OCR",
  "fieldLabel": "studentName",
  "ocrConfig": {
    "engine": "EASY_OCR",  # or "TESSERACT"
    "confidenceThreshold": 0.8,
    "languageCode": "en",
    "charset": "alphanumeric",  # digits, letters, alphanumeric, url_symbols
    "clearWhitespace": true,
    "maxLength": 50,
    "concatenation": {
      "enabled": true,
      "delimiter": "",  # or " ", "-", etc.
      "order": "topToBottom"  # or "leftToRight", "confidence"
    }
  }
}
```

---

## Data Flow

### 1. Template Definition → Field Setup

```
template.json
  └── fieldBlocks[]
        └── scanZone: { dimensions, margins }
        └── labelsPath: ["NAME"]
              │
              ├── fieldDetectionType: "OCR"
              └── emptyValue: ""

⬇️ Template.setup()

OCRField
  └── OCRScanBox
        └── scan_zone_rectangle (pre-computed with margins)
```

### 2. Detection Pass

```
ReadOMRProcessor.process(context)
  └── TemplateFileRunner.process_file(file_path)
        └── OCRDetectionPass.process_all_fields()
              └── For each OCR field:
                    OCRFieldDetection(field, gray_image, colored_image)
                      ├── Extract scan zone image
                      ├── EasyOCR.get_single_text_detection()
                      │     ├── reader.readtext(image)
                      │     ├── Filter by confidence >= 0.8
                      │     ├── Sort by score descending
                      │     └── Return highest confidence
                      ├── Convert to absolute coordinates
                      └── Create OCRFieldDetectionResult
                            ├── Save to DetectionRepository
                            └── Store in file_level_aggregates["ocr_fields"]
```

### 3. Interpretation Pass

```
OCRInterpretationPass.process_all_fields()
  └── For each OCR field:
        OCRFieldInterpretation(field, aggregates)
          ├── Load from repository: ocr_fields[field_label]
          ├── Map detections → interpretations
          ├── Concatenate detected texts
          └── Set is_attempted, is_multi_marked
```

### 4. Output

```
OCRFieldInterpretationDrawing.draw()
  ├── Draw individual bounding boxes
  ├── Draw combined bounding box
  └── Draw interpreted text label

CSV Output:
  field_label → interpreted_text
```

---

## Browser Migration Notes

### OCR Engine Migration

**Python (EasyOCR)**:
```python
import easyocr
reader = easyocr.Reader(["en"], gpu=True)
results = reader.readtext(image)
```

**Browser (Tesseract.js)**:
```typescript
import Tesseract from 'tesseract.js';

const worker = await Tesseract.createWorker('eng');
const { data } = await worker.recognize(imageData);
const text = data.text;
const confidence = data.confidence / 100; // 0-1 range

// With bounding boxes
const words = data.words.map(word => ({
  text: word.text,
  confidence: word.confidence / 100,
  bbox: word.bbox,
}));
```

**Migration Considerations**:
1. **EasyOCR → Tesseract.js**: Tesseract.js is the standard browser OCR library
2. **GPU Acceleration**: Not available in browser (CPU-only)
3. **Model Loading**: Tesseract.js downloads language models (~2-4MB) on first use
4. **Performance**: Slower than EasyOCR, expect 2-5 seconds per field
5. **Caching**: Cache worker instances to avoid re-initialization

### Coordinate System

**Python (NumPy arrays)**:
```python
bounding_box = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]  # 4-point polygon
```

**Browser (JavaScript arrays)**:
```typescript
interface Point {
  x: number;
  y: number;
}

interface BoundingBox {
  points: Point[]; // 4 corner points
  x0: number;      // Top-left x
  y0: number;      // Top-left y
  x1: number;      // Bottom-right x
  y1: number;      // Bottom-right y
}
```

### Text Post-Processing

**Python**:
```python
cleaned = TextOCR.postprocess_text(
    text,
    clear_whitespace=True,
    max_length=50,
    charset=TextOCR.alphanumeric_set
)
```

**Browser**:
```typescript
function postprocessText(
  text: string,
  options: {
    clearWhitespace?: boolean;
    maxLength?: number;
    charset?: Set<string>;
  }
): string {
  let cleaned = text.trim();

  // Remove non-ASCII
  cleaned = cleaned.replace(/[^\x00-\x7F]/g, '');

  if (options.clearWhitespace) {
    cleaned = cleaned.replace(/\s{2,}/g, ' ');
  }

  if (options.charset) {
    cleaned = [...cleaned].filter(c => options.charset.has(c)).join('');
  }

  if (options.maxLength && cleaned.length > options.maxLength) {
    cleaned = cleaned.substring(0, options.maxLength);
  }

  return cleaned;
}
```

### TypeScript Interfaces

```typescript
interface OCRDetection {
  detectedText: string;
  boundingBox: Point[];
  rotatedRectangle: Point[];
  confidenceScore: number;
  library: 'TESSERACT_JS';
  scanZoneRectangle: Point[];
}

interface OCRFieldDetectionResult {
  fieldId: string;
  fieldLabel: string;
  detections: OCRDetection[];
  confidence: number;
  timestamp: Date;
}

interface OCRConfig {
  engine: 'TESSERACT_JS';
  confidenceThreshold: number;
  languageCode: string;
  charset?: 'digits' | 'letters' | 'alphanumeric' | 'url_symbols';
  clearWhitespace: boolean;
  maxLength?: number;
}
```

---

## Related Documentation

- **Field Entity**: `modules/domain/field/concept.md`
- **FieldBlock Entity**: `modules/domain/field-block/concept.md`
- **ReadOMR Processor**: `modules/domain/detection/concept.md`
- **Barcode Detection**: `modules/domain/detection/barcode/concept.md`
- **Bubble Detection**: `modules/domain/detection/bubbles-threshold/concept.md`

---

## Summary

The OCR Detection System provides:

1. **Text Extraction**: Detect handwritten/printed text from scan zones
2. **Multi-Engine**: Support for EasyOCR (Python) and Tesseract.js (Browser)
3. **Confidence Scoring**: All detections include confidence scores for reliability
4. **Post-Processing**: Text cleanup, filtering, and normalization
5. **Typed Results**: Strongly-typed models for detection results
6. **Integration**: Seamless integration with ReadOMR and interpretation pipeline

**Default Setup**: EasyOCR with 0.8 confidence threshold, English language, GPU-accelerated when available.
