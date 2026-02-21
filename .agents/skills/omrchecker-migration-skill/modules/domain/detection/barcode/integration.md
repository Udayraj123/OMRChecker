# Barcode Detection - Integration Points

**Module**: Barcode Detection
**Python Reference**: `src/processors/detection/barcode/`
**Focus**: Integration with ReadOMR pipeline, template system, repository, evaluation

---

## Integration Overview

The Barcode Detection system integrates with multiple components in the OMRChecker pipeline:

```
Template Definition (JSON)
    ↓
Template Layout System
    ↓
ReadOMR Processor
    ↓
Template File Runner
    ↓
Barcode File Runner ← THIS MODULE
    ├─> Detection Pass
    └─> Interpretation Pass
    ↓
Detection Repository
    ↓
Response Generation
    ↓
Evaluation System
    ↓
CSV Output
```

---

## Integration 1: Template System

### Entry Point: Template JSON

```json
{
    "pageDimensions": [1200, 1800],
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

### Template Layout Processing

**File**: `src/processors/layout/template_layout.py`

```python
# Template layout detects barcode fields
def parse_field_blocks(field_blocks_config):
    for block_name, block_config in field_blocks_config.items():
        field_detection_type = block_config.get("fieldDetectionType")

        if field_detection_type == FieldDetectionType.BARCODE_QR:
            # Create BarcodeFieldBlock
            field_block = BarcodeFieldBlock(
                block_name,
                field_detection_type,
                block_config
            )
            template.field_blocks.append(field_block)
```

### Field Class Registration

**File**: `src/processors/layout/field_block/base.py`

```python
from src.processors.constants import FieldDetectionType
from src.processors.layout.field.barcode_field import BarcodeField

# Field class mapping
FIELD_CLASS_FOR_DETECTION_TYPE = {
    FieldDetectionType.BUBBLES_THRESHOLD: BubbleField,
    FieldDetectionType.OCR: OCRField,
    FieldDetectionType.BARCODE_QR: BarcodeField,  # ← Registered here
}

# Field creation
def create_field(field_detection_type, **kwargs):
    field_class = FIELD_CLASS_FOR_DETECTION_TYPE[field_detection_type]
    return field_class(**kwargs)
```

### BarcodeField Initialization

**File**: `src/processors/layout/field/barcode_field.py`

```python
class BarcodeField(Field):
    def __init__(self, direction, empty_value, field_block,
                 field_detection_type, field_label, origin):
        super().__init__(
            direction, empty_value, field_block,
            field_detection_type, field_label, origin
        )

    def setup_scan_boxes(self, field_block):
        """Create single scan box for barcode detection."""
        scan_zone = field_block.scan_zone
        origin = field_block.origin

        # Single scan box per barcode field
        scan_box = BarcodeScanBox(0, self, origin, scan_zone)
        self.scan_boxes = [scan_box]
```

**Key Integration Points**:
1. **Field Detection Type**: `BARCODE_QR` constant
2. **Empty Value**: Configured per field (default `""`)
3. **Scan Zone**: Single rectangular region (vs. bubble grid)
4. **Scan Box**: One per field (vs. multiple for bubbles)

---

## Integration 2: ReadOMR Processor

### Entry Point: ReadOMR.process()

**File**: `src/processors/detection/processor.py`

```python
class ReadOMRProcessor(Processor):
    """Main OMR detection orchestrator."""

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # 1. Resize images to template dimensions
        gray_image, colored_image = ImageUtils.resize_to_dimensions(
            template.template_dimensions, context.gray_image, context.colored_image
        )

        # 2. Normalize images
        gray_image, colored_image = ImageUtils.normalize(gray_image, colored_image)

        # 3. Run detection and interpretation
        # This includes ALL field types: bubbles, OCR, barcodes
        raw_omr_response = self.template_file_runner.read_omr_and_update_metrics(
            file_path, gray_image, colored_image
        )

        # 4. Get concatenated response
        concatenated_omr_response = template.get_concatenated_omr_response(
            raw_omr_response
        )

        # 5. Update context with results
        context.omr_response = concatenated_omr_response
        context.is_multi_marked = is_multi_marked
        context.field_id_to_interpretation = field_id_to_interpretation

        return context
```

**Integration Points**:
1. **Image Preprocessing**: Barcode detection uses same normalized grayscale image as bubbles
2. **Unified Pipeline**: All detection types processed together
3. **Response Generation**: Barcode fields included in `omr_response`
4. **Multi-Marking**: Barcode multi-marking combined with bubble multi-marking

---

## Integration 3: Template File Runner

### Orchestration: Template File Runner

**File**: `src/processors/detection/template_file_runner.py`

```python
class TemplateFileRunner:
    """Orchestrates detection for all field types in a template."""

    def __init__(self, template):
        self.template = template
        self.tuning_config = template.tuning_config

        # Create detection repository (shared across all field types)
        self.repository = DetectionRepository()

        # Initialize field type runners
        self.field_type_runners = {
            FieldDetectionType.BUBBLES_THRESHOLD: BubblesFileRunner(
                tuning_config, repository=self.repository
            ),
            FieldDetectionType.OCR: OCRFileRunner(
                tuning_config, repository=self.repository
            ),
            FieldDetectionType.BARCODE_QR: BarcodeFileRunner(  # ← Barcode runner
                tuning_config, repository=self.repository
            ),
        }
```

### Detection & Interpretation Flow

```python
def read_omr_and_update_metrics(self, file_path, gray_image, colored_image):
    # Phase 1: Detection Pass (all field types)
    for field_type, runner in self.field_type_runners.items():
        # Run detection pass
        file_level_aggregates = runner.detection_pass.process_file_fields(
            self.template, file_path, gray_image, colored_image
        )
        # Stores results in repository

    # Load detection results from repository
    file_detection_results = self.repository.get_file_results(file_path)
    # Contains: bubble_fields, ocr_fields, barcode_fields

    # Phase 2: Interpretation Pass (all field types)
    for field_type, runner in self.field_type_runners.items():
        # Run interpretation pass
        file_level_interpretation_aggregates = runner.interpretation_pass.process_file_fields(
            self.template, file_detection_results
        )

    # Phase 3: Generate response
    raw_omr_response = {}
    for field in self.template.all_fields:
        field_interpretation = file_level_interpretation_aggregates[field.id]
        raw_omr_response[field.field_label] = field_interpretation.get_field_interpretation_string()

    return raw_omr_response
    # Example: {"RollNumber": "STU-2024-12345", "Q1": "A", "Q2": "B"}
```

**Key Integration**:
1. **Shared Repository**: All field types use same `DetectionRepository`
2. **Unified Phases**: Detection → Interpretation (same as bubbles/OCR)
3. **Mixed Response**: Barcode fields mixed with bubble/OCR in output

---

## Integration 4: Barcode File Runner

### Runner Initialization

**File**: `src/processors/detection/barcode/file_runner.py`

```python
class BarcodeFileRunner(FieldTypeFileLevelRunner):
    """Orchestrates barcode detection and interpretation for a template."""

    def __init__(self, tuning_config, repository: DetectionRepository):
        field_detection_type = FieldDetectionType.BARCODE_QR

        # Create detection and interpretation passes
        detection_pass = BarcodeDetectionPass(
            tuning_config, field_detection_type, repository=repository
        )
        interpretation_pass = BarcodeInterpretationPass(
            tuning_config, field_detection_type, repository=repository
        )

        super().__init__(
            tuning_config, field_detection_type,
            detection_pass, interpretation_pass
        )
        self.repository = repository
```

**Integration with Base Class**:
- Inherits from `FieldTypeFileLevelRunner` (same as bubbles/OCR)
- Implements same interface: `detection_pass` and `interpretation_pass`
- Consistent API across all field types

---

## Integration 5: Detection Repository

### Repository Pattern

**File**: `src/processors/repositories/detection_repository.py`

```python
class DetectionRepository:
    """Centralized storage for all detection results."""

    def __init__(self):
        self.bubble_fields: dict[str, BubbleFieldDetectionResult] = {}
        self.ocr_fields: dict[str, OCRFieldDetectionResult] = {}
        self.barcode_fields: dict[str, BarcodeFieldDetectionResult] = {}  # ← Barcode storage

    def save_barcode_field(self, field_id: str,
                          result: BarcodeFieldDetectionResult):
        """Save barcode detection result."""
        self.barcode_fields[field_id] = result

    def get_barcode_field(self, field_id: str) -> BarcodeFieldDetectionResult:
        """Retrieve barcode detection result."""
        return self.barcode_fields[field_id]

    def get_file_results(self, file_path: str) -> FileDetectionResults:
        """Get all detection results for a file."""
        return FileDetectionResults(
            file_path=file_path,
            bubble_fields=self.bubble_fields,
            ocr_fields=self.ocr_fields,
            barcode_fields=self.barcode_fields  # ← Included in file results
        )
```

### Usage in Detection Pass

```python
# BarcodeDetectionPass.process_file_fields()
def update_field_level_aggregates_on_processed_field_detection(
    self, field: Field, field_detection: BarcodeFieldDetection
):
    # Save to repository
    self.repository.save_barcode_field(field.id, field_detection.result)

    # Update aggregates
    self.insert_field_level_aggregates({
        "detections": field_detection.detections
    })
```

### Usage in Interpretation Pass

```python
# BarcodeFieldInterpretation.initialize_from_file_level_aggregates()
def initialize_from_file_level_aggregates(
    self, field, file_level_detection_aggregates, ...
):
    # Load from repository via file_level_detection_aggregates
    barcode_fields = file_level_detection_aggregates["barcode_fields"]
    barcode_result = barcode_fields[field.field_label]
    detections = barcode_result.detections

    # Map to interpretations
    self.interpretations = [
        BarcodeInterpretation(detection)
        for detection in detections
    ]
```

**Benefits**:
1. **Decoupling**: Detection and interpretation separated
2. **Type Safety**: Strongly-typed models (`BarcodeFieldDetectionResult`)
3. **Caching**: Results stored, no re-detection needed
4. **Testing**: Easy to mock repository

---

## Integration 6: Response Generation

### Concatenated Response

**File**: `src/processors/template/template.py`

```python
def get_concatenated_omr_response(self, raw_omr_response):
    """Combine raw response with custom label concatenation."""

    concatenated_response = {}

    for field_label, field_value in raw_omr_response.items():
        # Check if field has custom label
        if field_label in self.custom_labels:
            # Concatenate with other fields (multi-field labels)
            custom_label = self.custom_labels[field_label]
            if custom_label not in concatenated_response:
                concatenated_response[custom_label] = []
            concatenated_response[custom_label].append(field_value)
        else:
            # Direct mapping
            concatenated_response[field_label] = field_value

    # Join multi-field labels
    for label, values in concatenated_response.items():
        if isinstance(values, list):
            concatenated_response[label] = "".join(values)

    return concatenated_response
```

**Example**:
```python
# Template config
custom_labels = {
    "RollPart1": "RollNumber",
    "RollPart2": "RollNumber"
}

# Raw response (barcode fields)
raw_omr_response = {
    "RollPart1": "STU",      # Barcode field 1
    "RollPart2": "12345",    # Barcode field 2
    "Q1": "A",               # Bubble field
    "Q2": "B"
}

# Concatenated response
concatenated_response = {
    "RollNumber": "STU12345",  # Concatenated barcode fields
    "Q1": "A",
    "Q2": "B"
}
```

**Integration Points**:
- Barcode fields can be concatenated with other barcodes
- Mixed with bubble/OCR fields in final response
- Custom labels apply to all field types

---

## Integration 7: Evaluation System

### Evaluation Config

**File**: `src/schemas/evaluation_schema.py`

```json
{
    "questions": {
        "RollNumber": {
            "correctAnswer": "STU-2024-12345",
            "score": 0,  // No points (just metadata)
            "type": "barcode"
        },
        "Q1": {
            "correctAnswer": "A",
            "score": 1,
            "type": "bubble"
        }
    }
}
```

### Evaluation Processing

```python
# Evaluate barcode field
def evaluate_field(field_label, student_response, evaluation_config):
    correct_answer = evaluation_config.questions[field_label].correctAnswer

    if student_response == correct_answer:
        return {
            "verdict": "correct",
            "score": evaluation_config.questions[field_label].score
        }
    else:
        return {
            "verdict": "incorrect",
            "score": 0
        }
```

**Use Cases**:
1. **Verification**: Check if barcode matches expected value
2. **Grouping**: Group sheets by barcode value (section, batch)
3. **Metadata**: Store barcode in results without scoring

---

## Integration 8: Visualization & Drawing

### Drawing Integration

**File**: `src/processors/detection/barcode/interpretation_drawing.py`

```python
class BarcodeFieldInterpretationDrawing(FieldInterpretationDrawing):
    """Draws barcode detection results on output image."""

    def draw_field_interpretation(
        self, marked_image, image_type, evaluation_meta, evaluation_config
    ):
        # 1. Draw individual bounding boxes
        for interpretation in self.field_interpretation.interpretations:
            bounding_box = interpretation.text_detection.bounding_box
            DrawingUtils.draw_contour(marked_image, bounding_box, color=CLR_BLACK)

        # 2. Draw combined bounding box (with evaluation color)
        combined_bbox = get_bounding_box_of_points(all_points)
        verdict_color = get_evaluation_verdict_color(...)  # Green/Red
        DrawingUtils.draw_contour(marked_image, combined_bbox, color=verdict_color)

        # 3. Draw interpreted text
        interpreted_text = self.field_interpretation.get_field_interpretation_string()
        DrawingUtils.draw_text_responsive(
            marked_image, interpreted_text, text_position,
            color=CLR_BLACK, thickness=3
        )
```

**Integration with Template Workflow**:
```python
# Template saves marked images with all field types drawn
template.save_image_ops.append_save_image(
    "Checked", range(4), marked_gray, marked_colored
)
# Includes: bubbles (circles), OCR (boxes), barcodes (boxes + text)
```

---

## Integration 9: CSV Output

### CSV Writer

**File**: `src/utils/csv.py`

```python
# Thread-safe CSV writer
def write_csv_row(file_path, row_data):
    with csv_lock:
        with open(file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row_data)
```

### CSV Row Format

```python
# Example CSV row (mixed field types)
row = {
    "file_path": "student_001.png",
    "RollNumber": "STU-2024-12345",  # Barcode field
    "SerialQR": "BATCH-A-001",       # Barcode field
    "Q1": "A",                        # Bubble field
    "Q2": "B",                        # Bubble field
    "Score": "2"                      # Evaluation
}
```

**Integration Points**:
- Barcode fields appear as regular columns
- No special handling needed (strings)
- Multi-marked barcodes: Concatenated value + warning column

---

## Integration 10: File Organization

### Organization by Barcode Value

**File**: `src/processors/organization/processor.py`

```json
{
    "organizationConfig": {
        "enabled": true,
        "outputDirectory": "./organized",
        "patterns": [
            {
                "name": "ByRollNumber",
                "format": "{RollNumber}",
                "mapping": [
                    {
                        "regex": "^STU-2024-.*",
                        "directory": "batch_2024"
                    },
                    {
                        "regex": "^STU-2025-.*",
                        "directory": "batch_2025"
                    }
                ]
            },
            {
                "name": "BySection",
                "format": "{Section}-{RollNumber}",
                "mapping": [
                    {
                        "regex": "^A-.*",
                        "directory": "section_A"
                    },
                    {
                        "regex": "^B-.*",
                        "directory": "section_B"
                    }
                ]
            }
        ]
    }
}
```

**Example**:
```
Input: student_001.png
Barcode RollNumber: "STU-2024-12345"
Barcode Section: "A"

Organization:
./organized/
  ├── batch_2024/
  │   └── section_A/
  │       └── A-STU-2024-12345.png
```

**Integration**:
- Barcode values used in directory structure
- Regex matching on barcode content
- Mixed with bubble/OCR fields for patterns

---

## Browser Migration Integration

### TypeScript Integration Points

```typescript
// 1. Template System
interface BarcodeFieldBlock {
    fieldDetectionType: 'BARCODE_QR';
    emptyValue: string;
    scanZone: {
        origin: [number, number];
        dimensions: [number, number];
        margins: [number, number, number, number];
    };
}

// 2. Detection Repository
class DetectionRepository {
    private barcodeFields: Map<string, BarcodeFieldDetectionResult> = new Map();

    saveBarcodeField(fieldId: string, result: BarcodeFieldDetectionResult): void {
        this.barcodeFields.set(fieldId, result);
    }

    getBarcodeField(fieldId: string): BarcodeFieldDetectionResult | undefined {
        return this.barcodeFields.get(fieldId);
    }
}

// 3. ReadOMR Processor
class ReadOMRProcessor {
    async process(context: ProcessingContext): Promise<ProcessingContext> {
        // Resize, normalize images
        const { grayImage, coloredImage } = await this.preprocessImages(context);

        // Run detection (including barcodes)
        const omrResponse = await this.templateFileRunner.readOMRAndUpdateMetrics(
            context.filePath, grayImage, coloredImage
        );

        // Update context
        context.omrResponse = omrResponse;
        return context;
    }
}

// 4. Barcode File Runner
class BarcodeFileRunner {
    constructor(
        private tuningConfig: TuningConfig,
        private repository: DetectionRepository
    ) {
        this.detectionPass = new BarcodeDetectionPass(tuningConfig, repository);
        this.interpretationPass = new BarcodeInterpretationPass(tuningConfig, repository);
    }
}

// 5. @zxing/library Integration
import { BrowserMultiFormatReader } from '@zxing/library';

class PyZBarBrowser {
    private reader: BrowserMultiFormatReader;

    async getSingleTextDetection(
        canvas: HTMLCanvasElement,
        confidenceThreshold: number = 0.8
    ): Promise<TextDetection | null> {
        try {
            const result = await this.reader.decodeFromCanvas(canvas);
            return {
                detected_text: result.getText(),
                bounding_box: this.calculateBoundingBox(result.getResultPoints()),
                rotated_rectangle: this.convertResultPoints(result.getResultPoints()),
                confident_score: 1.0  // ZXing doesn't provide quality
            };
        } catch (error) {
            return null;
        }
    }
}
```

---

## Integration Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Template Definition                         │
│  {                                                               │
│    "fieldBlocks": {                                              │
│      "RollNumber": { "fieldDetectionType": "BARCODE_QR", ... }   │
│    }                                                             │
│  }                                                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Template Layout System                        │
│  - Parse field blocks                                            │
│  - Create BarcodeField instances                                 │
│  - Register scan zones                                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ReadOMR Processor                            │
│  - Preprocess images (resize, normalize)                         │
│  - Call TemplateFileRunner                                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Template File Runner                           │
│  - Initialize DetectionRepository                                │
│  - Initialize field type runners (Bubbles, OCR, Barcode)         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Barcode File Runner                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Detection Pass                             │    │
│  │  - For each barcode field:                              │    │
│  │    1. Extract scan zone                                 │    │
│  │    2. PyZBar.decode()                                   │    │
│  │    3. Create BarcodeDetection                           │    │
│  │    4. Save to repository                                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           Interpretation Pass                           │    │
│  │  - For each barcode field:                              │    │
│  │    1. Load detections from repository                   │    │
│  │    2. Create interpretations                            │    │
│  │    3. Check multi-marking                               │    │
│  │    4. Generate field value string                       │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Detection Repository                           │
│  {                                                               │
│    "bubble_fields": { ... },                                     │
│    "ocr_fields": { ... },                                        │
│    "barcode_fields": {                                           │
│      "RollNumber": BarcodeFieldDetectionResult(...)              │
│    }                                                             │
│  }                                                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Response Generation                           │
│  {                                                               │
│    "RollNumber": "STU-2024-12345",  // Barcode                   │
│    "Q1": "A",                       // Bubble                    │
│    "Q2": "B"                        // Bubble                    │
│  }                                                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│   Evaluation System      │  │  Visualization System    │
│  - Verify barcode value  │  │  - Draw bounding boxes   │
│  - Compare with answer   │  │  - Display text          │
│  - Calculate score       │  │  - Color by verdict      │
└──────────────────────────┘  └──────────────────────────┘
              │                             │
              └──────────────┬──────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        CSV Output                                │
│  file_path,RollNumber,Q1,Q2,Score                                │
│  student_001.png,STU-2024-12345,A,B,2                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary

### Key Integration Points

1. **Template System**
   - Field type registration: `BARCODE_QR`
   - Field class: `BarcodeField`
   - Scan zone definition: Single rectangular region

2. **ReadOMR Pipeline**
   - Unified preprocessing (same images as bubbles/OCR)
   - Parallel processing (all field types together)
   - Shared response generation

3. **Detection Repository**
   - Typed storage: `BarcodeFieldDetectionResult`
   - Decoupling: Detection and interpretation separated
   - Shared across field types

4. **Response & Evaluation**
   - Mixed with bubble/OCR fields
   - Custom label concatenation
   - Evaluation support (verification, grouping)

5. **Output Systems**
   - CSV: Regular string columns
   - Visualization: Bounding boxes + text
   - File organization: Regex matching on barcode values

### Browser Migration Considerations

- **Same Integration Points**: TypeScript interfaces match Python classes
- **Async/Await**: Add async to all integration methods
- **Type Safety**: Use TypeScript interfaces for all models
- **Repository Pattern**: Use Map instead of dict
- **@zxing/library**: Drop-in replacement for PyZBar (with adaptations)

### Testing Integration

```python
# Integration test: End-to-end barcode detection
def test_barcode_integration():
    # 1. Load template with barcode fields
    template = load_template("barcode_template.json")
    assert template.has_field_type(FieldDetectionType.BARCODE_QR)

    # 2. Process image
    processor = ReadOMRProcessor(template)
    context = ProcessingContext(file_path="test.png", ...)
    result_context = processor.process(context)

    # 3. Verify response
    assert result_context.omr_response["RollNumber"] == "STU-2024-12345"

    # 4. Verify repository
    repository = processor.template_file_runner.repository
    barcode_result = repository.get_barcode_field("RollNumber")
    assert len(barcode_result.detections) == 1
```

**Next Steps**: See `concept.md` for barcode detection architecture overview.
