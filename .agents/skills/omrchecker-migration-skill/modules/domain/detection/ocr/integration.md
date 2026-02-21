# OCR Detection Integration

**Module**: Domain - Detection - OCR
**Python Reference**: `src/processors/detection/ocr/`
**Last Updated**: 2026-02-21

---

## Overview

This document describes how the OCR detection system integrates with other components of OMRChecker, including ReadOMR, template system, detection repository, interpretation pipeline, and output generation.

---

## Integration with ReadOMR Processor

### Overview

ReadOMR is the main orchestrator for all detection types (bubbles, OCR, barcodes). OCR detection is one of three detection strategies managed by ReadOMR.

### Integration Points

```
ReadOMRProcessor
    │
    ├── Template (contains OCR fields)
    │     └── OCRField[]
    │
    └── TemplateFileRunner
          │
          └── Detection Passes
                ├── BubbleDetectionPass
                ├── OCRDetectionPass ← OCR Integration
                └── BarcodeDetectionPass
```

### Flow

```python
# 1. ReadOMR Initialization
class ReadOMRProcessor(Processor):
    def __init__(self, template, ml_model_path=None):
        self.template = template
        self.template_file_runner = TemplateFileRunner(template)

# 2. Template contains OCR fields
template.ocr_fields = [
    OCRField(field_label="studentName", ...),
    OCRField(field_label="rollNumber", ...),
]

# 3. Processing a file
def process(self, context: ProcessingContext):
    # Resize and normalize image
    gray_image = context.gray_image
    colored_image = context.colored_image

    # Run all detection passes (including OCR)
    self.template_file_runner.process_file(
        file_path,
        gray_image,
        colored_image
    )
```

### Detection Pass Execution

```python
# Inside TemplateFileRunner.process_file()
class TemplateFileRunner:
    def process_file(self, file_path, gray_image, colored_image):
        # Create detection passes
        ocr_detection_pass = OCRDetectionPass(
            tuning_config=self.tuning_config,
            repository=self.detection_repository
        )

        # Initialize aggregates
        ocr_detection_pass.initialize_file_level_aggregates(file_path)

        # Process all OCR fields
        for field in self.template.ocr_fields:
            field_detection = ocr_detection_pass.get_field_detection(
                field,
                gray_image,
                colored_image
            )
            # Update aggregates and repository
            ocr_detection_pass.update_field_level_aggregates_on_processed_field_detection(
                field,
                field_detection
            )

        # Get results
        file_level_detection_aggregates = (
            ocr_detection_pass.get_file_level_aggregates()
        )
```

---

## Integration with Template System

### Template JSON Definition

OCR fields are defined in the template JSON with `fieldType: "OCR"`.

```json
{
  "fieldBlocks": [
    {
      "fieldType": "OCR",
      "origin": [100, 200],
      "scanZone": {
        "dimensions": [200, 50],
        "margins": {
          "top": 5,
          "right": 5,
          "bottom": 5,
          "left": 5
        }
      },
      "labelsPath": ["studentName"],
      "emptyValue": "",
      "direction": "vertical"
    }
  ]
}
```

### Template Initialization

```python
# 1. Parse template JSON
template_data = json.load(template_file)

# 2. Create Template object
template = Template(template_data, tuning_config)

# 3. Template.setup() generates OCR fields
template.setup()
    ├── Parse field blocks
    ├── For each block with fieldType="OCR":
    │     ├── Create OCRField for each label
    │     └── Setup OCRScanBox with scan_zone_rectangle
    └── Store in template.ocr_fields[]

# 4. OCR fields are ready for detection
template.ocr_fields = [
    OCRField(field_label="studentName", scan_boxes=[OCRScanBox(...)]),
    ...
]
```

### Field Block to OCR Field Mapping

```python
# Field Block (from template JSON)
field_block = {
    "fieldType": "OCR",
    "labelsPath": ["firstName", "lastName"],
    "scanZone": {
        "dimensions": [200, 50],
        "margins": {"top": 5, ...}
    },
    "emptyValue": "",
    "direction": "vertical"
}

# Generated OCR Fields (2 fields from 1 block)
ocr_fields = [
    OCRField(
        field_label="firstName",
        empty_value="",
        direction="vertical",
        origin=[100, 200],
        scan_boxes=[
            OCRScanBox(
                scan_zone_rectangle=[[100,200], [300,200], [300,250], [100,250]],
                dimensions=[200, 50],
                margins={...}
            )
        ]
    ),
    OCRField(
        field_label="lastName",
        empty_value="",
        direction="vertical",
        origin=[100, 260],  # Shifted by field height + spacing
        scan_boxes=[OCRScanBox(...)]
    )
]
```

---

## Integration with Detection Repository

### Purpose

DetectionRepository stores OCR detection results for access during interpretation pass.

### Storage Flow

```python
# Detection Pass stores results
class OCRDetectionPass:
    def __init__(self, repository: DetectionRepository):
        self.repository = repository

    def update_field_level_aggregates_on_processed_field_detection(
        self, field: Field, field_detection: OCRFieldDetection
    ):
        # Store typed result in repository
        self.repository.save_ocr_field(
            field.id,
            field_detection.result  # OCRFieldDetectionResult
        )
```

### Repository Implementation

```python
class DetectionRepository:
    def __init__(self):
        self.ocr_fields: dict[str, OCRFieldDetectionResult] = {}

    def save_ocr_field(
        self,
        field_id: str,
        result: OCRFieldDetectionResult
    ):
        self.ocr_fields[field_id] = result

    def get_ocr_field(self, field_id: str) -> OCRFieldDetectionResult:
        return self.ocr_fields.get(field_id)

    def get_all_ocr_fields(self) -> dict[str, OCRFieldDetectionResult]:
        return self.ocr_fields
```

### Retrieval Flow

```python
# Interpretation Pass retrieves results
class OCRFieldInterpretation:
    def initialize_from_file_level_aggregates(
        self, field, file_level_detection_aggregates
    ):
        # Get repository data from aggregates
        ocr_fields = file_level_detection_aggregates["ocr_fields"]

        # Retrieve specific field result
        ocr_result = ocr_fields[field.field_label]  # OCRFieldDetectionResult

        # Access detections
        detections = ocr_result.detections  # List[OCRDetection]
```

### Aggregates Structure

```python
file_level_detection_aggregates = {
    "ocr_fields": {  # From repository
        "studentName": OCRFieldDetectionResult(
            field_id="studentName_0",
            field_label="studentName",
            detections=[
                OCRDetection(
                    detected_text="John",
                    bounding_box=[[110,205], ...],
                    confident_score=0.95,
                    ...
                )
            ],
            confidence=0.95
        ),
        "rollNumber": OCRFieldDetectionResult(...),
    },
    "fields_count": {"processed": 2}
}
```

---

## Integration with Interpretation Pipeline

### Two-Pass Architecture

```
Pass 1: Detection
  └── OCRDetectionPass
        └── For each OCR field:
              └── OCRFieldDetection
                    └── EasyOCR.get_single_text_detection()
                          └── Store in DetectionRepository

Pass 2: Interpretation
  └── OCRInterpretationPass
        └── For each OCR field:
              └── OCRFieldInterpretation
                    └── Load from DetectionRepository
                          └── Concatenate detections
                                └── Return interpreted string
```

### Detection Pass

```python
class OCRDetectionPass(FieldTypeDetectionPass):
    def get_field_detection(
        self, field: Field, gray_image, colored_image
    ) -> OCRFieldDetection:
        # Create detection (runs OCR during __init__)
        return OCRFieldDetection(field, gray_image, colored_image)

    def update_field_level_aggregates_on_processed_field_detection(
        self, field: Field, field_detection: OCRFieldDetection
    ) -> None:
        # Save to repository
        self.repository.save_ocr_field(field.id, field_detection.result)

        # Update aggregates
        self.insert_field_level_aggregates({
            "detections": field_detection.detections
        })
```

### Interpretation Pass

```python
class OCRInterpretationPass(FieldTypeInterpretationPass):
    def get_field_interpretation(
        self,
        field: Field,
        file_level_detection_aggregates,
        file_level_interpretation_aggregates,
    ) -> OCRFieldInterpretation:
        # Create interpretation
        return OCRFieldInterpretation(
            self.tuning_config,
            field,
            file_level_detection_aggregates,
            file_level_interpretation_aggregates,
        )

    def update_field_level_aggregates_on_processed_field_interpretation(
        self, field: Field, field_interpretation: OCRFieldInterpretation
    ) -> None:
        # Track multi-mark status
        self.insert_field_level_aggregates({
            "is_multi_marked": field_interpretation.is_multi_marked,
        })
```

### Data Flow Between Passes

```python
# Pass 1: Detection
ocr_detection = OCRFieldDetection(field, gray_image, colored_image)
# ↓ Stores in repository
repository.save_ocr_field(field.id, ocr_detection.result)

# Pass 2: Interpretation (later)
ocr_interpretation = OCRFieldInterpretation(
    tuning_config,
    field,
    file_level_detection_aggregates,  # Contains repository data
    ...
)
# ↓ Retrieves from repository
ocr_fields = file_level_detection_aggregates["ocr_fields"]
ocr_result = ocr_fields[field.field_label]
detections = ocr_result.detections

# ↓ Interprets
interpreted_text = ocr_interpretation.get_field_interpretation_string()
# "John" or "" (if empty)
```

---

## Integration with Evaluation System

### Evaluation Config

OCR fields can have correct answers defined for evaluation.

```json
{
  "questions": {
    "studentName": {
      "answer": "JOHN DOE",
      "matchType": "exact"
    },
    "rollNumber": {
      "answer": "123456",
      "matchType": "exact"
    }
  }
}
```

### Evaluation Flow

```python
# After interpretation
interpreted_text = "John Doe"  # From OCR
expected_answer = "JOHN DOE"   # From evaluation config

# Match type: exact (case-insensitive)
is_correct = interpreted_text.upper() == expected_answer.upper()

# Store in evaluation meta
evaluation_meta["questions_meta"]["studentName"] = {
    "is_correct": True,
    "bonus_type": None,
    "verdict": "correct"
}
```

### Drawing Integration

```python
class OCRFieldInterpretationDrawing:
    def draw_field_interpretation(
        self, marked_image, image_type, evaluation_meta, evaluation_config
    ):
        # Determine color based on evaluation
        if evaluation_meta and field_label in evaluation_meta["questions_meta"]:
            question_meta = evaluation_meta["questions_meta"][field_label]

            verdict_color = evaluation_config.get_evaluation_meta_for_question(
                question_meta, field_interpretation, image_type
            )[1]  # Get verdict color

            # Green for correct, red for wrong, orange for multi-marked
            combined_bounding_box_color = verdict_color
        else:
            # Black if no evaluation
            combined_bounding_box_color = CLR_BLACK

        # Draw with appropriate color
        DrawingUtils.draw_contour(
            marked_image,
            combined_bounding_box,
            color=combined_bounding_box_color
        )
```

---

## Integration with Output Generation

### CSV Output

```python
# OCR interpretation results are written to CSV
class ResultsWriter:
    def write_results(self, file_path, field_interpretations):
        row = {}

        for field_label, interpretation in field_interpretations.items():
            if isinstance(interpretation, OCRFieldInterpretation):
                # Get interpreted string
                text_value = interpretation.get_field_interpretation_string()
                row[field_label] = text_value

        # Write to CSV
        csv_writer.writerow(row)
```

**CSV Example**:
```csv
fileName,studentName,rollNumber,q1,q2,q3
image1.jpg,John Doe,123456,A,B,C
image2.jpg,Jane Smith,789012,B,A,D
image3.jpg,,345678,C,C,A
```

**Note**: Empty OCR fields show as empty strings in CSV.

---

### Image Output (Drawing)

```python
# Drawing pass
class OCRFieldInterpretationDrawing:
    def draw_field_interpretation(
        self, marked_image, image_type, evaluation_meta, evaluation_config
    ):
        # 1. Draw individual detection bounding boxes
        for interpretation in field_interpretation.interpretations:
            bounding_box = interpretation.text_detection.bounding_box
            DrawingUtils.draw_contour(marked_image, bounding_box, color=CLR_BLACK)

        # 2. Draw combined bounding box
        combined_bounding_box = compute_combined_box(all_bounding_boxes)
        DrawingUtils.draw_contour(
            marked_image,
            combined_bounding_box,
            color=verdict_color
        )

        # 3. Draw interpreted text
        interpreted_text = field_interpretation.get_field_interpretation_string()
        DrawingUtils.draw_text_responsive(
            marked_image,
            interpreted_text,
            text_position,
            color=CLR_BLACK,
            thickness=3
        )
```

**Output Image Example**:
```
┌─────────────────────────────┐
│  File: image1.jpg           │
│                             │
│  "John Doe"  ← Interpreted  │
│  ┌──────────────────────┐   │
│  │  John Doe            │ ← Bounding box (green if correct)
│  └──────────────────────┘   │
│                             │
│  "123456"                   │
│  ┌──────────────────────┐   │
│  │  123456              │   │
│  └──────────────────────┘   │
└─────────────────────────────┘
```

---

## Integration with Multi-Format Detection

### Detection Type Dispatch

ReadOMR handles multiple field types via detection passes.

```python
class TemplateFileRunner:
    def process_file(self, file_path, gray_image, colored_image):
        # Create all detection passes
        bubble_pass = BubbleDetectionPass(...)
        ocr_pass = OCRDetectionPass(...)
        barcode_pass = BarcodeDetectionPass(...)

        # Process each field type
        self.run_detection_pass(bubble_pass, template.bubble_fields)
        self.run_detection_pass(ocr_pass, template.ocr_fields)
        self.run_detection_pass(barcode_pass, template.barcode_fields)

        # Combine results
        file_level_detection_aggregates = {
            "bubble_fields": bubble_pass.get_file_level_aggregates()["bubble_fields"],
            "ocr_fields": ocr_pass.get_file_level_aggregates()["ocr_fields"],
            "barcode_fields": barcode_pass.get_file_level_aggregates()["barcode_fields"],
        }
```

### Mixed Field Templates

A single template can have bubbles, OCR, and barcodes.

```json
{
  "fieldBlocks": [
    {
      "fieldType": "BUBBLES_THRESHOLD",
      "labelsPath": ["q1", "q2", "q3"],
      ...
    },
    {
      "fieldType": "OCR",
      "labelsPath": ["studentName"],
      ...
    },
    {
      "fieldType": "BARCODE_QR",
      "labelsPath": ["studentId"],
      ...
    }
  ]
}
```

**Detection Flow**:
```
Process File
  ├── Bubble Detection (q1, q2, q3)
  ├── OCR Detection (studentName)
  └── Barcode Detection (studentId)

CSV Output:
q1,q2,q3,studentName,studentId
A,B,C,John Doe,STU123456
```

---

## Integration with Preprocessing

### Image Channels

OCR detection can use different image channels for better accuracy.

```python
# Current: Uses gray_image
def run_detection(self, field: OCRField, gray_image, colored_image):
    image_zone = extract_image_from_zone_rectangle(
        gray_image,  # Grayscale
        zone_label,
        scan_zone_rectangle
    )

# Future: Use "ocr" channel with custom preprocessing
def run_detection(self, field: OCRField, image_channels, colored_image):
    # Get OCR-specific preprocessed image
    ocr_image = image_channels.get("ocr", gray_image)

    # OCR channel could have:
    # - Enhanced contrast
    # - Adaptive thresholding
    # - Noise reduction
    # - Deskewing

    image_zone = extract_image_from_zone_rectangle(
        ocr_image,
        zone_label,
        scan_zone_rectangle
    )
```

### Preprocessing Configuration

```json
{
  "preProcessingConfig": {
    "channels": {
      "gray": {
        "processors": ["GaussianBlur"]
      },
      "ocr": {
        "processors": [
          "GaussianBlur",
          "Contrast",
          "AdaptiveThreshold"
        ]
      }
    }
  }
}
```

---

## Browser Integration

### Architecture

```
React/Vue Component
    │
    ├── OCR Manager (singleton)
    │     └── Tesseract Worker Pool
    │
    ├── Template Loader
    │     └── Parse template JSON
    │           └── Create OCRField objects
    │
    ├── Image Processor
    │     └── Canvas API for image manipulation
    │
    └── Detection Pipeline
          └── For each OCR field:
                └── Extract scan zone (Canvas)
                      └── Send to OCR Worker
                            └── Tesseract.js recognition
                                  └── Return OCRDetection
```

### Worker Pool Implementation

```typescript
// ocr-manager.ts
class OCRManager {
  private static instance: OCRManager;
  private workerPool: TesseractWorkerPool;

  static async getInstance(): Promise<OCRManager> {
    if (!OCRManager.instance) {
      OCRManager.instance = new OCRManager();
      await OCRManager.instance.initialize();
    }
    return OCRManager.instance;
  }

  private async initialize() {
    const poolSize = navigator.hardwareConcurrency || 2;
    this.workerPool = new TesseractWorkerPool();
    await this.workerPool.initialize(Math.min(poolSize, 4));
  }

  async detectOCRField(
    field: OCRField,
    imageData: ImageData
  ): Promise<OCRFieldDetectionResult> {
    // Extract scan zone
    const scanZone = this.extractScanZone(imageData, field.scanBox);

    // Run OCR
    const detection = await this.workerPool.recognize(scanZone);

    // Create result
    return {
      fieldId: field.id,
      fieldLabel: field.fieldLabel,
      detections: detection ? [detection] : [],
      confidence: detection?.confidenceScore || 0.0,
      timestamp: new Date(),
    };
  }
}
```

### React Component Integration

```typescript
// OCRDetectionComponent.tsx
import { OCRManager } from './ocr-manager';

function OCRDetectionComponent({ template, imageFile }) {
  const [results, setResults] = useState<OCRFieldDetectionResult[]>([]);
  const [progress, setProgress] = useState(0);

  async function processImage() {
    // Load image
    const imageData = await loadImageToCanvas(imageFile);

    // Get OCR manager
    const ocrManager = await OCRManager.getInstance();

    // Process all OCR fields
    const ocrFields = template.fields.filter(f => f.fieldType === 'OCR');
    const totalFields = ocrFields.length;

    const detectionResults: OCRFieldDetectionResult[] = [];

    for (let i = 0; i < ocrFields.length; i++) {
      const field = ocrFields[i];

      // Detect
      const result = await ocrManager.detectOCRField(field, imageData);
      detectionResults.push(result);

      // Update progress
      setProgress(((i + 1) / totalFields) * 100);
    }

    setResults(detectionResults);
  }

  return (
    <div>
      <button onClick={processImage}>Process OCR</button>
      <ProgressBar value={progress} />
      <ResultsTable results={results} />
    </div>
  );
}
```

---

## API Interfaces

### Python API

```python
# Public interfaces for external integration

# 1. Detect OCR field
def detect_ocr_field(
    field: OCRField,
    gray_image: np.ndarray,
    confidence_threshold: float = 0.8
) -> OCRFieldDetectionResult:
    """Detect text in OCR field.

    Args:
        field: OCRField definition with scan zone
        gray_image: Grayscale image (NumPy array)
        confidence_threshold: Minimum confidence (0.0-1.0)

    Returns:
        OCRFieldDetectionResult with detections and confidence
    """
    detection = OCRFieldDetection(field, gray_image, None)
    return detection.result

# 2. Interpret OCR detection
def interpret_ocr_field(
    detection_result: OCRFieldDetectionResult,
    empty_value: str = ""
) -> str:
    """Interpret OCR detection as string.

    Args:
        detection_result: Detection result from detect_ocr_field()
        empty_value: Value to return if no detection

    Returns:
        Interpreted text string
    """
    if not detection_result.detections:
        return empty_value

    # Concatenate all detected texts
    texts = [d.detected_text for d in detection_result.detections]
    return "".join(texts)
```

### Browser API

```typescript
// Public interfaces for external integration

// 1. Detect OCR field
async function detectOCRField(
  field: OCRField,
  imageData: ImageData,
  options?: {
    confidenceThreshold?: number;
    language?: string;
  }
): Promise<OCRFieldDetectionResult> {
  const manager = await OCRManager.getInstance();
  return manager.detectOCRField(field, imageData, options);
}

// 2. Interpret OCR detection
function interpretOCRField(
  detectionResult: OCRFieldDetectionResult,
  emptyValue: string = ""
): string {
  if (detectionResult.detections.length === 0) {
    return emptyValue;
  }

  // Concatenate all detected texts
  const texts = detectionResult.detections.map(d => d.detectedText);
  return texts.join("");
}

// 3. Batch process multiple fields
async function detectAllOCRFields(
  fields: OCRField[],
  imageData: ImageData,
  onProgress?: (progress: number) => void
): Promise<Map<string, OCRFieldDetectionResult>> {
  const manager = await OCRManager.getInstance();
  const results = new Map<string, OCRFieldDetectionResult>();

  for (let i = 0; i < fields.length; i++) {
    const field = fields[i];
    const result = await manager.detectOCRField(field, imageData);
    results.set(field.fieldLabel, result);

    if (onProgress) {
      onProgress((i + 1) / fields.length);
    }
  }

  return results;
}
```

---

## Testing Integration

### Unit Tests

```python
# test_ocr_detection.py
def test_ocr_field_detection(sample_ocr_field, sample_image):
    # Create detection
    detection = OCRFieldDetection(
        sample_ocr_field,
        sample_image,
        None
    )

    # Verify result
    assert detection.result is not None
    assert detection.result.field_label == sample_ocr_field.field_label
    assert isinstance(detection.result, OCRFieldDetectionResult)

def test_ocr_interpretation(sample_detection_result):
    # Create mock field
    field = create_mock_ocr_field()

    # Create interpretation
    interpretation = OCRFieldInterpretation(
        tuning_config=TuningConfig(),
        field=field,
        file_level_detection_aggregates={
            "ocr_fields": {field.field_label: sample_detection_result}
        },
        file_level_interpretation_aggregates={}
    )

    # Run interpretation
    interpretation.run_interpretation(field, {}, {})

    # Verify result
    result = interpretation.get_field_interpretation_string()
    assert result == "expected_text"
```

### Integration Tests

```python
# test_ocr_integration.py
def test_full_ocr_pipeline(test_template, test_image):
    # Create ReadOMR processor
    processor = ReadOMRProcessor(test_template)

    # Process image
    context = ProcessingContext(
        input_file_path="test.jpg",
        gray_image=test_image,
        colored_image=test_image
    )

    result_context = processor.process(context)

    # Verify OCR results in context
    assert "studentName" in result_context.output_dict
    assert result_context.output_dict["studentName"] != ""
```

---

## Related Documentation

- **Concept**: `modules/domain/detection/ocr/concept.md`
- **Flows**: `modules/domain/detection/ocr/flows.md`
- **Decisions**: `modules/domain/detection/ocr/decisions.md`
- **Constraints**: `modules/domain/detection/ocr/constraints.md`
- **ReadOMR**: `modules/domain/detection/concept.md`
- **Template**: `modules/domain/template/concept.md`

---

## Summary

OCR Detection integrates with:

1. **ReadOMR**: Via detection pass architecture
2. **Template System**: Field blocks → OCR fields with scan zones
3. **Detection Repository**: Type-safe storage of detection results
4. **Interpretation Pipeline**: Two-pass detection → interpretation
5. **Evaluation System**: Color-coded correctness feedback
6. **Output Generation**: CSV export and image drawing
7. **Multi-Format Detection**: Coexists with bubbles and barcodes
8. **Browser**: Tesseract.js with Web Workers and Canvas API

**Key Integration Points**: DetectionRepository for pass separation, typed results for safety, unified API for consistency across detection types.
