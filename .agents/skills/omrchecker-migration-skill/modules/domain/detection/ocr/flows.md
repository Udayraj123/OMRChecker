# OCR Detection Flows

**Module**: Domain - Detection - OCR
**Python Reference**: `src/processors/detection/ocr/`
**Last Updated**: 2026-02-21

---

## Overview

This document details the step-by-step flows for OCR text detection, including EasyOCR integration, text extraction, post-processing, and interpretation.

---

## Flow 1: OCR Field Setup (Template Initialization)

### Trigger
Template initialization during `Template.setup()`

### Steps

```
1. Parse Template JSON
   Input: template.json with OCR field blocks
   {
     "fieldBlocks": [{
       "fieldType": "OCR",
       "labelsPath": ["studentName"],
       "scanZone": {
         "dimensions": [200, 50],
         "margins": {"top": 5, "right": 5, "bottom": 5, "left": 5}
       },
       "emptyValue": ""
     }]
   }

2. Create Field Block
   FieldBlock(
     field_detection_type="OCR",
     field_labels=["studentName"],
     scan_zone={dimensions: [200, 50], margins: {...}}
   )

3. Generate OCR Fields
   For each label in labelsPath:
     OCRField(
       field_label="studentName",
       field_detection_type="OCR",
       empty_value="",
       direction="vertical",  # or horizontal
       origin=[x, y],
       field_block=field_block
     )

4. Setup Scan Boxes
   OCRField.setup_scan_boxes(field_block)
     ├── Extract scan_zone from field_block
     ├── Create OCRScanBox(
     │     field_index=0,
     │     origin=[x, y],
     │     dimensions=[200, 50],
     │     margins={top: 5, ...}
     │   )
     └── Compute scan_zone_rectangle
           ShapeUtils.compute_scan_zone_rectangle(
             zone_description,
             include_margins=True
           )
           → Returns [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

5. Store in Template
   template.fields.append(ocr_field)
   template.ocr_fields.append(ocr_field)
```

### Output
- OCRField with pre-computed scan zone rectangle
- Ready for detection pass

---

## Flow 2: EasyOCR Detection (Single Field)

### Trigger
`OCRFieldDetection.run_detection()` during detection pass

### Prerequisites
- Aligned grayscale image
- OCRField with scan zone defined

### Steps

```
1. Extract Field Metadata
   field: OCRField
   scan_box = field.scan_boxes[0]  # Currently single scan box per field
   zone_label = scan_box.zone_description["label"]
   scan_zone_rectangle = scan_box.scan_zone_rectangle
   # Example: [[100, 200], [300, 200], [300, 250], [100, 250]]

2. Extract Image Zone
   image_zone = ShapeUtils.extract_image_from_zone_rectangle(
     gray_image,
     zone_label,
     scan_zone_rectangle
   )
   # Returns cropped image (NumPy array) of scan zone

3. Initialize EasyOCR (Lazy Loading)
   if EasyOCR.reader is None:
     EasyOCR.initialize()
       ├── import easyocr
       ├── reader = easyocr.Reader(["en"], gpu=True)
       └── Cache in EasyOCR.reader (singleton)
   # Note: Initialization happens once per process

4. Run OCR Detection
   text_detection = EasyOCR.get_single_text_detection(
     image_zone,
     confidence_threshold=0.8,
     clear_whitespace=True
   )

   ↓ Calls read_texts_with_boxes()

   4a. Read All Texts
       text_results = EasyOCR.reader.readtext(image_zone)
       # Returns: [
       #   ([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], "John", 0.95),
       #   ([[x5,y5], [x6,y6], [x7,y7], [x8,y8]], "Doe", 0.82),
       # ]

   4b. Filter by Confidence
       filtered = [
         (box, text, score)
         for (box, text, score) in text_results
         if score >= 0.8
       ]

   4c. Sort by Score (Descending)
       sorted_texts = sorted(
         filtered,
         key=operator.itemgetter(2),  # Sort by score (index 2)
         reverse=True
       )
       # Highest confidence first

   4d. Select Single Detection
       if len(sorted_texts) == 0:
         return None  # No detection above threshold

       box, text, score = sorted_texts[0]  # Highest confidence

       if score <= confidence_threshold:
         return None

   4e. Convert to TextDetection
       return EasyOCR.convert_to_text_detection(
         box, text, score, clear_whitespace=True
       )

5. Convert to TextDetection (Zone-Relative)
   EasyOCR.convert_to_text_detection(box, text, score, clear_whitespace)

   5a. Order Box Points
       ordered_box, _ = MathUtils.order_four_points(box)
       # Ensures consistent ordering: TL, TR, BR, BL

   5b. Post-Process Text
       processed_text = TextOCR.postprocess_text(
         text,
         clear_whitespace=True
       )
       ├── Strip whitespace: text.strip()
       ├── Remove non-ASCII: ord(c) < 128
       ├── Clear excess whitespace: re.sub("\\s{2,}", " ", text)
       └── Return cleaned text

   5c. Create TextDetection
       return TextDetection(
         detected_text=processed_text,  # "John"
         bounding_box=ordered_box,      # Zone-relative coordinates
         rotated_rectangle=ordered_box, # EasyOCR doesn't provide rotation
         confident_score=score          # 0.95
       )
       # Note: Coordinates are relative to image_zone, not full image

6. Convert to Absolute Coordinates
   if text_detection is not None:
     ocr_detection = OCRDetection.from_scan_zone_detection(
       scan_zone_rectangle,
       text_detection
     )

     ↓ Inside from_scan_zone_detection()

     6a. Get Zone Start
         zone_start = scan_zone_rectangle[0]  # Top-left corner [100, 200]

     6b. Shift Bounding Box
         absolute_bounding_box = MathUtils.shift_points_from_origin(
           zone_start,
           text_detection.bounding_box
         )
         # Adds zone_start to each point in bounding_box
         # Zone-relative [10, 5] → Absolute [110, 205]

     6c. Shift Rotated Rectangle
         absolute_rotated_rectangle = MathUtils.shift_points_from_origin(
           zone_start,
           text_detection.rotated_rectangle
         )

     6d. Create OCRDetection
         return OCRDetection(
           scan_zone_rectangle=scan_zone_rectangle,
           detected_text=text_detection.detected_text,
           bounding_box=absolute_bounding_box,
           rotated_rectangle=absolute_rotated_rectangle,
           confident_score=text_detection.confident_score
         )

7. Create Typed Result
   self.detections = [ocr_detection] if ocr_detection else []

   confidence = (
     self.detections[0].confident_score
     if self.detections
     else 0.0
   )

   self.result = OCRFieldDetectionResult(
     field_id=field.id,
     field_label=field.field_label,
     detections=self.detections,
     confidence=confidence,
   )

8. Store in Detection Pass
   Return self.detections (list of OCRDetection)
```

### Output
- `OCRFieldDetection` with:
  - `detections`: List of OCRDetection (0 or 1 item)
  - `result`: OCRFieldDetectionResult (typed model)
- Coordinates in absolute image space
- Confidence score from EasyOCR

### Edge Cases

**No Text Detected**:
```python
# text_results is empty or all below threshold
text_detection = None
self.detections = []
confidence = 0.0
```

**Multiple Texts in Zone** (Future Enhancement):
```python
# Current: Only returns highest confidence
# Future: Return all detections above threshold
text_detection = EasyOCR.get_all_text_detections(
  image_zone, confidence_threshold=0.8
)
# Returns list of TextDetection objects
```

**Low Confidence Detection**:
```python
# score = 0.75, threshold = 0.8
# Detection is filtered out
if score <= confidence_threshold:
  return None
```

---

## Flow 3: OCR Detection Pass (All Fields)

### Trigger
`ReadOMRProcessor.process()` → `OCRDetectionPass.process_all_fields()`

### Steps

```
1. Initialize Detection Pass
   OCRDetectionPass(
     tuning_config=template.tuning_config,
     repository=DetectionRepository()
   )

2. Initialize Aggregates
   2a. Directory Level
       OCRDetectionPass.initialize_directory_level_aggregates(directory_path)
       # No specific OCR directory aggregates currently

   2b. File Level
       OCRDetectionPass.initialize_file_level_aggregates(file_path)
       # No specific OCR file aggregates currently

3. Process Each OCR Field
   For each field in template.ocr_fields:

     3a. Create Field Detection
         field_detection = self.get_field_detection(
           field,
           gray_image,
           colored_image
         )
         # Returns: OCRFieldDetection instance
         # Runs detection during __init__ (see Flow 2)

     3b. Update Field Level Aggregates
         self.update_field_level_aggregates_on_processed_field_detection(
           field,
           field_detection
         )

         ↓ Inside update function

         # Save to repository
         self.repository.save_ocr_field(
           field.id,
           field_detection.result  # OCRFieldDetectionResult
         )

         # Update aggregates
         self.insert_field_level_aggregates({
           "detections": field_detection.detections
         })

     3c. Update File Level Aggregates
         self.update_file_level_aggregates_on_processed_field_detection(
           field,
           field_detection,
           field_level_aggregates
         )

         file_level_aggregates = self.get_file_level_aggregates()
         file_level_aggregates["fields_count"].push("processed")

4. Complete Detection Pass
   Return file_level_aggregates containing:
   {
     "fields_count": {"processed": N},
     "ocr_fields": {  # From repository
       "studentName": OCRFieldDetectionResult(...),
       "rollNumber": OCRFieldDetectionResult(...),
       ...
     }
   }
```

### Output
- All OCR fields detected
- Results saved in DetectionRepository
- Aggregates available for interpretation pass

---

## Flow 4: OCR Interpretation (Single Field)

### Trigger
`OCRInterpretationPass.process_all_fields()` → `OCRFieldInterpretation.run_interpretation()`

### Prerequisites
- Detection pass completed
- Results stored in repository/aggregates

### Steps

```
1. Create Field Interpretation
   OCRFieldInterpretation(
     tuning_config,
     field,
     file_level_detection_aggregates,
     file_level_interpretation_aggregates
   )

2. Initialize from Aggregates
   OCRFieldInterpretation.initialize_from_file_level_aggregates(
     field,
     file_level_detection_aggregates,
     file_level_interpretation_aggregates
   )

   2a. Retrieve Detection Results
       ocr_fields = file_level_detection_aggregates["ocr_fields"]

       if field.field_label not in ocr_fields:
         raise KeyError(f"Field {field_label} not found in ocr_fields")

       ocr_result = ocr_fields[field.field_label]
       # ocr_result: OCRFieldDetectionResult

       detections = ocr_result.detections
       # detections: List[OCRDetection]

   2b. Map Detections to Interpretations
       self.interpretations = [
         OCRInterpretation(detection)
         for detection in detections
       ]
       # Each OCRInterpretation wraps an OCRDetection

   2c. Handle Empty Detections
       if len(self.interpretations) == 0:
         logger.warning(f"No OCR detection for field: {field.id}")

3. Update Common Interpretations
   OCRFieldInterpretation.update_common_interpretations()

   3a. Get Marked Interpretations
       marked_interpretations = [
         interpretation.get_value()
         for interpretation in self.interpretations
         if interpretation.is_attempted
       ]
       # is_attempted = True if detection is not null
       # get_value() returns detected_text

   3b. Set Interpretation Flags
       self.is_attempted = len(marked_interpretations) > 0
       self.is_multi_marked = len(marked_interpretations) > 1
       # Currently max 1 detection per field
       # Future: Multiple detections → multi_marked = True

4. Generate Interpretation String
   OCRFieldInterpretation.get_field_interpretation_string()

   4a. Get All Marked Texts
       marked_interpretations = [
         interpretation.get_value()
         for interpretation in self.interpretations
         if interpretation.is_attempted
       ]

   4b. Handle Empty Case
       if len(marked_interpretations) == 0:
         return self.empty_value  # "" by default

   4c. Concatenate Texts
       return "".join(marked_interpretations)
       # Example: ["John", "Doe"] → "JohnDoe"
       # Future: Support configurable delimiter

5. Store Interpretation
   Return OCRFieldInterpretation with:
   - interpretations: List[OCRInterpretation]
   - is_attempted: bool
   - is_multi_marked: bool
   - get_field_interpretation_string() → final text value
```

### Output
- Interpreted text for field
- Flags for attempted/multi-marked
- Ready for CSV export

### Concatenation Examples

**Single Detection**:
```python
detections = [OCRDetection(detected_text="John", ...)]
interpretations = [OCRInterpretation(detection)]
result = "".join(["John"])  # "John"
```

**Multiple Detections** (Future):
```python
detections = [
  OCRDetection(detected_text="John", ...),
  OCRDetection(detected_text="Doe", ...)
]
interpretations = [OCRInterpretation(d) for d in detections]
result = "".join(["John", "Doe"])  # "JohnDoe"
```

**No Detection**:
```python
detections = []
interpretations = []
result = self.empty_value  # ""
```

---

## Flow 5: Text Post-Processing

### Trigger
During `EasyOCR.convert_to_text_detection()`

### Steps

```
1. Input
   raw_text = "  John  Doe  "  # From EasyOCR

2. Strip Whitespace
   stripped_text = raw_text.strip()
   # "John  Doe"

3. Cleanup Non-ASCII Characters
   TextOCR.cleanup_text(stripped_text)

   printable_text = "".join([
     c for c in stripped_text
     if ord(c) < 128
   ])
   # Removes characters like: é, ñ, 中文, emoji
   # "John  Doe" → "John  Doe" (no change in this case)

4. Clear Excess Whitespace (if enabled)
   if clear_whitespace:
     cleaned_text = re.sub("\\s{2,}", " ", printable_text)
     # "John  Doe" → "John Doe"

5. Filter by Character Set (if specified)
   if charset is not None:
     cleaned_text = TextOCR.filter_text(cleaned_text, charset)
     # Example: charset = TextOCR.alphanumeric_set
     # "John123!@#" → "John123"

6. Clip to Max Length (if specified)
   if max_length is not None and len(cleaned_text) > max_length:
     cleaned_text = cleaned_text[:max_length]
     # "JohnDoeSmith" with max_length=8 → "JohnDoeS"

7. Output
   return cleaned_text
```

### Character Set Options

**Digits Only**:
```python
TextOCR.postprocess_text(
  "Roll123ABC",
  charset=TextOCR.digits_set
)
# Output: "123"
```

**Letters Only**:
```python
TextOCR.postprocess_text(
  "John123",
  charset=TextOCR.letters_set
)
# Output: "John"
```

**Alphanumeric**:
```python
TextOCR.postprocess_text(
  "ID-456ABC",
  charset=TextOCR.alphanumeric_set
)
# Output: "456ABC"
```

**URL Symbols**:
```python
TextOCR.postprocess_text(
  "user@example.com",
  charset=TextOCR.url_symbols_set
)
# Output: "@."
```

---

## Flow 6: Drawing OCR Detections

### Trigger
`OCRFieldInterpretationDrawing.draw_field_interpretation()` during output generation

### Prerequisites
- Interpretation completed
- Output image (colored)

### Steps

```
1. Check if Interpretations Exist
   if len(field_interpretation.interpretations) == 0:
     return  # Nothing to draw

2. Draw Individual Bounding Boxes
   all_bounding_box_points = []

   For each interpretation in field_interpretation.interpretations:
     bounding_box = interpretation.text_detection.bounding_box
     # bounding_box: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

     DrawingUtils.draw_contour(
       marked_image,
       bounding_box,
       color=CLR_BLACK  # (0, 0, 0)
     )

     all_bounding_box_points.extend(bounding_box)

3. Compute Combined Bounding Box
   combined_bounding_box, dimensions = (
     MathUtils.get_bounding_box_of_points(all_bounding_box_points)
   )
   # Returns minimum axis-aligned rectangle enclosing all points
   # Format: [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]

4. Determine Bounding Box Color
   combined_bounding_box_color = CLR_BLACK  # Default

   if evaluation_meta is not None:
     # Color-code based on correctness
     if field_label in evaluation_meta["questions_meta"]:
       question_meta = evaluation_meta["questions_meta"][field_label]

       if field_interpretation.is_attempted or question_meta["bonus_type"]:
         verdict_color = (
           evaluation_config_for_response.get_evaluation_meta_for_question(
             question_meta,
             field_interpretation,
             image_type
           )[1]  # Index 1 is verdict_color
         )
         combined_bounding_box_color = verdict_color

5. Draw Combined Bounding Box
   DrawingUtils.draw_contour(
     marked_image,
     combined_bounding_box,
     color=combined_bounding_box_color
   )

6. Draw Interpreted Text
   6a. Calculate Text Position
       bounding_box_start = combined_bounding_box[0]
       text_position = MathUtils.add_points(
         bounding_box_start,
         (0, max(-10, -1 * bounding_box_start[1]))
       )
       # Position text 10 pixels above bounding box
       # Or at top of image if bounding box is near top

   6b. Get Interpreted Text
       interpreted_text = field_interpretation.get_field_interpretation_string()
       # Example: "John"

   6c. Draw Text
       DrawingUtils.draw_text_responsive(
         marked_image,
         interpreted_text,
         text_position,
         color=CLR_BLACK,
         thickness=3
       )

7. Output
   marked_image (modified in-place) with:
   - Individual detection bounding boxes (black)
   - Combined bounding box (black or verdict color)
   - Interpreted text label above box
```

### Drawing Examples

**Single Detection**:
```
┌─────────────────┐
│  "John"         │  ← Interpreted text (above box)
│  ┌───────────┐  │
│  │   John    │  │  ← Individual bounding box
│  └───────────┘  │  ← Combined bounding box (same as individual)
└─────────────────┘
```

**Multiple Detections** (Future):
```
┌─────────────────────┐
│  "JohnDoe"          │  ← Concatenated text
│  ┌──────┐  ┌─────┐  │
│  │ John │  │ Doe │  │  ← Individual bounding boxes
│  └──────┘  └─────┘  │
│  └──────────────┘   │  ← Combined bounding box
└─────────────────────┘
```

**Evaluation Color Coding**:
```
Green box: Correct answer
Red box: Wrong answer
Orange box: Multi-marked
Black box: No evaluation / not attempted
```

---

## Flow 7: Tesseract Integration (Future)

### Planned Implementation

```
1. Initialize Tesseract
   TesseractOCR.initialize()
   # pytesseract setup

2. Run OCR
   text_detection = TesseractOCR.get_single_text_detection(
     image_zone,
     confidence_threshold=0.8
   )

   2a. Extract Text with Data
       data = pytesseract.image_to_data(
         image_zone,
         output_type=Output.DICT
       )
       # Returns: {
       #   'text': ['John', 'Doe'],
       #   'conf': [95, 88],
       #   'left': [10, 50],
       #   'top': [5, 5],
       #   'width': [40, 35],
       #   'height': [15, 15]
       # }

   2b. Filter by Confidence
       valid_detections = [
         i for i, conf in enumerate(data['conf'])
         if conf >= confidence_threshold * 100  # Tesseract uses 0-100 scale
       ]

   2c. Construct Bounding Boxes
       For each valid detection:
         x = data['left'][i]
         y = data['top'][i]
         w = data['width'][i]
         h = data['height'][i]

         bounding_box = [
           [x, y],
           [x + w, y],
           [x + w, y + h],
           [x, y + h]
         ]

   2d. Select Highest Confidence
       Sort by confidence, return first

3. Convert to TextDetection
   # Same as EasyOCR flow
```

---

## Browser Migration Notes

### Tesseract.js Integration

```typescript
import Tesseract from 'tesseract.js';

// Initialize worker (cache for reuse)
let worker: Tesseract.Worker | null = null;

async function initializeTesseractWorker(): Promise<void> {
  if (worker) return;

  worker = await Tesseract.createWorker('eng', undefined, {
    logger: (m) => console.log(m), // Progress logging
  });
}

async function getSingleTextDetection(
  imageData: ImageData,
  confidenceThreshold: number = 0.8
): Promise<OCRDetection | null> {
  await initializeTesseractWorker();

  // Run OCR
  const { data } = await worker!.recognize(imageData);

  // Get words above threshold
  const validWords = data.words.filter(
    word => (word.confidence / 100) >= confidenceThreshold
  );

  if (validWords.length === 0) return null;

  // Sort by confidence, get highest
  const bestWord = validWords.sort(
    (a, b) => b.confidence - a.confidence
  )[0];

  // Construct bounding box
  const { x0, y0, x1, y1 } = bestWord.bbox;
  const boundingBox = [
    { x: x0, y: y0 },
    { x: x1, y: y0 },
    { x: x1, y: y1 },
    { x: x0, y: y1 },
  ];

  // Post-process text
  const processedText = postprocessText(bestWord.text, {
    clearWhitespace: true,
  });

  return {
    detectedText: processedText,
    boundingBox,
    rotatedRectangle: boundingBox,
    confidenceScore: bestWord.confidence / 100,
    library: 'TESSERACT_JS',
    scanZoneRectangle,
  };
}
```

### Performance Optimization

**Web Worker Pattern**:
```typescript
// offscreen-ocr-worker.ts
import Tesseract from 'tesseract.js';

let worker: Tesseract.Worker;

self.onmessage = async (e) => {
  const { imageData, threshold } = e.data;

  if (!worker) {
    worker = await Tesseract.createWorker('eng');
  }

  const result = await getSingleTextDetection(imageData, threshold);

  self.postMessage({ result });
};

// main.ts
const ocrWorker = new Worker('./offscreen-ocr-worker.ts');

ocrWorker.postMessage({ imageData, threshold: 0.8 });
ocrWorker.onmessage = (e) => {
  const { result } = e.data;
  // Process OCR result
};
```

---

## Performance Considerations

### EasyOCR Performance

**Initialization**:
- First call: ~5-10 seconds (model download + GPU setup)
- Subsequent calls: <100ms (cached reader)

**Detection Time**:
- Small zone (200×50px): ~200-500ms
- Medium zone (400×100px): ~500-1000ms
- Large zone (800×200px): ~1-2 seconds

**GPU Acceleration**:
- Enabled by default: `gpu=True`
- Fallback to CPU if GPU unavailable
- CPU ~3-5x slower than GPU

### Browser (Tesseract.js) Performance

**Initialization**:
- First call: ~2-5 seconds (language model download ~2-4MB)
- Model cached in browser storage

**Detection Time**:
- Small zone: ~1-2 seconds
- Medium zone: ~2-4 seconds
- Large zone: ~4-8 seconds
- **Much slower than Python EasyOCR**

**Optimization**:
1. Cache worker instance (avoid re-initialization)
2. Use Web Workers (offload from main thread)
3. Process multiple fields in parallel (worker pool)
4. Pre-download language models
5. Use WASM-optimized build

---

## Related Documentation

- **Concept**: `modules/domain/detection/ocr/concept.md`
- **Decisions**: `modules/domain/detection/ocr/decisions.md`
- **Constraints**: `modules/domain/detection/ocr/constraints.md`
- **Integration**: `modules/domain/detection/ocr/integration.md`
- **ReadOMR**: `modules/domain/detection/concept.md`

---

## Summary

OCR detection flows provide:

1. **Field Setup**: Template-driven scan zone definition
2. **EasyOCR Detection**: High-confidence text extraction with lazy loading
3. **Detection Pass**: Batch processing across all OCR fields
4. **Interpretation**: Text concatenation and empty value handling
5. **Post-Processing**: Text cleanup, filtering, normalization
6. **Visualization**: Bounding boxes and text labels on output images
7. **Browser Migration**: Tesseract.js with Web Worker optimization

**Typical Flow**: Template Setup → Scan Zone Extraction → EasyOCR Detection → Coordinate Transformation → Typed Result Storage → Interpretation → Drawing → CSV Export
