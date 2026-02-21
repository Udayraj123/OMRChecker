# Barcode Detection - Flow Details

**Module**: Barcode Detection
**Python Reference**: `src/processors/detection/barcode/`
**Focus**: Detection algorithms, decoding process, PyZbar integration

---

## Flow 1: Detection Pass - Barcode Extraction

### Entry Point
```python
# src/processors/detection/barcode/detection_pass.py
class BarcodeDetectionPass(FieldTypeDetectionPass):
    """Orchestrates barcode detection for all barcode fields."""
```

### Step-by-Step Flow

```
START: process_file_fields(template, file_path, gray_image, colored_image)
    ↓
    1. Initialize file-level aggregates
       ├─> fields_count = {"processed": []}
       └─> repository.initialize_file(file_path)
    ↓
    2. Filter barcode fields from template
       ├─> For field_block in template.field_blocks:
       │       If field_block.field_detection_type == "BARCODE_QR":
       │           For field in field_block.fields:
       │               barcode_fields.append(field)
    ↓
    3. Process each barcode field
       ├─> For field in barcode_fields:
       │   ├─> Extract scan zone coordinates
       │   │   scan_box = field.scan_boxes[0]
       │   │   zone_rectangle = scan_box.scan_zone_rectangle
       │   │   # Example: [(100,50), (400,50), (400,130), (100,130)]
       │   │
       │   ├─> Crop image to scan zone
       │   │   image_zone = extract_image_from_zone_rectangle(
       │   │       gray_image, zone_rectangle
       │   │   )
       │   │   # Now: image_zone is 300x80 px (from original coordinates)
       │   │
       │   ├─> Decode barcode with PyZBar
       │   │   text_detection = PyZBar.get_single_text_detection(
       │   │       image_zone,
       │   │       confidence_threshold=0.8
       │   │   )
       │   │   # Returns: TextDetection(text, bbox, polygon, score)
       │   │   #    OR: None (if no barcode found)
       │   │
       │   ├─> Convert to absolute coordinates
       │   │   if text_detection:
       │   │       zone_start = zone_rectangle[0]  # (100, 50)
       │   │
       │   │       # Shift bounding box from zone coords to image coords
       │   │       absolute_bbox = shift_points_from_origin(
       │   │           zone_start,
       │   │           text_detection.bounding_box
       │   │       )
       │   │       # Example:
       │   │       #   Zone bbox: [(10,5), (290,5), (290,75), (10,75)]
       │   │       #   Absolute bbox: [(110,55), (390,55), (390,125), (110,125)]
       │   │
       │   │       absolute_polygon = shift_points_from_origin(
       │   │           zone_start,
       │   │           text_detection.rotated_rectangle
       │   │       )
       │   │
       │   │       detection = BarcodeDetection(
       │   │           scan_zone_rectangle=zone_rectangle,
       │   │           detected_text=text_detection.detected_text,
       │   │           bounding_box=absolute_bbox,
       │   │           rotated_rectangle=absolute_polygon,
       │   │           confident_score=text_detection.confident_score
       │   │       )
       │   │       detections = [detection]
       │   │   else:
       │   │       detections = []  # No barcode found
       │   │
       │   ├─> Create typed result
       │   │   result = BarcodeFieldDetectionResult(
       │   │       field_id=field.id,
       │   │       field_label=field.field_label,
       │   │       detections=detections,
       │   │       timestamp=datetime.now()
       │   │   )
       │   │
       │   └─> Save to repository
       │       repository.save_barcode_field(field.id, result)
       │       fields_count["processed"].append(field.id)
    ↓
    4. Return file-level aggregates
       └─> {
               "fields_count": {"processed": ["RollNumber", "SerialQR"]},
               "barcode_fields": {  # Populated by repository
                   "RollNumber": BarcodeFieldDetectionResult(...),
                   "SerialQR": BarcodeFieldDetectionResult(...)
               }
           }
    ↓
END: Detection results stored in repository
```

### Code Example

```python
# BarcodeDetectionPass.process_file_fields()
def process_file_fields(self, template, file_path, gray_image, colored_image):
    # 1. Initialize
    self.initialize_file_level_aggregates(file_path)

    # 2. Get barcode fields
    barcode_fields = [
        field
        for field_block in template.field_blocks
        for field in field_block.fields
        if field_block.field_detection_type == FieldDetectionType.BARCODE_QR
    ]

    # 3. Process each field
    for field in barcode_fields:
        # Create field detection
        field_detection = self.get_field_detection(
            field, gray_image, colored_image
        )

        # Update aggregates and save to repository
        self.update_field_level_aggregates_on_processed_field_detection(
            field, field_detection
        )

        self.update_file_level_aggregates_on_processed_field_detection(
            field, field_detection, field_level_aggregates
        )

    # 4. Return aggregates
    return self.get_file_level_aggregates()
```

---

## Flow 2: PyZBar Decoding - Single Barcode

### Entry Point
```python
# src/processors/detection/barcode/lib/pyzbar.py
PyZBar.get_single_text_detection(image, confidence_threshold=0.8)
```

### Algorithm

```
START: get_single_text_detection(image_zone, confidence_threshold)
    ↓
    1. Check if PyZBar initialized
       ├─> if PyZBar.decode_barcode is None:
       │       logger.info("Initializing PyZBar reader...")
       │       from pyzbar.pyzbar import decode as decode_barcode
       │       PyZBar.decode_barcode = decode_barcode
    ↓
    2. Decode all barcodes in image
       ├─> text_results = PyZBar.decode_barcode(image_zone)
       │   # Returns: List[Decoded] objects from ZBar C library
       │   # Example:
       │   #   [
       │   #       Decoded(
       │   #           data=b'STU-2024-12345',
       │   #           type='QRCODE',
       │   #           quality=95,
       │   #           rect=Rect(top=10, left=5, width=280, height=70),
       │   #           polygon=[Point(5,10), Point(285,10), ...]
       │   #       )
       │   #   ]
    ↓
    3. Filter by confidence threshold
       ├─> filtered_results = [
       │       result
       │       for result in text_results
       │       if result.quality >= confidence_threshold * 100
       │   ]
       │   # quality is 0-100 scale, threshold is 0.0-1.0
    ↓
    4. Parse to standard form
       ├─> For each result:
       │   ├─> Extract bounding box
       │   │   rect = result.rect
       │   │   bounding_box = get_rectangle_points(
       │   │       rect.top, rect.left, rect.width, rect.height
       │   │   )
       │   │   # Returns: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
       │   │
       │   ├─> Extract detected text
       │   │   detected_text = str(result.data)
       │   │   # Converts bytes to string
       │   │   # b'STU-2024-12345' → 'STU-2024-12345'
       │   │
       │   ├─> Extract score
       │   │   score = float(result.quality) / 100.0
       │   │   # Normalize 0-100 to 0.0-1.0
       │   │
       │   └─> Extract rotated polygon
       │       rotated_rectangle = [
       │           [point.x, point.y]
       │           for point in result.polygon
       │       ]
       │       # Actual barcode corners (may be rotated)
    ↓
    5. Sort by score (descending)
       ├─> filtered_results.sort(key=lambda x: x[2], reverse=True)
       │   # Index 2 is score in (bbox, text, score, polygon) tuple
    ↓
    6. Select best result
       ├─> if len(filtered_results) == 0:
       │       return None  # No barcode found or all below threshold
       │
       │   box, text, score, polygon = filtered_results[0]
       │
       │   # Check threshold again (redundant but safe)
       │   if score <= confidence_threshold:
       │       return None
    ↓
    7. Convert to TextDetection
       ├─> Order bounding box points
       │   ordered_box, _ = order_four_points(box)
       │   # Ensures consistent ordering: TL, TR, BR, BL
       │
       ├─> Postprocess text
       │   processed_text = TextBarcode.postprocess_text(
       │       text,
       │       clear_whitespace=True
       │   )
       │   # Steps:
       │   #   1. Strip leading/trailing whitespace
       │   #   2. Remove non-ASCII characters
       │   #   3. Collapse multiple spaces to single space
       │   #   4. Optional: filter by charset
       │   #   5. Optional: clip to max length
       │
       └─> return TextDetection(
               detected_text=processed_text,
               bounding_box=ordered_box,
               rotated_rectangle=polygon,
               confident_score=score
           )
    ↓
END: Return TextDetection or None
```

### Text Postprocessing Details

```python
# src/processors/detection/barcode/lib/text_barcode.py
def postprocess_text(text, clear_whitespace=False, max_length=None, charset=None):
    # 1. Strip whitespace
    stripped_text = text.strip()
    # "  STU-2024-12345  " → "STU-2024-12345"

    # 2. Remove non-ASCII characters
    printable_text = "".join([c for c in stripped_text if ord(c) < 128])
    # "STU-2024-12345™" → "STU-2024-12345"

    # 3. Collapse multiple whitespace (if enabled)
    if clear_whitespace:
        cleaned_text = re.sub(r'\s{2,}', ' ', printable_text)
        # "STU  2024    12345" → "STU 2024 12345"

    # 4. Filter by charset (if provided)
    if charset is not None:
        cleaned_text = "".join([c for c in cleaned_text if c in charset])
        # charset="0123456789-" → "2024-12345"

    # 5. Clip to max length (if provided)
    if max_length is not None and len(cleaned_text) > max_length:
        cleaned_text = cleaned_text[:max_length]
        # max_length=10 → "STU-2024-1"

    return cleaned_text
```

---

## Flow 3: Interpretation Pass - Text Extraction

### Entry Point
```python
# src/processors/detection/barcode/interpretation_pass.py
class BarcodeInterpretationPass(FieldTypeInterpretationPass):
    """Interprets barcode detections into field values."""
```

### Algorithm

```
START: process_file_fields(template, file_level_detection_aggregates)
    ↓
    1. Initialize file-level interpretation aggregates
       ├─> self.initialize_file_level_aggregates(file_path)
       └─> file_level_aggregates = {}
    ↓
    2. Get barcode fields (same as detection pass)
       ├─> barcode_fields = filter(
       │       lambda field: field.field_detection_type == "BARCODE_QR",
       │       template.all_fields
       │   )
    ↓
    3. Process each field
       ├─> For field in barcode_fields:
       │   │
       │   ├─> Create field interpretation
       │   │   field_interpretation = BarcodeFieldInterpretation(
       │   │       tuning_config,
       │   │       field,
       │   │       file_level_detection_aggregates,
       │   │       file_level_aggregates
       │   │   )
       │   │
       │   ├─> Load detections from repository
       │   │   barcode_fields = file_level_detection_aggregates["barcode_fields"]
       │   │   barcode_result = barcode_fields[field.field_label]
       │   │   detections = barcode_result.detections
       │   │   # Example:
       │   │   #   [BarcodeDetection(text="STU-2024-12345", score=0.95)]
       │   │
       │   ├─> Map detections to interpretations
       │   │   interpretations = [
       │   │       BarcodeInterpretation(detection)
       │   │       for detection in detections
       │   │   ]
       │   │   # Each interpretation wraps a detection
       │   │
       │   ├─> Check if attempted
       │   │   marked_interpretations = [
       │   │       interpretation.get_value()
       │   │       for interpretation in interpretations
       │   │       if interpretation.is_attempted
       │   │   ]
       │   │   is_attempted = len(marked_interpretations) > 0
       │   │   is_multi_marked = len(marked_interpretations) > 1
       │   │   # is_attempted: True if any barcode detected
       │   │   # is_multi_marked: True if multiple barcodes (unusual)
       │   │
       │   ├─> Generate field value string
       │   │   if len(marked_interpretations) == 0:
       │   │       field_value = empty_value  # "" or configured default
       │   │   else:
       │   │       field_value = "".join(marked_interpretations)
       │   │   # Example: "STU-2024-12345"
       │   │
       │   └─> Update aggregates
       │       field_level_aggregates = {
       │           "is_multi_marked": is_multi_marked,
       │           "field_value": field_value
       │       }
       │       file_level_aggregates[field.id] = field_level_aggregates
    ↓
    4. Return interpretation results
       └─> {
               "RollNumber": "STU-2024-12345",
               "SerialQR": "BATCH-A-001",
               "read_response_flags": {
                   "is_multi_marked": False
               }
           }
    ↓
END: Interpreted values available for response generation
```

### Code Example

```python
# BarcodeFieldInterpretation.get_field_interpretation_string()
def get_field_interpretation_string(self):
    # Get all detected barcodes
    marked_interpretations = [
        interpretation.get_value()
        for interpretation in self.interpretations
        if interpretation.is_attempted
    ]

    # Handle empty case
    if len(marked_interpretations) == 0:
        return self.empty_value

    # Concatenate multiple barcodes (if enabled)
    # TODO: Add configuration for concatenation vs. first-only
    return "".join(marked_interpretations)
```

---

## Flow 4: Drawing - Visualization

### Entry Point
```python
# src/processors/detection/barcode/interpretation_drawing.py
class BarcodeFieldInterpretationDrawing(FieldInterpretationDrawing):
    """Draws barcode detection results on output image."""
```

### Algorithm

```
START: draw_field_interpretation(marked_image, image_type, evaluation_meta, evaluation_config)
    ↓
    1. Check if any detections exist
       ├─> if len(field_interpretation.interpretations) == 0:
       │       return  # Nothing to draw
    ↓
    2. Draw individual bounding boxes
       ├─> all_bounding_box_points = []
       │
       │   For interpretation in field_interpretation.interpretations:
       │       bounding_box = interpretation.text_detection.bounding_box
       │
       │       # Draw black contour around barcode
       │       DrawingUtils.draw_contour(
       │           marked_image,
       │           bounding_box,
       │           color=CLR_BLACK,
       │           thickness=2
       │       )
       │
       │       # Collect all points for combined box
       │       all_bounding_box_points.extend(bounding_box)
    ↓
    3. Calculate combined bounding box
       ├─> combined_bbox, dimensions = get_bounding_box_of_points(
       │       all_bounding_box_points
       │   )
       │   # Smallest box containing all individual boxes
       │   # Example: [(110,55), (390,55), (390,125), (110,125)]
    ↓
    4. Determine verdict color (if evaluation enabled)
       ├─> combined_bbox_color = CLR_BLACK  # Default
       │
       │   if evaluation_meta and evaluation_config:
       │       if field_label in evaluation_meta["questions_meta"]:
       │           question_meta = evaluation_meta["questions_meta"][field_label]
       │
       │           if field_interpretation.is_attempted or question_meta["bonus_type"]:
       │               verdict_color = get_evaluation_meta_for_question(
       │                   question_meta, field_interpretation, image_type
       │               )
       │               # Returns:
       │               #   GREEN: Correct barcode
       │               #   RED: Incorrect barcode
       │               #   ORANGE: Partial credit
       │
       │               combined_bbox_color = verdict_color
    ↓
    5. Draw combined bounding box
       ├─> DrawingUtils.draw_contour(
       │       marked_image,
       │       combined_bbox,
       │       color=combined_bbox_color,
       │       thickness=3
       │   )
    ↓
    6. Calculate text position
       ├─> bbox_start = combined_bbox[0]  # Top-left corner
       │
       │   # Position text above bounding box
       │   text_position = add_points(
       │       bbox_start,
       │       (0, max(-10, -1 * bbox_start[1]))
       │   )
       │   # Offset: 0 pixels right, 10 pixels up (or to image top)
       │   # max() ensures text stays within image bounds
    ↓
    7. Draw interpreted text
       ├─> interpreted_text = field_interpretation.get_field_interpretation_string()
       │   # Example: "STU-2024-12345"
       │
       │   DrawingUtils.draw_text_responsive(
       │       marked_image,
       │       interpreted_text,
       │       text_position,
       │       color=CLR_BLACK,
       │       thickness=3
       │   )
       │   # draw_text_responsive() auto-scales font size based on image resolution
    ↓
END: Barcode visualization complete
```

### Visual Output

```
Image Layout:
┌────────────────────────────────────────┐
│                                        │
│  STU-2024-12345  ← Interpreted text   │
│  ┌──────────────────────────────────┐ │
│  │ ┌─────────────────────────────┐  │ │ ← Combined box (green/red/black)
│  │ │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   │  │ │
│  │ │  ▓  Barcode Data Here  ▓    │  │ │ ← Individual box (black)
│  │ │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   │  │ │
│  │ └─────────────────────────────┘  │ │
│  └──────────────────────────────────┘ │
│                                        │
└────────────────────────────────────────┘
```

---

## Flow 5: Multi-Barcode Support (Future Enhancement)

### Current Limitation
```python
# Currently: Only returns single best barcode
text_detection = PyZBar.get_single_text_detection(image_zone)
```

### Proposed Enhancement

```
START: Support multiple barcodes per field
    ↓
    1. Update detection configuration
       ├─> field_block["barcodeConfig"] = {
       │       "maxBarcodes": 3,           # Allow up to 3 barcodes
       │       "concatenationEnabled": True,
       │       "concatenationSeparator": ","
       │   }
    ↓
    2. Use get_all_text_detections()
       ├─> text_detections = PyZBar.get_all_text_detections(
       │       image_zone,
       │       confidence_threshold=0.8
       │   )
       │   # Returns: List[TextDetection]
    ↓
    3. Limit to maxBarcodes
       ├─> text_detections = text_detections[:max_barcodes]
    ↓
    4. Create multiple BarcodeDetection objects
       ├─> detections = [
       │       BarcodeDetection.from_scan_zone_detection(
       │           scan_zone_rectangle, td
       │       )
       │       for td in text_detections
       │   ]
    ↓
    5. Interpretation: Concatenate or select first
       ├─> if concatenation_enabled:
       │       field_value = separator.join(
       │           [d.detected_text for d in detections]
       │       )
       │       # Example: "ABC,DEF,GHI"
       │   else:
       │       field_value = detections[0].detected_text
       │       # Example: "ABC"
    ↓
END: Multi-barcode support
```

---

## Flow 6: Format-Specific Decoding

### PyZBar Format Detection

```python
# PyZBar automatically detects barcode type
result.type  # Returns: 'QRCODE', 'CODE128', 'EAN13', etc.

# Example results:
{
    'QRCODE': {
        'data': b'https://example.com/student/12345',
        'quality': 95,
        'polygon': 4 points  # Always rectangular
    },
    'CODE128': {
        'data': b'STU-2024-12345',
        'quality': 88,
        'polygon': 4 points  # May be skewed
    },
    'EAN13': {
        'data': b'1234567890128',
        'quality': 92,
        'polygon': 4 points
    }
}
```

### Format-Specific Handling (Future)

```python
# Proposed: Format-specific configuration
field_block["barcodeConfig"] = {
    "allowedFormats": ["QRCODE", "CODE128"],
    "formatPriority": ["QRCODE", "CODE128"],  # Prefer QR over CODE128
    "formatSpecificConfig": {
        "QRCODE": {
            "errorCorrection": "H",  # High error correction
            "version": "auto"
        },
        "CODE128": {
            "charset": "ASCII",
            "maxLength": 50
        }
    }
}

# Detection: Filter by allowed formats
def filter_by_format(results, allowed_formats):
    return [
        r for r in results
        if r.type in allowed_formats
    ]

# Prioritize by format
def prioritize_by_format(results, format_priority):
    for format_type in format_priority:
        matching = [r for r in results if r.type == format_type]
        if matching:
            return matching[0]
    return results[0] if results else None
```

---

## Browser Migration: @zxing/library Flows

### Flow: Decode from Canvas

```typescript
// Browser equivalent of PyZBar.get_single_text_detection()
import { BrowserMultiFormatReader, DecodeHintType, BarcodeFormat } from '@zxing/library';

async function getSingleTextDetection(
    canvas: HTMLCanvasElement,
    confidenceThreshold: number = 0.8
): Promise<BarcodeDetectionResult | null> {
    // 1. Initialize reader with format hints
    const hints = new Map();
    hints.set(DecodeHintType.POSSIBLE_FORMATS, [
        BarcodeFormat.QR_CODE,
        BarcodeFormat.CODE_128,
        BarcodeFormat.EAN_13
    ]);

    const reader = new BrowserMultiFormatReader(hints);

    try {
        // 2. Decode barcode
        const result = await reader.decodeFromCanvas(canvas);

        // 3. Extract result points (corners)
        const resultPoints = result.getResultPoints();
        const polygon = resultPoints.map(pt => [pt.getX(), pt.getY()]);

        // 4. Calculate bounding box from polygon
        const xs = polygon.map(p => p[0]);
        const ys = polygon.map(p => p[1]);
        const bounding_box = [
            [Math.min(...xs), Math.min(...ys)],
            [Math.max(...xs), Math.min(...ys)],
            [Math.max(...xs), Math.max(...ys)],
            [Math.min(...xs), Math.max(...ys)]
        ];

        // 5. Return TextDetection equivalent
        return {
            detected_text: result.getText(),
            bounding_box: bounding_box,
            rotated_rectangle: polygon,
            confident_score: 1.0,  // ZXing doesn't provide quality score
            format: result.getBarcodeFormat()
        };

    } catch (error) {
        // No barcode found or decoding failed
        console.warn('Barcode decode failed:', error);
        return null;
    } finally {
        reader.reset();
    }
}
```

### Flow: Web Worker Parallel Processing

```typescript
// Main thread: Dispatch barcode fields to workers
async function processBarcodesInParallel(
    fields: BarcodeField[],
    image: ImageData
): Promise<BarcodeDetectionResult[]> {
    const workerPool = new BarcodeWorkerPool(4);

    const tasks = fields.map(field => ({
        field_id: field.id,
        scan_zone: field.scan_boxes[0].scan_zone_rectangle,
        image: image
    }));

    const results = await workerPool.executeTasks(tasks);
    return results;
}

// Worker thread: Decode barcode
// barcode-worker.js
import { BrowserMultiFormatReader } from '@zxing/library';

self.onmessage = async (event) => {
    const { field_id, scan_zone, image } = event.data;

    // 1. Extract scan zone from image
    const zone_image = extractScanZone(image, scan_zone);

    // 2. Create canvas from ImageData
    const canvas = new OffscreenCanvas(zone_image.width, zone_image.height);
    const ctx = canvas.getContext('2d');
    ctx.putImageData(zone_image, 0, 0);

    // 3. Decode barcode
    const reader = new BrowserMultiFormatReader();
    try {
        const result = await reader.decodeFromCanvas(canvas);

        self.postMessage({
            field_id: field_id,
            detected_text: result.getText(),
            success: true
        });
    } catch (error) {
        self.postMessage({
            field_id: field_id,
            detected_text: null,
            success: false
        });
    }
};
```

---

## Performance Metrics

### Python (PyZBar)

```
Barcode Detection Performance:
- QR Code (300x300): ~15-30ms
- CODE-128 (300x80): ~10-20ms
- Multiple formats: ~20-40ms
- Memory per field: ~1-2MB

Optimization:
- Scan zone cropping: 2x-3x faster
- Grayscale images: 1.5x faster
- Format hints: 1.2x faster (future)
```

### Browser (@zxing/library)

```
Barcode Detection Performance:
- QR Code (300x300): ~50-100ms
- CODE-128 (300x80): ~40-80ms
- Multiple formats: ~80-150ms
- Memory per field: ~3-5MB

Optimization:
- Web Workers: 3x-4x faster (parallel)
- OffscreenCanvas: 1.3x faster
- Format hints: 1.5x faster
- WASM backend: 2x faster (if available)
```

---

## Summary

**Key Flows**:
1. **Detection Pass**: Extract barcode from scan zone, save to repository
2. **PyZBar Decoding**: Singleton initialization, confidence filtering, text postprocessing
3. **Interpretation Pass**: Load detections, generate field value, check multi-marking
4. **Drawing**: Visualize bounding boxes, display interpreted text, color by evaluation
5. **Multi-Barcode**: Future support for multiple barcodes per field
6. **Format-Specific**: Future support for format filtering and prioritization

**Browser Migration**:
- Replace `PyZBar.get_single_text_detection()` with `BrowserMultiFormatReader.decodeFromCanvas()`
- Use Web Workers for parallel processing
- Handle missing quality scores (ZXing limitation)
- Implement coordinate conversion from ResultPoints

**Next**: See `decisions.md` for format selection logic and configuration patterns.
