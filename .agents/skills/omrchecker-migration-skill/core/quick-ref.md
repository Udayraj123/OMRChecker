# OMRChecker Quick Reference

**Document Type**: Quick Reference for Developers
**Last Updated**: 2026-02-20

---

## Common Operations

### 1. Process an OMR Sheet (End-to-End)

**Python Flow** (src/entry.py):
```
Input Image → Template Loading → Preprocessing → Alignment → Detection → Evaluation → Output CSV
```

**Key Steps**:
1. Load image (gray + colored): `ImageUtils.read_image_util()`
2. Load template: `Template(template_path, tuning_config, args)`
3. Create context: `ProcessingContext(file_path, gray_image, colored_image, template)`
4. Run pipeline: `template.pipeline.process_file()`
5. Extract response: `context.omr_response`
6. Evaluate (if config exists): `evaluate_concatenated_response()`
7. Write CSV: `thread_safe_csv_append(results_file, results_line)`

**Code Reference**: `src/entry.py:277-530` (process_single_file function)

---

### 2. Detect Bubbles (Threshold-Based)

**Algorithm** (src/processors/detection/bubbles_threshold/bubble_interpretation.py):

1. **Extract ROI**: Crop bubble region from aligned image
2. **Threshold**: Apply global/adaptive/local threshold
3. **Calculate Darkness**: `bubble_value = sum(thresholded_pixels) / total_pixels`
4. **Interpret**:
   - If `bubble_value > MARKER_FILL_THRESHOLD` → Marked
   - If `bubble_value <= MARKER_FILL_THRESHOLD` → Unmarked
5. **Multi-mark Detection**: If multiple bubbles marked in single field → `is_multi_marked = True`

**Key Parameters**:
- `MARKER_FILL_THRESHOLD`: Default 0.5 (50% darkness)
- Adjustable per field block in template

**Code Reference**: `src/processors/detection/bubbles_threshold/bubble_interpretation.py`

---

### 3. Align Image to Template

**Strategies** (src/processors/alignment/):

| Strategy | Use Case | Code Reference |
|----------|----------|----------------|
| **SIFT** | Feature-rich images | `alignment/sift_matcher.py` |
| **Phase Correlation** | Translation-only shifts | `alignment/phase_correlation.py` |
| **Template Matching** | Simple alignment | `alignment/template_alignment.py` |
| **Piecewise Affine** | Non-linear distortions | `alignment/piecewise_affine_delaunay.py` |

**Common Flow**:
1. Detect features in input image
2. Match features with template
3. Compute transformation matrix (homography/affine)
4. Warp image to align with template

**Code Reference**: `src/processors/alignment/processor.py:46-120`

---

### 4. Preprocess Image

**Pipeline** (src/processors/image/coordinator.py):

```
Input → AutoRotate → CropOnMarkers/CropPage → Filters (Blur/Contrast/Levels) → Aligned Image
```

**Common Preprocessors**:
- `AutoRotate`: Correct image rotation (90°, 180°, 270°)
- `CropOnMarkers`: Crop based on detected markers (dots, lines, custom)
- `CropPage`: Detect page boundaries and crop
- `GaussianBlur`: Reduce noise
- `Contrast`: Enhance contrast
- `Levels`: Adjust black/white levels

**Configuration** (template.json):
```json
{
  "preProcessors": [
    {"name": "AutoRotate"},
    {"name": "CropOnMarkers", "options": {"markerType": "dots"}},
    {"name": "GaussianBlur", "options": {"kSize": 5}}
  ]
}
```

**Code Reference**: `src/processors/image/coordinator.py`

---

### 5. Load and Validate Template

**Template Schema** (src/schemas/template_schema.py):

```json
{
  "pageDimensions": [200, 400],
  "bubbleDimensions": [10, 10],
  "preProcessors": [...],
  "fieldBlocks": {
    "block1": {
      "fieldType": "QTYPE_INT",
      "bubbleValues": ["1", "2", "3", "4"],
      "origin": [10, 20],
      "fieldLabels": ["q1", "q2", "q3"]
    }
  }
}
```

**Validation**:
1. JSON syntax check
2. Pydantic schema validation
3. Field block dimension validation
4. Bubble value uniqueness
5. Origin coordinates within page bounds

**Code Reference**: `src/schemas/template_schema.py`, `src/processors/template/template.py`

---

### 6. Evaluate Response

**Evaluation Flow** (src/processors/evaluation/):

1. **Load Answer Key** (evaluation.json)
2. **Match Response to Answer Key**: Field-by-field comparison
3. **Apply Marking Scheme**:
   - Correct answer: +marks
   - Wrong answer: -marks (negative marking)
   - Unattempted: 0
4. **Calculate Score**: Sum across all sections
5. **Generate Explanation**: Show correct/wrong/unattempted breakdown

**Marking Scheme Types**:
- Default: +1 correct, 0 wrong
- Negative marking: +4 correct, -1 wrong
- Partial credit: +1 full, +0.5 partial
- Section-wise: Different schemes per section

**Code Reference**: `src/processors/evaluation/evaluation_meta.py:10-150`

---

### 7. Handle Multi-Marked Fields

**Detection** (src/processors/detection/bubbles_threshold/interpretation_pass.py):

```python
marked_bubbles = [b for b in bubbles if b.value > threshold]
if len(marked_bubbles) > 1:
    is_multi_marked = True
    # Field value = concatenated marked values
    field_value = ",".join([b.label for b in marked_bubbles])
```

**Handling**:
- **Strict Mode**: Multi-marked = Error (file moved to multi-marked dir)
- **Lenient Mode**: Multi-marked = Record all marked values (comma-separated)

**Configuration** (config.json):
```json
{
  "outputs": {
    "filter_out_multimarked_files": true
  }
}
```

**Code Reference**: `src/processors/detection/bubbles_threshold/interpretation_pass.py`

---

### 8. Export Results to CSV

**CSV Format**:
```
file_name, input_path, output_path, score, q1, q2, q3, ..., qN
```

**Thread-Safe Writing** (src/utils/csv.py):
```python
from threading import Lock
csv_lock = Lock()

def thread_safe_csv_append(file_path, row):
    with csv_lock:
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
```

**Output Files**:
- `Results.csv`: Successfully processed files
- `Errors.csv`: Files with errors (no markers, etc.)
- `MultiMarked.csv`: Files with multi-marked fields

**Code Reference**: `src/utils/csv.py`, `src/entry.py:485-522`

---

### 9. Debug with Visual Outputs

**Debug Levels** (config.json):
```json
{
  "outputs": {
    "show_image_level": 4  // 0=none, 1-6=increasing detail
  }
}
```

**Save Operations** (src/processors/template/template.py):
```python
template.save_image_ops.append_save_image(
    "Preprocessing Step 1",
    range(1, 7),  # Show at debug levels 1-6
    gray_image,
    colored_image
)

template.save_image_ops.save_image_stacks(
    file_path,
    output_dir,
    images_per_row=5
)
```

**Output Structure**:
```
outputs/
├── CheckedOMRs/
│   ├── file1.png (final marked image)
│   └── colored/file1.png (colored version)
└── ImageAnalysis/
    └── file1/ (debug images for each step)
```

**Code Reference**: `src/processors/template/template.py`

---

### 10. Configure Parallel Processing

**ThreadPoolExecutor** (src/entry.py):
```python
max_workers = tuning_config.processing.max_parallel_workers

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_single_file, info): info
               for info in file_tasks}

    for future in as_completed(futures):
        result = future.result()
```

**Configuration** (config.json):
```json
{
  "processing": {
    "max_parallel_workers": 4  // 1-8 recommended
  }
}
```

**Note**: Parallel processing disabled if `show_image_level > 0` (interactive mode)

**Code Reference**: `src/entry.py:532-642`

---

## Browser Migration Quick Tips

### Python → JavaScript Equivalents

| Python | JavaScript/Browser |
|--------|-------------------|
| OpenCV (`cv2`) | OpenCV.js (WebAssembly) |
| NumPy arrays | TypedArrays (Uint8Array, Float32Array) |
| Pydantic schemas | Zod validation or JSON Schema |
| ThreadPoolExecutor | Web Workers |
| File I/O (`pathlib`) | File API, Blob, FileReader |
| Argparse CLI | UI controls (buttons, inputs) |
| PyZbar | @zxing/library or jsQR |
| EasyOCR/Tesseract | Tesseract.js |

### Key Adaptations

1. **Image Loading**: Use `<input type="file">` + FileReader + Canvas
2. **Parallel Processing**: Spawn Web Workers for each file
3. **Memory Management**: Process images in batches, clear arrays explicitly
4. **Output**: Download CSV as Blob, LocalStorage for caching
5. **ML Models**: Convert YOLO to TensorFlow.js or ONNX Runtime Web

---

## Most Common Edge Cases

### 1. Rotated Images
- **Detection**: Check orientation via EXIF or image analysis
- **Correction**: Rotate by 90°, 180°, 270° increments
- **Code**: `src/processors/image/AutoRotate.py`

### 2. Poor Lighting
- **Solution**: Adaptive thresholding instead of global
- **Config**: `thresholding.threshold_mode = "adaptive"`
- **Code**: `src/processors/threshold/adaptive_threshold.py`

### 3. Perspective Distortion
- **Solution**: 4-point perspective correction
- **Detection**: Find page corners via contours
- **Code**: `src/processors/image/page_detection.py`, `warp_strategies.py`

### 4. Xeroxed/Degraded Sheets
- **Solution**: Increase blur, adjust contrast
- **Config**: Add `GaussianBlur` and `Contrast` preprocessors
- **Code**: `src/processors/image/GaussianBlur.py`, `Contrast.py`

### 5. Stray Marks
- **Solution**: Increase `MARKER_FILL_THRESHOLD` to ignore light marks
- **Config**: Adjust per field block in template
- **Code**: `src/processors/detection/bubbles_threshold/interpretation.py`

---

## Code Reference Format

All code references follow the pattern:
```
<file_path>:<line_range>
```

Example:
- `src/entry.py:277-530` → Function process_single_file, lines 277-530
- `src/processors/alignment/sift_matcher.py` → Entire file

Use these to validate current behavior against documentation.

---

## Next Steps for Migration

1. **Start with Core Flow**: Implement main pipeline (Task 3.1)
2. **Add Preprocessing**: Implement rotation, cropping (Tasks 3.2-3.7)
3. **Implement Detection**: Start with threshold-based bubbles (Tasks 4.2-4.7)
4. **Add Alignment**: Implement SIFT or template matching (Tasks 3.8-3.13)
5. **Add Evaluation**: Implement scoring logic (Tasks 6.1-6.5)
6. **Test Edge Cases**: Validate against Python version's test suite

---

**For detailed documentation on any topic, load the corresponding module from `.agents/skills/omrchecker-migration-skill/modules/`**
