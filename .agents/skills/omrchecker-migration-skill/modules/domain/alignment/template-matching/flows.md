# Template Matching Alignment - Execution Flow

## Overview

**Template Matching** is a template-based alignment method that uses **normalized cross-correlation** to detect the position of a reference image within a scanned image. While NOT currently used as the primary alignment strategy in `template_alignment.py`, it serves as a foundational technique used in several OMRChecker components.

**Primary Use Cases**:
- **AutoRotate**: Detecting reference markers for rotation correction
- **CropOnCustomMarkers**: Locating custom markers for perspective correction
- **Multi-scale marker detection**: Finding markers at different scales

**Algorithm**: Normalized Cross-Correlation (`cv2.TM_CCOEFF_NORMED`)

---

## What is Template Matching?

Template matching is a technique for finding a small template image within a larger image by computing correlation scores at every possible position.

### Visual Concept

```
Large Image (Scanned Sheet):
┌─────────────────────────────┐
│                             │
│    ┌──┐ ← Template          │
│    │  │   (Reference Marker)│
│    └──┘                     │
│                             │
│         Slide template      │
│         across image,       │
│         compute similarity  │
│         at each position    │
└─────────────────────────────┘

Result Matrix (Match Scores):
┌─────────────────────────────┐
│ 0.1 0.2 0.3 0.2 0.1        │
│ 0.2 0.4 0.5 0.3 0.2        │
│ 0.3 0.5 0.95 0.4 0.3  ← Peak│
│ 0.2 0.3 0.4 0.2 0.1        │
└─────────────────────────────┘
           ↑
    Best match location
```

---

## Core Algorithm: Normalized Cross-Correlation

### Mathematical Foundation

**Formula** (TM_CCOEFF_NORMED):

```
R(x,y) = Σ(T'(x',y') · I'(x+x', y+y')) / √[Σ(T'²) · Σ(I'²)]

Where:
T'(x',y') = T(x',y') - mean(T)      (Template, mean-normalized)
I'(x,y)   = I(x,y) - mean(I(x,y))   (Image patch, mean-normalized)
```

**Range**: -1.0 to +1.0
- `+1.0`: Perfect positive match
- `0.0`: No correlation
- `-1.0`: Perfect negative match (inverse)

**Normalization Benefits**:
- Invariant to brightness changes
- Invariant to contrast changes (to some extent)
- More robust than raw cross-correlation

---

## Step-by-Step Flow

### Use Case 1: AutoRotate (Rotation Detection)

**File**: `src/processors/image/AutoRotate.py`

**Purpose**: Find best rotation angle (0°, 90°, 180°, 270°) by matching reference marker

---

#### Step 1: Initialization

```python
class AutoRotate(ImageTemplatePreprocessor):
    def __init__(self, *args, **kwargs):
        # Load reference marker image
        path = self.get_relative_path(self.options["reference_image"])
        self.reference_image = ImageUtils.load_image(path, cv2.IMREAD_GRAYSCALE)

        # Optional: Resize reference to fixed dimensions
        self.marker_dimensions = self.options.get("marker_dimensions", None)
        if self.marker_dimensions:
            self.resized_reference = ImageUtils.resize_to_dimensions(
                self.marker_dimensions, self.reference_image
            )

        # Optional: Match score threshold
        self.threshold = self.options.get("threshold", None)
```

**Configuration Example**:
```json
{
  "preProcessors": [
    {
      "name": "AutoRotate",
      "options": {
        "reference_image": "./markers/top_left_marker.png",
        "marker_dimensions": [50, 50],
        "threshold": {
          "value": 0.6,
          "passthrough": false
        }
      }
    }
  ]
}
```

---

#### Step 2: Template Matching Across Rotations

```python
def apply_filter(self, image, colored_image, _template, _file_path):
    best_val, best_rotation = -1, None
    rotations = [
        None,                           # 0° (no rotation)
        cv2.ROTATE_90_CLOCKWISE,        # 90° CW
        cv2.ROTATE_180,                 # 180°
        cv2.ROTATE_90_COUNTERCLOCKWISE, # 270° CW (90° CCW)
    ]

    for rotation in rotations:
        # Apply rotation
        rotated_img = image if rotation is None else ImageUtils.rotate(
            image, rotation, keep_original_shape=True
        )

        # Template matching
        res = cv2.matchTemplate(
            rotated_img,              # Source image (larger)
            self.resized_reference,   # Template (smaller)
            cv2.TM_CCOEFF_NORMED      # Method
        )

        # Find best match
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)

        # Track best rotation
        if max_val > best_val:
            best_val = max_val
            best_rotation = rotation
```

**Key Operations**:
1. **cv2.matchTemplate()**: Compute correlation matrix
2. **cv2.minMaxLoc()**: Find peak match score and location

---

#### Step 3: Result Matrix Analysis

**Input**:
```python
image.shape = (1200, 800)         # Scanned image
template.shape = (50, 50)         # Reference marker
```

**Output**:
```python
result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
result.shape = (1151, 751)        # (H - h + 1, W - w + 1)

# Result matrix contains match scores:
result = [
    [0.12, 0.15, 0.18, ..., 0.22],
    [0.14, 0.19, 0.23, ..., 0.20],
    [0.16, 0.21, 0.87, ..., 0.18],  ← Row with peak
    ...                        ↑
]                           Peak score (0.87)
```

**minMaxLoc**:
```python
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Example output:
min_val = -0.23        # Worst match score
max_val = 0.87         # Best match score
min_loc = (120, 340)   # Position of worst match
max_loc = (250, 180)   # Position of best match (top-left corner)
```

---

#### Step 4: Threshold Validation

```python
if self.threshold is not None and self.threshold["value"] > best_val:
    if self.threshold["passthrough"]:
        logger.warning("Score below threshold. Continuing due to passthrough.")
    else:
        logger.error("Score below threshold. Alignment failed.")
        raise ImageProcessingError(
            "The autorotate score is below threshold",
            context={
                "score": best_val,
                "min_threshold": self.threshold["value"],
            }
        )
```

**Example**:
```python
best_val = 0.55
threshold = {"value": 0.6, "passthrough": False}

# 0.55 < 0.6 → FAIL
# Raises ImageProcessingError
```

---

#### Step 5: Apply Best Rotation

```python
logger.info(f"AutoRotate Applied with rotation {best_rotation} and value {best_val}")

if best_rotation is not None:
    image = ImageUtils.rotate(image, best_rotation, keep_original_shape=True)
    if config.outputs.colored_outputs_enabled:
        colored_image = ImageUtils.rotate(
            colored_image, best_rotation, keep_original_shape=True
        )

return image, colored_image, _template
```

---

## Use Case 2: Custom Marker Detection (Multi-Scale)

**File**: `src/processors/image/crop_on_patches/marker_detection.py`

**Purpose**: Locate custom markers accounting for scale variations

---

### Enhanced Flow: Multi-Scale Template Matching

#### Step 1: Prepare Marker Template

```python
def prepare_marker_template(
    reference_image: np.ndarray,
    reference_zone: dict,
    marker_dimensions: Optional[Tuple[int, int]] = None,
    blur_kernel: Tuple[int, int] = (5, 5),
    apply_erode_subtract: bool = True,
) -> np.ndarray:
    # Extract ROI
    origin, dimensions = reference_zone["origin"], reference_zone["dimensions"]
    x, y = origin
    w, h = dimensions
    marker = reference_image[y : y + h, x : x + w]

    # Resize if specified
    if marker_dimensions is not None:
        marker = ImageUtils.resize_to_dimensions(marker_dimensions, marker)

    # Blur to reduce noise
    marker = cv2.GaussianBlur(marker, blur_kernel, 0)

    # Normalize to full range [0, 255]
    marker = cv2.normalize(
        marker, None, alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # Edge enhancement (optional)
    if apply_erode_subtract:
        marker = marker - cv2.erode(marker, kernel=np.ones((5, 5)), iterations=5)
        marker = cv2.normalize(marker, None, alpha=0, beta=255,
                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return marker
```

**Preprocessing Steps**:
1. **Extract**: Crop marker from reference image
2. **Resize**: Standardize marker size
3. **Blur**: Reduce high-frequency noise
4. **Normalize**: Stretch histogram to [0, 255]
5. **Edge Enhance**: Subtract erosion to emphasize edges

---

#### Step 2: Multi-Scale Matching

```python
def multi_scale_template_match(
    patch: np.ndarray,
    marker: np.ndarray,
    scale_range: Tuple[int, int] = (85, 115),
    scale_steps: int = 5,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, Optional[int]]:
    """
    Try template matching at multiple scales.

    Args:
        patch: Image to search in
        marker: Template to find
        scale_range: (min_percent, max_percent) e.g., (85, 115) → 85%-115%
        scale_steps: Number of scale increments (e.g., 5 → test 85%, 92%, 100%, 107%, 115%)

    Returns:
        (position, optimal_marker, confidence, optimal_scale_percent)
    """
    best_match_val = -1
    best_match_loc = None
    best_scaled_marker = None
    best_scale = None

    # Generate scale percentages
    scales = np.linspace(scale_range[0], scale_range[1], scale_steps)

    for scale_percent in scales:
        # Resize marker to current scale
        scale_factor = scale_percent / 100.0
        scaled_h = int(marker.shape[0] * scale_factor)
        scaled_w = int(marker.shape[1] * scale_factor)
        scaled_marker = cv2.resize(marker, (scaled_w, scaled_h))

        # Skip if scaled marker larger than patch
        if scaled_h > patch.shape[0] or scaled_w > patch.shape[1]:
            continue

        # Template matching
        result = cv2.matchTemplate(patch, scaled_marker, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Track best match across all scales
        if max_val > best_match_val:
            best_match_val = max_val
            best_match_loc = max_loc
            best_scaled_marker = scaled_marker
            best_scale = scale_percent

    return best_match_loc, best_scaled_marker, best_match_val, best_scale
```

**Example**:
```python
# Test 5 scales: 85%, 92%, 100%, 107%, 115%
scales = [85, 92, 100, 107, 115]

Results:
Scale 85%:  max_val = 0.72, position = (120, 80)
Scale 92%:  max_val = 0.68, position = (118, 82)
Scale 100%: max_val = 0.91, position = (115, 85)  ← Best
Scale 107%: max_val = 0.75, position = (112, 88)
Scale 115%: max_val = 0.64, position = (110, 90)

Best match: Scale 100%, score 0.91, at (115, 85)
```

---

## Comparison with SIFT Alignment

| Aspect | Template Matching | SIFT Alignment |
|--------|-------------------|----------------|
| **Algorithm** | Cross-correlation | Feature detection + matching |
| **Input** | Template image + target image | Two images (no template) |
| **Output** | Single position + confidence | Multiple displacement pairs |
| **Invariance** | Limited (brightness, some contrast) | Scale, rotation, partial illumination |
| **Use Case** | Finding known marker/pattern | General alignment |
| **Speed** | Fast (O(WH)) with FFT | Slower (O(n log n) features) |
| **Robustness** | Sensitive to scale/rotation | Robust to transformations |
| **Memory** | Low (just result matrix) | High (128-dim descriptors) |
| **Match Type** | Single best match | Multiple correspondences |

---

## Integration in Template Alignment

**Note**: Template matching is **NOT** the primary method in `template_alignment.py`. The current implementation uses:
- **Method 3 (Active)**: K-Nearest Interpolation with SIFT features

However, template matching **could** be integrated as:

### Potential Integration (Not Implemented)

```python
# Hypothetical Method 4: Template-based alignment
def apply_template_based_alignment(
    field_block,
    block_gray_image,
    block_gray_alignment_image,
    config,
):
    # Use alignment image as template
    result = cv2.matchTemplate(
        block_gray_image,           # Scanned image (larger)
        block_gray_alignment_image, # Template (reference)
        cv2.TM_CCOEFF_NORMED
    )

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # max_loc is top-left corner of best match
    # Compute shift from expected position (0, 0)
    shift_x = max_loc[0]
    shift_y = max_loc[1]

    # Apply shift to all scan boxes
    for field in field_block.fields:
        for scan_box in field.scan_boxes:
            scan_box.shifts = [shift_x, shift_y]

    return [shift_x, shift_y]
```

**Why Not Used?**:
- **Limitation**: Only detects translation (no rotation/warping)
- **Scale Sensitivity**: Requires exact scale match
- **Single Point**: Only one shift for entire field block (less flexible than SIFT's multiple anchors)

**When Useful**:
- Preprocessing alignment (coarse alignment before SIFT)
- Marker-based alignment (when markers are known/fixed)
- Fast initial estimate for iterative refinement

---

## Performance Characteristics

### Time Complexity

**Naive Sliding Window**: O(W × H × w × h)
- W, H: Image dimensions
- w, h: Template dimensions

**FFT-Based (OpenCV default for large templates)**: O((W × H) × log(W × H))

**Example**:
```python
image.shape = (1200, 800)     # 960,000 pixels
template.shape = (50, 50)     # 2,500 pixels

Naive: 960,000 × 2,500 = 2.4 billion operations
FFT: 960,000 × log₂(960,000) ≈ 19 million operations

Speed: ~10-30ms (FFT) vs ~500-1000ms (naive)
```

### Memory Usage

```python
# Result matrix:
result.shape = (H - h + 1, W - w + 1)
             = (1151, 751)
             = 864,401 floats × 4 bytes
             ≈ 3.3 MB

# Temporary buffers for FFT: 2-3× result size
Total: ~10-15 MB per matchTemplate call
```

---

## Edge Cases & Constraints

See **constraints.md** for detailed analysis of:
1. Template size constraints
2. Score threshold selection
3. Multiple matches (ambiguity)
4. Scale/rotation sensitivity
5. Border effects
6. Computational limits
7. Browser migration challenges

---

## Browser Migration Notes

### OpenCV.js Support

**Good News**: Template matching is fully supported!

```javascript
// Basic template matching
const result = new cv.Mat();
cv.matchTemplate(
  sourceImage,      // cv.Mat (larger image)
  templateImage,    // cv.Mat (template)
  result,           // cv.Mat (output)
  cv.TM_CCOEFF_NORMED  // Method
);

// Find best match
const minMax = cv.minMaxLoc(result);
console.log('Best match score:', minMax.maxVal);
console.log('Best match position:', minMax.maxLoc);  // {x, y}

// Clean up
result.delete();
```

### Memory Management

```javascript
function templateMatch(sourceImg, templateImg) {
  const result = new cv.Mat();

  try {
    cv.matchTemplate(sourceImg, templateImg, result, cv.TM_CCOEFF_NORMED);
    const minMax = cv.minMaxLoc(result);
    return {
      position: minMax.maxLoc,
      confidence: minMax.maxVal
    };
  } finally {
    result.delete();  // CRITICAL: Prevent memory leak
  }
}
```

### Multi-Scale in Browser

```javascript
async function multiScaleMatch(patch, marker, scaleRange = [85, 115], steps = 5) {
  let bestMatch = { score: -1, position: null, scale: null };

  const scales = [];
  for (let i = 0; i < steps; i++) {
    scales.push(scaleRange[0] + (scaleRange[1] - scaleRange[0]) * i / (steps - 1));
  }

  for (const scale of scales) {
    const scaledMarker = new cv.Mat();
    const dsize = new cv.Size(
      Math.floor(marker.cols * scale / 100),
      Math.floor(marker.rows * scale / 100)
    );

    try {
      cv.resize(marker, scaledMarker, dsize);

      const result = templateMatch(patch, scaledMarker);

      if (result.confidence > bestMatch.score) {
        bestMatch = {
          score: result.confidence,
          position: result.position,
          scale: scale
        };
      }
    } finally {
      scaledMarker.delete();
    }
  }

  return bestMatch;
}
```

### Web Worker Integration

```javascript
// worker.js
self.addEventListener('message', (event) => {
  const { sourceData, templateData } = event.data;

  const source = cv.matFromImageData(sourceData);
  const template = cv.matFromImageData(templateData);

  try {
    const result = templateMatch(source, template);
    self.postMessage({ success: true, result });
  } catch (error) {
    self.postMessage({ success: false, error: error.message });
  } finally {
    source.delete();
    template.delete();
  }
});
```

---

## Related Modules

- **AutoRotate** (`modules/domain/preprocessing/auto-rotate/flows.md`)
- **CropOnMarkers** (`modules/domain/preprocessing/crop-on-markers/flows.md`)
- **SIFT Alignment** (`../sift/flows.md`)
- **Phase Correlation** (`../phase-correlation/flows.md`)
- **Alignment Main** (`../flows.md`)

---

## Summary

Template matching provides a **simple, fast, and reliable** method for finding known patterns in images. While not used as the primary alignment strategy in OMRChecker, it serves critical roles in:
- Rotation detection (AutoRotate)
- Marker detection (CropOnCustomMarkers)
- Initial coarse alignment

**Key Strengths**:
- Fast (FFT-based)
- Simple to implement
- Deterministic results
- Browser-friendly

**Key Limitations**:
- Sensitive to scale/rotation
- Single match point (less flexible than SIFT)
- Requires known template

For general field block alignment, SIFT-based methods (Method 3) remain superior due to their robustness and flexibility.
