# Alignment Flow - Core Concepts

## Overview

The **Alignment System** is a critical component that corrects misalignments between scanned OMR sheets and the reference template. Even minor shifts, rotations, or distortions during scanning can cause bubble detection to fail. The alignment flow addresses these issues using multiple strategies to ensure accurate field detection.

## What is Alignment?

Alignment is the process of matching a scanned image to a reference template image by:
- Detecting displacement (shifts in x/y)
- Computing transformations (translation, rotation, warping)
- Applying corrections to either the image OR the template coordinates

**Key Insight**: OMRChecker supports TWO approaches:
1. **Warp the image** - Apply transformation to align scanned image with template
2. **Warp the coordinates** - Adjust template bubble positions to match scanned image (CURRENT DEFAULT)

## Why is Alignment Needed?

### Common Scan Issues
- **Translation**: Paper not perfectly centered on scanner
- **Rotation**: Small angle misalignment (usually < 5 degrees)
- **Scale Variance**: Different DPI settings between template and scan
- **Local Distortions**: Page warping, bending, or scanner artifacts
- **Field Block Shifts**: Individual sections may shift independently

### Without Alignment
```
Template Bubble:     [Expected: (100, 200)]
Scanned Bubble:      [Actual: (105, 198)]
Detection Result:    MISS (bubble not at expected location)
```

### With Alignment
```
Template Bubble:     [Expected: (100, 200)]
Detected Shift:      [+5, -2]
Adjusted Position:   [100 + 5, 200 - 2] = (105, 198)
Detection Result:    HIT (bubble found at shifted location)
```

## Core Architecture

### Entry Point
```python
# src/processors/alignment/processor.py
class AlignmentProcessor(Processor):
    """Main alignment processor orchestrator"""
```

### Primary Flow
```
AlignmentProcessor
    ↓
apply_template_alignment()
    ↓
For each field_block:
    1. Extract ROI (region of interest)
    2. Compute displacement using strategy
    3. Apply shifts to bubble coordinates
```

### Alignment Strategies (4 Methods)

The codebase contains **4 different alignment methods** (commented code shows evolution):

| Method | Status | Approach | Use Case |
|--------|--------|----------|----------|
| **Method 1: Phase Correlation** | Commented | FFT-based shift detection | Simple translation-only shifts |
| **Method 2: SIFT + Piecewise Affine** | Commented | Feature matching + triangulation warping | Complex warping, rotation |
| **Method 3: K-Nearest Interpolation** | **ACTIVE** | SIFT features + coordinate shifting | Current production method |
| **Method 4: Field-level Warping** | TODO | Per-field image transformation | Future: overlapping field blocks |

## Configuration Schema

### Template-Level Alignment Config
```json
{
  "alignment": {
    "margins": {
      "top": 50,
      "bottom": 50,
      "left": 50,
      "right": 50
    },
    "max_displacement": 30
  }
}
```

### Field Block-Level Override
```json
{
  "fieldBlocks": {
    "Q1-30": {
      "alignment": {
        "margins": { "top": 20, "bottom": 20, "left": 20, "right": 20 },
        "max_displacement": 15
      }
    }
  }
}
```

## Key Concepts

### 1. Alignment Image (Reference)
- Pre-processed reference image stored in template
- Acts as ground truth for alignment
- Loaded from `gray_alignment_image` and `colored_alignment_image`

### 2. Margins
- Extra space around field block to include context for matching
- Larger margins = more context but slower computation
- **Trade-off**: Accuracy vs. Performance

### 3. Max Displacement
- Maximum allowed pixel shift in any direction
- Acts as constraint filter for SIFT matches
- **Edge Case**: `max_displacement = 0` → Skip alignment

### 4. Zone-Based Alignment
- Alignment computed per field block, not globally
- Allows handling of local distortions
- Each field block gets independent shift values

### 5. Shift Application
- Shifts applied to `scan_box.shifts` (bubble coordinates)
- NOT applied to image itself (Method 3 approach)
- Allows overlapping field blocks (future feature)

## Data Flow

### Input
```python
context.gray_image           # Preprocessed scanned image
context.colored_image        # Color version
context.template.alignment   # Alignment configuration
```

### Processing
```python
# For each field block:
1. Extract zone_start, zone_end (with margins)
2. Crop ROI from both images
3. Compute displacement_pairs (SIFT features)
4. Find k-nearest anchors for field block center
5. Average displacements → field_block.shifts
6. Apply shifts to all scan_boxes in field
```

### Output
```python
context.template             # Updated with shift values
# Each scan_box now has:
scan_box.shifts = [dx, dy]   # Pixel displacement
```

## Edge Cases & Constraints

### 1. No Alignment Image Provided
```python
if "gray_alignment_image" not in template.alignment:
    # Skip alignment entirely
    return context
```

### 2. Zero Max Displacement
```python
if max_displacement == 0:
    continue  # Skip this field block
```

### 3. Insufficient SIFT Matches
```python
if len(good_matches) < MIN_MATCH_COUNT:  # MIN_MATCH_COUNT = 10
    logger.critical("Not enough matches found")
    # Alignment fails, shifts remain [0, 0]
```

### 4. Out-of-Bounds Displacements
```python
# Filter displacement pairs outside rectangle
if not rectangle_contains(point, warped_rectangle):
    # Reject this displacement pair
```

### 5. Zone Clipping
```python
# Zones may extend beyond image bounds
zone_start = max(0, zone_start)
zone_end = min(image.shape, zone_end)
```

## Dependencies

### Computer Vision
- **OpenCV SIFT**: Feature detection (`cv2.SIFT_create()`)
- **FLANN Matcher**: Fast approximate matching
- **Homography**: Geometric transformation estimation

### Internal Modules
- `ImageUtils.resize_to_dimensions()` - Normalize image sizes
- `MathUtils.distance()` - Euclidean distance for filtering
- `SiftMatcher` - Singleton SIFT feature extractor

## Performance Considerations

### Computational Cost
1. **SIFT Feature Detection**: O(n) per field block
2. **FLANN Matching**: O(k log n) approximate
3. **K-Nearest Search**: O(m * k) where m = number of bubbles

### Optimization Strategies
- Singleton SIFT/FLANN initialization (reused across images)
- Zone-based processing (only compute alignment where needed)
- Skip blocks with `max_displacement = 0`

## Browser Migration Notes

### Critical Challenges

#### 1. SIFT Availability
**Python**: `cv2.SIFT_create()` (non-free module)
**Browser**:
- OpenCV.js may NOT include SIFT (patent-encumbered)
- **Alternatives**:
  - **ORB** (cv.ORB_create()) - Patent-free, less accurate
  - **AKAZE** (cv.AKAZE_create()) - Good alternative
  - **Phase Correlation** (FFT-based, no features needed)

#### 2. FLANN Matcher
**Python**: `cv2.FlannBasedMatcher()`
**Browser**:
- OpenCV.js FLANN support is LIMITED
- **Fallback**: BFMatcher (Brute Force) - slower but works

#### 3. NumPy FFT
**Python**: `np.fft.fft2()` for phase correlation
**Browser**:
- Use **fft.js** or **ml-fft**
- Or implement via Web Workers

#### 4. Performance
- SIFT computation is CPU-intensive
- **Solution**: Offload to Web Worker
- Consider downsampling images for alignment

### Recommended Browser Strategy

```javascript
// Feature detection fallback chain
if (cv.SIFT_create) {
  matcher = new cv.SIFT_create();
} else if (cv.ORB_create) {
  matcher = new cv.ORB_create();
} else {
  // Fallback to phase correlation
  usePhaseCorrelation();
}
```

## Related Modules

- **SIFT Alignment** (`sift/`) - Feature-based matching
- **Phase Correlation** (`phase-correlation/`) - FFT-based translation
- **Template Matching** (`template-matching/`) - Cross-correlation
- **Piecewise Affine** (`piecewise-affine/`) - Delaunay warping
- **K-Nearest Interpolation** (`k-nearest/`) - Current production method

## Next Steps

1. Read `flows.md` for detailed execution flow
2. Read `decisions.md` for strategy comparison and trade-offs
3. Read `constraints.md` for edge case handling
4. Read `integration.md` for template loading and context management
