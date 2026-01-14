# Dot and Line Detection Refactoring

**Date**: January 13, 2026
**Status**: ✅ Module Extracted (CropOnDotLines refactoring pending)

## Overview

Extracted dot and line detection algorithms from `CropOnDotLines` into a dedicated `dot_line_detection.py` module. This provides focused, testable implementations of morphological operations and edge detection for OMR sheet processing.

## Changes Made

### 1. New Module: `dot_line_detection.py` (401 lines)

Created a comprehensive module for dot and line detection with the following functions:

#### Preprocessing Functions
- **`preprocess_dot_zone`**: Prepares image zones for dot detection
  - Handles blur, white padding, morphological opening, thresholding
- **`preprocess_line_zone`**: Prepares image zones for line detection
  - Handles blur, gamma adjustment, thresholding, morphological opening

#### Core Detection Functions
- **`detect_contours_using_canny`**: Detects contours using Canny edge detection
  - Returns sorted contours (largest first)
- **`extract_patch_corners_and_edges`**: Extracts corners and edge contours
  - Supports both PATCH_DOT and PATCH_LINE scanner types
  - Handles rotated rectangles for lines

#### Main Entry Points
- **`detect_dot_corners`**: Complete dot detection pipeline
  - Combines preprocessing, contour detection, corner extraction
  - Returns absolute corner coordinates
- **`detect_line_corners_and_edges`**: Complete line detection pipeline
  - Returns both corners and edge contours map
  - Handles multiple edge types (TOP, BOTTOM, LEFT, RIGHT)

#### Utility Functions
- **`validate_blur_kernel`**: Validates blur kernel size against zone dimensions
- **`create_structuring_element`**: Creates morphological structuring elements
  - Supports 'rect', 'ellipse', 'cross' shapes

### 2. Key Algorithms

#### Dot Detection Pipeline
```python
# 1. Preprocess
zone → blur (optional) → white padding → morphological opening →
       threshold → normalize → remove padding

# 2. Edge Detection
normalized_zone → Canny → contours (sorted by area)

# 3. Corner Extraction
largest_contour → bounding_rect → corners → shift_to_absolute
```

#### Line Detection Pipeline
```python
# 1. Preprocess
zone → blur (optional) → gamma_adjust → threshold → normalize →
       white padding → threshold → morphological opening → remove padding

# 2. Edge Detection
morphed_zone → Canny → contours (sorted by area)

# 3. Corner & Edge Extraction
largest_contour → rotated_rect → corners + edge_contours →
                  split_by_edges → shift_to_absolute
```

### 3. Morphological Operations

**Dot Detection**:
- Kernel: Small rectangular (e.g., 5x5)
- Operation: MORPH_OPEN (erode then dilate)
- Purpose: Remove noise while preserving dot shapes
- Iterations: 3

**Line Detection**:
- Kernel: Elongated rectangular (e.g., 2x10 for vertical lines)
- Operation: MORPH_OPEN (erode then dilate)
- Purpose: Enhance line structures, remove perpendicular noise
- Iterations: 3

### 4. Test Coverage (28 tests, 100% passing)

**Preprocessing Tests** (8 tests):
- Dot zone preprocessing: basic, with blur, with threshold
- Line zone preprocessing: basic, with gamma, with blur

**Contour Detection Tests** (3 tests):
- Successful detection
- Empty zone handling
- Contour sorting by area

**Corner Extraction Tests** (3 tests):
- Dot scanner corners
- Line scanner corners
- Unsupported scanner type error

**Integration Tests** (8 tests):
- Dot detection: success, with offset, not found
- Line detection: success, with offset, not found, edge contours

**Utility Tests** (6 tests):
- Blur kernel validation (valid, too large, equal size, with label)
- Structuring element creation (rect, ellipse, cross, invalid)

## Metrics

| Metric | Value |
|--------|-------|
| **New Module** | 401 LOC |
| **Test Coverage** | 28 tests (100% passing) |
| **Functions** | 8 core + 2 utilities |
| **Scanner Types** | 2 (PATCH_DOT, PATCH_LINE) |

## Technical Details

### Scanner Types

**PATCH_DOT**:
- Uses axis-aligned bounding rectangle
- Good for circular or square markers
- Faster computation

**PATCH_LINE**:
- Uses rotated bounding rectangle
- Handles slight rotations
- Better for alignment markers

### Edge Types

The module supports 4 edge types for lines:
- `TOP`: Upper edge of detected line
- `BOTTOM`: Lower edge of detected line
- `LEFT`: Left edge of detected line
- `RIGHT`: Right edge of detected line

Each edge type maps to specific contour segments, enabling precise alignment.

### Preprocessing Pipeline

**White Padding**:
- Added before morphological operations
- Prevents edge artifacts
- Size: `2 * kernel_size`
- Removed after processing

**Thresholding**:
- THRESH_TRUNC: Caps values at threshold
- Preserves relative intensities below threshold
- Better than binary for grayscale processing

**Normalization**:
- Stretches histogram to full 0-255 range
- Enhances contrast
- Applied after each major operation

## Benefits

1. **Separation of Concerns**: Detection logic independent from CropOnDotLines
2. **Testability**: 28 comprehensive unit tests
3. **Reusability**: Can be used by other processors
4. **Clarity**: Clear preprocessing → detection → extraction pipeline
5. **Maintainability**: Smaller, focused functions
6. **Debugging**: Intermediate steps can be inspected
7. **Documentation**: Well-documented algorithms and parameters

## Files Created

```
src/processors/image/dot_line_detection.py        (401 lines)
tests/processors/image/test_dot_line_detection.py (434 lines)
```

## Next Steps

**CropOnDotLines Refactoring** (Pending):
- Update `find_dot_corners_from_options` to use `detect_dot_corners`
- Update `find_line_corners_and_contours` to use `detect_line_corners_and_edges`
- Simplify `find_corners_and_contours_map_using_canny` to delegate to module
- Expected reduction: ~150-200 LOC (30-40%)

## Usage Example

```python
from src.processors.image.dot_line_detection import (
    detect_dot_corners,
    detect_line_corners_and_edges,
    create_structuring_element,
)

# Create kernels
dot_kernel = create_structuring_element('rect', (5, 5))
line_kernel = create_structuring_element('rect', (2, 10))

# Detect dot
dot_corners = detect_dot_corners(
    zone=image_patch,
    zone_offset=(100, 100),
    dot_kernel=dot_kernel,
    dot_threshold=150,
    blur_kernel=(5, 5),
)

# Detect line
line_corners, edge_contours = detect_line_corners_and_edges(
    zone=image_patch,
    zone_offset=(100, 100),
    line_kernel=line_kernel,
    gamma_low=0.5,
    line_threshold=180,
    blur_kernel=(5, 5),
)
```

## Backward Compatibility

✅ **Fully backward compatible**
- New module doesn't modify existing code
- Original `CropOnDotLines` functionality unchanged
- Refactoring to use new module will be transparent

## Related Documentation

- [Progress Summary](./PROGRESS_SUMMARY.md)
- [WarpOnPointsCommon Refactoring](./WARP_ON_POINTS_REFACTORING.md)
- [CropPage Refactoring](./CROP_PAGE_REFACTORING.md)
- [CropOnCustomMarkers Refactoring](./CROP_ON_CUSTOM_MARKERS_REFACTORING.md)

