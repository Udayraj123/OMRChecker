# CropOnCustomMarkers Refactoring

**Date**: January 2026
**Status**: ✅ Completed

## Overview

Refactored `CropOnCustomMarkers` by extracting marker detection logic into a dedicated `marker_detection.py` module. This improves code organization, testability, and reusability.

## Changes Made

### 1. New Module: `marker_detection.py` (328 lines)

Created a focused module for marker template matching with the following functions:

#### Core Detection Functions
- **`prepare_marker_template`**: Extracts and preprocesses marker template from reference image
  - Handles extraction, resizing, blurring, normalization, and edge enhancement
- **`detect_marker_in_patch`**: Main entry point for marker detection
  - Combines multi-scale matching, best match selection, and corner extraction

#### Multi-Scale Matching
- **`match_template_at_scale`**: Performs template matching at a single scale
- **`find_best_match_multi_scale`**: Searches across multiple scales to find best match

#### Helper Functions
- **`extract_marker_corners`**: Converts marker position to corner coordinates
- **`validate_marker_dimensions`**: Validates marker size is reasonable
- **`compute_marker_statistics`**: Provides debugging statistics

### 2. Refactored: `CropOnCustomMarkers.py`

**Original**: 481 lines → **Refactored**: 425 lines (56 lines reduced, ~12%)

#### Key Changes

**Simplified `extract_marker_from_reference`** (27 lines → 20 lines):
```python
# Before: Manual preprocessing with cv2/numpy operations
marker = reference_image[y : y + h, x : x + w]
marker = cv2.GaussianBlur(marker, blur_kernel, 0)
marker = cv2.normalize(marker, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
if self.apply_erode_subtract:
    marker -= cv2.erode(marker, kernel=np.ones((5, 5)), iterations=5)

# After: Delegate to marker_detection module
return prepare_marker_template(
    reference_image,
    reference_zone,
    marker_dimensions=marker_dimensions,
    blur_kernel=blur_kernel,
    apply_erode_subtract=self.apply_erode_subtract,
)
```

**Replaced `get_best_match` and `find_marker_corners_in_patch`** (135 lines → 42 lines):
- Removed the complex 135-line `get_best_match` method (C901 complexity warning)
- Simplified `find_marker_corners_in_patch` to delegate to `detect_marker_in_patch`
- Extracted visualization logic to separate `_visualize_marker_detection` method

**Before**:
```python
def find_marker_corners_in_patch(self, zone_description, image, _file_path):
    zone_label = zone_description["label"]
    patch_zone, zone_start, _zone_end = self.compute_scan_zone_util(...)
    (marker_position, optimal_marker) = self.get_best_match(zone_label, patch_zone)
    # ... manual corner calculation

def get_best_match(self, zone_label, patch_zone):
    # ... 135 lines of complex multi-scale matching logic
    # ... inline visualization code
    # ... inline error handling
```

**After**:
```python
def find_marker_corners_in_patch(self, zone_description, image, file_path):
    patch_zone, zone_start, _zone_end = self.compute_scan_zone_util(...)
    marker = self.marker_for_zone_label[zone_label]

    # Delegate to marker_detection module
    corners = detect_marker_in_patch(
        patch_zone, marker, zone_offset=zone_start,
        scale_range=self.marker_rescale_range,
        scale_steps=self.marker_rescale_steps,
        min_confidence=self.min_matching_threshold,
    )

    # Separate visualization
    if config.outputs.show_image_level >= 1:
        self._visualize_marker_detection(...)
```

## Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **CropOnCustomMarkers** | 481 LOC | 425 LOC | -56 (-12%) |
| **Marker Detection Module** | 0 LOC | 328 LOC | +328 (new) |
| **Test Coverage** | 0 tests | 21 tests | +21 |
| **Cyclomatic Complexity** | High (C901) | Low | ✅ Reduced |

### Test Coverage (21 tests, 100% passing)

- **Template Preparation**: 3 tests (extraction, resizing, edge enhancement)
- **Single-Scale Matching**: 4 tests (scale variations, oversized rejection)
- **Multi-Scale Matching**: 3 tests (success, correct scale, low confidence)
- **Corner Extraction**: 2 tests (with/without offset)
- **Integration**: 4 tests (detection success, offset handling, not found)
- **Validation**: 4 tests (dimension validation)
- **Utilities**: 1 test (statistics computation)

## Benefits

1. **Separation of Concerns**: Marker detection logic is now independent and reusable
2. **Testability**: 21 new unit tests provide comprehensive coverage
3. **Maintainability**: Smaller, focused functions are easier to understand and modify
4. **Reduced Complexity**: Removed C901 complexity warning from `get_best_match`
5. **Better Organization**: Clear API with documented functions
6. **Debugging**: Helper functions like `compute_marker_statistics` aid troubleshooting

## Files Created

```
src/processors/image/marker_detection.py       (328 lines)
tests/processors/image/test_marker_detection.py (390 lines)
```

## Files Modified

```
src/processors/image/CropOnCustomMarkers.py     (481 → 425 lines)
```

## Backward Compatibility

✅ **Fully backward compatible**
- All public APIs remain unchanged
- Existing templates and configurations work without modification
- Behavior is identical (delegates to same underlying logic)

## Next Steps

This refactoring pattern can be applied to:
- `CropOnDotLines` (dot line detection)
- `CropOnPatches` (patch extraction)
- Other marker-based processors

