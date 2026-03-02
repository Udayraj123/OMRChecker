# Test Coverage Audit: Python vs TypeScript

**Last Updated:** 2026-03-02  
**Purpose:** Track which TypeScript tests were translated from Python vs created from scratch

## Principle: Translate, Don't Invent

**Rule:** All TypeScript tests SHOULD be translations of existing Python tests.  
**Reason:** Ensures behavioral parity between Python and TypeScript implementations.

## Test Coverage Status

### ✅ Utils - Properly Translated

| Module | Python Test | TypeScript Test | Status | Notes |
|--------|------------|-----------------|--------|-------|
| geometry.py | test_geometry.py (4.7KB) | unit/geometry.test.ts | ✅ Translated | Based on Python tests |
| drawing.py | test_drawing.py (3.5KB) | browser/drawing.test.ts | ⚠️ Needs Review | May have extra tests |

### ⚠️ Utils - Created from Scratch (Need Python Tests)

| Module | Python Test | TypeScript Test | Status | Notes |
|--------|------------|-----------------|--------|-------|
| math.py | ❌ None | unit/math.test.ts (59 tests) | ⚠️ No Python equivalent | Need to check if Python has inline tests |
| stats.py | ❌ None | unit/stats.test.ts (36 tests) | ⚠️ No Python equivalent | Need to check if Python has inline tests |
| checksum.py | ❌ None (browser-only) | unit/checksum.test.ts | ⚠️ Browser-only | 7 tests failing, should be browser tests |

### ❌ Utils - Not Yet Tested

| Module | Python Test | TypeScript Test | Status | Action Needed |
|--------|------------|-----------------|--------|---------------|
| image.py | test_image_utils.py | ❌ None | ❌ Missing | Translate Python tests for resize, normalize, autoCanny, rotate, overlay |

### ✅ Processors - Created from Scratch (No Python Tests)

| Processor | Python Test | TypeScript Test | Status | Notes |
|-----------|------------|-----------------|--------|-------|
| GaussianBlur | ❌ None | browser/processors/image-filters.test.ts | ✅ OK | Python has no processor unit tests |
| MedianBlur | ❌ None | browser/processors/image-filters.test.ts | ✅ OK | Python has no processor unit tests |
| Contrast | ❌ None | browser/processors/image-filters.test.ts | ✅ OK | Python has no processor unit tests |
| Levels | ❌ None | browser/processors/image-filters.test.ts | ✅ OK | Python has no processor unit tests |
| AutoRotate | ❌ None | browser/processors/image-filters.test.ts | ✅ OK | Python has no processor unit tests |

**Note:** Python tests processors via integration tests (test_all_samples.py), not unit tests.

### 🔬 Special Test Types

| Test Type | Purpose | Status | Notes |
|-----------|---------|--------|-------|
| parity/python-comparison.test.ts | Verify TS matches Python output | ✅ 24/24 passing | Uses Python fixture generator |
| browser/opencv-smoke.test.ts | OpenCV.js initialization | ✅ 10/10 passing | Browser-specific |
| browser/memory-leak.test.ts | Mat memory management | ✅ Passing | Browser-specific |
| unit/exceptions.test.ts | Exception hierarchy | ✅ 235/235 passing | Based on Python structure |

## Action Items

### High Priority
1. **Audit math.test.ts** - Check if Python has inline/doctest tests for math utils
2. **Audit stats.test.ts** - Check if Python has inline/doctest tests for stats utils
3. **Translate test_image_utils.py** - Create tests for ImageUtils functions we migrated
4. **Review drawing tests** - Compare browser/drawing.test.ts with test_drawing.py line by line

### Medium Priority
5. **Review geometry tests** - Verify unit/geometry.test.ts matches test_geometry.py exactly
6. **Move checksum tests** - Convert unit/checksum.test.ts to browser-only tests

### Low Priority  
7. **Document policy** - Update AGENTS.md with "translate, don't invent" principle
8. **Create checklist** - Add test translation checklist to migration workflow

## Python Test Files Reference

**Location:** `/Users/udayraj.deshmukh/Personals/OMRChecker/src/tests/`

### Utils Tests
- `utils/__tests__/test_drawing.py` (3.5KB)
- `utils/__tests__/test_geometry.py` (4.7KB)
- `utils/__tests__/test_image_utils.py` (covers load_image only)
- `utils/__tests__/test_csv.py`
- `utils/__tests__/test_file.py`
- `utils/__tests__/test_file_pattern_resolver.py`
- `utils/__tests__/test_logger.py`
- `utils/__tests__/test_serialization.py`

### Processor Tests
- `processors/__tests__/test_pipeline.py`
- `processors/image/__tests__/test_WarpOnPointsCommon.py`
- `processors/image/__tests__/test_warp_strategies.py`
- `processors/image/__tests__/test_point_utils.py`
- `processors/image/__tests__/test_page_detection.py`
- `processors/image/__tests__/test_marker_detection.py`
- `processors/image/__tests__/test_dot_line_detection.py`

### Integration Tests
- `__tests__/test_all_samples.py` - End-to-end processor testing
- `__tests__/test_entry.py`
- `__tests__/test_main.py`

## Verification Process

When migrating a Python module to TypeScript:

1. **Before writing ANY tests**, check: `ls src/tests/**/test_{module_name}.py`
2. **If Python tests exist:**
   - Read the Python test file completely
   - Translate each test case to TypeScript
   - Match test names, assertions, edge cases
   - Do NOT add extra tests without Python equivalent
3. **If NO Python tests exist:**
   - Document why (e.g., "Python has no unit tests for processors")
   - Create minimal tests for TypeScript (OpenCV.js compatibility)
   - Consider contributing tests back to Python codebase

## Notes

- **Parity tests** (python-comparison.test.ts) are the gold standard - they literally compare outputs
- **Browser tests** for OpenCV.js are necessary because OpenCV.js ≠ cv2 in all aspects
- **Integration tests** in Python cover processors; we need unit tests for TypeScript compilation

## Detailed Test-by-Test Audit

### geometry.test.ts Analysis

**Python Source:** `src/tests/utils/__tests__/test_geometry.py` (17 tests)
**TypeScript:** `tests/unit/geometry.test.ts` (16 tests)
**Status:** ✅ Complete - all Python tests translated, no extra tests

**Python Tests (17 total):**
1. test_euclidean_distance_2d → ✅ Translated ("calculate distance between two points")
2. test_euclidean_distance_same_point → ✅ Translated ("return 0 for same point")
3. test_euclidean_distance_negative_coords → ✅ Translated ("handle negative coordinates")
4. test_euclidean_distance_float_coords → ✅ Translated ("handle floating point coordinates")
5. test_vector_magnitude_zero → ✅ Translated ("return 0 for zero vector")
6. test_vector_magnitude_unit_vector → ❌ MISSING
7. test_vector_magnitude_2d → ✅ Translated ("calculate magnitude of 2D vector")
8. test_vector_magnitude_negative → ✅ Translated ("handle negative components")
9. test_vector_magnitude_3d → ✅ Translated ("calculate magnitude of 3D vector")
10. test_bbox_center_unit_square → ⚠️ Different (origin [0,0] dims [2,2] vs [0,0] dims [100,200])
11. test_bbox_center_offset_box → ✅ Translated ("handle non-zero origin")
12. test_bbox_center_float_coords → ✅ Translated ("handle fractional dimensions")
13. test_bbox_center_zero_dimensions → ✅ Translated ("handle zero dimensions")
14. test_bbox_center_large_box → ✅ Translated ("handle large coordinates")
15. test_geometry_consistency → ❌ MISSING (tests bbox_center + euclideanDistance together)
16. test_pythagorean_theorem → ❌ MISSING (validates right triangle)

**Extra Tests in TypeScript (6 invented tests):**
1. euclideanDistance - handle large distances (NO Python equivalent)
2. euclideanDistance - be symmetric (NO Python equivalent)
3. vectorMagnitude - handle single element vector (NO Python equivalent)
4. vectorMagnitude - handle empty vector (NO Python equivalent)
5. vectorMagnitude - handle floating point components (NO Python equivalent)
6. vectorMagnitude - handle high dimensional vectors (NO Python equivalent)

**Completed Actions (commit 672ab67):**
- ✅ Removed 6 extra tests from geometry.test.ts
- ✅ Added 3 missing tests: test_vector_magnitude_unit_vector, test_geometry_consistency, test_pythagorean_theorem
- ✅ Fixed bbox_center test #10 to match Python (unit square)
- ✅ All 16 tests passing

### drawing.test.ts Analysis

**Python Source:** `src/tests/utils/__tests__/test_drawing.py` (10 tests)
**TypeScript:** ❌ File does not exist
**Status:** ❌ Not migrated

**Missing Tests (10 total):**
1. test_draw_box_diagonal
2. test_draw_box_hollow
3. test_draw_box_filled
4. test_draw_box_centered
5. test_draw_text
6. test_draw_text_centered
7. test_draw_line
8. test_draw_polygon_closed
9. test_draw_polygon_open
10. test_draw_contour

**Action Required:**
- Create tests/unit/drawing.test.ts
- Translate all 10 Python tests exactly
- No extra tests should be added
