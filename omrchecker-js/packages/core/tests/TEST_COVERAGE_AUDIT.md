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
