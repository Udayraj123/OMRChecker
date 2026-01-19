# BubblesFieldDetection Test Mapping

## Python Source File
**File:** `src/tests/test_bubbles_threshold_detection.py` (407 lines)

## TypeScript Test File
**File:** `omrchecker-js/packages/core/src/processors/detection/__tests__/bubblesFieldDetection.test.ts` (329 lines)

## Test Class Mapping

### 1. `TestBubblesFieldDetection` (Python) → `describe('BubblesFieldDetection')` (TypeScript)

#### ✅ Mapped Tests:

| Python Test | TypeScript Test | Status |
|------------|----------------|--------|
| `test_basic_detection` | `should create detection and run detection automatically` | ✅ Mapped |
| `test_bubble_mean_values` | `should calculate mean value for uniform region` | ✅ Mapped |
| `test_scan_quality_excellent` | `should detect marked vs unmarked bubbles correctly` | ✅ Mapped (partially) |
| `test_scan_quality_poor` | `should handle poor quality scans (all similar values)` | ✅ Mapped |
| `test_read_bubble_mean_value_static_method` | `readBubbleMeanValue static method` tests | ✅ Mapped |
| `test_empty_field` | `should handle empty bubble array` | ✅ Mapped |
| `test_single_bubble` | `should handle single bubble` | ✅ Mapped |
| `test_backward_compatibility` | `should populate fieldBubbleMeans for backward compatibility` | ✅ Mapped |

#### ⚠️ Partially Mapped / Missing Tests:

| Python Test | TypeScript Equivalent | Notes |
|------------|----------------------|-------|
| `test_jumps_calculation` | ❌ Missing | Tests calculation of jumps between bubble means |
| `test_max_jump` | ❌ Missing | Tests max_jump property |
| `test_sorted_bubble_means` | ❌ Missing | Tests sorted bubble means |
| `test_min_max_mean_values` | ❌ Missing | Tests min_mean and max_mean properties |

### 2. `TestBubbleMeanValue` (Python) → Not explicitly tested in TypeScript

| Python Test | TypeScript Equivalent | Notes |
|------------|----------------------|-------|
| `test_sorting_behavior` | ❌ Missing | Tests BubbleMeanValue sorting |
| `test_string_representation` | ❌ Missing | Tests string representation |
| `test_comparison_operators` | ❌ Missing | Tests comparison operators |

### 3. `TestIntegrationWithRealScenarios` (Python) → `describe('integration with realistic data')` (TypeScript)

| Python Test | TypeScript Test | Status |
|------------|----------------|--------|
| `test_typical_mcq_4_options` | `should detect marked vs unmarked bubbles correctly` | ✅ Mapped (similar) |
| `test_no_answer_marked` | `should handle poor quality scans (all similar values)` | ✅ Mapped (similar) |
| `test_multiple_answers_marked` | ❌ Missing | Tests multi-mark scenario |

## Additional TypeScript Tests (Not in Python)

1. **`should handle different intensity regions`** - Tests dark vs light bubble regions
2. **`should calculate correct position`** - Tests position calculation
3. **`should throw error if detection not run`** - Tests error handling
4. **`should handle large number of bubbles`** - Tests edge case with 50 bubbles

## Summary

### Coverage Status:
- **Mapped:** ~8 tests (core functionality)
- **Missing:** ~7 tests (jumps, sorting, min/max, BubbleMeanValue model tests, multi-mark scenario)
- **TypeScript-only:** ~4 tests (additional edge cases)

### Key Missing Functionality:
1. **Jumps calculation** - Tests for calculating jumps between bubble means
2. **Max jump** - Tests for max_jump property
3. **Sorted bubble means** - Tests for sorted_bubble_means property
4. **Min/Max mean values** - Tests for min_mean and max_mean properties
5. **BubbleMeanValue model tests** - Sorting, string representation, comparison operators
6. **Multi-mark scenario** - Testing when multiple answers are marked

### Recommendations:
1. Add tests for jumps calculation and max_jump
2. Add tests for sorted_bubble_means, min_mean, and max_mean
3. Add tests for BubbleMeanValue model (if used directly)
4. Add test for multi-mark scenario (multiple answers marked)

## FILE_MAPPING.json Reference

According to `FILE_MAPPING.json`:
- **Python Source:** `src/processors/detection/bubbles_threshold/detection.py`
- **TypeScript Source:** `omrchecker-js/packages/core/src/processors/detection/bubbles_threshold/detection.ts`
- **Test File:** `omrchecker-js/packages/core/src/processors/detection/__tests__/bubblesFieldDetection.test.ts`
- **Python Test:** `src/tests/test_bubbles_threshold_detection.py`

