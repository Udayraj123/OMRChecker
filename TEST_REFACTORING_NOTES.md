# Test Refactoring Notes

## Issue: Unnecessary `patch()` Usage

### Problem
Some tests were using `patch()` to bypass schema validation, which is hacky and doesn't test the actual code paths that users would encounter.

### Root Cause
Schema validation happens in `open_template_with_defaults()` before `TemplateLayout.__init__()` runs. This means:
- Invalid configurations are caught at schema validation level
- Code-level validation methods may never be reached for schema-invalid configs
- Tests that bypass schema validation don't reflect real-world usage

### Solution

#### 1. Removed Hacky Patches
**Before:** Tests used `patch("open_template_with_defaults")` to bypass schema validation
```python
with patch("src.processors.layout.template_layout.open_template_with_defaults") as mock_open:
    mock_open.return_value = invalid_json
    # Test code logic with invalid config
```

**After:** Test validation methods directly
```python
# Test the validation method directly with invalid input
sample_template_layout.validate_field_blocks(invalid_field_blocks)
```

#### 2. Kept Legitimate Patches
**Legitimate use:** Mocking I/O operations (file reading, image loading)
```python
# This is fine - we're mocking file I/O, not bypassing validation
with patch("src.processors.layout.template_layout.ImageUtils.read_image_util") as mock_read:
    mock_read.return_value = (np.zeros((100, 100), dtype=np.uint8), None)
    layout = TemplateLayout(...)
```

### Tests Refactored

1. **`test_validate_field_blocks_invalid_bubble_type`**
   - **Before:** Used patch to bypass schema validation
   - **After:** Tests `validate_field_blocks()` method directly with invalid bubble type
   - **Why:** Tests the actual validation logic, not schema validation

2. **`test_validate_field_blocks_missing_labels_gap`**
   - **Before:** Used patch to bypass schema validation
   - **After:** Tests `validate_field_blocks()` method directly with missing labelsGap
   - **Why:** Tests the actual validation logic, not schema validation

### Principles

1. **Test What Users Experience:** If schema validation catches invalid configs, that's the user experience. Don't bypass it.
2. **Test Code Logic Directly:** If you want to test code-level validation, call the validation methods directly.
3. **Mock I/O, Not Validation:** It's fine to mock file reading/image loading, but don't mock validation layers.
4. **Schema Validation is a Feature:** If schema validation prevents invalid configs, that's good! Don't test invalid configs that schema already catches.

### Remaining Patches

All remaining `patch()` usage is for legitimate I/O mocking:
- `ImageUtils.read_image_util` - Mocking image file reading (legitimate)
- No patches for bypassing validation (removed)

### Benefits

1. **More Realistic Tests:** Tests reflect actual code paths users encounter
2. **Better Separation:** Schema validation tests vs code logic tests
3. **Less Hacky:** No bypassing of validation layers
4. **Easier to Maintain:** Tests are clearer about what they're testing

