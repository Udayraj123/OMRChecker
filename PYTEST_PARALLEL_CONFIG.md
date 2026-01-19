# Pytest Parallel Execution Configuration

## Overview

Tests are configured to run in parallel using `pytest-xdist` with `--dist=loadscope` for better resource isolation. The `loadscope` distribution groups tests by module/class, ensuring tests in the same module run on the same worker, which prevents conflicts with shared file resources.

## Sequential Test Marking

Tests that use shared resources are marked with `@pytest.mark.serial` for documentation and potential future use with serial execution plugins.

### Tests Requiring Sequential Execution

#### 1. **test_edge_cases.py::test_different_bubble_dimensions**
- **Reason:** Uses shared file paths (`BASE_RESULTS_CSV_PATH`, `BASE_MULTIMARKED_CSV_PATH`)
- **Issue:** Removes and writes to shared CSV files, causing race conditions in parallel execution
- **Solution:** Marked with `@pytest.mark.serial`

#### 2. **test_all_samples.py** (All tests)
- **Reason:** All tests write to shared `outputs/` directories
- **Issue:** Multiple tests writing to same output paths can cause file conflicts
- **Solution:** Consider marking entire file or using `--dist=loadscope` for module-level grouping

#### 3. **test_template_validations.py** (All tests)
- **Reason:** Uses `write_jsons_and_run` which writes to shared sample paths
- **Issue:** Tests modify shared template/config files
- **Solution:** Consider marking entire file or using `--dist=loadscope`

## Configuration

### Current Setup

```ini
# pytest.ini
addopts =
    -n auto  # Auto-detect CPU cores for parallel execution
```

### Running Sequential Tests

When running tests, pytest-xdist will automatically:
1. Group tests marked with `@pytest.mark.serial` to run sequentially
2. Use `--dist=loadscope` to group tests by class/module (better for shared resources)

### Recommended Commands

```bash
# Run all tests in parallel with loadscope (default, groups by module/class)
uv run pytest src/tests -n auto --dist=loadscope

# Run all tests in parallel with worksteal (better load balancing)
uv run pytest src/tests -n auto --dist=worksteal

# Run only non-serial tests in parallel
uv run pytest src/tests -n auto --dist=loadscope -m "not serial"

# Run only serial tests (sequentially, no parallel)
uv run pytest src/tests -n 0 -m serial

# Run with fixed number of workers
uv run pytest src/tests -n 4 --dist=loadscope
```

## Alternative: Module-Level Grouping

Instead of marking individual tests, you can use `--dist=loadscope` which groups tests by:
- Test class (all tests in same class run on same worker)
- Test module (all tests in same module run on same worker)

This is often better for tests that share file system resources.

## Global State Concerns

### STATS Object in entry.py

The `STATS` global object in `src/entry.py` could theoretically cause issues, but:
- It's only used in integration tests (`test_all_samples.py`, `test_edge_cases.py`)
- These tests are already marked or should be grouped
- The object is reset between test runs

## Best Practices

1. **Use `tmp_path` fixture** for test-specific files (already done in most tests)
2. **Mark shared-resource tests** with `@pytest.mark.serial`
3. **Use `--dist=loadscope`** for better resource isolation
4. **Avoid global state** in test code when possible
5. **Use unique file paths** per test when using shared directories

## Verification

To verify tests work correctly in parallel:

```bash
# Run tests multiple times to catch race conditions
for i in {1..5}; do
    uv run pytest src/tests -n auto --dist=loadscope
done
```

## Future Improvements

1. Refactor `test_edge_cases.py::test_different_bubble_dimensions` to use `tmp_path` instead of shared paths
2. Refactor `test_all_samples.py` tests to use unique output directories per test
3. Consider using `pytest-xdist`'s `--dist=worksteal` for better load balancing
4. Add more `@pytest.mark.serial` markers if race conditions are discovered

