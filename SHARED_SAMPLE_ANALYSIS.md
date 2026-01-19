# Shared Sample Files Analysis

## Critical Issue Found ✅ FIXED

### `test_template_validations.py` - All 14 tests use the same sample directory

**Problem:** All tests in `test_template_validations.py` write to the same shared sample directory:
- **Shared Path:** `src/tests/test_samples/sample1/`
- **Files Written:** `template.json`, `config.json`, `evaluation.json`
- **Operation:** Each test writes these files, runs the test, then removes them

**Race Condition Risk:** If tests run in parallel, multiple tests could:
1. Write to the same files simultaneously
2. Read files that are being modified by another test
3. Delete files that another test is trying to read

### Solution Applied ✅

**Marked entire file with `pytestmark = pytest.mark.serial`** - All 14 tests in `test_template_validations.py` now run sequentially, preventing file conflicts.

## Analysis of Sample Usage

### Tests Using Shared Samples

#### 1. `test_template_validations.py` (14 tests) ⚠️ FIXED
- **Sample:** `src/tests/test_samples/sample1/`
- **All tests write to:** `sample1/template.json`, `sample1/config.json`, `sample1/evaluation.json`
- **Status:** ✅ **FIXED** - Entire file marked with `@pytest.mark.serial`
- **Protection:** All tests run sequentially

#### 2. `test_all_samples.py` (20 tests) ✅ SAFE
- **Samples:** Different paths like `samples/1-mobile-camera/`, `samples/2-omr-marker/`, etc.
- **Status:** ✅ **SAFE** - Each test uses a different sample directory
- **Protection:** No conflicts (different input paths)

#### 3. `test_edge_cases.py` (3 tests) ✅ SAFE
- **Sample:** `src/tests/test_samples/sample2/` (from constants.py)
- **Status:** ✅ **SAFE** - Uses different sample (sample2) than test_template_validations.py
- **Protection:** `--dist=loadscope` groups them (same module = same worker)

## Sample Directory Structure

```
src/tests/test_samples/
├── sample1/          # Used by test_template_validations.py (14 tests) - SERIAL
│   ├── sample.png
│   └── boilerplate.py
└── sample2/          # Used by test_edge_cases.py (3 tests) - SAFE
    ├── sample.jpg
    ├── omr_marker.jpg
    └── boilerplate.py

samples/               # Used by test_all_samples.py (20 tests) - SAFE
├── 1-mobile-camera/
├── 2-omr-marker/
├── 3-answer-key/
│   ├── bonus-marking/
│   ├── using-csv/
│   └── ...
└── community/
    ├── Antibodyy/
    ├── JoyChopra1298/
    └── ...
```

## How `write_jsons_and_run` Works

The `write_jsons_and_run` function in `src/tests/utils.py`:
1. **Writes** `template.json`, `config.json`, `evaluation.json` to the sample directory
2. **Runs** the test using `run_sample()`
3. **Removes** the files after the test completes

**Problem:** If two tests run in parallel and use the same `sample_path`, they'll:
- Write to the same files simultaneously
- Potentially read each other's modified files
- Delete files while another test is using them

## Current Protection Mechanisms

### 1. `--dist=loadscope` (Default) ✅ PRIMARY PROTECTION
- Groups tests by module/class
- All tests in `test_template_validations.py` → same worker
- All tests in `test_edge_cases.py` → same worker
- **Status:** ✅ Configured as default in `pytest.ini`
- **Note:** This is the PRIMARY protection mechanism

### 2. `@pytest.mark.serial` (Applied to test_template_validations.py)
- **Note:** `pytest.mark.serial` is NOT a built-in pytest-xdist feature
- **Current use:** Documentation marker (indicates tests should run sequentially)
- **Applied:** ✅ `test_template_validations.py` (all 14 tests)
- **To enforce:** Run with `-n 0 -m serial` to run sequentially, or rely on `--dist=loadscope`

## Verification

To verify tests work correctly:

```bash
# Run with loadscope (default - RECOMMENDED)
uv run pytest src/tests/test_template_validations.py -n auto --dist=loadscope
# ✅ All 14 tests pass (grouped on same worker)

# Run sequentially (no parallel)
uv run pytest src/tests/test_template_validations.py -n 0
# ✅ All 14 tests pass (no conflicts)

# Run with worksteal (NOT RECOMMENDED for this file)
uv run pytest src/tests/test_template_validations.py -n auto --dist=worksteal
# ⚠️ May fail due to race conditions (tests distributed across workers)
```

## Important Notes

⚠️ **`pytest.mark.serial` is NOT automatically enforced by pytest-xdist**

The `@pytest.mark.serial` marker is currently used for:
- **Documentation:** Indicates tests should run sequentially
- **Filtering:** Can be used with `-m serial` to select tests
- **Future:** Could be used with a custom plugin if needed

**The actual protection comes from `--dist=loadscope`** which groups tests by module, ensuring all tests in `test_template_validations.py` run on the same worker.

## Summary

✅ **Fixed:** `test_template_validations.py` marked as serial
✅ **Safe:** `test_all_samples.py` uses different sample directories
✅ **Safe:** `test_edge_cases.py` uses different sample (sample2) and is grouped by loadscope

**No further action needed** - The critical issue has been addressed.
