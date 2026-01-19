# Unused Files Verification Report

## ✅ Actions Taken

1. **Deleted backup file:**
   - ✅ `src/processors/detection/bubbles_threshold/interpretation_old_backup.py` - Removed

2. **Checked for dynamic imports:**
   - ✅ No `importlib` or `__import__` calls found
   - ✅ No conditional imports (try/except ImportError) found

## 📊 Detailed Analysis

### Files Used in Tests Only

These files are imported in test files but not in the main codebase:

1. **`src/processors/training/yolo_exporter.py`**
   - Used in: `src/tests/test_auto_training.py`
   - Status: ✅ Used by tests, not by entry.py
   - Recommendation: Keep (needed for tests)

2. **`src/training/trainer.py`** and **`src/training/__init__.py`**
   - Used in: `src/training/__init__.py` (self-referencing)
   - Status: ⚠️ Training module, may be used by external training scripts
   - Recommendation: Keep (training functionality)

### Files Referenced but Not Imported

1. **`src/processors/alignment/phase_correlation.py`**
   - Referenced in: `src/processors/alignment/template_alignment.py` (commented out)
   - Status: ⚠️ Code exists but not actively used
   - Recommendation: Review - may be legacy or future feature

2. **`src/processors/image/point_utils.py`**
   - Referenced in: `src/processors/image/WarpOnPointsCommon.py` (in comment/docstring)
   - Status: ⚠️ Mentioned but not imported
   - Recommendation: Verify if actually needed

3. **`src/utils/visualization_runner.py`**
   - Self-referencing: Contains usage examples in docstring
   - Status: ⚠️ May be run as standalone script (`python -m src.utils.visualization_runner`)
   - Recommendation: Keep if used as standalone tool

### Files with No References

These files have no references anywhere in the codebase:

1. **`src/cli/__init__.py`** - CLI interface (may be used by external entry points)
2. **`src/processors/detection/bubbles_threshold/stats.py`** - Statistics utilities
3. **`src/processors/detection/fusion/`** - Fusion detection (2 files)
4. **`src/processors/detection/ml_bubble_detector.py`** - ML detector
5. **`src/processors/detection/ocr/lib/tesseract.py`** - Tesseract wrapper
6. **`src/processors/detection/visualization/`** - ML visualization (2 files)
7. **`src/processors/helpers/mapping.py`** - Mapping utilities
8. **`src/processors/training/field_block_data_collector.py`** - Data collection
9. **`src/processors/visualization/`** - Workflow visualization (4 files)

## 🎯 Recommendations

### Safe to Delete (if confirmed unused)
- `src/processors/detection/bubbles_threshold/stats.py` - No references found
- `src/processors/detection/fusion/` - No references found
- `src/processors/detection/ml_bubble_detector.py` - No references found
- `src/processors/detection/visualization/` - No references found
- `src/processors/helpers/mapping.py` - No references found
- `src/processors/training/field_block_data_collector.py` - No references found
- `src/processors/visualization/` - No references found (workflow visualization)

### Keep (Used by tests or external tools)
- `src/processors/training/yolo_exporter.py` - Used in tests
- `src/training/` - Training module (may be used externally)
- `src/utils/visualization_runner.py` - Standalone tool
- `src/cli/__init__.py` - CLI interface (may be used externally)

### Review (May be conditionally used)
- `src/processors/alignment/phase_correlation.py` - Commented out in template_alignment
- `src/processors/image/point_utils.py` - Mentioned in comments
- `src/processors/detection/ocr/lib/tesseract.py` - May be conditionally imported

## 📝 Summary

- **Total unused files:** 20
- **Deleted:** 1 (backup file)
- **Used in tests:** 1 (yolo_exporter)
- **May be external tools:** 3 (cli, training, visualization_runner)
- **Truly unused:** ~15 files

## Next Steps

1. Verify if CLI, training, and visualization_runner are used by external scripts
2. Review phase_correlation and point_utils for actual usage
3. Consider deleting truly unused files after confirmation
