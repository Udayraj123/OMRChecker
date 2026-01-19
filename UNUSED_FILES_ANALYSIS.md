# Unused/Unimported Python Files Analysis

## Summary
- **Total Python files:** 203
- **Files in import tree:** 150
- **Unused files (excluding tests/backups):** 20

## Unused Files List

### 1. CLI Module
- `src/cli/__init__.py` - CLI interface (may be used by external scripts)

### 2. Alignment
- `src/processors/alignment/phase_correlation.py` - Phase correlation algorithm (may be imported conditionally)

### 3. Detection - Bubbles Threshold
- `src/processors/detection/bubbles_threshold/stats.py` - Statistics utilities (may be unused)

### 4. Detection - Fusion
- `src/processors/detection/fusion/__init__.py` - Fusion detection module
- `src/processors/detection/fusion/detection_fusion.py` - Detection fusion logic

### 5. Detection - ML
- `src/processors/detection/ml_bubble_detector.py` - ML-based bubble detector

### 6. Detection - OCR
- `src/processors/detection/ocr/lib/tesseract.py` - Tesseract OCR wrapper (may be conditionally imported)

### 7. Detection - Visualization
- `src/processors/detection/visualization/__init__.py` - ML detection visualization
- `src/processors/detection/visualization/ml_detection_viz.py` - ML detection visualization

### 8. Helpers
- `src/processors/helpers/mapping.py` - Mapping utilities

### 9. Image Processing
- `src/processors/image/point_utils.py` - Point utilities (may be imported conditionally)

### 10. Training
- `src/processors/training/field_block_data_collector.py` - Field block data collection
- `src/processors/training/yolo_exporter.py` - YOLO export utilities

### 11. Visualization
- `src/processors/visualization/__init__.py` - Workflow visualization
- `src/processors/visualization/html_exporter.py` - HTML export
- `src/processors/visualization/workflow_session.py` - Workflow session tracking
- `src/processors/visualization/workflow_tracker.py` - Workflow tracking

### 12. Training Module
- `src/training/__init__.py` - Training module
- `src/training/trainer.py` - Training utilities

### 13. Utils
- `src/utils/visualization_runner.py` - Visualization runner

## Notes

Many of these files may be:
1. **Conditionally imported** - Only imported when certain features are enabled
2. **Dynamically imported** - Imported via `importlib` or string-based imports
3. **Used by external scripts** - CLI, training scripts, etc.
4. **Future features** - Planned but not yet integrated
5. **Legacy code** - Old implementations kept for reference

## Recommendations

1. **Verify dynamic imports** - Check for `importlib.import_module()` or `__import__()` calls
2. **Check conditional imports** - Look for try/except import blocks
3. **Review CLI usage** - Check if CLI module is used by external entry points
4. **Check training scripts** - Verify if training module is used separately
5. **Review visualization** - Check if visualization is used in interactive mode only

## Backup Files

- `src/processors/detection/bubbles_threshold/interpretation_old_backup.py` - Old backup file (should be deleted)
