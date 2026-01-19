# Sync Status Analysis

## Current Status (from sync_tool.py)

- **Total mapped files:** 150
- **✅ In sync:** 77 (51.3%)
- **⚠️ Partially synced:** 4 (2.7%)
- **❌ Not started:** 69 (46.0%)

## Phase 1 Files Analysis

### Files Marked as "Partial" (Phase 1, High/Medium Priority)

1. **`src/exceptions.py`** → `omrchecker-js/packages/core/src/core/exceptions.ts`
   - **Status:** Partial
   - **Python:** ~30 exception classes (738 lines)
   - **TypeScript:** 10 exception classes
   - **Missing exceptions:**
     - `TemplateNotFoundError` - Used for template loading
     - `TemplateLoadError` - Used for template loading errors
     - `AnswerKeyError` - Used for evaluation
     - `ScoringError` - Used for evaluation
     - `PathTraversalError` - Security validation
     - `EvaluationConfigNotFoundError` - Evaluation config
     - `EvaluationConfigLoadError` - Evaluation config loading
     - `PreprocessorError` - Preprocessor failures
     - `MarkerDetectionError` - Marker detection failures
     - `BubbleDetectionError` - Bubble detection failures
     - `OCRError` - OCR failures
     - `BarcodeDetectionError` - Barcode detection failures
     - `AlignmentError` - Alignment failures
     - `OutputError`, `OutputDirectoryError`, `FileWriteError` - File operations (not needed in browser)
     - `InputDirectoryNotFoundError` - Directory operations (not needed in browser)
   - **Assessment:** Status "partial" is correct. However, some exceptions like `TemplateNotFoundError`, `TemplateLoadError`, `AnswerKeyError`, and `ScoringError` might be needed for browser functionality.

2. **`src/utils/interaction.py`** → `omrchecker-js/packages/core/src/utils/InteractionUtils.ts`
   - **Status:** Partial
   - **Python:** GUI/window management, ROI selection, image display
   - **TypeScript:** Browser-compatible image display, debug container
   - **Missing features:**
     - `SelectROI` class (ROI selection with mouse) - Not applicable to browser
     - `ImageMetrics` class (window positioning) - Not applicable to browser
     - `Stats` class (thread-safe stats) - Not needed in browser
     - `show_for_roi()` method - Not applicable to browser
   - **Assessment:** Status "partial" is correct. Browser version has equivalent functionality for image display, but lacks GUI-specific features that aren't applicable.

## Recommendations

### 1. Add Missing Critical Exceptions

The following exceptions should be added to TypeScript as they're likely needed:

```typescript
// Template exceptions
export class TemplateNotFoundError extends TemplateError {
  searchPath: string;
  constructor(searchPath: string) {
    super(`No template.json found in directory tree of '${searchPath}'`, { searchPath });
    this.searchPath = searchPath;
  }
}

export class TemplateLoadError extends TemplateError {
  path: string;
  reason: string;
  constructor(path: string, reason: string) {
    super(`Failed to load template '${path}': ${reason}`, { path, reason });
    this.path = path;
    this.reason = reason;
  }
}

// Evaluation exceptions
export class AnswerKeyError extends EvaluationError {
  constructor(message: string, context?: Record<string, unknown>) {
    super(message, context);
  }
}

export class ScoringError extends EvaluationError {
  constructor(message: string, context?: Record<string, unknown>) {
    super(message, context);
  }
}

// Security exception
export class PathTraversalError extends OMRCheckerError {
  path: string;
  basePath?: string;
  constructor(path: string, basePath?: string) {
    const msg = basePath 
      ? `Path traversal detected: '${path}' outside base '${basePath}'`
      : `Path traversal detected: '${path}'`;
    super(msg, { path, basePath });
    this.path = path;
    this.basePath = basePath;
  }
}
```

### 2. Update FILE_MAPPING.json Status

After adding the missing exceptions, the status for `exceptions.py` can remain "partial" (since many file system exceptions are intentionally omitted), but the notes should be updated to reflect that all browser-relevant exceptions are now included.

### 3. InteractionUtils Status

The `interaction.py` → `InteractionUtils.ts` mapping is correctly marked as "partial" since it intentionally omits GUI-specific features. This status is appropriate.

## Next Steps

1. ✅ Verify sync status using sync_tool.py - **DONE**
2. ⏳ Add missing critical exceptions to TypeScript
3. ⏳ Update FILE_MAPPING.json notes if needed
4. ⏳ Verify all Phase 1 files are correctly marked

