# Diff Plan: Python → TypeScript Image Processors (Post–Python Changes)

These four Python files are **modified** in the working tree (per git status). This doc is a quick diff plan to bring the TypeScript port up to date.

| Python | TypeScript |
|--------|------------|
| `src/processors/image/base.py` | `omrchecker-js/packages/core/src/processors/image/base.ts` |
| `src/processors/image/CropPage.py` | `omrchecker-js/packages/core/src/processors/image/CropPage.ts` |
| `src/processors/image/WarpOnPointsCommon.py` | `omrchecker-js/packages/core/src/processors/image/WarpOnPointsCommon.ts` |
| `src/processors/image/CropOnCustomMarkers.py` | `omrchecker-js/packages/core/src/processors/image/CropOnCustomMarkers.ts` |

---

## 1. base.py ↔ base.ts

**Sync checklist**

- **Option keys**: Python uses `options.get("processing_image_shape", default)` and `options.get("tuning_options", {})` (snake_case). TypeScript uses `options.processingImageShape` and `options.tuning_options`. Ensure config passed into the TS preprocessor uses camelCase (or add fallbacks for snake_case) so `processingImageShape` is never null when it should fall back to `defaultProcessingImageShape`.
- **Resize behavior**: Python always resizes using `self.processing_image_shape`. TS only resizes when `this.processingImageShape` is truthy. Align: either guarantee TS always has a shape (from default) or document the intentional difference.
- **Debug logging**: Python has  
  `logger.debug(f"processing_image_shape: {self.processing_image_shape}, gray_image: {gray_image.shape}, ...")`  
  before resize. Consider adding the same debug line in TS (with camelCase names) for parity.
- **`exclude_files`**: Both have it; no change needed if behavior is the same.

---

## 2. CropPage.py ↔ CropPage.ts

**Sync checklist**

- **`__is_internal_preprocessor__`**: Python has `False`, TypeScript has `true`. Confirm intended semantics (e.g. “internal” = not user-facing). If Python’s `False` is correct, set TS `__isInternalPreprocessor` to `false` and ensure no logic depends on it being true.
- **Options schema**: Python returns snake_case keys (`morph_kernel`, `use_colored_canny`, `max_points_per_edge`, `tuning_options.warp_method`, etc.). TS uses camelCase (`morphKernel`, `useColoredCanny`, …). Ensure:
  - Parent class (WarpOnPointsCommon) in TS reads the same logical options (camelCase in TS is fine if the object is built in TS with camelCase).
  - Any shared config or JSON that uses snake_case is normalized in one place so both sides behave the same.
- **`validate_and_remap_options_schema` vs `validateAndMerge`**: Python does validate + merge in base; TS uses a static `validateAndMerge` before `super()`. Confirm that the merged options in TS include all fields that Python’s `validate_and_remap_options_schema` + `merge_tuning_options` produce (including nested `tuning_options`).
- **`prepare_image_before_extraction` / `extract_control_destination_points`**: Compare with TS `prepareImageBeforeExtraction` and `extractControlDestinationPoints` for any new branches or parameters (e.g. `debug_image`, `use_colored_canny`) and align.

---

## 3. WarpOnPointsCommon.py ↔ WarpOnPointsCommon.ts

**Sync checklist**

- **Constructor flow**: Python: `validate_and_remap_options_schema` → `merge_tuning_options` → `super().__init__(merged_options, ...)`. TS: subclasses call a static validate/merge then `super(merged, ...)`. Verify that `merge_tuning_options` (Python) and the TS merge (e.g. `deepMerge` or equivalent) produce equivalent merged options (same keys and nesting, modulo camelCase).
- **Option keys**: Python uses `options.get("enable_cropping", False)`, `tuning_options.get("warp_method", ...)`, `tuning_options.get("warp_method_flag", 'INTER_LINEAR')`. TS uses `opts.enableCropping`, `tuningOptions.warp_method`, `tuningOptions.warp_method_flag`. Ensure defaulting and key names match (camelCase in TS, or dual support if config is shared).
- **`warp_method_flags_map`**: Both map INTER_LINEAR, INTER_CUBIC, INTER_NEAREST to OpenCV constants. Confirm the enum/constant names match (e.g. `WarpMethodFlags`).
- **Rest of pipeline**: `prepare_points_for_strategy`, `apply_filter` / `applyWarpStrategy`, and any new helpers in Python should have corresponding TS implementations; diff the full files if you made non-trivial changes in Python.

---

## 4. CropOnCustomMarkers.py ↔ CropOnCustomMarkers.ts

**Sync checklist**

- **Class vars**: Python uses `__is_internal_preprocessor__ = True`, `scan_zone_presets_for_layout`, `default_scan_zone_descriptions`, `default_points_selector_map`. TS has static overrides for these. Confirm values and structure match (e.g. `default_scan_zone_descriptions` built from `MARKER_ZONE_TYPES_IN_ORDER` in Python vs static object in TS).
- **Constructor / tuning options**: Python sets `min_matching_threshold`, `marker_rescale_range`, `marker_rescale_steps`, `apply_erode_subtract` from `tuning_options`. Ensure TS reads the same (camelCase) and uses the same defaults (e.g. `0.3`, `(85, 115)`, `5`, `True`).
- **`init_resized_markers` / `initResizedMarkers`**: Compare implementations; port any Python changes (e.g. rescale range, steps, or erode logic).
- **`validate_and_remap_options_schema`**: Python builds `parsed_options` from `reference_image`, `type`, `tuning_options`, etc. TS should build the same logical structure (with camelCase) so parent and downstream code see the same behavior.

---

## Suggested order of work

1. **base.ts** – Option handling and resize behavior so all preprocessors get a consistent context.
2. **WarpOnPointsCommon.ts** – Merge and option defaults so all warp-based processors (including CropPage and CropOnCustomMarkers) are aligned.
3. **CropPage.ts** – `__isInternalPreprocessor` and options schema.
4. **CropOnCustomMarkers.ts** – Tuning options and `initResizedMarkers` / validation.

After syncing, run the TS test suite and, if available, a small Python/TS comparison (same image + config) to confirm identical behavior.
