# Template Layout, Drawing, Schemas, and Utils Port Plan

## Overview

This plan covers porting the remaining functionality from:
1. `src/processors/layout/template_layout.py` - Missing TemplateLayout class methods
2. `src/processors/layout/template_drawing.py` - TemplateDrawing functionality (not ported)
3. `src/schemas/defaults/` - Default configuration values
4. `src/schemas/models/` - Typed configuration models
5. `src/utils/` - Missing utility functions

## Phase 9: Complete TemplateLayout Port

### 9.1 Port Missing TemplateLayout Methods

**Files to update:**
- `omrchecker-js/packages/core/src/template/TemplateLoader.ts` (or create new `TemplateLayout.ts`)

**Missing methods from Python `TemplateLayout` class:**

1. **`get_copy_for_shifting()`**
   - Creates shallow copy of template layout for alignment processing
   - Deep copies field blocks (which can be mutated)
   - **Location**: Add to `TemplateLayout` interface or create class wrapper

2. **`apply_preprocessors(file_path, gray_image, colored_image)`**
   - Applies preprocessing pipeline to images
   - Resizes to `processing_image_shape`
   - Uses unified processor interface
   - **Note**: May already be handled by `ProcessingPipeline` in TypeScript

3. **`setup_alignment(alignment_object, template_path_parent)`**
   - Sets up alignment configuration
   - Loads reference images if specified
   - **Location**: Add to `TemplateLoader` or `TemplateLayout` class

4. **`get_exclude_files()`**
   - Returns list of files to exclude from processing
   - Currently returns reference image path if present
   - **Location**: Add to `TemplateLayout` interface/class

5. **`reset_all_shifts()`**
   - Resets shifts for all field blocks and fields
   - **Location**: Add to `TemplateLayout` interface/class

6. **`validate_field_blocks(field_blocks_object)`**
   - Validates field block configuration
   - Checks for required properties
   - **Location**: Already partially in `TemplateLoader`, may need enhancement

7. **`parse_custom_bubble_field_types(custom_bubble_field_types)`**
   - Parses and validates custom bubble field types
   - **Location**: Already in `TemplateLoader.loadLayoutFromJSON()`

8. **`parse_custom_labels(custom_labels_object)`**
   - Parses custom labels configuration
   - **Location**: Already in `TemplateLoader.loadLayoutFromJSON()`

9. **`parse_output_columns(output_columns)`**
   - Parses output columns configuration
   - **Location**: Already in `TemplateLoader.loadLayoutFromJSON()`

10. **`fill_output_columns(non_custom_columns, all_custom_columns, output_columns)`**
    - Fills output columns if empty
    - **Location**: Already in `TemplateLoader.loadLayoutFromJSON()`

11. **`validate_template_columns(non_custom_columns, all_custom_columns)`**
    - Validates template columns configuration
    - **Location**: May need to add

**Implementation strategy:**
- Option A: Create `TemplateLayout` class that wraps the interface and adds methods
- Option B: Add methods to `TemplateLoader` as static or instance methods
- Option C: Create separate `TemplateLayoutManager` class

**Recommendation**: Option A - Create `TemplateLayout` class that extends/uses the interface

### 9.2 Port TemplateDrawing Functionality

**Files to create:**
- `omrchecker-js/packages/core/src/template/TemplateDrawing.ts`

**Classes to port:**

1. **`TemplateDrawing` class**
   - Wrapper for template drawing operations
   - Methods:
     - `draw_template_layout(gray_image, colored_image, config, ...args)`
     - `draw_only_field_blocks(image, shifted, should_copy, thickness, border)`
     - `draw_field_blocks_layout_util(template, image, shifted, should_copy, thickness, border)` (static)

2. **`TemplateDrawingUtils` class**
   - Static utility methods for template drawing
   - Methods:
     - `draw_template_layout(gray_image, colored_image, template, config, ...args)`
     - `draw_template_layout_util(image, image_type, template, config, field_id_to_interpretation, evaluation_meta, evaluation_config_for_response, shifted, border)`
     - `draw_all_fields(marked_image, image_type, template, field_id_to_interpretation, evaluation_meta, evaluation_config_for_response)`
     - `draw_evaluation_summary(marked_image, evaluation_meta, evaluation_config_for_response)`

**Dependencies:**
- `DrawingUtils` (already ported)
- `ImageUtils` (already ported)
- `InteractionUtils` (may need porting - see Phase 10)
- Field interpretation drawing (from `FieldInterpretation.drawing`)

**Implementation notes:**
- Integrate with existing `DrawingUtils`
- Support both grayscale and colored image drawing
- Handle evaluation summary drawing
- Support transparency/overlay effects

## Phase 10: Port Schemas (Defaults and Models)

### 10.1 Port Schema Defaults

**Files to create:**
- `omrchecker-js/packages/core/src/schemas/defaults/template.ts`
- `omrchecker-js/packages/core/src/schemas/defaults/config.ts`
- `omrchecker-js/packages/core/src/schemas/defaults/evaluation.ts`

**What to port:**

1. **`src/schemas/defaults/template.py`**
   - `TEMPLATE_DEFAULTS` constant
   - Default `TemplateConfig` instance
   - **Location**: `omrchecker-js/packages/core/src/schemas/defaults/template.ts`
   - **Status**: Currently defaults are in `template/types.ts` as `DEFAULT_TEMPLATE_CONFIG`

2. **`src/schemas/defaults/config.py`**
   - `CONFIG_DEFAULTS` constant
   - Default `Config` instance with all nested configs
   - **Location**: `omrchecker-js/packages/core/src/schemas/defaults/config.ts`
   - **Status**: Not ported

3. **`src/schemas/defaults/evaluation.py`**
   - `EVALUATION_CONFIG_DEFAULTS` constant
   - Default `EvaluationConfig` instance
   - **Location**: `omrchecker-js/packages/core/src/schemas/defaults/evaluation.ts`
   - **Status**: Not ported

**Implementation notes:**
- Extract defaults from existing TypeScript files where they exist
- Create typed default constants matching Python structure
- Use TypeScript const assertions for type safety

### 10.2 Port Schema Models

**Files to create/update:**
- `omrchecker-js/packages/core/src/schemas/models/config.ts`
- `omrchecker-js/packages/core/src/schemas/models/evaluation.ts`
- Update: `omrchecker-js/packages/core/src/template/types.ts` (already has template models)

**What to port:**

1. **`src/schemas/models/config.py`**
   - `ThresholdingConfig` dataclass
   - `GroupingRule` dataclass
   - `FileGroupingConfig` dataclass (with validation)
   - `OutputsConfig` dataclass
   - `ProcessingConfig` dataclass
   - `ShiftDetectionConfig` dataclass
   - `MLConfig` dataclass
   - `VisualizationConfig` dataclass
   - `Config` dataclass (main config)
   - **Location**: `omrchecker-js/packages/core/src/schemas/models/config.ts`
   - **Status**: Not ported (config types may be scattered)

2. **`src/schemas/models/evaluation.py`**
   - Evaluation configuration dataclasses
   - `EvaluationConfig` main class
   - `OutputsConfiguration` class
   - Drawing config classes (`DrawScoreConfig`, `DrawAnswersSummaryConfig`, etc.)
   - **Location**: `omrchecker-js/packages/core/src/schemas/models/evaluation.ts`
   - **Status**: Not ported

3. **`src/schemas/models/template.py`**
   - `AlignmentMarginsConfig` dataclass
   - `AlignmentConfig` dataclass
   - `OutputColumnsConfig` dataclass
   - `SortFilesConfig` dataclass
   - `TemplateConfig` dataclass
   - **Location**: Already partially in `omrchecker-js/packages/core/src/template/types.ts`
   - **Status**: Interfaces exist, may need to add methods like `from_dict()`, `to_dict()`

**Implementation notes:**
- Convert Python dataclasses to TypeScript interfaces/classes
- Add `from_dict()` and `to_dict()` methods for JSON serialization
- Implement validation methods where present
- Use TypeScript types for better type safety

## Phase 11: Port Missing Utils

### 11.1 Port File Utilities

**Files to create/update:**
- `omrchecker-js/packages/core/src/utils/file.ts` (already exists, may need updates)
- `omrchecker-js/packages/core/src/utils/filePatternResolver.ts` (new)

**What to port:**

1. **`src/utils/file.py`**
   - File path utilities
   - File pattern matching
   - **Status**: Partially ported, check what's missing

2. **`src/utils/file_pattern_resolver.py`**
   - File pattern resolution for grouping
   - Pattern matching with field substitution
   - **Location**: `omrchecker-js/packages/core/src/utils/filePatternResolver.ts`
   - **Status**: Not ported

### 11.2 Port Serialization Utilities

**Files to create:**
- `omrchecker-js/packages/core/src/utils/serialization.ts`

**What to port:**
- `dataclass_to_dict()` function
- JSON serialization helpers
- **Status**: Not ported (may be using native JSON.stringify)

### 11.3 Port Interaction Utilities

**Files to create:**
- `omrchecker-js/packages/core/src/utils/interaction.ts` (or update existing)

**What to port:**
- `InteractionUtils.show()` - Image display (browser equivalent)
- Window/popup utilities
- **Status**: Partially ported, may need browser-specific implementation
- **Note**: Browser environment doesn't need all Python interaction features

### 11.4 Port Other Utilities

**Files to check/port:**
- `src/utils/checksum.py` - Checksum calculation
- `src/utils/validations.py` - Validation utilities
- `src/utils/image_warp.py` - Image warping utilities (may be in ImageUtils)
- `src/utils/visualization_runner.py` - Visualization runner (may not be needed for browser)

**Status**: Check FILE_MAPPING.json for current status

## Implementation Order

1. **Phase 9.1** - Port missing TemplateLayout methods (enables alignment and preprocessing)
2. **Phase 9.2** - Port TemplateDrawing (enables visualization)
3. **Phase 10.1** - Port schema defaults (enables proper config handling)
4. **Phase 10.2** - Port schema models (enables typed configs)
5. **Phase 11** - Port missing utils (as needed)

## Dependencies

- **Phase 9.1 → Phase 9.2**: TemplateLayout needed for TemplateDrawing
- **Phase 10.1 → Phase 10.2**: Defaults reference models
- **Phase 11**: Can be done in parallel with other phases

## Files Summary

### To Create:
- `omrchecker-js/packages/core/src/template/TemplateLayout.ts` (class wrapper)
- `omrchecker-js/packages/core/src/template/TemplateDrawing.ts`
- `omrchecker-js/packages/core/src/schemas/defaults/template.ts`
- `omrchecker-js/packages/core/src/schemas/defaults/config.ts`
- `omrchecker-js/packages/core/src/schemas/defaults/evaluation.ts`
- `omrchecker-js/packages/core/src/schemas/models/config.ts`
- `omrchecker-js/packages/core/src/schemas/models/evaluation.ts`
- `omrchecker-js/packages/core/src/utils/filePatternResolver.ts`
- `omrchecker-js/packages/core/src/utils/serialization.ts`

### To Update:
- `omrchecker-js/packages/core/src/template/TemplateLoader.ts` (add missing methods or refactor)
- `omrchecker-js/packages/core/src/template/types.ts` (add methods to interfaces)
- `omrchecker-js/packages/core/src/utils/file.ts` (check completeness)
- `omrchecker-js/packages/core/src/utils/interaction.ts` (if exists, update)

## Success Criteria

1. ✅ All TemplateLayout methods ported
2. ✅ TemplateDrawing functionality working
3. ✅ Schema defaults match Python exactly
4. ✅ Schema models with proper typing
5. ✅ Missing utils ported
6. ✅ All typechecks pass
7. ✅ Tests created for new functionality

