# Multi-Pass Detection and Interpretation Sync Plan

## Overview

This plan outlines the steps to sync TypeScript code with Python's multi-pass detection and interpretation architecture. The Python implementation uses a sophisticated two-pass system (detection pass + interpretation pass) with aggregate management at field, file, and directory levels. The current TypeScript implementation uses a simplified single-pass approach.

## Goals

1. Port the multi-pass architecture from Python to TypeScript
2. Enable advanced features like confidence metrics, multi-marking detection, and aggregate statistics
3. Support future ML fallback and hybrid detection strategies
4. Match Python architecture exactly for easier maintenance and feature parity

## Architecture Overview

### Python Multi-Pass Flow

```
TemplateFileRunner
  ├── TemplateDetectionPass (directory/file/field aggregates)
  ├── TemplateInterpretationPass (directory/file/field aggregates)
  └── FieldTypeFileLevelRunner (per field type)
        ├── FieldTypeDetectionPass
        │     └── BubblesThresholdDetectionPass
        └── FieldTypeInterpretationPass
              └── BubblesThresholdInterpretationPass
```

### Current TypeScript Flow (Simplified)

```
OMRProcessor
  └── BubblesFieldDetection (single pass, no aggregates)
```

### Target TypeScript Flow

```
OMRProcessor
  └── TemplateFileRunner
        ├── TemplateDetectionPass
        ├── TemplateInterpretationPass
        └── FieldTypeFileLevelRunner
              ├── FieldTypeDetectionPass
              │     └── BubblesThresholdDetectionPass
              └── FieldTypeInterpretationPass
                    └── BubblesThresholdInterpretationPass
```

## Phase 0: Prerequisites - Layout Infrastructure

The multi-pass detection architecture requires the layout classes (`Field`, `FieldBlock`, etc.) to be ported first, as they are used throughout all detection and interpretation passes.

### 0.1 Port Field Base Classes
**Files to create:**
- `omrchecker-js/packages/core/src/processors/layout/field/base.ts`

**Classes to port:**
- `Field` - Abstract base class for all field types
- `ScanBox` - Base class for individual scan boxes (bubbles, OCR zones, etc.)

**Key properties (Field):**
- `id` - Unique identifier (field_block_name::field_label)
- `field_label` - Label for the field
- `field_detection_type` - Type of detection (BUBBLES_THRESHOLD, OCR, BARCODE_QR)
- `field_block` - Reference to parent FieldBlock
- `scan_boxes` - List of ScanBox instances
- `direction` - Layout direction (horizontal/vertical)
- `empty_value` - Default value when field is empty
- `origin` - Origin point [x, y]

**Key methods (Field):**
- `setup_scan_boxes(field_block)` - Abstract method to populate scan boxes
- `get_drawing_instance()` - Abstract method to get drawing instance
- `reset_all_shifts()` - Reset shifts for all scan boxes
- `to_json()` - Serialize to JSON

**Key properties (ScanBox):**
- `field_index` - Index within the field
- `field` - Reference to parent Field
- `dimensions` - [width, height]
- `margins` - {top, bottom, left, right}
- `origin` - Origin point [x, y]
- `x`, `y` - Rounded coordinates
- `name` - Unique name (field_label_field_index)
- `shifts` - [x_shift, y_shift] for runtime adjustments

**Key methods (ScanBox):**
- `reset_shifts()` - Reset shifts to [0, 0]
- `get_shifted_position(shifts?)` - Get position with shifts applied
- `to_json()` - Serialize to JSON

**Implementation notes:**
- Use abstract classes with TypeScript
- Support JSON serialization for browser compatibility
- Match Python structure exactly

### 0.2 Port Bubble Field Classes
**Files to create:**
- `omrchecker-js/packages/core/src/processors/layout/field/bubbleField.ts`

**Classes to port:**
- `BubbleField` - Extends `Field` for bubble fields
- `BubblesScanBox` - Extends `ScanBox` for individual bubbles

**Key properties (BubbleField):**
- `bubble_dimensions` - [width, height] for bubbles
- `bubble_values` - Array of bubble values (e.g., ["A", "B", "C", "D"])
- `bubbles_gap` - Gap between bubbles in pixels
- `bubble_field_type` - Type of bubble field (QTYPE_MCQ4, QTYPE_INT, etc.)

**Key methods (BubbleField):**
- `setup_scan_boxes(field_block)` - Creates BubblesScanBox instances for each bubble value

**Key properties (BubblesScanBox):**
- `bubble_value` - Value of this bubble (e.g., "A", "B", "1", "2")
- `bubble_field_type` - Type of bubble field
- `bubble_dimensions` - Dimensions of the bubble

**Implementation notes:**
- Integrate with existing template types
- Support bubble value expansion (e.g., "A..D" → ["A", "B", "C", "D"])
- Match Python structure exactly

### 0.3 Port Field Block Base Class
**Files to create:**
- `omrchecker-js/packages/core/src/processors/layout/fieldBlock/base.ts`

**Classes to port:**
- `FieldBlock` - Container for a group of related fields

**Key properties (FieldBlock):**
- `name` - Name of the field block
- `origin` - Origin point [x, y]
- `fields` - List of Field instances
- `field_detection_type` - Type of detection for fields in this block
- `direction` - Layout direction (horizontal/vertical)
- `empty_value` - Default value for empty fields
- `labels_gap` - Gap between field labels
- `shifts` - [x_shift, y_shift] for runtime adjustments
- `bounding_box_origin` - Bounding box origin [x, y]
- `bounding_box_dimensions` - Bounding box dimensions [width, height]

**Key methods (FieldBlock):**
- `setup_field_block(field_block_object, field_blocks_offset)` - Initialize from template config
- `generate_fields()` - Create Field instances from parsed labels
- `reset_all_shifts()` - Reset shifts for block and all fields
- `get_shifted_origin()` - Get origin with shifts applied
- `update_bounding_box()` - Calculate bounding box from all scan boxes
- `get_copy_for_shifting()` - Create shallow copy for alignment processing
- `to_json()` - Serialize to JSON

**Field detection type mapping:**
```typescript
const field_detection_type_to_field_class = {
  FieldDetectionType.BUBBLES_THRESHOLD: BubbleField,
  FieldDetectionType.OCR: OCRField, // Future
  FieldDetectionType.BARCODE_QR: BarcodeField, // Future
};
```

**Implementation notes:**
- Support field label expansion (e.g., "q1..10" → ["q1", "q2", ..., "q10"])
- Handle bubble-specific setup (bubble_dimensions, bubble_values, bubbles_gap)
- Support OCR and barcode field blocks (for future)
- Integrate with TemplateLoader

### 0.4 Port Field Drawing Classes
**Files to create:**
- `omrchecker-js/packages/core/src/processors/layout/field/fieldDrawing.ts`

**Classes to port:**
- `FieldDrawing` - Base class for field drawing
- `BubbleFieldDrawing` - Drawing for bubble fields
- `BarcodeFieldDrawing` - Drawing for barcode fields (stub for future)
- `OCRFieldDrawing` - Drawing for OCR fields (stub for future)

**Key methods (FieldDrawing):**
- `draw_scan_boxes(marked_image, shifts, thickness_factor, border)` - Draw all scan boxes
- `draw_scan_boxes_util(field, marked_image, shifts, thickness_factor, border)` - Static utility

**Implementation notes:**
- Integrate with existing DrawingUtils
- Support shift adjustments for alignment
- Match Python drawing behavior

### 0.5 Port Field Block Drawing
**Files to create:**
- `omrchecker-js/packages/core/src/processors/layout/fieldBlock/fieldBlockDrawing.ts`

**Classes to port:**
- `FieldBlockDrawing` - Drawing for field blocks

**Implementation notes:**
- Can be stubbed initially if visualization not needed
- Integrate with DrawingUtils for bounding box visualization

### 0.6 Port Other Field Types (Optional for Phase 0)
**Files to create (for future support):**
- `omrchecker-js/packages/core/src/processors/layout/field/barcodeField.ts`
- `omrchecker-js/packages/core/src/processors/layout/field/ocrField.ts`

**Classes to port:**
- `BarcodeField` - For barcode/QR code fields
- `OCRField` - For OCR text fields

**Implementation notes:**
- Can be stubbed initially if only bubble detection is needed
- Full implementation needed for Phase 2 features

### 0.7 Update TemplateLoader Integration
**Files to update:**
- `omrchecker-js/packages/core/src/template/TemplateLoader.ts`

**Changes needed:**
- Generate `FieldBlock` instances from template config
- Generate `Field` instances within each `FieldBlock`
- Populate `all_fields` list for template
- Support field label expansion

**Implementation notes:**
- Integrate with existing template loading logic
- Ensure FieldBlock and Field instances are created correctly
- Match Python TemplateLayout behavior
- Use `parse_fields` utility for field label expansion (e.g., "q1..10" → ["q1", "q2", ..., "q10"])

### 0.8 Port Field Label Parsing Utility
**Files to create/update:**
- `omrchecker-js/packages/core/src/utils/parsing.ts` (if not exists)

**Functions to port:**
- `parse_fields(key: string, fields: string[]): string[]` - Expand field labels (e.g., "q1..10" → ["q1", "q2", ..., "q10"])

**Implementation notes:**
- Support range expansion (e.g., "q1..10")
- Support individual labels (e.g., ["q1", "q2", "q3"])
- Support mixed arrays
- Match Python regex-based parsing

### 0.9 Testing for Phase 0
**Test files to create:**
- `omrchecker-js/packages/core/src/processors/layout/field/__tests__/base.test.ts`
- `omrchecker-js/packages/core/src/processors/layout/field/__tests__/bubbleField.test.ts`
- `omrchecker-js/packages/core/src/processors/layout/fieldBlock/__tests__/base.test.ts`
- `omrchecker-js/packages/core/src/template/__tests__/TemplateLoader.layout.test.ts`

**Test coverage:**
- Field creation and serialization
- ScanBox shift handling
- BubbleField setup with bubble values
- FieldBlock field generation
- Field label expansion (e.g., "q1..10")
- TemplateLoader integration with FieldBlock/Field

## Phase 1: Foundation - Common Pass Infrastructure

### 1.1 Port Stats Utilities
**Files to create:**
- `omrchecker-js/packages/core/src/utils/stats.ts`

**Classes to port:**
- `StatsByLabel` - Label-based statistics tracking
- `NumberAggregate` - Numeric aggregation with running averages

**Implementation notes:**
- Use TypeScript classes with proper typing
- Support JSON serialization for browser compatibility
- Match Python API exactly

### 1.2 Port FilePassAggregates Base Class
**Files to create:**
- `omrchecker-js/packages/core/src/processors/detection/base/commonPass.ts`

**Class to port:**
- `FilePassAggregates` - Base class for managing aggregates at three levels:
  - Field level: Per-field aggregates
  - File level: Per-image aggregates
  - Directory level: Cross-image aggregates

**Key methods:**
- `initialize_field_level_aggregates(field)`
- `initialize_file_level_aggregates(file_path)`
- `initialize_directory_level_aggregates(initial_directory_path)`
- `get_field_level_aggregates()`
- `get_file_level_aggregates()`
- `get_directory_level_aggregates()`
- `insert_field_level_aggregates(next_aggregates)`
- `insert_file_level_aggregates(next_aggregates)`
- `insert_directory_level_aggregates(next_aggregates)`
- `update_field_level_aggregates_on_processed_field(field)`
- `update_file_level_aggregates_on_processed_field(field, field_level_aggregates)`
- `update_directory_level_aggregates_on_processed_field(field, field_level_aggregates)`
- `update_aggregates_on_processed_file(file_path)`

**Implementation notes:**
- Use TypeScript interfaces for aggregate structures
- Support generic types for type safety
- Match Python structure exactly

## Phase 2: Detection Pass Infrastructure

### 2.1 Port Detection Pass Base Classes
**Files to create:**
- `omrchecker-js/packages/core/src/processors/detection/base/detectionPass.ts`

**Classes to port:**
- `FieldTypeDetectionPass` - Abstract base for field type detection passes
- `TemplateDetectionPass` - Template-level detection pass

**Key methods (FieldTypeDetectionPass):**
- `get_field_detection(field, gray_image, colored_image)` - Abstract
- `update_aggregates_on_processed_field_detection(field, field_detection)`
- `update_field_level_aggregates_on_processed_field_detection(field, field_detection)`
- `update_file_level_aggregates_on_processed_field_detection(field, field_detection, field_level_aggregates)`
- `update_directory_level_aggregates_on_processed_field_detection(field, field_detection, field_level_aggregates)`

**Key methods (TemplateDetectionPass):**
- `initialize_directory_level_aggregates(initial_directory_path, all_field_detection_types)`
- `initialize_file_level_aggregates(file_path, all_field_detection_types)`
- `update_aggregates_on_processed_field_detection(field, field_detection)`
- `update_aggregates_on_processed_file(file_path, field_detection_type_file_runners)`

**Implementation notes:**
- Use abstract classes with TypeScript
- Support field detection type tracking
- Integrate with StatsByLabel for counting

### 2.2 Port Bubbles Threshold Detection Pass
**Files to create:**
- `omrchecker-js/packages/core/src/processors/detection/bubbles_threshold/detectionPass.ts`

**Class to port:**
- `BubblesThresholdDetectionPass` - Extends `FieldTypeDetectionPass`

**Key features:**
- Uses `BubblesFieldDetection` for actual detection
- Manages file-level aggregates (global_max_jump, all_field_bubble_means, all_field_bubble_means_std)
- Supports DetectionRepository pattern (optional, for future)
- Uses typed models exclusively (BubbleFieldDetectionResult)

**Key methods:**
- `get_field_detection(field, gray_image, colored_image)` - Returns BubblesFieldDetection
- `initialize_directory_level_aggregates(initial_directory_path)`
- `initialize_file_level_aggregates(file_path)`
- `update_field_level_aggregates_on_processed_field_detection(field, field_detection)`
- `update_file_level_aggregates_on_processed_field_detection(field, field_detection, field_level_aggregates)`

**Implementation notes:**
- Integrate with existing `BubblesFieldDetection` class
- Store detection results in aggregates using typed models
- Refactor `BubblesFieldDetection` to work with new architecture

## Phase 3: Interpretation Pass Infrastructure

### 3.1 Port Interpretation Base Classes
**Files to create:**
- `omrchecker-js/packages/core/src/processors/detection/base/interpretation.ts` (update existing)
- `omrchecker-js/packages/core/src/processors/detection/base/interpretationPass.ts` (new)

**Classes to port:**
- `BaseInterpretation` - Base for single interpretation result
- `FieldInterpretation` - Base for field interpretation (already exists, needs update)
- `FieldTypeInterpretationPass` - Abstract base for field type interpretation passes
- `TemplateInterpretationPass` - Template-level interpretation pass

**Key methods (FieldTypeInterpretationPass):**
- `get_field_interpretation(field, file_level_detection_aggregates, file_level_interpretation_aggregates)` - Abstract
- `initialize_field_level_aggregates(field)`
- `initialize_file_level_aggregates(file_path, field_detection_type_wise_detection_aggregates, field_label_wise_detection_aggregates)`
- `run_field_level_interpretation(field, file_level_detection_aggregates)`
- `update_field_level_aggregates_on_processed_field_interpretation(field, field_interpretation)`
- `update_file_level_aggregates_on_processed_field_interpretation(field, field_interpretation, field_level_aggregates)`
- `update_directory_level_aggregates_on_processed_field_interpretation(field, field_interpretation, field_level_aggregates)`

**Key methods (TemplateInterpretationPass):**
- `initialize_file_level_aggregates(file_path, all_field_detection_types, ...)`
- `initialize_directory_level_aggregates(initial_directory_path, all_field_detection_types)`
- `run_field_level_interpretation(field, field_interpretation, field_type_runner_field_level_aggregates, current_omr_response)`
- `update_aggregates_on_processed_file(file_path, field_detection_type_file_runners)`

**Implementation notes:**
- Update existing `FieldInterpretation` to match Python structure
- Support confidence metrics tracking
- Handle multi-marking detection
- Support read_response_flags

### 3.2 Port Bubbles Threshold Interpretation
**Files to create:**
- `omrchecker-js/packages/core/src/processors/detection/bubbles_threshold/interpretation.ts` (new)
- `omrchecker-js/packages/core/src/processors/detection/bubbles_threshold/interpretationPass.ts` (new)

**Classes to port:**
- `BubbleInterpretation` - Single bubble interpretation result
- `BubblesFieldInterpretation` - Field-level bubble interpretation (already exists, needs major update)
- `BubblesThresholdInterpretationPass` - Extends `FieldTypeInterpretationPass`

**Key features:**
- Threshold calculation using strategies (already ported)
- Multi-marking detection
- Confidence metrics calculation
- Integration with detection aggregates

**Key methods (BubblesFieldInterpretation):**
- `run_interpretation(field, file_level_detection_aggregates, file_level_interpretation_aggregates)`
- `get_field_interpretation_string()` - Returns final answer string
- `_extract_detection_result(field, file_level_detection_aggregates)`
- `_create_threshold_config(file_level_interpretation_aggregates)`
- `_calculate_threshold(detection_result, file_level_interpretation_aggregates, config)`
- `_interpret_bubbles(detection_result)`
- `_check_multi_marking()`
- `_calculate_confidence_metrics(detection_result, file_level_aggregates)`
- `_calculate_overall_confidence_score(detection_result, disparity_bubbles)`

**Key methods (BubblesThresholdInterpretationPass):**
- `get_field_interpretation(field, file_level_detection_aggregates, file_level_aggregates)`
- `initialize_file_level_aggregates(file_path, field_detection_type_wise_detection_aggregates, field_label_wise_detection_aggregates)`
- `get_outlier_deviation_threshold(all_outlier_deviations)`
- `get_fallback_threshold(field_wise_means_and_refs)`
- `update_field_level_aggregates_on_processed_field_interpretation(field, field_interpretation)`
- `update_file_level_aggregates_on_processed_field_interpretation(field, field_interpretation, field_level_aggregates)`

**Implementation notes:**
- Complete refactor of existing `BubblesFieldDetection` - separate detection from interpretation
- Remove single-pass logic, use multi-pass architecture exclusively
- Use threshold strategies already ported
- Support confidence metrics for ML training
- Handle multi-marking edge cases

## Phase 4: File Runner Infrastructure

### 4.1 Port File Runner Base Classes
**Files to create:**
- `omrchecker-js/packages/core/src/processors/detection/base/fileRunner.ts`

**Classes to port:**
- `FileLevelRunner<DetectionPassT, InterpretationPassT>` - Generic runner coordinating detection and interpretation
- `FieldTypeFileLevelRunner` - Specialized for specific field types

**Key methods (FileLevelRunner):**
- `initialize_directory_level_aggregates(initial_directory_path)`
- `get_field_level_detection_aggregates()`
- `initialize_file_level_detection_aggregates(file_path)`
- `get_file_level_detection_aggregates()`
- `update_detection_aggregates_on_processed_file(file_path)`
- `get_directory_level_detection_aggregates()`
- `get_field_level_interpretation_aggregates()`
- `initialize_file_level_interpretation_aggregates(file_path, field_detection_type_wise_detection_aggregates, field_label_wise_detection_aggregates)`
- `get_file_level_interpretation_aggregates()`
- `update_interpretation_aggregates_on_processed_file(file_path)`
- `get_directory_level_interpretation_aggregates()`

**Key methods (FieldTypeFileLevelRunner):**
- `run_field_level_detection(field, gray_image, colored_image)` - Returns FieldDetection
- `run_field_level_interpretation(field)` - Returns FieldInterpretation

**Implementation notes:**
- Use TypeScript generics for type safety
- Coordinate between detection and interpretation passes
- Maintain aggregate state

### 4.2 Port Bubbles Threshold File Runner
**Files to create:**
- `omrchecker-js/packages/core/src/processors/detection/bubbles_threshold/fileRunner.ts`

**Class to port:**
- `BubblesThresholdFileRunner` - Extends `FieldTypeFileLevelRunner`

**Implementation notes:**
- Instantiate `BubblesThresholdDetectionPass` and `BubblesThresholdInterpretationPass`
- Pass tuning config to both passes
- Set field_detection_type

## Phase 5: Template File Runner

### 5.1 Port Template File Runner
**Files to create:**
- `omrchecker-js/packages/core/src/processors/detection/templateFileRunner.ts`

**Class to port:**
- `TemplateFileRunner` - Extends `FileLevelRunner<TemplateDetectionPass, TemplateInterpretationPass>`

**Key features:**
- Manages field type runners (bubbles, OCR, barcode)
- Coordinates detection and interpretation at template level
- Handles directory-level aggregates

**Key methods:**
- `initialize_field_file_runners(template)`
- `get_field_detection_type_file_runner(field_detection_type)`
- `read_omr_and_update_metrics(file_path, gray_image, colored_image)`
- `run_file_level_detection(file_path, gray_image, colored_image)`
- `run_field_level_detection(field, gray_image, colored_image)`
- `run_file_level_interpretation(file_path, gray_image, colored_image)`
- `run_field_level_interpretation(field, current_omr_response)`
- `initialize_directory_level_aggregates(template)`
- `initialize_file_level_detection_aggregates(file_path)`
- `initialize_file_level_interpretation_aggregates(file_path)`
- `update_detection_aggregates_on_processed_file(file_path)`
- `update_interpretation_aggregates_on_processed_file(file_path)`
- `finish_processing_directory()`

**Field detection type mapping:**
```typescript
const field_detection_type_to_runner = {
  FieldDetectionType.BUBBLES_THRESHOLD: BubblesThresholdFileRunner,
  FieldDetectionType.OCR: OCRFileRunner, // Future
  FieldDetectionType.BARCODE_QR: BarcodeFileRunner, // Future
};
```

**Implementation notes:**
- Integrate with TemplateLoader
- Support multiple field detection types
- Clean API matching Python exactly

## Phase 6: Integration with OMRProcessor

### 6.1 Update OMRProcessor
**Files to update:**
- `omrchecker-js/packages/core/src/core/OMRProcessor.ts`

**Changes needed:**
- Replace direct `BubblesFieldDetection` calls with `TemplateFileRunner`
- Support two-pass processing:
  1. Detection pass: Run detection on all fields, collect aggregates
  2. Interpretation pass: Use aggregates to interpret fields
- Refactor API to match Python architecture

**New flow:**
```typescript
class OMRProcessor {
  private templateFileRunner: TemplateFileRunner | null = null;

  async processImage(image: File | ImageData, template: TemplateConfig): Promise<OMRResult> {
    // Initialize template file runner if needed
    if (!this.templateFileRunner || this.templateFileRunner.template !== template) {
      this.templateFileRunner = new TemplateFileRunner(template);
    }

    // Load and preprocess image
    const grayImage = await ImageUtils.loadImage(image);
    const coloredImage = await ImageUtils.loadImage(image, { grayscale: false });

    // Run two-pass detection and interpretation
    const omrResponse = this.templateFileRunner.read_omr_and_update_metrics(
      image.name || 'unknown',
      grayImage,
      coloredImage
    );

    // Convert to OMRResult format with aggregates and confidence metrics
    return this._convertToOMRResult(omrResponse);
  }

  // New method to access aggregates for advanced use cases
  getAggregates(): DirectoryLevelAggregates {
    return this.templateFileRunner?.get_directory_level_aggregates() || null;
  }
}
```

**Implementation notes:**
- Refactor `processImage()` and `processBatch()` to use TemplateFileRunner
- Include aggregates and confidence metrics in results by default
- Match Python API structure exactly

## Phase 7: Testing and Validation

### 7.1 Unit Tests
**Test files to create:**
- `omrchecker-js/packages/core/src/utils/__tests__/stats.test.ts`
- `omrchecker-js/packages/core/src/processors/detection/base/__tests__/commonPass.test.ts`
- `omrchecker-js/packages/core/src/processors/detection/base/__tests__/detectionPass.test.ts`
- `omrchecker-js/packages/core/src/processors/detection/base/__tests__/interpretationPass.test.ts`
- `omrchecker-js/packages/core/src/processors/detection/base/__tests__/fileRunner.test.ts`
- `omrchecker-js/packages/core/src/processors/detection/bubbles_threshold/__tests__/detectionPass.test.ts`
- `omrchecker-js/packages/core/src/processors/detection/bubbles_threshold/__tests__/interpretationPass.test.ts`
- `omrchecker-js/packages/core/src/processors/detection/bubbles_threshold/__tests__/fileRunner.test.ts`
- `omrchecker-js/packages/core/src/processors/detection/__tests__/templateFileRunner.test.ts`

**Test coverage:**
- Aggregate initialization and updates
- Detection pass flow
- Interpretation pass flow
- Multi-marking detection
- Confidence metrics calculation
- Threshold calculation integration

### 7.2 Integration Tests
**Test files to update:**
- `omrchecker-js/packages/core/src/__tests__/e2e-complete-pipeline.test.ts`

**Test scenarios:**
- End-to-end processing with multi-pass architecture
- Aggregate collection across multiple images
- Confidence metrics in results
- Multi-marking detection

### 7.3 Performance Tests
**Test files to create:**
- `omrchecker-js/packages/core/src/core/__tests__/OMRProcessor.performance.test.ts`

**Test scenarios:**
- Multi-pass processing performance benchmarks
- Aggregate collection overhead
- Memory usage with multiple images
- Comparison with Python implementation performance

## Phase 8: Documentation and Migration

### 8.1 Update Documentation
- Update README with multi-pass architecture explanation
- Document aggregate access for advanced use cases
- Document new API structure matching Python
- Add examples for accessing confidence metrics and aggregates

### 8.2 Update FILE_MAPPING.json
- Mark detection pass files as synced
- Mark interpretation pass files as synced
- Mark file runner files as synced
- Update statistics

## Implementation Order

1. **Phase 0** - Layout infrastructure (Field, FieldBlock) - **REQUIRED PREREQUISITE**
2. **Phase 1** - Foundation (stats, common pass) - Enables everything else
3. **Phase 2** - Detection pass - Can be tested independently
4. **Phase 3** - Interpretation pass - Depends on detection pass
5. **Phase 4** - File runners - Depends on detection and interpretation passes
6. **Phase 5** - Template file runner - Depends on file runners
7. **Phase 6** - OMRProcessor integration - Depends on template file runner
8. **Phase 7** - Testing - Throughout all phases
9. **Phase 8** - Documentation - After implementation

## Breaking Changes

Since backward compatibility is not maintained, the following breaking changes will occur:

### API Changes
- `OMRProcessor.processImage()` will now use `TemplateFileRunner` internally
- Result format will include aggregates and confidence metrics by default
- Direct `BubblesFieldDetection` usage will be replaced with multi-pass flow

### Code Structure Changes
- `BubblesFieldDetection` will be refactored to work only within detection pass
- Single-pass detection logic will be removed
- Interpretation logic will be separated into `BubblesFieldInterpretation`

### Data Format Changes
- Results will use typed models exclusively (no legacy dict format)
- Aggregates will be available in results structure
- Confidence metrics included in field-level results

## Key Design Decisions

### 1. Architecture Matching
- Match Python architecture exactly for easier maintenance
- Use typed models exclusively (no legacy dict format)
- Clean API that reflects multi-pass structure

### 2. Type Safety
- Use TypeScript generics for pass types
- Strong typing for aggregates
- Interface-based aggregate structures

### 3. Performance
- Lazy initialization of aggregates
- Efficient aggregate updates
- Minimal memory overhead

### 4. Extensibility
- Abstract base classes for easy extension
- Strategy pattern for threshold calculation (already implemented)
- Support for future field detection types (OCR, barcode)

## Success Criteria

1. ✅ Layout infrastructure ported (Field, FieldBlock, BubbleField)
2. ✅ All Python multi-pass classes ported to TypeScript
3. ✅ Two-pass processing working (detection → interpretation)
4. ✅ Aggregates collected at field, file, and directory levels
5. ✅ Confidence metrics calculated and available
6. ✅ Multi-marking detection working
7. ✅ Architecture matches Python exactly
8. ✅ All tests passing
9. ✅ Performance acceptable (< 2x overhead vs single-pass)
10. ✅ Documentation updated
11. ✅ API clean and matches Python structure

## Risks and Mitigations

### Risk 1: Performance Degradation
**Mitigation:** Profile and optimize aggregate updates, use lazy initialization

### Risk 2: Breaking Changes
**Mitigation:** Clear migration documentation, version bump, comprehensive changelog

### Risk 3: Complexity Increase
**Mitigation:** Clear documentation, well-structured code, comprehensive tests

### Risk 4: Type Safety Issues
**Mitigation:** Strong typing, generic constraints, comprehensive type tests

## Timeline Estimate

- **Phase 0**: 4-5 days (Layout infrastructure - Field, FieldBlock, BubbleField)
- **Phase 1**: 2-3 days
- **Phase 2**: 3-4 days
- **Phase 3**: 4-5 days
- **Phase 4**: 2-3 days
- **Phase 5**: 3-4 days
- **Phase 6**: 2-3 days
- **Phase 7**: 3-4 days (parallel with implementation)
- **Phase 8**: 1-2 days

**Total**: ~24-33 days (5-7 weeks)

## Next Steps

1. Review and approve this plan
2. **Start with Phase 0 (Layout Infrastructure)** - Required prerequisite
3. Then proceed with Phase 1 (Foundation)
4. Implement incrementally with tests
5. Regular sync with Python codebase
6. Update FILE_MAPPING.json as files are ported

## Dependencies

- **Phase 0 → Phase 1**: Field and FieldBlock classes needed for FilePassAggregates
- **Phase 0 → Phase 2**: Field classes needed for detection passes
- **Phase 0 → Phase 3**: Field classes needed for interpretation passes
- **Phase 0 → Phase 5**: FieldBlock needed for TemplateFileRunner
- **Phase 1 → Phase 2**: FilePassAggregates needed for detection passes
- **Phase 2 → Phase 3**: Detection passes needed for interpretation passes
- **Phase 2+3 → Phase 4**: Detection and interpretation passes needed for file runners
- **Phase 4 → Phase 5**: File runners needed for template file runner
- **Phase 5 → Phase 6**: Template file runner needed for OMRProcessor

