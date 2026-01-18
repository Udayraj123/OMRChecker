# Test Coverage Analysis for OMRChecker

## Executive Summary

This document identifies critical test gaps in both Python and TypeScript codebases to achieve high test coverage (>85%). The analysis focuses on core functionality, edge cases, error handling, and integration scenarios.

---

## 1. Core Processing Pipeline (HIGH PRIORITY)

### 1.1 Entry Point (`src/entry.py`)
**Status**: ⚠️ **MISSING TESTS**

**Critical Functions to Test**:
- `entry_point()` - Main entry function
- `process_directory_wise()` - Recursive directory processing
- `print_config_summary()` - Configuration display

**Test Cases Needed**:
```python
# test_entry.py
- test_entry_point_with_valid_directory()
- test_entry_point_with_invalid_directory()
- test_process_directory_wise_recursive()
- test_process_directory_wise_with_local_config()
- test_process_directory_wise_with_local_template()
- test_process_directory_wise_with_evaluation_config()
- test_print_config_summary_all_fields()
- test_process_directory_wise_error_handling()
- test_concurrent_processing()
```

**TypeScript Equivalent**: `OMRProcessor` tests exist but need expansion

---

### 1.2 Template Class (`src/processors/template/template.py`)
**Status**: ⚠️ **MISSING TESTS**

**Critical Methods to Test**:
- `__init__()` - Template initialization
- `apply_preprocessors()` - Preprocessor pipeline
- `reset_and_setup_for_directory()` - Directory reset
- `get_exclude_files()` - File exclusion logic
- `get_pre_processors()` - Preprocessor retrieval
- `get_concatenated_omr_response()` - Response concatenation

**Test Cases Needed**:
```python
# test_template.py
- test_template_initialization()
- test_template_with_custom_preprocessors()
- test_apply_preprocessors_sequence()
- test_apply_preprocessors_with_alignment()
- test_reset_and_setup_for_directory()
- test_get_exclude_files_with_alignment()
- test_get_exclude_files_with_preprocessors()
- test_get_concatenated_omr_response_single_column()
- test_get_concatenated_omr_response_multi_column()
- test_get_concatenated_omr_response_custom_labels()
```

**TypeScript**: `TemplateLoader.test.ts` exists but needs `TemplateLayout` method tests

---

### 1.3 TemplateLayout (`src/processors/layout/template_layout.py`)
**Status**: ⚠️ **MISSING TESTS**

**Critical Methods to Test** (17 methods):
- `__init__()` - Layout initialization
- `get_copy_for_shifting()` - Shallow/deep copy logic
- `apply_preprocessors()` - Full preprocessor pipeline
- `setup_pre_processors()` - Preprocessor setup
- `setup_alignment()` - Alignment configuration
- `parse_output_columns()` - Column parsing
- `parse_custom_bubble_field_types()` - Custom types
- `validate_field_blocks()` - Field block validation
- `parse_custom_labels()` - Custom label parsing
- `get_concatenated_omr_response()` - Response concatenation
- `fill_output_columns()` - Column filling
- `validate_template_columns()` - Column validation
- `parse_and_add_field_block()` - Dynamic field block addition
- `prefill_field_block()` - Field block prefilling
- `validate_parsed_field_block()` - Field block validation
- `reset_all_shifts()` - Shift reset
- `to_json()` - Serialization

**Test Cases Needed**:
```python
# test_template_layout.py
- test_template_layout_initialization()
- test_get_copy_for_shifting_shallow_copy()
- test_get_copy_for_shifting_deep_copy_field_blocks()
- test_apply_preprocessors_full_pipeline()
- test_apply_preprocessors_with_save_image_ops()
- test_setup_pre_processors_with_valid_processors()
- test_setup_pre_processors_with_invalid_processor()
- test_setup_alignment_with_reference_image()
- test_setup_alignment_without_reference_image()
- test_parse_output_columns_custom_sort()
- test_parse_output_columns_numeric_sort()
- test_parse_custom_bubble_field_types()
- test_validate_field_blocks_valid()
- test_validate_field_blocks_invalid_bubble_type()
- test_validate_field_blocks_missing_labels_gap()
- test_parse_custom_labels_valid()
- test_parse_custom_labels_overlapping()
- test_parse_custom_labels_missing_labels()
- test_get_concatenated_omr_response_single_column()
- test_get_concatenated_omr_response_multi_column()
- test_fill_output_columns_auto()
- test_validate_template_columns_valid()
- test_validate_template_columns_missing()
- test_parse_and_add_field_block()
- test_prefill_field_block()
- test_validate_parsed_field_block()
- test_reset_all_shifts()
- test_to_json_serialization()
```

**TypeScript**: Missing comprehensive `TemplateLayout` tests

---

## 2. Multi-Pass Detection Architecture (HIGH PRIORITY)

### 2.1 TemplateFileRunner (`src/processors/detection/template_file_runner.py`)
**Status**: ⚠️ **MISSING TESTS**

**Critical Methods to Test**:
- `__init__()` - Runner initialization
- `initialize_field_file_runners()` - Field runner setup
- `read_omr_and_update_metrics()` - Two-pass processing
- `run_file_level_detection()` - Detection pass
- `run_field_level_detection()` - Field-level detection
- `run_file_level_interpretation()` - Interpretation pass
- `run_field_level_interpretation()` - Field-level interpretation
- `initialize_directory_level_aggregates()` - Aggregate initialization
- `update_detection_aggregates_on_processed_file()` - Aggregate updates
- `update_interpretation_aggregates_on_processed_file()` - Interpretation aggregates
- `finish_processing_directory()` - Directory completion
- `get_export_omr_metrics_for_file()` - Metrics export

**Test Cases Needed**:
```python
# test_template_file_runner.py
- test_initialize_field_file_runners_single_type()
- test_initialize_field_file_runners_multiple_types()
- test_read_omr_and_update_metrics_two_pass()
- test_run_file_level_detection_all_fields()
- test_run_field_level_detection_bubbles()
- test_run_field_level_detection_ocr()
- test_run_field_level_detection_barcode()
- test_run_file_level_interpretation()
- test_run_field_level_interpretation()
- test_initialize_directory_level_aggregates()
- test_update_detection_aggregates_on_processed_file()
- test_update_interpretation_aggregates_on_processed_file()
- test_aggregate_collection_across_files()
- test_finish_processing_directory()
- test_get_export_omr_metrics_for_file()
- test_multi_file_processing_aggregates()
```

**TypeScript**: `templateFileRunner.test.ts` exists but needs expansion

---

### 2.2 Detection Pass (`src/processors/detection/base/detection_pass.py`)
**Status**: ⚠️ **PARTIAL TESTS**

**Critical Methods to Test**:
- `run_detection()` - Detection execution
- `initialize_field_level_aggregates()` - Field aggregate init
- `update_aggregates_on_processed_field_detection()` - Aggregate updates

**Test Cases Needed**:
```python
# test_detection_pass.py
- test_run_detection_single_field()
- test_run_detection_multiple_fields()
- test_initialize_field_level_aggregates()
- test_update_aggregates_on_processed_field_detection()
- test_detection_aggregate_accumulation()
```

**TypeScript**: `detectionPass.test.ts` exists

---

### 2.3 Interpretation Pass (`src/processors/detection/base/interpretation_pass.py`)
**Status**: ⚠️ **PARTIAL TESTS**

**Critical Methods to Test**:
- `run_interpretation()` - Interpretation execution
- `update_aggregates_on_processed_field_interpretation()` - Aggregate updates

**Test Cases Needed**:
```python
# test_interpretation_pass.py
- test_run_interpretation_single_field()
- test_run_interpretation_multiple_fields()
- test_update_aggregates_on_processed_field_interpretation()
- test_interpretation_aggregate_accumulation()
- test_confidence_metrics_collection()
```

**TypeScript**: `interpretationPass.test.ts` exists

---

### 2.4 File Runner Base Classes (`src/processors/detection/base/file_runner.py`)
**Status**: ⚠️ **MISSING TESTS**

**Critical Classes to Test**:
- `FileLevelRunner` - Base file runner
- `FieldTypeFileLevelRunner` - Field type runner

**Test Cases Needed**:
```python
# test_file_runner.py
- test_file_level_runner_initialization()
- test_field_type_file_level_runner_initialization()
- test_aggregate_management_hierarchy()
- test_file_level_aggregate_updates()
- test_directory_level_aggregate_updates()
```

**TypeScript**: `fileRunner.test.ts` exists

---

## 3. Evaluation System (HIGH PRIORITY)

### 3.1 EvaluationConfigForSet (`src/processors/evaluation/evaluation_config_for_set.py`)
**Status**: ⚠️ **MISSING TESTS**

**Critical Methods to Test** (29 methods):
- `__init__()` - Configuration initialization
- `parse_local_question_answers()` - Local parsing
- `parse_csv_question_answers()` - CSV parsing
- `merge_parsed_questions_and_schemes_with_parent()` - Parent merging
- `validate_questions()` - Question validation
- `set_parsed_marking_schemes()` - Marking scheme setup
- `update_marking_schemes_from_parent()` - Parent scheme updates
- `validate_marking_schemes()` - Scheme validation
- `parse_answers_and_map_questions()` - Answer parsing
- `validate_answers()` - Answer validation
- `validate_format_strings()` - Format string validation
- `prepare_and_validate_omr_response()` - OMR response validation
- `match_answer_for_question()` - Answer matching
- `get_evaluation_meta_for_question()` - Evaluation metadata
- `get_formatted_answers_summary()` - Summary formatting
- `get_formatted_score()` - Score formatting
- `reset_evaluation()` - Evaluation reset

**Test Cases Needed**:
```python
# test_evaluation_config_for_set.py
- test_initialization_with_local_answers()
- test_initialization_with_csv_answers()
- test_initialization_with_image_answer_key()
- test_merge_with_parent_config()
- test_validate_questions_equal_lengths()
- test_validate_questions_unequal_lengths()
- test_set_parsed_marking_schemes_default()
- test_set_parsed_marking_schemes_custom()
- test_update_marking_schemes_from_parent()
- test_validate_marking_schemes_no_overlap()
- test_validate_marking_schemes_with_overlap()
- test_validate_marking_schemes_missing_questions()
- test_parse_answers_and_map_questions_standard()
- test_parse_answers_and_map_questions_multiple_correct()
- test_parse_answers_and_map_questions_weighted()
- test_validate_answers_no_multimarked()
- test_validate_answers_with_multimarked()
- test_validate_format_strings_valid()
- test_validate_format_strings_invalid()
- test_prepare_and_validate_omr_response_valid()
- test_prepare_and_validate_omr_response_missing_keys()
- test_match_answer_for_question_correct()
- test_match_answer_for_question_incorrect()
- test_match_answer_for_question_unmarked()
- test_get_evaluation_meta_for_question()
- test_get_formatted_answers_summary()
- test_get_formatted_score()
- test_reset_evaluation()
```

**TypeScript**: `EvaluationConfig.test.ts` exists but needs expansion

---

### 3.2 EvaluationConfig (`src/processors/evaluation/evaluation_config.py`)
**Status**: ⚠️ **MISSING TESTS**

**Critical Methods to Test**:
- `__init__()` - Config initialization
- `validate_conditional_sets()` - Conditional set validation
- `get_evaluation_config_for_response()` - Response matching
- `get_matching_set()` - Set matching logic

**Test Cases Needed**:
```python
# test_evaluation_config.py
- test_initialization_with_default_only()
- test_initialization_with_conditional_sets()
- test_validate_conditional_sets_unique_names()
- test_validate_conditional_sets_duplicate_names()
- test_get_evaluation_config_for_response_default()
- test_get_evaluation_config_for_response_matched_set()
- test_get_matching_set_by_field_value()
- test_get_matching_set_by_file_path()
```

**TypeScript**: `EvaluationConfig.test.ts` exists

---

### 3.3 AnswerMatcher (`src/processors/evaluation/answer_matcher.py`)
**Status**: ✅ **HAS TESTS** (`test_refactored_detection.py`)

**Additional Test Cases Needed**:
- Edge cases for weighted answers
- Multi-marking scenarios
- Empty value handling

---

### 3.4 SectionMarkingScheme (`src/processors/evaluation/section_marking_scheme.py`)
**Status**: ⚠️ **MISSING TESTS**

**Test Cases Needed**:
```python
# test_section_marking_scheme.py
- test_default_marking_scheme()
- test_custom_marking_scheme()
- test_verdict_level_streak()
- test_section_level_streak()
- test_bonus_questions()
```

**TypeScript**: `SectionMarkingScheme.test.ts` exists

---

## 4. Image Processors (MEDIUM PRIORITY)

### 4.1 PROCESSOR_MANAGER (`src/processors/manager.py`)
**Status**: ⚠️ **MISSING TESTS**

**Test Cases Needed**:
```python
# test_processor_manager.py
- test_all_supported_processors_registered()
- test_processor_keys_match_supported_names()
- test_processor_instantiation()
```

**TypeScript**: Missing tests

---

### 4.2 Preprocessor Pipeline (`src/processors/image/coordinator.py`)
**Status**: ⚠️ **MISSING TESTS**

**Test Cases Needed**:
```python
# test_preprocessing_coordinator.py
- test_preprocessing_sequence()
- test_preprocessing_with_save_image_ops()
- test_preprocessing_error_handling()
```

**TypeScript**: `coordinator.test.ts` exists

---

## 5. Template Drawing (MEDIUM PRIORITY)

### 5.1 TemplateDrawing (`src/processors/layout/template_drawing.py`)
**Status**: ⚠️ **MISSING TESTS**

**Critical Methods to Test**:
- `draw_template_layout()` - Full template drawing
- `draw_template_layout_util()` - Drawing utility
- `draw_all_fields()` - Field drawing
- `draw_evaluation_summary()` - Summary drawing

**Test Cases Needed**:
```python
# test_template_drawing.py
- test_draw_template_layout_without_interpretations()
- test_draw_template_layout_with_interpretations()
- test_draw_template_layout_with_evaluation()
- test_draw_all_fields_bubbles()
- test_draw_all_fields_ocr()
- test_draw_all_fields_barcode()
- test_draw_evaluation_summary()
- test_draw_evaluation_summary_with_score()
```

**TypeScript**: Missing tests

---

## 6. Utilities (MEDIUM PRIORITY)

### 6.1 FilePatternResolver (`src/utils/file_pattern_resolver.py`)
**Status**: ✅ **HAS TESTS** (`test_file_pattern_resolver.py`)

**Additional Test Cases**:
- Collision handling edge cases
- Path sanitization

---

### 6.2 Serialization (`src/utils/serialization.py`)
**Status**: ✅ **HAS TESTS** (`test_dataclass_serialization.py`)

---

### 6.3 Math Utils (`src/utils/math.py`)
**Status**: ⚠️ **MISSING TESTS**

**Test Cases Needed**:
```python
# test_math_utils.py
- test_order_four_points()
- test_get_rectangle_points_from_box()
- test_shift_points_from_origin()
- test_to_bgr_color_conversion()
```

**TypeScript**: Missing tests

---

### 6.4 Drawing Utils (`src/utils/drawing.py`)
**Status**: ✅ **HAS TESTS** (`test_drawing.py`)

**Additional Test Cases**:
- `draw_symbol()` - Symbol drawing
- `draw_group()` - Group drawing
- Edge cases for text positioning

---

## 7. Error Handling & Edge Cases (HIGH PRIORITY)

### 7.1 Exception Handling
**Status**: ✅ **HAS TESTS** (`test_exceptions.py`)

**Additional Test Cases Needed**:
- Template validation errors
- Image processing errors
- Configuration errors
- Evaluation errors

---

### 7.2 Edge Cases
**Status**: ✅ **HAS TESTS** (`test_edge_cases.py`)

**Additional Test Cases**:
- Empty templates
- Templates with single field
- Very large templates
- Multi-marking scenarios
- Low confidence detections

---

## 8. Integration Tests (HIGH PRIORITY)

### 8.1 End-to-End Pipeline
**Status**: ⚠️ **PARTIAL TESTS**

**Test Cases Needed**:
```python
# test_e2e_pipeline.py
- test_complete_pipeline_single_image()
- test_complete_pipeline_multiple_images()
- test_pipeline_with_preprocessing()
- test_pipeline_with_alignment()
- test_pipeline_with_evaluation()
- test_pipeline_error_recovery()
- test_pipeline_performance_large_batch()
```

**TypeScript**: `e2e-complete-pipeline.test.ts` exists

---

### 8.2 Multi-Pass Architecture Integration
**Status**: ⚠️ **PARTIAL TESTS**

**Test Cases Needed**:
```python
# test_multi_pass_integration.py
- test_two_pass_detection_interpretation()
- test_aggregate_collection_across_files()
- test_global_threshold_calculation()
- test_local_threshold_calculation()
- test_confidence_metrics_accumulation()
```

**TypeScript**: `e2e-multi-pass-architecture.test.ts` exists

---

## 9. Schema Validation (MEDIUM PRIORITY)

### 9.1 Template Schema
**Status**: ✅ **HAS TESTS** (`test_template_validations.py`)

---

### 9.2 Config Schema
**Status**: ✅ **HAS TESTS** (`test_config_validations.py`)

---

### 9.3 Evaluation Schema
**Status**: ⚠️ **MISSING TESTS**

**Test Cases Needed**:
```python
# test_evaluation_schema.py
- test_evaluation_config_validation()
- test_marking_scheme_validation()
- test_conditional_sets_validation()
```

**TypeScript**: `evaluationSchema.test.ts` exists

---

## 10. TypeScript-Specific Test Gaps

### 10.1 TemplateLayout Class
**Status**: ⚠️ **MISSING TESTS**

**Test Cases Needed**:
```typescript
// TemplateLayout.test.ts
- All 17 methods from Python test_template_layout.py
- Async method testing (setupAlignment, applyPreprocessors)
- Browser compatibility tests
```

---

### 10.2 EvaluationConfigForSet
**Status**: ⚠️ **NEEDS EXPANSION**

**Test Cases Needed**:
```typescript
// EvaluationConfigForSet.test.ts
- All validation methods
- Format string validation
- Answer validation
- Marking scheme validation
```

---

### 10.3 PROCESSOR_MANAGER
**Status**: ⚠️ **MISSING TESTS**

**Test Cases Needed**:
```typescript
// processorManager.test.ts
- Factory function instantiation
- Processor validation
- Error handling for unknown processors
```

---

## Priority Summary

### 🔴 CRITICAL (Must Have - Blocks Core Functionality)
1. **TemplateLayout** - 17 methods, core template processing
2. **TemplateFileRunner** - Multi-pass architecture orchestration
3. **EvaluationConfigForSet** - 29 methods, evaluation logic
4. **Entry Point** - Main processing flow
5. **Template Class** - Template lifecycle management

### 🟡 HIGH (Important - Affects Quality)
6. **Detection/Interpretation Passes** - Core detection logic
7. **EvaluationConfig** - Conditional set matching
8. **Integration Tests** - End-to-end scenarios
9. **Error Handling** - Robustness

### 🟢 MEDIUM (Nice to Have - Completeness)
10. **TemplateDrawing** - Visualization
11. **PROCESSOR_MANAGER** - Processor registry
12. **Math Utils** - Utility functions
13. **Schema Validation** - Input validation

---

## Recommended Test Implementation Order

### Phase 1: Core Infrastructure (Week 1)
1. `test_template_layout.py` - All 17 methods
2. `test_template_file_runner.py` - Multi-pass orchestration
3. `test_evaluation_config_for_set.py` - All 29 methods

### Phase 2: Integration & Edge Cases (Week 2)
4. `test_entry.py` - Entry point and directory processing
5. `test_template.py` - Template class lifecycle
6. `test_e2e_pipeline.py` - End-to-end scenarios

### Phase 3: Supporting Systems (Week 3)
7. `test_template_drawing.py` - Drawing functionality
8. `test_processor_manager.py` - Processor registry
9. `test_math_utils.py` - Utility functions

### Phase 4: TypeScript Parity (Week 4)
10. TypeScript equivalents of all Python tests
11. Browser compatibility tests
12. Performance benchmarks

---

## Test Coverage Goals

| Component | Current | Target | Priority |
|-----------|---------|--------|----------|
| TemplateLayout | ~20% | 90% | 🔴 Critical |
| TemplateFileRunner | ~30% | 90% | 🔴 Critical |
| EvaluationConfigForSet | ~40% | 90% | 🔴 Critical |
| Entry Point | ~10% | 85% | 🔴 Critical |
| Template Class | ~15% | 85% | 🔴 Critical |
| Detection Passes | ~50% | 85% | 🟡 High |
| Interpretation Passes | ~50% | 85% | 🟡 High |
| TemplateDrawing | ~10% | 80% | 🟢 Medium |
| Utilities | ~60% | 80% | 🟢 Medium |

**Overall Target**: 85%+ coverage for critical paths, 75%+ overall

---

## Notes

- Focus on **1:1 mapping** between Python and TypeScript tests
- Prioritize **integration tests** over unit tests for complex flows
- Include **error handling** and **edge cases** in all test suites
- Use **fixtures** for common test data (templates, images, configs)
- Add **performance benchmarks** for critical paths

