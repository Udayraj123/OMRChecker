# Unit Test Coverage Summary

**Generated:** $(date)

## Overall Statistics

- **Total Test Files:** 26
- **Total Test Cases:** 395 (collected)
- **Test Execution:** 389 passed, 4 failed, 3 skipped
- **Overall Code Coverage:** **71%**
  - **Statements:** 12,415 total, 3,295 missing
  - **Branches:** 3,172 total, 307 missing
  - **Lines:** 9,120 covered out of 12,415

## Test Files Breakdown

### Core Functionality Tests

#### 1. **test_template_layout.py** (43 tests)
- TemplateLayout class initialization and methods
- Template class lifecycle and interactions
- Field block parsing and validation
- Output column management
- Custom labels and bubble field types
- Preprocessor integration
- Template serialization

#### 2. **test_template_file_runner.py** (12 tests)
- Multi-pass detection and interpretation architecture
- TemplateFileRunner initialization
- File-level and field-level detection/interpretation
- Aggregate management across files and directories
- Export metrics collection

#### 3. **test_evaluation_config_for_set.py** (27 tests)
- EvaluationConfigForSet initialization
- Answer matching and scoring
- Marking scheme validation
- Format string validation
- OMR response validation
- Evaluation metadata generation

#### 4. **test_entry.py** (10 tests)
- Entry point functionality
- Directory-wise processing
- Config summary printing
- Template and evaluation config loading

#### 5. **test_exceptions.py** (60 tests - parameterized)
- Custom exception hierarchy
- Exception context preservation
- Error message formatting
- Exception catching behavior
- All exception types (Input, Processing, Validation, etc.)

### Detection & Processing Tests

#### 6. **test_bubbles_threshold_detection.py** (18 tests)
- Bubble field detection
- Threshold calculation
- Scan quality assessment
- Bubble mean value operations
- Multi-marking detection

#### 7. **test_shift_detection.py** (13 tests)
- Shift detection processor
- ML alignment validation
- Confidence reduction calculation
- Bubble and field response comparison

#### 8. **test_stn_integration.py** (Skipped - requires ML dependencies)
- Spatial Transformer Network (STN) modules
- STN forward pass and transformations
- Translation-only STN variants
- STN utility functions
- Image transformation application

### Utility & Helper Tests

#### 9. **test_file.py** (7 tests)
- File utilities (load_json, PathUtils)
- Output directory creation
- Path manipulation utilities

#### 10. **test_file_pattern_resolver.py** (16 tests)
- File pattern resolution
- Collision handling strategies
- Field substitution in patterns
- Path sanitization

#### 11. **test_drawing.py** (10 tests)
- Drawing utilities (boxes, lines, text, polygons)
- Contour drawing
- Text positioning and centering

#### 12. **test_image_utils.py** (6 tests)
- Image loading (grayscale, color)
- Image error handling
- Image mode conversion

#### 13. **test_geometry.py** (17 tests)
- Bounding box calculations
- Euclidean distance
- Vector magnitude
- Geometric consistency checks

#### 14. **test_math.py** (Covered in other tests)
- Math utilities and color conversion

#### 15. **test_csv.py** (4 tests)
- Thread-safe CSV operations
- CSV append functionality

#### 16. **test_logger.py** (11 tests)
- Logger initialization
- Log level management
- Message formatting
- Multiple log types

### Validation & Configuration Tests

#### 17. **test_config_validations.py** (2 tests)
- Config validation rules
- Parallel workers validation

#### 18. **test_template_validations.py** (14 tests)
- Template schema validation
- Field block validation
- Output column validation
- Overflow detection

#### 19. **test_file_grouping_validation.py** (14 tests)
- File grouping config validation
- Pattern validation
- Field availability checks
- Error message quality

### Integration & Sample Tests

#### 20. **test_all_samples.py** (20 tests)
- End-to-end sample processing
- Community-contributed samples
- Various feature scenarios
- Answer key processing

#### 21. **test_edge_cases.py** (3 tests)
- Low dimension error cases
- Different bubble dimensions
- Configuration edge cases

### Advanced Features Tests

#### 22. **test_augmentation.py** (24 tests)
- Data augmentation types
- Augmentation ordering
- Field block shifting
- Normalization
- Photometric augmentations

#### 23. **test_auto_training.py** (11 tests)
- ML training CLI arguments
- Confidence score calculation
- Hybrid detection strategy
- Training data collection
- YOLO annotation export

#### 24. **test_file_organizer.py** (14 tests)
- File organization processor
- Rule matching and priority
- Collision strategies
- QR code sorting

#### 25. **test_processors.py** (8 tests)
- Processing pipeline execution
- Processor management
- Processing context
- ReadOMR processor flow
- Alignment processor

#### 26. **test_workflow_visualization.py** (25 tests)
- Workflow tracking
- Processor state management
- Workflow graph construction
- HTML export
- JSON serialization
- Image encoding/decoding

### Serialization & Data Tests

#### 27. **test_dataclass_serialization.py** (8 tests)
- Dataclass to dict conversion
- Nested dataclass serialization
- Enum serialization
- Path serialization

## Coverage by Module

### High Coverage (90%+)
- `src/utils/constants.py`: 100%
- `src/utils/csv.py`: 100%
- `src/utils/geometry.py`: 100%
- `src/utils/env.py`: 100%
- `src/utils/file.py`: 95%
- `src/utils/logger.py`: 95%
- `src/utils/parsing.py`: 95%

### Medium Coverage (70-89%)
- `src/utils/file_pattern_resolver.py`: 88%
- `src/utils/math.py`: 88%
- `src/utils/image.py`: 89%
- `src/utils/drawing.py`: 79%
- `src/utils/serialization.py`: 87%
- `src/utils/shapes.py`: 87%

### Low Coverage (<70%)
- `src/utils/image_warp.py`: 22%
- `src/utils/interaction.py`: 30%
- `src/utils/stats.py`: 65%
- `src/utils/validations.py`: 57%
- `src/utils/visualization_runner.py`: 0%

## Test Execution Status

### Passing Tests: 389 ✅
- All core functionality tests passing
- All utility tests passing
- Most integration tests passing

### Failing Tests: 4 ❌
1. `test_template_validations.py::test_invalid_bubble_field_type`
2. `test_template_validations.py::test_empty_template`
3. `test_template_validations.py::test_overflow_labels`
4. `test_all_samples.py::test_run_answer_key_using_image`

### Skipped Tests: 3 ⏭️
- All tests in `test_stn_integration.py` (requires ML dependencies: torch)

## Coverage Gaps

### Critical Areas Needing More Tests
1. **Image Warping** (22% coverage)
   - `src/utils/image_warp.py` - Only 22% covered
   - Needs tests for warping transformations

2. **User Interaction** (30% coverage)
   - `src/utils/interaction.py` - Only 30% covered
   - Needs tests for user interaction flows

3. **Visualization Runner** (0% coverage)
   - `src/utils/visualization_runner.py` - No coverage
   - Needs comprehensive test suite

4. **Validations** (57% coverage)
   - `src/utils/validations.py` - 57% covered
   - Schema validation edge cases need more tests

5. **Stats** (65% coverage)
   - `src/utils/stats.py` - 65% covered
   - Statistics tracking needs more tests

## Test Quality Improvements

### Recent Improvements Made
1. ✅ Parameterized exception tests (reduced duplication)
2. ✅ Created shared fixtures (`conftest.py`)
3. ✅ Removed redundant tests
4. ✅ Fixed duplicate test names
5. ✅ Added edge case tests
6. ✅ Configured parallel test execution

### Recommendations
1. **Increase coverage for low-coverage modules**
   - Focus on `image_warp.py`, `interaction.py`, `visualization_runner.py`

2. **Fix failing tests**
   - Investigate and fix 4 failing tests in template validations and samples

3. **Add integration tests**
   - More end-to-end scenarios
   - Multi-file processing workflows

4. **Performance tests**
   - Add benchmarks for critical paths
   - Test parallel processing efficiency

## Test Execution Commands

```bash
# Run all tests
uv run pytest src/tests

# Run with coverage
uv run pytest src/tests --cov=src --cov-report=html

# Run specific test file
uv run pytest src/tests/test_template_layout.py

# Run in parallel (auto-detect workers)
uv run pytest src/tests -n auto

# Run with verbose output
uv run pytest src/tests -v

# Run only failing tests
uv run pytest src/tests --lf
```

## Notes

- Tests are configured to run in parallel using `pytest-xdist`
- Coverage reports are generated in `htmlcov/` directory
- ML-dependent tests are automatically skipped when torch is not available
- Test fixtures are shared via `conftest.py` to reduce duplication

