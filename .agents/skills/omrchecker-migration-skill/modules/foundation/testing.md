# Testing Patterns

**Module**: Foundation
**Python Reference**: `src/tests/conftest.py`, `pytest.ini`, `src/tests/__fixtures__/pytest_image_snapshot.py`
**Last Updated**: 2026-02-20

---

## Overview

OMRChecker uses **pytest** with custom fixtures for image snapshot testing, parallel execution, and comprehensive test coverage (~70%).

**Key Features**:
- Image snapshot testing (visual regression)
- Parallel test execution (pytest-xdist)
- Shared fixtures via conftest.py
- Custom markers for test organization
- Coverage reporting (pytest-cov)
- Mock-based unit testing

---

## Pytest Configuration

**Code Reference**: `pytest.ini`

### Key Settings
```ini
minversion = 7.0
addopts =
    --disable-warnings
    --cov-branch
    --cov-report=html
    --cov-report=term-missing
    --import-mode=importlib
    --dist=loadscope  # Group tests by module for shared resources

norecursedirs = .* __pycache__ htmlcov
pythonpath = "."
python_files = test_*.py
testpaths = src/tests
```

### Custom Markers
- `serial`: Tests that must run sequentially (shared resources)
- `integration`: Integration tests (longer duration)
- `file_io`: Tests with file I/O on shared paths
- `sample_*`: Sample-specific test markers (14+ samples)

**Parallel Execution**: `pytest -n auto` (auto-detect CPUs)

---

## Shared Fixtures

**Code Reference**: `src/tests/conftest.py`

### mock_template
```python
@pytest.fixture
def mock_template():
    template = Mock()
    template.tuning_config = CONFIG_DEFAULTS
    template.all_fields = []
    return template
```

### minimal_template_json
```python
@pytest.fixture
def minimal_template_json():
    return {
        "templateDimensions": [1000, 800],
        "bubbleDimensions": [20, 20],
        "fieldBlocks": { ... },
        "preProcessors": [],
        # ...
    }
```

### minimal_evaluation_json
```python
@pytest.fixture
def minimal_evaluation_json():
    return {
        "source_type": "local",
        "options": {
            "questions_in_order": ["q1", "q2"],
            "answers_in_order": ["A", "B"],
        },
        "marking_schemes": { "DEFAULT": { ... } },
        # ...
    }
```

### minimal_args
```python
@pytest.fixture
def minimal_args():
    return {
        "debug": False,
        "outputMode": "default",
        "setLayout": False,
        # ...
    }
```

### temp_template_path
```python
@pytest.fixture
def temp_template_path(tmp_path):
    return tmp_path / "template.json"
```

---

## Image Snapshot Testing

**Code Reference**: `src/tests/__fixtures__/pytest_image_snapshot.py`

### Purpose
Visual regression testing - compare generated images with baseline snapshots

### CLI Options
```bash
--image-snapshot-update  # Update baseline snapshots
--show-images-on-fail    # Display diff on failure
```

### Usage Pattern
```python
def test_bubble_detection(image_snapshot):
    output_image = detect_bubbles(input_image)
    image_snapshot(output_image, snapshot_path="snapshots/bubble_detection.png")
```

### Image Comparison
```python
def image_diff(image1, image2):
    difference = np.absolute(image1 - image2)
    return difference.any()
```

**Raises AssertionError** if images differ

**Snapshot Directory**: `src/tests/__image_snapshots__/`

---

## Test Organization

### Directory Structure
```
src/tests/
├── conftest.py                 # Shared fixtures
├── __fixtures__/
│   ├── pytest_image_snapshot.py
│   └── run_sample.py
├── __tests__/                  # Unit tests for src/
├── processors/                 # Processor-specific tests
│   ├── alignment/__tests__/
│   ├── detection/__tests__/
│   ├── evaluation/__tests__/
│   └── ...
├── schemas/__tests__/
└── utils/__tests__/
```

### Test File Naming
- `test_*.py` - Test files
- `*/__tests__/` - Test directory per module

---

## Common Test Patterns

### 1. Mock-Based Unit Tests
```python
def test_template_init(mock_template):
    assert mock_template.tuning_config == CONFIG_DEFAULTS
```

### 2. Fixture-Based Integration Tests
```python
def test_template_loading(minimal_template_json, temp_template_path):
    temp_template_path.write_text(json.dumps(minimal_template_json))
    template = Template(temp_template_path, CONFIG_DEFAULTS, {})
    assert template.field_blocks is not None
```

### 3. Parametrized Tests
```python
@pytest.mark.parametrize("input,expected", [
    ([1, 2, 3], 6),
    ([0, 0, 0], 0),
])
def test_sum(input, expected):
    assert sum(input) == expected
```

### 4. Exception Testing
```python
def test_invalid_template():
    with pytest.raises(TemplateValidationError):
        Template(invalid_path, config, args)
```

---

## Browser Migration Notes

### Jest/Vitest Configuration
```javascript
// vitest.config.js
export default {
  test: {
    globals: true,
    environment: 'jsdom',  // Browser environment
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
    },
  },
};
```

### Image Snapshot Testing
Use **jest-image-snapshot** or **pixelmatch**:
```javascript
import { toMatchImageSnapshot } from 'jest-image-snapshot';
expect.extend({ toMatchImageSnapshot });

test('bubble detection', () => {
  const outputImage = detectBubbles(inputImage);
  expect(outputImage).toMatchImageSnapshot();
});
```

### Mock Fixtures
```javascript
// fixtures/mockTemplate.js
export const mockTemplate = {
  tuningConfig: CONFIG_DEFAULTS,
  allFields: [],
  allFieldDetectionTypes: [],
};
```

### Parallel Testing
```bash
# Vitest auto-parallelizes by default
vitest --threads --minThreads=1 --maxThreads=8
```

---

## Summary

**Framework**: pytest 7.0+
**Coverage**: ~70% (branch coverage)
**Parallel**: pytest-xdist (auto CPU detection)
**Image Testing**: Custom snapshot fixture
**Fixtures**: 5+ shared fixtures in conftest.py
**Markers**: 14+ custom markers for organization

**Browser Migration**: Use Jest/Vitest with jest-image-snapshot, maintain fixture pattern
