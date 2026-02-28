# TypeScript Migration Testing Strategy

**Date**: 2026-02-28  
**Status**: DRAFT - Ready for implementation  
**Prerequisite for**: Phase 2+ migration

---

## Problem Statement

**Current Gap**: Phase 1 migrated 2,334 lines of TypeScript with 0 tests.

**Risk**: Cannot verify that TypeScript behaves identically to Python.

**Impact**:
- Regressions could go undetected
- Refactoring becomes dangerous
- Browser deployment could fail silently
- Memory leaks (cv.Mat) undetected

**Solution**: Comprehensive testing strategy before Phase 2.

---

## Testing Levels

### 1. Unit Tests (Utilities & Pure Functions)

**Scope**: math.ts, geometry.ts, stats.ts, checksum.ts

**Framework**: Vitest

**Coverage Target**: 90%+

**Example Test** (`math.test.ts`):
```typescript
import { describe, it, expect } from 'vitest';
import { MathUtils, Point } from '../utils/math';

describe('MathUtils', () => {
  describe('distance', () => {
    it('should calculate Euclidean distance between two points', () => {
      const p1: Point = [0, 0];
      const p2: Point = [3, 4];
      expect(MathUtils.distance(p1, p2)).toBe(5);
    });

    it('should handle negative coordinates', () => {
      const p1: Point = [-1, -1];
      const p2: Point = [2, 3];
      expect(MathUtils.distance(p1, p2)).toBeCloseTo(5, 1);
    });

    it('should return 0 for same point', () => {
      const p: Point = [5, 5];
      expect(MathUtils.distance(p, p)).toBe(0);
    });
  });

  describe('orderFourPoints', () => {
    it('should order points clockwise from top-left', () => {
      const points: Point[] = [[100, 100], [0, 0], [100, 0], [0, 100]];
      const [ordered, _] = MathUtils.orderFourPoints(points);
      expect(ordered[0]).toEqual([0, 0]); // top-left
      expect(ordered[1]).toEqual([100, 0]); // top-right
      expect(ordered[2]).toEqual([100, 100]); // bottom-right
      expect(ordered[3]).toEqual([0, 100]); // bottom-left
    });
  });
});
```

**Approach**:
1. Port existing Python tests from `src/tests/`
2. Generate test skeletons using `.agents/migration-toolkit/3-generate-tests.js`
3. Add edge cases discovered during migration

### 2. Schema Validation Tests (Zod Schemas)

**Scope**: template.ts, config.ts, evaluation.ts

**Framework**: Vitest + Zod

**Coverage Target**: 100% (all schema fields)

**Example Test** (`template.test.ts`):
```typescript
import { describe, it, expect } from 'vitest';
import { TemplateConfigSchema, createTemplateConfig } from '../schemas/models/template';

describe('TemplateConfig Schema', () => {
  it('should validate correct template config', () => {
    const validConfig = {
      bubbleDimensions: [10, 10],
      templateDimensions: [1200, 1600],
      alignment: { margins: { top: 0, bottom: 0, left: 0, right: 0 } }
    };
    expect(() => TemplateConfigSchema.parse(validConfig)).not.toThrow();
  });

  it('should reject invalid bubble dimensions', () => {
    const invalidConfig = {
      bubbleDimensions: [10], // Should be length 2
      templateDimensions: [1200, 1600]
    };
    expect(() => TemplateConfigSchema.parse(invalidConfig)).toThrow();
  });

  it('should apply defaults for missing optional fields', () => {
    const minimalConfig = {};
    const result = createTemplateConfig(minimalConfig);
    expect(result.bubbleDimensions).toEqual([10, 10]);
    expect(result.templateDimensions).toEqual([1200, 1600]);
  });
});
```

**Approach**:
1. Test each schema with valid/invalid inputs
2. Verify defaults are applied correctly
3. Test nested schema validation
4. Snapshot test for serialization/deserialization

### 3. Integration Tests (Processors)

**Scope**: base.ts, Pipeline.ts, coordinator.ts

**Framework**: Vitest + OpenCV.js (mocked initially)

**Coverage Target**: 80%+

**Example Test** (`Pipeline.test.ts`):
```typescript
import { describe, it, expect, beforeEach } from 'vitest';
import { ProcessingPipeline } from '../processors/Pipeline';
import { createProcessingContext } from '../processors/base';

describe('ProcessingPipeline', () => {
  let pipeline: ProcessingPipeline;
  let context: ProcessingContext;

  beforeEach(() => {
    pipeline = new ProcessingPipeline();
    context = createProcessingContext({
      filePath: 'test.jpg',
      grayImage: null,
      template: mockTemplate
    });
  });

  it('should execute processors in sequence', async () => {
    const executionOrder: string[] = [];
    
    const processor1 = {
      getName: () => 'Processor1',
      process: async (ctx) => {
        executionOrder.push('1');
        return ctx;
      }
    };
    
    const processor2 = {
      getName: () => 'Processor2',
      process: async (ctx) => {
        executionOrder.push('2');
        return ctx;
      }
    };

    pipeline.addProcessor(processor1);
    pipeline.addProcessor(processor2);

    await pipeline.process(context);

    expect(executionOrder).toEqual(['1', '2']);
  });

  it('should handle processor errors gracefully', async () => {
    const failingProcessor = {
      getName: () => 'FailingProcessor',
      process: async () => {
        throw new Error('Processing failed');
      }
    };

    pipeline.addProcessor(failingProcessor);

    await expect(pipeline.process(context)).rejects.toThrow('Processing failed');
  });
});
```

### 4. OpenCV.js Memory Leak Tests

**Scope**: drawing.ts, any file using cv.Mat

**Framework**: Custom test harness

**Coverage Target**: All cv.Mat usage

**Example Test** (`drawing.memory.test.ts`):
```typescript
import { describe, it, expect } from 'vitest';
import cv from '@techstark/opencv-js';
import { DrawingUtils } from '../utils/drawing';

describe('DrawingUtils Memory Management', () => {
  it('should not leak memory in drawMatches', () => {
    const img1 = cv.Mat.zeros(100, 100, cv.CV_8UC3);
    const img2 = cv.Mat.zeros(100, 100, cv.CV_8UC3);
    const points = [[10, 10], [20, 20]];

    const initialMatCount = cv.matCount(); // Custom utility

    const result = DrawingUtils.drawMatches(img1, points, img2, points);
    result.delete();
    img1.delete();
    img2.delete();

    const finalMatCount = cv.matCount();
    expect(finalMatCount).toBe(initialMatCount);
  });

  it('should clean up temporary Mats in drawContour', () => {
    const img = cv.Mat.zeros(100, 100, cv.CV_8UC3);
    const contour = [[10, 10], [20, 10], [20, 20], [10, 20]];

    const initialMatCount = cv.matCount();

    DrawingUtils.drawContour(img, contour);

    const finalMatCount = cv.matCount();
    // Only img should remain
    expect(finalMatCount - initialMatCount).toBe(0);

    img.delete();
  });
});
```

### 5. Snapshot Tests (Error Hierarchies)

**Scope**: exceptions.ts

**Framework**: Vitest snapshots

**Coverage Target**: All exception types

**Example Test** (`exceptions.test.ts`):
```typescript
import { describe, it, expect } from 'vitest';
import { 
  OMRCheckerError, 
  ImageProcessingError, 
  BubbleDetectionError 
} from '../utils/exceptions';

describe('Exception Hierarchy', () => {
  it('should create OMRCheckerError with context', () => {
    const error = new OMRCheckerError('Test error', { key: 'value' });
    expect(error.message).toBe('Test error');
    expect(error.context).toEqual({ key: 'value' });
    expect(error.toString()).toMatchSnapshot();
  });

  it('should support instanceof checks', () => {
    const error = new ImageProcessingError('Processing failed', { operation: 'threshold' });
    expect(error instanceof OMRCheckerError).toBe(true);
    expect(error instanceof ImageProcessingError).toBe(true);
    expect(error instanceof BubbleDetectionError).toBe(false);
  });

  it('should format context in toString', () => {
    const error = new BubbleDetectionError('test.jpg', 'Q1', 'No bubbles found');
    expect(error.toString()).toContain('test.jpg');
    expect(error.toString()).toContain('Q1');
    expect(error.toString()).toMatchSnapshot();
  });
});
```

### 6. Behavior Comparison Tests (Python ↔ TypeScript)

**Scope**: All migrated files

**Framework**: Custom test runner

**Coverage Target**: Representative samples

**Approach**:
```typescript
// Python test data generator (run once)
// python scripts/generate_test_fixtures.py

// TypeScript test
import { describe, it, expect } from 'vitest';
import { MathUtils } from '../utils/math';
import pythonOutputs from '../fixtures/math_outputs.json';

describe('Python Behavior Parity', () => {
  pythonOutputs.distance_tests.forEach((test) => {
    it(`distance(${JSON.stringify(test.input)}) matches Python`, () => {
      const result = MathUtils.distance(test.input.p1, test.input.p2);
      expect(result).toBeCloseTo(test.output, 6);
    });
  });
});
```

**Test Data Generation Script** (`scripts/generate_test_fixtures.py`):
```python
import json
from src.utils.math import MathUtils

test_cases = {
    "distance_tests": [
        {
            "input": {"p1": [0, 0], "p2": [3, 4]},
            "output": MathUtils.distance([0, 0], [3, 4])
        },
        # ... more test cases
    ]
}

with open('omrchecker-js/packages/core/fixtures/math_outputs.json', 'w') as f:
    json.dump(test_cases, f, indent=2)
```

---

## Test Organization

### Directory Structure
```
omrchecker-js/
├── packages/
│   └── core/
│       ├── src/
│       │   ├── utils/
│       │   ├── processors/
│       │   ├── schemas/
│       │   └── ...
│       ├── tests/
│       │   ├── unit/
│       │   │   ├── math.test.ts
│       │   │   ├── geometry.test.ts
│       │   │   ├── stats.test.ts
│       │   │   └── checksum.test.ts
│       │   ├── integration/
│       │   │   ├── Pipeline.test.ts
│       │   │   └── coordinator.test.ts
│       │   ├── schema/
│       │   │   ├── template.test.ts
│       │   │   ├── config.test.ts
│       │   │   └── evaluation.test.ts
│       │   ├── memory/
│       │   │   └── drawing.memory.test.ts
│       │   └── parity/
│       │       └── python-comparison.test.ts
│       ├── fixtures/
│       │   ├── math_outputs.json
│       │   ├── sample_template.json
│       │   └── sample_image.jpg
│       └── vitest.config.ts
```

### Vitest Configuration

**File**: `omrchecker-js/packages/core/vitest.config.ts`
```typescript
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'jsdom', // For browser APIs
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'tests/',
        '**/*.test.ts'
      ],
      thresholds: {
        lines: 90,
        functions: 90,
        branches: 85,
        statements: 90
      }
    },
    setupFiles: ['./tests/setup.ts']
  }
});
```

**Setup File**: `tests/setup.ts`
```typescript
import cv from '@techstark/opencv-js';

// Wait for OpenCV.js to load
beforeAll(async () => {
  await new Promise((resolve) => {
    cv.onRuntimeInitialized = resolve;
  });
});

// Track cv.Mat count for memory leak detection
let matCountSupported = false;
try {
  matCountSupported = typeof cv.matCount === 'function';
} catch (e) {
  console.warn('cv.matCount not available - memory leak tests will be skipped');
}

global.cv = cv;
global.matCountSupported = matCountSupported;
```

---

## Implementation Plan

### Phase 1 Tests (Utilities) - Week 1
- [ ] **Day 1**: Set up Vitest, create test structure
- [ ] **Day 2**: math.ts tests (19 methods)
- [ ] **Day 3**: geometry.ts, stats.ts tests
- [ ] **Day 4**: checksum.ts tests, verify 90% coverage
- [ ] **Day 5**: Buffer day + documentation

**Success Criteria**: 90%+ coverage, all tests passing

### Phase 2 Tests (Schemas) - Week 2
- [ ] **Day 1**: template.ts schema tests
- [ ] **Day 2**: config.ts schema tests
- [ ] **Day 3**: evaluation.ts schema tests
- [ ] **Day 4**: Snapshot tests for serialization
- [ ] **Day 5**: Buffer day + documentation

**Success Criteria**: 100% schema field coverage

### Phase 3 Tests (Processors) - Week 3
- [ ] **Day 1**: base.ts + ProcessingContext tests
- [ ] **Day 2**: Pipeline.ts integration tests
- [ ] **Day 3**: coordinator.ts tests
- [ ] **Day 4**: End-to-end processor chain test
- [ ] **Day 5**: Buffer day + documentation

**Success Criteria**: 80%+ coverage, async handling verified

### Phase 4 Tests (Memory & Parity) - Week 4
- [ ] **Day 1**: OpenCV.js memory leak detection utilities
- [ ] **Day 2**: drawing.ts memory tests
- [ ] **Day 3**: Generate Python test fixtures
- [ ] **Day 4**: Python-TypeScript parity tests
- [ ] **Day 5**: CI integration + documentation

**Success Criteria**: 0 memory leaks, parity verified

---

## Test Templates

### Unit Test Template
```typescript
import { describe, it, expect, beforeEach, afterEach } from 'vitest';

describe('ComponentName', () => {
  beforeEach(() => {
    // Setup
  });

  afterEach(() => {
    // Cleanup
  });

  describe('methodName', () => {
    it('should handle normal case', () => {
      // Arrange
      const input = ...;
      const expected = ...;

      // Act
      const result = ComponentName.methodName(input);

      // Assert
      expect(result).toBe(expected);
    });

    it('should handle edge case', () => {
      // Test edge case
    });

    it('should throw on invalid input', () => {
      expect(() => ComponentName.methodName(null)).toThrow();
    });
  });
});
```

### Memory Leak Test Template
```typescript
import { describe, it, expect } from 'vitest';
import cv from '@techstark/opencv-js';

describe('ComponentName Memory Management', () => {
  it('should not leak cv.Mat in methodName', () => {
    if (!global.matCountSupported) {
      console.warn('Skipping memory test - cv.matCount not available');
      return;
    }

    const initialCount = cv.matCount();

    // Create resources
    const mat = cv.Mat.zeros(100, 100, cv.CV_8UC3);

    // Execute function
    ComponentName.methodName(mat);

    // Cleanup
    mat.delete();

    const finalCount = cv.matCount();
    expect(finalCount).toBe(initialCount);
  });
});
```

---

## CI/CD Integration

### GitHub Actions Workflow

**File**: `.github/workflows/ts-tests.yml`
```yaml
name: TypeScript Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install dependencies
        run: cd omrchecker-js && npm install
      - name: Run tests
        run: cd omrchecker-js && npm test
      - name: Check coverage
        run: cd omrchecker-js && npm run coverage
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./omrchecker-js/coverage/coverage-final.json
```

### Pre-commit Hook

Add to `.pre-commit-config.yaml`:
```yaml
  - repo: local
    hooks:
      - id: ts-tests
        name: TypeScript Tests
        entry: bash -c 'cd omrchecker-js && npm test'
        language: system
        pass_filenames: false
        always_run: true
```

---

## Success Metrics

### Coverage Targets
- **Unit tests**: ≥90% line coverage
- **Schema tests**: 100% field coverage
- **Integration tests**: ≥80% coverage
- **Memory tests**: 100% cv.Mat usage covered

### Quality Gates
- [ ] All tests passing
- [ ] No memory leaks detected
- [ ] Python-TypeScript parity verified for key functions
- [ ] Test execution time <30 seconds
- [ ] CI passing on all PRs

### Documentation
- [ ] Test README with running instructions
- [ ] Commented test examples
- [ ] Coverage reports generated
- [ ] Testing strategy in AGENTS.md

---

## Migration Workflow Integration

### Updated Migration Checklist (Per File)

1. ✅ Claim issue (`bd update <id> --status in_progress`)
2. ✅ Generate skeleton (`generate_ts_suggestions.py`)
3. ✅ Enhance implementation (manual)
4. ✅ Validate structure (`validate_ts_migration.py`)
5. **🆕 Write tests** (`<file>.test.ts` with 90%+ coverage)
6. **🆕 Run tests** (`npm test <file>.test.ts`)
7. **🆕 Check memory** (if uses cv.Mat)
8. ✅ Commit (`git commit` with issue ref)
9. ✅ Close issue (`bd close <id>`)

**Time estimate per file**: +5-10 minutes for testing

---

## Next Actions

### Immediate (Before Phase 2)
1. Create Vitest setup (`vitest.config.ts`, `tests/setup.ts`)
2. Write test for ONE Phase 1 file (math.ts) as proof of concept
3. Document any gaps or challenges
4. Update migration skill with testing step

### Short-term (Week 1-2)
1. Complete all Phase 1 file tests
2. Achieve 90%+ coverage for Phase 1 files
3. Generate Python test fixtures
4. Verify parity for key functions

### Long-term (Week 3-4)
1. Create test generation automation
2. Set up CI/CD pipeline
3. Add pre-commit hooks
4. Create testing guide for contributors

---

## Open Questions

1. **OpenCV.js Loading**: How to handle async OpenCV.js initialization in tests?
   - **Answer**: Use `beforeAll` with promise wait
   
2. **Test Data**: Should we commit large binary fixtures (images)?
   - **Recommendation**: Use small synthetic images, document how to generate larger ones

3. **Memory Leak Detection**: What if cv.matCount() not available?
   - **Recommendation**: Skip memory tests gracefully, log warning

4. **Python Fixture Generation**: Automate or manual?
   - **Recommendation**: Script to generate, commit fixtures to git

---

## Conclusion

Testing is **mandatory** before Phase 2. The strategy provides:
- ✅ Comprehensive coverage (unit, integration, parity)
- ✅ Memory leak detection
- ✅ CI/CD integration
- ✅ Clear implementation plan

**Status**: Ready to implement  
**Estimated effort**: 4 weeks parallel to Phase 2 migration  
**Blocking**: Phase 2 can start, but files must have tests before marked complete

---

**Next**: Implement Vitest setup and create first test for math.ts as proof of concept.
