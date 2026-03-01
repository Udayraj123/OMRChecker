# OMRChecker TypeScript Tests

This directory contains comprehensive tests for the OMRChecker TypeScript implementation.

## Test Structure

```
tests/
├── unit/          # Unit tests for utilities and pure functions
├── integration/   # Integration tests for processors and pipelines
├── schema/        # Schema validation tests (Zod)
├── memory/        # Memory leak tests (OpenCV.js cv.Mat)
├── parity/        # Python-TypeScript behavior comparison tests
├── setup.ts       # Test environment setup
└── README.md      # This file
```

## Running Tests

### Run all tests
```bash
npm test
```

### Run tests in watch mode (interactive)
```bash
npm test
```

### Run tests once (CI mode)
```bash
npm run test:run
```

### Run tests with UI
```bash
npm run test:ui
```

### Run tests with coverage
```bash
npm run coverage
```

## Coverage Targets

- **Unit tests**: ≥90% line coverage
- **Schema tests**: 100% field coverage
- **Integration tests**: ≥80% coverage
- **Memory tests**: 100% cv.Mat usage covered

## Test Patterns

### Unit Test Example

```typescript
import { describe, it, expect } from 'vitest';
import { MathUtils } from '../../src/utils/math';

describe('MathUtils', () => {
  describe('distance', () => {
    it('should calculate Euclidean distance', () => {
      const result = MathUtils.distance([0, 0], [3, 4]);
      expect(result).toBe(5);
    });

    it('should handle negative coordinates', () => {
      const result = MathUtils.distance([-1, -1], [2, 3]);
      expect(result).toBeCloseTo(5, 1);
    });
  });
});
```

### Memory Leak Test Example

```typescript
import { describe, it, expect } from 'vitest';
import { getMatCount, skipIfNoMatCount } from '../setup';

describe('DrawingUtils Memory', () => {
  it('should not leak cv.Mat', skipIfNoMatCount(() => {
    const initialCount = getMatCount();
    
    // ... perform operations with cv.Mat
    
    const finalCount = getMatCount();
    expect(finalCount).toBe(initialCount);
  }));
});
```

### Schema Validation Test Example

```typescript
import { describe, it, expect } from 'vitest';
import { TemplateConfigSchema } from '../../src/schemas/models/template';

describe('TemplateConfig Schema', () => {
  it('should validate correct config', () => {
    const validConfig = { /* ... */ };
    expect(() => TemplateConfigSchema.parse(validConfig)).not.toThrow();
  });

  it('should reject invalid config', () => {
    const invalidConfig = { /* ... */ };
    expect(() => TemplateConfigSchema.parse(invalidConfig)).toThrow();
  });
});
```

## Test Environment

- **Framework**: Vitest
- **Environment**: jsdom (for browser APIs)
- **Setup**: OpenCV.js loading handled in setup.ts
- **Memory tracking**: cv.matCount wrapper for leak detection

## Notes

### OpenCV.js Testing

OpenCV.js requires a browser environment. For unit tests that don't need actual OpenCV functionality, operations can be mocked. For integration tests, the setup.ts file handles initialization gracefully, skipping tests if OpenCV is unavailable.

### Memory Leak Detection

The `getMatCount()` helper function tracks cv.Mat allocations. If cv.matCount is not available (e.g., in Node environment), memory tests are automatically skipped with a warning.

### Coverage Thresholds

Coverage thresholds are enforced in CI. If coverage falls below targets:
- Document why (e.g., browser-only code, error handling edge cases)
- Add TODO for future improvement
- Get approval for exception

## CI Integration

Tests run automatically on:
- Every commit (pre-commit hook)
- Every push (pre-push hook)
- Pull requests (GitHub Actions)

## Documentation

For detailed testing strategy, see:
- `/docs/TESTING_STRATEGY.md` - Comprehensive testing approach
- `/docs/TESTING_TASKS_BREAKDOWN.md` - Task breakdown with subagent assignments

## Status

**Current Coverage**: 0% (Phase 1 files migrated, tests pending)  
**Next Milestone**: Complete Phase 1 unit tests (90%+ coverage)
