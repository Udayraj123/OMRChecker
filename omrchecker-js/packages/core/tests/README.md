# OMRChecker TypeScript Tests

This directory contains comprehensive tests for the OMRChecker TypeScript implementation.

## Test Structure

```
tests/
├── unit/          # Unit tests for utilities and pure functions
├── browser/       # Browser tests using Playwright (OpenCV.js, Web Crypto)
├── integration/   # Integration tests for processors and pipelines
├── schema/        # Schema validation tests (Zod)
├── parity/        # Python-TypeScript behavior comparison tests
├── setup.ts       # Test environment setup (jsdom)
└── README.md      # This file
```

## Running Tests

### Run all tests (unit + browser)
```bash
npm run test:all
```

### Run unit tests only (jsdom)
```bash
npm run test:unit
```

### Run browser tests only (Playwright)
```bash
npm run test:browser
```

### Run tests in watch mode
```bash
# Unit tests watch
npm run test:watch

# Browser tests watch
npm run test:watch:browser
```

### Run tests with UI
```bash
npm run test:ui
```

### Run tests with coverage
```bash
# All tests with coverage
npm run coverage:all

# Unit tests only
npm run coverage:unit

# Browser tests only
npm run coverage:browser
```

## Coverage Targets

- **Unit tests**: ≥90% line coverage
- **Browser tests**: ≥85% line coverage (drawing, checksum, memory utilities)
- **Schema tests**: 100% field coverage
- **Integration tests**: ≥80% coverage

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

### Browser Test Example (Playwright)

```typescript
import { describe, it, expect, beforeAll } from 'vitest';
import { setupOpenCV, withMemoryTracking } from './browser-setup';

describe('DrawingUtils', () => {
  beforeAll(async () => {
    await setupOpenCV();
  });

  it('should draw rectangle without leaking memory', async () => {
    await withMemoryTracking(async (cv) => {
      const mat = cv.Mat.zeros(100, 100, cv.CV_8UC3);
      const rect = new cv.Rect(10, 10, 50, 50);
      const color = new cv.Scalar(255, 0, 0);
      
      cv.rectangle(mat, rect, color, 2);
      
      // Cleanup
      mat.delete();
      rect.delete();
      color.delete();
    });
  });
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

### Unit Tests
- **Framework**: Vitest
- **Environment**: jsdom (for browser APIs)
- **Setup**: Minimal test environment in setup.ts

### Browser Tests
- **Framework**: Vitest + Playwright
- **Environment**: Real Chromium browser
- **OpenCV.js**: Full OpenCV.js loaded from CDN (browser-setup.ts)
- **Memory tracking**: cv.getBuildInformation() and Mat tracking (memory-utils.ts)
- **Web Crypto**: Native SubtleCrypto API for checksums

## Notes

### OpenCV.js Testing

**Unit tests**: OpenCV functionality is mocked in jsdom since full OpenCV.js requires a real browser.

**Browser tests**: Full OpenCV.js is loaded from CDN using Playwright in real Chromium:
- `browser-setup.ts` handles asynchronous OpenCV loading
- `setupOpenCV()` must be called in `beforeAll()` hooks
- Tests wait for OpenCV.js to be fully initialized before running

### Memory Leak Detection

The `withMemoryTracking()` helper (in `memory-utils.ts`) automatically tracks cv.Mat allocations:
- Takes initial snapshot using `cv.getBuildInformation()`
- Runs test callback
- Takes final snapshot and compares
- Throws error if memory leaks detected
- Provides verbose logging with `getMemoryStats()`

### Coverage Thresholds

Coverage thresholds are enforced in CI. If coverage falls below targets:
- Document why (e.g., browser-only code, error handling edge cases)
- Add TODO for future improvement
- Get approval for exception

## CI Integration

Tests run automatically on:
- **Pre-commit hook**: Unit tests only (`npm run test:unit`) - fast feedback
- **Pre-push hook**: All tests (`npm run test:all`) - full validation
- **Pull requests**: GitHub Actions with coverage reporting

## Documentation

For detailed testing strategy, see:
- `/docs/TESTING_STRATEGY.md` - Comprehensive testing approach
- `/docs/TESTING_TASKS_BREAKDOWN.md` - Task breakdown with subagent assignments

## Status

**Browser Tests**: ✅ Complete (Phase 1-3)
- 10 OpenCV smoke tests (opencv-smoke.test.ts)
- 27 drawing tests (drawing.test.ts)
- 10 checksum tests (checksum.test.ts)
- 10 memory leak tests (memory-leak.test.ts)

**Total**: 57 browser tests across 4 files

**Next Milestone**: Unit tests for utilities (math.ts, array.ts, etc.)
