# Revised Testing Strategy - Hybrid Approach

**Date**: 2026-03-01  
**Status**: CRITICAL REVISION  
**Issue**: Current jsdom tests skip OpenCV.js - not testing actual functionality

---

## Problem Analysis

### What We're Currently Testing ✅
- **Pure JavaScript utilities** (no browser APIs)
  - ✅ math.ts - geometric calculations
  - ✅ geometry.ts - distance/magnitude functions
  - ✅ stats.ts - StatsByLabel, NumberAggregate classes
  - ✅ exceptions.ts - error hierarchy

### What We're NOT Testing ❌
- **OpenCV.js functionality** (requires real browser/WASM)
  - ❌ drawing.ts - cv.Mat, cv.rectangle, cv.drawContours
  - ❌ Image processing - actual cv operations
  - ❌ Memory leaks - cv.Mat lifecycle
- **Web Crypto API** (requires browser)
  - ❌ checksum.ts - crypto.subtle.digest
- **Browser APIs**
  - ❌ Blob.arrayBuffer()
  - ❌ Canvas operations (toBgr in math.ts)

### Current Setup Issues
```typescript
// tests/setup.ts - Line 18
console.log('Test setup: OpenCV.js initialization skipped (requires browser)');
```

**Reality**: We're testing JavaScript logic but skipping all browser-specific functionality!

---

## Revised Hybrid Testing Strategy

### Layer 1: Unit Tests (Node.js/Vitest) - FAST ⚡
**Purpose**: Test pure JavaScript logic  
**Environment**: jsdom (current setup)  
**Coverage Target**: 90%+

**What to test**:
- ✅ Pure functions (math, geometry, stats)
- ✅ Data structures and algorithms
- ✅ Error handling and validation
- ✅ Type checking and interfaces
- ✅ Business logic without browser APIs

**Files suitable for Layer 1**:
- `src/utils/math.ts` (except `toBgr` - line 299-317)
- `src/utils/geometry.ts`
- `src/utils/stats.ts`
- `src/utils/exceptions.ts`
- `src/schemas/**/*.ts` (Zod validation)
- `src/processors/base.ts` (interfaces)
- `src/processors/Pipeline.ts` (orchestration logic)

### Layer 2: Browser Integration Tests - COMPREHENSIVE 🌐
**Purpose**: Test actual OpenCV.js and browser APIs  
**Environment**: Real browser (Playwright/Puppeteer)  
**Coverage Target**: Critical paths

**Framework**: Playwright + Vitest browser mode

**What to test**:
- ✅ OpenCV.js operations (cv.Mat, image processing)
- ✅ Memory leak detection (cv.matCount())
- ✅ Web Crypto API (crypto.subtle)
- ✅ Canvas operations
- ✅ Blob/ArrayBuffer handling
- ✅ End-to-end image processing pipeline

**Files requiring Layer 2**:
- `src/utils/drawing.ts` (ALL cv operations)
- `src/utils/checksum.ts` (crypto.subtle)
- `src/processors/image/**/*.ts` (image operations)
- Any file using cv.Mat

### Layer 3: Visual Regression Tests - VALIDATION 👁️
**Purpose**: Verify visual output correctness  
**Environment**: Headless browser + snapshot comparison  
**Coverage Target**: Key visual features

**What to test**:
- ✅ Template alignment visualization
- ✅ Bubble detection marking
- ✅ Contour drawing
- ✅ Arrow/box annotations
- ✅ Text rendering

---

## Implementation Plan

### Phase 1: Fix Current Setup (Immediate)

**1. Split test suite**:
```
tests/
├── unit/              # Layer 1: Pure JS (current)
│   ├── math.test.ts
│   ├── geometry.test.ts
│   ├── stats.test.ts
│   └── exceptions.test.ts
├── browser/           # Layer 2: Browser APIs (NEW)
│   ├── drawing.test.ts
│   ├── checksum.test.ts
│   ├── opencv-memory.test.ts
│   └── image-processing.test.ts
└── visual/            # Layer 3: Visual regression (NEW)
    └── snapshot.test.ts
```

**2. Add Playwright for browser tests**:
```bash
npm install -D @playwright/test playwright
npm install -D @vitest/browser
```

**3. Update vitest.config.ts**:
```typescript
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // Fast unit tests (default)
    globals: true,
    environment: 'jsdom',
    include: ['tests/unit/**/*.test.ts'],
    
    // Browser tests (separate)
    browser: {
      enabled: false, // Enable via --browser flag
      name: 'chromium',
      provider: 'playwright',
      headless: true,
    },
  },
});
```

### Phase 2: Browser Test Setup

**Create browser-specific test config**: `vitest.config.browser.ts`
```typescript
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    browser: {
      enabled: true,
      name: 'chromium',
      provider: 'playwright',
      headless: true,
    },
    include: ['tests/browser/**/*.test.ts'],
    setupFiles: ['./tests/browser-setup.ts'],
  },
});
```

**Create browser setup**: `tests/browser-setup.ts`
```typescript
import { beforeAll, afterAll } from 'vitest';

// Wait for OpenCV.js to load in REAL browser
beforeAll(async () => {
  await new Promise<void>((resolve) => {
    if (typeof cv !== 'undefined' && cv.Mat) {
      console.log('OpenCV.js already loaded');
      resolve();
      return;
    }

    // Load OpenCV.js
    const script = document.createElement('script');
    script.src = 'https://docs.opencv.org/4.9.0/opencv.js';
    script.onload = () => {
      cv.onRuntimeInitialized = () => {
        console.log('OpenCV.js loaded successfully');
        resolve();
      };
    };
    document.body.appendChild(script);
  });
}, 30000);

// Track cv.Mat count for memory leak detection
let initialMatCount = 0;

beforeAll(() => {
  if (typeof cv !== 'undefined' && cv.Mat) {
    initialMatCount = cv.matCount?.() ?? 0;
    console.log(`Initial cv.Mat count: ${initialMatCount}`);
  }
});

afterAll(() => {
  if (typeof cv !== 'undefined' && cv.matCount) {
    const finalCount = cv.matCount();
    const leaked = finalCount - initialMatCount;
    if (leaked > 0) {
      console.warn(`⚠️ Memory leak detected: ${leaked} cv.Mat objects not deleted`);
    } else {
      console.log('✅ No memory leaks detected');
    }
  }
});
```

### Phase 3: Rewrite Browser-Dependent Tests

**Example: drawing.ts browser test**:
```typescript
import { describe, it, expect } from 'vitest';
import { DrawingUtils, CLR_GREEN } from '../../src/utils/drawing';

describe('DrawingUtils (Browser)', () => {
  it('should draw contour on cv.Mat', () => {
    // Create actual cv.Mat
    const img = new cv.Mat(100, 100, cv.CV_8UC3);
    const contour: Point[] = [[10, 10], [90, 10], [90, 90], [10, 90]];
    
    // This actually runs OpenCV.js!
    DrawingUtils.drawContour(img, contour, CLR_GREEN, 2);
    
    // Verify drawing occurred (check pixel values)
    const pixel = img.ucharPtr(10, 10);
    expect(pixel[0]).toBeGreaterThan(0); // Green channel
    
    img.delete();
  });
  
  it('should not leak memory when drawing boxes', () => {
    const initialCount = cv.matCount();
    
    const img = new cv.Mat(200, 200, cv.CV_8UC3);
    DrawingUtils.drawBox(img, [50, 50], [100, 100]);
    img.delete();
    
    const finalCount = cv.matCount();
    expect(finalCount).toBe(initialCount);
  });
});
```

**Example: checksum.ts browser test**:
```typescript
import { describe, it, expect } from 'vitest';
import { calculateFileChecksum } from '../../src/utils/checksum';

describe('Checksum (Browser)', () => {
  it('should calculate SHA-256 with real Web Crypto API', async () => {
    const data = new TextEncoder().encode('hello world').buffer;
    const checksum = await calculateFileChecksum(data);
    
    // This actually uses crypto.subtle.digest()!
    expect(checksum).toBe('b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9');
  });
  
  it('should handle Blob with real arrayBuffer()', async () => {
    const blob = new Blob(['test data']);
    const checksum = await calculateFileChecksum(blob);
    
    expect(checksum).toHaveLength(64);
  });
});
```

---

## Updated npm Scripts

```json
{
  "scripts": {
    "test": "vitest",
    "test:unit": "vitest run --config vitest.config.ts",
    "test:browser": "vitest run --config vitest.config.browser.ts",
    "test:all": "npm run test:unit && npm run test:browser",
    "test:watch": "vitest",
    "test:ui": "vitest --ui",
    "coverage": "vitest run --coverage",
    "coverage:browser": "vitest run --config vitest.config.browser.ts --coverage"
  }
}
```

---

## Execution Strategy

### Development (Fast Feedback)
```bash
npm run test:unit  # Fast, runs in ~2s
```

### Pre-commit (Comprehensive)
```bash
npm run test:all   # Unit + Browser, runs in ~10s
```

### CI/CD (Full Coverage)
```bash
npm run test:all
npm run coverage
npm run test:visual  # Screenshot comparison
```

---

## Coverage Targets (Revised)

### Layer 1 (Unit Tests)
- **Target**: 95%+ for pure JS
- **Files**: math, geometry, stats, exceptions, schemas
- **Fast**: ~2 seconds

### Layer 2 (Browser Tests)
- **Target**: 80%+ for browser-specific code
- **Files**: drawing, checksum, image processors
- **Slower**: ~10 seconds (browser startup)

### Layer 3 (Visual Tests)
- **Target**: Key visual features
- **Count**: ~20 snapshot comparisons
- **Slow**: ~30 seconds (render + compare)

---

## What This Fixes

### Before (Current - BROKEN)
```
✅ math.ts tests pass
✅ 94% coverage reported
❌ BUT: OpenCV.js skipped!
❌ Drawing functions not actually tested
❌ False sense of security
```

### After (Revised - CORRECT)
```
✅ math.ts: 95% coverage (pure JS)
✅ drawing.ts: 85% coverage (real OpenCV.js)
✅ Memory leaks detected
✅ Actual browser APIs tested
✅ Visual output validated
```

---

## Migration Path

### Immediate Actions
1. ✅ Keep current unit tests (they're valid for pure JS)
2. 🆕 Install Playwright
3. 🆕 Create browser test config
4. 🆕 Move drawing/checksum tests to browser suite
5. 🆕 Add memory leak detection
6. 🆕 Update documentation

### DO NOT
- ❌ Delete current tests (they test pure JS correctly)
- ❌ Try to run OpenCV tests in jsdom
- ❌ Fake OpenCV.js with mocks (defeats the purpose)

---

## Success Metrics

### Unit Tests (Layer 1)
- ✅ 132 tests passing in ~2s
- ✅ No browser startup overhead
- ✅ Fast feedback loop

### Browser Tests (Layer 2)
- 🎯 50+ tests covering OpenCV.js
- 🎯 Memory leak detection working
- 🎯 Real Web Crypto API tested

### Visual Tests (Layer 3)
- 🎯 20 snapshot tests
- 🎯 Pixel-perfect validation

---

## Next Steps

1. Install Playwright: `npm install -D @playwright/test`
2. Create `vitest.config.browser.ts`
3. Create `tests/browser-setup.ts`
4. Move/rewrite browser-dependent tests
5. Update CI/CD pipeline
6. Document browser testing requirements

---

## Conclusion

**Key Insight**: We were testing JavaScript syntax but not actual functionality!

**Solution**: Hybrid approach
- Fast unit tests for pure JS (keep current)
- Real browser tests for OpenCV.js/Web APIs (add new)
- Visual tests for correctness validation (add new)

**Result**: Actual confidence that code works in browser!

---

**Status**: Ready to implement  
**Blocking**: None  
**Estimated effort**: 2-3 days for full migration
