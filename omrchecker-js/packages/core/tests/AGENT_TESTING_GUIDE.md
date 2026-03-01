# Agent Testing Guide

Quick reference for AI agents writing browser tests with Playwright and OpenCV.js.

## Critical Rules

### 1. Always Use `beforeEach` with Playwright Fixtures ⚠️

**DON'T**:
```typescript
test.beforeAll(async ({ page }) => {  // ❌ WILL FAIL
  await setupBrowser(page);
});
```

**DO**:
```typescript
test.beforeEach(async ({ page }) => {  // ✅ CORRECT
  await setupBrowser(page);
});
```

**Why**: Playwright `page` and `context` fixtures are created per-test, not per-suite. Using them in `beforeAll`/`afterAll` throws an error.

**Performance**: OpenCV.js is cached by browser after first load. `beforeEach` is fast (~2-3s first test, <1s subsequent tests).

### 2. Don't Rely on OpenCV.js Mat Counting 🚫

**DON'T**:
```typescript
const before = await getMatCount(page);  // ❌ Returns 0 or unreliable
await page.evaluate(() => { 
  const mat = new cv.Mat(10, 10, cv.CV_8UC1); 
});
const after = await getMatCount(page);  // ❌ Still 0
expect(after).toBeGreaterThan(before);  // ❌ WILL FAIL
```

**DO**:
```typescript
const result = await page.evaluate(() => {  // ✅ Single evaluation
  const mat = new cv.Mat(10, 10, cv.CV_8UC1);
  const valid = mat.rows === 10 && mat.cols === 10;
  mat.delete();
  return valid;
});
expect(result).toBe(true);  // ✅ WORKS
```

**Why**: OpenCV.js doesn't expose `matRegistry` or `getAllocatedMatCount()`. Mat count doesn't persist across `page.evaluate()` calls.

**Rule**: Always verify Mat operations within a single `page.evaluate()` call.

### 3. Test Mat Lifecycle, Not Exact Counts 📊

**DON'T**:
```typescript
// ❌ Trying to track exact Mat counts
const leaked = await getMatCount(page) - initialCount;
expect(leaked).toBe(0);
```

**DO**:
```typescript
// ✅ Test operational correctness
await page.evaluate(() => {
  const mat1 = new cv.Mat(10, 10, cv.CV_8UC1);
  const mat2 = mat1.clone();
  
  // Verify operations work
  const same = mat2.rows === mat1.rows;
  
  // Clean up
  mat1.delete();
  mat2.delete();
  
  return same;
});
```

**Why**: Focus on correctness of operations (create, clone, delete) rather than internal memory tracking.

### 4. crypto.subtle Requires Secure Context 🔒

**Problem**: `crypto.subtle` is `undefined` in `about:blank` or data URLs.

**Quick Fix**:
```typescript
test.beforeEach(async ({ page }) => {
  // Navigate to data URL for secure context
  await page.goto('data:text/html,<!DOCTYPE html><html><body></body></html>');
});
```

**Better Fix**: Add `webServer` to `playwright.config.ts`:
```typescript
export default defineConfig({
  webServer: {
    command: 'python -m http.server 8000',
    url: 'http://localhost:8000',
    reuseExistingServer: !process.env.CI,
  },
});
```

### 5. Memory Tracking Limitations ⚠️

**`withMemoryTracking()` limitations**:
- Relies on `getMatCount()` which may return 0
- Best used for complete operations, not exact leak detection
- Don't depend on it for critical assertions

**Better approach**:
```typescript
// ✅ Test proper cleanup patterns
await page.evaluate(() => {
  const mats = [];
  try {
    mats.push(new cv.Mat(10, 10, cv.CV_8UC1));
    mats.push(new cv.Mat(20, 20, cv.CV_8UC3));
    // ... operations ...
  } finally {
    mats.forEach(mat => mat.delete());  // Proper cleanup
  }
});
```

## Quick Checklist for New Tests

- [ ] Use `beforeEach` (not `beforeAll`) for page fixtures
- [ ] Keep Mat operations within single `page.evaluate()` calls
- [ ] Test operational correctness, not internal counts
- [ ] Always call `mat.delete()` for cleanup
- [ ] Use `try/finally` for guaranteed cleanup
- [ ] For crypto tests, ensure secure context (localhost or data URL)
- [ ] Set reasonable timeouts (60s for OpenCV.js loading)

## Common Patterns

### Pattern: Test Mat Creation and Cleanup
```typescript
test('should create and delete Mat', async ({ page }) => {
  const result = await page.evaluate(() => {
    const mat = new cv.Mat(10, 10, cv.CV_8UC1);
    const created = mat.rows === 10;
    mat.delete();
    return created;
  });
  expect(result).toBe(true);
});
```

### Pattern: Test Multiple Mats
```typescript
test('should handle multiple Mats', async ({ page }) => {
  const result = await page.evaluate(() => {
    const mat1 = new cv.Mat(5, 5, cv.CV_8UC1);
    const mat2 = new cv.Mat(10, 10, cv.CV_8UC3);
    const mat3 = new cv.Mat(3, 3, cv.CV_32FC1);
    
    const correct = (
      mat1.channels() === 1 &&
      mat2.channels() === 3 &&
      mat3.channels() === 1
    );
    
    mat1.delete();
    mat2.delete();
    mat3.delete();
    
    return correct;
  });
  expect(result).toBe(true);
});
```

### Pattern: Test with Guaranteed Cleanup
```typescript
test('should cleanup even on error', async ({ page }) => {
  const result = await page.evaluate(() => {
    let mat;
    try {
      mat = new cv.Mat(10, 10, cv.CV_8UC1);
      // ... operations that might throw ...
      const value = mat.rows;
      return value === 10;
    } finally {
      if (mat) mat.delete();
    }
  });
  expect(result).toBe(true);
});
```

## Performance Tips

1. **OpenCV.js caching**: After first test loads OpenCV.js (~10s), subsequent tests are fast (<1s)
2. **Parallel execution**: Tests run in parallel by default (4 workers)
3. **Timeout management**: Set `test.setTimeout(60000)` for OpenCV.js loading
4. **Browser reuse**: Playwright reuses browser context for speed

## Error Messages & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `"page" fixtures not supported in beforeAll` | Using page in beforeAll | Change to beforeEach |
| `Cannot read properties of undefined (reading 'digest')` | crypto.subtle needs secure context | Use data URL or webServer |
| `getMatCount() returns 0` | OpenCV.js doesn't expose count API | Test within single evaluate() |
| `TimeoutError: page.goto` | Network issue loading OpenCV.js | Increase timeout or use local CDN |

## Related Files

- `tests/README.md` - Full testing documentation with troubleshooting
- `tests/browser/browser-setup.ts` - OpenCV.js loading utilities
- `tests/browser/memory-utils.ts` - Memory tracking (use with caution)
- `playwright.config.ts` - Playwright configuration

## When in Doubt

1. Look at `tests/browser/opencv-smoke.test.ts` for working examples
2. Keep Mat operations in single `page.evaluate()` calls
3. Focus on operational correctness over internal memory tracking
4. Always clean up Mats with `mat.delete()`
