/**
 * Smoke tests for OpenCV.js initialization in Playwright browser
 *
 * These tests verify that:
 * 1. OpenCV.js loads successfully from CDN
 * 2. WASM module initializes properly
 * 3. cv.Mat objects can be created and deleted
 * 4. Memory tracking utilities detect leaks correctly
 *
 * Run with: npm run test:browser
 */

import { test, expect } from '@playwright/test';
import { setupBrowser, teardownBrowser, isOpenCVReady, getOpenCVBuildInfo } from './browser-setup';
import { getMatCount, withMemoryTracking, getMemoryStats } from './memory-utils';

// Increase timeout for browser tests (OpenCV.js loading can take time)
test.setTimeout(60000);

test.describe('OpenCV.js Smoke Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Setup OpenCV.js for each test
    // Note: OpenCV.js is cached by the browser, so subsequent loads are fast
    await setupBrowser(page, { verbose: false });
  });

  test('should load OpenCV.js successfully', async ({ page }) => {
    // Verify OpenCV.js is loaded
    const ready = await isOpenCVReady(page);
    expect(ready).toBe(true);

    // Check that cv object exists
    const hasCv = await page.evaluate(() => {
      return typeof window.cv !== 'undefined';
    });
    expect(hasCv).toBe(true);
  });

  test('should have cv.Mat constructor available', async ({ page }) => {
    const hasMatConstructor = await page.evaluate(() => {
      return (
        typeof window.cv !== 'undefined' &&
        typeof window.cv.Mat !== 'undefined' &&
        typeof window.cv.Mat === 'function'
      );
    });
    expect(hasMatConstructor).toBe(true);
  });

  test('should provide build information', async ({ page }) => {
    const buildInfo = await getOpenCVBuildInfo(page);
    expect(buildInfo).toBeTruthy();
    expect(typeof buildInfo).toBe('string');
    console.log('OpenCV.js Build Info:', buildInfo?.substring(0, 100) + '...');
  });

  test('should create and delete cv.Mat without leaking memory', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      await page.evaluate(() => {
        // Create a simple 10x10 8-bit single-channel Mat
        const mat = new window.cv.Mat(10, 10, window.cv.CV_8UC1);

        // Verify properties
        if (mat.rows !== 10 || mat.cols !== 10) {
          throw new Error('Mat dimensions incorrect');
        }

        // Clean up - CRITICAL for no memory leaks
        mat.delete();
      });
    });

    // If we reach here, no memory leak was detected
  });

  test('should detect memory leaks when Mat is not deleted', async ({ page }) => {
    // This test intentionally leaks memory to verify detection works
    let leakDetected = false;

    try {
      await withMemoryTracking(page, async () => {
        await page.evaluate(() => {
          // Create Mat but DON'T delete it - this will leak!
          new window.cv.Mat(5, 5, window.cv.CV_8UC1);
          // Intentionally not calling mat.delete()
        });
      });
    } catch (error) {
      // We expect an error about memory leak
      if (error instanceof Error && error.message.includes('Memory leak detected')) {
        leakDetected = true;
      } else {
        throw error; // Re-throw if it's a different error
      }
    }

    expect(leakDetected).toBe(true);
  });

  test('should track Mat count accurately', async ({ page }) => {
    // Get initial count
    const initialCount = await getMatCount(page);

    // Create some Mats
    await page.evaluate(() => {
      // Create 3 Mats
      const mat1 = new window.cv.Mat(5, 5, window.cv.CV_8UC1);
      const mat2 = new window.cv.Mat(10, 10, window.cv.CV_8UC3);
      const mat3 = new window.cv.Mat(3, 3, window.cv.CV_32FC1);

      // Store them temporarily (we'll clean up in next evaluate)
      (window as any).testMats = [mat1, mat2, mat3];
    });

    // Count should increase by 3
    const afterCreate = await getMatCount(page);
    expect(afterCreate).toBeGreaterThanOrEqual(initialCount + 3);

    // Delete the Mats
    await page.evaluate(() => {
      const mats = (window as any).testMats;
      mats.forEach((mat: any) => mat.delete());
      delete (window as any).testMats;
    });

    // Count should return to initial
    const afterDelete = await getMatCount(page);
    expect(afterDelete).toBe(initialCount);
  });

  test('should get memory statistics', async ({ page }) => {
    const stats = await getMemoryStats(page);

    expect(stats).toBeDefined();
    expect(typeof stats.matCount).toBe('number');
    expect(typeof stats.hasGC).toBe('boolean');

    console.log('Memory Stats:', {
      matCount: stats.matCount,
      hasGC: stats.hasGC,
      cvVersion: stats.cvVersion,
    });

    // Mat count should be >= 0
    expect(stats.matCount).toBeGreaterThanOrEqual(0);
  });

  test('should support common OpenCV constants', async ({ page }) => {
    const hasConstants = await page.evaluate(() => {
      const cv = window.cv;
      return {
        hasCV_8UC1: typeof cv.CV_8UC1 !== 'undefined',
        hasCV_8UC3: typeof cv.CV_8UC3 !== 'undefined',
        hasCV_8UC4: typeof cv.CV_8UC4 !== 'undefined',
        hasCV_32FC1: typeof cv.CV_32FC1 !== 'undefined',
        hasCV_64FC1: typeof cv.CV_64FC1 !== 'undefined',
      };
    });

    // All common constants should be available
    expect(hasConstants.hasCV_8UC1).toBe(true);
    expect(hasConstants.hasCV_8UC3).toBe(true);
    expect(hasConstants.hasCV_8UC4).toBe(true);
    expect(hasConstants.hasCV_32FC1).toBe(true);
    expect(hasConstants.hasCV_64FC1).toBe(true);
  });

  test('should support Mat data access', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mat = new window.cv.Mat(2, 2, window.cv.CV_8UC1);

        // Set some data
        mat.data[0] = 42;
        mat.data[1] = 100;
        mat.data[2] = 200;
        mat.data[3] = 255;

        // Read it back
        const data = {
          first: mat.data[0],
          second: mat.data[1],
          third: mat.data[2],
          fourth: mat.data[3],
        };

        mat.delete();
        return data;
      });

      expect(result.first).toBe(42);
      expect(result.second).toBe(100);
      expect(result.third).toBe(200);
      expect(result.fourth).toBe(255);
    });
  });

  test('should support Mat clone operation', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        // Create original Mat
        const original = new window.cv.Mat(3, 3, window.cv.CV_8UC1);
        original.data[0] = 123;

        // Clone it
        const cloned = original.clone();

        // Verify clone has same data
        const same = cloned.data[0] === 123;

        // Clean up both
        original.delete();
        cloned.delete();

        return same;
      });

      expect(result).toBe(true);
    });
  });
});
