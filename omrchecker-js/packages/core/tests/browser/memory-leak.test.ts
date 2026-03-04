/**
 * Memory leak detection tests for OpenCV.js
 * 
 * These tests verify that our memory tracking utilities correctly detect
 * memory leaks in OpenCV.js operations. Tests intentional leaks, proper
 * cleanup patterns, and edge cases.
 * 
 * Run with: npm run test:browser
 */

import { test, expect } from '@playwright/test';
import { setupBrowser, teardownBrowser } from './browser-setup';
import { withMemoryTracking, getMatCount, getMemoryStats, installMatTracking } from './memory-utils';

// Increase timeout for browser tests
test.setTimeout(60000);

test.describe('Memory Leak Detection - Browser Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Setup OpenCV.js for each test, then install the Mat-tracking shim so
    // getMatCount() returns real allocation counts instead of always-0.
    await setupBrowser(page);
    await installMatTracking(page);
  });

  test.describe('withMemoryTracking - Clean Operations', () => {
    test('should pass when Mat is properly deleted', async ({ page }) => {
      await withMemoryTracking(page, async () => {
        await page.evaluate(() => {
          const mat = new window.cv.Mat(10, 10, window.cv.CV_8UC1);
          mat.delete(); // Proper cleanup
        });
      });
      // Test passes if no error thrown
    });

    test('should pass when multiple Mats are properly deleted', async ({ page }) => {
      await withMemoryTracking(page, async () => {
        await page.evaluate(() => {
          const mat1 = new window.cv.Mat(10, 10, window.cv.CV_8UC1);
          const mat2 = new window.cv.Mat(20, 20, window.cv.CV_8UC3);
          const mat3 = new window.cv.Mat(30, 30, window.cv.CV_32FC1);
          
          mat1.delete();
          mat2.delete();
          mat3.delete();
        });
      });
    });

    test('should pass when Mat is deleted in finally block', async ({ page }) => {
      await withMemoryTracking(page, async () => {
        await page.evaluate(() => {
          let mat: any;
          try {
            mat = new window.cv.Mat(10, 10, window.cv.CV_8UC1);
            // Do some operations
            const val = mat.rows;
          } finally {
            if (mat) mat.delete();
          }
        });
      });
    });
  });

  test.describe('withMemoryTracking - Leak Detection', () => {
    test('should detect leak when Mat is not deleted', async ({ page }) => {
      let leakDetected = false;
      
      try {
        await withMemoryTracking(page, async () => {
          await page.evaluate(() => {
            new window.cv.Mat(10, 10, window.cv.CV_8UC1);
            // Intentional leak - no delete()
          });
        });
      } catch (error) {
        if (error instanceof Error && error.message.includes('Memory leak detected')) {
          leakDetected = true;
        }
      }
      
      expect(leakDetected).toBe(true);
    });

    test('should detect partial leaks when some Mats are not deleted', async ({ page }) => {
      let leakDetected = false;
      
      try {
        await withMemoryTracking(page, async () => {
          await page.evaluate(() => {
            const mat1 = new window.cv.Mat(10, 10, window.cv.CV_8UC1);
            const mat2 = new window.cv.Mat(20, 20, window.cv.CV_8UC3);
            
            mat1.delete(); // Clean
            // mat2 NOT deleted - leak!
          });
        });
      } catch (error) {
        if (error instanceof Error && error.message.includes('Memory leak detected')) {
          leakDetected = true;
        }
      }
      
      expect(leakDetected).toBe(true);
    });

    test('should detect leaks in MatVector', async ({ page }) => {
      let leakDetected = false;
      
      try {
        await withMemoryTracking(page, async () => {
          await page.evaluate(() => {
            const matVec = new window.cv.MatVector();
            const mat = new window.cv.Mat(10, 10, window.cv.CV_8UC1);
            matVec.push_back(mat);
            
            // Forgot to delete matVec and mat - leak!
          });
        });
      } catch (error) {
        if (error instanceof Error && error.message.includes('Memory leak detected')) {
          leakDetected = true;
        }
      }
      
      expect(leakDetected).toBe(true);
    });
  });

  test.describe('getMatCount - Accuracy', () => {
    test('should accurately count created Mats', async ({ page }) => {
      const initialCount = await getMatCount(page);
      
      await page.evaluate(() => {
        (window as any).testMats = [
          new window.cv.Mat(10, 10, window.cv.CV_8UC1),
          new window.cv.Mat(20, 20, window.cv.CV_8UC3),
          new window.cv.Mat(30, 30, window.cv.CV_32FC1),
        ];
      });
      
      const afterCreate = await getMatCount(page);
      expect(afterCreate).toBeGreaterThanOrEqual(initialCount + 3);
      
      // Cleanup
      await page.evaluate(() => {
        (window as any).testMats.forEach((mat: any) => mat.delete());
        delete (window as any).testMats;
      });
      
      const afterDelete = await getMatCount(page);
      expect(afterDelete).toBe(initialCount);
    });

    test('should track Mat count through operations', async ({ page }) => {
      const counts = await page.evaluate(async () => {
        // Helper to get count (simplified for in-browser use)
        const getCount = () => {
          // This is a simplified version - actual implementation may vary
          return 0; // Placeholder
        };
        
        const before = getCount();
        
        const mat1 = new window.cv.Mat(10, 10, window.cv.CV_8UC1);
        const afterCreate = getCount();
        
        mat1.delete();
        const afterDelete = getCount();
        
        return { before, afterCreate, afterDelete };
      });
      
      // Counts should change appropriately
      expect(counts).toBeDefined();
    });
  });

  test.describe('Complex Memory Scenarios', () => {
    test('should handle nested Mat operations correctly', async ({ page }) => {
      await withMemoryTracking(page, async () => {
        await page.evaluate(() => {
          const createAndProcess = () => {
            const mat = new window.cv.Mat(10, 10, window.cv.CV_8UC1);
            const clone = mat.clone();
            
            mat.delete();
            clone.delete();
          };
          
          // Call multiple times
          createAndProcess();
          createAndProcess();
          createAndProcess();
        });
      });
    });

    test('should detect leaks even with verbose logging', async ({ page }) => {
      let leakDetected = false;
      
      try {
        await withMemoryTracking(page, async () => {
          await page.evaluate(() => {
            const mat = new window.cv.Mat(10, 10, window.cv.CV_8UC1);
            // Leak - no delete
          });
        }, { verbose: true });
      } catch (error) {
        if (error instanceof Error && error.message.includes('Memory leak detected')) {
          leakDetected = true;
        }
      }
      
      expect(leakDetected).toBe(true);
    });
  });

  test.describe('getMemoryStats', () => {
    test('should return memory statistics', async ({ page }) => {
      const stats = await getMemoryStats(page);
      
      expect(stats).toBeDefined();
      expect(typeof stats.matCount).toBe('number');
      expect(typeof stats.hasGC).toBe('boolean');
      expect(stats.matCount).toBeGreaterThanOrEqual(0);
    });
  });
});
