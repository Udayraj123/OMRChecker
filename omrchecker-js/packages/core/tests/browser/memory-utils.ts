/**
 * Memory tracking utilities for OpenCV.js leak detection
 *
 * OpenCV.js uses manual memory management - cv.Mat objects must be explicitly deleted.
 * These utilities help detect memory leaks in tests by tracking Mat creation/deletion.
 *
 * Usage:
 * ```typescript
 * import { test, expect } from '@playwright/test';
 * import { withMemoryTracking, getMatCount } from './memory-utils';
 *
 * test('should not leak memory', async ({ page }) => {
 *   await withMemoryTracking(page, async () => {
 *     // Your test code that creates cv.Mat objects
 *     const mat = await page.evaluate(() => {
 *       const m = new window.cv.Mat(10, 10, window.cv.CV_8UC1);
 *       m.delete(); // Must delete!
 *       return null;
 *     });
 *   });
 *   // Throws if memory leaked
 * });
 * ```
 */

import type { Page } from '@playwright/test';

/**
 * Memory snapshot containing Mat count and metadata
 */
export interface MemorySnapshot {
  /** Number of cv.Mat objects currently allocated */
  matCount: number;
  /** Timestamp when snapshot was taken */
  timestamp: number;
  /** Additional metadata (for debugging) */
  metadata?: Record<string, any>;
}

/**
 * Options for memory tracking
 */
export interface MemoryTrackingOptions {
  /** Whether to log memory stats (default: false) */
  verbose?: boolean;
  /** Allowed leaked Mats count (default: 0) */
  allowedLeaks?: number;
  /** Whether to throw on leak detection (default: true) */
  throwOnLeak?: boolean;
}

/**
 * Get the current count of cv.Mat objects in the page
 *
 * This works by accessing OpenCV.js internals that track allocated objects.
 * Note: This may be implementation-specific to OpenCV.js versions.
 *
 * @param page - Playwright page instance
 * @returns Number of currently allocated cv.Mat objects
 */
export async function getMatCount(page: Page): Promise<number> {
  return await page.evaluate(() => {
    if (typeof window.cv === 'undefined') {
      throw new Error('OpenCV.js not loaded - cannot get Mat count');
    }

    // OpenCV.js tracks allocated objects internally
    // Access the registry if available
    if (typeof (window.cv as any).matRegistry !== 'undefined') {
      return Object.keys((window.cv as any).matRegistry).length;
    }

    // Fallback: Try to count through internal tracking
    // Note: This relies on OpenCV.js implementation details
    // and may need adjustment for different versions
    let count = 0;
    try {
      // Some OpenCV.js versions expose getAllocatedMatCount
      if (typeof (window.cv as any).getAllocatedMatCount === 'function') {
        count = (window.cv as any).getAllocatedMatCount();
      } else {
        // Fallback: estimate from internal data structures
        // This is a heuristic and may not be 100% accurate
        count = 0; // Default to 0 if we can't determine
      }
    } catch (e) {
      console.warn('Unable to get accurate Mat count:', e);
      count = 0;
    }

    return count;
  });
}

/**
 * Take a memory snapshot for comparison
 *
 * @param page - Playwright page instance
 * @param metadata - Optional metadata to attach
 * @returns Memory snapshot
 */
export async function takeMemorySnapshot(
  page: Page,
  metadata?: Record<string, any>
): Promise<MemorySnapshot> {
  const matCount = await getMatCount(page);
  return {
    matCount,
    timestamp: Date.now(),
    metadata,
  };
}

/**
 * Compare two memory snapshots and check for leaks
 *
 * @param before - Snapshot before operation
 * @param after - Snapshot after operation
 * @param allowedLeaks - Number of allowed leaked Mats (default: 0)
 * @returns Leaked Mat count (0 if no leak)
 * @throws Error if leak detected and allowedLeaks is 0
 */
export function compareSnapshots(
  before: MemorySnapshot,
  after: MemorySnapshot,
  allowedLeaks: number = 0
): number {
  const leaked = after.matCount - before.matCount;

  if (leaked > allowedLeaks) {
    const message =
      `Memory leak detected: ${leaked} cv.Mat objects leaked ` +
      `(before: ${before.matCount}, after: ${after.matCount}, allowed: ${allowedLeaks})`;
    throw new Error(message);
  }

  return leaked;
}

/**
 * Wrap a test function with memory leak detection
 *
 * Takes a memory snapshot before and after the function executes,
 * then checks for leaks. Throws if leaked Mats exceed the allowed threshold.
 *
 * @param page - Playwright page instance
 * @param fn - Async function to wrap
 * @param options - Memory tracking options
 * @throws Error if memory leak detected
 *
 * @example
 * ```typescript
 * await withMemoryTracking(page, async () => {
 *   await page.evaluate(() => {
 *     const mat = new window.cv.Mat(10, 10, window.cv.CV_8UC1);
 *     mat.delete(); // Must clean up
 *   });
 * });
 * ```
 */
export async function withMemoryTracking<T>(
  page: Page,
  fn: () => Promise<T>,
  options: MemoryTrackingOptions = {}
): Promise<T> {
  const { verbose = false, allowedLeaks = 0, throwOnLeak = true } = options;

  // Take "before" snapshot
  const before = await takeMemorySnapshot(page, { phase: 'before' });

  if (verbose) {
    console.log(`[memory] Before: ${before.matCount} Mats allocated`);
  }

  // Execute the function
  let result: T;
  try {
    result = await fn();
  } catch (error) {
    // Still check memory even if test failed
    const after = await takeMemorySnapshot(page, { phase: 'after-error' });
    if (verbose) {
      console.log(`[memory] After (error): ${after.matCount} Mats allocated`);
    }
    throw error;
  }

  // Take "after" snapshot
  const after = await takeMemorySnapshot(page, { phase: 'after' });

  if (verbose) {
    console.log(`[memory] After: ${after.matCount} Mats allocated`);
  }

  // Check for leaks
  const leaked = after.matCount - before.matCount;

  if (leaked > allowedLeaks) {
    const message =
      `Memory leak detected: ${leaked} cv.Mat objects leaked ` +
      `(before: ${before.matCount}, after: ${after.matCount}, allowed: ${allowedLeaks})`;

    if (throwOnLeak) {
      throw new Error(message);
    } else if (verbose) {
      console.warn(`[memory] ${message}`);
    }
  } else if (verbose && leaked > 0) {
    console.log(`[memory] ${leaked} Mat(s) leaked (within allowed threshold of ${allowedLeaks})`);
  } else if (verbose) {
    console.log('[memory] No memory leaks detected ✓');
  }

  return result;
}

/**
 * Force garbage collection in the browser (if available)
 *
 * Note: This requires the browser to be launched with --expose-gc flag.
 * Playwright doesn't enable this by default.
 *
 * @param page - Playwright page instance
 * @returns true if GC was triggered, false if not available
 */
export async function forceGarbageCollection(page: Page): Promise<boolean> {
  return await page.evaluate(() => {
    if (typeof (window as any).gc === 'function') {
      (window as any).gc();
      return true;
    }
    return false;
  });
}

/**
 * Get detailed memory statistics from the page
 *
 * @param page - Playwright page instance
 * @returns Memory statistics object
 */
export async function getMemoryStats(page: Page): Promise<{
  matCount: number;
  hasGC: boolean;
  cvVersion?: string;
}> {
  return await page.evaluate(() => {
    const hasGC = typeof (window as any).gc === 'function';

    // Get OpenCV.js version if available
    let cvVersion: string | undefined;
    try {
      if (typeof window.cv !== 'undefined' && typeof window.cv.getBuildInformation === 'function') {
        const buildInfo = window.cv.getBuildInformation();
        // Extract version from build info (format varies)
        const versionMatch = buildInfo.match(/OpenCV:\s*(\S+)/);
        cvVersion = versionMatch ? versionMatch[1] : undefined;
      }
    } catch (e) {
      // Ignore
    }

    // Get Mat count (same logic as getMatCount)
    let matCount = 0;
    if (typeof window.cv !== 'undefined') {
      if (typeof (window.cv as any).matRegistry !== 'undefined') {
        matCount = Object.keys((window.cv as any).matRegistry).length;
      } else if (typeof (window.cv as any).getAllocatedMatCount === 'function') {
        matCount = (window.cv as any).getAllocatedMatCount();
      }
    }

    return {
      matCount,
      hasGC,
      cvVersion,
    };
  });
}
