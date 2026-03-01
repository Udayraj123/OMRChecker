/**
 * Browser setup utilities for Playwright browser testing with OpenCV.js
 *
 * This module provides helpers to load and initialize OpenCV.js in a real browser environment
 * using Playwright. It handles WASM loading, initialization timeouts, and proper cleanup.
 *
 * Usage:
 * ```typescript
 * import { test, expect } from '@playwright/test';
 * import { setupBrowser, teardownBrowser } from './browser-setup';
 *
 * test.beforeAll(async ({ page }) => {
 *   await setupBrowser(page);
 * });
 *
 * test.afterAll(async ({ page }) => {
 *   await teardownBrowser(page);
 * });
 * ```
 */

import type { Page } from '@playwright/test';

/**
 * OpenCV.js global object interface
 * Extends the Window interface with cv global
 */
declare global {
  interface Window {
    cv: any; // OpenCV.js exposes cv globally
  }
}

export interface BrowserSetupOptions {
  /** Timeout in ms to wait for OpenCV.js to initialize (default: 30000ms) */
  timeout?: number;
  /** OpenCV.js CDN URL or local path (default: CDN) */
  opencvUrl?: string;
  /** Whether to enable verbose logging (default: false) */
  verbose?: boolean;
}

const DEFAULT_OPENCV_URL = 'https://docs.opencv.org/4.x/opencv.js';
const DEFAULT_TIMEOUT = 30000; // 30 seconds

/**
 * Setup OpenCV.js in the browser page
 *
 * Loads OpenCV.js from CDN or local path and waits for WASM initialization to complete.
 * Throws an error if initialization times out or fails.
 *
 * @param page - Playwright page instance
 * @param options - Optional configuration
 * @throws Error if OpenCV.js fails to load or initialize
 */
export async function setupBrowser(
  page: Page,
  options: BrowserSetupOptions = {}
): Promise<void> {
  const { timeout = DEFAULT_TIMEOUT, opencvUrl = DEFAULT_OPENCV_URL, verbose = false } = options;

  if (verbose) {
    console.log(`[browser-setup] Loading OpenCV.js from: ${opencvUrl}`);
  }

  try {
    // Navigate to a blank page first
    await page.goto('about:blank');

    // Inject OpenCV.js script
    await page.addScriptTag({
      url: opencvUrl,
    });

    if (verbose) {
      console.log('[browser-setup] OpenCV.js script loaded, waiting for WASM initialization...');
    }

    // Wait for OpenCV.js to initialize (cv.Mat becomes available when ready)
    await page.waitForFunction(
      () => {
        return (
          typeof window.cv !== 'undefined' &&
          typeof window.cv.Mat !== 'undefined' &&
          window.cv.Mat.name === 'Mat'
        );
      },
      { timeout }
    );

    if (verbose) {
      console.log('[browser-setup] OpenCV.js initialized successfully');
    }

    // Verify cv is actually usable by creating a simple Mat
    const isUsable = await page.evaluate(() => {
      try {
        const testMat = new window.cv.Mat(1, 1, window.cv.CV_8UC1);
        const isValid = testMat.rows === 1 && testMat.cols === 1;
        testMat.delete(); // Clean up
        return isValid;
      } catch (e) {
        return false;
      }
    });

    if (!isUsable) {
      throw new Error('OpenCV.js loaded but cv.Mat is not functional');
    }

    if (verbose) {
      console.log('[browser-setup] OpenCV.js verification successful');
    }
  } catch (error) {
    if (error instanceof Error) {
      if (error.message.includes('Timeout')) {
        throw new Error(
          `OpenCV.js initialization timed out after ${timeout}ms. ` +
            `Check network connectivity and WASM loading. URL: ${opencvUrl}`
        );
      }
      throw new Error(`Failed to setup OpenCV.js: ${error.message}`);
    }
    throw error;
  }
}

/**
 * Teardown OpenCV.js and cleanup resources
 *
 * Currently a no-op, but reserved for future cleanup needs (e.g., releasing memory,
 * removing global cv object, etc.)
 *
 * @param page - Playwright page instance
 */
export async function teardownBrowser(page: Page): Promise<void> {
  // Future: Add cleanup logic here if needed
  // For now, Playwright handles page cleanup automatically
  // We could explicitly delete the cv global if needed:
  // await page.evaluate(() => { delete (window as any).cv; });
}

/**
 * Check if OpenCV.js is loaded and ready in the page
 *
 * @param page - Playwright page instance
 * @returns true if OpenCV.js is ready, false otherwise
 */
export async function isOpenCVReady(page: Page): Promise<boolean> {
  try {
    return await page.evaluate(() => {
      return (
        typeof window.cv !== 'undefined' &&
        typeof window.cv.Mat !== 'undefined' &&
        window.cv.Mat.name === 'Mat'
      );
    });
  } catch {
    return false;
  }
}

/**
 * Get OpenCV.js build information from the browser
 *
 * @param page - Playwright page instance
 * @returns OpenCV build info string or null if not available
 */
export async function getOpenCVBuildInfo(page: Page): Promise<string | null> {
  try {
    return await page.evaluate(() => {
      if (typeof window.cv !== 'undefined' && typeof window.cv.getBuildInformation === 'function') {
        return window.cv.getBuildInformation();
      }
      return null;
    });
  } catch {
    return null;
  }
}
