/**
 * Test setup file for OpenCV.js initialization
 *
 * Ensures OpenCV.js is properly loaded and initialized before tests run.
 * This is required because @techstark/opencv-js uses WebAssembly which needs
 * to be initialized before use.
 *
 * For @techstark/opencv-js v4.12.0-release.1, the module works directly in Node.js
 * but we verify it's properly initialized before tests run.
 */

import { beforeAll } from 'vitest';
import * as cv from '@techstark/opencv-js';

/**
 * Wait for OpenCV.js to be fully initialized
 *
 * For @techstark/opencv-js v4.11+, the module may export a Promise.
 * For earlier versions, we wait for onRuntimeInitialized.
 * This function handles both cases.
 */
export async function waitForOpenCV(): Promise<void> {
  // Check if OpenCV is already initialized
  if (typeof cv.Mat !== 'undefined' && typeof cv.CV_8UC1 !== 'undefined') {
    return;
  }

  // For v4.11+, the default export might be a Promise
  // Try to await it if it's a Promise
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const cvModule = cv as unknown as { then?: () => Promise<unknown>; onRuntimeInitialized?: () => void };
  if (cvModule && typeof cvModule.then === 'function') {
    await cvModule.then();
    return;
  }

  // For older versions, wait for onRuntimeInitialized
  if (cvModule.onRuntimeInitialized) {
    return new Promise<void>((resolve) => {
      if (cvModule.onRuntimeInitialized) {
        cvModule.onRuntimeInitialized = () => {
          resolve();
        };
      } else {
        // Already initialized
        resolve();
      }
    });
  }

  // Fallback: wait a bit and check again
  // This handles cases where OpenCV loads synchronously in Node.js
  await new Promise((resolve) => setTimeout(resolve, 100));

  if (typeof cv.Mat === 'undefined') {
    throw new Error(
      'OpenCV.js failed to initialize. ' +
      'Make sure @techstark/opencv-js is properly installed and ' +
      'your Node.js environment supports WebAssembly.'
    );
  }
}

// Initialize OpenCV before all tests
beforeAll(async () => {
  await waitForOpenCV();

  // Verify OpenCV is ready by checking for key classes and constants
  if (typeof cv.Mat === 'undefined') {
    throw new Error('OpenCV.js not loaded after initialization');
  }

  // Additional verification - try to create a small Mat to ensure it's fully functional
  try {
    const testMat = new cv.Mat(1, 1, cv.CV_8UC1);
    testMat.delete();
  } catch (error) {
    throw new Error(`OpenCV.js initialized but not functional: ${error}`);
  }
});

