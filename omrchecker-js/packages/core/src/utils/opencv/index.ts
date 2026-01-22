/**
 * Centralized OpenCV initialization and access module.
 *
 * This module provides a proper async initialization pattern for OpenCV.js
 * following the TechStark/opencv-js recommended approach.
 *
 * @see https://github.com/TechStark/opencv-js/
 */

import cvModule from '@techstark/opencv-js';

// global instance
let cvInstance: any = null;

/**
 * Initialize OpenCV asynchronously.
 *
 * This handles both Promise and callback-based initialization patterns
 * since OpenCV.js is a WebAssembly module that needs proper initialization.
 *
 * @returns Promise resolving to the initialized OpenCV instance
 */
export async function initOpenCV(): Promise<any> {
  if (cvInstance) return cvInstance;

  if (cvModule instanceof Promise) {
    // Promise API
    cvInstance = await cvModule;
  } else if ((cvModule as any).Mat) {
    // Already initialized
    cvInstance = cvModule as any;
  } else {
    // Callback API
    await new Promise<void>((resolve) => {
      (cvModule as { onRuntimeInitialized: () => void }).onRuntimeInitialized =
        resolve;
    });
    cvInstance = cvModule as any;
  }
  return cvInstance;
}

/**
 * Get the initialized OpenCV instance.
 *
 * @throws Error if OpenCV has not been initialized yet
 * @returns The initialized OpenCV instance
 */
export function getCV(): any {
  if (!cvInstance) {
    throw new Error('OpenCV not initialized. Call initOpenCV() first.');
  }
  return cvInstance;
}

/**
 * Check if OpenCV has been initialized.
 *
 * @returns true if OpenCV is ready to use
 */
export function isOpenCVInitialized(): boolean {
  return cvInstance !== null;
}

// Default export for simpler import syntax
export default cvInstance;

