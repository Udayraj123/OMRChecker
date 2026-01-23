/**
 * Centralized OpenCV initialization and access module.
 *
 * This module provides a proper async initialization pattern for OpenCV.js
 * supporting both browser (global cv) and Node.js (local file) environments.
 *
 * Browser: Expects opencv.js to be loaded via script tag before use
 * Node.js: Loads from local lib/opencv.js file (for tests)
 */

import path from 'path';

// Global instance
let cvInstance: typeof cv | null = null;

/**
 * Initialize OpenCV asynchronously.
 *
 * This handles both browser and Node.js environments:
 * - Browser: Uses global cv object loaded via script tag
 * - Node.js: Loads from local lib/opencv.js file
 *
 * @returns Promise resolving to the initialized OpenCV instance
 */
export async function initOpenCV(): Promise<typeof cv> {
  if (cvInstance) return cvInstance;

  if (typeof window === 'undefined') {
    // Node.js environment (tests) - load from local file
    const opencvPath = path.resolve(__dirname, '../../../lib/opencv.js');
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const cvModule = require(opencvPath);

    if (cvModule instanceof Promise) {
      cvInstance = await cvModule;
    } else if (cvModule.onRuntimeInitialized) {
      await new Promise<void>((resolve) => {
        cvModule.onRuntimeInitialized = resolve;
      });
      cvInstance = cvModule;
    } else {
      cvInstance = cvModule;
    }
  } else {
    // Browser environment - use global cv loaded via script tag
    if (!(window as any).cv) {
      throw new Error(
        'OpenCV.js not loaded. Add <script src="opencv.js"> before using this library.'
      );
    }
    cvInstance = (window as any).cv;
  }

  return cvInstance!;
}

/**
 * Get the initialized OpenCV instance.
 *
 * @throws Error if OpenCV has not been initialized yet
 * @returns The initialized OpenCV instance
 */
export function getCV(): typeof cv {
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
export default getCV();
