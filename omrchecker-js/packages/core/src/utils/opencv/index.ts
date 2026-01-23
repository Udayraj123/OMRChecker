/**
 * Centralized OpenCV initialization and access module.
 *
 * This module provides a proper async initialization pattern for OpenCV.js
 * supporting both browser (global cv) and Node.js (local file) environments.
 *
 * Browser: Expects opencv.js to be loaded via script tag before use
 * Node.js: Loads from local lib/opencv.js file (for tests)
 */

import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { createRequire } from 'module';

// Get __dirname equivalent for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

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
    const opencvPath = resolve(__dirname, '../../../lib/opencv.js');
    const nodeRequire = createRequire(import.meta.url);
    const opencvDir = dirname(opencvPath);

    // Create a fake module object for the UMD wrapper
    const fakeModule = { exports: {} };

    // Save original values
    const originalDirname = (global as any).__dirname;
    const originalFilename = (global as any).__filename;
    const originalRequire = (global as any).require;
    const originalModule = (global as any).module;
    const originalExports = (global as any).exports;

    // Set up CommonJS environment for opencv.js
    (global as any).__dirname = opencvDir;
    (global as any).__filename = opencvPath;
    (global as any).require = nodeRequire;
    (global as any).module = fakeModule;
    (global as any).exports = fakeModule.exports;

    try {
      // Load the module - opencv.js uses UMD pattern and will use module.exports path
      // We need to ensure 'this' in the IIFE refers to something valid
      // The UMD checks module.exports first, so it should work
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      let cvModule = nodeRequire(opencvPath);

      // The module might export cv directly or have it as a property
      if (cvModule && typeof cvModule === 'object' && 'cv' in cvModule) {
        cvModule = cvModule.cv;
      }

      // Also check fakeModule.exports
      if (!cvModule && fakeModule.exports && typeof fakeModule.exports === 'object') {
        if ('cv' in fakeModule.exports) {
          cvModule = (fakeModule.exports as any).cv;
        } else {
          cvModule = fakeModule.exports;
        }
      }

      // Handle initialization
      if (cvModule instanceof Promise) {
        cvInstance = await cvModule;
      } else if (cvModule && typeof cvModule === 'object' && 'onRuntimeInitialized' in cvModule) {
        // Wait for WASM initialization
        await new Promise<void>((resolve) => {
          if (cvModule.onRuntimeInitialized) {
            cvModule.onRuntimeInitialized = resolve;
          } else {
            // If already initialized, resolve immediately
            resolve();
          }
        });
        cvInstance = cvModule;
      } else {
        cvInstance = cvModule;
      }

      // Also check if cv was set on global
      if (!cvInstance && (global as any).cv) {
        cvInstance = (global as any).cv;
      }

      if (!cvInstance) {
        throw new Error('Failed to load OpenCV.js - cv object not found');
      }
    } finally {
      // Restore original values
      if (originalDirname !== undefined) {
        (global as any).__dirname = originalDirname;
      } else {
        delete (global as any).__dirname;
      }
      if (originalFilename !== undefined) {
        (global as any).__filename = originalFilename;
      } else {
        delete (global as any).__filename;
      }
      if (originalRequire !== undefined) {
        (global as any).require = originalRequire;
      } else {
        delete (global as any).require;
      }
      if (originalModule !== undefined) {
        (global as any).module = originalModule;
      } else {
        delete (global as any).module;
      }
      if (originalExports !== undefined) {
        (global as any).exports = originalExports;
      } else {
        delete (global as any).exports;
      }
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
// Note: This will throw if OpenCV is not initialized
// Use getCV() after calling initOpenCV() or import { getCV } directly
export default getCV;
