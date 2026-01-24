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
    // Based on official OpenCV.js Node.js pattern:
    // https://docs.opencv.org/4.x/dc/de6/tutorial_js_nodejs.html
    const opencvPath = resolve(__dirname, '../../../lib/opencv.js');
    const nodeRequire = createRequire(import.meta.url);
    const opencvDir = dirname(opencvPath);

    // Create a Module object with onRuntimeInitialized callback
    // This MUST be set up BEFORE requiring opencv.js
    let initResolve: () => void;
    const initPromise = new Promise<void>((resolve) => {
      initResolve = resolve;
    });

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
    const fakeModule = { exports: {} };
    (global as any).module = fakeModule;
    (global as any).exports = fakeModule.exports;

    // Define Module globally BEFORE requiring opencv.js
    // This is the key pattern from the official docs
    // The Module object must have onRuntimeInitialized callback
    const ModuleObj: any = {
      onRuntimeInitialized: () => {
        console.log('OpenCV onRuntimeInitialized called');
        initResolve();
      },
      // OpenCV.js may need these properties
      locateFile: (path: string) => {
        // Return path relative to opencv.js location for WASM files
        return resolve(opencvDir, path);
      },
    };
    (global as any).Module = ModuleObj;

    try {
      // Load the module - opencv.js will use the global Module object
      // The UMD wrapper exports cv(Module), which returns Module after initialization
      console.log('Loading opencv.js...');
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const cvModule = nodeRequire(opencvPath);
      console.log('opencv.js loaded');

      // Check if Module.run exists and needs to be called
      // Some versions of opencv.js require explicit run() call
      if (ModuleObj.run && typeof ModuleObj.run === 'function' && !ModuleObj.calledRun) {
        console.log('Calling Module.run()...');
        ModuleObj.run();
      }

      console.log('Waiting for initialization...');
      // Wait for WASM initialization with timeout
      await Promise.race([
        initPromise,
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error('OpenCV initialization timeout after 10s')), 10000)
        ),
      ]);
      console.log('OpenCV initialized');

      // After initialization, Module should have all OpenCV functions
      // The cv function returns the Module object, which is now populated
      // Check both the returned module and the global Module
      if (cvModule && typeof cvModule === 'object' && 'Scalar' in cvModule) {
        cvInstance = cvModule;
      } else if (ModuleObj && typeof ModuleObj === 'object' && 'Scalar' in ModuleObj) {
        cvInstance = ModuleObj;
      } else if ((global as any).Module && typeof (global as any).Module === 'object' && 'Scalar' in (global as any).Module) {
        cvInstance = (global as any).Module;
      } else {
        throw new Error('OpenCV.js not properly initialized - cv object not found');
      }

      // Verify that Scalar is available (indicates proper initialization)
      if (!cvInstance || typeof (cvInstance as any).Scalar !== 'function') {
        throw new Error('OpenCV.js not properly initialized - Scalar constructor not found');
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
      // Don't delete Module - tests need it
      // if (originalGlobalModule !== undefined) {
      //   (global as any).Module = originalGlobalModule;
      // } else {
      //   delete (global as any).Module;
      // }
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
// Use a Proxy to forward property access to the cv instance
// This allows code to use cv.Rect, cv.Mat, etc. directly
// Note: This will throw if OpenCV is not initialized
// Use getCV() after calling initOpenCV() or import { getCV } directly
const cvProxy = new Proxy({} as typeof cv, {
  get(_target, prop) {
    const cv = getCV();
    return (cv as any)[prop];
  },
  set(_target, prop, value) {
    const cv = getCV();
    (cv as any)[prop] = value;
    return true;
  },
  has(_target, prop) {
    const cv = getCV();
    return prop in cv;
  },
  ownKeys(_target) {
    const cv = getCV();
    return Object.keys(cv);
  },
  getOwnPropertyDescriptor(_target, prop) {
    const cv = getCV();
    return Object.getOwnPropertyDescriptor(cv, prop);
  },
});

export default cvProxy;
