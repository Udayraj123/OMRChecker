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
 * - Node.js: Loads from local lib/opencv.js file or uses global.cv from setup.ts
 *
 * @returns Promise resolving to the initialized OpenCV instance
 */
export async function initOpenCV(): Promise<typeof cv> {
  if (cvInstance) return cvInstance;

  if (typeof window === 'undefined') {
    // Node.js environment (tests)
    cvInstance = await initOpenCVNode();
  } else {
    // Browser environment
    cvInstance = initOpenCVBrowser();
  }

  return cvInstance;
}

/**
 * Initialize OpenCV in Node.js environment.
 * First checks if global.cv is available (from setup.ts), otherwise loads from file.
 */
async function initOpenCVNode(): Promise<typeof cv> {
  // Check if cv is already available from setup.ts
  const globalCv = (global as any).cv;
  if (globalCv && typeof globalCv.Scalar === 'function') {
    return globalCv;
  }

  // Load from local file
  return loadOpenCVFromFile();
}

/**
 * Load OpenCV.js from local file in Node.js environment.
 * Based on official OpenCV.js Node.js pattern:
 * https://docs.opencv.org/4.x/dc/de6/tutorial_js_nodejs.html
 */
async function loadOpenCVFromFile(): Promise<typeof cv> {
  const opencvPath = resolve(__dirname, '../../../lib/opencv.js');
  const opencvDir = dirname(opencvPath);
  const nodeRequire = createRequire(import.meta.url);

  // Set up initialization promise
  let initResolve: () => void;
  const initPromise = new Promise<void>((resolve) => {
    initResolve = resolve;
  });

  // Create Module object with onRuntimeInitialized callback
  // This MUST be set up BEFORE requiring opencv.js
  const ModuleObj: any = {
    onRuntimeInitialized: () => initResolve(),
    locateFile: (path: string) => resolve(opencvDir, path),
  };

  // Save and set up CommonJS environment
  const savedGlobals = saveAndSetupGlobals(opencvDir, opencvPath, nodeRequire, ModuleObj);

  try {
    // Load opencv.js - it will use the global Module object
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const cvModule = nodeRequire(opencvPath);

    // Some versions require explicit run() call
    if (ModuleObj.run && typeof ModuleObj.run === 'function' && !ModuleObj.calledRun) {
      ModuleObj.run();
    }

    // Wait for WASM initialization with timeout
    await Promise.race([
      initPromise,
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error('OpenCV initialization timeout after 10s')), 10000)
      ),
    ]);

    // Find the initialized cv object
    const cv = findInitializedCV(cvModule, ModuleObj);
    if (!cv || typeof (cv as any).Scalar !== 'function') {
      throw new Error('OpenCV.js not properly initialized - Scalar constructor not found');
    }

    return cv;
  } finally {
    restoreGlobals(savedGlobals);
  }
}

/**
 * Save current global values and set up CommonJS environment for opencv.js.
 */
function saveAndSetupGlobals(
  opencvDir: string,
  opencvPath: string,
  nodeRequire: NodeRequire,
  ModuleObj: any
) {
  const saved = {
    __dirname: (global as any).__dirname,
    __filename: (global as any).__filename,
    require: (global as any).require,
    module: (global as any).module,
    exports: (global as any).exports,
  };

  // Set up CommonJS environment
  (global as any).__dirname = opencvDir;
  (global as any).__filename = opencvPath;
  (global as any).require = nodeRequire;
  (global as any).module = { exports: {} };
  (global as any).exports = (global as any).module.exports;
  (global as any).Module = ModuleObj;

  return saved;
}

/**
 * Restore original global values.
 */
function restoreGlobals(saved: {
  __dirname?: string;
  __filename?: string;
  require?: NodeRequire;
  module?: any;
  exports?: any;
}) {
  const restore = (key: string, value: any) => {
    if (value !== undefined) {
      (global as any)[key] = value;
    } else {
      delete (global as any)[key];
    }
  };

  restore('__dirname', saved.__dirname);
  restore('__filename', saved.__filename);
  restore('require', saved.require);
  restore('module', saved.module);
  restore('exports', saved.exports);
  // Don't delete Module - tests need it
}

/**
 * Find the initialized cv object from various possible locations.
 */
function findInitializedCV(cvModule: any, ModuleObj: any): typeof cv | null {
  // Check returned module
  if (cvModule && typeof cvModule === 'object' && 'Scalar' in cvModule) {
    return cvModule;
  }
  // Check Module object
  if (ModuleObj && typeof ModuleObj === 'object' && 'Scalar' in ModuleObj) {
    return ModuleObj;
  }
  // Check global Module
  const globalModule = (global as any).Module;
  if (globalModule && typeof globalModule === 'object' && 'Scalar' in globalModule) {
    return globalModule;
  }
  return null;
}

/**
 * Initialize OpenCV in browser environment.
 */
function initOpenCVBrowser(): typeof cv {
  if (!(window as any).cv) {
    throw new Error(
      'OpenCV.js not loaded. Add <script src="opencv.js"> before using this library.'
    );
  }
  return (window as any).cv;
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
