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
  console.log('[initOpenCV] Starting initialization...');

  if (cvInstance) {
    console.log('[initOpenCV] cvInstance already exists, returning it');
    return cvInstance;
  }

  console.log('[initOpenCV] Checking environment...');
  console.log('[initOpenCV] typeof window:', typeof window);
  console.log('[initOpenCV] window exists:', typeof window !== 'undefined');
  console.log('[initOpenCV] global.cv exists:', !!(global as any).cv);
  console.log('[initOpenCV] window.cv exists:', typeof window !== 'undefined' && !!(window as any).cv);

  // In jsdom (test environment), window is defined but we should check for global.cv first
  // If global.cv exists, we're in Node.js test environment
  const globalCv = (global as any).cv;
  if (globalCv && typeof globalCv.Scalar === 'function') {
    console.log('[initOpenCV] Found global.cv, using Node.js path...');
    cvInstance = globalCv;
    console.log('[initOpenCV] Using global.cv directly');
  } else if (typeof window === 'undefined') {
    // Node.js environment (tests)
    console.log('[initOpenCV] Node.js environment detected (no window), calling initOpenCVNode()...');
    cvInstance = await initOpenCVNode();
    console.log('[initOpenCV] initOpenCVNode() completed');
  } else {
    // Browser environment (or jsdom)
    console.log('[initOpenCV] Browser/jsdom environment detected, calling initOpenCVBrowser()...');
    cvInstance = initOpenCVBrowser();
    console.log('[initOpenCV] initOpenCVBrowser() completed');
  }

  console.log('[initOpenCV] Initialization complete, cvInstance:', cvInstance ? 'found' : 'null');
  return cvInstance;
}

/**
 * Initialize OpenCV in Node.js environment.
 * First checks if global.cv is available (from setup.ts), otherwise loads from file.
 */
async function initOpenCVNode(): Promise<typeof cv> {
  console.log('[initOpenCVNode] Starting Node.js initialization...');
  // Check if cv is already available from setup.ts
  const globalCv = (global as any).cv;
  console.log('[initOpenCVNode] Checking global.cv:', globalCv ? 'exists' : 'not found');
  if (globalCv && typeof globalCv.Scalar === 'function') {
    console.log('[initOpenCVNode] Found global.cv with Scalar, using it');
    return globalCv;
  }

  // Load from local file
  console.log('[initOpenCVNode] global.cv not available, loading from file...');
  const cv = await loadOpenCVFromFile();
  console.log('[initOpenCVNode] loadOpenCVFromFile() completed');
  return cv;
}

/**
 * Load OpenCV.js from local file in Node.js environment.
 * Based on official OpenCV.js Node.js pattern:
 * https://docs.opencv.org/4.x/dc/de6/tutorial_js_nodejs.html
 */
async function loadOpenCVFromFile(): Promise<typeof cv> {
  console.log('[loadOpenCVFromFile] Starting to load OpenCV from file...');
  const opencvPath = resolve(__dirname, '../../../lib/opencv.js');
  const opencvDir = dirname(opencvPath);
  console.log('[loadOpenCVFromFile] opencvPath:', opencvPath);
  console.log('[loadOpenCVFromFile] opencvDir:', opencvDir);

  const nodeRequire = createRequire(import.meta.url);
  console.log('[loadOpenCVFromFile] Created nodeRequire');

  // Set up initialization promise
  let initResolve: () => void;
  const initPromise = new Promise<void>((resolve) => {
    initResolve = resolve;
    console.log('[loadOpenCVFromFile] Created initPromise');
  });

  // Create Module object with onRuntimeInitialized callback
  // This MUST be set up BEFORE requiring opencv.js
  console.log('[loadOpenCVFromFile] Creating Module object...');
  const ModuleObj: any = {
    onRuntimeInitialized: () => {
      console.log('[loadOpenCVFromFile] onRuntimeInitialized callback called!');
      initResolve();
    },
    locateFile: (path: string) => {
      const resolved = resolve(opencvDir, path);
      console.log('[loadOpenCVFromFile] locateFile called with path:', path, '-> resolved to:', resolved);
      return resolved;
    },
  };
  console.log('[loadOpenCVFromFile] Module object created');

  // Save and set up CommonJS environment
  console.log('[loadOpenCVFromFile] Saving and setting up globals...');
  const savedGlobals = saveAndSetupGlobals(opencvDir, opencvPath, nodeRequire, ModuleObj);
  console.log('[loadOpenCVFromFile] Globals set up, saved globals:', Object.keys(savedGlobals));

  try {
    // Load opencv.js - it will use the global Module object
    console.log('[loadOpenCVFromFile] About to require opencv.js...');
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const cvModule = nodeRequire(opencvPath);
    console.log('[loadOpenCVFromFile] opencv.js required, cvModule:', typeof cvModule, Object.keys(cvModule || {}));

    // Some versions require explicit run() call
    console.log('[loadOpenCVFromFile] Checking if ModuleObj.run exists...');
    if (ModuleObj.run && typeof ModuleObj.run === 'function' && !ModuleObj.calledRun) {
      console.log('[loadOpenCVFromFile] Calling ModuleObj.run()...');
      ModuleObj.run();
      console.log('[loadOpenCVFromFile] ModuleObj.run() completed');
    } else {
      console.log('[loadOpenCVFromFile] ModuleObj.run not available or already called');
    }

    // Wait for WASM initialization with timeout
    console.log('[loadOpenCVFromFile] Waiting for WASM initialization (Promise.race)...');
    await Promise.race([
      initPromise.then(() => {
        console.log('[loadOpenCVFromFile] initPromise resolved!');
      }),
      new Promise((_, reject) => {
        console.log('[loadOpenCVFromFile] Setting up timeout (10s)...');
        setTimeout(() => {
          console.log('[loadOpenCVFromFile] Timeout reached, rejecting...');
          reject(new Error('OpenCV initialization timeout after 10s'));
        }, 10000);
      }),
    ]);
    console.log('[loadOpenCVFromFile] Promise.race completed');

    // Find the initialized cv object
    console.log('[loadOpenCVFromFile] Finding initialized cv object...');
    const cv = findInitializedCV(cvModule, ModuleObj);
    console.log('[loadOpenCVFromFile] findInitializedCV returned:', cv ? 'found' : 'null');
    if (!cv || typeof (cv as any).Scalar !== 'function') {
      console.error('[loadOpenCVFromFile] cv object validation failed');
      throw new Error('OpenCV.js not properly initialized - Scalar constructor not found');
    }
    console.log('[loadOpenCVFromFile] cv object validated successfully');

    return cv;
  } catch (error) {
    console.error('[loadOpenCVFromFile] Error occurred:', error);
    throw error;
  } finally {
    console.log('[loadOpenCVFromFile] Restoring globals...');
    restoreGlobals(savedGlobals);
    console.log('[loadOpenCVFromFile] Globals restored');
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
  console.log('[saveAndSetupGlobals] Saving current global values...');
  const saved = {
    __dirname: (global as any).__dirname,
    __filename: (global as any).__filename,
    require: (global as any).require,
    module: (global as any).module,
    exports: (global as any).exports,
  };
  console.log('[saveAndSetupGlobals] Saved globals:', {
    has__dirname: !!saved.__dirname,
    has__filename: !!saved.__filename,
    hasRequire: !!saved.require,
    hasModule: !!saved.module,
    hasExports: !!saved.exports,
  });

  // Set up CommonJS environment
  console.log('[saveAndSetupGlobals] Setting up CommonJS environment...');
  (global as any).__dirname = opencvDir;
  (global as any).__filename = opencvPath;
  (global as any).require = nodeRequire;
  (global as any).module = { exports: {} };
  (global as any).exports = (global as any).module.exports;
  (global as any).Module = ModuleObj;
  console.log('[saveAndSetupGlobals] CommonJS environment set up, global.Module:', typeof (global as any).Module);

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
  console.log('[restoreGlobals] Restoring global values...');
  const restore = (key: string, value: any) => {
    if (value !== undefined) {
      (global as any)[key] = value;
      console.log(`[restoreGlobals] Restored ${key}`);
    } else {
      delete (global as any)[key];
      console.log(`[restoreGlobals] Deleted ${key}`);
    }
  };

  restore('__dirname', saved.__dirname);
  restore('__filename', saved.__filename);
  restore('require', saved.require);
  restore('module', saved.module);
  restore('exports', saved.exports);
  // Don't delete Module - tests need it
  console.log('[restoreGlobals] Globals restored (Module kept for tests)');
}

/**
 * Find the initialized cv object from various possible locations.
 */
function findInitializedCV(cvModule: any, ModuleObj: any): typeof cv | null {
  console.log('[findInitializedCV] Searching for cv object...');
  console.log('[findInitializedCV] cvModule type:', typeof cvModule, cvModule ? Object.keys(cvModule).slice(0, 10) : 'null');
  console.log('[findInitializedCV] ModuleObj type:', typeof ModuleObj, ModuleObj ? Object.keys(ModuleObj).slice(0, 10) : 'null');

  // Check returned module
  if (cvModule && typeof cvModule === 'object' && 'Scalar' in cvModule) {
    console.log('[findInitializedCV] Found cv in cvModule');
    return cvModule;
  }
  // Check Module object
  if (ModuleObj && typeof ModuleObj === 'object' && 'Scalar' in ModuleObj) {
    console.log('[findInitializedCV] Found cv in ModuleObj');
    return ModuleObj;
  }
  // Check global Module
  const globalModule = (global as any).Module;
  console.log('[findInitializedCV] globalModule type:', typeof globalModule, globalModule ? Object.keys(globalModule).slice(0, 10) : 'null');
  if (globalModule && typeof globalModule === 'object' && 'Scalar' in globalModule) {
    console.log('[findInitializedCV] Found cv in globalModule');
    return globalModule;
  }
  console.log('[findInitializedCV] cv object not found in any location');
  return null;
}

/**
 * Initialize OpenCV in browser environment.
 */
function initOpenCVBrowser(): typeof cv {
  console.log('[initOpenCVBrowser] Starting browser initialization...');
  console.log('[initOpenCVBrowser] window.cv exists:', !!(window as any).cv);
  if (!(window as any).cv) {
    console.error('[initOpenCVBrowser] window.cv not found!');
    throw new Error(
      'OpenCV.js not loaded. Add <script src="opencv.js"> before using this library.'
    );
  }
  console.log('[initOpenCVBrowser] window.cv found, checking Scalar...');
  const cv = (window as any).cv;
  if (typeof cv.Scalar !== 'function') {
    console.error('[initOpenCVBrowser] window.cv.Scalar is not a function!');
    throw new Error('OpenCV.js not properly initialized - Scalar constructor not found');
  }
  console.log('[initOpenCVBrowser] window.cv validated successfully');
  return cv;
}

/**
 * Get the initialized OpenCV instance.
 *
 * @throws Error if OpenCV has not been initialized yet
 * @returns The initialized OpenCV instance
 */
export function getCV(): typeof cv {
  cvInstance = cvInstance || (global as any).cv; // temp
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
