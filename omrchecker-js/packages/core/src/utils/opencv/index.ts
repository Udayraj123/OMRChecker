/**
 * Centralized OpenCV initialization and access module.
 *
 * This module provides a proper async initialization pattern for OpenCV.js
 * supporting both browser (global cv) and Node.js (local file) environments.
 *
 * Browser: Expects opencv.js to be loaded via script tag before use
 * Node.js: Loads from local lib/opencv.js file (for tests)
 */

// Global instance
let cvInstance: typeof cv | null = window.cv;


/**
 * Get the initialized OpenCV instance.
 *
 * In test environments, if cvInstance is not set but global.cv exists,
 * we use it as a fallback. This allows tests to work without calling initOpenCV()
 * explicitly, as long as global.cv is set up in the test setup.
 *
 * @throws Error if OpenCV has not been initialized and global.cv is not available
 * @returns The initialized OpenCV instance
 */
export function getCV(): typeof cv {
  // If cvInstance is already set, use it
  if (cvInstance) {
    return cvInstance;
  }

  // Fallback: Check if global.cv is available (for test environments)
  // This allows tests to work without explicitly calling initOpenCV()
  const globalCv = (global as any).cv;
  if (globalCv && typeof globalCv.Scalar === 'function') {
    // Cache it for future use
    cvInstance = globalCv as typeof cv;
    return cvInstance;
  }

  // Check window.cv as a last resort (for browser/jsdom environments)
  if (typeof window !== 'undefined' && (window as any).cv) {
    const windowCv = (window as any).cv;
    if (typeof windowCv.Scalar === 'function') {
      cvInstance = windowCv as typeof cv;
      return cvInstance;
    }
  }

  throw new Error('OpenCV not initialized. Call initOpenCV() first, or ensure global.cv is set in test environment.');
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
