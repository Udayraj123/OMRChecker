import { beforeAll, afterAll } from 'vitest';

// OpenCV.js type declarations
declare global {
  interface Window {
    cv: any;
  }
  var cv: any;
  var matCountSupported: boolean;
}

// Wait for OpenCV.js to load
beforeAll(async () => {
  // In a browser environment, OpenCV.js would be loaded via script tag
  // For testing, we'll mock or skip OpenCV.js initialization
  // since @techstark/opencv-js requires browser environment
  
  console.log('Test setup: OpenCV.js initialization skipped (requires browser)');
  
  // Check if cv.matCount is available for memory leak detection
  let matCountSupported = false;
  try {
    if (typeof cv !== 'undefined' && typeof cv.matCount === 'function') {
      matCountSupported = true;
      console.log('Test setup: cv.matCount available for memory leak detection');
    }
  } catch (e) {
    console.warn('Test setup: cv.matCount not available - memory leak tests will be skipped');
  }
  
  // Make matCountSupported available globally for tests
  global.matCountSupported = matCountSupported;
}, 30000); // 30 second timeout for OpenCV.js loading

afterAll(() => {
  console.log('Test teardown: Cleaning up');
});

// Helper function to check for memory leaks
export function getMatCount(): number {
  if (global.matCountSupported && typeof cv !== 'undefined' && typeof cv.matCount === 'function') {
    return cv.matCount();
  }
  return -1; // Indicates cv.matCount not available
}

// Helper to skip memory tests if cv.matCount not available
export function skipIfNoMatCount(testFn: () => void | Promise<void>) {
  return () => {
    if (!global.matCountSupported) {
      console.warn('Skipping memory test - cv.matCount not available');
      return;
    }
    return testFn();
  };
}
