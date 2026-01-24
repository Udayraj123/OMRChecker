// https://github.com/TechStark/opencv-js/blob/main/test/cv.ts

import { readFileSync } from 'fs';
import { resolve } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

// Setup OpenCV.js for tests using jsdom (browser-like environment)

// Load opencv.js file
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const opencvPath = resolve(__dirname, '../../lib/opencv.js');
const opencvCode = readFileSync(opencvPath, 'utf-8');

// Set up Module with onRuntimeInitialized callback BEFORE loading opencv.js
let initResolve: () => void;
const initPromise = new Promise<void>((resolve) => {
  initResolve = resolve;
});

const window = globalThis as any;

// Ensure window.window exists (jsdom should provide this)
if (!window.window) {
  window.window = window;
}

// Remove module and require to force browser path in UMD wrapper
// The UMD checks module.exports first, so we need to hide it
const originalModule = window.module;
const originalRequire = window.require;
delete window.module;
delete window.require;

// Set up Module - opencv.js will use this
window.Module = {
  onRuntimeInitialized: () => {
    initResolve();
  },
  // Add other properties opencv.js might need
  locateFile: (path: string) => {
    // Return path as-is for jsdom
    return path;
  },
};

// Module set up, loading opencv.js...

// Patch the code to work in jsdom:
// 1. Replace UMD wrapper's 'this' with 'window'
// 2. Make require available as a stub to prevent errors
const patchedCode = opencvCode
  // Replace UMD wrapper pattern: }(this, function () { with window
  .replace(/}\(this,\s*function\s*\(\)\s*\{/g, '}(window, function () {')
  // Also handle the case where 'this' is used alone in the IIFE
  .replace(/\(function\s*\(root,\s*factory\)\s*\{[\s\S]*?\}\(this,/g, (match) => {
    return match.replace(/\(this,/, '(window,');
  });

// Provide a stub require that returns an empty module
// This prevents errors when opencv.js checks for require
window.require = (() => {
  const stubModule = { exports: {} };
  const requireFn = (_id: string) => {
    // Return empty module for any require call
    return stubModule.exports;
  };
  requireFn.resolve = () => '';
  return requireFn;
})();

// Execute directly in window context using eval
try {
  window.eval(patchedCode);
} catch (error) {
  // Fallback: try as script element
  const script = window.document.createElement('script');
  script.textContent = patchedCode;
  window.document.head.appendChild(script);
}

// Restore module/require if they existed
if (originalModule !== undefined) {
  window.module = originalModule;
}
if (originalRequire !== undefined) {
  window.require = originalRequire;
}

// Wait for initialization with timeout
let initialized = false;

await Promise.race([
  initPromise.then(() => {
    initialized = true;
  }),
  new Promise<void>((resolve) => {
    // Check if cv is already available (polling)
    let attempts = 0;
    const maxAttempts = 600; // 30 seconds at 50ms intervals
    const checkInterval = setInterval(() => {
      attempts++;
      if (window.cv && typeof window.cv.Scalar === 'function') {
        clearInterval(checkInterval);
        initResolve();
        initialized = true;
        resolve();
      }

      if (attempts >= maxAttempts) {
        clearInterval(checkInterval);
        throw new Error('OpenCV.js initialization timeout');
      }
    }, 50);
  }),
]);

if (!initialized) {
  throw new Error('OpenCV initialization did not complete');
}

// Ensure cv is available
if (!window.cv || typeof window.cv.Scalar !== 'function') {
  throw new Error('OpenCV.js failed to initialize - cv object not properly initialized');
}

// Make cv available on globalThis for tests
(globalThis as any).cv = window.cv;