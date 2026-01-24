// https://github.com/TechStark/opencv-js/blob/main/test/cv.ts

import { readFileSync } from 'fs';
import { resolve } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { initOpenCV } from '../utils/opencv';

console.log('[OpenCV Setup] Module imports completed');
console.log('[OpenCV Setup] initOpenCV imported:', typeof initOpenCV);

// Setup OpenCV.js for tests using jsdom (browser-like environment)

console.log('[OpenCV Setup] Starting OpenCV initialization...');

// Load opencv.js file
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const opencvPath = resolve(__dirname, '../../lib/opencv.js');
console.log('[OpenCV Setup] Loading opencv.js from:', opencvPath);
const opencvCode = readFileSync(opencvPath, 'utf-8');
console.log('[OpenCV Setup] Loaded opencv.js, size:', opencvCode.length, 'bytes');

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
console.log('[OpenCV Setup] Executing opencv.js code...');
try {
  window.eval(patchedCode);
  console.log('[OpenCV Setup] opencv.js code executed successfully');
} catch (error) {
  console.error('[OpenCV Setup] Error executing opencv.js with eval:', error);
  // Fallback: try as script element
  try {
    const script = window.document.createElement('script');
    script.textContent = patchedCode;
    // Add error handler to script
    script.onerror = (err) => {
      console.error('[OpenCV Setup] Error loading opencv.js as script element:', err);
    };
    window.document.head.appendChild(script);
    console.log('[OpenCV Setup] opencv.js loaded as script element');
  } catch (fallbackError) {
    console.error('[OpenCV Setup] Fallback script execution also failed:', fallbackError);
    throw new Error(`Failed to execute opencv.js: ${error instanceof Error ? error.message : String(error)}`);
  }
}

// Restore module/require if they existed
if (originalModule !== undefined) {
  window.module = originalModule;
}
if (originalRequire !== undefined) {
  window.require = originalRequire;
}

// Wait for initialization with timeout
console.log('[OpenCV Setup] Waiting for OpenCV initialization...');
let initialized = false;

// Use a simpler approach: just poll for cv object
// Also listen for onRuntimeInitialized callback, but don't block on it
initPromise.then(() => {
  console.log('[OpenCV Setup] onRuntimeInitialized callback called');
  if (!initialized) {
    initialized = true;
  }
}).catch((err) => {
  console.warn('[OpenCV Setup] initPromise rejected (this is OK if polling succeeds):', err);
});

// Poll for cv object with timeout using recursive promise-based approach
// This avoids potential issues with setInterval and resolve()
const pollForCV = async (): Promise<void> => {
  const maxAttempts = 600; // 30 seconds at 50ms intervals
  const startTime = Date.now();
  const timeout = 30000; // 30 seconds

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    // Check if cv is ready
    if (window.cv && typeof window.cv.Scalar === 'function') {
      const elapsed = Date.now() - startTime;
      console.log(`[OpenCV Setup] Found cv object after ${attempt + 1} attempts (${elapsed}ms)`);
      initialized = true;
      return;
    }

    // Log progress every 100 attempts (5 seconds)
    if (attempt > 0 && attempt % 100 === 0) {
      const elapsed = Date.now() - startTime;
      console.log(`[OpenCV Setup] Still waiting for cv object... (attempt ${attempt}/${maxAttempts}, ${elapsed}ms elapsed)`);
    }

    // Check timeout
    if (Date.now() - startTime > timeout) {
      throw new Error('OpenCV.js initialization timeout - cv object not found after 30 seconds');
    }

    // Wait 50ms before next check
    await new Promise(resolve => setTimeout(resolve, 50));
  }

  throw new Error('OpenCV.js initialization timeout - cv object not found after max attempts');
};

await pollForCV();

console.log('[OpenCV Setup] Polling completed, initialized flag:', initialized);

if (!initialized) {
  throw new Error('OpenCV initialization did not complete');
}

console.log('[OpenCV Setup] Initialization promise resolved');

// Ensure cv is available
if (!window.cv || typeof window.cv.Scalar !== 'function') {
  throw new Error('OpenCV.js failed to initialize - cv object not properly initialized');
}

console.log('[OpenCV Setup] cv object verified, making available on globalThis and global');

// Make cv available on both globalThis and global for tests
// This ensures initOpenCV() can find it
(globalThis as any).cv = window.cv;
(global as any).cv = window.cv;
console.log('[OpenCV Setup] cv set on both globalThis and global');
console.log('[OpenCV Setup] Verifying global.cv:', {
  exists: !!(global as any).cv,
  hasScalar: !!(global as any).cv && typeof (global as any).cv.Scalar === 'function',
});

console.log('[OpenCV Setup] Setup complete!');