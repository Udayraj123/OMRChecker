// cv-shim.ts — re-exports window.cv as the default export
// This is used by esbuild to alias @techstark/opencv-js → cv-shim.ts
// so the browser bundle uses the OpenCV.js already loaded from CDN.
export default (globalThis as any).cv as typeof import('@techstark/opencv-js').default;
