// build-browser.mjs — esbuild script for browser IIFE bundle
import { build } from 'esbuild';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

await build({
  entryPoints: ['src/index.ts'],
  bundle: true,
  format: 'iife',
  globalName: 'OMRChecker',
  platform: 'browser',
  target: ['es2020', 'chrome80', 'firefox75', 'safari13'],
  alias: {
    '@techstark/opencv-js': path.join(__dirname, 'src/cv-shim.ts'),
  },
  outfile: 'dist/omrchecker.iife.js',
  minify: false,   // keep readable for debugging
  sourcemap: true,
  treeShaking: true,
  logLevel: 'info',
});

console.log('✓ IIFE bundle written to dist/omrchecker.iife.js');
