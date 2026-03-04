import { test, expect } from '@playwright/test';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { setupBrowser } from './browser-setup';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

test.setTimeout(60_000);

test('IIFE bundle exposes OMRChecker on window', async ({ page }) => {
  await setupBrowser(page);

  // Inject the compiled IIFE bundle
  const bundlePath = join(__dirname, '../../dist/omrchecker.iife.js');
  const bundleCode = readFileSync(bundlePath, 'utf8');
  await page.addScriptTag({ content: bundleCode });

  const result = await page.evaluate(() => {
    const w = window as any;
    // The IIFE globalName 'OMRChecker' holds all module exports as a namespace.
    // The OMRChecker class is a named export, accessible as w.OMRChecker.OMRChecker.
    // processSingleFile is a static method on the OMRChecker class.
    const ns = w.OMRChecker;
    const OMRCheckerClass = ns?.OMRChecker;
    return {
      hasOMRChecker: typeof ns !== 'undefined',
      hasOMRCheckerClass: typeof OMRCheckerClass === 'function',
      hasProcessSingleFile: typeof OMRCheckerClass?.processSingleFile === 'function',
      hasTemplate: typeof ns?.Template !== 'undefined' || true, // may or may not be exported
    };
  });

  expect(result.hasOMRChecker).toBe(true);
  expect(result.hasOMRCheckerClass).toBe(true);
  expect(result.hasProcessSingleFile).toBe(true);
});
