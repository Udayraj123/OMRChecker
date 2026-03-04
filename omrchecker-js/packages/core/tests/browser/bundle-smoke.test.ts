/**
 * Browser smoke test for the compiled IIFE bundle (dist/omrchecker.iife.js).
 *
 * Validates the public API surface exposed on window.OMRChecker after the
 * bundle is injected into a real Chromium browser.
 *
 * Full pipeline coverage (field labels + Python snapshot values) lives in
 * tests/browser/e2e-sample1.test.ts to avoid duplication.
 *
 * Run:
 *   cd omrchecker-js/packages/core
 *   npx playwright test tests/browser/bundle-smoke.test.ts --reporter=list
 */

import { test, expect } from '@playwright/test';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { setupBrowser } from './browser-setup';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const BUNDLE_PATH = join(__dirname, '../../dist/omrchecker.iife.js');

async function injectBundle(page: any): Promise<void> {
  await page.addScriptTag({ content: readFileSync(BUNDLE_PATH, 'utf8') });
}

// ── tests ─────────────────────────────────────────────────────────────────────

test('IIFE bundle exposes OMRChecker API on window', async ({ page }) => {
  test.setTimeout(60_000);
  await setupBrowser(page);
  await injectBundle(page);

  const result = await page.evaluate(() => {
    const w = window as any;
    // globalName:'OMRChecker' → window.OMRChecker is the module namespace.
    // The OMRChecker class is a named export: window.OMRChecker.OMRChecker.
    const ns  = w.OMRChecker;
    const cls = ns?.OMRChecker;
    return {
      hasNamespace:         typeof ns  === 'object',
      hasClass:             typeof cls === 'function',
      hasProcessSingleFile: typeof cls?.processSingleFile === 'function',
      hasTemplate:          typeof ns?.Template === 'function',
      hasBubbleReader:      typeof ns?.BubbleReader === 'function',
      hasCropPage:          typeof ns?.CropPage === 'function',
      hasCropOnMarkers:     typeof ns?.CropOnMarkers === 'function',
    };
  });

  expect(result.hasNamespace,         'window.OMRChecker namespace').toBe(true);
  expect(result.hasClass,             'OMRChecker class').toBe(true);
  expect(result.hasProcessSingleFile, 'processSingleFile static method').toBe(true);
  expect(result.hasTemplate,          'Template class exported').toBe(true);
  expect(result.hasBubbleReader,      'BubbleReader class exported').toBe(true);
  expect(result.hasCropPage,          'CropPage class exported').toBe(true);
  expect(result.hasCropOnMarkers,     'CropOnMarkers class exported').toBe(true);
});

// Full pipeline coverage (field labels + Python snapshot) is in e2e-sample1.test.ts.
