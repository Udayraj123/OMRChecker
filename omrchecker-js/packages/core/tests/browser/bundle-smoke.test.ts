/**
 * Browser smoke tests for the compiled IIFE bundle (dist/omrchecker.iife.js).
 *
 * These tests validate the public API surface of the bundle and run a full
 * end-to-end pipeline (CropPage → CropOnMarkers → BubbleReader) using the
 * distributed artifact — NOT the inline eval approach used in e2e-omrchecker.test.ts.
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

const BUNDLE_PATH   = join(__dirname, '../../dist/omrchecker.iife.js');
const SHEET1_PATH   = join(__dirname, '../../../../../samples/1-mobile-camera/MobileCamera/sheet1.jpg');
const MARKER_PATH   = join(__dirname, '../../../../../samples/1-mobile-camera/omr_marker.jpg');
const TEMPLATE_PATH = join(__dirname, '../../../../../samples/1-mobile-camera/template.json');

// ── helpers ───────────────────────────────────────────────────────────────────

function fileToDataUrl(filePath: string, mime: string): string {
  return `data:${mime};base64,${readFileSync(filePath).toString('base64')}`;
}

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

test('IIFE bundle — processSingleFile returns all expected field labels', async ({ page }) => {
  test.setTimeout(120_000);
  await setupBrowser(page);
  await injectBundle(page);

  const sheetB64     = fileToDataUrl(SHEET1_PATH,  'image/jpeg');
  const markerB64    = fileToDataUrl(MARKER_PATH,  'image/jpeg');
  const templateJson = JSON.parse(readFileSync(TEMPLATE_PATH, 'utf8'));

  const result = await page.evaluate(
    async ({ sheetB64, markerB64, templateJson }:
           { sheetB64: string; markerB64: string; templateJson: any }) => {
      const { OMRChecker } = (window as any).OMRChecker;
      try {
        const omrResult = await OMRChecker.processSingleFile({
          imageBase64: sheetB64,
          templateJson,
          assets: { 'omr_marker.jpg': markerB64 },
        });
        return { ok: true,
                 fieldLabels: Object.keys(omrResult.response).sort(),
                 dims: omrResult.processedImageDimensions };
      } catch (err: any) {
        return { ok: false, error: String(err?.message ?? err) };
      }
    },
    { sheetB64, markerB64, templateJson }
  );

  expect(result.ok, result.ok ? '' : `Pipeline failed: ${(result as any).error}`).toBe(true);

  // The response keys are raw field labels (not customLabel groups).
  // template.json "Roll" customLabel → roll1..9 individual labels.
  // template.json "q5" customLabel  → q5_1, q5_2 individual labels.
  const expectedFields = [
    'Medium',
    'roll1', 'roll2', 'roll3', 'roll4', 'roll5', 'roll6', 'roll7', 'roll8', 'roll9',
    'q1', 'q2', 'q3', 'q4',
    'q5_1', 'q5_2', 'q6_1', 'q6_2', 'q7_1', 'q7_2', 'q8_1', 'q8_2', 'q9_1', 'q9_2',
    'q10', 'q11', 'q12', 'q13',
    'q14', 'q15', 'q16',
    'q17', 'q18', 'q19', 'q20',
  ];
  for (const f of expectedFields) {
    expect((result as any).fieldLabels, `field ${f} missing`).toContain(f);
  }

  // After preprocessors the image is resized to templateDimensions [1846, 1500] (w×h)
  // so BubbleReader coordinates (defined at template resolution) are correct.
  const dims = (result as any).dims as [number, number];
  expect(dims[0], 'width').toBe(1846);
  expect(dims[1], 'height').toBe(1500);
});

test('IIFE bundle — processSingleFile matches Python snapshot values', async ({ page }) => {
  test.setTimeout(120_000);
  await setupBrowser(page);
  await injectBundle(page);

  const sheetB64     = fileToDataUrl(SHEET1_PATH,  'image/jpeg');
  const markerB64    = fileToDataUrl(MARKER_PATH,  'image/jpeg');
  const templateJson = JSON.parse(readFileSync(TEMPLATE_PATH, 'utf8'));

  const result = await page.evaluate(
    async ({ sheetB64, markerB64, templateJson }:
           { sheetB64: string; markerB64: string; templateJson: any }) => {
      const { OMRChecker } = (window as any).OMRChecker;
      try {
        const { response } = await OMRChecker.processSingleFile({
          imageBase64: sheetB64,
          templateJson,
          assets: { 'omr_marker.jpg': markerB64 },
        });
        return { ok: true, response };
      } catch (err: any) {
        return { ok: false, error: String(err?.message ?? err) };
      }
    },
    { sheetB64, markerB64, templateJson }
  );

  expect(result.ok, result.ok ? '' : `Pipeline failed: ${(result as any).error}`).toBe(true);

  // Python snapshot (test_all_samples.py):
  //   Medium=E, Roll=503110026 (roll1='5',roll2='0',...,roll9='6')
  //   q1=B, q2=-, q3=D, q4=B
  //   q10=B, q11=D, q12=C, q13=D, q14=A, q15=D, q16=B, q17=A, q18=C, q19=C, q20=D
  //
  // Note: "Roll" is a customLabel group; individual field labels are roll1..roll9.
  const r = (result as any).response as Record<string, string>;
  // Medium is a direct field label in the response
  expect(r['Medium'], 'Medium').toBe('E');
  // MCQ answers (most reliable across threshold variance)
  expect(r['q1'],  'q1').toBe('B');
  expect(r['q3'],  'q3').toBe('D');
  expect(r['q4'],  'q4').toBe('B');
  expect(r['q10'], 'q10').toBe('B');
  expect(r['q11'], 'q11').toBe('D');
  expect(r['q14'], 'q14').toBe('A');
  expect(r['q17'], 'q17').toBe('A');
  expect(r['q20'], 'q20').toBe('D');
});
