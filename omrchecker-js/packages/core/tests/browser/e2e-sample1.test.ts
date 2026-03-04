/**
 * E2E browser test: Sample 1 Mobile Camera — OMR pipeline via IIFE bundle
 *
 * Runs the real samples/1-mobile-camera/MobileCamera/sheet1.jpg through the
 * complete OMR pipeline using the compiled omrchecker.iife.js bundle in a
 * real Chromium browser.
 *
 * Pipeline (executed by OMRChecker.processSingleFile):
 *   [1] CropPage       — detect sheet boundary from dark background
 *   [2] CropOnMarkers  — detect 4 corner markers, perspective-warp to align
 *   [3] BubbleReader   — detect & interpret filled bubbles
 *
 * Python expected output (test_all_samples.py::test_run_omr_marker_mobile):
 *   Medium=E, Roll=503110026
 *   q1=B, q2=-, q3=D, q4=B, q5=6, q6=11, q7=20, q8=7, q9=16
 *   q10=B, q11=D, q12=C, q13=D, q14=A, q15=D, q16=B, q17=A, q18=C, q19=C, q20=D
 *
 * bd issue: omr-b4q
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

// Allow extra time: OpenCV.js CDN load + WASM init + full pipeline
test.setTimeout(120_000);

function fileToDataUrl(filePath: string, mime: string): string {
  return `data:${mime};base64,${readFileSync(filePath).toString('base64')}`;
}

async function injectBundle(page: any): Promise<void> {
  await page.addScriptTag({ content: readFileSync(BUNDLE_PATH, 'utf8') });
}

// ── tests ─────────────────────────────────────────────────────────────────────

test.describe('E2E: Sample 1 Mobile Camera — full pipeline via bundle', () => {
  const sheetB64     = fileToDataUrl(SHEET1_PATH, 'image/jpeg');
  const markerB64    = fileToDataUrl(MARKER_PATH, 'image/jpeg');
  const templateJson = JSON.parse(readFileSync(TEMPLATE_PATH, 'utf8'));

  test.beforeEach(async ({ page }) => {
    await setupBrowser(page);
    await injectBundle(page);
  });

  // ── Stage 0: raw image sanity ─────────────────────────────────────────────
  test('stage 0 — sheet1.jpg loads as 747×1024 grayscale image', async ({ page }) => {
    const r = await page.evaluate(async ({ sheetB64 }: { sheetB64: string }) => {
      const img = new Image();
      img.src = sheetB64;
      await new Promise<void>((res, rej) => { img.onload = () => res(); img.onerror = rej; });
      const canvas = document.createElement('canvas');
      canvas.width = img.width; canvas.height = img.height;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(img, 0, 0);
      const rgba = window.cv.matFromImageData(ctx.getImageData(0, 0, img.width, img.height));
      const gray = new window.cv.Mat();
      window.cv.cvtColor(rgba, gray, window.cv.COLOR_RGBA2GRAY);
      rgba.delete();
      try {
        return {
          rows: gray.rows,
          cols: gray.cols,
          channels: gray.channels(),
          nonZero: window.cv.countNonZero(gray),
          mean: window.cv.mean(gray)[0],
        };
      } finally { gray.delete(); }
    }, { sheetB64 });

    expect(r.rows).toBe(1024);
    expect(r.cols).toBe(747);
    expect(r.channels).toBe(1);
    // Photo includes dark background — still >10% non-zero pixels
    expect(r.nonZero).toBeGreaterThan(747 * 1024 * 0.1);
    // Mean should be mid-range (not all black, not all white)
    expect(r.mean).toBeGreaterThan(20);
    expect(r.mean).toBeLessThan(200);
  });

  // ── Full pipeline: all field labels present ───────────────────────────────
  test('full pipeline — processSingleFile returns all expected field labels', async ({ page }) => {
    const result = await page.evaluate(
      async ({ sheetB64, markerB64, templateJson }: any) => {
        const { OMRChecker } = (window as any).OMRChecker;
        try {
          const omrResult = await OMRChecker.processSingleFile({
            imageBase64: sheetB64,
            templateJson,
            assets: { 'omr_marker.jpg': markerB64 },
          });
          return { ok: true, fieldLabels: Object.keys(omrResult.response).sort(), dims: omrResult.processedImageDimensions };
        } catch (err: any) {
          return { ok: false, error: String(err?.message ?? err) };
        }
      },
      { sheetB64, markerB64, templateJson }
    );

    expect(result.ok, result.ok ? '' : `Pipeline failed: ${(result as any).error}`).toBe(true);

    // "Roll" customLabel → roll1..roll9 individual labels
    // "q5".."q9" customLabels → q5_1,q5_2 .. q9_1,q9_2 two-digit fields
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

    // Final image is resized to templateDimensions [1846, 1500] (width, height)
    const dims = (result as any).dims as [number, number];
    expect(dims[0], 'width').toBe(1846);
    expect(dims[1], 'height').toBe(1500);
  });

  // ── Full pipeline: Python snapshot values ─────────────────────────────────
  test('full pipeline — processSingleFile matches Python snapshot values', async ({ page }) => {
    const result = await page.evaluate(
      async ({ sheetB64, markerB64, templateJson }: any) => {
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

    const r = (result as any).response as Record<string, string>;

    // Python snapshot (test_all_samples.py::test_run_omr_marker_mobile):
    //   Medium=E
    expect(r['Medium'], 'Medium').toBe('E');

    // Roll=503110026 → roll1='5', roll2='0', roll3='3', ..., roll9='6'
    expect(r['roll1'], 'roll1').toBe('5');
    expect(r['roll2'], 'roll2').toBe('0');
    expect(r['roll3'], 'roll3').toBe('3');
    expect(r['roll4'], 'roll4').toBe('1');
    expect(r['roll5'], 'roll5').toBe('1');
    expect(r['roll6'], 'roll6').toBe('0');
    expect(r['roll7'], 'roll7').toBe('0');
    expect(r['roll8'], 'roll8').toBe('2');
    expect(r['roll9'], 'roll9').toBe('6');

    // MCQ answers (q1–q4, q10–q20)
    expect(r['q1'],  'q1').toBe('B');
    expect(r['q2'],  'q2').toBe('');  // blank — TS uses '' (empty_value default), Python CSV shows '-'
    expect(r['q3'],  'q3').toBe('D');
    expect(r['q4'],  'q4').toBe('B');
    expect(r['q10'], 'q10').toBe('B');
    expect(r['q11'], 'q11').toBe('D');
    expect(r['q12'], 'q12').toBe('C');
    expect(r['q13'], 'q13').toBe('D');
    expect(r['q14'], 'q14').toBe('A');
    expect(r['q15'], 'q15').toBe('D');
    expect(r['q16'], 'q16').toBe('B');
    expect(r['q17'], 'q17').toBe('A');
    expect(r['q18'], 'q18').toBe('C');
    expect(r['q19'], 'q19').toBe('C');
    expect(r['q20'], 'q20').toBe('D');

    // Two-digit answers: q5=6, q6=11, q7=20, q8=7, q9=16
    // Stored as individual digit fields: q5_1, q5_2
    expect(r['q5_1'], 'q5_1').toBe('6');
    expect(r['q5_2'], 'q5_2').toBe('');  // single digit → second bubble is blank
    expect(r['q6_1'], 'q6_1').toBe('1');
    expect(r['q6_2'], 'q6_2').toBe('1');
    expect(r['q7_1'], 'q7_1').toBe('2');
    expect(r['q7_2'], 'q7_2').toBe('0');
    expect(r['q8_1'], 'q8_1').toBe('7');
    expect(r['q8_2'], 'q8_2').toBe('');  // single digit → second bubble is blank
    expect(r['q9_1'], 'q9_1').toBe('1');
    expect(r['q9_2'], 'q9_2').toBe('6');
  });
});
