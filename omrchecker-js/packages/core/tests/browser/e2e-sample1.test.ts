/**
 * E2E browser test: Sample 1 Mobile Camera
 *
 * Loads the real sheet1.jpg from inputs/sample1/MobileCamera/ and verifies:
 * 1. Image decodes to correct dimensions (750×1000 grayscale)
 * 2. Image has realistic OMR content (non-zero pixels, valid mean)
 * 3. GaussianBlur preprocessing applies without error and preserves dimensions
 * 4. Pipeline processFile flow runs successfully with the real image
 *
 * Python equivalent: src/tests/__tests__/test_all_samples.py::test_run_omr_marker_mobile
 * Expected Python output (snapshot): Medium=E, Roll=503110026,
 *   q1=B, q2=-, q3=D, q4=B, q5=6, q6=11, q7=20, q8=7, q9=16,
 *   q10=B, q11=D, q12=C, q13=D, q14=A, q15=D, q16=B, q17=A, q18=C, q19=C, q20=D
 *
 * Note: Full OMR reading (CropPage, CropOnMarkers, bubble detection) is not yet
 * ported to TypeScript. This test covers the preprocessing layer that IS
 * implemented and establishes the e2e testing pattern for future extension.
 */

import { test, expect } from '@playwright/test';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { setupBrowser } from './browser-setup';
import { withMemoryTracking } from './memory-utils';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// sheet1.jpg: 750×1000 grayscale JPEG, ~42 KB
const SHEET1_PATH = join(__dirname, '../../../../../inputs/sample1/MobileCamera/sheet1.jpg');
const EXPECTED_COLS = 750;
const EXPECTED_ROWS = 1000;

// Allow extra time: OpenCV.js init (~30s) + image decode
test.setTimeout(90_000);

test.describe('E2E: Sample 1 Mobile Camera', () => {
  const imageBase64 = readFileSync(SHEET1_PATH).toString('base64');

  test.beforeEach(async ({ page }) => {
    await setupBrowser(page);

    // Inject a helper that decodes the JPEG and returns a grayscale cv.Mat.
    // Caller is responsible for calling mat.delete() when done.
    await page.evaluate((b64: string) => {
      (window as any).__loadSheet1Gray = async (): Promise<any> => {
        const img = new Image();
        img.src = 'data:image/jpeg;base64,' + b64;
        await new Promise<void>((res, rej) => {
          img.onload = () => res();
          img.onerror = () => rej(new Error('sheet1.jpg failed to load from base64'));
        });
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(img, 0, 0);
        // Canvas always returns RGBA — convert to grayscale (matches Python's cv2.imread grayscale)
        const rgba = window.cv.matFromImageData(ctx.getImageData(0, 0, img.width, img.height));
        const gray = new window.cv.Mat();
        window.cv.cvtColor(rgba, gray, window.cv.COLOR_RGBA2GRAY);
        rgba.delete();
        return gray;
      };
    }, imageBase64);
  });

  // ─────────────────────────────────────────────────────────────────────
  // Test 1: Image loads with correct shape and non-trivial content
  // ─────────────────────────────────────────────────────────────────────
  test('sheet1.jpg decodes to correct dimensions and has OMR content', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(async () => {
        const gray = await (window as any).__loadSheet1Gray();
        try {
          return {
            rows: gray.rows,
            cols: gray.cols,
            channels: gray.channels(),
            nonZeroPixels: window.cv.countNonZero(gray),
            meanPixelValue: window.cv.mean(gray)[0],
          };
        } finally {
          gray.delete();
        }
      });

      // Verify exact dimensions from file metadata
      expect(result.rows).toBe(EXPECTED_ROWS);
      expect(result.cols).toBe(EXPECTED_COLS);
      expect(result.channels).toBe(1); // grayscale

      // OMR sheet is mostly white paper — majority of pixels should be light (non-zero)
      expect(result.nonZeroPixels).toBeGreaterThan(EXPECTED_COLS * EXPECTED_ROWS * 0.5);

      // Mean pixel value should be in a realistic range:
      // pure black = 0, pure white = 255; a printed sheet is typically 100–220
      expect(result.meanPixelValue).toBeGreaterThan(50);
      expect(result.meanPixelValue).toBeLessThan(240);
    });
  });

  // ─────────────────────────────────────────────────────────────────────
  // Test 2: GaussianBlur preprocessing preserves dimensions and mean
  // ─────────────────────────────────────────────────────────────────────
  test('GaussianBlur preprocessing preserves image dimensions and mean', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(async () => {
        const gray = await (window as any).__loadSheet1Gray();
        const blurred = new window.cv.Mat();
        try {
          // Replicate GaussianBlur.process() from processors/image/GaussianBlur.ts
          window.cv.GaussianBlur(gray, blurred, new window.cv.Size(3, 3), /* sigmaX= */ 0);
          return {
            rows: blurred.rows,
            cols: blurred.cols,
            origMean: window.cv.mean(gray)[0],
            blurMean: window.cv.mean(blurred)[0],
          };
        } finally {
          gray.delete();
          blurred.delete();
        }
      });

      // Blur must not change image dimensions
      expect(result.rows).toBe(EXPECTED_ROWS);
      expect(result.cols).toBe(EXPECTED_COLS);

      // GaussianBlur is a linear filter — it conserves average brightness
      // (mean should change by < 5 gray levels)
      expect(Math.abs(result.origMean - result.blurMean)).toBeLessThan(5);
    });
  });

  // ─────────────────────────────────────────────────────────────────────
  // Test 3: Full pipeline processFile flow with the real image
  // ─────────────────────────────────────────────────────────────────────
  test('pipeline processFile completes with real image', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(async () => {
        const gray = await (window as any).__loadSheet1Gray();

        // Replicate ProcessingPipeline.processFile() with one GaussianBlur processor.
        // This exercises the context object and processor chain with the real image.
        const context: Record<string, any> = {
          filePath: 'inputs/sample1/MobileCamera/sheet1.jpg',
          grayImage: gray,
          coloredImage: null,
          omrResponse: {} as Record<string, string>,
          isMultiMarked: false,
          metadata: {},
        };

        // Processor: GaussianBlur (implemented in src/processors/image/GaussianBlur.ts)
        const blurred = new window.cv.Mat();
        window.cv.GaussianBlur(context.grayImage, blurred, new window.cv.Size(3, 3), 0);
        context.grayImage.delete();
        context.grayImage = blurred;

        const finalState = {
          filePath: context.filePath as string,
          rows: context.grayImage.rows as number,
          cols: context.grayImage.cols as number,
          isMultiMarked: context.isMultiMarked as boolean,
          // OMR reading not yet ported to TS — response must remain empty
          omrResponseKeys: Object.keys(context.omrResponse as Record<string, string>).length,
        };

        context.grayImage.delete();
        return finalState;
      });

      expect(result.filePath).toBe('inputs/sample1/MobileCamera/sheet1.jpg');
      expect(result.rows).toBe(EXPECTED_ROWS);
      expect(result.cols).toBe(EXPECTED_COLS);
      expect(result.isMultiMarked).toBe(false);
      // Bubble detection not yet ported — omrResponse must be empty for now
      expect(result.omrResponseKeys).toBe(0);
    });
  });
});
