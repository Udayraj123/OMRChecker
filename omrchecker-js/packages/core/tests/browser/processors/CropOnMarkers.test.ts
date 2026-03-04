/**
 * Browser tests for CropOnMarkers.ts using OpenCV.js (Playwright).
 *
 * Port of Python CropOnMarkers (FOUR_MARKERS type only) verified against:
 *   samples/1-mobile-camera/MobileCamera/sheet1.jpg + omr_marker.jpg
 *
 * Tests:
 *   1. detectsMarkersInSampleImage — full pipeline on real image
 *   2. throwsWhenMarkerNotFound    — blank white image → ImageProcessingError
 *
 * Run with:
 *   cd omrchecker-js/packages/core && npx playwright test tests/browser/processors/CropOnMarkers.test.ts --reporter=list
 */

import { test, expect } from '@playwright/test';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { setupBrowser } from '../browser-setup';
import { withMemoryTracking } from '../memory-utils';
import { SHARED_UTILS_SCRIPT } from './shared-browser-utils';
import { CROP_ON_MARKERS_SCRIPT } from './shared-crop-on-markers-script';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Real sample images
const SHEET1_PATH = join(
  __dirname,
  '../../../../../../samples/1-mobile-camera/MobileCamera/sheet1.jpg'
);
const MARKER_PATH = join(
  __dirname,
  '../../../../../../samples/1-mobile-camera/omr_marker.jpg'
);

// ─────────────────────────────────────────────────────────────────────────────
// SETUP_SCRIPT: assembled from shared modules (no inline duplication).
//
// Provides:
//   window.__loadGray(b64)    → Promise<cv.Mat>  (grayscale, caller deletes)
//   window.__cropPage(gray)   → { warped, corners } (caller deletes warped)
//   window.CropOnMarkers      → class (FOUR_MARKERS only)
// ─────────────────────────────────────────────────────────────────────────────

const SETUP_SCRIPT = SHARED_UTILS_SCRIPT + '\n' + CROP_ON_MARKERS_SCRIPT;

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

test.describe('CropOnMarkers - Browser Tests', () => {
  // Allow extra time for OpenCV.js CDN load + WASM init + image processing
  test.use({ timeout: 90_000 });

  const sheet1B64 = readFileSync(SHEET1_PATH).toString('base64');
  const markerB64 = readFileSync(MARKER_PATH).toString('base64');

  test.beforeEach(async ({ page }) => {
    await setupBrowser(page);
    await page.evaluate(SETUP_SCRIPT);
  });

  // ─── Test 1: detectsMarkersInSampleImage ─────────────────────────────────────
  //
  // Load samples/1-mobile-camera/MobileCamera/sheet1.jpg, pass through CropPage,
  // then run CropOnMarkers with the omr_marker.jpg asset (decoded via canvas).
  // Verify: output is a valid grayscale image with reasonable dimensions.
  //
  test('detectsMarkersInSampleImage', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(
        async ([sheetB64, mrkB64]: string[]) => {
          const mats: any[] = [];
          try {
            // Stage 1: load and CropPage
            const gray = await (window as any).__loadGray(sheetB64);
            mats.push(gray);
            const { warped: croppedPage } = (window as any).__cropPage(gray);
            mats.push(croppedPage);

            // Stage 2: CropOnMarkers (use static factory to decode marker via canvas)
            const processor = await (window as any).CropOnMarkers.fromBase64(
              {
                type: 'FOUR_MARKERS',
                reference_image: 'omr_marker.jpg',
                marker_dimensions: [35, 35],
                tuning_options: {
                  marker_rescale_range: [80, 120],
                  marker_rescale_steps: 5,
                  min_matching_threshold: 0.2,
                  apply_erode_subtract: true,
                },
              },
              { 'omr_marker.jpg': mrkB64 }
            );

            const [warped] = processor.applyFilter(croppedPage, null, null, 'sheet1.jpg');
            mats.push(warped);
            processor.dispose();

            return {
              rows: warped.rows,
              cols: warped.cols,
              channels: warped.channels(),
              mean: (window as any).cv.mean(warped)[0],
            };
          } finally {
            mats.forEach((m: any) => { try { m.delete(); } catch (_) {} });
          }
        },
        [sheet1B64, markerB64]
      );

      // Output must be a valid grayscale image
      expect(result.channels).toBe(1);
      // The warped sheet should be at least 100px in each dimension
      expect(result.rows).toBeGreaterThan(100);
      expect(result.cols).toBeGreaterThan(100);
      // Aligned OMR sheet should be mostly white/light (OMR forms have white background)
      expect(result.mean).toBeGreaterThan(50);
    });
  });

  // ─── Test 2: throwsWhenMarkerNotFound ────────────────────────────────────────
  //
  // Pass a blank white 300x400 image — no markers present.
  // CropOnMarkers should throw an error (ImageProcessingError / Error).
  //
  test('throwsWhenMarkerNotFound', async ({ page }) => {
    const result = await page.evaluate(
      async (mrkB64: string) => {
        const mats: any[] = [];
        try {
          // Create a blank white image (no markers)
          const blankImage = new (window as any).cv.Mat(
            400, 300, (window as any).cv.CV_8UC1,
            new (window as any).cv.Scalar(255)
          );
          mats.push(blankImage);

          // Use a very high confidence threshold → marker won't be found in blank image
          const processor = await (window as any).CropOnMarkers.fromBase64(
            {
              type: 'FOUR_MARKERS',
              reference_image: 'omr_marker.jpg',
              marker_dimensions: [35, 35],
              tuning_options: {
                marker_rescale_range: [80, 120],
                marker_rescale_steps: 5,
                min_matching_threshold: 0.99, // very high → will not match in blank image
                apply_erode_subtract: true,
              },
            },
            { 'omr_marker.jpg': mrkB64 }
          );

          try {
            processor.applyFilter(blankImage, null, null, 'blank.jpg');
            processor.dispose();
            return { threw: false, message: '' };
          } catch (e: any) {
            processor.dispose();
            return { threw: true, message: e.message || String(e) };
          }
        } finally {
          mats.forEach((m: any) => { try { m.delete(); } catch (_) {} });
        }
      },
      markerB64
    );

    expect(result.threw).toBe(true);
    // Should mention "marker" or "zone" or similar in the error message
    expect(result.message.toLowerCase()).toMatch(/marker|zone|patch/i);
  });
});
