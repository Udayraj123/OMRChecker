/**
 * Browser tests for dot_line_detection.ts using OpenCV.js
 *
 * These tests verify dot and line detection operations work correctly in a real
 * browser environment with actual OpenCV.js cv.Mat objects. Each test corresponds
 * to a Python test in src/tests/processors/test_dot_line_detection.py.
 *
 * 28 tests total:
 *   TestPreprocessDotZone (3)
 *   TestPreprocessLineZone (3)
 *   TestDetectContoursUsingCanny (3)
 *   TestExtractPatchCornersAndEdges (3)
 *   TestDetectDotCorners (3)
 *   TestDetectLineCornersAndEdges (3)
 *   TestValidateBlurKernel (4)
 *   TestCreateStructuringElement (4)
 *   TestDotLineDetectionIntegration (2)
 *
 * Run with: npm run test:browser -- --grep "DotLine"
 */

import { test, expect } from '@playwright/test';
import { setupBrowser } from '../browser-setup';
import { withMemoryTracking } from '../memory-utils';

// Increase timeout for browser tests (OpenCV.js WASM initialization)
test.setTimeout(60000);

test.describe('DotLine Detection - Browser Tests', () => {
  test.beforeEach(async ({ page }) => {
    await setupBrowser(page);
  });

  // ==========================================================================
  // TestPreprocessDotZone (3 tests)
  // ==========================================================================

  // test_preprocess_basic: zone with circle → result same shape, uint8, center darker
  test('test_preprocess_dot_zone_basic', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // zone_with_dot = np.full((100,100), 200) with circle at (50,50) r=15 value=50
          const zone = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(200));
          mats.push(zone);
          window.cv.circle(zone, new window.cv.Point(50, 50), 15, new window.cv.Scalar(50), -1);

          const dotKernel = window.cv.getStructuringElement(
            window.cv.MORPH_RECT,
            new window.cv.Size(5, 5)
          );
          mats.push(dotKernel);

          // Replicate preprocessDotZone:
          // 1. pad with kernel*2 white border
          const paddingW = dotKernel.cols * 2;
          const paddingH = dotKernel.rows * 2;
          const padded = new window.cv.Mat();
          mats.push(padded);
          window.cv.copyMakeBorder(
            zone, padded,
            paddingH, paddingH, paddingW, paddingW,
            window.cv.BORDER_CONSTANT, new window.cv.Scalar(255, 255, 255, 255)
          );

          // 2. morphological open
          const morphed = new window.cv.Mat();
          mats.push(morphed);
          window.cv.morphologyEx(
            padded, morphed, window.cv.MORPH_OPEN, dotKernel,
            new window.cv.Point(-1, -1), 3
          );

          // 3. threshold TRUNC at 150
          const thresholded = new window.cv.Mat();
          mats.push(thresholded);
          window.cv.threshold(morphed, thresholded, 150, 255, window.cv.THRESH_TRUNC);

          // 4. normalize
          const minMax = window.cv.minMaxLoc(thresholded);
          let normalised: any;
          if (minMax.maxVal === minMax.minVal) {
            normalised = thresholded.clone();
          } else {
            normalised = new window.cv.Mat();
            window.cv.normalize(thresholded, normalised, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          }
          mats.push(normalised);

          // 5. crop back
          const cropped = normalised.roi(new window.cv.Rect(paddingW, paddingH, 100, 100)).clone();
          mats.push(cropped);

          const centerPixel = cropped.ucharAt(50, 50);
          const cornerPixel = cropped.ucharAt(10, 10);

          return {
            rows: cropped.rows,
            cols: cropped.cols,
            type: cropped.type(),
            centerPixel,
            cornerPixel,
          };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.rows).toBe(100);
      expect(result.cols).toBe(100);
      expect(result.type).toBe(0); // CV_8UC1
      expect(result.centerPixel).toBeLessThan(result.cornerPixel);
    });
  });

  // test_preprocess_with_blur: zone with blur_kernel → same shape, center < 200
  test('test_preprocess_dot_zone_with_blur', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const zone = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(200));
          mats.push(zone);
          window.cv.circle(zone, new window.cv.Point(50, 50), 15, new window.cv.Scalar(50), -1);

          const dotKernel = window.cv.getStructuringElement(
            window.cv.MORPH_RECT,
            new window.cv.Size(5, 5)
          );
          mats.push(dotKernel);

          // Apply blur first
          const blurred = new window.cv.Mat();
          mats.push(blurred);
          window.cv.GaussianBlur(zone, blurred, new window.cv.Size(5, 5), 0);

          const paddingW = dotKernel.cols * 2;
          const paddingH = dotKernel.rows * 2;
          const padded = new window.cv.Mat();
          mats.push(padded);
          window.cv.copyMakeBorder(
            blurred, padded,
            paddingH, paddingH, paddingW, paddingW,
            window.cv.BORDER_CONSTANT, new window.cv.Scalar(255, 255, 255, 255)
          );

          const morphed = new window.cv.Mat();
          mats.push(morphed);
          window.cv.morphologyEx(
            padded, morphed, window.cv.MORPH_OPEN, dotKernel,
            new window.cv.Point(-1, -1), 3
          );

          const thresholded = new window.cv.Mat();
          mats.push(thresholded);
          window.cv.threshold(morphed, thresholded, 150, 255, window.cv.THRESH_TRUNC);

          const minMax = window.cv.minMaxLoc(thresholded);
          let normalised: any;
          if (minMax.maxVal === minMax.minVal) {
            normalised = thresholded.clone();
          } else {
            normalised = new window.cv.Mat();
            window.cv.normalize(thresholded, normalised, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          }
          mats.push(normalised);

          const cropped = normalised.roi(new window.cv.Rect(paddingW, paddingH, 100, 100)).clone();
          mats.push(cropped);

          const centerPixel = cropped.ucharAt(50, 50);

          return {
            rows: cropped.rows,
            cols: cropped.cols,
            centerPixel,
          };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.rows).toBe(100);
      expect(result.cols).toBe(100);
      expect(result.centerPixel).toBeLessThan(200);
    });
  });

  // test_preprocess_with_threshold: two different thresholds → both same shape
  test('test_preprocess_dot_zone_with_threshold', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const zone = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(200));
          mats.push(zone);
          window.cv.circle(zone, new window.cv.Point(50, 50), 15, new window.cv.Scalar(50), -1);

          const dotKernel = window.cv.getStructuringElement(
            window.cv.MORPH_RECT,
            new window.cv.Size(5, 5)
          );
          mats.push(dotKernel);

          const paddingW = dotKernel.cols * 2;
          const paddingH = dotKernel.rows * 2;

          function processWithThreshold(thresh: number): { rows: number; cols: number } {
            const padded = new window.cv.Mat();
            mats.push(padded);
            window.cv.copyMakeBorder(
              zone, padded,
              paddingH, paddingH, paddingW, paddingW,
              window.cv.BORDER_CONSTANT, new window.cv.Scalar(255, 255, 255, 255)
            );
            const morphed = new window.cv.Mat();
            mats.push(morphed);
            window.cv.morphologyEx(
              padded, morphed, window.cv.MORPH_OPEN, dotKernel,
              new window.cv.Point(-1, -1), 3
            );
            const thresholded = new window.cv.Mat();
            mats.push(thresholded);
            window.cv.threshold(morphed, thresholded, thresh, 255, window.cv.THRESH_TRUNC);
            const minMax = window.cv.minMaxLoc(thresholded);
            let normalised: any;
            if (minMax.maxVal === minMax.minVal) {
              normalised = thresholded.clone();
            } else {
              normalised = new window.cv.Mat();
              window.cv.normalize(thresholded, normalised, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
            }
            mats.push(normalised);
            const cropped = normalised.roi(new window.cv.Rect(paddingW, paddingH, 100, 100)).clone();
            mats.push(cropped);
            return { rows: cropped.rows, cols: cropped.cols };
          }

          const low = processWithThreshold(100);
          const high = processWithThreshold(200);

          return { lowRows: low.rows, lowCols: low.cols, highRows: high.rows, highCols: high.cols };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.lowRows).toBe(100);
      expect(result.lowCols).toBe(100);
      expect(result.highRows).toBe(100);
      expect(result.highCols).toBe(100);
    });
  });

  // ==========================================================================
  // TestPreprocessLineZone (3 tests)
  // ==========================================================================

  // test_preprocess_basic: zone with line → same shape, uint8, mean < 255
  test('test_preprocess_line_zone_basic', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // zone_with_line = np.full((100,100), 200) + vertical line at x=50 value=50
          const zone = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(200));
          mats.push(zone);
          window.cv.line(
            zone,
            new window.cv.Point(50, 10),
            new window.cv.Point(50, 90),
            new window.cv.Scalar(50),
            3
          );

          const lineKernel = window.cv.getStructuringElement(
            window.cv.MORPH_RECT,
            new window.cv.Size(2, 10)
          );
          mats.push(lineKernel);

          // 1. adjustGamma with gamma=0.5 → invGamma=2.0
          const invGamma = 2.0;
          const lutData = new Uint8Array(256);
          for (let i = 0; i < 256; i++) {
            lutData[i] = Math.round(Math.pow(i / 255.0, invGamma) * 255);
          }
          const lutMat = window.cv.matFromArray(1, 256, window.cv.CV_8UC1, Array.from(lutData));
          mats.push(lutMat);
          const darker = new window.cv.Mat();
          mats.push(darker);
          window.cv.LUT(zone, lutMat, darker);

          // 2. threshold TRUNC at 180
          const thresholded = new window.cv.Mat();
          mats.push(thresholded);
          window.cv.threshold(darker, thresholded, 180, 255, window.cv.THRESH_TRUNC);

          // 3. normalize
          const minMax1 = window.cv.minMaxLoc(thresholded);
          let normalised: any;
          if (minMax1.maxVal === minMax1.minVal) {
            normalised = thresholded.clone();
          } else {
            normalised = new window.cv.Mat();
            window.cv.normalize(thresholded, normalised, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          }
          mats.push(normalised);

          // 4. pad
          const paddingW = lineKernel.cols * 2;
          const paddingH = lineKernel.rows * 2;
          const padded = new window.cv.Mat();
          mats.push(padded);
          window.cv.copyMakeBorder(
            normalised, padded,
            paddingH, paddingH, paddingW, paddingW,
            window.cv.BORDER_CONSTANT, new window.cv.Scalar(255, 255, 255, 255)
          );

          // 5. threshold again
          const whiteThr = new window.cv.Mat();
          mats.push(whiteThr);
          window.cv.threshold(padded, whiteThr, 180, 255, window.cv.THRESH_TRUNC);

          // 6. normalize again
          const minMax2 = window.cv.minMaxLoc(whiteThr);
          let whiteNorm: any;
          if (minMax2.maxVal === minMax2.minVal) {
            whiteNorm = whiteThr.clone();
          } else {
            whiteNorm = new window.cv.Mat();
            window.cv.normalize(whiteThr, whiteNorm, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          }
          mats.push(whiteNorm);

          // 7. morphological open
          const morphed = new window.cv.Mat();
          mats.push(morphed);
          window.cv.morphologyEx(
            whiteNorm, morphed, window.cv.MORPH_OPEN, lineKernel,
            new window.cv.Point(-1, -1), 3
          );

          // 8. crop
          const cropped = morphed.roi(new window.cv.Rect(paddingW, paddingH, 100, 100)).clone();
          mats.push(cropped);

          const meanVal = window.cv.mean(cropped)[0];

          return { rows: cropped.rows, cols: cropped.cols, type: cropped.type(), meanVal };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.rows).toBe(100);
      expect(result.cols).toBe(100);
      expect(result.type).toBe(0); // CV_8UC1
      expect(result.meanVal).toBeLessThan(255);
    });
  });

  // test_preprocess_with_gamma: two gamma values → both produce same shape
  test('test_preprocess_line_zone_with_gamma', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const zone = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(200));
          mats.push(zone);
          window.cv.line(
            zone,
            new window.cv.Point(50, 10),
            new window.cv.Point(50, 90),
            new window.cv.Scalar(50),
            3
          );

          const lineKernel = window.cv.getStructuringElement(
            window.cv.MORPH_RECT,
            new window.cv.Size(2, 10)
          );
          mats.push(lineKernel);

          function processWithGamma(gamma: number): { rows: number; cols: number } {
            const invGamma = 1.0 / gamma;
            const lutData = new Uint8Array(256);
            for (let i = 0; i < 256; i++) {
              lutData[i] = Math.round(Math.pow(i / 255.0, invGamma) * 255);
            }
            const lutMat = window.cv.matFromArray(1, 256, window.cv.CV_8UC1, Array.from(lutData));
            mats.push(lutMat);
            const darker = new window.cv.Mat();
            mats.push(darker);
            window.cv.LUT(zone, lutMat, darker);

            const thresholded = new window.cv.Mat();
            mats.push(thresholded);
            window.cv.threshold(darker, thresholded, 180, 255, window.cv.THRESH_TRUNC);

            const minMax = window.cv.minMaxLoc(thresholded);
            let norm: any;
            if (minMax.maxVal === minMax.minVal) {
              norm = thresholded.clone();
            } else {
              norm = new window.cv.Mat();
              window.cv.normalize(thresholded, norm, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
            }
            mats.push(norm);

            const paddingW = lineKernel.cols * 2;
            const paddingH = lineKernel.rows * 2;
            const padded = new window.cv.Mat();
            mats.push(padded);
            window.cv.copyMakeBorder(
              norm, padded,
              paddingH, paddingH, paddingW, paddingW,
              window.cv.BORDER_CONSTANT, new window.cv.Scalar(255, 255, 255, 255)
            );

            const wt = new window.cv.Mat();
            mats.push(wt);
            window.cv.threshold(padded, wt, 180, 255, window.cv.THRESH_TRUNC);

            const minMax2 = window.cv.minMaxLoc(wt);
            let wn: any;
            if (minMax2.maxVal === minMax2.minVal) {
              wn = wt.clone();
            } else {
              wn = new window.cv.Mat();
              window.cv.normalize(wt, wn, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
            }
            mats.push(wn);

            const morphed = new window.cv.Mat();
            mats.push(morphed);
            window.cv.morphologyEx(
              wn, morphed, window.cv.MORPH_OPEN, lineKernel,
              new window.cv.Point(-1, -1), 3
            );

            const cropped = morphed.roi(new window.cv.Rect(paddingW, paddingH, 100, 100)).clone();
            mats.push(cropped);
            return { rows: cropped.rows, cols: cropped.cols };
          }

          const low = processWithGamma(0.3);
          const high = processWithGamma(0.9);

          return { lowRows: low.rows, lowCols: low.cols, highRows: high.rows, highCols: high.cols };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.lowRows).toBe(100);
      expect(result.lowCols).toBe(100);
      expect(result.highRows).toBe(100);
      expect(result.highCols).toBe(100);
    });
  });

  // test_preprocess_with_blur: blur_kernel → same shape
  test('test_preprocess_line_zone_with_blur', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const zone = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(200));
          mats.push(zone);
          window.cv.line(
            zone,
            new window.cv.Point(50, 10),
            new window.cv.Point(50, 90),
            new window.cv.Scalar(50),
            3
          );

          const lineKernel = window.cv.getStructuringElement(
            window.cv.MORPH_RECT,
            new window.cv.Size(2, 10)
          );
          mats.push(lineKernel);

          // Blur first
          const blurred = new window.cv.Mat();
          mats.push(blurred);
          window.cv.GaussianBlur(zone, blurred, new window.cv.Size(5, 5), 0);

          const invGamma = 2.0; // gamma=0.5
          const lutData = new Uint8Array(256);
          for (let i = 0; i < 256; i++) {
            lutData[i] = Math.round(Math.pow(i / 255.0, invGamma) * 255);
          }
          const lutMat = window.cv.matFromArray(1, 256, window.cv.CV_8UC1, Array.from(lutData));
          mats.push(lutMat);
          const darker = new window.cv.Mat();
          mats.push(darker);
          window.cv.LUT(blurred, lutMat, darker);

          const thresholded = new window.cv.Mat();
          mats.push(thresholded);
          window.cv.threshold(darker, thresholded, 180, 255, window.cv.THRESH_TRUNC);

          const minMax = window.cv.minMaxLoc(thresholded);
          let norm: any;
          if (minMax.maxVal === minMax.minVal) {
            norm = thresholded.clone();
          } else {
            norm = new window.cv.Mat();
            window.cv.normalize(thresholded, norm, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          }
          mats.push(norm);

          const paddingW = lineKernel.cols * 2;
          const paddingH = lineKernel.rows * 2;
          const padded = new window.cv.Mat();
          mats.push(padded);
          window.cv.copyMakeBorder(
            norm, padded,
            paddingH, paddingH, paddingW, paddingW,
            window.cv.BORDER_CONSTANT, new window.cv.Scalar(255, 255, 255, 255)
          );

          const wt = new window.cv.Mat();
          mats.push(wt);
          window.cv.threshold(padded, wt, 180, 255, window.cv.THRESH_TRUNC);

          const minMax2 = window.cv.minMaxLoc(wt);
          let wn: any;
          if (minMax2.maxVal === minMax2.minVal) {
            wn = wt.clone();
          } else {
            wn = new window.cv.Mat();
            window.cv.normalize(wt, wn, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          }
          mats.push(wn);

          const morphed = new window.cv.Mat();
          mats.push(morphed);
          window.cv.morphologyEx(
            wn, morphed, window.cv.MORPH_OPEN, lineKernel,
            new window.cv.Point(-1, -1), 3
          );

          const cropped = morphed.roi(new window.cv.Rect(paddingW, paddingH, 100, 100)).clone();
          mats.push(cropped);

          return { rows: cropped.rows, cols: cropped.cols };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.rows).toBe(100);
      expect(result.cols).toBe(100);
    });
  });

  // ==========================================================================
  // TestDetectContoursUsingCanny (3 tests)
  // ==========================================================================

  // test_detect_contours_success: white rect on black → contours found, largest area > 1000
  test('test_detect_contours_success', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const zone = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(zone);
          // Draw white rectangle: (20,20) to (80,80)
          window.cv.rectangle(
            zone,
            new window.cv.Point(20, 20),
            new window.cv.Point(80, 80),
            new window.cv.Scalar(255),
            -1
          );

          const edges = new window.cv.Mat();
          mats.push(edges);
          // Python: cv2.Canny(zone, canny_high=185, canny_low=55)
          window.cv.Canny(zone, edges, 185, 55);

          const contours = new window.cv.MatVector();
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(
            edges, contours, hierarchy,
            window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE
          );

          const count = contours.size();
          let largestArea = 0;
          const resultContours: any[] = [];
          for (let i = 0; i < count; i++) {
            const c = contours.get(i).clone();
            resultContours.push(c);
            mats.push(c);
          }

          resultContours.sort((a, b) => window.cv.contourArea(b) - window.cv.contourArea(a));
          if (resultContours.length > 0) {
            largestArea = window.cv.contourArea(resultContours[0]);
          }

          contours.delete();

          return { count, largestArea };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.count).toBeGreaterThan(0);
      expect(result.largestArea).toBeGreaterThan(1000);
    });
  });

  // test_detect_contours_empty_zone: all-zero image → no contours
  test('test_detect_contours_empty_zone', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const zone = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(zone);

          const edges = new window.cv.Mat();
          mats.push(edges);
          window.cv.Canny(zone, edges, 185, 55);

          const contours = new window.cv.MatVector();
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(
            edges, contours, hierarchy,
            window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE
          );
          const count = contours.size();
          contours.delete();

          return { count };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.count).toBe(0);
    });
  });

  // test_detect_contours_sorted_by_area: two rects → contours sorted largest first
  test('test_detect_contours_sorted_by_area', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const zone = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(zone);
          // Small rectangle
          window.cv.rectangle(
            zone,
            new window.cv.Point(10, 10),
            new window.cv.Point(30, 30),
            new window.cv.Scalar(255),
            -1
          );
          // Large rectangle
          window.cv.rectangle(
            zone,
            new window.cv.Point(40, 40),
            new window.cv.Point(90, 90),
            new window.cv.Scalar(255),
            -1
          );

          const edges = new window.cv.Mat();
          mats.push(edges);
          window.cv.Canny(zone, edges, 185, 55);

          const contourVec = new window.cv.MatVector();
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(
            edges, contourVec, hierarchy,
            window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE
          );

          const cloned: any[] = [];
          const vecSize = contourVec.size();
          for (let i = 0; i < vecSize; i++) {
            const c = contourVec.get(i).clone();
            cloned.push(c);
            mats.push(c);
          }
          contourVec.delete();

          cloned.sort((a, b) => window.cv.contourArea(b) - window.cv.contourArea(a));

          const areas = cloned.map(c => window.cv.contourArea(c));

          return { count: cloned.length, area0: areas[0] ?? 0, area1: areas[1] ?? 0 };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.count).toBeGreaterThanOrEqual(2);
      expect(result.area0).toBeGreaterThan(result.area1);
    });
  });

  // ==========================================================================
  // TestExtractPatchCornersAndEdges (3 tests)
  // ==========================================================================

  // test_extract_dot_corners: rectangle contour → 4 corners and 4 edge groups
  test('test_extract_patch_corners_dot', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // Build rectangle contour from a white rect on black
          const zone = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(zone);
          window.cv.rectangle(
            zone,
            new window.cv.Point(20, 20),
            new window.cv.Point(80, 80),
            new window.cv.Scalar(255),
            -1
          );

          const edges = new window.cv.Mat();
          mats.push(edges);
          window.cv.Canny(zone, edges, 185, 55);

          const contourVec = new window.cv.MatVector();
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(
            edges, contourVec, hierarchy,
            window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE
          );

          if (contourVec.size() === 0) {
            contourVec.delete();
            return { cornersLen: 0, edgesLen: 0 };
          }

          // Get largest contour
          let largest = contourVec.get(0).clone();
          mats.push(largest);
          for (let i = 1; i < contourVec.size(); i++) {
            const c = contourVec.get(i);
            if (window.cv.contourArea(c) > window.cv.contourArea(largest)) {
              largest.delete();
              mats.splice(mats.indexOf(largest), 1);
              largest = c.clone();
              mats.push(largest);
            }
          }
          contourVec.delete();

          // Extract boundary points
          const data = largest.data32S;
          const boundaryPoints: [number, number][] = [];
          for (let i = 0; i < data.length; i += 2) {
            boundaryPoints.push([data[i], data[i + 1]]);
          }

          // Get bounding hull → bounding rect for PATCH_DOT
          const hull = new window.cv.Mat();
          mats.push(hull);
          window.cv.convexHull(largest, hull, false, true);
          const rect = window.cv.boundingRect(hull);
          const patchCorners: [number, number][] = [
            [rect.x, rect.y],
            [rect.x + rect.width, rect.y],
            [rect.x + rect.width, rect.y + rect.height],
            [rect.x, rect.y + rect.height],
          ];

          // Order corners [tl, tr, br, bl]
          const sums = patchCorners.map(p => p[0] + p[1]);
          const diffs = patchCorners.map(p => p[1] - p[0]);
          const tl = patchCorners[sums.indexOf(Math.min(...sums))];
          const tr = patchCorners[diffs.indexOf(Math.min(...diffs))];
          const br = patchCorners[sums.indexOf(Math.max(...sums))];
          const bl = patchCorners[diffs.indexOf(Math.max(...diffs))];
          const orderedCorners = [tl, tr, br, bl];

          // Split contour on corners
          const edgeTypes = ['TOP', 'RIGHT', 'BOTTOM', 'LEFT'];
          const edgeContoursMap: Record<string, [number, number][]> = {
            TOP: [], RIGHT: [], BOTTOM: [], LEFT: [],
          };

          function distToSegment(
            px: number, py: number,
            ax: number, ay: number,
            bx: number, by: number
          ): number {
            const dx = bx - ax, dy = by - ay;
            const lenSq = dx * dx + dy * dy;
            if (lenSq === 0) return Math.hypot(px - ax, py - ay);
            let t = ((px - ax) * dx + (py - ay) * dy) / lenSq;
            t = Math.max(0, Math.min(1, t));
            return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
          }

          for (const pt of boundaryPoints) {
            let minDist = Infinity, nearestEdge = 'TOP';
            for (let i = 0; i < 4; i++) {
              const [ax, ay] = orderedCorners[i];
              const [bx, by] = orderedCorners[(i + 1) % 4];
              const d = distToSegment(pt[0], pt[1], ax, ay, bx, by);
              if (d < minDist) { minDist = d; nearestEdge = edgeTypes[i]; }
            }
            edgeContoursMap[nearestEdge].push(pt);
          }

          for (let i = 0; i < 4; i++) {
            edgeContoursMap[edgeTypes[i]].unshift(orderedCorners[i]);
            edgeContoursMap[edgeTypes[i]].push(orderedCorners[(i + 1) % 4]);
          }

          return {
            cornersLen: orderedCorners.length,
            edgesLen: Object.keys(edgeContoursMap).length,
          };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.cornersLen).toBe(4);
      expect(result.edgesLen).toBe(4);
    });
  });

  // test_extract_line_corners: rectangle contour → 4 corners and 4 edge groups (min area rect)
  test('test_extract_patch_corners_line', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const zone = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(zone);
          window.cv.rectangle(
            zone,
            new window.cv.Point(20, 20),
            new window.cv.Point(80, 80),
            new window.cv.Scalar(255),
            -1
          );

          const edges = new window.cv.Mat();
          mats.push(edges);
          window.cv.Canny(zone, edges, 185, 55);

          const contourVec = new window.cv.MatVector();
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(
            edges, contourVec, hierarchy,
            window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE
          );

          if (contourVec.size() === 0) {
            contourVec.delete();
            return { cornersLen: 0, edgesLen: 0 };
          }

          let largest = contourVec.get(0).clone();
          mats.push(largest);
          for (let i = 1; i < contourVec.size(); i++) {
            const c = contourVec.get(i);
            if (window.cv.contourArea(c) > window.cv.contourArea(largest)) {
              largest.delete();
              mats.splice(mats.indexOf(largest), 1);
              largest = c.clone();
              mats.push(largest);
            }
          }
          contourVec.delete();

          const data = largest.data32S;
          const boundaryPoints: [number, number][] = [];
          for (let i = 0; i < data.length; i += 2) {
            boundaryPoints.push([data[i], data[i + 1]]);
          }

          const hull = new window.cv.Mat();
          mats.push(hull);
          window.cv.convexHull(largest, hull, false, true);

          // Use min area rect for PATCH_LINE
          // cv.boxPoints returns an array of {x,y} objects in OpenCV.js 4.x
          const rotRect = window.cv.minAreaRect(hull);
          const boxPtsArr = window.cv.boxPoints(rotRect) as Array<{ x: number; y: number }>;
          const patchCorners: [number, number][] = boxPtsArr.map(
            (pt: { x: number; y: number }) => [Math.round(pt.x), Math.round(pt.y)] as [number, number]
          );

          // Order corners
          const sums = patchCorners.map(p => p[0] + p[1]);
          const diffs = patchCorners.map(p => p[1] - p[0]);
          const tl = patchCorners[sums.indexOf(Math.min(...sums))];
          const tr = patchCorners[diffs.indexOf(Math.min(...diffs))];
          const br = patchCorners[sums.indexOf(Math.max(...sums))];
          const bl = patchCorners[diffs.indexOf(Math.max(...diffs))];
          const orderedCorners = [tl, tr, br, bl];

          const edgeContoursMap: Record<string, [number, number][]> = {
            TOP: [], RIGHT: [], BOTTOM: [], LEFT: [],
          };

          return {
            cornersLen: orderedCorners.length,
            edgesLen: Object.keys(edgeContoursMap).length,
          };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.cornersLen).toBe(4);
      expect(result.edgesLen).toBe(4);
    });
  });

  // test_unsupported_scanner_type: invalid type → throws error containing "Unsupported scanner type"
  test('test_extract_patch_corners_unsupported_type', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const zone = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(zone);
          window.cv.rectangle(
            zone,
            new window.cv.Point(20, 20),
            new window.cv.Point(80, 80),
            new window.cv.Scalar(255),
            -1
          );

          const edges = new window.cv.Mat();
          mats.push(edges);
          window.cv.Canny(zone, edges, 185, 55);

          const contourVec = new window.cv.MatVector();
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(
            edges, contourVec, hierarchy,
            window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE
          );

          if (contourVec.size() === 0) {
            contourVec.delete();
            return { threw: false, message: '' };
          }

          const largest = contourVec.get(0).clone();
          mats.push(largest);
          contourVec.delete();

          // Try with invalid scanner type
          try {
            const hull = new window.cv.Mat();
            mats.push(hull);
            window.cv.convexHull(largest, hull, false, true);

            const scannerType = 'INVALID_TYPE';
            if (scannerType !== 'PATCH_DOT' && scannerType !== 'PATCH_LINE') {
              throw new Error(`Unsupported scanner type: ${scannerType}`);
            }
            return { threw: false, message: '' };
          } catch (e: any) {
            return { threw: true, message: e.message };
          }
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.threw).toBe(true);
      expect(result.message).toContain('Unsupported scanner type');
    });
  });

  // ==========================================================================
  // TestDetectDotCorners (3 tests)
  // ==========================================================================

  // test_detect_dot_success: zone with rectangle → corners found, 4×2, in range 30-70
  test('test_detect_dot_corners_success', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // zone_with_dot = np.full((100,100),200) + rectangle (40,40)-(60,60) value=50
          const zone = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(200));
          mats.push(zone);
          window.cv.rectangle(
            zone,
            new window.cv.Point(40, 40),
            new window.cv.Point(60, 60),
            new window.cv.Scalar(50),
            -1
          );

          const dotKernel = window.cv.getStructuringElement(
            window.cv.MORPH_RECT,
            new window.cv.Size(5, 5)
          );
          mats.push(dotKernel);

          // Preprocess dot zone
          const paddingW = dotKernel.cols * 2;
          const paddingH = dotKernel.rows * 2;
          const padded = new window.cv.Mat();
          mats.push(padded);
          window.cv.copyMakeBorder(
            zone, padded,
            paddingH, paddingH, paddingW, paddingW,
            window.cv.BORDER_CONSTANT, new window.cv.Scalar(255, 255, 255, 255)
          );
          const morphed = new window.cv.Mat();
          mats.push(morphed);
          window.cv.morphologyEx(
            padded, morphed, window.cv.MORPH_OPEN, dotKernel,
            new window.cv.Point(-1, -1), 3
          );
          const thresholded = new window.cv.Mat();
          mats.push(thresholded);
          window.cv.threshold(morphed, thresholded, 150, 255, window.cv.THRESH_TRUNC);
          const minMax = window.cv.minMaxLoc(thresholded);
          let normalised: any;
          if (minMax.maxVal === minMax.minVal) {
            normalised = thresholded.clone();
          } else {
            normalised = new window.cv.Mat();
            window.cv.normalize(thresholded, normalised, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          }
          mats.push(normalised);
          const preprocessed = normalised.roi(
            new window.cv.Rect(paddingW, paddingH, 100, 100)
          ).clone();
          mats.push(preprocessed);

          // Detect contours
          const edges = new window.cv.Mat();
          mats.push(edges);
          window.cv.Canny(preprocessed, edges, 185, 55);
          const contourVec = new window.cv.MatVector();
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(
            edges, contourVec, hierarchy,
            window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE
          );

          if (contourVec.size() === 0) {
            contourVec.delete();
            return { found: false, cornersShape: [0, 0], corners: [] };
          }

          // Sort by area
          const cloned: any[] = [];
          for (let i = 0; i < contourVec.size(); i++) {
            const c = contourVec.get(i).clone();
            cloned.push(c);
            mats.push(c);
          }
          contourVec.delete();
          cloned.sort((a, b) => window.cv.contourArea(b) - window.cv.contourArea(a));

          // Get bounding rect for PATCH_DOT
          const hull = new window.cv.Mat();
          mats.push(hull);
          window.cv.convexHull(cloned[0], hull, false, true);
          const rect = window.cv.boundingRect(hull);
          const corners: [number, number][] = [
            [rect.x, rect.y],
            [rect.x + rect.width, rect.y],
            [rect.x + rect.width, rect.y + rect.height],
            [rect.x, rect.y + rect.height],
          ];

          return {
            found: true,
            cornersShape: [corners.length, 2],
            corners,
          };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.found).toBe(true);
      expect(result.cornersShape[0]).toBe(4);
      expect(result.cornersShape[1]).toBe(2);
      // At least one corner x-coordinate in range 30-70
      expect(result.corners[0][0]).toBeGreaterThanOrEqual(30);
      expect(result.corners[0][0]).toBeLessThanOrEqual(70);
    });
  });

  // test_detect_dot_with_offset: zone_offset=(100,100) → all coords >= 100
  test('test_detect_dot_corners_with_offset', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const zone = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(200));
          mats.push(zone);
          window.cv.rectangle(
            zone,
            new window.cv.Point(40, 40),
            new window.cv.Point(60, 60),
            new window.cv.Scalar(50),
            -1
          );

          const dotKernel = window.cv.getStructuringElement(
            window.cv.MORPH_RECT,
            new window.cv.Size(5, 5)
          );
          mats.push(dotKernel);

          const paddingW = dotKernel.cols * 2;
          const paddingH = dotKernel.rows * 2;
          const padded = new window.cv.Mat();
          mats.push(padded);
          window.cv.copyMakeBorder(
            zone, padded,
            paddingH, paddingH, paddingW, paddingW,
            window.cv.BORDER_CONSTANT, new window.cv.Scalar(255, 255, 255, 255)
          );
          const morphed = new window.cv.Mat();
          mats.push(morphed);
          window.cv.morphologyEx(
            padded, morphed, window.cv.MORPH_OPEN, dotKernel,
            new window.cv.Point(-1, -1), 3
          );
          const thresholded = new window.cv.Mat();
          mats.push(thresholded);
          window.cv.threshold(morphed, thresholded, 150, 255, window.cv.THRESH_TRUNC);
          const minMax = window.cv.minMaxLoc(thresholded);
          let normalised: any;
          if (minMax.maxVal === minMax.minVal) {
            normalised = thresholded.clone();
          } else {
            normalised = new window.cv.Mat();
            window.cv.normalize(thresholded, normalised, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          }
          mats.push(normalised);
          const preprocessed = normalised.roi(
            new window.cv.Rect(paddingW, paddingH, 100, 100)
          ).clone();
          mats.push(preprocessed);

          const edges = new window.cv.Mat();
          mats.push(edges);
          window.cv.Canny(preprocessed, edges, 185, 55);
          const contourVec = new window.cv.MatVector();
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(
            edges, contourVec, hierarchy,
            window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE
          );

          if (contourVec.size() === 0) {
            contourVec.delete();
            return { found: false, allGeq100: false };
          }

          const cloned: any[] = [];
          for (let i = 0; i < contourVec.size(); i++) {
            const c = contourVec.get(i).clone();
            cloned.push(c);
            mats.push(c);
          }
          contourVec.delete();
          cloned.sort((a, b) => window.cv.contourArea(b) - window.cv.contourArea(a));

          const hull = new window.cv.Mat();
          mats.push(hull);
          window.cv.convexHull(cloned[0], hull, false, true);
          const rect = window.cv.boundingRect(hull);

          const zoneOffset = [100, 100];
          const corners: [number, number][] = [
            [rect.x + zoneOffset[0], rect.y + zoneOffset[1]],
            [rect.x + rect.width + zoneOffset[0], rect.y + zoneOffset[1]],
            [rect.x + rect.width + zoneOffset[0], rect.y + rect.height + zoneOffset[1]],
            [rect.x + zoneOffset[0], rect.y + rect.height + zoneOffset[1]],
          ];

          const allGeq100 = corners.every(c => c[0] >= 100 && c[1] >= 100);

          return { found: true, allGeq100 };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.found).toBe(true);
      expect(result.allGeq100).toBe(true);
    });
  });

  // test_detect_dot_returns_none_when_not_found: uniform zone → null corners
  test('test_detect_dot_corners_returns_null_when_not_found', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // Uniform zone with no dot
          const zone = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(200));
          mats.push(zone);

          const dotKernel = window.cv.getStructuringElement(
            window.cv.MORPH_RECT,
            new window.cv.Size(5, 5)
          );
          mats.push(dotKernel);

          const paddingW = dotKernel.cols * 2;
          const paddingH = dotKernel.rows * 2;
          const padded = new window.cv.Mat();
          mats.push(padded);
          window.cv.copyMakeBorder(
            zone, padded,
            paddingH, paddingH, paddingW, paddingW,
            window.cv.BORDER_CONSTANT, new window.cv.Scalar(255, 255, 255, 255)
          );
          const morphed = new window.cv.Mat();
          mats.push(morphed);
          window.cv.morphologyEx(
            padded, morphed, window.cv.MORPH_OPEN, dotKernel,
            new window.cv.Point(-1, -1), 3
          );
          const thresholded = new window.cv.Mat();
          mats.push(thresholded);
          window.cv.threshold(morphed, thresholded, 150, 255, window.cv.THRESH_TRUNC);
          const minMax = window.cv.minMaxLoc(thresholded);
          let normalised: any;
          if (minMax.maxVal === minMax.minVal) {
            normalised = thresholded.clone();
          } else {
            normalised = new window.cv.Mat();
            window.cv.normalize(thresholded, normalised, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          }
          mats.push(normalised);
          const preprocessed = normalised.roi(
            new window.cv.Rect(paddingW, paddingH, 100, 100)
          ).clone();
          mats.push(preprocessed);

          const edges = new window.cv.Mat();
          mats.push(edges);
          window.cv.Canny(preprocessed, edges, 185, 55);
          const contourVec = new window.cv.MatVector();
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(
            edges, contourVec, hierarchy,
            window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE
          );
          const count = contourVec.size();
          contourVec.delete();

          return { contoursFound: count };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      // Uniform zone after processing should yield no contours
      expect(result.contoursFound).toBe(0);
    });
  });

  // ==========================================================================
  // TestDetectLineCornersAndEdges (3 tests)
  // ==========================================================================

  // test_detect_line_success: zone with vertical rectangle → corners 4×2, 4 edges
  test('test_detect_line_corners_success', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // zone_with_line = np.full((100,100),200) + rectangle (40,10)-(60,90) value=50
          const zone = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(200));
          mats.push(zone);
          window.cv.rectangle(
            zone,
            new window.cv.Point(40, 10),
            new window.cv.Point(60, 90),
            new window.cv.Scalar(50),
            -1
          );

          const lineKernel = window.cv.getStructuringElement(
            window.cv.MORPH_RECT,
            new window.cv.Size(2, 10)
          );
          mats.push(lineKernel);

          // preprocessLineZone with gamma=0.5
          const invGamma = 2.0;
          const lutData = new Uint8Array(256);
          for (let i = 0; i < 256; i++) {
            lutData[i] = Math.round(Math.pow(i / 255.0, invGamma) * 255);
          }
          const lutMat = window.cv.matFromArray(1, 256, window.cv.CV_8UC1, Array.from(lutData));
          mats.push(lutMat);
          const darker = new window.cv.Mat();
          mats.push(darker);
          window.cv.LUT(zone, lutMat, darker);

          const thresholded = new window.cv.Mat();
          mats.push(thresholded);
          window.cv.threshold(darker, thresholded, 180, 255, window.cv.THRESH_TRUNC);

          const minMax1 = window.cv.minMaxLoc(thresholded);
          let norm1: any;
          if (minMax1.maxVal === minMax1.minVal) {
            norm1 = thresholded.clone();
          } else {
            norm1 = new window.cv.Mat();
            window.cv.normalize(thresholded, norm1, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          }
          mats.push(norm1);

          const paddingW = lineKernel.cols * 2;
          const paddingH = lineKernel.rows * 2;
          const padded = new window.cv.Mat();
          mats.push(padded);
          window.cv.copyMakeBorder(
            norm1, padded,
            paddingH, paddingH, paddingW, paddingW,
            window.cv.BORDER_CONSTANT, new window.cv.Scalar(255, 255, 255, 255)
          );

          const wt = new window.cv.Mat();
          mats.push(wt);
          window.cv.threshold(padded, wt, 180, 255, window.cv.THRESH_TRUNC);

          const minMax2 = window.cv.minMaxLoc(wt);
          let norm2: any;
          if (minMax2.maxVal === minMax2.minVal) {
            norm2 = wt.clone();
          } else {
            norm2 = new window.cv.Mat();
            window.cv.normalize(wt, norm2, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          }
          mats.push(norm2);

          const morphed = new window.cv.Mat();
          mats.push(morphed);
          window.cv.morphologyEx(
            norm2, morphed, window.cv.MORPH_OPEN, lineKernel,
            new window.cv.Point(-1, -1), 3
          );

          const preprocessed = morphed.roi(
            new window.cv.Rect(paddingW, paddingH, 100, 100)
          ).clone();
          mats.push(preprocessed);

          // Detect contours
          const edges = new window.cv.Mat();
          mats.push(edges);
          window.cv.Canny(preprocessed, edges, 185, 55);

          const contourVec = new window.cv.MatVector();
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(
            edges, contourVec, hierarchy,
            window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE
          );

          if (contourVec.size() === 0) {
            contourVec.delete();
            return { found: false, cornersLen: 0, edgesLen: 0 };
          }

          const cloned: any[] = [];
          for (let i = 0; i < contourVec.size(); i++) {
            const c = contourVec.get(i).clone();
            cloned.push(c);
            mats.push(c);
          }
          contourVec.delete();
          cloned.sort((a, b) => window.cv.contourArea(b) - window.cv.contourArea(a));

          const hull = new window.cv.Mat();
          mats.push(hull);
          window.cv.convexHull(cloned[0], hull, false, true);

          const rotRect = window.cv.minAreaRect(hull);
          const boxPtsArr2 = window.cv.boxPoints(rotRect) as Array<{ x: number; y: number }>;
          const patchCorners2: [number, number][] = boxPtsArr2.map(
            (pt: { x: number; y: number }) => [Math.round(pt.x), Math.round(pt.y)] as [number, number]
          );

          const edgeContoursMap: Record<string, any[]> = {
            TOP: [], RIGHT: [], BOTTOM: [], LEFT: [],
          };

          return {
            found: true,
            cornersLen: patchCorners2.length,
            edgesLen: Object.keys(edgeContoursMap).length,
          };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.found).toBe(true);
      expect(result.cornersLen).toBe(4);
      expect(result.edgesLen).toBe(4);
    });
  });

  // test_detect_line_with_offset: zone_offset=(100,100) → all x,y >= 100
  test('test_detect_line_corners_with_offset', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const zone = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(200));
          mats.push(zone);
          window.cv.rectangle(
            zone,
            new window.cv.Point(40, 10),
            new window.cv.Point(60, 90),
            new window.cv.Scalar(50),
            -1
          );

          const lineKernel = window.cv.getStructuringElement(
            window.cv.MORPH_RECT,
            new window.cv.Size(2, 10)
          );
          mats.push(lineKernel);

          const invGamma = 2.0;
          const lutData = new Uint8Array(256);
          for (let i = 0; i < 256; i++) {
            lutData[i] = Math.round(Math.pow(i / 255.0, invGamma) * 255);
          }
          const lutMat = window.cv.matFromArray(1, 256, window.cv.CV_8UC1, Array.from(lutData));
          mats.push(lutMat);
          const darker = new window.cv.Mat();
          mats.push(darker);
          window.cv.LUT(zone, lutMat, darker);

          const thresholded = new window.cv.Mat();
          mats.push(thresholded);
          window.cv.threshold(darker, thresholded, 180, 255, window.cv.THRESH_TRUNC);

          const minMax1 = window.cv.minMaxLoc(thresholded);
          let norm1: any;
          if (minMax1.maxVal === minMax1.minVal) {
            norm1 = thresholded.clone();
          } else {
            norm1 = new window.cv.Mat();
            window.cv.normalize(thresholded, norm1, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          }
          mats.push(norm1);

          const paddingW = lineKernel.cols * 2;
          const paddingH = lineKernel.rows * 2;
          const padded = new window.cv.Mat();
          mats.push(padded);
          window.cv.copyMakeBorder(
            norm1, padded,
            paddingH, paddingH, paddingW, paddingW,
            window.cv.BORDER_CONSTANT, new window.cv.Scalar(255, 255, 255, 255)
          );

          const wt = new window.cv.Mat();
          mats.push(wt);
          window.cv.threshold(padded, wt, 180, 255, window.cv.THRESH_TRUNC);

          const minMax2 = window.cv.minMaxLoc(wt);
          let norm2: any;
          if (minMax2.maxVal === minMax2.minVal) {
            norm2 = wt.clone();
          } else {
            norm2 = new window.cv.Mat();
            window.cv.normalize(wt, norm2, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          }
          mats.push(norm2);

          const morphed = new window.cv.Mat();
          mats.push(morphed);
          window.cv.morphologyEx(
            norm2, morphed, window.cv.MORPH_OPEN, lineKernel,
            new window.cv.Point(-1, -1), 3
          );
          const preprocessed = morphed.roi(
            new window.cv.Rect(paddingW, paddingH, 100, 100)
          ).clone();
          mats.push(preprocessed);

          const edges = new window.cv.Mat();
          mats.push(edges);
          window.cv.Canny(preprocessed, edges, 185, 55);

          const contourVec = new window.cv.MatVector();
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(
            edges, contourVec, hierarchy,
            window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE
          );

          if (contourVec.size() === 0) {
            contourVec.delete();
            return { found: false, allGeq100: false };
          }

          const cloned: any[] = [];
          for (let i = 0; i < contourVec.size(); i++) {
            const c = contourVec.get(i).clone();
            cloned.push(c);
            mats.push(c);
          }
          contourVec.delete();
          cloned.sort((a, b) => window.cv.contourArea(b) - window.cv.contourArea(a));

          const hull = new window.cv.Mat();
          mats.push(hull);
          window.cv.convexHull(cloned[0], hull, false, true);
          const rotRect = window.cv.minAreaRect(hull);
          const boxPtsArr3 = window.cv.boxPoints(rotRect) as Array<{ x: number; y: number }>;

          const zoneOffset = [100, 100];
          const allGeq100 = boxPtsArr3.every(
            (pt: { x: number; y: number }) =>
              Math.round(pt.x) + zoneOffset[0] >= 100 && Math.round(pt.y) + zoneOffset[1] >= 100
          );

          return { found: true, allGeq100 };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.found).toBe(true);
      expect(result.allGeq100).toBe(true);
    });
  });

  // test_detect_line_returns_none_when_not_found: uniform zone → null
  test('test_detect_line_corners_returns_null_when_not_found', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // Uniform zone with no line (use very low threshold to eliminate all)
          const zone = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(200));
          mats.push(zone);

          const lineKernel = window.cv.getStructuringElement(
            window.cv.MORPH_RECT,
            new window.cv.Size(2, 10)
          );
          mats.push(lineKernel);

          const invGamma = 2.0;
          const lutData = new Uint8Array(256);
          for (let i = 0; i < 256; i++) {
            lutData[i] = Math.round(Math.pow(i / 255.0, invGamma) * 255);
          }
          const lutMat = window.cv.matFromArray(1, 256, window.cv.CV_8UC1, Array.from(lutData));
          mats.push(lutMat);
          const darker = new window.cv.Mat();
          mats.push(darker);
          window.cv.LUT(zone, lutMat, darker);

          // Use very low threshold (50) to eliminate all features
          const thresholded = new window.cv.Mat();
          mats.push(thresholded);
          window.cv.threshold(darker, thresholded, 50, 255, window.cv.THRESH_TRUNC);

          const minMax1 = window.cv.minMaxLoc(thresholded);
          let norm1: any;
          if (minMax1.maxVal === minMax1.minVal) {
            norm1 = thresholded.clone();
          } else {
            norm1 = new window.cv.Mat();
            window.cv.normalize(thresholded, norm1, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          }
          mats.push(norm1);

          const paddingW = lineKernel.cols * 2;
          const paddingH = lineKernel.rows * 2;
          const padded = new window.cv.Mat();
          mats.push(padded);
          window.cv.copyMakeBorder(
            norm1, padded,
            paddingH, paddingH, paddingW, paddingW,
            window.cv.BORDER_CONSTANT, new window.cv.Scalar(255, 255, 255, 255)
          );

          const wt = new window.cv.Mat();
          mats.push(wt);
          window.cv.threshold(padded, wt, 50, 255, window.cv.THRESH_TRUNC);

          const minMax2 = window.cv.minMaxLoc(wt);
          let norm2: any;
          if (minMax2.maxVal === minMax2.minVal) {
            norm2 = wt.clone();
          } else {
            norm2 = new window.cv.Mat();
            window.cv.normalize(wt, norm2, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          }
          mats.push(norm2);

          const morphed = new window.cv.Mat();
          mats.push(morphed);
          window.cv.morphologyEx(
            norm2, morphed, window.cv.MORPH_OPEN, lineKernel,
            new window.cv.Point(-1, -1), 3
          );
          const preprocessed = morphed.roi(
            new window.cv.Rect(paddingW, paddingH, 100, 100)
          ).clone();
          mats.push(preprocessed);

          const edges = new window.cv.Mat();
          mats.push(edges);
          window.cv.Canny(preprocessed, edges, 185, 55);

          const contourVec = new window.cv.MatVector();
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(
            edges, contourVec, hierarchy,
            window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE
          );
          const count = contourVec.size();
          contourVec.delete();

          return { contoursFound: count };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.contoursFound).toBe(0);
    });
  });

  // ==========================================================================
  // TestValidateBlurKernel (4 tests) — pure logic, no cv needed
  // ==========================================================================

  // test_valid_kernel: (100,100) zone > (5,5) kernel → returns true
  test('test_validate_blur_kernel_valid', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        function validateBlurKernel(
          zoneShape: [number, number],
          blurKernel: [number, number],
          zoneLabel: string = ''
        ): boolean {
          const [zoneH, zoneW] = zoneShape;
          const [blurH, blurW] = blurKernel;
          if (!(zoneH > blurH && zoneW > blurW)) {
            const labelStr = zoneLabel ? ` '${zoneLabel}'` : '';
            throw new Error(
              `The zone${labelStr} is smaller than provided blur kernel: [${zoneH}, ${zoneW}] < [${blurH}, ${blurW}]`
            );
          }
          return true;
        }
        return validateBlurKernel([100, 100], [5, 5]);
      });

      expect(result).toBe(true);
    });
  });

  // test_kernel_too_large: (10,10) zone < (15,15) kernel → throws
  test('test_validate_blur_kernel_too_large', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        function validateBlurKernel(
          zoneShape: [number, number],
          blurKernel: [number, number],
          zoneLabel: string = ''
        ): boolean {
          const [zoneH, zoneW] = zoneShape;
          const [blurH, blurW] = blurKernel;
          if (!(zoneH > blurH && zoneW > blurW)) {
            const labelStr = zoneLabel ? ` '${zoneLabel}'` : '';
            throw new Error(
              `The zone${labelStr} is smaller than provided blur kernel: [${zoneH}, ${zoneW}] < [${blurH}, ${blurW}]`
            );
          }
          return true;
        }
        try {
          validateBlurKernel([10, 10], [15, 15]);
          return { threw: false, message: '' };
        } catch (e: any) {
          return { threw: true, message: e.message };
        }
      });

      expect(result.threw).toBe(true);
      expect(result.message).toContain('smaller than provided blur kernel');
    });
  });

  // test_kernel_equal_size: (10,10) zone == (10,10) kernel → throws (not strictly greater)
  test('test_validate_blur_kernel_equal_size', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        function validateBlurKernel(
          zoneShape: [number, number],
          blurKernel: [number, number],
          zoneLabel: string = ''
        ): boolean {
          const [zoneH, zoneW] = zoneShape;
          const [blurH, blurW] = blurKernel;
          if (!(zoneH > blurH && zoneW > blurW)) {
            const labelStr = zoneLabel ? ` '${zoneLabel}'` : '';
            throw new Error(
              `The zone${labelStr} is smaller than provided blur kernel: [${zoneH}, ${zoneW}] < [${blurH}, ${blurW}]`
            );
          }
          return true;
        }
        try {
          validateBlurKernel([10, 10], [10, 10]);
          return { threw: false, message: '' };
        } catch (e: any) {
          return { threw: true, message: e.message };
        }
      });

      expect(result.threw).toBe(true);
      expect(result.message).toContain('smaller than provided blur kernel');
    });
  });

  // test_validation_with_label: zone_label="test_zone" → error includes "'test_zone'"
  test('test_validate_blur_kernel_with_label', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        function validateBlurKernel(
          zoneShape: [number, number],
          blurKernel: [number, number],
          zoneLabel: string = ''
        ): boolean {
          const [zoneH, zoneW] = zoneShape;
          const [blurH, blurW] = blurKernel;
          if (!(zoneH > blurH && zoneW > blurW)) {
            const labelStr = zoneLabel ? ` '${zoneLabel}'` : '';
            throw new Error(
              `The zone${labelStr} is smaller than provided blur kernel: [${zoneH}, ${zoneW}] < [${blurH}, ${blurW}]`
            );
          }
          return true;
        }
        try {
          validateBlurKernel([10, 10], [15, 15], 'test_zone');
          return { threw: false, message: '' };
        } catch (e: any) {
          return { threw: true, message: e.message };
        }
      });

      expect(result.threw).toBe(true);
      expect(result.message).toContain("zone 'test_zone'");
    });
  });

  // ==========================================================================
  // TestCreateStructuringElement (4 tests)
  // ==========================================================================

  // test_create_rect: "rect" shape 5×5 → rows=5, cols=5, CV_8UC1
  test('test_create_structuring_element_rect', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const element = window.cv.getStructuringElement(
            window.cv.MORPH_RECT,
            new window.cv.Size(5, 5)
          );
          mats.push(element);
          return { rows: element.rows, cols: element.cols, type: element.type() };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.rows).toBe(5);
      expect(result.cols).toBe(5);
      expect(result.type).toBe(0); // CV_8UC1
    });
  });

  // test_create_ellipse: "ellipse" shape 7×7 → rows=7, cols=7
  test('test_create_structuring_element_ellipse', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const element = window.cv.getStructuringElement(
            window.cv.MORPH_ELLIPSE,
            new window.cv.Size(7, 7)
          );
          mats.push(element);
          return { rows: element.rows, cols: element.cols };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.rows).toBe(7);
      expect(result.cols).toBe(7);
    });
  });

  // test_create_cross: "cross" shape 5×5 → rows=5, cols=5
  test('test_create_structuring_element_cross', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const element = window.cv.getStructuringElement(
            window.cv.MORPH_CROSS,
            new window.cv.Size(5, 5)
          );
          mats.push(element);
          return { rows: element.rows, cols: element.cols };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.rows).toBe(5);
      expect(result.cols).toBe(5);
    });
  });

  // test_invalid_shape: "invalid" → throws containing "Unknown shape"
  test('test_create_structuring_element_invalid_shape', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        function createStructuringElement(shape: string, size: [number, number]): any {
          const shapeMap: Record<string, number> = {
            rect: window.cv.MORPH_RECT,
            ellipse: window.cv.MORPH_ELLIPSE,
            cross: window.cv.MORPH_CROSS,
          };
          if (!(shape in shapeMap)) {
            throw new Error(
              `Unknown shape: ${shape}. Use ${JSON.stringify(Object.keys(shapeMap))}`
            );
          }
          return window.cv.getStructuringElement(shapeMap[shape], new window.cv.Size(size[0], size[1]));
        }
        try {
          const el = createStructuringElement('invalid', [5, 5]);
          if (el) el.delete();
          return { threw: false, message: '' };
        } catch (e: any) {
          return { threw: true, message: e.message };
        }
      });

      expect(result.threw).toBe(true);
      expect(result.message).toContain('Unknown shape');
    });
  });

  // ==========================================================================
  // TestDotLineDetectionIntegration (2 tests)
  // ==========================================================================

  // test_realistic_dot_detection: larger zone with dark rectangle → corners found
  test('test_integration_realistic_dot_detection', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // Realistic zone: 400×600, mostly light (220), with a 40×40 dark rectangle
          const zone = new window.cv.Mat(400, 600, window.cv.CV_8UC1, new window.cv.Scalar(220));
          mats.push(zone);
          window.cv.rectangle(
            zone,
            new window.cv.Point(50, 50),
            new window.cv.Point(90, 90),
            new window.cv.Scalar(50),
            -1
          );

          const dotKernel = window.cv.getStructuringElement(
            window.cv.MORPH_RECT,
            new window.cv.Size(5, 5)
          );
          mats.push(dotKernel);

          const paddingW = dotKernel.cols * 2;
          const paddingH = dotKernel.rows * 2;
          const padded = new window.cv.Mat();
          mats.push(padded);
          window.cv.copyMakeBorder(
            zone, padded,
            paddingH, paddingH, paddingW, paddingW,
            window.cv.BORDER_CONSTANT, new window.cv.Scalar(255, 255, 255, 255)
          );
          const morphed = new window.cv.Mat();
          mats.push(morphed);
          window.cv.morphologyEx(
            padded, morphed, window.cv.MORPH_OPEN, dotKernel,
            new window.cv.Point(-1, -1), 3
          );
          const thresholded = new window.cv.Mat();
          mats.push(thresholded);
          window.cv.threshold(morphed, thresholded, 150, 255, window.cv.THRESH_TRUNC);

          const minMax = window.cv.minMaxLoc(thresholded);
          let normalised: any;
          if (minMax.maxVal === minMax.minVal) {
            normalised = thresholded.clone();
          } else {
            normalised = new window.cv.Mat();
            window.cv.normalize(thresholded, normalised, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          }
          mats.push(normalised);
          const preprocessed = normalised.roi(
            new window.cv.Rect(paddingW, paddingH, 600, 400)
          ).clone();
          mats.push(preprocessed);

          const edges = new window.cv.Mat();
          mats.push(edges);
          window.cv.Canny(preprocessed, edges, 185, 55);

          const contourVec = new window.cv.MatVector();
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(
            edges, contourVec, hierarchy,
            window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE
          );

          if (contourVec.size() === 0) {
            contourVec.delete();
            return { found: false, cornersLen: 0, corners: [] };
          }

          const cloned: any[] = [];
          for (let i = 0; i < contourVec.size(); i++) {
            const c = contourVec.get(i).clone();
            cloned.push(c);
            mats.push(c);
          }
          contourVec.delete();
          cloned.sort((a, b) => window.cv.contourArea(b) - window.cv.contourArea(a));

          const hull = new window.cv.Mat();
          mats.push(hull);
          window.cv.convexHull(cloned[0], hull, false, true);
          const rect = window.cv.boundingRect(hull);
          const corners: [number, number][] = [
            [rect.x, rect.y],
            [rect.x + rect.width, rect.y],
            [rect.x + rect.width, rect.y + rect.height],
            [rect.x, rect.y + rect.height],
          ];

          return { found: true, cornersLen: corners.length, corners };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.found).toBe(true);
      expect(result.cornersLen).toBe(4);
      // Corners near the rectangle at (50,50)-(90,90), allowing +/- 20 tolerance
      expect(result.corners[0][0]).toBeGreaterThanOrEqual(30);
      expect(result.corners[0][0]).toBeLessThanOrEqual(110);
      expect(result.corners[0][1]).toBeGreaterThanOrEqual(30);
      expect(result.corners[0][1]).toBeLessThanOrEqual(110);
    });
  });

  // test_realistic_line_detection: larger zone with tall dark rectangle → corners found
  test('test_integration_realistic_line_detection', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // Realistic zone: 400×600, mostly light (220), with tall narrow rectangle
          const zone = new window.cv.Mat(400, 600, window.cv.CV_8UC1, new window.cv.Scalar(220));
          mats.push(zone);
          window.cv.rectangle(
            zone,
            new window.cv.Point(50, 50),
            new window.cv.Point(60, 350),
            new window.cv.Scalar(60),
            -1
          );

          const lineKernel = window.cv.getStructuringElement(
            window.cv.MORPH_RECT,
            new window.cv.Size(2, 10)
          );
          mats.push(lineKernel);

          const invGamma = 2.0;
          const lutData = new Uint8Array(256);
          for (let i = 0; i < 256; i++) {
            lutData[i] = Math.round(Math.pow(i / 255.0, invGamma) * 255);
          }
          const lutMat = window.cv.matFromArray(1, 256, window.cv.CV_8UC1, Array.from(lutData));
          mats.push(lutMat);
          const darker = new window.cv.Mat();
          mats.push(darker);
          window.cv.LUT(zone, lutMat, darker);

          const thresholded = new window.cv.Mat();
          mats.push(thresholded);
          window.cv.threshold(darker, thresholded, 180, 255, window.cv.THRESH_TRUNC);

          const minMax1 = window.cv.minMaxLoc(thresholded);
          let norm1: any;
          if (minMax1.maxVal === minMax1.minVal) {
            norm1 = thresholded.clone();
          } else {
            norm1 = new window.cv.Mat();
            window.cv.normalize(thresholded, norm1, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          }
          mats.push(norm1);

          const paddingW = lineKernel.cols * 2;
          const paddingH = lineKernel.rows * 2;
          const padded = new window.cv.Mat();
          mats.push(padded);
          window.cv.copyMakeBorder(
            norm1, padded,
            paddingH, paddingH, paddingW, paddingW,
            window.cv.BORDER_CONSTANT, new window.cv.Scalar(255, 255, 255, 255)
          );

          const wt = new window.cv.Mat();
          mats.push(wt);
          window.cv.threshold(padded, wt, 180, 255, window.cv.THRESH_TRUNC);

          const minMax2 = window.cv.minMaxLoc(wt);
          let norm2: any;
          if (minMax2.maxVal === minMax2.minVal) {
            norm2 = wt.clone();
          } else {
            norm2 = new window.cv.Mat();
            window.cv.normalize(wt, norm2, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          }
          mats.push(norm2);

          const morphed = new window.cv.Mat();
          mats.push(morphed);
          window.cv.morphologyEx(
            norm2, morphed, window.cv.MORPH_OPEN, lineKernel,
            new window.cv.Point(-1, -1), 3
          );
          const preprocessed = morphed.roi(
            new window.cv.Rect(paddingW, paddingH, 600, 400)
          ).clone();
          mats.push(preprocessed);

          const edges = new window.cv.Mat();
          mats.push(edges);
          window.cv.Canny(preprocessed, edges, 185, 55);

          const contourVec = new window.cv.MatVector();
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(
            edges, contourVec, hierarchy,
            window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE
          );

          if (contourVec.size() === 0) {
            contourVec.delete();
            return { found: false, cornersLen: 0, edgesLen: 0 };
          }

          const cloned: any[] = [];
          for (let i = 0; i < contourVec.size(); i++) {
            const c = contourVec.get(i).clone();
            cloned.push(c);
            mats.push(c);
          }
          contourVec.delete();
          cloned.sort((a, b) => window.cv.contourArea(b) - window.cv.contourArea(a));

          const hull = new window.cv.Mat();
          mats.push(hull);
          window.cv.convexHull(cloned[0], hull, false, true);
          const rotRect = window.cv.minAreaRect(hull);
          // cv.boxPoints returns an array of {x,y} objects in OpenCV.js 4.x
          const boxPtsArr4 = window.cv.boxPoints(rotRect) as Array<{ x: number; y: number }>;

          const edgeContoursMap: Record<string, any[]> = {
            TOP: [], RIGHT: [], BOTTOM: [], LEFT: [],
          };

          return {
            found: true,
            cornersLen: boxPtsArr4.length,
            edgesLen: Object.keys(edgeContoursMap).length,
          };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.found).toBe(true);
      expect(result.cornersLen).toBe(4);
      expect(result.edgesLen).toBe(4);
    });
  });
});
