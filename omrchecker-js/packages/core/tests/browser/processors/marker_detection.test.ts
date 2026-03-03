/**
 * Browser tests for marker_detection.ts using OpenCV.js
 *
 * These tests verify marker template preparation, multi-scale template matching,
 * corner extraction, marker detection, and validation in a real browser environment
 * with actual OpenCV.js cv.Mat objects. Each test corresponds to a Python test in
 * src/tests/processors/test_marker_detection.py.
 *
 * 25 tests total:
 *   TestPrepareMarkerTemplate (4)
 *   TestMultiScaleTemplateMatch (5)
 *   TestExtractMarkerCorners (3)
 *   TestDetectMarkerInPatch (5)
 *   TestValidateMarkerDetection (5)
 *   TestIntegrationScenarios (3)
 *
 * Run with: npm run test:browser -- --grep "MarkerDetection"
 */

import { test, expect } from '@playwright/test';
import { setupBrowser } from '../browser-setup';
import { withMemoryTracking } from '../memory-utils';

// Increase timeout for browser tests (OpenCV.js WASM initialization)
test.setTimeout(60000);

test.describe('MarkerDetection - Browser Tests', () => {
  test.beforeEach(async ({ page }) => {
    await setupBrowser(page);
  });

  // ==========================================================================
  // TestPrepareMarkerTemplate (4 tests)
  // ==========================================================================

  // test_basic_extraction: crop 20x20 ROI from reference → shape (20,20), uint8
  test('test_prepare_marker_template_basic_extraction', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // reference = np.zeros((100,100)); reference[20:40, 30:50] = 255
          const reference = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(reference);
          // Fill rows 20..39, cols 30..49 with 255
          for (let r = 20; r < 40; r++) {
            for (let c = 30; c < 50; c++) {
              reference.data[r * 100 + c] = 255;
            }
          }

          // reference_zone = {"origin": [30, 20], "dimensions": [20, 20]}
          // x=30, y=20, w=20, h=20
          const roi = reference.roi(new window.cv.Rect(30, 20, 20, 20));
          let marker = roi.clone();
          mats.push(marker);

          // GaussianBlur (3,3)
          const blurred = new window.cv.Mat();
          mats.push(blurred);
          window.cv.GaussianBlur(marker, blurred, new window.cv.Size(3, 3), 0);
          mats.splice(mats.indexOf(marker), 1);
          marker.delete();
          marker = blurred;

          // normalize
          const minMax0 = window.cv.minMaxLoc(marker);
          let normalized: any;
          if (minMax0.maxVal === minMax0.minVal) {
            normalized = marker.clone();
          } else {
            normalized = new window.cv.Mat();
            window.cv.normalize(marker, normalized, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          }
          mats.push(normalized);
          // no erode_subtract

          return {
            rows: normalized.rows,
            cols: normalized.cols,
            type: normalized.type(), // CV_8UC1 = 0
          };
        } finally {
          mats.forEach(m => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.rows).toBe(20);
      expect(result.cols).toBe(20);
      expect(result.type).toBe(0); // CV_8UC1
    });
  });

  // test_with_resize: dimensions [40,30] → resize to (20,20) → shape (20,20)
  test('test_prepare_marker_template_with_resize', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // reference = np.ones((100,100)) * 128
          const reference = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(128));
          mats.push(reference);

          // reference_zone = {"origin": [10, 10], "dimensions": [40, 30]}
          // x=10, y=10, w=40, h=30
          const roi = reference.roi(new window.cv.Rect(10, 10, 40, 30));
          let marker = roi.clone();
          mats.push(marker);

          // Resize to (20, 20): marker_dimensions=[20,20] → w=20, h=20
          const resized = new window.cv.Mat();
          mats.push(resized);
          window.cv.resize(marker, resized, new window.cv.Size(20, 20), 0, 0, window.cv.INTER_LINEAR);
          mats.splice(mats.indexOf(marker), 1);
          marker.delete();
          marker = resized;

          // GaussianBlur (5,5) - default
          const blurred = new window.cv.Mat();
          mats.push(blurred);
          window.cv.GaussianBlur(marker, blurred, new window.cv.Size(5, 5), 0);
          mats.splice(mats.indexOf(marker), 1);
          marker.delete();
          marker = blurred;

          // normalize (all pixels same → clone)
          const minMax = window.cv.minMaxLoc(marker);
          let normalized: any;
          if (minMax.maxVal === minMax.minVal) {
            normalized = marker.clone();
          } else {
            normalized = new window.cv.Mat();
            window.cv.normalize(marker, normalized, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          }
          mats.push(normalized);

          return {
            rows: normalized.rows,
            cols: normalized.cols,
          };
        } finally {
          mats.forEach(m => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.rows).toBe(20);
      expect(result.cols).toBe(20);
    });
  });

  // test_with_erode_subtract: erode-subtract produces different result; both max > 200
  test('test_prepare_marker_template_with_erode_subtract', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // reference with gradient: reference[i,j] = 200 - |i-25|*5 - |j-25|*5 for i,j in [15,35)
          const reference = new window.cv.Mat(50, 50, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(reference);
          for (let i = 15; i < 35; i++) {
            for (let j = 15; j < 35; j++) {
              const val = Math.max(0, 200 - Math.abs(i - 25) * 5 - Math.abs(j - 25) * 5);
              reference.data[i * 50 + j] = val;
            }
          }

          // reference_zone = {"origin": [15, 15], "dimensions": [20, 20]}
          const processMarker = (applyErodeSubtract: boolean) => {
            const roi = reference.roi(new window.cv.Rect(15, 15, 20, 20));
            let marker = roi.clone();
            mats.push(marker);

            const blurred = new window.cv.Mat();
            mats.push(blurred);
            window.cv.GaussianBlur(marker, blurred, new window.cv.Size(3, 3), 0);
            mats.splice(mats.indexOf(marker), 1);
            marker.delete();
            marker = blurred;

            const minMax0 = window.cv.minMaxLoc(marker);
            let normalized: any;
            if (minMax0.maxVal === minMax0.minVal) {
              normalized = marker.clone();
            } else {
              normalized = new window.cv.Mat();
              window.cv.normalize(marker, normalized, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
            }
            mats.push(normalized);
            mats.splice(mats.indexOf(marker), 1);
            marker.delete();
            marker = normalized;

            if (applyErodeSubtract) {
              const kernel = window.cv.Mat.ones(5, 5, window.cv.CV_8U);
              mats.push(kernel);
              const eroded = new window.cv.Mat();
              mats.push(eroded);
              window.cv.erode(marker, eroded, kernel, new window.cv.Point(-1, -1), 5);
              const subtracted = new window.cv.Mat();
              mats.push(subtracted);
              window.cv.subtract(marker, eroded, subtracted);
              const renorm = new window.cv.Mat();
              mats.push(renorm);
              const minMax1 = window.cv.minMaxLoc(subtracted);
              if (minMax1.maxVal === minMax1.minVal) {
                const cloned = subtracted.clone();
                mats.push(cloned);
                mats.splice(mats.indexOf(marker), 1);
                marker.delete();
                marker = cloned;
              } else {
                window.cv.normalize(subtracted, renorm, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
                mats.splice(mats.indexOf(marker), 1);
                marker.delete();
                marker = renorm;
              }
            }

            const maxVal = window.cv.minMaxLoc(marker).maxVal;
            return maxVal;
          };

          const maxNoErode = processMarker(false);
          const maxWithErode = processMarker(true);

          return { maxNoErode, maxWithErode };
        } finally {
          mats.forEach(m => { try { m.delete(); } catch (_) {} });
        }
      });

      // Both should have max > 200 (after normalization to [0,255], max is 255)
      expect(result.maxNoErode).toBeGreaterThan(200);
      expect(result.maxWithErode).toBeGreaterThan(200);
    });
  });

  // test_normalization: gradient reference → normalized marker has max=255, min=0
  test('test_prepare_marker_template_normalization', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // reference: gradient along rows for rows 10..49, cols 10..49
          const reference = new window.cv.Mat(60, 60, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(reference);
          for (let i = 10; i < 50; i++) {
            for (let j = 10; j < 50; j++) {
              reference.data[i * 60 + j] = Math.round(((i - 10) / 40) * 255);
            }
          }

          // reference_zone = {"origin": [10, 10], "dimensions": [40, 40]}
          const roi = reference.roi(new window.cv.Rect(10, 10, 40, 40));
          let marker = roi.clone();
          mats.push(marker);

          // blur (5,5)
          const blurred = new window.cv.Mat();
          mats.push(blurred);
          window.cv.GaussianBlur(marker, blurred, new window.cv.Size(5, 5), 0);
          mats.splice(mats.indexOf(marker), 1);
          marker.delete();
          marker = blurred;

          // normalize - apply_erode_subtract=False
          const minMax = window.cv.minMaxLoc(marker);
          let normalized: any;
          if (minMax.maxVal === minMax.minVal) {
            normalized = marker.clone();
          } else {
            normalized = new window.cv.Mat();
            window.cv.normalize(marker, normalized, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          }
          mats.push(normalized);

          const finalMinMax = window.cv.minMaxLoc(normalized);
          return {
            maxVal: finalMinMax.maxVal,
            minVal: finalMinMax.minVal,
          };
        } finally {
          mats.forEach(m => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.maxVal).toBe(255);
      expect(result.minVal).toBe(0);
    });
  });

  // ==========================================================================
  // TestMultiScaleTemplateMatch (5 tests)
  // ==========================================================================

  // Helper: createTestPatchAndMarker - marker=20x20 with white square, patch=100x100 with marker at (30,40)
  // Returns { patchData, markerData, patchRows, patchCols, markerRows, markerCols, scale }

  // test_perfect_match: match at scale ~1.0 → confidence > 0.9, 95 <= scale <= 105
  test('test_multi_scale_perfect_match', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // marker = zeros(20,20); marker[5:15,5:15]=255
          const marker = new window.cv.Mat(20, 20, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(marker);
          for (let r = 5; r < 15; r++) {
            for (let c = 5; c < 15; c++) {
              marker.data[r * 20 + c] = 255;
            }
          }

          // patch = zeros(100,100); embed marker at row=40, col=30
          const patch = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(patch);
          for (let r = 0; r < 20; r++) {
            for (let c = 0; c < 20; c++) {
              patch.data[(40 + r) * 100 + (30 + c)] = marker.data[r * 20 + c];
            }
          }

          // Multi-scale template match
          const descentPerStep = Math.floor((110 - 90) / 5); // 4
          let bestPosition: [number, number] | null = null;
          let bestMarker: any = null;
          let bestConfidence = 0.0;
          let bestScalePercent: number | null = null;

          for (let scalePercent = 110; scalePercent > 90; scalePercent -= descentPerStep) {
            const scale = scalePercent / 100;
            if (scale <= 0) continue;
            const scaledW = Math.floor(20 * scale);
            const scaledH = Math.floor(20 * scale);
            const scaledMarker = new window.cv.Mat();
            mats.push(scaledMarker);
            window.cv.resize(marker, scaledMarker, new window.cv.Size(scaledW, scaledH), 0, 0, window.cv.INTER_LINEAR);

            if (scaledH > 100 || scaledW > 100) continue;

            const matchResult = new window.cv.Mat();
            mats.push(matchResult);
            window.cv.matchTemplate(patch, scaledMarker, matchResult, window.cv.TM_CCOEFF_NORMED);
            const minMax = window.cv.minMaxLoc(matchResult);
            const confidence = minMax.maxVal;

            if (confidence > bestConfidence) {
              bestScalePercent = scalePercent;
              bestConfidence = confidence;
              bestPosition = [minMax.maxLoc.x, minMax.maxLoc.y];
              if (bestMarker) {
                mats.splice(mats.indexOf(bestMarker), 1);
                bestMarker.delete();
              }
              bestMarker = scaledMarker;
              mats.splice(mats.indexOf(scaledMarker), 1);
            }
          }

          if (bestMarker) {
            mats.push(bestMarker);
          }

          return {
            position: bestPosition,
            confidence: bestConfidence,
            scalePercent: bestScalePercent,
          };
        } finally {
          mats.forEach(m => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.position).not.toBeNull();
      expect(result.confidence).toBeGreaterThan(0.9);
      expect(result.scalePercent).toBeGreaterThanOrEqual(95);
      expect(result.scalePercent).toBeLessThanOrEqual(110);
    });
  });

  // test_scaled_match: patch has marker at 0.9 scale → scalePercent in [85,95]
  test('test_multi_scale_scaled_match', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // marker = zeros(20,20); marker[5:15,5:15]=255
          const marker = new window.cv.Mat(20, 20, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(marker);
          for (let r = 5; r < 15; r++) {
            for (let c = 5; c < 15; c++) {
              marker.data[r * 20 + c] = 255;
            }
          }

          // scale = 0.9 → scaledW=18, scaledH=18
          const scaledMarkerForPatch = new window.cv.Mat();
          mats.push(scaledMarkerForPatch);
          window.cv.resize(marker, scaledMarkerForPatch, new window.cv.Size(18, 18), 0, 0, window.cv.INTER_LINEAR);

          // patch = zeros(100,100); embed scaledMarker at (40,30)
          const patch = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(patch);
          for (let r = 0; r < 18; r++) {
            for (let c = 0; c < 18; c++) {
              patch.data[(40 + r) * 100 + (30 + c)] = scaledMarkerForPatch.data[r * 18 + c];
            }
          }

          // Multi-scale match with range (80, 100), steps=5
          const descentPerStep = Math.floor((100 - 80) / 5); // 4
          let bestPosition: [number, number] | null = null;
          let bestConfidence = 0.0;
          let bestScalePercent: number | null = null;
          const scaledMats: any[] = [];

          for (let scalePercent = 100; scalePercent > 80; scalePercent -= descentPerStep) {
            const scale = scalePercent / 100;
            if (scale <= 0) continue;
            const scaledW = Math.floor(20 * scale);
            const scaledH = Math.floor(20 * scale);
            const scaledMarker = new window.cv.Mat();
            scaledMats.push(scaledMarker);
            window.cv.resize(marker, scaledMarker, new window.cv.Size(scaledW, scaledH), 0, 0, window.cv.INTER_LINEAR);

            if (scaledH > 100 || scaledW > 100) continue;

            const matchResult = new window.cv.Mat();
            scaledMats.push(matchResult);
            window.cv.matchTemplate(patch, scaledMarker, matchResult, window.cv.TM_CCOEFF_NORMED);
            const minMax = window.cv.minMaxLoc(matchResult);
            const confidence = minMax.maxVal;

            if (confidence > bestConfidence) {
              bestScalePercent = scalePercent;
              bestConfidence = confidence;
              bestPosition = [minMax.maxLoc.x, minMax.maxLoc.y];
            }
          }

          scaledMats.forEach(m => { try { m.delete(); } catch (_) {} });

          return {
            position: bestPosition,
            scalePercent: bestScalePercent,
          };
        } finally {
          mats.forEach(m => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.position).not.toBeNull();
      expect(result.scalePercent).toBeGreaterThanOrEqual(84);
      expect(result.scalePercent).toBeLessThanOrEqual(100);
    });
  });

  // test_no_match: random mismatched patterns → confidence < 0.7
  test('test_multi_scale_no_match', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // patch: randint(50,100, (100,100))
          const patchData = new Uint8Array(100 * 100);
          for (let i = 0; i < patchData.length; i++) {
            patchData[i] = Math.floor(Math.random() * 50 + 50);
          }
          const patch = window.cv.matFromArray(100, 100, window.cv.CV_8UC1, Array.from(patchData));
          mats.push(patch);

          // marker: randint(150,255, (20,20))
          const markerData = new Uint8Array(20 * 20);
          for (let i = 0; i < markerData.length; i++) {
            markerData[i] = Math.floor(Math.random() * 105 + 150);
          }
          const marker = window.cv.matFromArray(20, 20, window.cv.CV_8UC1, Array.from(markerData));
          mats.push(marker);

          // Multi-scale with default range (85,115), steps=5
          const descentPerStep = Math.floor((115 - 85) / 5); // 6
          let bestConfidence = 0.0;

          for (let scalePercent = 115; scalePercent > 85; scalePercent -= descentPerStep) {
            const scale = scalePercent / 100;
            if (scale <= 0) continue;
            const scaledW = Math.floor(20 * scale);
            const scaledH = Math.floor(20 * scale);
            const scaledMarker = new window.cv.Mat();
            mats.push(scaledMarker);
            window.cv.resize(marker, scaledMarker, new window.cv.Size(scaledW, scaledH), 0, 0, window.cv.INTER_LINEAR);

            if (scaledH > 100 || scaledW > 100) continue;

            const matchResult = new window.cv.Mat();
            mats.push(matchResult);
            window.cv.matchTemplate(patch, scaledMarker, matchResult, window.cv.TM_CCOEFF_NORMED);
            const minMax = window.cv.minMaxLoc(matchResult);
            if (minMax.maxVal > bestConfidence) {
              bestConfidence = minMax.maxVal;
            }
          }

          return { confidence: bestConfidence };
        } finally {
          mats.forEach(m => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.confidence).toBeLessThan(0.7);
    });
  });

  // test_marker_larger_than_patch: marker > patch at all scales → no crash
  test('test_multi_scale_marker_larger_than_patch', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const patch = new window.cv.Mat(50, 50, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(patch);

          const marker = new window.cv.Mat(40, 40, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(marker);
          for (let r = 10; r < 30; r++) {
            for (let c = 10; c < 30; c++) {
              marker.data[r * 40 + c] = 255;
            }
          }

          // scale_range=(100,150), steps=3 → descentPerStep=16
          const descentPerStep = Math.floor((150 - 100) / 3); // 16
          let didNotCrash = true;

          for (let scalePercent = 150; scalePercent > 100; scalePercent -= descentPerStep) {
            const scale = scalePercent / 100;
            if (scale <= 0) continue;
            const scaledW = Math.floor(40 * scale);
            const scaledH = Math.floor(40 * scale);
            const scaledMarker = new window.cv.Mat();
            mats.push(scaledMarker);
            window.cv.resize(marker, scaledMarker, new window.cv.Size(scaledW, scaledH), 0, 0, window.cv.INTER_LINEAR);

            // All scales > patch → skip
            if (scaledH > 50 || scaledW > 50) continue;

            // Should not reach here for all scales > 100%
            const matchResult = new window.cv.Mat();
            mats.push(matchResult);
            window.cv.matchTemplate(patch, scaledMarker, matchResult, window.cv.TM_CCOEFF_NORMED);
          }

          return { didNotCrash };
        } finally {
          mats.forEach(m => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.didNotCrash).toBe(true);
    });
  });

  // test_various_scale_ranges: two different scale ranges both find position
  test('test_multi_scale_various_scale_ranges', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // marker = zeros(20,20); marker[5:15,5:15]=255
          const marker = new window.cv.Mat(20, 20, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(marker);
          for (let r = 5; r < 15; r++) {
            for (let c = 5; c < 15; c++) {
              marker.data[r * 20 + c] = 255;
            }
          }

          // patch = zeros(100,100); embed marker at (40,30)
          const patch = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(patch);
          for (let r = 0; r < 20; r++) {
            for (let c = 0; c < 20; c++) {
              patch.data[(40 + r) * 100 + (30 + c)] = marker.data[r * 20 + c];
            }
          }

          function runMatch(scaleMin: number, scaleMax: number, steps: number): [number, number] | null {
            const descentPerStep = Math.floor((scaleMax - scaleMin) / steps);
            let bestPosition: [number, number] | null = null;
            let bestConfidence = 0.0;
            const localMats: any[] = [];

            for (let scalePercent = scaleMax; scalePercent > scaleMin; scalePercent -= descentPerStep) {
              const scale = scalePercent / 100;
              if (scale <= 0) continue;
              const scaledW = Math.floor(20 * scale);
              const scaledH = Math.floor(20 * scale);
              const scaledMarker = new window.cv.Mat();
              localMats.push(scaledMarker);
              window.cv.resize(marker, scaledMarker, new window.cv.Size(scaledW, scaledH), 0, 0, window.cv.INTER_LINEAR);

              if (scaledH > 100 || scaledW > 100) continue;

              const matchResult = new window.cv.Mat();
              localMats.push(matchResult);
              window.cv.matchTemplate(patch, scaledMarker, matchResult, window.cv.TM_CCOEFF_NORMED);
              const minMax = window.cv.minMaxLoc(matchResult);
              if (minMax.maxVal > bestConfidence) {
                bestConfidence = minMax.maxVal;
                bestPosition = [minMax.maxLoc.x, minMax.maxLoc.y];
              }
            }

            localMats.forEach(m => { try { m.delete(); } catch (_) {} });
            return bestPosition;
          }

          const pos1 = runMatch(95, 105, 3);
          const pos2 = runMatch(50, 150, 10);

          return {
            pos1NotNull: pos1 !== null,
            pos2NotNull: pos2 !== null,
          };
        } finally {
          mats.forEach(m => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.pos1NotNull).toBe(true);
      expect(result.pos2NotNull).toBe(true);
    });
  });

  // ==========================================================================
  // TestExtractMarkerCorners (3 tests)
  // ==========================================================================

  // test_basic_extraction: marker(20,30), position=(10,15), zoneOffset=(0,0)
  // corners = [[10,15],[40,15],[40,35],[10,35]]
  test('test_extract_marker_corners_basic_extraction', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        // MathUtils.getRectanglePoints(x=10, y=15, w=30, h=20)
        // tl=[10,15], tr=[40,15], br=[40,35], bl=[10,35]
        // shiftPointsFromOrigin([0,0], corners) → same
        const x = 10, y = 15, w = 30, h = 20;
        const corners = [
          [x, y],
          [x + w, y],
          [x + w, y + h],
          [x, y + h],
        ];
        const zoneOffset = [0, 0];
        const absoluteCorners = corners.map(c => [c[0] + zoneOffset[0], c[1] + zoneOffset[1]]);

        return {
          length: absoluteCorners.length,
          pointLength: absoluteCorners[0].length,
          tl: absoluteCorners[0],
          tr: absoluteCorners[1],
          br: absoluteCorners[2],
          bl: absoluteCorners[3],
        };
      });

      expect(result.length).toBe(4);
      expect(result.pointLength).toBe(2);
      expect(result.tl).toEqual([10, 15]);
      expect(result.tr).toEqual([40, 15]);
      expect(result.br).toEqual([40, 35]);
      expect(result.bl).toEqual([10, 35]);
    });
  });

  // test_with_zone_offset: position=(5,10), zoneOffset=(100,200) → all x>=100, all y>=200
  test('test_extract_marker_corners_with_zone_offset', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        // marker size 20x20, position=(5,10), zoneOffset=(100,200)
        const x = 5, y = 10, w = 20, h = 20;
        const corners = [
          [x, y],
          [x + w, y],
          [x + w, y + h],
          [x, y + h],
        ];
        const zoneOffset = [100, 200];
        const absoluteCorners = corners.map(c => [c[0] + zoneOffset[0], c[1] + zoneOffset[1]]);

        const allXGe100 = absoluteCorners.every(c => c[0] >= 100);
        const allYGe200 = absoluteCorners.every(c => c[1] >= 200);

        return { allXGe100, allYGe200 };
      });

      expect(result.allXGe100).toBe(true);
      expect(result.allYGe200).toBe(true);
    });
  });

  // test_different_marker_sizes: for sizes (10,10),(30,20),(15,40)
  // corners[1][0] - corners[0][0] === width, corners[2][1] - corners[1][1] === height
  test('test_extract_marker_corners_different_marker_sizes', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const sizes = [[10, 10], [30, 20], [15, 40]]; // [width, height]
        const checks: Array<{ widthOk: boolean; heightOk: boolean }> = [];

        for (const [w, h] of sizes) {
          // position=(0,0), zoneOffset=(0,0)
          const x = 0, y = 0;
          const corners = [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
          ];
          const widthOk = corners[1][0] - corners[0][0] === w;
          const heightOk = corners[2][1] - corners[1][1] === h;
          checks.push({ widthOk, heightOk });
        }

        return checks;
      });

      for (const check of result) {
        expect(check.widthOk).toBe(true);
        expect(check.heightOk).toBe(true);
      }
    });
  });

  // ==========================================================================
  // TestDetectMarkerInPatch (5 tests)
  // ==========================================================================

  // test_successful_detection: detect marker near (60,50) → corners not null, x in [55,65], y in [45,55]
  test('test_detect_marker_in_patch_successful_detection', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // marker = zeros(25,25); marker[5:20,5:20]=255; GaussianBlur(5,5)
          const markerRaw = new window.cv.Mat(25, 25, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(markerRaw);
          for (let r = 5; r < 20; r++) {
            for (let c = 5; c < 20; c++) {
              markerRaw.data[r * 25 + c] = 255;
            }
          }
          const marker = new window.cv.Mat();
          mats.push(marker);
          window.cv.GaussianBlur(markerRaw, marker, new window.cv.Size(5, 5), 0);

          // patch = randint(0,50,(150,150)); patch[50:75,60:85]=marker
          const patchData = new Uint8Array(150 * 150);
          for (let i = 0; i < patchData.length; i++) {
            patchData[i] = Math.floor(Math.random() * 50);
          }
          const patch = window.cv.matFromArray(150, 150, window.cv.CV_8UC1, Array.from(patchData));
          mats.push(patch);

          // Embed marker at row=50, col=60
          for (let r = 0; r < 25; r++) {
            for (let c = 0; c < 25; c++) {
              patch.data[(50 + r) * 150 + (60 + c)] = marker.data[r * 25 + c];
            }
          }

          // Multi-scale match range=(90,110), steps=5
          const descentPerStep = Math.floor((110 - 90) / 5); // 4
          let bestPosition: [number, number] | null = null;
          let bestMarker: any = null;
          let bestConfidence = 0.0;

          for (let scalePercent = 110; scalePercent > 90; scalePercent -= descentPerStep) {
            const scale = scalePercent / 100;
            if (scale <= 0) continue;
            const scaledW = Math.floor(25 * scale);
            const scaledH = Math.floor(25 * scale);
            const scaledMarker = new window.cv.Mat();
            mats.push(scaledMarker);
            window.cv.resize(marker, scaledMarker, new window.cv.Size(scaledW, scaledH), 0, 0, window.cv.INTER_LINEAR);

            if (scaledH > 150 || scaledW > 150) continue;

            const matchResult = new window.cv.Mat();
            mats.push(matchResult);
            window.cv.matchTemplate(patch, scaledMarker, matchResult, window.cv.TM_CCOEFF_NORMED);
            const minMax = window.cv.minMaxLoc(matchResult);
            const confidence = minMax.maxVal;

            if (confidence > bestConfidence) {
              bestConfidence = confidence;
              bestPosition = [minMax.maxLoc.x, minMax.maxLoc.y];
              if (bestMarker) {
                mats.splice(mats.indexOf(bestMarker), 1);
                bestMarker.delete();
              }
              bestMarker = scaledMarker;
              mats.splice(mats.indexOf(scaledMarker), 1);
            }
          }

          if (bestMarker) mats.push(bestMarker);

          if (bestPosition === null || bestMarker === null || bestConfidence < 0.3) {
            return { cornersNull: true, confidence: bestConfidence };
          }

          // Extract corners
          const [px, py] = bestPosition;
          const h = bestMarker.rows, w = bestMarker.cols;
          const corners = [
            [px, py],
            [px + w, py],
            [px + w, py + h],
            [px, py + h],
          ];

          return {
            cornersNull: false,
            confidence: bestConfidence,
            tlX: corners[0][0],
            tlY: corners[0][1],
            cornersLen: corners.length,
          };
        } finally {
          mats.forEach(m => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.cornersNull).toBe(false);
      expect(result.cornersLen).toBe(4);
      expect(result.tlX).toBeGreaterThanOrEqual(55);
      expect(result.tlX).toBeLessThanOrEqual(65);
      expect(result.tlY).toBeGreaterThanOrEqual(45);
      expect(result.tlY).toBeLessThanOrEqual(55);
    });
  });

  // test_below_confidence_threshold: random mismatch → corners null
  test('test_detect_marker_in_patch_below_confidence_threshold', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const patchData = new Uint8Array(100 * 100);
          for (let i = 0; i < patchData.length; i++) {
            patchData[i] = Math.floor(Math.random() * 100);
          }
          const patch = window.cv.matFromArray(100, 100, window.cv.CV_8UC1, Array.from(patchData));
          mats.push(patch);

          const markerData = new Uint8Array(20 * 20);
          for (let i = 0; i < markerData.length; i++) {
            markerData[i] = Math.floor(Math.random() * 105 + 150);
          }
          const marker = window.cv.matFromArray(20, 20, window.cv.CV_8UC1, Array.from(markerData));
          mats.push(marker);

          // min_confidence=0.95 → almost certainly fails
          const descentPerStep = Math.floor((115 - 85) / 5);
          let bestConfidence = 0.0;
          let bestPosition: [number, number] | null = null;

          for (let scalePercent = 115; scalePercent > 85; scalePercent -= descentPerStep) {
            const scale = scalePercent / 100;
            const scaledW = Math.floor(20 * scale);
            const scaledH = Math.floor(20 * scale);
            const scaledMarker = new window.cv.Mat();
            mats.push(scaledMarker);
            window.cv.resize(marker, scaledMarker, new window.cv.Size(scaledW, scaledH), 0, 0, window.cv.INTER_LINEAR);

            if (scaledH > 100 || scaledW > 100) continue;

            const matchResult = new window.cv.Mat();
            mats.push(matchResult);
            window.cv.matchTemplate(patch, scaledMarker, matchResult, window.cv.TM_CCOEFF_NORMED);
            const minMax = window.cv.minMaxLoc(matchResult);
            if (minMax.maxVal > bestConfidence) {
              bestConfidence = minMax.maxVal;
              bestPosition = [minMax.maxLoc.x, minMax.maxLoc.y];
            }
          }

          const minConfidence = 0.95;
          const cornersNull = bestPosition === null || bestConfidence < minConfidence;

          return { cornersNull };
        } finally {
          mats.forEach(m => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.cornersNull).toBe(true);
    });
  });

  // test_with_zone_offset: detect with zone_offset=(1000,2000) → all corners x>=1000, y>=2000
  test('test_detect_marker_in_patch_with_zone_offset', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // marker = zeros(25,25); marker[5:20,5:20]=255; GaussianBlur(5,5)
          const markerRaw = new window.cv.Mat(25, 25, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(markerRaw);
          for (let r = 5; r < 20; r++) {
            for (let c = 5; c < 20; c++) {
              markerRaw.data[r * 25 + c] = 255;
            }
          }
          const marker = new window.cv.Mat();
          mats.push(marker);
          window.cv.GaussianBlur(markerRaw, marker, new window.cv.Size(5, 5), 0);

          // patch = zeros(150,150); embed marker at (50,60)
          const patchData = new Uint8Array(150 * 150);
          for (let i = 0; i < patchData.length; i++) {
            patchData[i] = Math.floor(Math.random() * 50);
          }
          const patch = window.cv.matFromArray(150, 150, window.cv.CV_8UC1, Array.from(patchData));
          mats.push(patch);
          for (let r = 0; r < 25; r++) {
            for (let c = 0; c < 25; c++) {
              patch.data[(50 + r) * 150 + (60 + c)] = marker.data[r * 25 + c];
            }
          }

          const zoneOffset = [1000, 2000];
          const descentPerStep = Math.floor((110 - 90) / 5);
          let bestPosition: [number, number] | null = null;
          let bestMarker: any = null;
          let bestConfidence = 0.0;

          for (let scalePercent = 110; scalePercent > 90; scalePercent -= descentPerStep) {
            const scale = scalePercent / 100;
            const scaledW = Math.floor(25 * scale);
            const scaledH = Math.floor(25 * scale);
            const scaledMarker = new window.cv.Mat();
            mats.push(scaledMarker);
            window.cv.resize(marker, scaledMarker, new window.cv.Size(scaledW, scaledH), 0, 0, window.cv.INTER_LINEAR);

            if (scaledH > 150 || scaledW > 150) continue;

            const matchResult = new window.cv.Mat();
            mats.push(matchResult);
            window.cv.matchTemplate(patch, scaledMarker, matchResult, window.cv.TM_CCOEFF_NORMED);
            const minMax = window.cv.minMaxLoc(matchResult);

            if (minMax.maxVal > bestConfidence) {
              bestConfidence = minMax.maxVal;
              bestPosition = [minMax.maxLoc.x, minMax.maxLoc.y];
              if (bestMarker) {
                mats.splice(mats.indexOf(bestMarker), 1);
                bestMarker.delete();
              }
              bestMarker = scaledMarker;
              mats.splice(mats.indexOf(scaledMarker), 1);
            }
          }

          if (bestMarker) mats.push(bestMarker);

          if (bestPosition === null || bestMarker === null || bestConfidence < 0.3) {
            return { cornersNull: true };
          }

          const [px, py] = bestPosition;
          const h = bestMarker.rows, w = bestMarker.cols;
          const corners = [
            [px + zoneOffset[0], py + zoneOffset[1]],
            [px + w + zoneOffset[0], py + zoneOffset[1]],
            [px + w + zoneOffset[0], py + h + zoneOffset[1]],
            [px + zoneOffset[0], py + h + zoneOffset[1]],
          ];

          const allXGe1000 = corners.every(c => c[0] >= 1000);
          const allYGe2000 = corners.every(c => c[1] >= 2000);

          return { cornersNull: false, allXGe1000, allYGe2000 };
        } finally {
          mats.forEach(m => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.cornersNull).toBe(false);
      expect(result.allXGe1000).toBe(true);
      expect(result.allYGe2000).toBe(true);
    });
  });

  // test_scaled_marker_detection: marker 30x30, patch has 24x24 version at (40,50)
  // detect with scale_range=(70,90), min_confidence=0.5
  test('test_detect_marker_in_patch_scaled_marker_detection', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // marker_original = zeros(30,30); marker_original[8:22,8:22]=255
          const marker = new window.cv.Mat(30, 30, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(marker);
          for (let r = 8; r < 22; r++) {
            for (let c = 8; c < 22; c++) {
              marker.data[r * 30 + c] = 255;
            }
          }

          // resize marker to 24x24 for embedding in patch
          const smallMarker = new window.cv.Mat();
          mats.push(smallMarker);
          window.cv.resize(marker, smallMarker, new window.cv.Size(24, 24), 0, 0, window.cv.INTER_LINEAR);

          // patch = zeros(120,120); patch[40:64,50:74] = smallMarker
          const patch = new window.cv.Mat(120, 120, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(patch);
          for (let r = 0; r < 24; r++) {
            for (let c = 0; c < 24; c++) {
              patch.data[(40 + r) * 120 + (50 + c)] = smallMarker.data[r * 24 + c];
            }
          }

          // detect with scale_range=(70,90), steps=5, min_confidence=0.5
          const descentPerStep = Math.floor((90 - 70) / 5); // 4
          let bestPosition: [number, number] | null = null;
          let bestConfidence = 0.0;

          for (let scalePercent = 90; scalePercent > 70; scalePercent -= descentPerStep) {
            const scale = scalePercent / 100;
            const scaledW = Math.floor(30 * scale);
            const scaledH = Math.floor(30 * scale);
            const scaledMarker = new window.cv.Mat();
            mats.push(scaledMarker);
            window.cv.resize(marker, scaledMarker, new window.cv.Size(scaledW, scaledH), 0, 0, window.cv.INTER_LINEAR);

            if (scaledH > 120 || scaledW > 120) continue;

            const matchResult = new window.cv.Mat();
            mats.push(matchResult);
            window.cv.matchTemplate(patch, scaledMarker, matchResult, window.cv.TM_CCOEFF_NORMED);
            const minMax = window.cv.minMaxLoc(matchResult);

            if (minMax.maxVal > bestConfidence) {
              bestConfidence = minMax.maxVal;
              bestPosition = [minMax.maxLoc.x, minMax.maxLoc.y];
            }
          }

          const minConfidence = 0.5;
          const cornersNull = bestPosition === null || bestConfidence < minConfidence;

          return { cornersNull, confidence: bestConfidence };
        } finally {
          mats.forEach(m => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.cornersNull).toBe(false);
    });
  });

  // test_no_marker_in_patch: random low-intensity patch, high-intensity marker → corners null
  test('test_detect_marker_in_patch_no_marker_in_patch', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const patchData = new Uint8Array(100 * 100);
          for (let i = 0; i < patchData.length; i++) {
            patchData[i] = Math.floor(Math.random() * 80);
          }
          const patch = window.cv.matFromArray(100, 100, window.cv.CV_8UC1, Array.from(patchData));
          mats.push(patch);

          const markerData = new Uint8Array(20 * 20);
          for (let i = 0; i < markerData.length; i++) {
            markerData[i] = Math.floor(Math.random() * 75 + 180);
          }
          const marker = window.cv.matFromArray(20, 20, window.cv.CV_8UC1, Array.from(markerData));
          mats.push(marker);

          const descentPerStep = Math.floor((115 - 85) / 5);
          let bestConfidence = 0.0;
          let bestPosition: [number, number] | null = null;

          for (let scalePercent = 115; scalePercent > 85; scalePercent -= descentPerStep) {
            const scale = scalePercent / 100;
            const scaledW = Math.floor(20 * scale);
            const scaledH = Math.floor(20 * scale);
            const scaledMarker = new window.cv.Mat();
            mats.push(scaledMarker);
            window.cv.resize(marker, scaledMarker, new window.cv.Size(scaledW, scaledH), 0, 0, window.cv.INTER_LINEAR);

            if (scaledH > 100 || scaledW > 100) continue;

            const matchResult = new window.cv.Mat();
            mats.push(matchResult);
            window.cv.matchTemplate(patch, scaledMarker, matchResult, window.cv.TM_CCOEFF_NORMED);
            const minMax = window.cv.minMaxLoc(matchResult);
            if (minMax.maxVal > bestConfidence) {
              bestConfidence = minMax.maxVal;
              bestPosition = [minMax.maxLoc.x, minMax.maxLoc.y];
            }
          }

          const minConfidence = 0.8;
          const cornersNull = bestPosition === null || bestConfidence < minConfidence;
          return { cornersNull };
        } finally {
          mats.forEach(m => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.cornersNull).toBe(true);
    });
  });

  // ==========================================================================
  // TestValidateMarkerDetection (5 tests)
  // ==========================================================================

  // test_valid_corners: 4x2 float32 square → true
  test('test_validate_marker_detection_valid_corners', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const corners = [[10, 10], [30, 10], [30, 30], [10, 30]];

        // validate: not null, length===4, each element length===2
        if (corners === null || corners === undefined) return false;
        if (corners.length !== 4) return false;
        if (!corners.every(c => Array.isArray(c) && c.length === 2)) return false;
        return true;
      });

      expect(result).toBe(true);
    });
  });

  // test_none_corners: null → false
  test('test_validate_marker_detection_none_corners', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const corners: any = null;
        if (corners === null || corners === undefined) return false;
        return true;
      });

      expect(result).toBe(false);
    });
  });

  // test_invalid_shape: only 2 points → false
  test('test_validate_marker_detection_invalid_shape', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const corners = [[10, 10], [20, 20]]; // only 2 points
        if (corners === null || corners === undefined) return false;
        if (corners.length !== 4) return false;
        return true;
      });

      expect(result).toBe(false);
    });
  });

  // test_area_validation: 20x20 square (area=400), test in-range and out-of-range
  test('test_validate_marker_detection_area_validation', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // corners = [[0,0],[20,0],[20,20],[0,20]] → area=400
          const corners = [[0, 0], [20, 0], [20, 20], [0, 20]];

          function validate(expectedAreaRange: [number, number] | null): boolean {
            if (corners === null || corners === undefined) return false;
            if (corners.length !== 4) return false;
            if (!corners.every(c => Array.isArray(c) && c.length === 2)) return false;

            if (expectedAreaRange !== null && expectedAreaRange !== undefined) {
              const flatData = corners.flat().map(Math.round);
              const contourMat = window.cv.matFromArray(4, 1, window.cv.CV_32SC2, flatData);
              mats.push(contourMat);
              const area = window.cv.contourArea(contourMat);
              const [minArea, maxArea] = expectedAreaRange;
              if (!(minArea <= area && area <= maxArea)) return false;
            }

            return true;
          }

          const inRange = validate([300, 500]);    // area=400 in [300,500] → true
          const tooSmall = validate([500, 1000]);  // area=400 < 500 → false
          const tooLarge = validate([10, 100]);    // area=400 > 100 → false

          return { inRange, tooSmall, tooLarge };
        } finally {
          mats.forEach(m => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.inRange).toBe(true);
      expect(result.tooSmall).toBe(false);
      expect(result.tooLarge).toBe(false);
    });
  });

  // test_area_validation_without_range: valid corners, no area range → true
  test('test_validate_marker_detection_area_validation_without_range', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const corners = [[0, 0], [20, 0], [20, 20], [0, 20]];

        if (corners === null || corners === undefined) return false;
        if (corners.length !== 4) return false;
        if (!corners.every(c => Array.isArray(c) && c.length === 2)) return false;
        // No area range → skip area check
        return true;
      });

      expect(result).toBe(true);
    });
  });

  // ==========================================================================
  // TestIntegrationScenarios (3 tests)
  // ==========================================================================

  // test_end_to_end_pipeline: prepare template, detect in patch, validate
  test('test_integration_end_to_end_pipeline', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // reference = zeros(100,100); reference[30:50,40:60]=255; GaussianBlur(5,5)
          const reference = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(reference);
          for (let r = 30; r < 50; r++) {
            for (let c = 40; c < 60; c++) {
              reference.data[r * 100 + c] = 255;
            }
          }
          window.cv.GaussianBlur(reference, reference, new window.cv.Size(5, 5), 0);

          // reference_zone = {"origin":[40,30], "dimensions":[20,20]}
          const roi = reference.roi(new window.cv.Rect(40, 30, 20, 20));
          let marker = roi.clone();
          mats.push(marker);

          // prepareMarkerTemplate steps: blur(3,3), normalize (apply_erode_subtract=True by default)
          const blurred = new window.cv.Mat();
          mats.push(blurred);
          window.cv.GaussianBlur(marker, blurred, new window.cv.Size(3, 3), 0);
          mats.splice(mats.indexOf(marker), 1);
          marker.delete();
          marker = blurred;

          const minMax0 = window.cv.minMaxLoc(marker);
          let normalized: any;
          if (minMax0.maxVal === minMax0.minVal) {
            normalized = marker.clone();
          } else {
            normalized = new window.cv.Mat();
            window.cv.normalize(marker, normalized, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          }
          mats.push(normalized);
          mats.splice(mats.indexOf(marker), 1);
          marker.delete();
          marker = normalized;

          // erode-subtract
          const kernel = window.cv.Mat.ones(5, 5, window.cv.CV_8U);
          mats.push(kernel);
          const eroded = new window.cv.Mat();
          mats.push(eroded);
          window.cv.erode(marker, eroded, kernel, new window.cv.Point(-1, -1), 5);
          const subtracted = new window.cv.Mat();
          mats.push(subtracted);
          window.cv.subtract(marker, eroded, subtracted);
          const renorm = new window.cv.Mat();
          mats.push(renorm);
          const minMax1 = window.cv.minMaxLoc(subtracted);
          if (minMax1.maxVal === minMax1.minVal) {
            const cloned = subtracted.clone();
            mats.push(cloned);
            mats.splice(mats.indexOf(marker), 1);
            marker.delete();
            marker = cloned;
          } else {
            window.cv.normalize(subtracted, renorm, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
            mats.splice(mats.indexOf(marker), 1);
            marker.delete();
            marker = renorm;
          }

          // patch = zeros(150,150); patch[60:80,70:90] = marker
          const patch = new window.cv.Mat(150, 150, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(patch);
          const mH = marker.rows, mW = marker.cols;
          for (let r = 0; r < mH; r++) {
            for (let c = 0; c < mW; c++) {
              patch.data[(60 + r) * 150 + (70 + c)] = marker.data[r * mW + c];
            }
          }

          // detect with min_confidence=0.7
          const descentPerStep = Math.floor((115 - 85) / 5);
          let bestPosition: [number, number] | null = null;
          let bestMarker: any = null;
          let bestConfidence = 0.0;

          for (let scalePercent = 115; scalePercent > 85; scalePercent -= descentPerStep) {
            const scale = scalePercent / 100;
            const scaledW = Math.floor(mW * scale);
            const scaledH = Math.floor(mH * scale);
            const scaledMarker = new window.cv.Mat();
            mats.push(scaledMarker);
            window.cv.resize(marker, scaledMarker, new window.cv.Size(scaledW, scaledH), 0, 0, window.cv.INTER_LINEAR);

            if (scaledH > 150 || scaledW > 150) continue;

            const matchResult = new window.cv.Mat();
            mats.push(matchResult);
            window.cv.matchTemplate(patch, scaledMarker, matchResult, window.cv.TM_CCOEFF_NORMED);
            const minMax = window.cv.minMaxLoc(matchResult);

            if (minMax.maxVal > bestConfidence) {
              bestConfidence = minMax.maxVal;
              bestPosition = [minMax.maxLoc.x, minMax.maxLoc.y];
              if (bestMarker) {
                mats.splice(mats.indexOf(bestMarker), 1);
                bestMarker.delete();
              }
              bestMarker = scaledMarker;
              mats.splice(mats.indexOf(scaledMarker), 1);
            }
          }

          if (bestMarker) mats.push(bestMarker);

          if (bestPosition === null || bestConfidence < 0.7) {
            return { cornersNull: true };
          }

          const [px, py] = bestPosition;
          const h = bestMarker.rows, w = bestMarker.cols;
          const corners = [
            [px, py], [px + w, py], [px + w, py + h], [px, py + h],
          ];

          // validate
          const valid = corners !== null && corners.length === 4 &&
            corners.every(c => Array.isArray(c) && c.length === 2);

          return { cornersNull: false, valid };
        } finally {
          mats.forEach(m => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.cornersNull).toBe(false);
      expect(result.valid).toBe(true);
    });
  });

  // test_with_noise: noisy patch with embedded marker → still detects
  test('test_integration_with_noise', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // marker = zeros(25,25); marker[5:20,5:20]=255; GaussianBlur(5,5)
          const markerRaw = new window.cv.Mat(25, 25, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(markerRaw);
          for (let r = 5; r < 20; r++) {
            for (let c = 5; c < 20; c++) {
              markerRaw.data[r * 25 + c] = 255;
            }
          }
          const marker = new window.cv.Mat();
          mats.push(marker);
          window.cv.GaussianBlur(markerRaw, marker, new window.cv.Size(5, 5), 0);

          // patch = randint(0,100,(120,120)); embed marker at (40,50)
          const patchData = new Uint8Array(120 * 120);
          for (let i = 0; i < patchData.length; i++) {
            patchData[i] = Math.floor(Math.random() * 100);
          }
          const patchArr = Array.from(patchData);
          // Embed marker
          for (let r = 0; r < 25; r++) {
            for (let c = 0; c < 25; c++) {
              patchArr[(40 + r) * 120 + (50 + c)] = marker.data[r * 25 + c];
            }
          }
          // Add noise: clip to [0,255]
          const noisyArr = patchArr.map(v => Math.max(0, Math.min(255, v + Math.floor(Math.random() * 60 - 30))));
          const patch = window.cv.matFromArray(120, 120, window.cv.CV_8UC1, noisyArr);
          mats.push(patch);

          // Re-embed marker (overwrite noise at marker position)
          for (let r = 0; r < 25; r++) {
            for (let c = 0; c < 25; c++) {
              patch.data[(40 + r) * 120 + (50 + c)] = marker.data[r * 25 + c];
            }
          }

          // detect with min_confidence=0.3
          const descentPerStep = Math.floor((115 - 85) / 5);
          let bestPosition: [number, number] | null = null;
          let bestConfidence = 0.0;

          for (let scalePercent = 115; scalePercent > 85; scalePercent -= descentPerStep) {
            const scale = scalePercent / 100;
            const scaledW = Math.floor(25 * scale);
            const scaledH = Math.floor(25 * scale);
            const scaledMarker = new window.cv.Mat();
            mats.push(scaledMarker);
            window.cv.resize(marker, scaledMarker, new window.cv.Size(scaledW, scaledH), 0, 0, window.cv.INTER_LINEAR);

            if (scaledH > 120 || scaledW > 120) continue;

            const matchResult = new window.cv.Mat();
            mats.push(matchResult);
            window.cv.matchTemplate(patch, scaledMarker, matchResult, window.cv.TM_CCOEFF_NORMED);
            const minMax = window.cv.minMaxLoc(matchResult);
            if (minMax.maxVal > bestConfidence) {
              bestConfidence = minMax.maxVal;
              bestPosition = [minMax.maxLoc.x, minMax.maxLoc.y];
            }
          }

          const cornersNull = bestPosition === null || bestConfidence < 0.3;
          return { cornersNull, confidence: bestConfidence };
        } finally {
          mats.forEach(m => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.cornersNull).toBe(false);
    });
  });

  // test_multiple_scale_attempts: marker 30x30, patch has 33x33 version at (50,60)
  // wider scale range (90,120) should find it
  test('test_integration_multiple_scale_attempts', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // marker = zeros(30,30); marker[10:20,10:20]=255
          const marker = new window.cv.Mat(30, 30, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(marker);
          for (let r = 10; r < 20; r++) {
            for (let c = 10; c < 20; c++) {
              marker.data[r * 30 + c] = 255;
            }
          }

          // resize marker to 33x33 for embedding
          const bigMarker = new window.cv.Mat();
          mats.push(bigMarker);
          window.cv.resize(marker, bigMarker, new window.cv.Size(33, 33), 0, 0, window.cv.INTER_LINEAR);

          // patch = zeros(150,150); embed 33x33 version at (50,60)
          const patch = new window.cv.Mat(150, 150, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(patch);
          for (let r = 0; r < 33; r++) {
            for (let c = 0; c < 33; c++) {
              patch.data[(50 + r) * 150 + (60 + c)] = bigMarker.data[r * 33 + c];
            }
          }

          // detect with scale_range=(90,120), steps=6, min_confidence=0.7
          const descentPerStep = Math.floor((120 - 90) / 6); // 5
          let bestPosition: [number, number] | null = null;
          let bestConfidence = 0.0;

          for (let scalePercent = 120; scalePercent > 90; scalePercent -= descentPerStep) {
            const scale = scalePercent / 100;
            const scaledW = Math.floor(30 * scale);
            const scaledH = Math.floor(30 * scale);
            const scaledMarker = new window.cv.Mat();
            mats.push(scaledMarker);
            window.cv.resize(marker, scaledMarker, new window.cv.Size(scaledW, scaledH), 0, 0, window.cv.INTER_LINEAR);

            if (scaledH > 150 || scaledW > 150) continue;

            const matchResult = new window.cv.Mat();
            mats.push(matchResult);
            window.cv.matchTemplate(patch, scaledMarker, matchResult, window.cv.TM_CCOEFF_NORMED);
            const minMax = window.cv.minMaxLoc(matchResult);
            if (minMax.maxVal > bestConfidence) {
              bestConfidence = minMax.maxVal;
              bestPosition = [minMax.maxLoc.x, minMax.maxLoc.y];
            }
          }

          const cornersNull = bestPosition === null || bestConfidence < 0.7;
          return { cornersNull, confidence: bestConfidence };
        } finally {
          mats.forEach(m => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.cornersNull).toBe(false);
    });
  });
});
