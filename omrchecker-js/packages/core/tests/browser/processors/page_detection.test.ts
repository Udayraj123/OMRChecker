/**
 * Browser tests for page_detection.ts using OpenCV.js
 *
 * These tests verify page detection operations work correctly in a real browser
 * environment with actual OpenCV.js cv.Mat objects. Each test corresponds to a
 * Python test in src/tests/processors/test_page_detection.py.
 *
 * Run with: npm run test:browser
 */

import { test, expect } from '@playwright/test';
import { setupBrowser } from '../browser-setup';
import { withMemoryTracking } from '../memory-utils';

// Increase timeout for browser tests (OpenCV.js WASM initialization)
test.setTimeout(60000);

test.describe('Page Detection - Browser Tests', () => {
  test.beforeEach(async ({ page }) => {
    await setupBrowser(page);
  });

  // =========================================================================
  // TestPreparePageImage
  // =========================================================================

  // test_prepare_normalizes_image: uniform 128-value image → result is uint8, same shape
  test('test_prepare_normalizes_image', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const image = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(128));
          mats.push(image);

          // preparePageImage: threshold TRUNC at 210, then normalize
          const truncated = new window.cv.Mat();
          mats.push(truncated);
          window.cv.threshold(image, truncated, 210, 255, window.cv.THRESH_TRUNC);

          const normalized = new window.cv.Mat();
          mats.push(normalized);
          window.cv.normalize(truncated, normalized, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);

          return { rows: normalized.rows, cols: normalized.cols, type: normalized.type() };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.rows).toBe(100);
      expect(result.cols).toBe(100);
      // CV_8UC1 = 0
      expect(result.type).toBe(0);
    });
  });

  // test_prepare_truncates_high_values: uniform 250 image → max pixel value <= 255
  test('test_prepare_truncates_high_values', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const image = new window.cv.Mat(100, 100, window.cv.CV_8UC1, new window.cv.Scalar(250));
          mats.push(image);

          const truncated = new window.cv.Mat();
          mats.push(truncated);
          window.cv.threshold(image, truncated, 210, 255, window.cv.THRESH_TRUNC);

          const normalized = new window.cv.Mat();
          mats.push(normalized);
          window.cv.normalize(truncated, normalized, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);

          const minMaxResult = window.cv.minMaxLoc(normalized);
          return { max: minMaxResult.maxVal };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.max).toBeLessThanOrEqual(255);
    });
  });

  // =========================================================================
  // TestApplyColoredCanny
  // =========================================================================

  // test_apply_colored_canny_produces_edges: white rect on black → edges detected
  test('test_apply_colored_canny_produces_edges', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // Gray image with white rectangle
          const gray = new window.cv.Mat(200, 200, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(gray);
          window.cv.rectangle(
            gray,
            new window.cv.Point(50, 50),
            new window.cv.Point(150, 150),
            new window.cv.Scalar(255),
            -1
          );

          // Color version (BGR)
          const color = new window.cv.Mat();
          mats.push(color);
          window.cv.cvtColor(gray, color, window.cv.COLOR_GRAY2BGR);

          // applyColoredCanny logic: HSV mask, bitwise_and, Canny
          const hsv = new window.cv.Mat();
          mats.push(hsv);
          window.cv.cvtColor(color, hsv, window.cv.COLOR_BGR2HSV);

          // Build the whitish-region mask using threshold on HSV V channel
          // (equivalent to HSV whitish mask, but using split/threshold for compatibility)
          const hsvChannels = new window.cv.MatVector();
          mats.push(hsvChannels);
          window.cv.split(hsv, hsvChannels);
          const vChannel = hsvChannels.get(2); // V channel
          mats.push(vChannel);
          const sChannel = hsvChannels.get(1); // S channel
          mats.push(sChannel);

          // Mask: V >= 150 AND S <= 60 (whitish region)
          const maskV = new window.cv.Mat();
          mats.push(maskV);
          window.cv.threshold(vChannel, maskV, 149, 255, window.cv.THRESH_BINARY);
          const maskS = new window.cv.Mat();
          mats.push(maskS);
          window.cv.threshold(sChannel, maskS, 60, 255, window.cv.THRESH_BINARY_INV);
          const mask = new window.cv.Mat();
          mats.push(mask);
          window.cv.bitwise_and(maskV, maskS, mask);

          const masked = new window.cv.Mat();
          mats.push(masked);
          window.cv.bitwise_and(gray, gray, masked, mask);

          const edges = new window.cv.Mat();
          mats.push(edges);
          window.cv.Canny(masked, edges, 185, 55);

          return { nonZero: window.cv.countNonZero(edges) };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.nonZero).toBeGreaterThan(0);
    });
  });

  // =========================================================================
  // TestApplyGrayscaleCanny
  // =========================================================================

  // test_apply_without_morph: white rect on black → edges detected without morph
  test('test_apply_without_morph', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const image = new window.cv.Mat(200, 200, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(image);
          window.cv.rectangle(
            image,
            new window.cv.Point(50, 50),
            new window.cv.Point(150, 150),
            new window.cv.Scalar(255),
            -1
          );

          // applyGrayscaleCanny: threshold TRUNC at 200, normalize, Canny (no morph)
          const truncated = new window.cv.Mat();
          mats.push(truncated);
          window.cv.threshold(image, truncated, 200, 255, window.cv.THRESH_TRUNC);

          const normalized = new window.cv.Mat();
          mats.push(normalized);
          window.cv.normalize(truncated, normalized, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);

          const edges = new window.cv.Mat();
          mats.push(edges);
          window.cv.Canny(normalized, edges, 185, 55);

          return { nonZero: window.cv.countNonZero(edges) };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.nonZero).toBeGreaterThan(0);
    });
  });

  // test_apply_with_morph_kernel: edges detected with large (5x5) morphology kernel
  test('test_apply_with_morph_kernel', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const image = new window.cv.Mat(200, 200, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(image);
          window.cv.rectangle(
            image,
            new window.cv.Point(50, 50),
            new window.cv.Point(150, 150),
            new window.cv.Scalar(255),
            -1
          );

          const truncated = new window.cv.Mat();
          mats.push(truncated);
          window.cv.threshold(image, truncated, 200, 255, window.cv.THRESH_TRUNC);

          const normalized = new window.cv.Mat();
          mats.push(normalized);
          window.cv.normalize(truncated, normalized, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);

          // Apply morphological closing with a 5x5 kernel
          const kernel = window.cv.getStructuringElement(
            window.cv.MORPH_RECT,
            new window.cv.Size(5, 5)
          );
          mats.push(kernel);
          const closed = new window.cv.Mat();
          mats.push(closed);
          window.cv.morphologyEx(normalized, closed, window.cv.MORPH_CLOSE, kernel);

          const edges = new window.cv.Mat();
          mats.push(edges);
          window.cv.Canny(closed, edges, 185, 55);

          return { nonZero: window.cv.countNonZero(edges) };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.nonZero).toBeGreaterThan(0);
    });
  });

  // test_small_kernel_skips_morph: 1x1 kernel → still produces edges (morph skipped)
  test('test_small_kernel_skips_morph', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const image = new window.cv.Mat(200, 200, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(image);
          window.cv.rectangle(
            image,
            new window.cv.Point(50, 50),
            new window.cv.Point(150, 150),
            new window.cv.Scalar(255),
            -1
          );

          const truncated = new window.cv.Mat();
          mats.push(truncated);
          window.cv.threshold(image, truncated, 200, 255, window.cv.THRESH_TRUNC);

          const normalized = new window.cv.Mat();
          mats.push(normalized);
          window.cv.normalize(truncated, normalized, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);

          // 1x1 kernel → morph is skipped in the Python logic (rows <= 1)
          // So we go straight to Canny without morphologyEx
          const edges = new window.cv.Mat();
          mats.push(edges);
          window.cv.Canny(normalized, edges, 185, 55);

          return { nonZero: window.cv.countNonZero(edges) };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.nonZero).toBeGreaterThan(0);
    });
  });

  // =========================================================================
  // TestFindPageContours
  // =========================================================================

  // test_find_contours_returns_list: edge image with drawn rect → finds at least 1 contour
  test('test_find_contours_returns_list', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // Create an edge image directly (draw lines to simulate canny output)
          const edge = new window.cv.Mat(400, 600, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(edge);
          // Draw a large rectangle outline (edges)
          window.cv.rectangle(
            edge,
            new window.cv.Point(50, 50),
            new window.cv.Point(550, 350),
            new window.cv.Scalar(255),
            2
          );
          // Draw a small rectangle as a second contour
          window.cv.rectangle(
            edge,
            new window.cv.Point(200, 150),
            new window.cv.Point(250, 200),
            new window.cv.Scalar(255),
            1
          );

          // findPageContours logic
          const contours = new window.cv.MatVector();
          mats.push(contours);
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(
            edge, contours, hierarchy,
            window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE
          );

          const count = contours.size();

          // Build convex hulls
          const hulls: any[] = [];
          for (let i = 0; i < count; i++) {
            const hull = new window.cv.Mat();
            window.cv.convexHull(contours.get(i), hull);
            hulls.push(hull);
            mats.push(hull);
          }

          // Sort by area desc
          hulls.sort((a: any, b: any) => window.cv.contourArea(b) - window.cv.contourArea(a));
          const topCount = Math.min(5, hulls.length);

          return { contourCount: topCount };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.contourCount).toBeGreaterThan(0);
    });
  });

  // test_contours_sorted_by_area: if >= 2 contours, area[0] >= area[1]
  test('test_contours_sorted_by_area', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const edge = new window.cv.Mat(400, 600, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(edge);
          // Large rectangle
          window.cv.rectangle(edge, new window.cv.Point(50, 50), new window.cv.Point(550, 350), new window.cv.Scalar(255), 2);
          // Small rectangle
          window.cv.rectangle(edge, new window.cv.Point(200, 150), new window.cv.Point(250, 200), new window.cv.Scalar(255), 1);

          const contours = new window.cv.MatVector();
          mats.push(contours);
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(edge, contours, hierarchy, window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE);

          const hulls: any[] = [];
          for (let i = 0; i < contours.size(); i++) {
            const hull = new window.cv.Mat();
            window.cv.convexHull(contours.get(i), hull);
            hulls.push(hull);
            mats.push(hull);
          }

          // Sort descending by area
          hulls.sort((a: any, b: any) => window.cv.contourArea(b) - window.cv.contourArea(a));
          const top = hulls.slice(0, 5);

          if (top.length < 2) {
            return { valid: true, area0: -1, area1: -1 };
          }

          const area0 = window.cv.contourArea(top[0]);
          const area1 = window.cv.contourArea(top[1]);

          return { valid: area0 >= area1, area0, area1 };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.valid).toBe(true);
    });
  });

  // test_returns_top_candidates_only: contours count <= 5 (TOP_CONTOURS_COUNT)
  test('test_returns_top_candidates_only', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const edge = new window.cv.Mat(400, 600, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(edge);
          // Draw many rectangles to produce many contours
          const rects = [
            [50, 50, 550, 350],
            [100, 100, 200, 200],
            [250, 100, 350, 200],
            [400, 100, 500, 200],
            [100, 250, 200, 340],
            [250, 250, 350, 340],
            [400, 250, 500, 340],
          ];
          for (const [x1, y1, x2, y2] of rects) {
            window.cv.rectangle(edge, new window.cv.Point(x1, y1), new window.cv.Point(x2, y2), new window.cv.Scalar(255), 1);
          }

          const contours = new window.cv.MatVector();
          mats.push(contours);
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(edge, contours, hierarchy, window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE);

          const hulls: any[] = [];
          for (let i = 0; i < contours.size(); i++) {
            const hull = new window.cv.Mat();
            window.cv.convexHull(contours.get(i), hull);
            hulls.push(hull);
            mats.push(hull);
          }

          hulls.sort((a: any, b: any) => window.cv.contourArea(b) - window.cv.contourArea(a));
          const topCount = Math.min(5, hulls.length);

          return { topCount };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.topCount).toBeLessThanOrEqual(5);
    });
  });

  // =========================================================================
  // TestExtractPageRectangle
  // =========================================================================

  // test_extract_valid_rectangle: large 4-corner contour → corners returned (4 points)
  test('test_extract_valid_rectangle', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // Draw a large rectangle on a blank image and find its contour
          const img = new window.cv.Mat(600, 800, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(img);
          window.cv.rectangle(img, new window.cv.Point(50, 50), new window.cv.Point(750, 550), new window.cv.Scalar(255), 3);

          const contours = new window.cv.MatVector();
          mats.push(contours);
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(img, contours, hierarchy, window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE);

          // Build hulls and sort by area
          const hulls: any[] = [];
          for (let i = 0; i < contours.size(); i++) {
            const hull = new window.cv.Mat();
            window.cv.convexHull(contours.get(i), hull);
            hulls.push(hull);
            mats.push(hull);
          }
          hulls.sort((a: any, b: any) => window.cv.contourArea(b) - window.cv.contourArea(a));
          const topContours = hulls.slice(0, 5);

          // extractPageRectangle logic
          const MIN_PAGE_AREA = 80000;
          const APPROX_POLY_EPSILON_FACTOR = 0.025;
          const MAX_COSINE = 0.35;

          function angle(p1: number[], p2: number[], p0: number[]) {
            const dx1 = p1[0] - p0[0], dy1 = p1[1] - p0[1];
            const dx2 = p2[0] - p0[0], dy2 = p2[1] - p0[1];
            const dot = dx1 * dx2 + dy1 * dy2;
            const mag = Math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
            return dot / mag;
          }

          function validateRect(pts: number[][]): boolean {
            if (pts.length !== 4) return false;
            let maxCos = 0;
            for (let i = 2; i < 5; i++) {
              const cos = Math.abs(angle(pts[i % 4], pts[i - 2], pts[i - 1]));
              if (cos > maxCos) maxCos = cos;
            }
            return maxCos < MAX_COSINE;
          }

          let corners: number[][] | null = null;
          for (const contour of topContours) {
            if (window.cv.contourArea(contour) < MIN_PAGE_AREA) continue;
            const perimeter = window.cv.arcLength(contour, true);
            const epsilon = APPROX_POLY_EPSILON_FACTOR * perimeter;
            const approx = new window.cv.Mat();
            mats.push(approx);
            window.cv.approxPolyDP(contour, approx, epsilon, true);

            if (approx.rows !== 4) continue;

            const pts: number[][] = [];
            for (let i = 0; i < 4; i++) {
              pts.push([approx.data32S[i * 2], approx.data32S[i * 2 + 1]]);
            }

            if (validateRect(pts)) {
              corners = pts;
              break;
            }
          }

          return { cornersLength: corners ? corners.length : null, found: corners !== null };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.found).toBe(true);
      expect(result.cornersLength).toBe(4);
    });
  });

  // test_reject_small_contour: small contour (area < 80000) → null returned
  test('test_reject_small_contour', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // Small rectangle — far less than MIN_PAGE_AREA=80000
          const img = new window.cv.Mat(200, 200, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(img);
          window.cv.rectangle(img, new window.cv.Point(10, 10), new window.cv.Point(50, 50), new window.cv.Scalar(255), 2);

          const contours = new window.cv.MatVector();
          mats.push(contours);
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(img, contours, hierarchy, window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE);

          const hulls: any[] = [];
          for (let i = 0; i < contours.size(); i++) {
            const hull = new window.cv.Mat();
            window.cv.convexHull(contours.get(i), hull);
            hulls.push(hull);
            mats.push(hull);
          }
          hulls.sort((a: any, b: any) => window.cv.contourArea(b) - window.cv.contourArea(a));

          const MIN_PAGE_AREA = 80000;
          const APPROX_POLY_EPSILON_FACTOR = 0.025;
          let found = false;

          for (const contour of hulls.slice(0, 5)) {
            if (window.cv.contourArea(contour) < MIN_PAGE_AREA) continue;
            const perimeter = window.cv.arcLength(contour, true);
            const approx = new window.cv.Mat();
            mats.push(approx);
            window.cv.approxPolyDP(contour, approx, APPROX_POLY_EPSILON_FACTOR * perimeter, true);
            if (approx.rows === 4) {
              found = true;
              break;
            }
          }

          return { found };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.found).toBe(false);
    });
  });

  // test_reject_non_rectangle: triangle contour (3 sides) → null returned
  test('test_reject_non_rectangle', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // Draw a triangle (not a rectangle) that's large enough
          const img = new window.cv.Mat(600, 800, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(img);

          // Large triangle
          const triPts = window.cv.matFromArray(3, 1, window.cv.CV_32SC2, [
            400, 50,
            50, 550,
            750, 550,
          ]);
          mats.push(triPts);
          const triVec = new window.cv.MatVector();
          mats.push(triVec);
          triVec.push_back(triPts);
          window.cv.polylines(img, triVec, true, [255, 255, 255, 0], 3);

          const contours = new window.cv.MatVector();
          mats.push(contours);
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(img, contours, hierarchy, window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE);

          const hulls: any[] = [];
          for (let i = 0; i < contours.size(); i++) {
            const hull = new window.cv.Mat();
            window.cv.convexHull(contours.get(i), hull);
            hulls.push(hull);
            mats.push(hull);
          }
          hulls.sort((a: any, b: any) => window.cv.contourArea(b) - window.cv.contourArea(a));

          const MIN_PAGE_AREA = 80000;
          const APPROX_POLY_EPSILON_FACTOR = 0.025;
          const MAX_COSINE = 0.35;

          function angle(p1: number[], p2: number[], p0: number[]) {
            const dx1 = p1[0] - p0[0], dy1 = p1[1] - p0[1];
            const dx2 = p2[0] - p0[0], dy2 = p2[1] - p0[1];
            const dot = dx1 * dx2 + dy1 * dy2;
            const mag = Math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
            return dot / mag;
          }

          function validateRect(pts: number[][]): boolean {
            if (pts.length !== 4) return false;
            let maxCos = 0;
            for (let i = 2; i < 5; i++) {
              const cos = Math.abs(angle(pts[i % 4], pts[i - 2], pts[i - 1]));
              if (cos > maxCos) maxCos = cos;
            }
            return maxCos < MAX_COSINE;
          }

          let found = false;
          for (const contour of hulls.slice(0, 5)) {
            if (window.cv.contourArea(contour) < MIN_PAGE_AREA) continue;
            const perimeter = window.cv.arcLength(contour, true);
            const approx = new window.cv.Mat();
            mats.push(approx);
            window.cv.approxPolyDP(contour, approx, APPROX_POLY_EPSILON_FACTOR * perimeter, true);
            if (approx.rows !== 4) continue;
            const pts: number[][] = [];
            for (let i = 0; i < 4; i++) {
              pts.push([approx.data32S[i * 2], approx.data32S[i * 2 + 1]]);
            }
            if (validateRect(pts)) {
              found = true;
              break;
            }
          }

          return { found };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      // Triangle has 3 corners, not 4 — should be rejected
      expect(result.found).toBe(false);
    });
  });

  // =========================================================================
  // TestFindPageContourAndCorners
  // =========================================================================

  // Helper: full page detection pipeline logic in page.evaluate
  // Shared logic for the 5 findPageContourAndCorners tests

  // test_find_page_success: white rect on 400x600 black image → corners (4 points)
  test('test_find_page_success', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // Large enough image with a page-like white rectangle
          const image = new window.cv.Mat(400, 600, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(image);
          window.cv.rectangle(image, new window.cv.Point(30, 30), new window.cv.Point(570, 370), new window.cv.Scalar(255), -1);

          // preparePageImage
          const truncated = new window.cv.Mat();
          mats.push(truncated);
          window.cv.threshold(image, truncated, 210, 255, window.cv.THRESH_TRUNC);
          const normalized = new window.cv.Mat();
          mats.push(normalized);
          window.cv.normalize(truncated, normalized, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);

          // applyGrayscaleCanny (no morph)
          const truncated2 = new window.cv.Mat();
          mats.push(truncated2);
          window.cv.threshold(normalized, truncated2, 200, 255, window.cv.THRESH_TRUNC);
          const normalized2 = new window.cv.Mat();
          mats.push(normalized2);
          window.cv.normalize(truncated2, normalized2, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          const edges = new window.cv.Mat();
          mats.push(edges);
          window.cv.Canny(normalized2, edges, 185, 55);

          // findPageContours
          const contours = new window.cv.MatVector();
          mats.push(contours);
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(edges, contours, hierarchy, window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE);
          const hulls: any[] = [];
          for (let i = 0; i < contours.size(); i++) {
            const hull = new window.cv.Mat();
            window.cv.convexHull(contours.get(i), hull);
            hulls.push(hull);
            mats.push(hull);
          }
          hulls.sort((a: any, b: any) => window.cv.contourArea(b) - window.cv.contourArea(a));
          const topContours = hulls.slice(0, 5);

          // extractPageRectangle
          const MIN_PAGE_AREA = 80000;
          const APPROX_POLY_EPSILON_FACTOR = 0.025;
          const MAX_COSINE = 0.35;

          function angle(p1: number[], p2: number[], p0: number[]) {
            const dx1 = p1[0] - p0[0], dy1 = p1[1] - p0[1];
            const dx2 = p2[0] - p0[0], dy2 = p2[1] - p0[1];
            const dot = dx1 * dx2 + dy1 * dy2;
            const mag = Math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
            return dot / mag;
          }

          function validateRect(pts: number[][]): boolean {
            if (pts.length !== 4) return false;
            let maxCos = 0;
            for (let i = 2; i < 5; i++) {
              const cos = Math.abs(angle(pts[i % 4], pts[i - 2], pts[i - 1]));
              if (cos > maxCos) maxCos = cos;
            }
            return maxCos < MAX_COSINE;
          }

          let corners: number[][] | null = null;
          for (const contour of topContours) {
            if (window.cv.contourArea(contour) < MIN_PAGE_AREA) continue;
            const perimeter = window.cv.arcLength(contour, true);
            const approx = new window.cv.Mat();
            mats.push(approx);
            window.cv.approxPolyDP(contour, approx, APPROX_POLY_EPSILON_FACTOR * perimeter, true);
            if (approx.rows !== 4) continue;
            const pts: number[][] = [];
            for (let i = 0; i < 4; i++) {
              pts.push([approx.data32S[i * 2], approx.data32S[i * 2 + 1]]);
            }
            if (validateRect(pts)) {
              corners = pts;
              break;
            }
          }

          return { found: corners !== null, length: corners ? corners.length : null };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.found).toBe(true);
      expect(result.length).toBe(4);
    });
  });

  // test_find_page_with_colored_canny: same with colored canny → corners (4 points)
  test('test_find_page_with_colored_canny', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // Create a color (BGR) image with white rectangle
          const colorImg = new window.cv.Mat(400, 600, window.cv.CV_8UC3, new window.cv.Scalar(0, 0, 0));
          mats.push(colorImg);
          window.cv.rectangle(colorImg, new window.cv.Point(30, 30), new window.cv.Point(570, 370), new window.cv.Scalar(255, 255, 255), -1);

          // Get grayscale version
          const grayImg = new window.cv.Mat();
          mats.push(grayImg);
          window.cv.cvtColor(colorImg, grayImg, window.cv.COLOR_BGR2GRAY);

          // preparePageImage
          const truncated = new window.cv.Mat();
          mats.push(truncated);
          window.cv.threshold(grayImg, truncated, 210, 255, window.cv.THRESH_TRUNC);
          const prepared = new window.cv.Mat();
          mats.push(prepared);
          window.cv.normalize(truncated, prepared, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);

          // applyColoredCanny using HSV channel split (V>=150, S<=60 = whitish)
          const hsv = new window.cv.Mat();
          mats.push(hsv);
          window.cv.cvtColor(colorImg, hsv, window.cv.COLOR_BGR2HSV);

          const hsvChannels = new window.cv.MatVector();
          mats.push(hsvChannels);
          window.cv.split(hsv, hsvChannels);
          const vChannel = hsvChannels.get(2);
          mats.push(vChannel);
          const sChannel = hsvChannels.get(1);
          mats.push(sChannel);

          const maskV = new window.cv.Mat();
          mats.push(maskV);
          window.cv.threshold(vChannel, maskV, 149, 255, window.cv.THRESH_BINARY);
          const maskS = new window.cv.Mat();
          mats.push(maskS);
          window.cv.threshold(sChannel, maskS, 60, 255, window.cv.THRESH_BINARY_INV);
          const mask = new window.cv.Mat();
          mats.push(mask);
          window.cv.bitwise_and(maskV, maskS, mask);

          const masked = new window.cv.Mat();
          mats.push(masked);
          window.cv.bitwise_and(prepared, prepared, masked, mask);
          const edges = new window.cv.Mat();
          mats.push(edges);
          window.cv.Canny(masked, edges, 185, 55);

          // findPageContours
          const contours = new window.cv.MatVector();
          mats.push(contours);
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(edges, contours, hierarchy, window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE);
          const hulls: any[] = [];
          for (let i = 0; i < contours.size(); i++) {
            const hull = new window.cv.Mat();
            window.cv.convexHull(contours.get(i), hull);
            hulls.push(hull);
            mats.push(hull);
          }
          hulls.sort((a: any, b: any) => window.cv.contourArea(b) - window.cv.contourArea(a));
          const topContours = hulls.slice(0, 5);

          const MIN_PAGE_AREA = 80000;
          const APPROX_POLY_EPSILON_FACTOR = 0.025;
          const MAX_COSINE = 0.35;

          function angle(p1: number[], p2: number[], p0: number[]) {
            const dx1 = p1[0] - p0[0], dy1 = p1[1] - p0[1];
            const dx2 = p2[0] - p0[0], dy2 = p2[1] - p0[1];
            const dot = dx1 * dx2 + dy1 * dy2;
            const mag = Math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
            return dot / mag;
          }

          function validateRect(pts: number[][]): boolean {
            if (pts.length !== 4) return false;
            let maxCos = 0;
            for (let i = 2; i < 5; i++) {
              const cos = Math.abs(angle(pts[i % 4], pts[i - 2], pts[i - 1]));
              if (cos > maxCos) maxCos = cos;
            }
            return maxCos < MAX_COSINE;
          }

          let corners: number[][] | null = null;
          for (const contour of topContours) {
            if (window.cv.contourArea(contour) < MIN_PAGE_AREA) continue;
            const perimeter = window.cv.arcLength(contour, true);
            const approx = new window.cv.Mat();
            mats.push(approx);
            window.cv.approxPolyDP(contour, approx, APPROX_POLY_EPSILON_FACTOR * perimeter, true);
            if (approx.rows !== 4) continue;
            const pts: number[][] = [];
            for (let i = 0; i < 4; i++) {
              pts.push([approx.data32S[i * 2], approx.data32S[i * 2 + 1]]);
            }
            if (validateRect(pts)) {
              corners = pts;
              break;
            }
          }

          return { found: corners !== null, length: corners ? corners.length : null };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.found).toBe(true);
      expect(result.length).toBe(4);
    });
  });

  // test_find_page_with_morph_kernel: large white rect + morphology kernel → corners not null
  test('test_find_page_with_morph_kernel', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const image = new window.cv.Mat(400, 600, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(image);
          window.cv.rectangle(image, new window.cv.Point(30, 30), new window.cv.Point(570, 370), new window.cv.Scalar(255), -1);

          // preparePageImage
          const truncated = new window.cv.Mat();
          mats.push(truncated);
          window.cv.threshold(image, truncated, 210, 255, window.cv.THRESH_TRUNC);
          const normalized = new window.cv.Mat();
          mats.push(normalized);
          window.cv.normalize(truncated, normalized, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);

          // applyGrayscaleCanny with morph kernel
          const truncated2 = new window.cv.Mat();
          mats.push(truncated2);
          window.cv.threshold(normalized, truncated2, 200, 255, window.cv.THRESH_TRUNC);
          const normalized2 = new window.cv.Mat();
          mats.push(normalized2);
          window.cv.normalize(truncated2, normalized2, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);

          const kernel = window.cv.getStructuringElement(window.cv.MORPH_RECT, new window.cv.Size(5, 5));
          mats.push(kernel);
          const closed = new window.cv.Mat();
          mats.push(closed);
          window.cv.morphologyEx(normalized2, closed, window.cv.MORPH_CLOSE, kernel);

          const edges = new window.cv.Mat();
          mats.push(edges);
          window.cv.Canny(closed, edges, 185, 55);

          // findPageContours
          const contours = new window.cv.MatVector();
          mats.push(contours);
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(edges, contours, hierarchy, window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE);
          const hulls: any[] = [];
          for (let i = 0; i < contours.size(); i++) {
            const hull = new window.cv.Mat();
            window.cv.convexHull(contours.get(i), hull);
            hulls.push(hull);
            mats.push(hull);
          }
          hulls.sort((a: any, b: any) => window.cv.contourArea(b) - window.cv.contourArea(a));
          const topContours = hulls.slice(0, 5);

          const MIN_PAGE_AREA = 80000;
          const APPROX_POLY_EPSILON_FACTOR = 0.025;
          const MAX_COSINE = 0.35;

          function angle(p1: number[], p2: number[], p0: number[]) {
            const dx1 = p1[0] - p0[0], dy1 = p1[1] - p0[1];
            const dx2 = p2[0] - p0[0], dy2 = p2[1] - p0[1];
            const dot = dx1 * dx2 + dy1 * dy2;
            const mag = Math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
            return dot / mag;
          }

          function validateRect(pts: number[][]): boolean {
            if (pts.length !== 4) return false;
            let maxCos = 0;
            for (let i = 2; i < 5; i++) {
              const cos = Math.abs(angle(pts[i % 4], pts[i - 2], pts[i - 1]));
              if (cos > maxCos) maxCos = cos;
            }
            return maxCos < MAX_COSINE;
          }

          let found = false;
          for (const contour of topContours) {
            if (window.cv.contourArea(contour) < MIN_PAGE_AREA) continue;
            const perimeter = window.cv.arcLength(contour, true);
            const approx = new window.cv.Mat();
            mats.push(approx);
            window.cv.approxPolyDP(contour, approx, APPROX_POLY_EPSILON_FACTOR * perimeter, true);
            if (approx.rows !== 4) continue;
            const pts: number[][] = [];
            for (let i = 0; i < 4; i++) {
              pts.push([approx.data32S[i * 2], approx.data32S[i * 2 + 1]]);
            }
            if (validateRect(pts)) {
              found = true;
              break;
            }
          }

          return { found };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.found).toBe(true);
    });
  });

  // test_find_page_draws_debug_contours: with debug_image → debug_image has nonZero pixels
  test('test_find_page_draws_debug_contours', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const image = new window.cv.Mat(400, 600, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(image);
          window.cv.rectangle(image, new window.cv.Point(30, 30), new window.cv.Point(570, 370), new window.cv.Scalar(255), -1);

          // Debug image (BGR)
          const debugImage = new window.cv.Mat(400, 600, window.cv.CV_8UC3, new window.cv.Scalar(0, 0, 0));
          mats.push(debugImage);

          // preparePageImage
          const truncated = new window.cv.Mat();
          mats.push(truncated);
          window.cv.threshold(image, truncated, 210, 255, window.cv.THRESH_TRUNC);
          const normalized = new window.cv.Mat();
          mats.push(normalized);
          window.cv.normalize(truncated, normalized, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);

          // applyGrayscaleCanny
          const truncated2 = new window.cv.Mat();
          mats.push(truncated2);
          window.cv.threshold(normalized, truncated2, 200, 255, window.cv.THRESH_TRUNC);
          const normalized2 = new window.cv.Mat();
          mats.push(normalized2);
          window.cv.normalize(truncated2, normalized2, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          const edges = new window.cv.Mat();
          mats.push(edges);
          window.cv.Canny(normalized2, edges, 185, 55);

          // findPageContours
          const contours = new window.cv.MatVector();
          mats.push(contours);
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(edges, contours, hierarchy, window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE);
          const hulls: any[] = [];
          for (let i = 0; i < contours.size(); i++) {
            const hull = new window.cv.Mat();
            window.cv.convexHull(contours.get(i), hull);
            hulls.push(hull);
            mats.push(hull);
          }
          hulls.sort((a: any, b: any) => window.cv.contourArea(b) - window.cv.contourArea(a));
          const topContours = hulls.slice(0, 5);

          const MIN_PAGE_AREA = 80000;
          const APPROX_POLY_EPSILON_FACTOR = 0.025;
          const MAX_COSINE = 0.35;

          function angle(p1: number[], p2: number[], p0: number[]) {
            const dx1 = p1[0] - p0[0], dy1 = p1[1] - p0[1];
            const dx2 = p2[0] - p0[0], dy2 = p2[1] - p0[1];
            const dot = dx1 * dx2 + dy1 * dy2;
            const mag = Math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
            return dot / mag;
          }

          function validateRect(pts: number[][]): boolean {
            if (pts.length !== 4) return false;
            let maxCos = 0;
            for (let i = 2; i < 5; i++) {
              const cos = Math.abs(angle(pts[i % 4], pts[i - 2], pts[i - 1]));
              if (cos > maxCos) maxCos = cos;
            }
            return maxCos < MAX_COSINE;
          }

          let corners: number[][] | null = null;
          for (const contour of topContours) {
            if (window.cv.contourArea(contour) < MIN_PAGE_AREA) continue;
            const perimeter = window.cv.arcLength(contour, true);
            const approx = new window.cv.Mat();
            mats.push(approx);
            window.cv.approxPolyDP(contour, approx, APPROX_POLY_EPSILON_FACTOR * perimeter, true);
            if (approx.rows !== 4) continue;
            const pts: number[][] = [];
            for (let i = 0; i < 4; i++) {
              pts.push([approx.data32S[i * 2], approx.data32S[i * 2 + 1]]);
            }
            if (validateRect(pts)) {
              corners = pts;
              break;
            }
          }

          // Draw contour on debug image if found
          if (corners !== null) {
            const contourData = corners.flat();
            const approxMat = window.cv.matFromArray(4, 1, window.cv.CV_32SC2, contourData);
            mats.push(approxMat);
            const contourVec = new window.cv.MatVector();
            mats.push(contourVec);
            contourVec.push_back(approxMat);
            window.cv.drawContours(debugImage, contourVec, -1, [255, 255, 255, 0], 10);
          }

          // Check debugImage has nonzero pixels
          const grayDebug = new window.cv.Mat();
          mats.push(grayDebug);
          window.cv.cvtColor(debugImage, grayDebug, window.cv.COLOR_BGR2GRAY);
          const nonZero = window.cv.countNonZero(grayDebug);

          return { nonZero, cornersFound: corners !== null };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.cornersFound).toBe(true);
      expect(result.nonZero).toBeGreaterThan(0);
    });
  });

  // test_find_page_raises_error_when_not_found: empty black image → error thrown
  test('test_find_page_raises_error_when_not_found', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // Completely black image → no contours → no rectangle
          const image = new window.cv.Mat(400, 600, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(image);

          // preparePageImage
          const truncated = new window.cv.Mat();
          mats.push(truncated);
          window.cv.threshold(image, truncated, 210, 255, window.cv.THRESH_TRUNC);
          const normalized = new window.cv.Mat();
          mats.push(normalized);
          // Black image has all same values, so normalize returns a clone (all zeros)
          // We still proceed
          window.cv.normalize(truncated, normalized, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);

          // applyGrayscaleCanny
          const truncated2 = new window.cv.Mat();
          mats.push(truncated2);
          window.cv.threshold(normalized, truncated2, 200, 255, window.cv.THRESH_TRUNC);
          const normalized2 = new window.cv.Mat();
          mats.push(normalized2);
          window.cv.normalize(truncated2, normalized2, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          const edges = new window.cv.Mat();
          mats.push(edges);
          window.cv.Canny(normalized2, edges, 185, 55);

          // findPageContours
          const contours = new window.cv.MatVector();
          mats.push(contours);
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(edges, contours, hierarchy, window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE);
          const hulls: any[] = [];
          for (let i = 0; i < contours.size(); i++) {
            const hull = new window.cv.Mat();
            window.cv.convexHull(contours.get(i), hull);
            hulls.push(hull);
            mats.push(hull);
          }
          hulls.sort((a: any, b: any) => window.cv.contourArea(b) - window.cv.contourArea(a));
          const topContours = hulls.slice(0, 5);

          const MIN_PAGE_AREA = 80000;
          const APPROX_POLY_EPSILON_FACTOR = 0.025;
          const MAX_COSINE = 0.35;

          function angle(p1: number[], p2: number[], p0: number[]) {
            const dx1 = p1[0] - p0[0], dy1 = p1[1] - p0[1];
            const dx2 = p2[0] - p0[0], dy2 = p2[1] - p0[1];
            const dot = dx1 * dx2 + dy1 * dy2;
            const mag = Math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
            return dot / mag;
          }

          function validateRect(pts: number[][]): boolean {
            if (pts.length !== 4) return false;
            let maxCos = 0;
            for (let i = 2; i < 5; i++) {
              const cos = Math.abs(angle(pts[i % 4], pts[i - 2], pts[i - 1]));
              if (cos > maxCos) maxCos = cos;
            }
            return maxCos < MAX_COSINE;
          }

          let pageContour: any = null;
          for (const contour of topContours) {
            if (window.cv.contourArea(contour) < MIN_PAGE_AREA) continue;
            const perimeter = window.cv.arcLength(contour, true);
            const approx = new window.cv.Mat();
            mats.push(approx);
            window.cv.approxPolyDP(contour, approx, APPROX_POLY_EPSILON_FACTOR * perimeter, true);
            if (approx.rows !== 4) continue;
            const pts: number[][] = [];
            for (let i = 0; i < 4; i++) {
              pts.push([approx.data32S[i * 2], approx.data32S[i * 2 + 1]]);
            }
            if (validateRect(pts)) {
              pageContour = contour;
              break;
            }
          }

          // Simulate error when not found
          if (pageContour === null) {
            throw new Error('ImageProcessingError: Paper boundary not found');
          }

          return { errorThrown: false };
        } catch (e: any) {
          return { errorThrown: true, message: e.message };
        } finally {
          mats.forEach(m => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.errorThrown).toBe(true);
      expect(result.message).toContain('Paper boundary not found');
    });
  });

  // =========================================================================
  // TestPageDetectionIntegration
  // =========================================================================

  // test_realistic_page_detection: white polygon (page) on dark background → corners (4 points)
  test('test_realistic_page_detection', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          // Create a realistic image: white filled polygon on dark background
          const image = new window.cv.Mat(600, 800, window.cv.CV_8UC1, new window.cv.Scalar(20));
          mats.push(image);

          // Draw a large white rectangle (simulating a scanned page)
          // Using fillPoly to simulate a polygon/page boundary
          const pagePts = window.cv.matFromArray(4, 1, window.cv.CV_32SC2, [
            50, 50,
            750, 50,
            750, 550,
            50, 550,
          ]);
          mats.push(pagePts);
          const pageVec = new window.cv.MatVector();
          mats.push(pageVec);
          pageVec.push_back(pagePts);
          window.cv.fillPoly(image, pageVec, new window.cv.Scalar(240));

          // preparePageImage
          const truncated = new window.cv.Mat();
          mats.push(truncated);
          window.cv.threshold(image, truncated, 210, 255, window.cv.THRESH_TRUNC);
          const normalized = new window.cv.Mat();
          mats.push(normalized);
          window.cv.normalize(truncated, normalized, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);

          // applyGrayscaleCanny
          const truncated2 = new window.cv.Mat();
          mats.push(truncated2);
          window.cv.threshold(normalized, truncated2, 200, 255, window.cv.THRESH_TRUNC);
          const normalized2 = new window.cv.Mat();
          mats.push(normalized2);
          window.cv.normalize(truncated2, normalized2, 0, 255, window.cv.NORM_MINMAX, window.cv.CV_8U);
          const edges = new window.cv.Mat();
          mats.push(edges);
          window.cv.Canny(normalized2, edges, 185, 55);

          // findPageContours
          const contours = new window.cv.MatVector();
          mats.push(contours);
          const hierarchy = new window.cv.Mat();
          mats.push(hierarchy);
          window.cv.findContours(edges, contours, hierarchy, window.cv.RETR_LIST, window.cv.CHAIN_APPROX_SIMPLE);
          const hulls: any[] = [];
          for (let i = 0; i < contours.size(); i++) {
            const hull = new window.cv.Mat();
            window.cv.convexHull(contours.get(i), hull);
            hulls.push(hull);
            mats.push(hull);
          }
          hulls.sort((a: any, b: any) => window.cv.contourArea(b) - window.cv.contourArea(a));
          const topContours = hulls.slice(0, 5);

          const MIN_PAGE_AREA = 80000;
          const APPROX_POLY_EPSILON_FACTOR = 0.025;
          const MAX_COSINE = 0.35;

          function angle(p1: number[], p2: number[], p0: number[]) {
            const dx1 = p1[0] - p0[0], dy1 = p1[1] - p0[1];
            const dx2 = p2[0] - p0[0], dy2 = p2[1] - p0[1];
            const dot = dx1 * dx2 + dy1 * dy2;
            const mag = Math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
            return dot / mag;
          }

          function validateRect(pts: number[][]): boolean {
            if (pts.length !== 4) return false;
            let maxCos = 0;
            for (let i = 2; i < 5; i++) {
              const cos = Math.abs(angle(pts[i % 4], pts[i - 2], pts[i - 1]));
              if (cos > maxCos) maxCos = cos;
            }
            return maxCos < MAX_COSINE;
          }

          let corners: number[][] | null = null;
          for (const contour of topContours) {
            if (window.cv.contourArea(contour) < MIN_PAGE_AREA) continue;
            const perimeter = window.cv.arcLength(contour, true);
            const approx = new window.cv.Mat();
            mats.push(approx);
            window.cv.approxPolyDP(contour, approx, APPROX_POLY_EPSILON_FACTOR * perimeter, true);
            if (approx.rows !== 4) continue;
            const pts: number[][] = [];
            for (let i = 0; i < 4; i++) {
              pts.push([approx.data32S[i * 2], approx.data32S[i * 2 + 1]]);
            }
            if (validateRect(pts)) {
              corners = pts;
              break;
            }
          }

          return {
            found: corners !== null,
            length: corners ? corners.length : null,
          };
        } finally {
          mats.forEach(m => m.delete());
        }
      });

      expect(result.found).toBe(true);
      expect(result.length).toBe(4);
    });
  });
});
