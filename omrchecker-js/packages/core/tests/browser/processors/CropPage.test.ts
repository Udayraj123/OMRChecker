/**
 * Browser tests for CropPage.ts using OpenCV.js (Playwright).
 *
 * Tests the CropPage processor class which detects and crops the page boundary
 * from an image using edge detection and perspective warp.
 *
 * Run with: npx playwright test tests/browser/processors/CropPage.test.ts --reporter=list
 */

import { test, expect } from '@playwright/test';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { setupBrowser } from '../browser-setup';
import { withMemoryTracking } from '../memory-utils';
import { SHARED_UTILS_SCRIPT } from './shared-browser-utils';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Real sample image: 747×1024 mobile-camera photo on dark background
const SHEET1_PATH = join(__dirname, '../../../../../../samples/1-mobile-camera/MobileCamera/sheet1.jpg');

test.setTimeout(90_000);

// ── Setup script ─────────────────────────────────────────────────────────────
//
// SHARED_UTILS_SCRIPT provides: WarpMethod, MathUtils, ImageUtils,
// WarpStrategyFactory, WarpOnPointsCommon, __loadGray.
// CROP_PAGE_SCRIPT adds the page_detection helpers and CropPage class.

const CROP_PAGE_SCRIPT = `
(function() {
  const cv                 = window.cv;
  const WarpMethod         = window.WarpMethod;
  const WARP_METHOD_FLAG_VALUES = window.WARP_METHOD_FLAG_VALUES;
  const MathUtils          = window.MathUtils;
  const ImageUtils         = window.ImageUtils;
  const WarpOnPointsCommon = window.WarpOnPointsCommon;

  // ── page_detection constants ──────────────────────────────────────────────
  const THRESH_PAGE_TRUNCATE_HIGH      = 210;
  const THRESH_PAGE_TRUNCATE_SECONDARY = 200;
  const CANNY_THRESHOLD_HIGH           = 185;
  const CANNY_THRESHOLD_LOW            = 55;
  const MIN_PAGE_AREA                  = 80000;
  const APPROX_POLY_EPSILON_FACTOR     = 0.025;
  const TOP_CONTOURS_COUNT             = 5;

  // ── page_detection helpers ─────────────────────────────────────────────────
  function preparePageImage(image) {
    const cv = window.cv;
    const truncated = new cv.Mat();
    cv.threshold(image, truncated, THRESH_PAGE_TRUNCATE_HIGH, 255, cv.THRESH_TRUNC);
    const normalized = ImageUtils.normalizeSingle(truncated);
    truncated.delete();
    return normalized;
  }

  function applyGrayscaleCanny(image, morphKernel) {
    const cv = window.cv;
    const truncated = new cv.Mat();
    cv.threshold(image, truncated, THRESH_PAGE_TRUNCATE_SECONDARY, 255, cv.THRESH_TRUNC);
    const normalized = ImageUtils.normalizeSingle(truncated);
    truncated.delete();
    let processed;
    let ownedClosed = null;
    if (morphKernel && morphKernel.rows > 1) {
      ownedClosed = new cv.Mat();
      cv.morphologyEx(normalized, ownedClosed, cv.MORPH_CLOSE, morphKernel);
      processed = ownedClosed;
    } else {
      processed = normalized;
    }
    const edges = new cv.Mat();
    try {
      cv.Canny(processed, edges, CANNY_THRESHOLD_HIGH, CANNY_THRESHOLD_LOW);
      return edges.clone();
    } finally {
      edges.delete();
      normalized.delete();
      if (ownedClosed) ownedClosed.delete();
    }
  }

  function findPageContours(cannyEdge) {
    const cv = window.cv;
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    try {
      cv.findContours(cannyEdge, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
    } finally {
      hierarchy.delete();
    }
    const hullList = [];
    const size = contours.size();
    for (let i = 0; i < size; i++) {
      const hull = new cv.Mat();
      cv.convexHull(contours.get(i), hull);
      hullList.push(hull);
    }
    contours.delete();
    hullList.sort((a, b) => cv.contourArea(b) - cv.contourArea(a));
    const topContours = hullList.slice(0, TOP_CONTOURS_COUNT);
    for (let i = TOP_CONTOURS_COUNT; i < hullList.length; i++) hullList[i].delete();
    return topContours;
  }

  function extractPageRectangle(contours) {
    const cv = window.cv;
    for (const contour of contours) {
      const area = cv.contourArea(contour);
      if (area < MIN_PAGE_AREA) continue;
      const perimeter = cv.arcLength(contour, true);
      const epsilon = APPROX_POLY_EPSILON_FACTOR * perimeter;
      const approx = new cv.Mat();
      try {
        cv.approxPolyDP(contour, approx, epsilon, true);
        if (approx.rows !== 4) continue;
        const points = [];
        for (let i = 0; i < 4; i++) {
          points.push([approx.data32S[i * 2], approx.data32S[i * 2 + 1]]);
        }
        if (MathUtils.validateRect(points)) {
          const corners = points.map(([x, y]) => [x, y]);
          const fullContour = contour.clone();
          return [corners, fullContour];
        }
      } finally {
        approx.delete();
      }
    }
    return [null, null];
  }

  function findPageContourAndCorners(image, options) {
    options = options || {};
    const { morphKernel, filePath } = options;
    const prepared = preparePageImage(image);
    let cannyEdge;
    try {
      cannyEdge = applyGrayscaleCanny(prepared, morphKernel);
    } finally {
      prepared.delete();
    }
    let contours;
    try {
      contours = findPageContours(cannyEdge);
    } finally {
      cannyEdge.delete();
    }
    let corners = null;
    let pageContour = null;
    try {
      [corners, pageContour] = extractPageRectangle(contours);
    } finally {
      for (const c of contours) c.delete();
    }
    if (pageContour === null) {
      const err = new Error('Paper boundary not found');
      err.name = 'ImageProcessingError';
      err.filePath = filePath;
      err.reason = 'No valid rectangle found in top contour candidates';
      throw err;
    }
    return [corners, pageContour];
  }
  window.findPageContourAndCorners = findPageContourAndCorners;

  // ── CropPage ──────────────────────────────────────────────────────────────
  class CropPage extends WarpOnPointsCommon {
    constructor(options = {}) {
      super(options);
      const morphKernel = options.morph_kernel || options.morphKernel || [10, 10];
      this.useColoredCanny = options.use_colored_canny || options.useColoredCanny || false;
      this.morphKernel = window.cv.getStructuringElement(
        window.cv.MORPH_RECT,
        new window.cv.Size(morphKernel[0], morphKernel[1])
      );
    }

    validateAndRemapOptionsSchema(options) {
      const tuningOptions = options.tuning_options || options.tuningOptions || {};
      const morphKernel = options.morph_kernel || options.morphKernel || [10, 10];
      const useColoredCanny = options.use_colored_canny || options.useColoredCanny || false;
      return {
        morph_kernel: morphKernel,
        use_colored_canny: useColoredCanny,
        max_points_per_edge: options.max_points_per_edge || null,
        enable_cropping: true,
        tuning_options: {
          warp_method: tuningOptions.warp_method || WarpMethod.PERSPECTIVE_TRANSFORM,
          normalize_config: [],
          canny_config: [],
        },
      };
    }

    prepareImageBeforeExtraction(image) {
      return ImageUtils.normalizeSingle(image);
    }

    extractControlDestinationPoints(image, _coloredImage, filePath) {
      const [corners, pageContour] = window.findPageContourAndCorners(image, {
        morphKernel: this.morphKernel,
        useColoredCanny: this.useColoredCanny,
        filePath,
      });
      pageContour.delete();
      const [warpedPoints] = ImageUtils.getCroppedWarpedRectanglePoints(corners);
      return [corners, warpedPoints, null];
    }

    dispose() {
      if (this.morphKernel && !this.morphKernel.isDeleted()) {
        this.morphKernel.delete();
      }
    }
  }
  window.CropPage = CropPage;
})();
`;

// Combine shared base + CropPage-specific code.
const SETUP_SCRIPT = SHARED_UTILS_SCRIPT + '\n' + CROP_PAGE_SCRIPT;

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

test.describe('CropPage - Browser Tests', () => {
  test.beforeEach(async ({ page }) => {
    await setupBrowser(page);
    await page.evaluate(SETUP_SCRIPT);
  });

  // ── Test 1: constructor initializes with defaults ─────────────────────────
  test('constructor initializes with default options', async ({ page }) => {
    const result = await page.evaluate(() => {
      const cropPage = new window.CropPage({});
      const result = {
        enableCropping: cropPage.enableCropping,
        warpMethod: cropPage.warpMethod,
        useColoredCanny: cropPage.useColoredCanny,
        hasMorphKernel: cropPage.morphKernel != null && !cropPage.morphKernel.isDeleted(),
        morphKernelRows: cropPage.morphKernel ? cropPage.morphKernel.rows : -1,
        morphKernelCols: cropPage.morphKernel ? cropPage.morphKernel.cols : -1,
      };
      cropPage.dispose();
      return result;
    });

    expect(result.enableCropping).toBe(true);
    expect(result.warpMethod).toBe('PERSPECTIVE_TRANSFORM');
    expect(result.useColoredCanny).toBe(false);
    expect(result.hasMorphKernel).toBe(true);
    // Default morphKernel is [10, 10] — getStructuringElement returns 10×10 kernel
    expect(result.morphKernelRows).toBe(10);
    expect(result.morphKernelCols).toBe(10);
  });

  // ── Test 2: validateAndRemapOptionsSchema correctness ────────────────────
  test('validateAndRemapOptionsSchema returns correct schema', async ({ page }) => {
    const result = await page.evaluate(() => {
      const cropPage = new window.CropPage({
        morph_kernel: [15, 15],
        use_colored_canny: false,
      });
      const parsed = cropPage.validateAndRemapOptionsSchema({
        morph_kernel: [15, 15],
        use_colored_canny: true,
      });
      cropPage.dispose();
      return {
        enableCropping: parsed.enable_cropping,
        warpMethod: parsed.tuning_options.warp_method,
        morphKernel: parsed.morph_kernel,
        useColoredCanny: parsed.use_colored_canny,
        hasNormalizeConfig: Array.isArray(parsed.tuning_options.normalize_config),
        hasCannyConfig: Array.isArray(parsed.tuning_options.canny_config),
      };
    });

    expect(result.enableCropping).toBe(true);
    expect(result.warpMethod).toBe('PERSPECTIVE_TRANSFORM');
    expect(result.morphKernel).toEqual([15, 15]);
    expect(result.useColoredCanny).toBe(true);
    expect(result.hasNormalizeConfig).toBe(true);
    expect(result.hasCannyConfig).toBe(true);
  });

  // ── Test 3: applyFilter on real sample image ───────────────────────────────
  test('applyFilter on real sample image produces a valid cropped output', async ({ page }) => {
    const sheet1B64 = readFileSync(SHEET1_PATH).toString('base64');

    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(async (b64: string) => {
        const mats: any[] = [];
        try {
          const gray = await (window as any).__loadGray(b64);
          mats.push(gray);

          const cropPage = new (window as any).CropPage({ morph_kernel: [10, 10] });
          const [warped] = cropPage.applyFilter(gray, null, null, 'sheet1.jpg');
          mats.push(warped);
          cropPage.dispose();

          const mean = (window.cv as any).mean(warped)[0];
          return {
            rows: warped.rows,
            cols: warped.cols,
            channels: warped.channels(),
            mean,
          };
        } finally {
          mats.forEach((m: any) => { try { m.delete(); } catch (_) {} });
        }
      }, sheet1B64);

      // Output must be a valid single-channel image
      expect(result.channels).toBe(1);
      // Warped sheet should be at least 200px in each dimension
      expect(result.rows).toBeGreaterThan(200);
      expect(result.cols).toBeGreaterThan(200);
      // After cropping white OMR sheet from dark background, mean is brighter
      expect(result.mean).toBeGreaterThan(100);
    });
  });

  // ── Test 4: throws ImageProcessingError on solid black image ─────────────
  test('applyFilter throws ImageProcessingError when page not found', async ({ page }) => {
    const result = await page.evaluate(() => {
      const mats: any[] = [];
      try {
        // Solid black image — no page contour can be found
        const blackImage = new window.cv.Mat(200, 200, window.cv.CV_8UC1, new window.cv.Scalar(0));
        mats.push(blackImage);

        const cropPage = new window.CropPage({});
        try {
          cropPage.applyFilter(blackImage, null, null, 'black.jpg');
          cropPage.dispose();
          return { threw: false, message: '', errorName: '' };
        } catch (e: any) {
          cropPage.dispose();
          return { threw: true, message: e.message, errorName: e.name };
        }
      } finally {
        mats.forEach((m: any) => { try { m.delete(); } catch (_) {} });
      }
    });

    expect(result.threw).toBe(true);
    expect(result.message).toContain('Paper boundary not found');
  });

  // ── Test 5: custom morphKernel and dispose cleans up ─────────────────────
  test('custom morphKernel is applied and dispose releases memory', async ({ page }) => {
    const result = await page.evaluate(() => {
      const cropPage = new window.CropPage({ morph_kernel: [5, 5] });
      const kernelRows = cropPage.morphKernel.rows;
      const kernelCols = cropPage.morphKernel.cols;
      const beforeDispose = !cropPage.morphKernel.isDeleted();
      cropPage.dispose();
      const afterDispose = cropPage.morphKernel.isDeleted();
      return { kernelRows, kernelCols, beforeDispose, afterDispose };
    });

    expect(result.kernelRows).toBe(5);
    expect(result.kernelCols).toBe(5);
    expect(result.beforeDispose).toBe(true);
    expect(result.afterDispose).toBe(true);
  });
});
