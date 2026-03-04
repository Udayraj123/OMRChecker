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

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Real sample image: 747×1024 mobile-camera photo on dark background
const SHEET1_PATH = join(__dirname, '../../../../../../samples/1-mobile-camera/MobileCamera/sheet1.jpg');

test.setTimeout(90_000);

// ── Setup script ─────────────────────────────────────────────────────────────
//
// Playwright's page.evaluate() runs inside the browser, so TypeScript module
// imports are unavailable there. We inject all required helpers + the CropPage
// class as plain JavaScript strings via page.addInitScript / page.evaluate.

const SETUP_SCRIPT = `
(function() {
  // ── Constants ──────────────────────────────────────────────────────────────
  const WarpMethod = {
    PERSPECTIVE_TRANSFORM: 'PERSPECTIVE_TRANSFORM',
    HOMOGRAPHY: 'HOMOGRAPHY',
    REMAP_GRIDDATA: 'REMAP_GRIDDATA',
    DOC_REFINE: 'DOC_REFINE',
  };
  window.WarpMethod = WarpMethod;

  const WARP_METHOD_FLAG_VALUES = {
    INTER_LINEAR: 1,
    INTER_CUBIC: 2,
    INTER_NEAREST: 0,
  };
  window.WARP_METHOD_FLAG_VALUES = WARP_METHOD_FLAG_VALUES;

  // page_detection constants
  const THRESH_PAGE_TRUNCATE_HIGH = 210;
  const THRESH_PAGE_TRUNCATE_SECONDARY = 200;
  const CANNY_THRESHOLD_HIGH = 185;
  const CANNY_THRESHOLD_LOW = 55;
  const MIN_PAGE_AREA = 80000;
  const APPROX_POLY_EPSILON_FACTOR = 0.025;
  const TOP_CONTOURS_COUNT = 5;

  // ── MathUtils ────────────────────────────────────────────────────────────
  const MathUtils = {
    distance(p1, p2) {
      const dx = p1[0] - p2[0];
      const dy = p1[1] - p2[1];
      return Math.hypot(dx, dy);
    },
    getBoundingBoxOfPoints(points) {
      const xs = points.map(p => p[0]);
      const ys = points.map(p => p[1]);
      const minX = Math.min(...xs);
      const minY = Math.min(...ys);
      const maxX = Math.max(...xs);
      const maxY = Math.max(...ys);
      const boundingBox = [[minX, minY], [maxX, minY], [maxX, maxY], [minX, maxY]];
      const boxDimensions = [Math.floor(maxX - minX), Math.floor(maxY - minY)];
      return { boundingBox, boxDimensions };
    },
    shiftPointsFromOrigin(newOrigin, listOfPoints) {
      return listOfPoints.map(p => [newOrigin[0] + p[0], newOrigin[1] + p[1]]);
    },
    orderFourPoints(points) {
      const sums  = points.map(p => p[0] + p[1]);
      const diffs = points.map(p => p[1] - p[0]);
      const minSumIdx  = sums.indexOf(Math.min(...sums));
      const minDiffIdx = diffs.indexOf(Math.min(...diffs));
      const maxSumIdx  = sums.indexOf(Math.max(...sums));
      const maxDiffIdx = diffs.indexOf(Math.max(...diffs));
      const rect = [points[minSumIdx], points[minDiffIdx], points[maxSumIdx], points[maxDiffIdx]];
      return { rect, orderedIndices: [minSumIdx, minDiffIdx, maxSumIdx, maxDiffIdx] };
    },
    validateRect(points) {
      if (points.length !== 4) return false;
      // Check max cosine of angles at each corner < 0.35
      let maxCos = 0;
      for (let i = 2; i < 5; i++) {
        const p1 = points[i % 4];
        const p2 = points[(i - 2) % 4];
        const p0 = points[(i - 1) % 4];
        const dx1 = p1[0] - p0[0], dy1 = p1[1] - p0[1];
        const dx2 = p2[0] - p0[0], dy2 = p2[1] - p0[1];
        const cos = Math.abs((dx1*dx2 + dy1*dy2) / Math.sqrt((dx1**2+dy1**2)*(dx2**2+dy2**2) + 1e-10));
        if (cos > maxCos) maxCos = cos;
      }
      return maxCos < 0.35;
    },
  };
  window.MathUtils = MathUtils;

  // ── ImageUtils ─────────────────────────────────────────────────────────────
  const ImageUtils = {
    normalizeSingle(image, alpha, beta, normType) {
      const cv = window.cv;
      alpha = alpha != null ? alpha : 0;
      beta  = beta  != null ? beta  : 255;
      normType = normType != null ? normType : cv.NORM_MINMAX;
      if (!image || image.empty()) return image;
      const minMax = cv.minMaxLoc(image, new cv.Mat());
      if (minMax.maxVal === minMax.minVal) return image.clone();
      const normalized = new cv.Mat();
      cv.normalize(image, normalized, alpha, beta, normType);
      return normalized;
    },
    getCroppedWarpedRectanglePoints(orderedCorners) {
      const [tl, tr, br, bl] = orderedCorners;
      const maxWidth = Math.max(
        Math.floor(MathUtils.distance(tr, tl)),
        Math.floor(MathUtils.distance(br, bl))
      );
      const maxHeight = Math.max(
        Math.floor(MathUtils.distance(tr, br)),
        Math.floor(MathUtils.distance(tl, bl))
      );
      const warpedPoints = [
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1],
      ];
      return [warpedPoints, [maxWidth, maxHeight]];
    },
  };
  window.ImageUtils = ImageUtils;

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

  // ── WarpStrategyFactory ────────────────────────────────────────────────────
  const WarpStrategyFactory = {
    create(methodName) {
      return {
        methodName,
        warpImage(image, coloredImage, ctrl, dest, dims) {
          const cv = window.cv;
          const [w, h] = dims;
          const n = ctrl.length;
          const dsize = new cv.Size(w, h);
          let M;
          if (methodName === 'PERSPECTIVE_TRANSFORM') {
            const cMat = cv.matFromArray(4, 1, cv.CV_32FC2, ctrl.flat());
            const dMat = cv.matFromArray(4, 1, cv.CV_32FC2, dest.flat());
            M = cv.getPerspectiveTransform(cMat, dMat);
            cMat.delete(); dMat.delete();
          } else {
            const cMat = cv.matFromArray(n, 1, cv.CV_32FC2, ctrl.flat());
            const dMat = cv.matFromArray(n, 1, cv.CV_32FC2, dest.flat());
            M = cv.findHomography(cMat, dMat, 0, 3.0);
            cMat.delete(); dMat.delete();
          }
          const warpedGray = new cv.Mat();
          cv.warpPerspective(image, warpedGray, M, dsize, cv.INTER_LINEAR);
          let warpedColored = null;
          if (coloredImage) {
            warpedColored = new cv.Mat();
            cv.warpPerspective(coloredImage, warpedColored, M, dsize, cv.INTER_LINEAR);
          }
          M.delete();
          return { warpedGray, warpedColored, warpedDebug: null };
        },
      };
    },
  };
  window.WarpStrategyFactory = WarpStrategyFactory;

  // ── WarpOnPointsCommon (abstract base) ────────────────────────────────────
  class WarpOnPointsCommon {
    constructor(options = {}) {
      const parsed = this.validateAndRemapOptionsSchema(options);
      const tuningOptions = options.tuning_options != null ? options.tuning_options
        : (parsed.tuning_options != null ? parsed.tuning_options : {});
      this.enableCropping = parsed.enable_cropping != null ? parsed.enable_cropping
        : (options.enable_cropping != null ? options.enable_cropping : false);
      this.warpMethod = tuningOptions.warp_method != null ? tuningOptions.warp_method
        : (this.enableCropping ? WarpMethod.PERSPECTIVE_TRANSFORM : WarpMethod.HOMOGRAPHY);
      const flagKey = tuningOptions.warp_method_flag != null ? tuningOptions.warp_method_flag : 'INTER_LINEAR';
      this.warpMethodFlag = WARP_METHOD_FLAG_VALUES[flagKey] != null ? WARP_METHOD_FLAG_VALUES[flagKey] : 1;
      this.coloredOutputsEnabled = options.colored_outputs_enabled != null ? options.colored_outputs_enabled : false;
      this.warpStrategy = WarpStrategyFactory.create(this.warpMethod);
    }

    validateAndRemapOptionsSchema(_options) { throw new Error('Not implemented'); }
    prepareImageBeforeExtraction(_image)    { throw new Error('Not implemented'); }
    extractControlDestinationPoints(_image, _colored, _filePath) { throw new Error('Not implemented'); }

    appendSaveImage(..._args) {}

    applyFilter(image, coloredImage, template, filePath) {
      const prepared = this.prepareImageBeforeExtraction(image);
      const [controlPts, destPts, edgeMap] = this.extractControlDestinationPoints(prepared, coloredImage, filePath);
      const [parsedCtrl, parsedDest, dims] = this._parseAndPreparePoints(prepared, controlPts, destPts);
      const [warpedImage, warpedColored] = this._applyWarpStrategy(image, coloredImage, parsedCtrl, parsedDest, dims, edgeMap);
      this.appendSaveImage('Warped Image', [4, 5, 6], warpedImage, warpedColored);
      return [warpedImage, warpedColored, template];
    }

    _parseAndPreparePoints(image, controlPoints, destinationPoints) {
      const seen = new Map();
      const uniqueCtrl = [];
      const uniqueDest = [];
      for (let i = 0; i < controlPoints.length; i++) {
        const key = JSON.stringify(controlPoints[i]);
        if (!seen.has(key)) {
          seen.set(key, destinationPoints[i]);
          uniqueCtrl.push(controlPoints[i]);
          uniqueDest.push(destinationPoints[i]);
        }
      }
      const dims = this._calculateWarpedDimensions([image.cols, image.rows], uniqueDest);
      return [uniqueCtrl, uniqueDest, dims];
    }

    _calculateWarpedDimensions(defaultDims, destinationPoints) {
      if (!this.enableCropping) return defaultDims;
      const { boundingBox, boxDimensions } = MathUtils.getBoundingBoxOfPoints(destinationPoints);
      const fromOrigin = [-boundingBox[0][0], -boundingBox[0][1]];
      const shifted = MathUtils.shiftPointsFromOrigin(fromOrigin, destinationPoints);
      for (let i = 0; i < destinationPoints.length; i++) {
        destinationPoints[i] = shifted[i];
      }
      return boxDimensions;
    }

    _applyWarpStrategy(image, coloredImage, controlPoints, destinationPoints, warpedDimensions, _edgeContoursMap) {
      const [ctrl, dest, dims] = this._preparePointsForStrategy(controlPoints, destinationPoints, warpedDimensions);
      const coloredInput = this.coloredOutputsEnabled ? coloredImage : null;
      const result = this.warpStrategy.warpImage(image, coloredInput, ctrl, dest, dims);
      return [result.warpedGray, result.warpedColored];
    }

    _preparePointsForStrategy(controlPoints, destinationPoints, warpedDimensions) {
      if (this.warpMethod !== WarpMethod.PERSPECTIVE_TRANSFORM) {
        return [controlPoints, destinationPoints, warpedDimensions];
      }
      if (controlPoints.length !== 4) {
        throw new Error('Expected 4 control points for perspective transform, found ' + controlPoints.length + '.');
      }
      const { rect: orderedCtrl } = MathUtils.orderFourPoints(controlPoints);
      const [newDest, newDims] = ImageUtils.getCroppedWarpedRectanglePoints(orderedCtrl);
      return [orderedCtrl, newDest, newDims];
    }
  }
  window.WarpOnPointsCommon = WarpOnPointsCommon;

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

  // ── Image loader helper ────────────────────────────────────────────────────
  window.__loadGray = async function(b64) {
    const img = new Image();
    img.src = 'data:image/jpeg;base64,' + b64;
    await new Promise((res, rej) => { img.onload = res; img.onerror = rej; });
    const canvas = document.createElement('canvas');
    canvas.width = img.width; canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const rgba = window.cv.matFromImageData(ctx.getImageData(0, 0, img.width, img.height));
    const gray = new window.cv.Mat();
    window.cv.cvtColor(rgba, gray, window.cv.COLOR_RGBA2GRAY);
    rgba.delete();
    return gray;
  };
})();
`;

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
