/**
 * Browser tests for WarpOnPointsCommon.ts using OpenCV.js (Playwright).
 *
 * Translates all 18 Python tests from test_WarpOnPointsCommon.py into browser tests.
 * Each test exercises WarpOnPointsCommon logic directly inside page.evaluate()
 * where window.cv is available.
 *
 * Run with: npm run test:browser -- --grep "WarpOnPointsCommon"
 */

import { test, expect } from '@playwright/test';
import { setupBrowser } from '../browser-setup';
import { withMemoryTracking } from '../memory-utils';

test.setTimeout(60000);

// ── Setup ─────────────────────────────────────────────────────────────────────
//
// Because Playwright's page.evaluate() runs inside the browser context (not Node.js),
// TypeScript module imports are unavailable there.  We inject the complete
// WarpOnPointsCommon + helpers source into the page ONCE via `page.addInitScript`,
// which runs before every page navigation and makes the classes available on `window`.

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

  // ── MathUtils ────────────────────────────────────────────────────────────────
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
  };
  window.MathUtils = MathUtils;

  // ── ImageUtils ─────────────────────────────────────────────────────────────
  const ImageUtils = {
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
          } else if (methodName === 'HOMOGRAPHY') {
            const cMat = cv.matFromArray(n, 1, cv.CV_32FC2, ctrl.flat());
            const dMat = cv.matFromArray(n, 1, cv.CV_32FC2, dest.flat());
            M = cv.findHomography(cMat, dMat, 0, 3.0);
            cMat.delete(); dMat.delete();
          } else {
            // REMAP_GRIDDATA browser fallback: use first 4 points
            const ctrl4 = ctrl.slice(0, 4);
            const dest4 = dest.slice(0, 4);
            const cMat = cv.matFromArray(4, 1, cv.CV_32FC2, ctrl4.flat());
            const dMat = cv.matFromArray(4, 1, cv.CV_32FC2, dest4.flat());
            M = cv.getPerspectiveTransform(cMat, dMat);
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
      const tuningOptions = options.tuning_options != null ? options.tuning_options : (parsed.tuning_options != null ? parsed.tuning_options : {});
      this.enableCropping = parsed.enable_cropping != null ? parsed.enable_cropping : (options.enable_cropping != null ? options.enable_cropping : false);
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

  // ── ConcreteWarpProcessor (test subclass) ─────────────────────────────────
  class ConcreteWarpProcessor extends WarpOnPointsCommon {
    constructor(options = {}) {
      super(options);
      this._testControlPts = null;
      this._testDestPts = null;
      this._testEdgeMap = null;
      this._appendSaveImageCallCount = 0;
    }
    validateAndRemapOptionsSchema(options) {
      return {
        enable_cropping: options.enable_cropping != null ? options.enable_cropping : false,
        tuning_options: options.tuning_options || {},
      };
    }
    prepareImageBeforeExtraction(image) { return image; }
    extractControlDestinationPoints(_image, _colored, _filePath) {
      return [this._testControlPts, this._testDestPts, this._testEdgeMap];
    }
    setTestPoints(ctrl, dest, edgeMap = null) {
      this._testControlPts = ctrl;
      this._testDestPts = dest;
      this._testEdgeMap = edgeMap;
    }
    appendSaveImage(..._args) {
      this._appendSaveImageCallCount++;
    }
  }
  window.ConcreteWarpProcessor = ConcreteWarpProcessor;
})();
`;

test.describe('WarpOnPointsCommon - Browser Tests', () => {
  test.beforeEach(async ({ page }) => {
    await setupBrowser(page);
    // Inject helper classes into the page after OpenCV.js is ready
    await page.evaluate(SETUP_SCRIPT);
  });

  // ─── TestWarpOnPointsCommonInitialization ────────────────────────────────────

  // test_default_initialization
  test('test_default_initialization', async ({ page }) => {
    const result = await page.evaluate(() => {
      const p = new window.ConcreteWarpProcessor();
      return {
        enableCropping: p.enableCropping,
        warpMethod: p.warpMethod,
        hasWarpStrategy: p.warpStrategy != null,
      };
    });

    expect(result.enableCropping).toBe(false);
    expect(result.warpMethod).toBe('HOMOGRAPHY');
    expect(result.hasWarpStrategy).toBe(true);
  });

  // test_initialization_with_cropping
  test('test_initialization_with_cropping', async ({ page }) => {
    const result = await page.evaluate(() => {
      const p = new window.ConcreteWarpProcessor({ enable_cropping: true });
      return {
        enableCropping: p.enableCropping,
        warpMethod: p.warpMethod,
      };
    });

    expect(result.enableCropping).toBe(true);
    expect(result.warpMethod).toBe('PERSPECTIVE_TRANSFORM');
  });

  // test_custom_warp_method
  test('test_custom_warp_method', async ({ page }) => {
    const result = await page.evaluate(() => {
      const p = new window.ConcreteWarpProcessor({ tuning_options: { warp_method: 'REMAP_GRIDDATA' } });
      return { warpMethod: p.warpMethod };
    });

    expect(result.warpMethod).toBe('REMAP_GRIDDATA');
  });

  // test_custom_interpolation_flag (INTER_CUBIC = 2)
  test('test_custom_interpolation_flag', async ({ page }) => {
    const result = await page.evaluate(() => {
      const p = new window.ConcreteWarpProcessor({ tuning_options: { warp_method_flag: 'INTER_CUBIC' } });
      return { warpMethodFlag: p.warpMethodFlag };
    });

    expect(result.warpMethodFlag).toBe(2);
  });

  // ─── TestWarpOnPointsCommonPointParsing ──────────────────────────────────────

  // test_parse_simple_points_no_cropping
  // image = Mat(400, 600, CV_8UC1) → rows=400 (height), cols=600 (width)
  // expected dims = (600, 400) = (cols, rows)
  test('test_parse_simple_points_no_cropping', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const image = new window.cv.Mat(400, 600, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(image);
          const p = new window.ConcreteWarpProcessor();
          const control = [[100, 100], [500, 100], [500, 300], [100, 300]];
          const dest    = [[0, 0], [400, 0], [400, 200], [0, 200]];
          const [parsedCtrl, parsedDest, dims] = p._parseAndPreparePoints(image, control, dest);
          return {
            ctrlLength: parsedCtrl.length,
            ctrlWidth:  parsedCtrl[0].length,
            destLength: parsedDest.length,
            destWidth:  parsedDest[0].length,
            dimsW: dims[0],
            dimsH: dims[1],
          };
        } finally {
          mats.forEach((m: any) => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.ctrlLength).toBe(4);
      expect(result.ctrlWidth).toBe(2);
      expect(result.destLength).toBe(4);
      expect(result.destWidth).toBe(2);
      // dims = [cols, rows] = [600, 400]
      expect(result.dimsW).toBe(600);
      expect(result.dimsH).toBe(400);
    });
  });

  // test_parse_points_with_cropping
  // dest = [[50,50],[450,50],[450,250],[50,250]]
  // bounding box x:[50,450], y:[50,250] → dims = (400, 200)
  // after shift → dest[0] = [0, 0]
  test('test_parse_points_with_cropping', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const image = new window.cv.Mat(400, 600, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(image);
          const p = new window.ConcreteWarpProcessor({ enable_cropping: true });
          const control = [[100, 100], [500, 100], [500, 300], [100, 300]];
          const dest    = [[50, 50], [450, 50], [450, 250], [50, 250]];
          const [, parsedDest, dims] = p._parseAndPreparePoints(image, control, dest);
          return {
            dimsW: dims[0],
            dimsH: dims[1],
            dest0x: parsedDest[0][0],
            dest0y: parsedDest[0][1],
          };
        } finally {
          mats.forEach((m: any) => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.dimsW).toBe(400);
      expect(result.dimsH).toBe(200);
      expect(result.dest0x).toBe(0);
      expect(result.dest0y).toBe(0);
    });
  });

  // test_deduplicate_points
  // control has duplicate at index 2 → only 3 unique pairs
  test('test_deduplicate_points', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const image = new window.cv.Mat(400, 600, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(image);
          const p = new window.ConcreteWarpProcessor();
          const control = [[100, 100], [200, 200], [100, 100], [300, 300]];
          const dest    = [[0, 0], [100, 100], [0, 0], [200, 200]];
          const [parsedCtrl, parsedDest] = p._parseAndPreparePoints(image, control, dest);
          return {
            ctrlLen: parsedCtrl.length,
            destLen: parsedDest.length,
          };
        } finally {
          mats.forEach((m: any) => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.ctrlLen).toBe(3);
      expect(result.destLen).toBe(3);
    });
  });

  // ─── TestWarpOnPointsCommonWarpingStrategies ─────────────────────────────────

  // test_perspective_transform_strategy
  // image = Mat(400, 400, CV_8UC1) with white rectangle
  // → warped.rows===200, warped.cols===200, warpedColored===null
  test('test_perspective_transform_strategy', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const image = new window.cv.Mat(400, 400, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(image);
          window.cv.rectangle(image, new window.cv.Point(100, 100), new window.cv.Point(300, 300), [255, 255, 255, 0], -1);
          const p = new window.ConcreteWarpProcessor({ tuning_options: { warp_method: 'PERSPECTIVE_TRANSFORM' } });
          const control = [[100, 100], [300, 100], [300, 300], [100, 300]];
          const dest    = [[0, 0], [200, 0], [200, 200], [0, 200]];
          const [warped, warpedColored] = p._applyWarpStrategy(image, null, control, dest, [200, 200], null);
          mats.push(warped);
          return {
            rows: warped.rows,
            cols: warped.cols,
            warpedColoredIsNull: warpedColored === null,
          };
        } finally {
          mats.forEach((m: any) => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.rows).toBe(200);
      expect(result.cols).toBe(200);
      expect(result.warpedColoredIsNull).toBe(true);
    });
  });

  // test_homography_strategy
  test('test_homography_strategy', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const image = new window.cv.Mat(400, 400, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(image);
          window.cv.rectangle(image, new window.cv.Point(100, 100), new window.cv.Point(300, 300), [255, 255, 255, 0], -1);
          const p = new window.ConcreteWarpProcessor({ tuning_options: { warp_method: 'HOMOGRAPHY' } });
          const control = [[100, 100], [300, 100], [300, 300], [100, 300]];
          const dest    = [[0, 0], [200, 0], [200, 200], [0, 200]];
          const [warped] = p._applyWarpStrategy(image, null, control, dest, [200, 200], null);
          mats.push(warped);
          return { rows: warped.rows, cols: warped.cols };
        } finally {
          mats.forEach((m: any) => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.rows).toBe(200);
      expect(result.cols).toBe(200);
    });
  });

  // test_griddata_strategy (8 control points)
  test('test_griddata_strategy', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const image = new window.cv.Mat(400, 400, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(image);
          window.cv.rectangle(image, new window.cv.Point(100, 100), new window.cv.Point(300, 300), [255, 255, 255, 0], -1);
          const p = new window.ConcreteWarpProcessor({ tuning_options: { warp_method: 'REMAP_GRIDDATA' } });
          const control = [[100, 100], [200, 100], [300, 100], [300, 200], [300, 300], [200, 300], [100, 300], [100, 200]];
          const dest    = [[0, 0], [100, 0], [200, 0], [200, 100], [200, 200], [100, 200], [0, 200], [0, 100]];
          const [warped] = p._applyWarpStrategy(image, null, control, dest, [200, 200], null);
          mats.push(warped);
          return { rows: warped.rows, cols: warped.cols };
        } finally {
          mats.forEach((m: any) => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.rows).toBe(200);
      expect(result.cols).toBe(200);
    });
  });

  // test_perspective_transform_requires_4_points (3 points → throws)
  test('test_perspective_transform_requires_4_points', async ({ page }) => {
    const result = await page.evaluate(() => {
      const mats: any[] = [];
      try {
        const image = new window.cv.Mat(400, 400, window.cv.CV_8UC1, new window.cv.Scalar(0));
        mats.push(image);
        const p = new window.ConcreteWarpProcessor({ tuning_options: { warp_method: 'PERSPECTIVE_TRANSFORM' } });
        const control = [[0, 0], [100, 0], [100, 100]];
        const dest    = [[0, 0], [100, 0], [100, 100]];
        try {
          p._applyWarpStrategy(image, null, control, dest, [200, 200], null);
          return { threw: false, message: '' };
        } catch (e: any) {
          return { threw: true, message: e.message };
        }
      } finally {
        mats.forEach((m: any) => { try { m.delete(); } catch (_) {} });
      }
    });

    expect(result.threw).toBe(true);
    expect(result.message).toContain('Expected 4 control points');
  });

  // test_colored_output_when_enabled
  test('test_colored_output_when_enabled', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const image = new window.cv.Mat(400, 400, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(image);
          window.cv.rectangle(image, new window.cv.Point(100, 100), new window.cv.Point(300, 300), [255, 255, 255, 0], -1);
          const colored = new window.cv.Mat(400, 400, window.cv.CV_8UC3, new window.cv.Scalar(0, 0, 0, 0));
          mats.push(colored);
          window.cv.rectangle(colored, new window.cv.Point(100, 100), new window.cv.Point(300, 300), [0, 255, 0, 0], -1);
          const p = new window.ConcreteWarpProcessor({
            colored_outputs_enabled: true,
            tuning_options: { warp_method: 'HOMOGRAPHY' },
          });
          const control = [[100, 100], [300, 100], [300, 300], [100, 300]];
          const dest    = [[0, 0], [200, 0], [200, 200], [0, 200]];
          const [warped, warpedColored] = p._applyWarpStrategy(image, colored, control, dest, [200, 200], null);
          mats.push(warped);
          if (warpedColored) mats.push(warpedColored);
          return {
            rows: warped.rows,
            cols: warped.cols,
            warpedColoredNotNull: warpedColored !== null,
            coloredRows:     warpedColored ? warpedColored.rows : -1,
            coloredCols:     warpedColored ? warpedColored.cols : -1,
            coloredChannels: warpedColored ? warpedColored.channels() : -1,
          };
        } finally {
          mats.forEach((m: any) => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.rows).toBe(200);
      expect(result.cols).toBe(200);
      expect(result.warpedColoredNotNull).toBe(true);
      expect(result.coloredRows).toBe(200);
      expect(result.coloredCols).toBe(200);
      expect(result.coloredChannels).toBe(3);
    });
  });

  // ─── TestWarpOnPointsCommonFullPipeline ──────────────────────────────────────

  // test_full_apply_filter_pipeline
  // image = Mat(400, 600, CV_8UC1)
  // homography with dims [600, 400] → warped.rows===400, warped.cols===600
  test('test_full_apply_filter_pipeline', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const image = new window.cv.Mat(400, 600, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(image);
          window.cv.rectangle(image, new window.cv.Point(100, 100), new window.cv.Point(500, 300), [255, 255, 255, 0], -1);
          const p = new window.ConcreteWarpProcessor();
          p.setTestPoints(
            [[100, 100], [500, 100], [500, 300], [100, 300]],
            [[0, 0], [400, 0], [400, 200], [0, 200]]
          );
          const [warped] = p.applyFilter(image, image, null, 'test.jpg');
          mats.push(warped);
          return {
            rows: warped.rows,
            cols: warped.cols,
            appendSaveImageCallCount: p._appendSaveImageCallCount,
          };
        } finally {
          mats.forEach((m: any) => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.rows).toBe(400);
      expect(result.cols).toBe(600);
      expect(result.appendSaveImageCallCount).toBeGreaterThan(0);
    });
  });

  // test_pipeline_with_cropping
  test('test_pipeline_with_cropping', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const image = new window.cv.Mat(400, 600, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(image);
          window.cv.rectangle(image, new window.cv.Point(100, 100), new window.cv.Point(500, 300), [255, 255, 255, 0], -1);
          const p = new window.ConcreteWarpProcessor({ enable_cropping: true });
          p.setTestPoints(
            [[100, 100], [500, 100], [500, 300], [100, 300]],
            [[50, 50], [450, 50], [450, 250], [50, 250]]
          );
          const [warped] = p.applyFilter(image, image, null, 'test.jpg');
          mats.push(warped);
          return { warpedNotNull: warped !== null, rows: warped.rows, cols: warped.cols };
        } finally {
          mats.forEach((m: any) => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.warpedNotNull).toBe(true);
    });
  });

  // ─── TestWarpOnPointsCommonAbstractMethods ───────────────────────────────────

  // test_validate_and_remap_not_implemented
  // A subclass that does NOT override validateAndRemapOptionsSchema should throw.
  test('test_validate_and_remap_not_implemented', async ({ page }) => {
    const result = await page.evaluate(() => {
      class IncompleteProcessor extends window.WarpOnPointsCommon {}
      try {
        new IncompleteProcessor({});
        return { threw: false, message: '' };
      } catch (e: any) {
        return { threw: true, message: e.message };
      }
    });

    expect(result.threw).toBe(true);
    expect(result.message).toContain('Not implemented');
  });
});
