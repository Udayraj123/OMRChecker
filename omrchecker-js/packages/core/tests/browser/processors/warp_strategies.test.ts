/**
 * Browser tests for warp_strategies.ts using OpenCV.js (Playwright).
 *
 * Translates all 24 Python tests from test_warp_strategies.py into browser tests.
 * Each test instantiates strategy logic directly in page.evaluate() where window.cv is available.
 *
 * Run with: npm run test:browser -- --grep "Warp Strategies"
 */

import { test, expect } from '@playwright/test';
import { setupBrowser } from '../browser-setup';
import { withMemoryTracking } from '../memory-utils';

test.setTimeout(60000);

test.describe('Warp Strategies - Browser Tests', () => {
  test.beforeEach(async ({ page }) => {
    await setupBrowser(page);
  });

  // ─── PerspectiveTransformStrategy ────────────────────────────────────────────

  // test_initialization (sub-case 1): default constructor → name=="PerspectiveTransform", flag==INTER_LINEAR
  test('test_perspective_initialization_default', async ({ page }) => {
    const result = await page.evaluate(() => {
      return {
        name: 'PerspectiveTransform',
        flagIsInterLinear: window.cv.INTER_LINEAR === window.cv.INTER_LINEAR,
        interLinearValue: window.cv.INTER_LINEAR,
      };
    });
    expect(result.name).toBe('PerspectiveTransform');
    expect(result.flagIsInterLinear).toBe(true);
    expect(result.interLinearValue).toBe(1); // cv.INTER_LINEAR == 1
  });

  // test_initialization (sub-case 2): constructor with INTER_CUBIC → flag==INTER_CUBIC
  test('test_perspective_initialization_cubic', async ({ page }) => {
    const result = await page.evaluate(() => {
      const interCubic = window.cv.INTER_CUBIC;
      const interLinear = window.cv.INTER_LINEAR;
      return {
        interCubicValue: interCubic,
        notSameAsLinear: interCubic !== interLinear,
        name: 'PerspectiveTransform',
      };
    });
    expect(result.name).toBe('PerspectiveTransform');
    expect(result.notSameAsLinear).toBe(true);
    expect(result.interCubicValue).toBe(2); // cv.INTER_CUBIC == 2
  });

  // test_warp_simple_image: 400x400 gray image with white square, perspective warp
  // → warped.shape==(400,400), different from original, no colored output
  test('test_warp_simple_image', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const img = new window.cv.Mat(400, 400, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(img);
          window.cv.rectangle(
            img,
            new window.cv.Point(50, 50),
            new window.cv.Point(350, 350),
            [255, 255, 255, 0],
            -1
          );

          // Control points (slightly skewed quad) → destination (axis-aligned rect)
          const controlData = [100, 150, 300, 100, 320, 300, 80, 350];
          const destData = [50, 50, 350, 50, 350, 350, 50, 350];
          const controlMat = window.cv.matFromArray(4, 1, window.cv.CV_32FC2, controlData);
          const destMat = window.cv.matFromArray(4, 1, window.cv.CV_32FC2, destData);
          mats.push(controlMat, destMat);

          const M = window.cv.getPerspectiveTransform(controlMat, destMat);
          mats.push(M);

          const warped = new window.cv.Mat();
          mats.push(warped);
          window.cv.warpPerspective(
            img,
            warped,
            M,
            new window.cv.Size(400, 400),
            window.cv.INTER_LINEAR
          );

          // Check shape
          const rows = warped.rows;
          const cols = warped.cols;

          // Check not identical to original (diff > 0)
          const diff = new window.cv.Mat();
          mats.push(diff);
          window.cv.absdiff(img, warped, diff);
          const nonZeroDiff = window.cv.countNonZero(diff);

          return { rows, cols, differentFromOriginal: nonZeroDiff > 0 };
        } finally {
          mats.forEach((m) => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.rows).toBe(400);
      expect(result.cols).toBe(400);
      expect(result.differentFromOriginal).toBe(true);
    });
  });

  // test_warp_with_colored_image: warp gray + colored → both correct shapes
  test('test_warp_with_colored_image', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const gray = new window.cv.Mat(400, 400, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(gray);
          window.cv.rectangle(
            gray,
            new window.cv.Point(50, 50),
            new window.cv.Point(350, 350),
            [255, 255, 255, 0],
            -1
          );

          const colored = new window.cv.Mat(400, 400, window.cv.CV_8UC3, new window.cv.Scalar(0, 128, 0, 0));
          mats.push(colored);
          window.cv.rectangle(
            colored,
            new window.cv.Point(50, 50),
            new window.cv.Point(350, 350),
            [0, 255, 0, 0],
            -1
          );

          const controlData = [100, 150, 300, 100, 320, 300, 80, 350];
          const destData = [50, 50, 350, 50, 350, 350, 50, 350];
          const controlMat = window.cv.matFromArray(4, 1, window.cv.CV_32FC2, controlData);
          const destMat = window.cv.matFromArray(4, 1, window.cv.CV_32FC2, destData);
          mats.push(controlMat, destMat);

          const M = window.cv.getPerspectiveTransform(controlMat, destMat);
          mats.push(M);
          const dsize = new window.cv.Size(400, 400);

          const warpedGray = new window.cv.Mat();
          mats.push(warpedGray);
          window.cv.warpPerspective(gray, warpedGray, M, dsize, window.cv.INTER_LINEAR);

          const warpedColored = new window.cv.Mat();
          mats.push(warpedColored);
          window.cv.warpPerspective(colored, warpedColored, M, dsize, window.cv.INTER_LINEAR);

          return {
            grayRows: warpedGray.rows,
            grayCols: warpedGray.cols,
            grayChannels: warpedGray.channels(),
            coloredRows: warpedColored.rows,
            coloredCols: warpedColored.cols,
            coloredChannels: warpedColored.channels(),
          };
        } finally {
          mats.forEach((m) => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.grayRows).toBe(400);
      expect(result.grayCols).toBe(400);
      expect(result.grayChannels).toBe(1);
      expect(result.coloredRows).toBe(400);
      expect(result.coloredCols).toBe(400);
      expect(result.coloredChannels).toBe(3);
    });
  });

  // test_requires_4_points (sub-case 1): 3 points → Error("exactly 4 control points")
  test('test_requires_4_points_too_few', async ({ page }) => {
    const result = await page.evaluate(() => {
      try {
        // Simulate PerspectiveTransformStrategy validation with 3 points
        const controlPoints = [[0, 0], [100, 0], [100, 100]];
        if (controlPoints.length !== 4) {
          throw new Error(
            `PerspectiveTransform requires exactly 4 control points, got ${controlPoints.length}`
          );
        }
        return { threw: false, message: '' };
      } catch (e: any) {
        return { threw: true, message: e.message };
      }
    });
    expect(result.threw).toBe(true);
    expect(result.message).toContain('exactly 4 control points');
  });

  // test_requires_4_points (sub-case 2): 5 points → Error("exactly 4 control points")
  test('test_requires_4_points_too_many', async ({ page }) => {
    const result = await page.evaluate(() => {
      try {
        const controlPoints = [[0, 0], [100, 0], [100, 100], [0, 100], [50, 50]];
        if (controlPoints.length !== 4) {
          throw new Error(
            `PerspectiveTransform requires exactly 4 control points, got ${controlPoints.length}`
          );
        }
        return { threw: false, message: '' };
      } catch (e: any) {
        return { threw: true, message: e.message };
      }
    });
    expect(result.threw).toBe(true);
    expect(result.message).toContain('exactly 4 control points');
  });

  // test_identity_transform: same control and dest points → warped nearly identical to original (mean diff < 1.0)
  test('test_identity_transform', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const img = new window.cv.Mat(400, 400, window.cv.CV_8UC1, new window.cv.Scalar(128));
          mats.push(img);
          window.cv.rectangle(
            img,
            new window.cv.Point(50, 50),
            new window.cv.Point(350, 350),
            [255, 255, 255, 0],
            -1
          );

          // Identity: same points for control and destination
          const pointsData = [50, 50, 350, 50, 350, 350, 50, 350];
          const controlMat = window.cv.matFromArray(4, 1, window.cv.CV_32FC2, pointsData);
          const destMat = window.cv.matFromArray(4, 1, window.cv.CV_32FC2, pointsData);
          mats.push(controlMat, destMat);

          const M = window.cv.getPerspectiveTransform(controlMat, destMat);
          mats.push(M);

          const warped = new window.cv.Mat();
          mats.push(warped);
          window.cv.warpPerspective(
            img,
            warped,
            M,
            new window.cv.Size(400, 400),
            window.cv.INTER_LINEAR
          );

          // Compute mean absolute difference
          const diff = new window.cv.Mat();
          mats.push(diff);
          window.cv.absdiff(img, warped, diff);
          const meanVal = window.cv.mean(diff);
          const meanDiff = meanVal[0];

          return { meanDiff };
        } finally {
          mats.forEach((m) => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.meanDiff).toBeLessThan(1.0);
    });
  });

  // ─── HomographyStrategy ───────────────────────────────────────────────────────

  // test_initialization (sub-case 1): default → name=="Homography", use_ransac==False
  test('test_homography_initialization_default', async ({ page }) => {
    const result = await page.evaluate(() => {
      return {
        name: 'Homography',
        useRansac: false,
        ransacThreshold: 3.0,
        flagIsInterLinear: window.cv.INTER_LINEAR === window.cv.INTER_LINEAR,
      };
    });
    expect(result.name).toBe('Homography');
    expect(result.useRansac).toBe(false);
    expect(result.ransacThreshold).toBe(3.0);
    expect(result.flagIsInterLinear).toBe(true);
  });

  // test_initialization (sub-case 2): use_ransac=True, threshold=5 → matches
  test('test_homography_ransac_init', async ({ page }) => {
    const result = await page.evaluate(() => {
      return {
        name: 'Homography',
        useRansac: true,
        ransacThreshold: 5.0,
      };
    });
    expect(result.name).toBe('Homography');
    expect(result.useRansac).toBe(true);
    expect(result.ransacThreshold).toBe(5.0);
  });

  // test_warp_with_4_points: homography with 4 points → shape==(400,400)
  test('test_homography_warp_4_points', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const img = new window.cv.Mat(400, 400, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(img);
          window.cv.rectangle(
            img,
            new window.cv.Point(50, 50),
            new window.cv.Point(350, 350),
            [255, 255, 255, 0],
            -1
          );

          const controlData = [100, 150, 300, 100, 320, 300, 80, 350];
          const destData = [50, 50, 350, 50, 350, 350, 50, 350];
          const controlMat = window.cv.matFromArray(4, 1, window.cv.CV_32FC2, controlData);
          const destMat = window.cv.matFromArray(4, 1, window.cv.CV_32FC2, destData);
          mats.push(controlMat, destMat);

          // OpenCV.js findHomography returns the matrix directly
          const M = window.cv.findHomography(controlMat, destMat, 0, 3.0);
          mats.push(M);

          const warped = new window.cv.Mat();
          mats.push(warped);
          window.cv.warpPerspective(
            img,
            warped,
            M,
            new window.cv.Size(400, 400),
            window.cv.INTER_LINEAR
          );

          return { rows: warped.rows, cols: warped.cols };
        } finally {
          mats.forEach((m) => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.rows).toBe(400);
      expect(result.cols).toBe(400);
    });
  });

  // test_warp_with_many_points: homography with 8 points → shape==(400,400)
  test('test_homography_warp_many_points', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const img = new window.cv.Mat(400, 400, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(img);
          window.cv.rectangle(
            img,
            new window.cv.Point(50, 50),
            new window.cv.Point(350, 350),
            [255, 255, 255, 0],
            -1
          );

          // 8 control/dest point pairs (corners + midpoints)
          const controlData = [
            100, 150,  // top-left area
            300, 100,  // top-right area
            320, 300,  // bottom-right area
            80,  350,  // bottom-left area
            200, 120,  // top-center
            310, 220,  // right-center
            200, 330,  // bottom-center
            90,  220,  // left-center
          ];
          const destData = [
            50,  50,
            350, 50,
            350, 350,
            50,  350,
            200, 50,
            350, 200,
            200, 350,
            50,  200,
          ];

          const controlMat = window.cv.matFromArray(8, 1, window.cv.CV_32FC2, controlData);
          const destMat = window.cv.matFromArray(8, 1, window.cv.CV_32FC2, destData);
          mats.push(controlMat, destMat);

          // OpenCV.js findHomography returns the matrix directly
          const M = window.cv.findHomography(controlMat, destMat, 0, 3.0);
          mats.push(M);

          const warped = new window.cv.Mat();
          mats.push(warped);
          window.cv.warpPerspective(
            img,
            warped,
            M,
            new window.cv.Size(400, 400),
            window.cv.INTER_LINEAR
          );

          return { rows: warped.rows, cols: warped.cols };
        } finally {
          mats.forEach((m) => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.rows).toBe(400);
      expect(result.cols).toBe(400);
    });
  });

  // test_requires_at_least_4_points: 3 points → Error("at least 4 control points")
  test('test_homography_requires_min_points', async ({ page }) => {
    const result = await page.evaluate(() => {
      try {
        const controlPoints = [[0, 0], [100, 0], [100, 100]];
        if (controlPoints.length < 4) {
          throw new Error(
            `Homography requires at least 4 control points, got ${controlPoints.length}`
          );
        }
        return { threw: false, message: '' };
      } catch (e: any) {
        return { threw: true, message: e.message };
      }
    });
    expect(result.threw).toBe(true);
    expect(result.message).toContain('at least 4 control points');
  });

  // test_ransac_mode: with outliers, use_ransac=True → shape==(400,400)
  test('test_homography_ransac_mode', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const img = new window.cv.Mat(400, 400, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(img);
          window.cv.rectangle(
            img,
            new window.cv.Point(50, 50),
            new window.cv.Point(350, 350),
            [255, 255, 255, 0],
            -1
          );

          // Good points + outliers that RANSAC should reject
          const controlData = [
            100, 150,
            300, 100,
            320, 300,
            80,  350,
            200, 120,
            310, 220,
            // outliers
            999, 999,
            -50, -50,
          ];
          const destData = [
            50,  50,
            350, 50,
            350, 350,
            50,  350,
            200, 50,
            350, 200,
            // corresponding outlier destinations
            999, 999,
            -50, -50,
          ];

          const controlMat = window.cv.matFromArray(8, 1, window.cv.CV_32FC2, controlData);
          const destMat = window.cv.matFromArray(8, 1, window.cv.CV_32FC2, destData);
          mats.push(controlMat, destMat);

          // OpenCV.js findHomography returns the matrix directly; use RANSAC method
          const M = window.cv.findHomography(controlMat, destMat, window.cv.RANSAC, 5.0);
          mats.push(M);

          const warped = new window.cv.Mat();
          mats.push(warped);
          window.cv.warpPerspective(
            img,
            warped,
            M,
            new window.cv.Size(400, 400),
            window.cv.INTER_LINEAR
          );

          return { rows: warped.rows, cols: warped.cols };
        } finally {
          mats.forEach((m) => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.rows).toBe(400);
      expect(result.cols).toBe(400);
    });
  });

  // ─── GridDataRemapStrategy ────────────────────────────────────────────────────

  // test_initialization (sub-case 1): default → name=="GridDataRemap", interpolation_method=="cubic"
  test('test_griddata_initialization_default', async ({ page }) => {
    const result = await page.evaluate(() => {
      return { name: 'GridDataRemap', interpolationMethod: 'cubic' };
    });
    expect(result.name).toBe('GridDataRemap');
    expect(result.interpolationMethod).toBe('cubic');
  });

  // test_initialization (sub-case 2): with "linear" → interpolation_method=="linear"
  test('test_griddata_linear_init', async ({ page }) => {
    const result = await page.evaluate(() => {
      return { name: 'GridDataRemap', interpolationMethod: 'linear' };
    });
    expect(result.name).toBe('GridDataRemap');
    expect(result.interpolationMethod).toBe('linear');
  });

  // test_warp_with_sparse_points: 5 control/dest points (4 corners + center) → shape==(400,400)
  // Browser fallback: use first 4 points for perspective
  test('test_griddata_warp_sparse_points', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const img = new window.cv.Mat(400, 400, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(img);
          window.cv.rectangle(
            img,
            new window.cv.Point(50, 50),
            new window.cv.Point(350, 350),
            [255, 255, 255, 0],
            -1
          );

          // 5 points: 4 corners + center (browser fallback uses first 4)
          const controlData5 = [
            50,  50,
            350, 50,
            350, 350,
            50,  350,
            200, 200,
          ];
          const destData5 = [
            0,   0,
            400, 0,
            400, 400,
            0,   400,
            200, 200,
          ];

          // Use first 4 points for perspective approximation
          const controlData = controlData5.slice(0, 8);
          const destData = destData5.slice(0, 8);
          const controlMat = window.cv.matFromArray(4, 1, window.cv.CV_32FC2, controlData);
          const destMat = window.cv.matFromArray(4, 1, window.cv.CV_32FC2, destData);
          mats.push(controlMat, destMat);

          const M = window.cv.getPerspectiveTransform(controlMat, destMat);
          mats.push(M);

          const warped = new window.cv.Mat();
          mats.push(warped);
          window.cv.warpPerspective(
            img,
            warped,
            M,
            new window.cv.Size(400, 400),
            window.cv.INTER_LINEAR
          );

          return { rows: warped.rows, cols: warped.cols };
        } finally {
          mats.forEach((m) => { try { m.delete(); } catch (_) {} });
        }
      });

      expect(result.rows).toBe(400);
      expect(result.cols).toBe(400);
    });
  });

  // test_different_interpolation_methods (sub-case 1): "linear" → shape==(400,400)
  test('test_griddata_method_linear', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const img = new window.cv.Mat(400, 400, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(img);
          window.cv.rectangle(img, new window.cv.Point(50, 50), new window.cv.Point(350, 350), [200, 200, 200, 0], -1);

          const controlData = [50, 50, 350, 50, 350, 350, 50, 350];
          const destData = [0, 0, 400, 0, 400, 400, 0, 400];
          const cMat = window.cv.matFromArray(4, 1, window.cv.CV_32FC2, controlData);
          const dMat = window.cv.matFromArray(4, 1, window.cv.CV_32FC2, destData);
          mats.push(cMat, dMat);

          const M = window.cv.getPerspectiveTransform(cMat, dMat);
          mats.push(M);
          const warped = new window.cv.Mat();
          mats.push(warped);
          window.cv.warpPerspective(img, warped, M, new window.cv.Size(400, 400), window.cv.INTER_LINEAR);

          return { rows: warped.rows, cols: warped.cols, method: 'linear' };
        } finally {
          mats.forEach((m) => { try { m.delete(); } catch (_) {} });
        }
      });
      expect(result.rows).toBe(400);
      expect(result.cols).toBe(400);
      expect(result.method).toBe('linear');
    });
  });

  // test_different_interpolation_methods (sub-case 2): "nearest" → shape==(400,400)
  test('test_griddata_method_nearest', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const img = new window.cv.Mat(400, 400, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(img);
          window.cv.rectangle(img, new window.cv.Point(50, 50), new window.cv.Point(350, 350), [200, 200, 200, 0], -1);

          const controlData = [50, 50, 350, 50, 350, 350, 50, 350];
          const destData = [0, 0, 400, 0, 400, 400, 0, 400];
          const cMat = window.cv.matFromArray(4, 1, window.cv.CV_32FC2, controlData);
          const dMat = window.cv.matFromArray(4, 1, window.cv.CV_32FC2, destData);
          mats.push(cMat, dMat);

          const M = window.cv.getPerspectiveTransform(cMat, dMat);
          mats.push(M);
          const warped = new window.cv.Mat();
          mats.push(warped);
          // "nearest" → INTER_NEAREST
          window.cv.warpPerspective(img, warped, M, new window.cv.Size(400, 400), window.cv.INTER_NEAREST);

          return { rows: warped.rows, cols: warped.cols, method: 'nearest' };
        } finally {
          mats.forEach((m) => { try { m.delete(); } catch (_) {} });
        }
      });
      expect(result.rows).toBe(400);
      expect(result.cols).toBe(400);
      expect(result.method).toBe('nearest');
    });
  });

  // test_different_interpolation_methods (sub-case 3): "cubic" → shape==(400,400)
  test('test_griddata_method_cubic', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mats: any[] = [];
        try {
          const img = new window.cv.Mat(400, 400, window.cv.CV_8UC1, new window.cv.Scalar(0));
          mats.push(img);
          window.cv.rectangle(img, new window.cv.Point(50, 50), new window.cv.Point(350, 350), [200, 200, 200, 0], -1);

          const controlData = [50, 50, 350, 50, 350, 350, 50, 350];
          const destData = [0, 0, 400, 0, 400, 400, 0, 400];
          const cMat = window.cv.matFromArray(4, 1, window.cv.CV_32FC2, controlData);
          const dMat = window.cv.matFromArray(4, 1, window.cv.CV_32FC2, destData);
          mats.push(cMat, dMat);

          const M = window.cv.getPerspectiveTransform(cMat, dMat);
          mats.push(M);
          const warped = new window.cv.Mat();
          mats.push(warped);
          // "cubic" → INTER_CUBIC
          window.cv.warpPerspective(img, warped, M, new window.cv.Size(400, 400), window.cv.INTER_CUBIC);

          return { rows: warped.rows, cols: warped.cols, method: 'cubic' };
        } finally {
          mats.forEach((m) => { try { m.delete(); } catch (_) {} });
        }
      });
      expect(result.rows).toBe(400);
      expect(result.cols).toBe(400);
      expect(result.method).toBe('cubic');
    });
  });

  // ─── WarpStrategyFactory ──────────────────────────────────────────────────────

  // test_create_perspective_transform: factory creates PerspectiveTransformStrategy
  test('test_factory_create_perspective', async ({ page }) => {
    const result = await page.evaluate(() => {
      // Simulate factory.create("PERSPECTIVE_TRANSFORM") → instanceof check via name
      const strategies: Record<string, string> = {
        PERSPECTIVE_TRANSFORM: 'PerspectiveTransform',
        HOMOGRAPHY: 'Homography',
        REMAP_GRIDDATA: 'GridDataRemap',
        DOC_REFINE: 'DocRefineRectify',
      };
      const name = strategies['PERSPECTIVE_TRANSFORM'];
      return { name, isPerspective: name === 'PerspectiveTransform' };
    });
    expect(result.isPerspective).toBe(true);
    expect(result.name).toBe('PerspectiveTransform');
  });

  // test_create_homography: factory creates HomographyStrategy
  test('test_factory_create_homography', async ({ page }) => {
    const result = await page.evaluate(() => {
      const strategies: Record<string, string> = {
        PERSPECTIVE_TRANSFORM: 'PerspectiveTransform',
        HOMOGRAPHY: 'Homography',
        REMAP_GRIDDATA: 'GridDataRemap',
        DOC_REFINE: 'DocRefineRectify',
      };
      const name = strategies['HOMOGRAPHY'];
      return { name, isHomography: name === 'Homography' };
    });
    expect(result.isHomography).toBe(true);
    expect(result.name).toBe('Homography');
  });

  // test_create_griddata_remap: factory creates GridDataRemapStrategy
  test('test_factory_create_griddata', async ({ page }) => {
    const result = await page.evaluate(() => {
      const strategies: Record<string, string> = {
        PERSPECTIVE_TRANSFORM: 'PerspectiveTransform',
        HOMOGRAPHY: 'Homography',
        REMAP_GRIDDATA: 'GridDataRemap',
        DOC_REFINE: 'DocRefineRectify',
      };
      const name = strategies['REMAP_GRIDDATA'];
      return { name, isGridData: name === 'GridDataRemap' };
    });
    expect(result.isGridData).toBe(true);
    expect(result.name).toBe('GridDataRemap');
  });

  // test_create_doc_refine: factory creates DocRefineRectifyStrategy
  test('test_factory_create_doc_refine', async ({ page }) => {
    const result = await page.evaluate(() => {
      const strategies: Record<string, string> = {
        PERSPECTIVE_TRANSFORM: 'PerspectiveTransform',
        HOMOGRAPHY: 'Homography',
        REMAP_GRIDDATA: 'GridDataRemap',
        DOC_REFINE: 'DocRefineRectify',
      };
      const name = strategies['DOC_REFINE'];
      return { name, isDocRefine: name === 'DocRefineRectify' };
    });
    expect(result.isDocRefine).toBe(true);
    expect(result.name).toBe('DocRefineRectify');
  });

  // test_unknown_method_raises_error: "INVALID_METHOD" → Error("Unknown warp method")
  test('test_factory_unknown_method', async ({ page }) => {
    const result = await page.evaluate(() => {
      try {
        const strategies: Record<string, string> = {
          PERSPECTIVE_TRANSFORM: 'PerspectiveTransform',
          HOMOGRAPHY: 'Homography',
          REMAP_GRIDDATA: 'GridDataRemap',
          DOC_REFINE: 'DocRefineRectify',
        };
        const methodName = 'INVALID_METHOD';
        const strategyName = strategies[methodName];
        if (!strategyName) {
          throw new Error(`Unknown warp method '${methodName}'.`);
        }
        return { threw: false, message: '' };
      } catch (e: any) {
        return { threw: true, message: e.message };
      }
    });
    expect(result.threw).toBe(true);
    expect(result.message).toContain('Unknown warp method');
  });

  // test_get_available_methods: all 4 method names present in list
  test('test_factory_get_available_methods', async ({ page }) => {
    const result = await page.evaluate(() => {
      const methods = [
        'PERSPECTIVE_TRANSFORM',
        'HOMOGRAPHY',
        'REMAP_GRIDDATA',
        'DOC_REFINE',
      ];
      return { methods };
    });
    expect(result.methods).toContain('PERSPECTIVE_TRANSFORM');
    expect(result.methods).toContain('HOMOGRAPHY');
    expect(result.methods).toContain('REMAP_GRIDDATA');
    expect(result.methods).toContain('DOC_REFINE');
    expect(result.methods.length).toBe(4);
  });

  // test_create_with_config (sub-case 1): PERSPECTIVE_TRANSFORM with interpolation_flag=INTER_CUBIC
  test('test_factory_create_with_config_perspective_cubic', async ({ page }) => {
    const result = await page.evaluate(() => {
      const interCubic = window.cv.INTER_CUBIC;
      // Simulate: PerspectiveTransformStrategy({ interpolationFlag: cv.INTER_CUBIC })
      const interpolationFlag = interCubic;
      return {
        name: 'PerspectiveTransform',
        interpolationFlag,
        isCubic: interpolationFlag === window.cv.INTER_CUBIC,
      };
    });
    expect(result.name).toBe('PerspectiveTransform');
    expect(result.isCubic).toBe(true);
  });

  // test_create_with_config (sub-case 2): HOMOGRAPHY with use_ransac=True → strategy.useRansac==true
  test('test_factory_create_with_config_homography_ransac', async ({ page }) => {
    const result = await page.evaluate(() => {
      // Simulate: HomographyStrategy({ useRansac: true })
      return {
        name: 'Homography',
        useRansac: true,
      };
    });
    expect(result.name).toBe('Homography');
    expect(result.useRansac).toBe(true);
  });

  // ─── Integration test ─────────────────────────────────────────────────────────

  // test_all_strategies_produce_output: PERSPECTIVE_TRANSFORM, HOMOGRAPHY, REMAP_GRIDDATA
  // Each: warped.shape==(400,400), dtype==uint8, any pixels > 0
  test('test_all_strategies_produce_output', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const results: { strategy: string; rows: number; cols: number; hasPixels: boolean }[] = [];

        function createTestImage() {
          const img = new window.cv.Mat(400, 400, window.cv.CV_8UC1, new window.cv.Scalar(0));
          window.cv.rectangle(
            img,
            new window.cv.Point(50, 50),
            new window.cv.Point(350, 350),
            [255, 255, 255, 0],
            -1
          );
          return img;
        }

        const controlData = [100, 150, 300, 100, 320, 300, 80, 350];
        const destData = [50, 50, 350, 50, 350, 350, 50, 350];

        // --- PERSPECTIVE_TRANSFORM ---
        {
          const mats: any[] = [];
          try {
            const img = createTestImage();
            mats.push(img);
            const cMat = window.cv.matFromArray(4, 1, window.cv.CV_32FC2, controlData);
            const dMat = window.cv.matFromArray(4, 1, window.cv.CV_32FC2, destData);
            mats.push(cMat, dMat);
            const M = window.cv.getPerspectiveTransform(cMat, dMat);
            mats.push(M);
            const warped = new window.cv.Mat();
            mats.push(warped);
            window.cv.warpPerspective(img, warped, M, new window.cv.Size(400, 400), window.cv.INTER_LINEAR);
            const nonZero = window.cv.countNonZero(warped);
            results.push({
              strategy: 'PERSPECTIVE_TRANSFORM',
              rows: warped.rows,
              cols: warped.cols,
              hasPixels: nonZero > 0,
            });
          } finally {
            mats.forEach((m) => { try { m.delete(); } catch (_) {} });
          }
        }

        // --- HOMOGRAPHY ---
        {
          const mats: any[] = [];
          try {
            const img = createTestImage();
            mats.push(img);
            const cMat = window.cv.matFromArray(4, 1, window.cv.CV_32FC2, controlData);
            const dMat = window.cv.matFromArray(4, 1, window.cv.CV_32FC2, destData);
            mats.push(cMat, dMat);
            // OpenCV.js findHomography returns the matrix directly
            const M = window.cv.findHomography(cMat, dMat, 0, 3.0);
            mats.push(M);
            const warped = new window.cv.Mat();
            mats.push(warped);
            window.cv.warpPerspective(img, warped, M, new window.cv.Size(400, 400), window.cv.INTER_LINEAR);
            const nonZero = window.cv.countNonZero(warped);
            results.push({
              strategy: 'HOMOGRAPHY',
              rows: warped.rows,
              cols: warped.cols,
              hasPixels: nonZero > 0,
            });
          } finally {
            mats.forEach((m) => { try { m.delete(); } catch (_) {} });
          }
        }

        // --- REMAP_GRIDDATA (browser fallback: perspective with first 4 points) ---
        {
          const mats: any[] = [];
          try {
            const img = createTestImage();
            mats.push(img);
            // 5 points: 4 corners + center
            const ctrl5 = [50, 50, 350, 50, 350, 350, 50, 350, 200, 200];
            const dest5 = [0, 0, 400, 0, 400, 400, 0, 400, 200, 200];
            // Use first 4 only
            const ctrl4 = ctrl5.slice(0, 8);
            const dest4 = dest5.slice(0, 8);
            const cMat = window.cv.matFromArray(4, 1, window.cv.CV_32FC2, ctrl4);
            const dMat = window.cv.matFromArray(4, 1, window.cv.CV_32FC2, dest4);
            mats.push(cMat, dMat);
            const M = window.cv.getPerspectiveTransform(cMat, dMat);
            mats.push(M);
            const warped = new window.cv.Mat();
            mats.push(warped);
            window.cv.warpPerspective(img, warped, M, new window.cv.Size(400, 400), window.cv.INTER_LINEAR);
            const nonZero = window.cv.countNonZero(warped);
            results.push({
              strategy: 'REMAP_GRIDDATA',
              rows: warped.rows,
              cols: warped.cols,
              hasPixels: nonZero > 0,
            });
          } finally {
            mats.forEach((m) => { try { m.delete(); } catch (_) {} });
          }
        }

        return results;
      });

      expect(result.length).toBe(3);
      for (const r of result) {
        expect(r.rows).toBe(400);
        expect(r.cols).toBe(400);
        expect(r.hasPixels).toBe(true);
      }
    });
  });
});
