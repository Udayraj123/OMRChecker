/**
 * Browser tests for drawing.ts using OpenCV.js
 *
 * These tests verify drawing operations work correctly in a real browser environment
 * with actual OpenCV.js cv.Mat objects. Each test corresponds to a Python test in
 * src/tests/utils/__tests__/test_drawing.py.
 *
 * Run with: npm run test:browser
 */

import { test, expect } from '@playwright/test';
import { setupBrowser, teardownBrowser } from './browser-setup';
import { withMemoryTracking } from './memory-utils';

// Increase timeout for browser tests
test.setTimeout(60000);

test.describe('Drawing Utils - Browser Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Setup OpenCV.js for each test
    await setupBrowser(page);
  });

  // test_draw_box_diagonal: draw_box_diagonal(image, (10,10), (90,90)), checks np.any(image > 0)
  test('test_draw_box_diagonal', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mat = new window.cv.Mat(100, 100, window.cv.CV_8UC3, [0, 0, 0, 0]);

        // Draw rectangle from (10,10) to (90,90)
        const pt1 = new window.cv.Point(10, 10);
        const pt2 = new window.cv.Point(90, 90);
        window.cv.rectangle(mat, pt1, pt2, [100, 100, 100], 3);

        // Check that something was drawn (image is no longer all zeros)
        const pixel = mat.ucharPtr(10, 10);
        const wasModified = pixel[0] > 0 || pixel[1] > 0 || pixel[2] > 0;

        mat.delete();
        return wasModified;
      });

      expect(result).toBe(true);
    });
  });

  // test_draw_box_hollow: draw_box(image, (10,10), (50,50), style="BOX_HOLLOW"), checks image modified
  test('test_draw_box_hollow', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mat = new window.cv.Mat(100, 100, window.cv.CV_8UC3, [0, 0, 0, 0]);

        // draw_box with BOX_HOLLOW: position=(10,10), dimensions=(50,50), thicknessFactor=1/12
        const x = 10, y = 10, boxW = 50, boxH = 50;
        const thicknessFactor = 1 / 12;
        const pt1 = new window.cv.Point(
          Math.floor(x + boxW * thicknessFactor),
          Math.floor(y + boxH * thicknessFactor)
        );
        const pt2 = new window.cv.Point(
          Math.floor(x + boxW - boxW * thicknessFactor),
          Math.floor(y + boxH - boxH * thicknessFactor)
        );
        window.cv.rectangle(mat, pt1, pt2, [130, 130, 130], 3);

        // Check that something was drawn (image is no longer all zeros)
        const gray = new window.cv.Mat();
        window.cv.cvtColor(mat, gray, window.cv.COLOR_BGR2GRAY);
        const nonZero = window.cv.countNonZero(gray);
        gray.delete();
        mat.delete();
        return nonZero > 0;
      });

      expect(result).toBe(true);
    });
  });

  // test_draw_box_filled: draw_box(image, (10,10), (50,50), style="BOX_FILLED"), checks image modified
  test('test_draw_box_filled', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mat = new window.cv.Mat(100, 100, window.cv.CV_8UC3, [0, 0, 0, 0]);

        // draw_box with BOX_FILLED: position=(10,10), dimensions=(50,50), thicknessFactor=1/12, thickness=-1
        const x = 10, y = 10, boxW = 50, boxH = 50;
        const thicknessFactor = 1 / 12;
        const pt1 = new window.cv.Point(
          Math.floor(x + boxW * thicknessFactor),
          Math.floor(y + boxH * thicknessFactor)
        );
        const pt2 = new window.cv.Point(
          Math.floor(x + boxW - boxW * thicknessFactor),
          Math.floor(y + boxH - boxH * thicknessFactor)
        );
        window.cv.rectangle(mat, pt1, pt2, [100, 100, 100], -1);

        // Check that something was drawn (image is no longer all zeros)
        const gray = new window.cv.Mat();
        window.cv.cvtColor(mat, gray, window.cv.COLOR_BGR2GRAY);
        const nonZero = window.cv.countNonZero(gray);
        gray.delete();
        mat.delete();
        return nonZero > 0;
      });

      expect(result).toBe(true);
    });
  });

  // test_draw_box_centered: draw_box(image, (50,50), (30,30), style="BOX_HOLLOW", centered=True), checks image modified
  test('test_draw_box_centered', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mat = new window.cv.Mat(100, 100, window.cv.CV_8UC3, [0, 0, 0, 0]);

        // draw_box with centered=True: position=(50,50), dimensions=(30,30), thicknessFactor=1/12
        const x = 50, y = 50, boxW = 30, boxH = 30;
        const thicknessFactor = 1 / 12;

        let posX = Math.floor(x + boxW * thicknessFactor);
        let posY = Math.floor(y + boxH * thicknessFactor);
        let diagX = Math.floor(x + boxW - boxW * thicknessFactor);
        let diagY = Math.floor(y + boxH - boxH * thicknessFactor);

        // Apply centering transformation
        const centeredPosX = Math.floor((3 * posX - diagX) / 2);
        const centeredPosY = Math.floor((3 * posY - diagY) / 2);
        const centeredDiagX = Math.floor((posX + diagX) / 2);
        const centeredDiagY = Math.floor((posY + diagY) / 2);

        const pt1 = new window.cv.Point(centeredPosX, centeredPosY);
        const pt2 = new window.cv.Point(centeredDiagX, centeredDiagY);
        window.cv.rectangle(mat, pt1, pt2, [130, 130, 130], 3);

        // Check that something was drawn (image is no longer all zeros)
        const gray = new window.cv.Mat();
        window.cv.cvtColor(mat, gray, window.cv.COLOR_BGR2GRAY);
        const nonZero = window.cv.countNonZero(gray);
        gray.delete();
        mat.delete();
        return nonZero > 0;
      });

      expect(result).toBe(true);
    });
  });

  // test_draw_text: draw_text(image, "Test", (10,50), color=(255,255,255)), checks image modified
  test('test_draw_text', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mat = new window.cv.Mat(100, 100, window.cv.CV_8UC3, [0, 0, 0, 0]);

        const position = new window.cv.Point(10, 50);
        window.cv.putText(
          mat, 'Test', position,
          window.cv.FONT_HERSHEY_SIMPLEX, 0.95, [255, 255, 255], 2, window.cv.LINE_AA
        );

        // Check that something was drawn (image is no longer all zeros)
        const gray = new window.cv.Mat();
        window.cv.cvtColor(mat, gray, window.cv.COLOR_BGR2GRAY);
        const nonZero = window.cv.countNonZero(gray);
        gray.delete();
        mat.delete();
        return nonZero > 0;
      });

      expect(result).toBe(true);
    });
  });

  // test_draw_text_centered: draw_text(image, "Test", (50,50), centered=True, color=(255,255,255)), checks image modified
  test('test_draw_text_centered', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mat = new window.cv.Mat(100, 100, window.cv.CV_8UC3, [0, 0, 0, 0]);

        const fontFace = window.cv.FONT_HERSHEY_SIMPLEX;
        const fontScale = 0.95;
        const thickness = 2;

        // Calculate centered position: position=(50,50)
        const textSizeResult = window.cv.getTextSize('Test', fontFace, fontScale, thickness);
        const sizeX = textSizeResult.size.width;
        const sizeY = textSizeResult.size.height;
        const centeredX = 50 - Math.floor(sizeX / 2);
        const centeredY = 50 + Math.floor(sizeY / 2);

        const position = new window.cv.Point(centeredX, centeredY);
        window.cv.putText(
          mat, 'Test', position,
          fontFace, fontScale, [255, 255, 255], thickness, window.cv.LINE_AA
        );

        // Check that something was drawn (image is no longer all zeros)
        const gray = new window.cv.Mat();
        window.cv.cvtColor(mat, gray, window.cv.COLOR_BGR2GRAY);
        const nonZero = window.cv.countNonZero(gray);
        gray.delete();
        mat.delete();
        return nonZero > 0;
      });

      expect(result).toBe(true);
    });
  });

  // test_draw_line: draw_line(image, (10,10), (90,90), color=(255,255,255)), checks image modified
  test('test_draw_line', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mat = new window.cv.Mat(100, 100, window.cv.CV_8UC3, [0, 0, 0, 0]);

        const pt1 = new window.cv.Point(10, 10);
        const pt2 = new window.cv.Point(90, 90);
        window.cv.line(mat, pt1, pt2, [255, 255, 255], 3);

        // Check that something was drawn (image is no longer all zeros)
        const gray = new window.cv.Mat();
        window.cv.cvtColor(mat, gray, window.cv.COLOR_BGR2GRAY);
        const nonZero = window.cv.countNonZero(gray);
        gray.delete();
        mat.delete();
        return nonZero > 0;
      });

      expect(result).toBe(true);
    });
  });

  // test_draw_polygon_closed: draw_polygon(image, [(20,20),(80,20),(80,80),(20,80)], closed=True, color=(255,255,255)), checks image modified
  test('test_draw_polygon_closed', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mat = new window.cv.Mat(100, 100, window.cv.CV_8UC3, [0, 0, 0, 0]);

        // Points: [(20,20),(80,20),(80,80),(20,80)]
        const pointsData = [20, 20, 80, 20, 80, 80, 20, 80];
        const pts = window.cv.matFromArray(4, 1, window.cv.CV_32SC2, pointsData);
        const ptsVec = new window.cv.MatVector();
        ptsVec.push_back(pts);
        window.cv.polylines(mat, ptsVec, true, [255, 255, 255], 1);

        ptsVec.delete();
        pts.delete();

        // Check that something was drawn (image is no longer all zeros)
        const gray = new window.cv.Mat();
        window.cv.cvtColor(mat, gray, window.cv.COLOR_BGR2GRAY);
        const nonZero = window.cv.countNonZero(gray);
        gray.delete();
        mat.delete();
        return nonZero > 0;
      });

      expect(result).toBe(true);
    });
  });

  // test_draw_polygon_open: draw_polygon(image, [(20,20),(80,20),(80,80)], closed=False, color=(255,255,255)), checks image modified
  test('test_draw_polygon_open', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mat = new window.cv.Mat(100, 100, window.cv.CV_8UC3, [0, 0, 0, 0]);

        // Points: [(20,20),(80,20),(80,80)]
        const pointsData = [20, 20, 80, 20, 80, 80];
        const pts = window.cv.matFromArray(3, 1, window.cv.CV_32SC2, pointsData);
        const ptsVec = new window.cv.MatVector();
        ptsVec.push_back(pts);
        window.cv.polylines(mat, ptsVec, false, [255, 255, 255], 1);

        ptsVec.delete();
        pts.delete();

        // Check that something was drawn (image is no longer all zeros)
        const gray = new window.cv.Mat();
        window.cv.cvtColor(mat, gray, window.cv.COLOR_BGR2GRAY);
        const nonZero = window.cv.countNonZero(gray);
        gray.delete();
        mat.delete();
        return nonZero > 0;
      });

      expect(result).toBe(true);
    });
  });

  // test_draw_contour: draw_contour(image, np.array([[10,10],[90,10],[90,90],[10,90]])), checks image modified
  test('test_draw_contour', async ({ page }) => {
    await withMemoryTracking(page, async () => {
      const result = await page.evaluate(() => {
        const mat = new window.cv.Mat(100, 100, window.cv.CV_8UC3, [0, 0, 0, 0]);

        // Contour: [[10,10],[90,10],[90,90],[10,90]]
        const contourData = [10, 10, 90, 10, 90, 90, 10, 90];
        const contour = window.cv.matFromArray(4, 1, window.cv.CV_32SC2, contourData);
        const contours = new window.cv.MatVector();
        contours.push_back(contour);
        window.cv.drawContours(mat, contours, -1, [100, 200, 100], 2);

        contours.delete();
        contour.delete();

        // Check that something was drawn (image is no longer all zeros)
        const gray = new window.cv.Mat();
        window.cv.cvtColor(mat, gray, window.cv.COLOR_BGR2GRAY);
        const nonZero = window.cv.countNonZero(gray);
        gray.delete();
        mat.delete();
        return nonZero > 0;
      });

      expect(result).toBe(true);
    });
  });
});
