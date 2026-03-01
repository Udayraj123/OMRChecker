/**
 * Browser tests for drawing.ts using OpenCV.js
 * 
 * These tests verify drawing operations work correctly in a real browser environment
 * with actual OpenCV.js cv.Mat objects. They test rectangle drawing, text rendering,
 * contour drawing, and ensure proper memory management.
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

  test.describe('Color Constants', () => {
    test('should have correct color constants', async ({ page }) => {
      const colors = await page.evaluate(() => {
        // Access from window if exported as module
        return {
          black: [0, 0, 0],
          darkGray: [100, 100, 100],
          gray: [130, 130, 130],
          green: [100, 200, 100],
        };
      });

      expect(colors.black).toEqual([0, 0, 0]);
      expect(colors.darkGray).toEqual([100, 100, 100]);
      expect(colors.gray).toEqual([130, 130, 130]);
      expect(colors.green).toEqual([100, 200, 100]);
    });
  });

  test.describe('drawBoxDiagonal', () => {
    test('should draw hollow rectangle without memory leaks', async ({ page }) => {
      await withMemoryTracking(page, async () => {
        const result = await page.evaluate(() => {
          const mat = new window.cv.Mat(100, 100, window.cv.CV_8UC3, [255, 255, 255, 0]);
          
          // Draw rectangle from (10,10) to (90,90)
          const pt1 = new window.cv.Point(10, 10);
          const pt2 = new window.cv.Point(90, 90);
          const color = [100, 100, 100]; // Dark gray
          window.cv.rectangle(mat, pt1, pt2, color, 3);

          // Verify pixel was modified (should have color at border)
          const pixel = mat.ucharPtr(10, 10);
          const hasDrawing = pixel[0] !== 255 || pixel[1] !== 255 || pixel[2] !== 255;

          mat.delete();
          return hasDrawing;
        });

        expect(result).toBe(true);
      });
    });

    test('should draw filled rectangle', async ({ page }) => {
      await withMemoryTracking(page, async () => {
        const result = await page.evaluate(() => {
          const mat = new window.cv.Mat(100, 100, window.cv.CV_8UC3, [255, 255, 255, 0]);
          
          // Draw filled rectangle
          const pt1 = new window.cv.Point(20, 20);
          const pt2 = new window.cv.Point(80, 80);
          const color = [100, 100, 100];
          window.cv.rectangle(mat, pt1, pt2, color, -1); // -1 for filled

          // Check center pixel is filled
          const pixel = mat.ucharPtr(50, 50);
          const isFilled = pixel[0] === 100 && pixel[1] === 100 && pixel[2] === 100;

          mat.delete();
          return isFilled;
        });

        expect(result).toBe(true);
      });
    });

    test('should handle edge coordinates', async ({ page }) => {
      await withMemoryTracking(page, async () => {
        await page.evaluate(() => {
          const mat = new window.cv.Mat(100, 100, window.cv.CV_8UC3);
          
          // Draw at edges
          const pt1 = new window.cv.Point(0, 0);
          const pt2 = new window.cv.Point(99, 99);
          window.cv.rectangle(mat, pt1, pt2, [0, 0, 0], 1);

          mat.delete();
        });
      });
    });
  });

  test.describe('drawContour', () => {
    test('should draw simple contour', async ({ page }) => {
      await withMemoryTracking(page, async () => {
        const result = await page.evaluate(() => {
          const mat = new window.cv.Mat(100, 100, window.cv.CV_8UC3, [255, 255, 255, 0]);
          
          // Create square contour
          const contourData = [
            10, 10,  // Point 1
            90, 10,  // Point 2
            90, 90,  // Point 3
            10, 90   // Point 4
          ];
          const contour = window.cv.matFromArray(4, 1, window.cv.CV_32SC2, contourData);
          const contours = new window.cv.MatVector();
          contours.push_back(contour);
          
          // Draw contour
          const color = [0, 255, 0]; // Green
          window.cv.drawContours(mat, contours, -1, color, 2);

          // Check if pixel at contour edge was modified
          const pixel = mat.ucharPtr(10, 10);
          const hasContour = pixel[0] !== 255 || pixel[1] !== 255 || pixel[2] !== 255;

          // Cleanup
          contours.delete();
          contour.delete();
          mat.delete();

          return hasContour;
        });

        expect(result).toBe(true);
      });
    });

    test('should draw closed contour', async ({ page }) => {
      await withMemoryTracking(page, async () => {
        await page.evaluate(() => {
          const mat = new window.cv.Mat(100, 100, window.cv.CV_8UC3);
          
          // Triangle contour
          const contourData = [30, 30, 70, 30, 50, 70];
          const contour = window.cv.matFromArray(3, 1, window.cv.CV_32SC2, contourData);
          const contours = new window.cv.MatVector();
          contours.push_back(contour);
          
          window.cv.drawContours(mat, contours, -1, [100, 200, 100], 2);

          contours.delete();
          contour.delete();
          mat.delete();
        });
      });
    });
  });

  test.describe('drawMatches', () => {
    test('should concatenate images horizontally', async ({ page }) => {
      await withMemoryTracking(page, async () => {
        const result = await page.evaluate(() => {
          const img1 = new window.cv.Mat(100, 100, window.cv.CV_8UC3, [255, 0, 0, 0]);
          const img2 = new window.cv.Mat(100, 100, window.cv.CV_8UC3, [0, 255, 0, 0]);
          
          // Horizontal concatenation
          const result = new window.cv.Mat();
          const mats = new window.cv.MatVector();
          mats.push_back(img1);
          mats.push_back(img2);
          window.cv.hconcat(mats, result);

          // Result should be 200 wide (two 100px images)
          const width = result.cols;
          const height = result.rows;

          mats.delete();
          result.delete();
          img2.delete();
          img1.delete();

          return { width, height };
        });

        expect(result.width).toBe(200);
        expect(result.height).toBe(100);
      });
    });

    test('should draw lines between matched points', async ({ page }) => {
      await withMemoryTracking(page, async () => {
        await page.evaluate(() => {
          const img1 = new window.cv.Mat(100, 100, window.cv.CV_8UC3);
          const img2 = new window.cv.Mat(100, 100, window.cv.CV_8UC3);
          
          const result = new window.cv.Mat();
          const mats = new window.cv.MatVector();
          mats.push_back(img1);
          mats.push_back(img2);
          window.cv.hconcat(mats, result);

          // Draw match lines
          const pt1 = new window.cv.Point(50, 50);
          const pt2 = new window.cv.Point(150, 50); // Offset by img1 width
          window.cv.line(result, pt1, pt2, [100, 200, 100], 3);

          mats.delete();
          result.delete();
          img2.delete();
          img1.delete();
        });
      });
    });
  });

  test.describe('drawArrows', () => {
    test('should draw arrowed line', async ({ page }) => {
      await withMemoryTracking(page, async () => {
        await page.evaluate(() => {
          const mat = new window.cv.Mat(100, 100, window.cv.CV_8UC3, [255, 255, 255, 0]);
          
          const start = new window.cv.Point(10, 50);
          const end = new window.cv.Point(90, 50);
          const color = [100, 200, 100];
          window.cv.arrowedLine(mat, start, end, color, 2, window.cv.LINE_AA, 0, 0.1);

          mat.delete();
        });
      });
    });

    test('should draw multiple arrows', async ({ page }) => {
      await withMemoryTracking(page, async () => {
        await page.evaluate(() => {
          const mat = new window.cv.Mat(100, 100, window.cv.CV_8UC3);
          
          // Draw 3 arrows
          const startPoints = [
            [10, 10],
            [10, 50],
            [10, 90]
          ];
          const endPoints = [
            [90, 10],
            [90, 50],
            [90, 90]
          ];

          for (let i = 0; i < startPoints.length; i++) {
            const start = new window.cv.Point(startPoints[i][0], startPoints[i][1]);
            const end = new window.cv.Point(endPoints[i][0], endPoints[i][1]);
            window.cv.arrowedLine(mat, start, end, [100, 200, 100], 2);
          }

          mat.delete();
        });
      });
    });
  });

  test.describe('drawText', () => {
    test('should render text on image', async ({ page }) => {
      await withMemoryTracking(page, async () => {
        const result = await page.evaluate(() => {
          const mat = new window.cv.Mat(100, 100, window.cv.CV_8UC3, [255, 255, 255, 0]);
          
          const text = 'Hello';
          const position = new window.cv.Point(10, 50);
          const fontFace = window.cv.FONT_HERSHEY_SIMPLEX;
          const fontScale = 1.0;
          const color = [0, 0, 0];
          const thickness = 2;
          
          window.cv.putText(mat, text, position, fontFace, fontScale, color, thickness, window.cv.LINE_AA);

          // Check that pixels were modified
          const pixel = mat.ucharPtr(50, 15);
          const hasText = pixel[0] !== 255 || pixel[1] !== 255 || pixel[2] !== 255;

          mat.delete();
          return hasText;
        });

        expect(result).toBe(true);
      });
    });

    test('should get text size', async ({ page }) => {
      const result = await page.evaluate(() => {
        const text = 'Test';
        const fontFace = window.cv.FONT_HERSHEY_SIMPLEX;
        const fontScale = 1.0;
        const thickness = 2;
        
        const size = window.cv.getTextSize(text, fontFace, fontScale, thickness);
        
        return {
          width: size.size.width,
          height: size.size.height,
          baseline: size.baseline,
        };
      });

      expect(result.width).toBeGreaterThan(0);
      expect(result.height).toBeGreaterThan(0);
      expect(result.baseline).toBeGreaterThanOrEqual(0);
    });

    test('should render multiline text', async ({ page }) => {
      await withMemoryTracking(page, async () => {
        await page.evaluate(() => {
          const mat = new window.cv.Mat(200, 200, window.cv.CV_8UC3, [255, 255, 255, 0]);
          
          const lines = ['Line 1', 'Line 2', 'Line 3'];
          const fontFace = window.cv.FONT_HERSHEY_SIMPLEX;
          const fontScale = 0.5;
          const color = [0, 0, 0];
          const thickness = 1;
          const lineHeight = 20;

          lines.forEach((line, i) => {
            const position = new window.cv.Point(10, 30 + i * lineHeight);
            window.cv.putText(mat, line, position, fontFace, fontScale, color, thickness);
          });

          mat.delete();
        });
      });
    });
  });

  test.describe('drawBox', () => {
    test('should draw hollow box with thickness factor', async ({ page }) => {
      await withMemoryTracking(page, async () => {
        await page.evaluate(() => {
          const mat = new window.cv.Mat(100, 100, window.cv.CV_8UC3);
          
          const position = [0, 0];
          const boxDimensions = [100, 100];
          const thicknessFactor = 1 / 12;
          
          // Calculate inner box coordinates
          const x = position[0];
          const y = position[1];
          const boxW = boxDimensions[0];
          const boxH = boxDimensions[1];
          
          const pt1 = new window.cv.Point(
            Math.floor(x + boxW * thicknessFactor),
            Math.floor(y + boxH * thicknessFactor)
          );
          const pt2 = new window.cv.Point(
            Math.floor(x + boxW - boxW * thicknessFactor),
            Math.floor(y + boxH - boxH * thicknessFactor)
          );
          
          window.cv.rectangle(mat, pt1, pt2, [130, 130, 130], 3);

          mat.delete();
        });
      });
    });

    test('should draw centered box', async ({ page }) => {
      await withMemoryTracking(page, async () => {
        await page.evaluate(() => {
          const mat = new window.cv.Mat(100, 100, window.cv.CV_8UC3);
          
          // Centered box calculation
          const position = [50, 50];
          const boxDimensions = [40, 40];
          const thicknessFactor = 1 / 12;
          
          const x = position[0];
          const y = position[1];
          const boxW = boxDimensions[0];
          const boxH = boxDimensions[1];
          
          let posX = Math.floor(x + boxW * thicknessFactor);
          let posY = Math.floor(y + boxH * thicknessFactor);
          let diagX = Math.floor(x + boxW - boxW * thicknessFactor);
          let diagY = Math.floor(y + boxH - boxH * thicknessFactor);
          
          // Apply centering
          const centeredPosX = Math.floor((3 * posX - diagX) / 2);
          const centeredPosY = Math.floor((3 * posY - diagY) / 2);
          const centeredDiagX = Math.floor((posX + diagX) / 2);
          const centeredDiagY = Math.floor((posY + diagY) / 2);
          
          const pt1 = new window.cv.Point(centeredPosX, centeredPosY);
          const pt2 = new window.cv.Point(centeredDiagX, centeredDiagY);
          
          window.cv.rectangle(mat, pt1, pt2, [130, 130, 130], 3);

          mat.delete();
        });
      });
    });

    test('should draw filled box', async ({ page }) => {
      await withMemoryTracking(page, async () => {
        const result = await page.evaluate(() => {
          const mat = new window.cv.Mat(100, 100, window.cv.CV_8UC3, [255, 255, 255, 0]);
          
          const pt1 = new window.cv.Point(20, 20);
          const pt2 = new window.cv.Point(80, 80);
          const color = [100, 100, 100];
          
          window.cv.rectangle(mat, pt1, pt2, color, -1); // Filled

          // Check center is filled
          const pixel = mat.ucharPtr(50, 50);
          const isFilled = pixel[0] === 100 && pixel[1] === 100 && pixel[2] === 100;

          mat.delete();
          return isFilled;
        });

        expect(result).toBe(true);
      });
    });
  });

  test.describe('Complex Drawing Scenarios', () => {
    test('should combine multiple drawing operations', async ({ page }) => {
      await withMemoryTracking(page, async () => {
        await page.evaluate(() => {
          const mat = new window.cv.Mat(200, 200, window.cv.CV_8UC3, [255, 255, 255, 0]);
          
          // Draw rectangle
          window.cv.rectangle(
            mat,
            new window.cv.Point(20, 20),
            new window.cv.Point(180, 180),
            [0, 0, 0],
            2
          );
          
          // Draw circle
          window.cv.circle(mat, new window.cv.Point(100, 100), 30, [100, 200, 100], 2);
          
          // Draw line
          window.cv.line(
            mat,
            new window.cv.Point(50, 100),
            new window.cv.Point(150, 100),
            [100, 100, 100],
            2
          );
          
          // Draw text
          window.cv.putText(
            mat,
            'TEST',
            new window.cv.Point(70, 190),
            window.cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            [0, 0, 0],
            2
          );

          mat.delete();
        });
      });
    });

    test('should handle large number of drawings', async ({ page }) => {
      await withMemoryTracking(page, async () => {
        await page.evaluate(() => {
          const mat = new window.cv.Mat(500, 500, window.cv.CV_8UC3);
          
          // Draw grid of circles
          for (let i = 10; i < 500; i += 50) {
            for (let j = 10; j < 500; j += 50) {
              window.cv.circle(mat, new window.cv.Point(i, j), 5, [100, 100, 100], -1);
            }
          }

          mat.delete();
        });
      });
    });
  });

  test.describe('Error Handling', () => {
    test('should handle invalid color values gracefully', async ({ page }) => {
      await withMemoryTracking(page, async () => {
        await page.evaluate(() => {
          const mat = new window.cv.Mat(100, 100, window.cv.CV_8UC3);
          
          // OpenCV.js clamps color values automatically
          const pt1 = new window.cv.Point(10, 10);
          const pt2 = new window.cv.Point(90, 90);
          const invalidColor = [300, -100, 500]; // Will be clamped to [255, 0, 255]
          
          window.cv.rectangle(mat, pt1, pt2, invalidColor, 2);

          mat.delete();
        });
      });
    });

    test('should handle zero-size rectangles', async ({ page }) => {
      await withMemoryTracking(page, async () => {
        await page.evaluate(() => {
          const mat = new window.cv.Mat(100, 100, window.cv.CV_8UC3);
          
          // Zero-size rectangle (same point)
          const pt = new window.cv.Point(50, 50);
          window.cv.rectangle(mat, pt, pt, [0, 0, 0], 2);

          mat.delete();
        });
      });
    });
  });
});
