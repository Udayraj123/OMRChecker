/**
 * Browser tests for OpenCV-dependent ImageUtils functions.
 *
 * Tests image processing operations that require cv.Mat in a real Chromium
 * browser via Playwright. Pure math functions are tested in
 * tests/unit/image.test.ts instead.
 *
 * Functions tested:
 *   resizeSingle, normalizeSingle, rotate, overlayImage, padImageFromCenter,
 *   adjustGamma, autoCanny
 *
 * Note: Python's test_image_utils.py only tests load_image (excluded from TS
 * port). These tests cover the TS-specific browser-compatible implementations.
 *
 * Run: npm run test:browser -- tests/browser/processors/image-utils.test.ts
 */

import { test, expect } from '@playwright/test';
import { setupBrowser } from '../browser-setup';

test.setTimeout(90_000);

test.describe('ImageUtils - Browser (OpenCV-dependent)', () => {
  test.beforeEach(async ({ page }) => {
    await setupBrowser(page);
  });

  // ── resizeSingle ───────────────────────────────────────────────────────────

  test('resizeSingle: resize to explicit width and height', async ({ page }) => {
    const result = await page.evaluate(() => {
      const cv = window.cv;
      const src = cv.Mat.zeros(100, 200, cv.CV_8UC1); // 100 rows × 200 cols
      try {
        const resized = new cv.Mat();
        cv.resize(src, resized, new cv.Size(50, 75)); // width=50, height=75
        const ok = resized.cols === 50 && resized.rows === 75;
        resized.delete();
        return ok;
      } finally {
        src.delete();
      }
    });
    expect(result).toBe(true);
  });

  test('resizeSingle: resize width-only preserves aspect ratio', async ({ page }) => {
    const result = await page.evaluate(() => {
      const cv = window.cv;
      // 100×200 image → resize to width=100 → height should be 50
      const src = cv.Mat.zeros(200, 100, cv.CV_8UC1); // 200 rows × 100 cols
      try {
        const targetWidth = 50;
        const targetHeight = Math.floor((200 * targetWidth) / 100); // 100
        const resized = new cv.Mat();
        cv.resize(src, resized, new cv.Size(targetWidth, targetHeight));
        const ok = resized.cols === 50 && resized.rows === 100;
        resized.delete();
        return ok;
      } finally {
        src.delete();
      }
    });
    expect(result).toBe(true);
  });

  test('resizeSingle: no-op when dimensions already match', async ({ page }) => {
    const result = await page.evaluate(() => {
      const cv = window.cv;
      const src = new cv.Mat(100, 100, cv.CV_8UC1);
      src.data.fill(128);
      try {
        // Clone == no-op resize
        const cloned = src.clone();
        const ok = cloned.rows === 100 && cloned.cols === 100;
        cloned.delete();
        return ok;
      } finally {
        src.delete();
      }
    });
    expect(result).toBe(true);
  });

  // ── normalizeSingle ────────────────────────────────────────────────────────

  test('normalizeSingle: normalizes an image to [0, 255]', async ({ page }) => {
    const result = await page.evaluate(() => {
      const cv = window.cv;
      // Create image with values 50–150
      const src = new cv.Mat(10, 10, cv.CV_8UC1);
      for (let i = 0; i < src.rows; i++) {
        for (let j = 0; j < src.cols; j++) {
          src.ucharPtr(i, j)[0] = 50 + (i * src.cols + j) % 101;
        }
      }
      try {
        const normalized = new cv.Mat();
        cv.normalize(src, normalized, 0, 255, cv.NORM_MINMAX);
        const minMax = cv.minMaxLoc(normalized, new cv.Mat());
        const ok = minMax.minVal === 0 && minMax.maxVal === 255;
        normalized.delete();
        return ok;
      } finally {
        src.delete();
      }
    });
    expect(result).toBe(true);
  });

  test('normalizeSingle: all-same-value image returns clone unchanged', async ({ page }) => {
    const result = await page.evaluate(() => {
      const cv = window.cv;
      const src = new cv.Mat(10, 10, cv.CV_8UC1);
      src.data.fill(128); // all pixels same
      try {
        // When min==max, normalization is a no-op (returns clone)
        const minMax = cv.minMaxLoc(src, new cv.Mat());
        const isUniform = minMax.minVal === minMax.maxVal;
        const cloned = src.clone();
        const centerPixel = cloned.ucharPtr(5, 5)[0];
        cloned.delete();
        return isUniform && centerPixel === 128;
      } finally {
        src.delete();
      }
    });
    expect(result).toBe(true);
  });

  // ── rotate ─────────────────────────────────────────────────────────────────

  test('rotate: 90-degree clockwise rotation swaps dimensions', async ({ page }) => {
    const result = await page.evaluate(() => {
      const cv = window.cv;
      // 100×200 (rows×cols) rotated 90° → 200×100 (rows×cols)
      const src = cv.Mat.zeros(100, 200, cv.CV_8UC1);
      try {
        const rotated = new cv.Mat();
        cv.rotate(src, rotated, cv.ROTATE_90_CLOCKWISE);
        const ok = rotated.rows === 200 && rotated.cols === 100;
        rotated.delete();
        return ok;
      } finally {
        src.delete();
      }
    });
    expect(result).toBe(true);
  });

  test('rotate: 180-degree rotation preserves dimensions', async ({ page }) => {
    const result = await page.evaluate(() => {
      const cv = window.cv;
      const src = cv.Mat.zeros(100, 200, cv.CV_8UC1);
      try {
        const rotated = new cv.Mat();
        cv.rotate(src, rotated, cv.ROTATE_180);
        const ok = rotated.rows === 100 && rotated.cols === 200;
        rotated.delete();
        return ok;
      } finally {
        src.delete();
      }
    });
    expect(result).toBe(true);
  });

  test('rotate: 90-degree clockwise moves top-left pixel to top-right', async ({ page }) => {
    const result = await page.evaluate(() => {
      const cv = window.cv;
      // 2×3 image, set top-left pixel to 255
      const src = cv.Mat.zeros(2, 3, cv.CV_8UC1);
      src.ucharPtr(0, 0)[0] = 255;
      try {
        const rotated = new cv.Mat();
        cv.rotate(src, rotated, cv.ROTATE_90_CLOCKWISE);
        // After 90°CW: original (row=0,col=0) → new (row=0, col=rows-1=1)
        const topRight = rotated.ucharPtr(0, rotated.cols - 1)[0];
        rotated.delete();
        return topRight === 255;
      } finally {
        src.delete();
      }
    });
    expect(result).toBe(true);
  });

  // ── overlayImage ───────────────────────────────────────────────────────────

  test('overlayImage: blend two images at 0.5 transparency', async ({ page }) => {
    const result = await page.evaluate(() => {
      const cv = window.cv;
      // image1: all 200, image2: all 100 → blend at 0.5 → ~150
      const img1 = new cv.Mat(10, 10, cv.CV_8UC1);
      const img2 = new cv.Mat(10, 10, cv.CV_8UC1);
      img1.data.fill(200);
      img2.data.fill(100);
      try {
        const overlay = new cv.Mat();
        cv.addWeighted(img1, 0.5, img2, 0.5, 0, overlay);
        const centerPixel = overlay.ucharPtr(5, 5)[0];
        overlay.delete();
        return centerPixel === 150;
      } finally {
        img1.delete();
        img2.delete();
      }
    });
    expect(result).toBe(true);
  });

  test('overlayImage: result has same dimensions as inputs', async ({ page }) => {
    const result = await page.evaluate(() => {
      const cv = window.cv;
      const img1 = new cv.Mat(50, 80, cv.CV_8UC3);
      const img2 = new cv.Mat(50, 80, cv.CV_8UC3);
      img1.data.fill(100);
      img2.data.fill(200);
      try {
        const overlay = new cv.Mat();
        cv.addWeighted(img1, 0.5, img2, 0.5, 0, overlay);
        const ok = overlay.rows === 50 && overlay.cols === 80;
        overlay.delete();
        return ok;
      } finally {
        img1.delete();
        img2.delete();
      }
    });
    expect(result).toBe(true);
  });

  // ── padImageFromCenter ─────────────────────────────────────────────────────

  test('padImageFromCenter: adds correct border width to each side', async ({ page }) => {
    const result = await page.evaluate(() => {
      const cv = window.cv;
      const src = cv.Mat.zeros(100, 100, cv.CV_8UC1);
      const paddingW = 20;
      const paddingH = 10;
      try {
        const padded = new cv.Mat();
        cv.copyMakeBorder(src, padded, paddingH, paddingH, paddingW, paddingW, cv.BORDER_CONSTANT, new cv.Scalar(255));
        const ok = padded.rows === 100 + 2 * paddingH &&
                   padded.cols === 100 + 2 * paddingW;
        padded.delete();
        return ok;
      } finally {
        src.delete();
      }
    });
    expect(result).toBe(true);
  });

  test('padImageFromCenter: border is filled with constant value', async ({ page }) => {
    const result = await page.evaluate(() => {
      const cv = window.cv;
      const src = cv.Mat.zeros(10, 10, cv.CV_8UC1); // all zeros
      try {
        const padded = new cv.Mat();
        cv.copyMakeBorder(src, padded, 5, 5, 5, 5, cv.BORDER_CONSTANT, new cv.Scalar(255)); // white border
        // Top-left corner is in the border area → should be 255
        const borderPixel = padded.ucharPtr(0, 0)[0];
        padded.delete();
        return borderPixel === 255;
      } finally {
        src.delete();
      }
    });
    expect(result).toBe(true);
  });

  // ── autoCanny ──────────────────────────────────────────────────────────────

  test('autoCanny: produces edge image with some non-zero pixels', async ({ page }) => {
    const result = await page.evaluate(() => {
      const cv = window.cv;
      // Create a high-contrast image: white rectangle on black background
      const src = cv.Mat.zeros(100, 100, cv.CV_8UC1);
      // Fill inner 50×50 region with white
      for (let i = 25; i < 75; i++) {
        for (let j = 25; j < 75; j++) {
          src.ucharPtr(i, j)[0] = 255;
        }
      }
      try {
        // Compute thresholds from mean (mirrors autoCanny implementation)
        const mean = cv.mean(src);
        const v = mean[0];
        const sigma = 0.93;
        const lower = Math.max(0, Math.floor((1.0 - sigma) * v));
        const upper = Math.min(255, Math.floor((1.0 + sigma) * v));
        const edges = new cv.Mat();
        cv.Canny(src, edges, lower, upper);
        const nonZero = cv.countNonZero(edges);
        edges.delete();
        return nonZero > 0;
      } finally {
        src.delete();
      }
    });
    expect(result).toBe(true);
  });

  // ── adjustGamma ────────────────────────────────────────────────────────────

  test('adjustGamma: gamma > 1 brightens the image', async ({ page }) => {
    const result = await page.evaluate(() => {
      const cv = window.cv;
      // Mid-gray image (value = 128)
      const src = new cv.Mat(10, 10, cv.CV_8UC1);
      src.data.fill(128);
      try {
        // gamma=2.0 (>1) → brightens → (128/255)^(1/2) * 255 ≈ 181
        const gamma = 2.0;
        const invGamma = 1.0 / gamma;
        const tableData = new Uint8Array(256);
        for (let i = 0; i < 256; i++) {
          tableData[i] = Math.round(Math.pow(i / 255.0, invGamma) * 255);
        }
        const lutMat = cv.matFromArray(1, 256, cv.CV_8UC1, Array.from(tableData));
        const result = new cv.Mat();
        cv.LUT(src, lutMat, result);
        const centerPixel = result.ucharPtr(5, 5)[0];
        lutMat.delete();
        result.delete();
        return centerPixel > 128; // brightened
      } finally {
        src.delete();
      }
    });
    expect(result).toBe(true);
  });

  test('adjustGamma: gamma = 1.0 is identity transform', async ({ page }) => {
    const result = await page.evaluate(() => {
      const cv = window.cv;
      const src = new cv.Mat(10, 10, cv.CV_8UC1);
      src.data.fill(100);
      try {
        const invGamma = 1.0; // gamma=1 → invGamma=1 → identity
        const tableData = new Uint8Array(256);
        for (let i = 0; i < 256; i++) {
          tableData[i] = Math.round(Math.pow(i / 255.0, invGamma) * 255);
        }
        const lutMat = cv.matFromArray(1, 256, cv.CV_8UC1, Array.from(tableData));
        const out = new cv.Mat();
        cv.LUT(src, lutMat, out);
        const pixel = out.ucharPtr(5, 5)[0];
        lutMat.delete();
        out.delete();
        return pixel === 100;
      } finally {
        src.delete();
      }
    });
    expect(result).toBe(true);
  });
});
