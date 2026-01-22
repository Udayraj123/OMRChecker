/**
 * Unit tests for ImageUtils.
 *
 * TypeScript port of tests for src/utils/image.py
 */

import { describe, it, expect, beforeAll } from 'vitest';
const cv = global.cv;
import { ImageUtils } from '../ImageUtils';

describe('ImageUtils', () => {
  beforeAll(async () => {
    // Wait for OpenCV to be ready
    if (cv.getBuildInformation) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  });

  describe('resizeSingle', () => {
    it('should resize image to specified dimensions', () => {
      const image = new cv.Mat(100, 100, cv.CV_8UC1);
      const resized = ImageUtils.resizeSingle(image, 50, 50);

      expect(resized).toBeDefined();
      expect(resized!.rows).toBe(50);
      expect(resized!.cols).toBe(50);

      image.delete();
      resized!.delete();
    });

    it('should calculate height when only width provided', () => {
      const image = new cv.Mat(100, 200, cv.CV_8UC1); // Aspect ratio 1:2
      const resized = ImageUtils.resizeSingle(image, 100);

      expect(resized!.cols).toBe(100);
      expect(resized!.rows).toBe(50); // Maintains aspect ratio

      image.delete();
      resized!.delete();
    });

    it('should calculate width when only height provided', () => {
      const image = new cv.Mat(100, 200, cv.CV_8UC1);
      const resized = ImageUtils.resizeSingle(image, undefined, 50);

      expect(resized!.rows).toBe(50);
      expect(resized!.cols).toBe(100); // Maintains aspect ratio

      image.delete();
      resized!.delete();
    });

    it('should return original image if dimensions match', () => {
      const image = new cv.Mat(100, 100, cv.CV_8UC1);
      const resized = ImageUtils.resizeSingle(image, 100, 100);

      expect(resized).toBe(image); // Same reference

      image.delete();
    });

    it('should return null for null input', () => {
      const resized = ImageUtils.resizeSingle(null, 100, 100);
      expect(resized).toBeNull();
    });
  });

  describe('rotate', () => {
    it('should rotate image 90 degrees clockwise', () => {
      const image = new cv.Mat(100, 50, cv.CV_8UC1); // 100x50
      const rotated = ImageUtils.rotate(image, cv.ROTATE_90_CLOCKWISE, false);

      expect(rotated.rows).toBe(50); // Width becomes height
      expect(rotated.cols).toBe(100); // Height becomes width

      image.delete();
      rotated.delete();
    });

    it('should rotate 180 degrees', () => {
      const image = new cv.Mat(100, 50, cv.CV_8UC1);
      const rotated = ImageUtils.rotate(image, cv.ROTATE_180, false);

      expect(rotated.rows).toBe(100); // Same dimensions
      expect(rotated.cols).toBe(50);

      image.delete();
      rotated.delete();
    });

    it('should keep original shape when requested', () => {
      const image = new cv.Mat(100, 50, cv.CV_8UC1);
      const rotated = ImageUtils.rotate(image, cv.ROTATE_90_CLOCKWISE, true);

      expect(rotated.rows).toBe(100); // Original dimensions preserved
      expect(rotated.cols).toBe(50);

      image.delete();
      rotated.delete();
    });
  });

  describe('normalize', () => {
    it('should normalize image to 0-255 range', () => {
      const image = new cv.Mat(10, 10, cv.CV_8UC1);
      // Fill with values 50-150
      for (let i = 0; i < image.rows; i++) {
        for (let j = 0; j < image.cols; j++) {
          image.ucharPtr(i, j)[0] = 50 + (i * 10);
        }
      }

      const normalized = ImageUtils.normalizeSingle(image, 0, 255, cv.NORM_MINMAX) as cv.Mat;

      const minMax = cv.minMaxLoc(normalized);
      expect(minMax.minVal).toBe(0);
      expect(minMax.maxVal).toBe(255);

      image.delete();
      normalized.delete();
    });

    it('should return original image if all pixels are same value', () => {
      const image = new cv.Mat(10, 10, cv.CV_8UC1, new cv.Scalar(128));
      const normalized = ImageUtils.normalizeSingle(image, 0, 255, cv.NORM_MINMAX);

      expect(normalized).toBe(image); // Same reference for constant image

      image.delete();
    });

    it('should normalize multiple images', () => {
      const image1 = new cv.Mat(10, 10, cv.CV_8UC1, new cv.Scalar(50));
      const image2 = new cv.Mat(10, 10, cv.CV_8UC1, new cv.Scalar(100));

      const result = ImageUtils.normalize([image1, image2], 0, 255, cv.NORM_MINMAX) as cv.Mat[];

      expect(result).toHaveLength(2);
      expect(result[0]).toBeDefined();
      expect(result[1]).toBeDefined();

      image1.delete();
      image2.delete();
      result[0].delete();
      result[1].delete();
    });
  });

  describe('autoCanny', () => {
    it('should detect edges with automatic thresholds', () => {
      const image = new cv.Mat(100, 100, cv.CV_8UC1);
      // Create a pattern
      for (let i = 0; i < image.rows; i++) {
        for (let j = 0; j < image.cols; j++) {
          image.ucharPtr(i, j)[0] = (i + j) % 2 === 0 ? 0 : 255;
        }
      }

      const edges = ImageUtils.autoCanny(image, 0.93);

      expect(edges).toBeDefined();
      expect(edges.rows).toBe(100);
      expect(edges.cols).toBe(100);

      image.delete();
      edges.delete();
    });
  });

  describe('adjustGamma', () => {
    it('should apply gamma correction', () => {
      const image = new cv.Mat(10, 10, cv.CV_8UC1, new cv.Scalar(128));
      const adjusted = ImageUtils.adjustGamma(image, 2.0);

      expect(adjusted).toBeDefined();
      expect(adjusted.rows).toBe(10);
      expect(adjusted.cols).toBe(10);

      // With gamma=2.0, 128 should become darker (value ~64)
      const centerValue = adjusted.ucharPtr(5, 5)[0];
      expect(centerValue).toBeLessThan(128);

      image.delete();
      adjusted.delete();
    });

    it('should lighten image with gamma < 1', () => {
      const image = new cv.Mat(10, 10, cv.CV_8UC1, new cv.Scalar(128));
      const adjusted = ImageUtils.adjustGamma(image, 0.5);

      // With gamma=0.5, 128 should become lighter (value ~181)
      const centerValue = adjusted.ucharPtr(5, 5)[0];
      expect(centerValue).toBeGreaterThan(128);

      image.delete();
      adjusted.delete();
    });
  });

  describe('padding operations', () => {
    it('should pad image to height', () => {
      const image = new cv.Mat(50, 100, cv.CV_8UC1, new cv.Scalar(0));
      const padded = ImageUtils.padImageToHeight(image, 100, [255, 255, 255]);

      expect(padded.rows).toBe(100);
      expect(padded.cols).toBe(100);

      // Check padding is white
      const paddingValue = padded.ucharPtr(99, 50)[0];
      expect(paddingValue).toBe(255);

      image.delete();
      if (padded !== image) padded.delete();
    });

    it('should pad image to width', () => {
      const image = new cv.Mat(100, 50, cv.CV_8UC1, new cv.Scalar(0));
      const padded = ImageUtils.padImageToWidth(image, 100, [255, 255, 255]);

      expect(padded.rows).toBe(100);
      expect(padded.cols).toBe(100);

      // Check padding is white
      const paddingValue = padded.ucharPtr(50, 99)[0];
      expect(paddingValue).toBe(255);

      image.delete();
      if (padded !== image) padded.delete();
    });

    it('should pad image from center', () => {
      const image = new cv.Mat(50, 50, cv.CV_8UC1, new cv.Scalar(0));
      const [padded, padRange] = ImageUtils.padImageFromCenter(image, 25, 25, 255);

      expect(padded.rows).toBe(100); // 50 + 25*2
      expect(padded.cols).toBe(100); // 50 + 25*2
      expect(padRange).toEqual([25, 75, 25, 75]);

      // Check center is black (original)
      const centerValue = padded.ucharPtr(50, 50)[0];
      expect(centerValue).toBe(0);

      // Check corners are white (padding)
      const cornerValue = padded.ucharPtr(0, 0)[0];
      expect(cornerValue).toBe(255);

      image.delete();
      padded.delete();
    });

    it('should not pad if already at target size', () => {
      const image = new cv.Mat(100, 100, cv.CV_8UC1);
      const padded = ImageUtils.padImageToHeight(image, 100);

      expect(padded).toBe(image); // Same reference

      image.delete();
    });
  });

  describe('stacking operations', () => {
    it('should stack images horizontally with padding', () => {
      const image1 = new cv.Mat(50, 100, cv.CV_8UC1, new cv.Scalar(50));
      const image2 = new cv.Mat(100, 100, cv.CV_8UC1, new cv.Scalar(150));

      const stacked = ImageUtils.getPaddedHstack([image1, image2]);

      expect(stacked.rows).toBe(100); // Max height
      expect(stacked.cols).toBe(200); // Sum of widths

      image1.delete();
      image2.delete();
      stacked.delete();
    });

    it('should stack images vertically with padding', () => {
      const image1 = new cv.Mat(100, 50, cv.CV_8UC1, new cv.Scalar(50));
      const image2 = new cv.Mat(100, 100, cv.CV_8UC1, new cv.Scalar(150));

      const stacked = ImageUtils.getPaddedVstack([image1, image2]);

      expect(stacked.rows).toBe(200); // Sum of heights
      expect(stacked.cols).toBe(100); // Max width

      image1.delete();
      image2.delete();
      stacked.delete();
    });
  });

  describe('clipZoneToImageBounds', () => {
    it('should clip rectangle to image bounds', () => {
      const image = new cv.Mat(100, 100, cv.CV_8UC1);
      const rectangle: [[number, number], [number, number]] = [
        [-10, -10],
        [110, 110],
      ];

      const clipped = ImageUtils.clipZoneToImageBounds(rectangle, image);

      expect(clipped[0]).toEqual([0, 0]); // Clipped to top-left
      expect(clipped[1]).toEqual([100, 100]); // Clipped to bottom-right

      image.delete();
    });

    it('should not modify valid rectangle', () => {
      const image = new cv.Mat(100, 100, cv.CV_8UC1);
      const rectangle: [[number, number], [number, number]] = [
        [10, 10],
        [90, 90],
      ];

      const clipped = ImageUtils.clipZoneToImageBounds(rectangle, image);

      expect(clipped[0]).toEqual([10, 10]);
      expect(clipped[1]).toEqual([90, 90]);

      image.delete();
    });
  });

  describe('overlayImage', () => {
    it('should overlay two images with transparency', () => {
      const image1 = new cv.Mat(100, 100, cv.CV_8UC1, new cv.Scalar(100));
      const image2 = new cv.Mat(100, 100, cv.CV_8UC1, new cv.Scalar(200));

      const overlay = ImageUtils.overlayImage(image1, image2, 0.5);

      expect(overlay).toBeDefined();
      expect(overlay.rows).toBe(100);
      expect(overlay.cols).toBe(100);

      // Value should be average of 100 and 200
      const centerValue = overlay.ucharPtr(50, 50)[0];
      expect(centerValue).toBeCloseTo(150, 0);

      image1.delete();
      image2.delete();
      overlay.delete();
    });
  });

  describe('getCroppedWarpedRectanglePoints', () => {
    it('should calculate warped points and dimensions', () => {
      const corners: [number, number][] = [
        [0, 0], // tl
        [100, 0], // tr
        [100, 200], // br
        [0, 200], // bl
      ];

      const [warpedPoints, dimensions] = ImageUtils.getCroppedWarpedRectanglePoints(corners);

      expect(dimensions[0]).toBeCloseTo(100, 0); // width
      expect(dimensions[1]).toBeCloseTo(200, 0); // height
      expect(warpedPoints).toBeDefined();
      expect(warpedPoints.length).toBe(4);
      expect(warpedPoints[0]).toEqual([0, 0]);
      expect(warpedPoints[1]).toEqual([99, 0]);
      expect(warpedPoints[2]).toEqual([99, 199]);
      expect(warpedPoints[3]).toEqual([0, 199]);
    });

    it('should handle non-rectangular shapes', () => {
      const corners: [number, number][] = [
        [10, 10], // tl
        [110, 5], // tr (slightly tilted)
        [115, 205], // br
        [5, 210], // bl
      ];

      const [warpedPoints, dimensions] = ImageUtils.getCroppedWarpedRectanglePoints(corners);

      expect(dimensions[0]).toBeGreaterThan(90);
      expect(dimensions[0]).toBeLessThan(120);
      expect(dimensions[1]).toBeGreaterThan(190);
      expect(dimensions[1]).toBeLessThan(210);
      expect(warpedPoints).toBeDefined();
      expect(warpedPoints.length).toBe(4);
    });
  });
});

