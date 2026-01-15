/**
 * Tests for image utilities
 *
 * TypeScript port of relevant tests from Python test suite
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import cv from '@techstark/opencv-js';
import { ImageUtils } from '../image';
import { CLR_BLACK } from '../constants';

describe('ImageUtils', () => {
  let testMat: cv.Mat;

  beforeEach(() => {
    // Create a test image (100x100, grayscale)
    testMat = new cv.Mat(100, 100, cv.CV_8UC1, new cv.Scalar(128));
  });

  afterEach(() => {
    if (testMat && !testMat.isDeleted()) {
      testMat.delete();
    }
  });

  describe('Resize Operations (DRY Pattern)', () => {
    it('should resize single image with width only', () => {
      const resized = ImageUtils.resizeSingle(testMat, 50);

      expect(resized.cols).toBe(50);
      expect(resized.rows).toBe(50); // Should maintain aspect ratio

      resized.delete();
    });

    it('should resize single image with height only', () => {
      const resized = ImageUtils.resizeSingle(testMat, undefined, 75);

      expect(resized.cols).toBe(75); // Should maintain aspect ratio
      expect(resized.rows).toBe(75);

      resized.delete();
    });

    it('should resize single image with both dimensions', () => {
      const resized = ImageUtils.resizeSingle(testMat, 80, 60);

      expect(resized.cols).toBe(80);
      expect(resized.rows).toBe(60);

      resized.delete();
    });

    it('should clone image when dimensions match', () => {
      const resized = ImageUtils.resizeSingle(testMat, 100, 100);

      expect(resized.cols).toBe(100);
      expect(resized.rows).toBe(100);
      expect(resized.ptr(0, 0)).not.toBe(testMat.ptr(0, 0)); // Different memory

      resized.delete();
    });

    it('should resize multiple images', () => {
      const mat2 = new cv.Mat(100, 100, cv.CV_8UC1);
      const resized = ImageUtils.resizeMultiple([testMat, mat2], 50, 50);

      expect(resized).toHaveLength(2);
      expect(resized[0].cols).toBe(50);
      expect(resized[0].rows).toBe(50);
      expect(resized[1].cols).toBe(50);
      expect(resized[1].rows).toBe(50);

      resized.forEach(m => m.delete());
      mat2.delete();
    });

    it('should resize to shape [height, width]', () => {
      const [resized] = ImageUtils.resizeToShape([75, 125], testMat);

      expect(resized.cols).toBe(125);
      expect(resized.rows).toBe(75);

      resized.delete();
    });

    it('should resize to dimensions [width, height]', () => {
      const [resized] = ImageUtils.resizeToDimensions([125, 75], testMat);

      expect(resized.cols).toBe(125);
      expect(resized.rows).toBe(75);

      resized.delete();
    });
  });

  describe('Normalization Operations (DRY Pattern)', () => {
    it('should normalize single image', () => {
      const normalized = ImageUtils.normalizeSingle(testMat);

      expect(normalized.cols).toBe(testMat.cols);
      expect(normalized.rows).toBe(testMat.rows);

      normalized.delete();
    });

    it('should normalize multiple images', () => {
      const mat2 = new cv.Mat(100, 100, cv.CV_8UC1);
      const normalized = ImageUtils.normalize([testMat, mat2]);

      expect(normalized).toHaveLength(2);

      normalized.forEach(m => m.delete());
      mat2.delete();
    });

    it('should normalize with custom alpha and beta', () => {
      const normalized = ImageUtils.normalizeSingle(testMat, 50, 200);

      expect(normalized.cols).toBe(testMat.cols);
      expect(normalized.rows).toBe(testMat.rows);

      normalized.delete();
    });
  });

  describe('Padding Operations (DRY Pattern)', () => {
    it('should pad image to height', () => {
      const padded = ImageUtils.padImageToHeight(testMat, 150);

      expect(padded.cols).toBe(100);
      expect(padded.rows).toBe(150);

      padded.delete();
    });

    it('should not pad if already at target height', () => {
      const padded = ImageUtils.padImageToHeight(testMat, 80);

      expect(padded.cols).toBe(100);
      expect(padded.rows).toBe(100); // No padding needed

      padded.delete();
    });

    it('should pad image to width', () => {
      const padded = ImageUtils.padImageToWidth(testMat, 150);

      expect(padded.cols).toBe(150);
      expect(padded.rows).toBe(100);

      padded.delete();
    });

    it('should pad image from center', () => {
      const padded = ImageUtils.padImageFromCenter(testMat, 20, 10);

      expect(padded.cols).toBe(120); // 100 + 20
      expect(padded.rows).toBe(110); // 100 + 10

      padded.delete();
    });

    it('should pad images to max dimensions', () => {
      const mat1 = new cv.Mat(80, 60, cv.CV_8UC1);
      const mat2 = new cv.Mat(100, 120, cv.CV_8UC1);
      const mat3 = new cv.Mat(90, 110, cv.CV_8UC1);

      const padded = ImageUtils.padImagesToMax([mat1, mat2, mat3]);

      expect(padded).toHaveLength(3);
      padded.forEach(m => {
        expect(m.cols).toBe(120); // Max width
        expect(m.rows).toBe(100); // Max height
      });

      padded.forEach(m => m.delete());
      [mat1, mat2, mat3].forEach(m => m.delete());
    });

    it('should pad image uniformly', () => {
      const padded = ImageUtils.padImageUniform(testMat, 10);

      expect(padded.cols).toBe(120); // 100 + 10*2
      expect(padded.rows).toBe(120); // 100 + 10*2

      padded.delete();
    });

    it('should use custom padding color', () => {
      const padded = ImageUtils.padImageUniform(testMat, 5, CLR_BLACK);

      expect(padded.cols).toBe(110);
      expect(padded.rows).toBe(110);

      padded.delete();
    });
  });

  describe('Stacking Operations', () => {
    it('should stack images horizontally', () => {
      const mat1 = new cv.Mat(50, 60, cv.CV_8UC1);
      const mat2 = new cv.Mat(50, 80, cv.CV_8UC1);

      const stacked = ImageUtils.stackImagesHorizontal([mat1, mat2]);

      expect(stacked.cols).toBe(140); // 60 + 80
      expect(stacked.rows).toBe(50);

      stacked.delete();
      [mat1, mat2].forEach(m => m.delete());
    });

    it('should stack images vertically', () => {
      const mat1 = new cv.Mat(50, 60, cv.CV_8UC1);
      const mat2 = new cv.Mat(80, 60, cv.CV_8UC1);

      const stacked = ImageUtils.stackImagesVertical([mat1, mat2]);

      expect(stacked.cols).toBe(60);
      expect(stacked.rows).toBe(130); // 50 + 80

      stacked.delete();
      [mat1, mat2].forEach(m => m.delete());
    });

    it('should handle single image in stack', () => {
      const stacked = ImageUtils.stackImagesHorizontal([testMat]);

      expect(stacked.cols).toBe(100);
      expect(stacked.rows).toBe(100);

      stacked.delete();
    });

    it('should throw error for empty array', () => {
      expect(() => ImageUtils.stackImagesHorizontal([])).toThrow();
      expect(() => ImageUtils.stackImagesVertical([])).toThrow();
    });
  });

  describe('Color Operations', () => {
    it('should convert grayscale to BGR', () => {
      const bgr = ImageUtils.grayToBGR(testMat);

      expect(bgr.channels()).toBe(3);
      expect(bgr.cols).toBe(100);
      expect(bgr.rows).toBe(100);

      bgr.delete();
    });

    it('should return clone if already BGR', () => {
      const colorMat = new cv.Mat(100, 100, cv.CV_8UC3);
      const bgr = ImageUtils.grayToBGR(colorMat);

      expect(bgr.channels()).toBe(3);

      bgr.delete();
      colorMat.delete();
    });

    it('should convert BGR to grayscale', () => {
      const colorMat = new cv.Mat(100, 100, cv.CV_8UC3);
      const gray = ImageUtils.bgrToGray(colorMat);

      expect(gray.channels()).toBe(1);
      expect(gray.cols).toBe(100);
      expect(gray.rows).toBe(100);

      gray.delete();
      colorMat.delete();
    });

    it('should return clone if already grayscale', () => {
      const gray = ImageUtils.bgrToGray(testMat);

      expect(gray.channels()).toBe(1);

      gray.delete();
    });
  });

  describe('Utility Operations', () => {
    it('should get dimensions as [width, height]', () => {
      const dims = ImageUtils.getDimensions(testMat);

      expect(dims).toEqual([100, 100]);
    });

    it('should get shape as [height, width, channels]', () => {
      const shape = ImageUtils.getShape(testMat);

      expect(shape).toEqual([100, 100, 1]);
    });

    it('should detect grayscale image', () => {
      expect(ImageUtils.isGrayscale(testMat)).toBe(true);

      const colorMat = new cv.Mat(100, 100, cv.CV_8UC3);
      expect(ImageUtils.isGrayscale(colorMat)).toBe(false);

      colorMat.delete();
    });

    it('should clone image', () => {
      const cloned = ImageUtils.clone(testMat);

      expect(cloned.cols).toBe(100);
      expect(cloned.rows).toBe(100);
      expect(cloned.ptr(0, 0)).not.toBe(testMat.ptr(0, 0));

      cloned.delete();
    });
  });
});

