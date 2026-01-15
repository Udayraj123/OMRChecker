/**
 * Unit tests for GaussianBlur processor.
 *
 * TypeScript port of tests for src/processors/image/GaussianBlur.py
 */

import { describe, it, expect, beforeAll, afterEach } from 'vitest';
import * as cv from '@techstark/opencv-js';
import { GaussianBlur } from '../GaussianBlur';
import { createProcessingContext } from '../../base';

describe('GaussianBlur', () => {
  let testImage: cv.Mat;

  beforeAll(async () => {
    // Wait for OpenCV to be ready
    if (cv.getBuildInformation) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  });

  afterEach(() => {
    if (testImage && !testImage.isDeleted()) {
      testImage.delete();
    }
  });

  it('should initialize with default parameters', () => {
    const processor = new GaussianBlur();
    expect(processor.getName()).toBe('GaussianBlur');
  });

  it('should initialize with custom parameters', () => {
    const processor = new GaussianBlur({
      kSize: [5, 5],
      sigmaX: 1.5,
    });
    expect(processor.getName()).toBe('GaussianBlur');
  });

  it('should apply Gaussian blur to image', () => {
    testImage = new cv.Mat(100, 100, cv.CV_8UC1);
    // Fill with some pattern
    for (let i = 0; i < testImage.rows; i++) {
      for (let j = 0; j < testImage.cols; j++) {
        testImage.ucharPtr(i, j)[0] = (i + j) % 255;
      }
    }

    const colorImage = testImage.clone();
    const context = createProcessingContext('test.jpg', testImage, colorImage, {});

    const processor = new GaussianBlur({ kSize: [5, 5], sigmaX: 0 });
    const result = processor.process(context);

    expect(result.grayImage).toBeDefined();
    expect(result.grayImage.rows).toBe(100);
    expect(result.grayImage.cols).toBe(100);

    // Cleanup
    result.grayImage.delete();
    result.coloredImage.delete();
  });

  it('should blur the image (reduce high-frequency content)', () => {
    // Create image with high contrast
    testImage = new cv.Mat(100, 100, cv.CV_8UC1);
    for (let i = 0; i < testImage.rows; i++) {
      for (let j = 0; j < testImage.cols; j++) {
        testImage.ucharPtr(i, j)[0] = (i % 2) * 255; // Alternating black/white rows
      }
    }

    const originalData = new Uint8Array(testImage.data.length);
    originalData.set(testImage.data);

    const colorImage = testImage.clone();
    const context = createProcessingContext('test.jpg', testImage, colorImage, {});

    const processor = new GaussianBlur({ kSize: [15, 15], sigmaX: 5 });
    const result = processor.process(context);

    // Calculate variance of differences
    let diffSum = 0;
    for (let i = 0; i < originalData.length; i++) {
      diffSum += Math.abs(result.grayImage.data[i] - originalData[i]);
    }
    const avgDiff = diffSum / originalData.length;

    // After blur, the image should be different from original
    expect(avgDiff).toBeGreaterThan(10);

    // Cleanup
    result.grayImage.delete();
    result.coloredImage.delete();
  });
});

