/**
 * Unit tests for Contrast processor.
 *
 * TypeScript port of tests for src/processors/image/Contrast.py
 */

import { describe, it, expect, beforeAll, afterEach } from 'vitest';
const cv = global.cv;
import { Contrast } from '../Contrast';
import { createProcessingContext } from '../../base';

describe('Contrast', () => {
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

  it('should initialize with default parameters (manual mode)', () => {
    const processor = new Contrast();
    expect(processor.getName()).toBe('Contrast');
  });

  it('should initialize with custom manual parameters', () => {
    const processor = new Contrast({
      mode: 'manual',
      alpha: 2.0,
      beta: 10,
    });
    expect(processor.getName()).toBe('Contrast');
  });

  it('should initialize with auto mode', () => {
    const processor = new Contrast({
      mode: 'auto',
      clipPercentage: 2,
    });
    expect(processor.getName()).toBe('Contrast');
  });

  it('should apply manual contrast adjustment', () => {
    testImage = new cv.Mat(100, 100, cv.CV_8UC1);
    // Fill with gray values
    for (let i = 0; i < testImage.rows; i++) {
      for (let j = 0; j < testImage.cols; j++) {
        testImage.ucharPtr(i, j)[0] = 100;
      }
    }

    const colorImage = testImage.clone();
    const context = createProcessingContext('test.jpg', testImage, colorImage, {});

    const processor = new Contrast({
      mode: 'manual',
      alpha: 1.5,
      beta: 20,
    });
    const result = processor.process(context);

    expect(result.grayImage).toBeDefined();
    expect(result.grayImage.rows).toBe(100);
    expect(result.grayImage.cols).toBe(100);

    // Check that contrast was applied (100 * 1.5 + 20 = 170)
    const centerPixel = result.grayImage.ucharPtr(50, 50)[0];
    expect(centerPixel).toBeCloseTo(170, 0);

    // Cleanup
    result.grayImage.delete();
    result.coloredImage.delete();
  });

  it('should apply auto contrast adjustment', () => {
    testImage = new cv.Mat(100, 100, cv.CV_8UC1);
    // Fill with low-contrast values (80-120 range)
    for (let i = 0; i < testImage.rows; i++) {
      for (let j = 0; j < testImage.cols; j++) {
        testImage.ucharPtr(i, j)[0] = 80 + (i % 40);
      }
    }

    const colorImage = testImage.clone();
    const context = createProcessingContext('test.jpg', testImage, colorImage, {});

    const processor = new Contrast({
      mode: 'auto',
      clipPercentage: 1,
    });
    const result = processor.process(context);

    expect(result.grayImage).toBeDefined();

    // Check that the range was expanded
    let minVal = 255;
    let maxVal = 0;
    for (let i = 0; i < result.grayImage.rows; i++) {
      for (let j = 0; j < result.grayImage.cols; j++) {
        const val = result.grayImage.ucharPtr(i, j)[0];
        minVal = Math.min(minVal, val);
        maxVal = Math.max(maxVal, val);
      }
    }

    // Auto contrast should expand the range
    expect(maxVal - minVal).toBeGreaterThan(40); // Original range was 40

    // Cleanup
    result.grayImage.delete();
    result.coloredImage.delete();
  });

  it('should increase contrast of low-contrast image', () => {
    // Create low-contrast image (values between 100-150)
    testImage = new cv.Mat(100, 100, cv.CV_8UC1);
    for (let i = 0; i < testImage.rows; i++) {
      for (let j = 0; j < testImage.cols; j++) {
        testImage.ucharPtr(i, j)[0] = 100 + (i % 50);
      }
    }

    const colorImage = testImage.clone();
    const context = createProcessingContext('test.jpg', testImage, colorImage, {});

    const processor = new Contrast({
      mode: 'manual',
      alpha: 2.0,
      beta: -100,
    });
    const result = processor.process(context);

    // Calculate standard deviation of result
    let sum = 0;
    let count = 0;
    for (let i = 0; i < result.grayImage.rows; i++) {
      for (let j = 0; j < result.grayImage.cols; j++) {
        sum += result.grayImage.ucharPtr(i, j)[0];
        count++;
      }
    }
    const mean = sum / count;

    let variance = 0;
    for (let i = 0; i < result.grayImage.rows; i++) {
      for (let j = 0; j < result.grayImage.cols; j++) {
        const diff = result.grayImage.ucharPtr(i, j)[0] - mean;
        variance += diff * diff;
      }
    }
    const stdDev = Math.sqrt(variance / count);

    // After contrast enhancement, std dev should be higher
    expect(stdDev).toBeGreaterThan(15);

    // Cleanup
    result.grayImage.delete();
    result.coloredImage.delete();
  });
});

