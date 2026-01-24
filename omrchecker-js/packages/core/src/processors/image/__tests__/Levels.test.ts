/**
 * Unit tests for Levels processor.
 *
 * TypeScript port of tests for src/processors/image/Levels.py
 */

import { describe, it, expect, beforeAll, afterEach } from 'vitest';
import { Levels } from '../Levels';
import { createProcessingContext } from '../../base';

describe('Levels', () => {
  let testImage: any;
  const cv = global.cv;

  beforeAll(async () => {
    // Wait for OpenCV to be ready
    if (cv && cv.getBuildInformation) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  });

  afterEach(() => {
    if (testImage && !testImage.isDeleted()) {
      testImage.delete();
    }
  });

  it('should initialize with default parameters', () => {
    const processor = new Levels();
    expect(processor.getName()).toBe('Levels');
  });

  it('should initialize with custom parameters', () => {
    const processor = new Levels({
      low: 0.1,
      high: 0.9,
      gamma: 1.5,
    });
    expect(processor.getName()).toBe('Levels');
  });

  it('should apply levels adjustment', () => {
    testImage = new cv.Mat(100, 100, cv.CV_8UC1);
    // Fill with gradient
    for (let i = 0; i < testImage.rows; i++) {
      for (let j = 0; j < testImage.cols; j++) {
        testImage.ucharPtr(i, j)[0] = (i + j) % 256;
      }
    }

    const colorImage = testImage.clone();
    const context = createProcessingContext('test.jpg', testImage, colorImage, {});

    const processor = new Levels({
      low: 0.2,
      high: 0.8,
      gamma: 1.0,
    });
    const result = processor.process(context);

    expect(result.grayImage).toBeDefined();
    expect(result.grayImage.rows).toBe(100);
    expect(result.grayImage.cols).toBe(100);

    // Cleanup
    result.grayImage.delete();
    result.coloredImage.delete();
    processor.cleanup();
  });

  it('should darken low values with low threshold', () => {
    testImage = new cv.Mat(100, 100, cv.CV_8UC1);
    // Fill with low values (50)
    for (let i = 0; i < testImage.rows; i++) {
      for (let j = 0; j < testImage.cols; j++) {
        testImage.ucharPtr(i, j)[0] = 50;
      }
    }

    const colorImage = testImage.clone();
    const context = createProcessingContext('test.jpg', testImage, colorImage, {});

    const processor = new Levels({
      low: 0.3, // 76.5 in 0-255 range
      high: 1.0,
      gamma: 1.0,
    });
    const result = processor.process(context);

    // Values below low threshold (50 < 76.5) should be darkened
    const centerPixel = result.grayImage.ucharPtr(50, 50)[0];
    expect(centerPixel).toBeLessThan(50);

    // Cleanup
    result.grayImage.delete();
    result.coloredImage.delete();
    processor.cleanup();
  });

  it('should brighten high values with high threshold', () => {
    testImage = new cv.Mat(100, 100, cv.CV_8UC1);
    // Fill with high values (200)
    for (let i = 0; i < testImage.rows; i++) {
      for (let j = 0; j < testImage.cols; j++) {
        testImage.ucharPtr(i, j)[0] = 200;
      }
    }

    const colorImage = testImage.clone();
    const context = createProcessingContext('test.jpg', testImage, colorImage, {});

    const processor = new Levels({
      low: 0.0,
      high: 0.7, // 178.5 in 0-255 range
      gamma: 1.0,
    });
    const result = processor.process(context);

    // Values above high threshold (200 > 178.5) should become 255
    const centerPixel = result.grayImage.ucharPtr(50, 50)[0];
    expect(centerPixel).toBe(255);

    // Cleanup
    result.grayImage.delete();
    result.coloredImage.delete();
    processor.cleanup();
  });

  it('should apply gamma correction', () => {
    testImage = new cv.Mat(100, 100, cv.CV_8UC1);
    // Fill with mid-tone values (128)
    for (let i = 0; i < testImage.rows; i++) {
      for (let j = 0; j < testImage.cols; j++) {
        testImage.ucharPtr(i, j)[0] = 128;
      }
    }

    const colorImage = testImage.clone();
    const context = createProcessingContext('test.jpg', testImage, colorImage, {});

    // Gamma > 1 should darken mid-tones
    const processor = new Levels({
      low: 0.0,
      high: 1.0,
      gamma: 2.0,
    });
    const result = processor.process(context);

    const centerPixel = result.grayImage.ucharPtr(50, 50)[0];
    // With gamma=2.0, 128 (0.5 normalized) becomes 0.5^0.5 = 0.707, so ~180
    expect(centerPixel).toBeGreaterThan(128);
    expect(centerPixel).toBeLessThan(200);

    // Cleanup
    result.grayImage.delete();
    result.coloredImage.delete();
    processor.cleanup();
  });

  it('should expand limited range to full range', () => {
    testImage = new cv.Mat(100, 100, cv.CV_8UC1);
    // Fill with limited range (100-150)
    for (let i = 0; i < testImage.rows; i++) {
      for (let j = 0; j < testImage.cols; j++) {
        testImage.ucharPtr(i, j)[0] = 100 + (j % 50);
      }
    }

    const colorImage = testImage.clone();
    const context = createProcessingContext('test.jpg', testImage, colorImage, {});

    const processor = new Levels({
      low: 0.39, // ~100/255
      high: 0.59, // ~150/255
      gamma: 1.0,
    });
    const result = processor.process(context);

    // Check that range is expanded
    let minVal = 255;
    let maxVal = 0;
    for (let i = 0; i < result.grayImage.rows; i++) {
      for (let j = 0; j < result.grayImage.cols; j++) {
        const val = result.grayImage.ucharPtr(i, j)[0];
        minVal = Math.min(minVal, val);
        maxVal = Math.max(maxVal, val);
      }
    }

    // Range should be expanded close to 0-255
    expect(minVal).toBeLessThan(10);
    expect(maxVal).toBeGreaterThan(245);

    // Cleanup
    result.grayImage.delete();
    result.coloredImage.delete();
    processor.cleanup();
  });
});

