/**
 * Unit tests for MedianBlur processor.
 *
 * TypeScript port of tests for src/processors/image/MedianBlur.py
 */

import { describe, it, expect, beforeAll, afterEach } from 'vitest';
import * as cv from '@techstark/opencv-js';
import { MedianBlur } from '../MedianBlur';
import { createProcessingContext } from '../../base';

describe('MedianBlur', () => {
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
    const processor = new MedianBlur();
    expect(processor.getName()).toBe('MedianBlur');
  });

  it('should initialize with custom parameters', () => {
    const processor = new MedianBlur({ kSize: 7 });
    expect(processor.getName()).toBe('MedianBlur');
  });

  it('should apply median blur to image', () => {
    testImage = new cv.Mat(100, 100, cv.CV_8UC1);
    // Fill with some pattern
    for (let i = 0; i < testImage.rows; i++) {
      for (let j = 0; j < testImage.cols; j++) {
        testImage.ucharPtr(i, j)[0] = (i + j) % 255;
      }
    }

    const colorImage = testImage.clone();
    const context = createProcessingContext('test.jpg', testImage, colorImage, {});

    const processor = new MedianBlur({ kSize: 5 });
    const result = processor.process(context);

    expect(result.grayImage).toBeDefined();
    expect(result.grayImage.rows).toBe(100);
    expect(result.grayImage.cols).toBe(100);

    // Cleanup
    result.grayImage.delete();
    result.coloredImage.delete();
  });

  it('should remove salt-and-pepper noise', () => {
    // Create image with salt-and-pepper noise
    testImage = new cv.Mat(100, 100, cv.CV_8UC1);
    const baseValue = 128;

    for (let i = 0; i < testImage.rows; i++) {
      for (let j = 0; j < testImage.cols; j++) {
        let value = baseValue;
        const random = Math.random();
        // Add random salt-and-pepper noise (10% of pixels)
        if (random < 0.05) {
          value = 0; // Pepper
        } else if (random < 0.10) {
          value = 255; // Salt
        }
        testImage.ucharPtr(i, j)[0] = value;
      }
    }

    const colorImage = testImage.clone();
    const context = createProcessingContext('test.jpg', testImage, colorImage, {});

    const processor = new MedianBlur({ kSize: 5 });
    const result = processor.process(context);

    // Count pixels close to base value
    let closeToBase = 0;
    for (let i = 0; i < result.grayImage.rows; i++) {
      for (let j = 0; j < result.grayImage.cols; j++) {
        const val = result.grayImage.ucharPtr(i, j)[0];
        if (Math.abs(val - baseValue) < 30) {
          closeToBase++;
        }
      }
    }

    const ratio = closeToBase / (result.grayImage.rows * result.grayImage.cols);

    // After median blur, most pixels should be close to base value
    expect(ratio).toBeGreaterThan(0.8);

    // Cleanup
    result.grayImage.delete();
    result.coloredImage.delete();
  });
});

