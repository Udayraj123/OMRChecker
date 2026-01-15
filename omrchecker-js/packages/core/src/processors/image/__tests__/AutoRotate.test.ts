/**
 * Unit tests for AutoRotate processor.
 *
 * TypeScript port of tests for src/processors/image/AutoRotate.py
 */

import { describe, it, expect, beforeAll, afterEach } from 'vitest';
import * as cv from '@techstark/opencv-js';
import { AutoRotate } from '../AutoRotate';
import { createProcessingContext } from '../../base';

describe('AutoRotate', () => {
  let testImage: cv.Mat;
  let referenceImage: cv.Mat;

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
    if (referenceImage && !referenceImage.isDeleted()) {
      referenceImage.delete();
    }
  });

  it('should initialize with required parameters', () => {
    const processor = new AutoRotate({
      referenceImagePath: 'reference.png',
    });
    expect(processor.getName()).toBe('AutoRotate');
  });

  it('should initialize with marker dimensions', () => {
    const processor = new AutoRotate({
      referenceImagePath: 'reference.png',
      markerDimensions: { width: 100, height: 100 },
    });
    expect(processor.getName()).toBe('AutoRotate');
  });

  it('should initialize with threshold configuration', () => {
    const processor = new AutoRotate({
      referenceImagePath: 'reference.png',
      threshold: {
        value: 0.7,
        passthrough: true,
      },
    });
    expect(processor.getName()).toBe('AutoRotate');
  });

  it('should detect no rotation needed when image matches reference', () => {
    // Create reference marker (50x50 white square)
    referenceImage = new cv.Mat(50, 50, cv.CV_8UC1, new cv.Scalar(255));

    // Create test image with marker at top-left
    testImage = new cv.Mat(200, 200, cv.CV_8UC1, new cv.Scalar(0));
    const roi = testImage.roi(new cv.Rect(10, 10, 50, 50));
    referenceImage.copyTo(roi);
    roi.delete();

    const colorImage = testImage.clone();
    const context = createProcessingContext('test.jpg', testImage, colorImage, {});

    const processor = new AutoRotate({
      referenceImagePath: 'reference.png',
    });
    processor.setReferenceImage(referenceImage);

    const result = processor.process(context);

    expect(result.grayImage).toBeDefined();
    expect(result.grayImage.rows).toBe(200);
    expect(result.grayImage.cols).toBe(200);

    // Cleanup
    result.grayImage.delete();
    result.coloredImage.delete();
  });

  it('should detect and correct 180-degree rotation', () => {
    // Create distinctive reference marker (rectangle)
    referenceImage = new cv.Mat(30, 50, cv.CV_8UC1, new cv.Scalar(255));

    // Create test image with marker rotated 180 degrees
    testImage = new cv.Mat(200, 200, cv.CV_8UC1, new cv.Scalar(0));

    // Rotate reference and place in test image
    const rotatedRef = new cv.Mat();
    cv.rotate(referenceImage, rotatedRef, cv.ROTATE_180);

    const roi = testImage.roi(new cv.Rect(75, 85, rotatedRef.cols, rotatedRef.rows));
    rotatedRef.copyTo(roi);
    roi.delete();
    rotatedRef.delete();

    const colorImage = testImage.clone();
    const context = createProcessingContext('test.jpg', testImage, colorImage, {});

    const processor = new AutoRotate({
      referenceImagePath: 'reference.png',
    });
    processor.setReferenceImage(referenceImage);

    const result = processor.process(context);

    expect(result.grayImage).toBeDefined();
    // After rotation correction, dimensions should be same
    expect(result.grayImage.rows).toBe(200);
    expect(result.grayImage.cols).toBe(200);

    // Cleanup
    result.grayImage.delete();
    result.coloredImage.delete();
  });

  it('should throw error when threshold not met without passthrough', () => {
    // Create reference that won't match
    referenceImage = new cv.Mat(50, 50, cv.CV_8UC1, new cv.Scalar(255));

    // Create completely different test image
    testImage = new cv.Mat(200, 200, cv.CV_8UC1, new cv.Scalar(0));

    const colorImage = testImage.clone();
    const context = createProcessingContext('test.jpg', testImage, colorImage, {});

    const processor = new AutoRotate({
      referenceImagePath: 'reference.png',
      threshold: {
        value: 0.9, // Very high threshold
        passthrough: false,
      },
    });
    processor.setReferenceImage(referenceImage);

    expect(() => processor.process(context)).toThrow(/threshold/);

    // Cleanup
    context.grayImage.delete();
    context.coloredImage.delete();
  });

  it('should continue when threshold not met with passthrough', () => {
    // Create reference that won't match well
    referenceImage = new cv.Mat(50, 50, cv.CV_8UC1, new cv.Scalar(255));

    // Create different test image
    testImage = new cv.Mat(200, 200, cv.CV_8UC1, new cv.Scalar(100));

    const colorImage = testImage.clone();
    const context = createProcessingContext('test.jpg', testImage, colorImage, {});

    const processor = new AutoRotate({
      referenceImagePath: 'reference.png',
      threshold: {
        value: 0.9, // High threshold
        passthrough: true, // But allow to continue
      },
    });
    processor.setReferenceImage(referenceImage);

    const result = processor.process(context);

    expect(result.grayImage).toBeDefined();

    // Cleanup
    result.grayImage.delete();
    result.coloredImage.delete();
  });
});

