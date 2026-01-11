/**
 * Tests for image filter processors.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import * as cv from '@techstark/opencv-js';
import { GaussianBlur, MedianBlur, Contrast } from '../filters';
import { createProcessingContext, ProcessingContext } from '../../base';

describe('Image Filter Processors', () => {
  let context: ProcessingContext;
  let testImage: cv.Mat;

  beforeEach(() => {
    // Create a test grayscale image (100x100)
    testImage = cv.Mat.zeros(100, 100, cv.CV_8UC1);

    // Add some noise/pattern
    for (let i = 0; i < 100; i += 10) {
      for (let j = 0; j < 100; j += 10) {
        testImage.ucharPtr(i, j)[0] = 255;
      }
    }

    context = createProcessingContext('test.jpg', testImage.clone(), testImage.clone(), {});
  });

  describe('GaussianBlur', () => {
    it('should apply Gaussian blur with default options', () => {
      const processor = new GaussianBlur();

      const result = processor.process(context);

      expect(result).toBeDefined();
      expect(result.grayImage).toBeDefined();
      expect(result.grayImage.rows).toBe(100);
      expect(result.grayImage.cols).toBe(100);
    });

    it('should apply Gaussian blur with custom kernel size', () => {
      const processor = new GaussianBlur({ kSize: [5, 5], sigmaX: 1 });

      const result = processor.process(context);

      expect(result).toBeDefined();
      expect(processor.getName()).toBe('GaussianBlur');
    });

    it('should preserve context properties', () => {
      const processor = new GaussianBlur();
      context.metadata['test'] = 'value';

      const result = processor.process(context);

      expect(result.filePath).toBe('test.jpg');
      expect(result.metadata['test']).toBe('value');
    });
  });

  describe('MedianBlur', () => {
    it('should apply median blur with default options', () => {
      const processor = new MedianBlur();

      const result = processor.process(context);

      expect(result).toBeDefined();
      expect(result.grayImage).toBeDefined();
      expect(result.grayImage.rows).toBe(100);
      expect(result.grayImage.cols).toBe(100);
    });

    it('should apply median blur with custom kernel size', () => {
      const processor = new MedianBlur({ kSize: 7 });

      const result = processor.process(context);

      expect(result).toBeDefined();
      expect(processor.getName()).toBe('MedianBlur');
    });

    it('should handle odd kernel sizes only', () => {
      // Median blur requires odd kernel size
      const processor = new MedianBlur({ kSize: 5 });

      const result = processor.process(context);

      expect(result.grayImage).toBeDefined();
    });
  });

  describe('Contrast', () => {
    it('should apply manual contrast adjustment', () => {
      const processor = new Contrast({ mode: 'manual', alpha: 1.5, beta: 10 });

      const result = processor.process(context);

      expect(result).toBeDefined();
      expect(result.grayImage).toBeDefined();
      expect(processor.getName()).toBe('Contrast');
    });

    it('should apply auto contrast adjustment', () => {
      const processor = new Contrast({ mode: 'auto', clipPercentage: 1 });

      const result = processor.process(context);

      expect(result).toBeDefined();
      expect(result.grayImage).toBeDefined();
    });

    it('should use default manual mode', () => {
      const processor = new Contrast();

      const result = processor.process(context);

      expect(result).toBeDefined();
    });

    it('should preserve image dimensions', () => {
      const processor = new Contrast({ mode: 'auto' });

      const result = processor.process(context);

      expect(result.grayImage.rows).toBe(100);
      expect(result.grayImage.cols).toBe(100);
    });
  });

  describe('Filter chaining', () => {
    it('should chain multiple filters', () => {
      const blur = new GaussianBlur({ kSize: [3, 3] });
      const median = new MedianBlur({ kSize: 5 });
      const contrast = new Contrast({ mode: 'manual', alpha: 1.2 });

      let result = context;
      result = blur.process(result);
      result = median.process(result);
      result = contrast.process(result);

      expect(result).toBeDefined();
      expect(result.grayImage).toBeDefined();
    });

    it('should maintain context through filter chain', () => {
      context.metadata['initial'] = 'value';

      const blur = new GaussianBlur();
      const contrast = new Contrast();

      let result = context;
      result = blur.process(result);
      result = contrast.process(result);

      expect(result.metadata['initial']).toBe('value');
    });
  });
});

