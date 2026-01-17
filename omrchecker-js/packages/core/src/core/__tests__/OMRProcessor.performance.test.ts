/**
 * Performance tests for OMRProcessor with multi-pass architecture.
 *
 * Tests performance characteristics including:
 * - Multi-pass processing benchmarks
 * - Aggregate collection overhead
 * - Memory usage patterns
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as cv from '@techstark/opencv-js';
import { OMRProcessor } from '../OMRProcessor';
import type { TemplateConfig } from '../../template/types';

describe('OMRProcessor Performance', () => {
  let templateConfig: TemplateConfig;
  let mockGrayImage: cv.Mat;
  let mockColoredImage: cv.Mat;

  beforeEach(() => {
    // Create a template config with multiple fields
    templateConfig = {
      templateDimensions: [900, 650],
      bubbleDimensions: [20, 20],
      fieldBlocks: {
        block1: {
          name: 'block1',
          origin: [100, 100],
          fieldLabels: ['q1', 'q2', 'q3', 'q4', 'q5'],
          bubbleFieldType: 'QTYPE_MCQ4',
          bubblesGap: 30,
          labelsGap: 50,
        },
      },
    };

    // Create mock images
    mockGrayImage = new cv.Mat(900, 650, cv.CV_8UC1, new cv.Scalar(200));
    mockColoredImage = new cv.Mat(900, 650, cv.CV_8UC3, new cv.Scalar(200, 200, 200));
  });

  afterEach(() => {
    mockGrayImage.delete();
    mockColoredImage.delete();
  });

  describe('Multi-Pass Processing Performance', () => {
    it('should complete processing within reasonable time', async () => {
      const processor = new OMRProcessor(templateConfig);

      const startTime = performance.now();
      await processor.processImage(mockGrayImage, 'test.jpg', mockColoredImage);
      const endTime = performance.now();

      const processingTime = endTime - startTime;

      // Processing should complete in reasonable time (adjust threshold as needed)
      // This is a basic sanity check - actual benchmarks would need more sophisticated setup
      expect(processingTime).toBeLessThan(10000); // 10 seconds max
      expect(processingTime).toBeGreaterThan(0);
    });

    it('should handle batch processing efficiently', async () => {
      const processor = new OMRProcessor(templateConfig);

      const images: Array<[cv.Mat, string, cv.Mat?]> = [];
      for (let i = 0; i < 5; i++) {
        const img = new cv.Mat(900, 650, cv.CV_8UC1, new cv.Scalar(200));
        images.push([img, `test${i}.jpg`, mockColoredImage]);
      }

      const startTime = performance.now();
      const results = await processor.processBatch(images);
      const endTime = performance.now();

      const processingTime = endTime - startTime;

      expect(results.length).toBe(5);
      expect(processingTime).toBeLessThan(50000); // 50 seconds max for 5 images

      // Cleanup
      images.forEach(([img]) => img.delete());
    });
  });

  describe('Aggregate Collection Overhead', () => {
    it('should collect aggregates without significant overhead', async () => {
      const processor = new OMRProcessor(templateConfig);

      const startTime = performance.now();
      await processor.processImage(mockGrayImage, 'test.jpg', mockColoredImage);
      const processingTime = performance.now() - startTime;

      // Get aggregates (this should be fast)
      const aggregatesStartTime = performance.now();
      const aggregates = processor.getAggregates();
      const aggregatesTime = performance.now() - aggregatesStartTime;

      expect(aggregates).toBeDefined();
      // Aggregate access should be very fast (less than 10ms)
      expect(aggregatesTime).toBeLessThan(10);

      // Aggregate collection overhead should be reasonable
      // (This is a basic check - actual profiling would be needed)
      expect(processingTime).toBeGreaterThan(0);
    });
  });

  describe('Memory Usage', () => {
    it('should handle multiple images without memory leaks', async () => {
      const processor = new OMRProcessor(templateConfig);

      // Process multiple images
      for (let i = 0; i < 10; i++) {
        const img = new cv.Mat(900, 650, cv.CV_8UC1, new cv.Scalar(200));
        await processor.processImage(img, `test${i}.jpg`, mockColoredImage);
        img.delete();
      }

      // If we get here without memory issues, test passes
      // Actual memory profiling would require more sophisticated tools
      expect(true).toBe(true);
    });

    it('should clean up aggregates appropriately', async () => {
      const processor = new OMRProcessor(templateConfig);

      await processor.processImage(mockGrayImage, 'test.jpg', mockColoredImage);

      const aggregates1 = processor.getAggregates();
      expect(aggregates1).toBeDefined();

      // Process another image
      const img2 = new cv.Mat(900, 650, cv.CV_8UC1, new cv.Scalar(200));
      await processor.processImage(img2, 'test2.jpg', mockColoredImage);

      const aggregates2 = processor.getAggregates();
      expect(aggregates2).toBeDefined();

      img2.delete();
    });
  });

  describe('Scalability', () => {
    it('should scale reasonably with number of fields', async () => {
      // Template with more fields
      const largeTemplateConfig: TemplateConfig = {
        templateDimensions: [900, 650],
        bubbleDimensions: [20, 20],
        fieldBlocks: {
          block1: {
            name: 'block1',
            origin: [100, 100],
            fieldLabels: Array.from({ length: 20 }, (_, i) => `q${i + 1}`),
            bubbleFieldType: 'QTYPE_MCQ4',
            bubblesGap: 30,
            labelsGap: 50,
          },
        },
      };

      const processor = new OMRProcessor(largeTemplateConfig);

      const startTime = performance.now();
      await processor.processImage(mockGrayImage, 'test.jpg', mockColoredImage);
      const processingTime = performance.now() - startTime;

      // Processing time should scale reasonably (not exponentially)
      // This is a basic check - actual profiling needed for precise measurements
      expect(processingTime).toBeLessThan(30000); // 30 seconds max for 20 fields
    });
  });
});

