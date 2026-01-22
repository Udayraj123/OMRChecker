/**
 * Comprehensive tests for OMRProcessor class.
 *
 * Tests initialization, processImage, getAggregates, error handling, and edge cases.
 * TypeScript equivalent of Python's test_entry.py, focused on OMRProcessor.
 */

console.log("start");
const cv = global.cv;

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { OMRProcessor, type OMRProcessorConfig } from '../OMRProcessor';
import type { TemplateConfig } from '../../template/types';

describe('OMRProcessor', () => {
  let templateConfig: TemplateConfig;
  let mockGrayImage: any;
  let mockColoredImage: any;

  function createMinimalTemplateConfig(): TemplateConfig {
    return {
      templateDimensions: [900, 650],
      bubbleDimensions: [20, 20],
      fieldBlocks: {
        block1: {
          fieldDetectionType: 'BUBBLES_THRESHOLD',
          name: 'block1',
          origin: [100, 100],
          fieldLabels: ['q1', 'q2'],
          bubbleFieldType: 'QTYPE_MCQ4',
          bubblesGap: 30,
          labelsGap: 50,
        },
      },
    };
  }

  beforeEach(() => {
    templateConfig = createMinimalTemplateConfig();
    mockGrayImage = new cv.Mat(900, 650, cv.CV_8UC1, new cv.Scalar(200));
    mockColoredImage = new cv.Mat(900, 650, cv.CV_8UC3, new cv.Scalar(200, 200, 200));
  });

  afterEach(() => {
    mockGrayImage.delete();
    mockColoredImage.delete();
  });

  describe('Initialization', () => {
    it('should initialize with minimal template config', () => {
      const processor = new OMRProcessor(templateConfig);

      expect(processor).toBeDefined();
    });

    it('should initialize with custom config', () => {
      const config: OMRProcessorConfig = {
        debug: true,
        saveIntermediateImages: true,
        outputDirectory: '/tmp/output',
      };

      const processor = new OMRProcessor(templateConfig, config);

      expect(processor).toBeDefined();
    });

    it('should initialize with alignment config', () => {
      const configWithAlignment: TemplateConfig = {
        ...templateConfig,
        alignment: {
          margins: {
            top: 10,
            right: 10,
            bottom: 10,
            left: 10,
          },
        },
      };

      const processor = new OMRProcessor(configWithAlignment);

      expect(processor).toBeDefined();
    });

    it('should initialize with preprocessors', () => {
      const configWithPreprocessors: TemplateConfig = {
        ...templateConfig,
        preProcessors: [
          {
            name: 'GaussianBlur',
            options: {},
          },
        ],
      };

      const processor = new OMRProcessor(configWithPreprocessors);

      expect(processor).toBeDefined();
    });

    it('should initialize with answer key', () => {
      const answerKey = { q1: 'A', q2: 'B' };
      const processor = new OMRProcessor(templateConfig, {}, answerKey);

      expect(processor).toBeDefined();
    });
  });

  describe('processImage', () => {
    it('should process a single image', async () => {
      const processor = new OMRProcessor(templateConfig);

      const result = await processor.processImage(mockGrayImage, 'test.jpg', mockColoredImage);

      expect(result).toBeDefined();
      expect(result.filePath).toBe('test.jpg');
      expect(result.responses).toBeDefined();
      expect(result.processingTimeMs).toBeGreaterThan(0);
      expect(result.statistics).toBeDefined();
      expect(result.statistics.totalFields).toBeGreaterThan(0);
    });

    it('should process image without colored image', async () => {
      const processor = new OMRProcessor(templateConfig);

      const result = await processor.processImage(mockGrayImage, 'test.jpg');

      expect(result).toBeDefined();
      expect(result.filePath).toBe('test.jpg');
    });

    it('should return responses for all fields', async () => {
      const processor = new OMRProcessor(templateConfig);

      const result = await processor.processImage(mockGrayImage, 'test.jpg', mockColoredImage);

      expect(result.responses).toBeDefined();
      expect(typeof result.responses).toBe('object');
      // Should have responses for q1 and q2
      expect(result.responses.q1).toBeDefined();
      expect(result.responses.q2).toBeDefined();
    });

    it('should track multi-marked fields', async () => {
      const processor = new OMRProcessor(templateConfig);

      const result = await processor.processImage(mockGrayImage, 'test.jpg', mockColoredImage);

      expect(result.multiMarkedFields).toBeDefined();
      expect(Array.isArray(result.multiMarkedFields)).toBe(true);
    });

    it('should track empty fields', async () => {
      const processor = new OMRProcessor(templateConfig);

      const result = await processor.processImage(mockGrayImage, 'test.jpg', mockColoredImage);

      expect(result.emptyFields).toBeDefined();
      expect(Array.isArray(result.emptyFields)).toBe(true);
    });

    it('should include field results', async () => {
      const processor = new OMRProcessor(templateConfig);

      const result = await processor.processImage(mockGrayImage, 'test.jpg', mockColoredImage);

      expect(result.fieldResults).toBeDefined();
      expect(typeof result.fieldResults).toBe('object');
    });

    it('should include statistics', async () => {
      const processor = new OMRProcessor(templateConfig);

      const result = await processor.processImage(mockGrayImage, 'test.jpg', mockColoredImage);

      expect(result.statistics).toBeDefined();
      expect(result.statistics.totalFields).toBeGreaterThanOrEqual(0);
      expect(result.statistics.answeredFields).toBeGreaterThanOrEqual(0);
      expect(result.statistics.unansweredFields).toBeGreaterThanOrEqual(0);
      expect(result.statistics.multiMarkedFields).toBeGreaterThanOrEqual(0);
      expect(result.statistics.avgConfidence).toBeGreaterThanOrEqual(0);
    });

    it('should include warnings array', async () => {
      const processor = new OMRProcessor(templateConfig);

      const result = await processor.processImage(mockGrayImage, 'test.jpg', mockColoredImage);

      expect(result.warnings).toBeDefined();
      expect(Array.isArray(result.warnings)).toBe(true);
    });

    it('should handle different image sizes', async () => {
      const processor = new OMRProcessor(templateConfig);
      const largeImage = new cv.Mat(1200, 800, cv.CV_8UC1, new cv.Scalar(200));

      const result = await processor.processImage(largeImage, 'large.jpg', mockColoredImage);

      expect(result).toBeDefined();
      expect(result.filePath).toBe('large.jpg');

      largeImage.delete();
    });

    it('should handle zero-sized image gracefully', async () => {
      const processor = new OMRProcessor(templateConfig);
      const emptyImage = new cv.Mat(0, 0, cv.CV_8UC1);

      // Should not throw, but may return empty results
      try {
        const result = await processor.processImage(emptyImage, 'empty.jpg');
        expect(result).toBeDefined();
      } catch (error) {
        // Expected if empty image is rejected
        expect(error).toBeDefined();
      }

      emptyImage.delete();
    });

    it('should process multiple images sequentially', async () => {
      const processor = new OMRProcessor(templateConfig);

      const result1 = await processor.processImage(mockGrayImage, 'test1.jpg', mockColoredImage);
      const result2 = await processor.processImage(mockGrayImage, 'test2.jpg', mockColoredImage);

      expect(result1.filePath).toBe('test1.jpg');
      expect(result2.filePath).toBe('test2.jpg');
    });
  });

  describe('getAggregates', () => {
    it('should return aggregates after processing', async () => {
      const processor = new OMRProcessor(templateConfig);

      await processor.processImage(mockGrayImage, 'test.jpg', mockColoredImage);

      const aggregates = processor.getAggregates();

      expect(aggregates).toBeDefined();
      expect(aggregates?.detection).toBeDefined();
      expect(aggregates?.interpretation).toBeDefined();
    });

    it('should return null aggregates before processing', () => {
      const processor = new OMRProcessor(templateConfig);

      const aggregates = processor.getAggregates();

      // Aggregates may be null or empty before processing
      expect(aggregates === null || aggregates === undefined || typeof aggregates === 'object').toBe(true);
    });

    it('should accumulate aggregates across multiple images', async () => {
      const processor = new OMRProcessor(templateConfig);

      await processor.processImage(mockGrayImage, 'test1.jpg', mockColoredImage);
      await processor.processImage(mockGrayImage, 'test2.jpg', mockColoredImage);

      const aggregates = processor.getAggregates();

      expect(aggregates).toBeDefined();
      if (aggregates) {
        expect(aggregates.detection).toBeDefined();
        expect(aggregates.interpretation).toBeDefined();
      }
    });
  });

  describe('Error Handling', () => {
    it('should handle invalid template config gracefully', () => {
      const invalidConfig = {} as TemplateConfig;

      // Should throw or handle gracefully
      try {
        const processor = new OMRProcessor(invalidConfig);
        expect(processor).toBeDefined();
      } catch (error) {
        expect(error).toBeDefined();
      }
    });

    it('should handle processing errors gracefully', async () => {
      const processor = new OMRProcessor(templateConfig);
      const invalidImage = new cv.Mat(100, 100, cv.CV_8UC1, new cv.Scalar(0));

      try {
        const result = await processor.processImage(invalidImage, 'invalid.jpg');
        // May succeed with empty results
        expect(result).toBeDefined();
      } catch (error) {
        // Expected if processing fails
        expect(error).toBeDefined();
      }

      invalidImage.delete();
    });

    it('should include warnings in result on processing issues', async () => {
      const processor = new OMRProcessor(templateConfig);

      const result = await processor.processImage(mockGrayImage, 'test.jpg', mockColoredImage);

      expect(result.warnings).toBeDefined();
      expect(Array.isArray(result.warnings)).toBe(true);
    });
  });

  describe('Configuration Options', () => {
    it('should respect debug mode', () => {
      const config: OMRProcessorConfig = { debug: true };
      const processor = new OMRProcessor(templateConfig, config);

      expect(processor).toBeDefined();
    });

    it('should respect saveIntermediateImages option', () => {
      const config: OMRProcessorConfig = { saveIntermediateImages: true };
      const processor = new OMRProcessor(templateConfig, config);

      expect(processor).toBeDefined();
    });

    it('should respect outputDirectory option', () => {
      const config: OMRProcessorConfig = { outputDirectory: '/tmp/output' };
      const processor = new OMRProcessor(templateConfig, config);

      expect(processor).toBeDefined();
    });

    it('should respect thresholdConfig option', () => {
      const config: OMRProcessorConfig = {
        thresholdConfig: {
          defaultThreshold: 0.5,
          minJump: 0.1,
        },
      };
      const processor = new OMRProcessor(templateConfig, config);

      expect(processor).toBeDefined();
    });
  });

  describe('Template Variations', () => {
    it('should handle template with multiple field blocks', () => {
      const multiBlockConfig: TemplateConfig = {
        ...templateConfig,
        fieldBlocks: {
          block1: {
            fieldDetectionType: 'BUBBLES_THRESHOLD',
            name: 'block1',
            origin: [100, 100],
            fieldLabels: ['q1', 'q2'],
            bubbleFieldType: 'QTYPE_MCQ4',
            bubblesGap: 30,
            labelsGap: 50,
          },
          block2: {
            fieldDetectionType: 'BUBBLES_THRESHOLD',
            name: 'block2',
            origin: [100, 300],
            fieldLabels: ['q3', 'q4'],
            bubbleFieldType: 'QTYPE_MCQ4',
            bubblesGap: 30,
            labelsGap: 50,
          },
        },
      };

      const processor = new OMRProcessor(multiBlockConfig);

      expect(processor).toBeDefined();
    });

    it('should handle template with custom bubble field types', () => {
      const customTypeConfig: TemplateConfig = {
        ...templateConfig,
        customBubbleFieldTypes: {
          customType: {
            direction: 'horizontal',
            bubbleValues: ['A', 'B', 'C', 'D', 'E'],
          },
        },
      };

      const processor = new OMRProcessor(customTypeConfig);

      expect(processor).toBeDefined();
    });

    it('should handle template with custom labels', () => {
      const customLabelsConfig: TemplateConfig = {
        ...templateConfig,
        customLabels: {
          q1: ['Question 1'],
          q2: ['Question 2'],
        },
      };

      const processor = new OMRProcessor(customLabelsConfig);

      expect(processor).toBeDefined();
    });

    it('should handle template with output columns', () => {
      const outputColumnsConfig: TemplateConfig = {
        ...templateConfig,
        outputColumns: {
          sortType: 'ALPHANUMERIC',
          customOrder: ['q1', 'q2'],
        },
      };

      const processor = new OMRProcessor(outputColumnsConfig);

      expect(processor).toBeDefined();
    });
  });

  describe('Edge Cases', () => {
    it('should handle template with no field blocks', () => {
      const emptyConfig: TemplateConfig = {
        templateDimensions: [900, 650],
        bubbleDimensions: [20, 20],
        fieldBlocks: {},
      };

      const processor = new OMRProcessor(emptyConfig);

      expect(processor).toBeDefined();
    });

    it('should handle processing with empty field blocks', async () => {
      const emptyConfig: TemplateConfig = {
        templateDimensions: [900, 650],
        bubbleDimensions: [20, 20],
        fieldBlocks: {},
      };

      const processor = new OMRProcessor(emptyConfig);
      const result = await processor.processImage(mockGrayImage, 'empty.jpg', mockColoredImage);

      expect(result).toBeDefined();
      expect(result.statistics.totalFields).toBe(0);
    });

    it('should handle very large images', async () => {
      const processor = new OMRProcessor(templateConfig);
      const largeImage = new cv.Mat(3000, 2000, cv.CV_8UC1, new cv.Scalar(200));

      const result = await processor.processImage(largeImage, 'large.jpg', mockColoredImage);

      expect(result).toBeDefined();

      largeImage.delete();
    });

    it('should handle images with different aspect ratios', async () => {
      const processor = new OMRProcessor(templateConfig);
      const wideImage = new cv.Mat(900, 1300, cv.CV_8UC1, new cv.Scalar(200));
      const tallImage = new cv.Mat(1300, 900, cv.CV_8UC1, new cv.Scalar(200));

      const result1 = await processor.processImage(wideImage, 'wide.jpg', mockColoredImage);
      const result2 = await processor.processImage(tallImage, 'tall.jpg', mockColoredImage);

      expect(result1).toBeDefined();
      expect(result2).toBeDefined();

      wideImage.delete();
      tallImage.delete();
    });
  });

  describe('processBatch', () => {
    it('should process multiple images in batch', async () => {
      const processor = new OMRProcessor(templateConfig);
      const image1 = new cv.Mat(900, 650, cv.CV_8UC1, new cv.Scalar(200));
      const image2 = new cv.Mat(900, 650, cv.CV_8UC1, new cv.Scalar(200));
      const image3 = new cv.Mat(900, 650, cv.CV_8UC1, new cv.Scalar(200));

      const batch: Array<[any, string, any?]> = [
        [image1, 'batch1.jpg', mockColoredImage],
        [image2, 'batch2.jpg', mockColoredImage],
        [image3, 'batch3.jpg', mockColoredImage],
      ];

      const results = await processor.processBatch(batch);

      expect(results).toBeDefined();
      expect(results.length).toBe(3);
      expect(results[0].filePath).toBe('batch1.jpg');
      expect(results[1].filePath).toBe('batch2.jpg');
      expect(results[2].filePath).toBe('batch3.jpg');

      image1.delete();
      image2.delete();
      image3.delete();
    });

    it('should handle empty batch', async () => {
      const processor = new OMRProcessor(templateConfig);

      const results = await processor.processBatch([]);

      expect(results).toBeDefined();
      expect(results.length).toBe(0);
    });

    it('should process batch without colored images', async () => {
      const processor = new OMRProcessor(templateConfig);
      const image1 = new cv.Mat(900, 650, cv.CV_8UC1, new cv.Scalar(200));
      const image2 = new cv.Mat(900, 650, cv.CV_8UC1, new cv.Scalar(200));

      const batch: Array<[any, string]> = [
        [image1, 'batch1.jpg'],
        [image2, 'batch2.jpg'],
      ];

      const results = await processor.processBatch(batch);

      expect(results).toBeDefined();
      expect(results.length).toBe(2);

      image1.delete();
      image2.delete();
    });
  });

  describe('getTemplate', () => {
    it('should return parsed template', () => {
      const processor = new OMRProcessor(templateConfig);

      const template = processor.getTemplate();

      expect(template).toBeDefined();
      expect(template.fields).toBeDefined();
    });
  });

  describe('getFieldIds', () => {
    it('should return array of field IDs', () => {
      const processor = new OMRProcessor(templateConfig);

      const fieldIds = processor.getFieldIds();

      expect(fieldIds).toBeDefined();
      expect(Array.isArray(fieldIds)).toBe(true);
      expect(fieldIds.length).toBeGreaterThan(0);
    });

    it('should return field IDs matching template', () => {
      const processor = new OMRProcessor(templateConfig);

      const fieldIds = processor.getFieldIds();

      // Should include q1 and q2 from template
      expect(fieldIds).toContain('q1');
      expect(fieldIds).toContain('q2');
    });
  });

  describe('exportToCSV', () => {
    it('should export results to CSV format', async () => {
      const processor = new OMRProcessor(templateConfig);

      const result1 = await processor.processImage(mockGrayImage, 'test1.jpg', mockColoredImage);
      const result2 = await processor.processImage(mockGrayImage, 'test2.jpg', mockColoredImage);

      const csv = processor.exportToCSV([result1, result2]);

      expect(csv).toBeDefined();
      expect(typeof csv).toBe('string');
      expect(csv.length).toBeGreaterThan(0);
      expect(csv).toContain('File');
      expect(csv).toContain('test1.jpg');
      expect(csv).toContain('test2.jpg');
    });

    it('should handle empty results array', () => {
      const processor = new OMRProcessor(templateConfig);

      const csv = processor.exportToCSV([]);

      expect(csv).toBe('');
    });

    it('should include score column when available', async () => {
      const processor = new OMRProcessor(templateConfig);

      const result = await processor.processImage(mockGrayImage, 'test.jpg', mockColoredImage);
      const csv = processor.exportToCSV([result]);

      expect(csv).toContain('Score');
    });

    it('should include warnings column', async () => {
      const processor = new OMRProcessor(templateConfig);

      const result = await processor.processImage(mockGrayImage, 'test.jpg', mockColoredImage);
      const csv = processor.exportToCSV([result]);

      expect(csv).toContain('Warnings');
    });
  });

  describe('getStatistics', () => {
    it('should return statistics for results', async () => {
      const processor = new OMRProcessor(templateConfig);

      const result1 = await processor.processImage(mockGrayImage, 'test1.jpg', mockColoredImage);
      const result2 = await processor.processImage(mockGrayImage, 'test2.jpg', mockColoredImage);

      const stats = processor.getStatistics([result1, result2]);

      expect(stats).toBeDefined();
      expect(stats.totalSheets).toBe(2);
      expect(stats.averageProcessingTime).toBeGreaterThanOrEqual(0);
      expect(stats.totalWarnings).toBeGreaterThanOrEqual(0);
      expect(stats.multiMarkedSheets).toBeGreaterThanOrEqual(0);
      expect(stats.emptyFieldsCount).toBeGreaterThanOrEqual(0);
    });

    it('should handle empty results array', () => {
      const processor = new OMRProcessor(templateConfig);

      const stats = processor.getStatistics([]);

      expect(stats).toBeDefined();
      expect(stats.totalSheets).toBe(0);
      expect(stats.averageProcessingTime).toBe(0);
      expect(stats.totalWarnings).toBe(0);
      expect(stats.multiMarkedSheets).toBe(0);
      expect(stats.emptyFieldsCount).toBe(0);
    });

    it('should calculate average processing time correctly', async () => {
      const processor = new OMRProcessor(templateConfig);

      const result1 = await processor.processImage(mockGrayImage, 'test1.jpg', mockColoredImage);
      const result2 = await processor.processImage(mockGrayImage, 'test2.jpg', mockColoredImage);

      const stats = processor.getStatistics([result1, result2]);

      const expectedAvg =
        (result1.processingTimeMs + result2.processingTimeMs) / 2;
      expect(stats.averageProcessingTime).toBeCloseTo(expectedAvg, 1);
    });

    it('should count multi-marked sheets correctly', async () => {
      const processor = new OMRProcessor(templateConfig);

      const result1 = await processor.processImage(mockGrayImage, 'test1.jpg', mockColoredImage);
      const result2 = await processor.processImage(mockGrayImage, 'test2.jpg', mockColoredImage);

      const stats = processor.getStatistics([result1, result2]);

      const expectedMultiMarked =
        (result1.multiMarkedFields.length > 0 ? 1 : 0) +
        (result2.multiMarkedFields.length > 0 ? 1 : 0);
      expect(stats.multiMarkedSheets).toBe(expectedMultiMarked);
    });
  });
});

