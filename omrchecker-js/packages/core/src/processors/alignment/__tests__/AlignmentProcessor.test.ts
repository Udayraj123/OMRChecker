/**
 * Tests for AlignmentProcessor.
 *
 * Tests the template alignment processor.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { AlignmentProcessor } from '../AlignmentProcessor';
import { createProcessingContext } from '../../base';
const cv = global.cv;

describe('AlignmentProcessor', () => {
  let mockTemplate: any;
  let mockGrayImage: cv.Mat;
  let mockColoredImage: cv.Mat;
  let mockAlignmentImage: cv.Mat;

  beforeEach(() => {
    // Create mock images (empty Mats for testing)
    mockGrayImage = new cv.Mat();
    mockColoredImage = new cv.Mat();
    mockAlignmentImage = new cv.Mat();

    // Create mock template with alignment configuration
    mockTemplate = {
      tuningConfig: {
        outputs: {
          colored_outputs_enabled: true,
        },
      },
      alignment: {
        grayAlignmentImage: mockAlignmentImage,
        coloredAlignmentImage: mockAlignmentImage,
        margins: {
          left: 10,
          right: 10,
          top: 10,
          bottom: 10,
        },
        maxDisplacement: 20,
      },
      templateDimensions: [1000, 1500],
      fieldBlocks: [],
    };
  });

  describe('constructor', () => {
    it('should initialize with template', () => {
      const processor = new AlignmentProcessor(mockTemplate);

      expect(processor).toBeDefined();
      expect(processor.getName()).toBe('Alignment');
    });

    it('should handle template with snake_case properties', () => {
      const templateSnakeCase = {
        tuning_config: {
          outputs: {
            colored_outputs_enabled: false,
          },
        },
        alignment: {
          gray_alignment_image: mockAlignmentImage,
        },
      };

      const processor = new AlignmentProcessor(templateSnakeCase);
      expect(processor).toBeDefined();
    });
  });

  describe('getName', () => {
    it('should return "Alignment"', () => {
      const processor = new AlignmentProcessor(mockTemplate);
      expect(processor.getName()).toBe('Alignment');
    });
  });

  describe('process', () => {
    it('should process context with alignment configured', () => {
      const processor = new AlignmentProcessor(mockTemplate);
      const context = createProcessingContext(
        'test.jpg',
        mockGrayImage,
        mockColoredImage,
        mockTemplate
      );

      const result = processor.process(context);

      expect(result).toBeDefined();
      expect(result.filePath).toBe('test.jpg');
    });

    it('should skip alignment when no alignment image configured', () => {
      const templateNoAlignment = {
        ...mockTemplate,
        alignment: {
          margins: { left: 10, right: 10, top: 10, bottom: 10 },
          maxDisplacement: 20,
        },
      };

      const processor = new AlignmentProcessor(templateNoAlignment);
      const context = createProcessingContext(
        'test.jpg',
        mockGrayImage,
        mockColoredImage,
        templateNoAlignment
      );

      const result = processor.process(context);

      expect(result.grayImage).toBe(mockGrayImage);
      expect(result.coloredImage).toBe(mockColoredImage);
    });

    it('should skip alignment when gray image is null', () => {
      const processor = new AlignmentProcessor(mockTemplate);
      const context = createProcessingContext(
        'test.jpg',
        null as any,
        mockColoredImage,
        mockTemplate
      );

      const result = processor.process(context);

      expect(result.filePath).toBe('test.jpg');
    });

    it('should preserve context properties through processing', () => {
      const processor = new AlignmentProcessor(mockTemplate);
      const context = createProcessingContext(
        'test.jpg',
        mockGrayImage,
        mockColoredImage,
        mockTemplate
      );

      context.omrResponse = { Q1: 'A', Q2: 'B' };
      context.score = 85;

      const result = processor.process(context);

      expect(result.omrResponse).toEqual({ Q1: 'A', Q2: 'B' });
      expect(result.score).toBe(85);
    });

    it('should handle template with field blocks', () => {
      const templateWithBlocks = {
        ...mockTemplate,
        fieldBlocks: [
          {
            name: 'Block1',
            boundingBoxOrigin: [100, 100],
            boundingBoxDimensions: [200, 50],
            alignment: {
              maxDisplacement: 10,
            },
          },
        ],
      };

      const processor = new AlignmentProcessor(templateWithBlocks);
      const context = createProcessingContext(
        'test.jpg',
        mockGrayImage,
        mockColoredImage,
        templateWithBlocks
      );

      const result = processor.process(context);

      expect(result).toBeDefined();
      expect(result.filePath).toBe('test.jpg');
    });

    it('should handle both camelCase and snake_case alignment properties', () => {
      const templateSnakeCase = {
        tuning_config: {
          outputs: {},
        },
        alignment: {
          gray_alignment_image: mockAlignmentImage,
          margins: { left: 5, right: 5, top: 5, bottom: 5 },
          max_displacement: 15,
        },
        template_dimensions: [800, 1200],
        field_blocks: [],
      };

      const processor = new AlignmentProcessor(templateSnakeCase);
      const context = createProcessingContext(
        'test.jpg',
        mockGrayImage,
        mockColoredImage,
        templateSnakeCase
      );

      const result = processor.process(context);

      expect(result).toBeDefined();
    });
  });

  describe('integration', () => {
    it('should work in a processing pipeline', () => {
      const processor = new AlignmentProcessor(mockTemplate);

      // Simulate a pipeline with multiple contexts
      const context1 = createProcessingContext(
        'test1.jpg',
        mockGrayImage,
        mockColoredImage,
        mockTemplate
      );
      const context2 = createProcessingContext(
        'test2.jpg',
        mockGrayImage,
        mockColoredImage,
        mockTemplate
      );

      const result1 = processor.process(context1);
      const result2 = processor.process(context2);

      expect(result1.filePath).toBe('test1.jpg');
      expect(result2.filePath).toBe('test2.jpg');
    });
  });
});

