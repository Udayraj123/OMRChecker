/**
 * Tests for PreprocessingProcessor (coordinator).
 *
 * Tests the preprocessing coordinator that runs image processors in sequence.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { PreprocessingProcessor } from '../coordinator';
import { Processor, ProcessingContext, createProcessingContext } from '../../base';
import * as cv from '@techstark/opencv-js';

// Mock preprocessor for testing
class MockPreprocessor extends Processor {
  private processCalled = false;
  private name: string;

  constructor(name: string) {
    super();
    this.name = name;
  }

  getName(): string {
    return this.name;
  }

  process(context: ProcessingContext): ProcessingContext {
    this.processCalled = true;
    // Add marker to metadata to track execution
    context.metadata[`${this.name}_processed`] = true;
    return context;
  }

  wasProcessCalled(): boolean {
    return this.processCalled;
  }
}

describe('PreprocessingProcessor', () => {
  let mockTemplate: any;
  let mockGrayImage: cv.Mat;
  let mockColoredImage: cv.Mat;

  beforeEach(() => {
    // Create mock images (empty Mats for testing)
    mockGrayImage = new cv.Mat();
    mockColoredImage = new cv.Mat();

    // Create mock template with minimal structure
    mockTemplate = {
      tuningConfig: {
        outputs: {
          colored_outputs_enabled: true,
          show_preprocessors_diff: {},
        },
      },
      templateLayout: {
        processingImageShape: [800, 1200],
        outputImageShape: null,
        preProcessors: [],
      },
    };
  });

  describe('constructor', () => {
    it('should initialize with template', () => {
      const processor = new PreprocessingProcessor(mockTemplate);

      expect(processor).toBeDefined();
      expect(processor.getName()).toBe('Preprocessing');
    });

    it('should handle template with snake_case properties', () => {
      const templateSnakeCase = {
        tuning_config: {
          outputs: {
            colored_outputs_enabled: false,
          },
        },
        template_layout: {
          processing_image_shape: [1000, 1500],
          pre_processors: [],
        },
      };

      const processor = new PreprocessingProcessor(templateSnakeCase);
      expect(processor).toBeDefined();
    });
  });

  describe('getName', () => {
    it('should return "Preprocessing"', () => {
      const processor = new PreprocessingProcessor(mockTemplate);
      expect(processor.getName()).toBe('Preprocessing');
    });
  });

  describe('process', () => {
    it('should process context through all preprocessors', () => {
      const mockPreprocessor1 = new MockPreprocessor('Processor1');
      const mockPreprocessor2 = new MockPreprocessor('Processor2');
      mockTemplate.templateLayout.preProcessors = [mockPreprocessor1, mockPreprocessor2];

      const processor = new PreprocessingProcessor(mockTemplate);
      const context = createProcessingContext(
        'test.jpg',
        mockGrayImage,
        mockColoredImage,
        mockTemplate
      );

      const result = processor.process(context);

      expect(result.metadata['Processor1_processed']).toBe(true);
      expect(result.metadata['Processor2_processed']).toBe(true);
    });

    it('should handle empty preprocessors list', () => {
      mockTemplate.templateLayout.preProcessors = [];

      const processor = new PreprocessingProcessor(mockTemplate);
      const context = createProcessingContext(
        'test.jpg',
        mockGrayImage,
        mockColoredImage,
        mockTemplate
      );

      const result = processor.process(context);

      expect(result.filePath).toBe('test.jpg');
      expect(result.grayImage).toBeDefined();
    });

    it('should handle missing template layout gracefully', () => {
      const templateWithoutLayout = {
        tuningConfig: {
          outputs: {},
        },
      };

      const processor = new PreprocessingProcessor(templateWithoutLayout);
      const context = createProcessingContext(
        'test.jpg',
        mockGrayImage,
        mockColoredImage,
        templateWithoutLayout
      );

      const result = processor.process(context);

      expect(result.filePath).toBe('test.jpg');
    });

    it('should preserve context properties through processing', () => {
      const processor = new PreprocessingProcessor(mockTemplate);
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

    it('should run preprocessors in correct order', () => {
      const executionOrder: string[] = [];

      class OrderTrackingPreprocessor extends Processor {
        constructor(private name: string, private orderArray: string[]) {
          super();
        }

        getName(): string {
          return this.name;
        }

        process(context: ProcessingContext): ProcessingContext {
          this.orderArray.push(this.name);
          return context;
        }
      }

      mockTemplate.templateLayout.preProcessors = [
        new OrderTrackingPreprocessor('First', executionOrder),
        new OrderTrackingPreprocessor('Second', executionOrder),
        new OrderTrackingPreprocessor('Third', executionOrder),
      ];

      const processor = new PreprocessingProcessor(mockTemplate);
      const context = createProcessingContext(
        'test.jpg',
        mockGrayImage,
        mockColoredImage,
        mockTemplate
      );

      processor.process(context);

      expect(executionOrder).toEqual(['First', 'Second', 'Third']);
    });

    it('should handle context updates from preprocessors', () => {
      class ContextModifyingPreprocessor extends Processor {
        constructor(private name: string) {
          super();
        }

        getName(): string {
          return this.name;
        }

        process(context: ProcessingContext): ProcessingContext {
          context.metadata[this.name] = `processed_${this.name}`;
          return context;
        }
      }

      mockTemplate.templateLayout.preProcessors = [
        new ContextModifyingPreprocessor('Proc1'),
        new ContextModifyingPreprocessor('Proc2'),
      ];

      const processor = new PreprocessingProcessor(mockTemplate);
      const context = createProcessingContext(
        'test.jpg',
        mockGrayImage,
        mockColoredImage,
        mockTemplate
      );

      const result = processor.process(context);

      expect(result.metadata['Proc1']).toBe('processed_Proc1');
      expect(result.metadata['Proc2']).toBe('processed_Proc2');
    });
  });

  describe('integration with template configuration', () => {
    it('should handle camelCase template properties', () => {
      mockTemplate.templateLayout.preProcessors = [new MockPreprocessor('Test')];

      const processor = new PreprocessingProcessor(mockTemplate);
      const context = createProcessingContext(
        'test.jpg',
        mockGrayImage,
        mockColoredImage,
        mockTemplate
      );

      const result = processor.process(context);
      expect(result.metadata['Test_processed']).toBe(true);
    });

    it('should handle snake_case template properties', () => {
      const templateSnakeCase = {
        tuning_config: {
          outputs: {
            colored_outputs_enabled: false,
          },
        },
        template_layout: {
          processing_image_shape: [1000, 1500],
          pre_processors: [new MockPreprocessor('Test')],
        },
      };

      const processor = new PreprocessingProcessor(templateSnakeCase);
      const context = createProcessingContext(
        'test.jpg',
        mockGrayImage,
        mockColoredImage,
        templateSnakeCase
      );

      const result = processor.process(context);
      expect(result.metadata['Test_processed']).toBe(true);
    });
  });
});

