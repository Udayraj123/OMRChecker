/**
 * Tests for ProcessingPipeline.
 *
 * Tests the unified pipeline interface for orchestrating processors.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { ProcessingPipeline, PipelineConfig } from '../Pipeline';
import { Processor } from '../base';
const cv = global.cv;

// Mock processor for testing
class MockProcessor extends Processor {
  private name: string;
  private shouldThrow: boolean;

  constructor(name: string, shouldThrow: boolean = false) {
    super();
    this.name = name;
    this.shouldThrow = shouldThrow;
  }

  getName(): string {
    return this.name;
  }

  process(context: ProcessingContext): ProcessingContext {
    if (this.shouldThrow) {
      throw new Error(`Error in ${this.name}`);
    }

    // Update context to show this processor was called
    context.metadata[this.name] = true;
    return context;
  }
}

// Async mock processor
class AsyncMockProcessor extends Processor {
  private name: string;

  constructor(name: string) {
    super();
    this.name = name;
  }

  getName(): string {
    return this.name;
  }

  async process(context: ProcessingContext): Promise<ProcessingContext> {
    // Simulate async work
    await new Promise((resolve) => setTimeout(resolve, 10));

    context.metadata[this.name] = true;
    return context;
  }
}

describe('ProcessingPipeline', () => {
  let mockTemplate: any;
  let mockGrayImage: cv.Mat;
  let mockColoredImage: cv.Mat;

  beforeEach(() => {
    // Create mock template
    mockTemplate = {
      tuningConfig: {
        outputs: { show_image_level: 0 },
      },
    };

    // Create mock images (empty Mats for testing)
    mockGrayImage = new cv.Mat();
    mockColoredImage = new cv.Mat();
  });

  describe('constructor', () => {
    it('should initialize with template', () => {
      const pipeline = new ProcessingPipeline(mockTemplate);

      expect(pipeline).toBeDefined();
      expect(pipeline.getProcessors()).toEqual([]);
    });

    it('should initialize with config', () => {
      const config: PipelineConfig = {
        mlModelPath: 'test.pt',
        collectTrainingData: false,
      };

      const pipeline = new ProcessingPipeline(mockTemplate, config);

      expect(pipeline).toBeDefined();
    });
  });

  describe('processor management', () => {
    it('should add processors', () => {
      const pipeline = new ProcessingPipeline(mockTemplate);
      const processor1 = new MockProcessor('Processor1');
      const processor2 = new MockProcessor('Processor2');

      pipeline.addProcessor(processor1);
      pipeline.addProcessor(processor2);

      expect(pipeline.getProcessors()).toHaveLength(2);
      expect(pipeline.getProcessorNames()).toEqual(['Processor1', 'Processor2']);
    });

    it('should remove processors by name', () => {
      const pipeline = new ProcessingPipeline(mockTemplate);
      const processor1 = new MockProcessor('Processor1');
      const processor2 = new MockProcessor('Processor2');
      const processor3 = new MockProcessor('Processor3');

      pipeline.addProcessor(processor1);
      pipeline.addProcessor(processor2);
      pipeline.addProcessor(processor3);

      pipeline.removeProcessor('Processor2');

      expect(pipeline.getProcessors()).toHaveLength(2);
      expect(pipeline.getProcessorNames()).toEqual(['Processor1', 'Processor3']);
    });

    it('should get processor by name', () => {
      const pipeline = new ProcessingPipeline(mockTemplate);
      const processor = new MockProcessor('TestProcessor');

      pipeline.addProcessor(processor);

      const retrieved = pipeline.getProcessorByName('TestProcessor');
      expect(retrieved).toBe(processor);
    });

    it('should return undefined for non-existent processor', () => {
      const pipeline = new ProcessingPipeline(mockTemplate);

      const retrieved = pipeline.getProcessorByName('NonExistent');
      expect(retrieved).toBeUndefined();
    });

    it('should clear all processors', () => {
      const pipeline = new ProcessingPipeline(mockTemplate);
      const processor1 = new MockProcessor('Processor1');
      const processor2 = new MockProcessor('Processor2');

      pipeline.addProcessor(processor1);
      pipeline.addProcessor(processor2);

      pipeline.clearProcessors();

      expect(pipeline.getProcessors()).toHaveLength(0);
    });
  });

  describe('processFile', () => {
    it('should process file through all processors', async () => {
      const pipeline = new ProcessingPipeline(mockTemplate);
      const processor1 = new MockProcessor('Processor1');
      const processor2 = new MockProcessor('Processor2');
      const processor3 = new MockProcessor('Processor3');

      pipeline.addProcessor(processor1);
      pipeline.addProcessor(processor2);
      pipeline.addProcessor(processor3);

      const result = await pipeline.processFile('test.jpg', mockGrayImage, mockColoredImage);

      // All processors should have been called
      expect(result.metadata['Processor1']).toBe(true);
      expect(result.metadata['Processor2']).toBe(true);
      expect(result.metadata['Processor3']).toBe(true);
    });

    it('should handle async processors', async () => {
      const pipeline = new ProcessingPipeline(mockTemplate);
      const asyncProcessor = new AsyncMockProcessor('AsyncProcessor');
      const syncProcessor = new MockProcessor('SyncProcessor');

      pipeline.addProcessor(asyncProcessor);
      pipeline.addProcessor(syncProcessor);

      const result = await pipeline.processFile('test.jpg', mockGrayImage, mockColoredImage);

      // Both processors should have been called
      expect(result.metadata['AsyncProcessor']).toBe(true);
      expect(result.metadata['SyncProcessor']).toBe(true);
    });

    it('should preserve file path in context', async () => {
      const pipeline = new ProcessingPipeline(mockTemplate);
      const filePath = '/path/to/test.jpg';

      const result = await pipeline.processFile(filePath, mockGrayImage, mockColoredImage);

      expect(result.filePath).toBe(filePath);
    });

    it('should preserve images in context', async () => {
      const pipeline = new ProcessingPipeline(mockTemplate);

      const result = await pipeline.processFile('test.jpg', mockGrayImage, mockColoredImage);

      expect(result.grayImage).toBe(mockGrayImage);
      expect(result.coloredImage).toBe(mockColoredImage);
    });

    it('should throw error if processor fails', async () => {
      const pipeline = new ProcessingPipeline(mockTemplate);
      const goodProcessor = new MockProcessor('GoodProcessor');
      const badProcessor = new MockProcessor('BadProcessor', true);

      pipeline.addProcessor(goodProcessor);
      pipeline.addProcessor(badProcessor);

      await expect(
        pipeline.processFile('test.jpg', mockGrayImage, mockColoredImage)
      ).rejects.toThrow('Error in BadProcessor');
    });

    it('should process in correct order', async () => {
      const pipeline = new ProcessingPipeline(mockTemplate);
      const order: string[] = [];

      class OrderTrackingProcessor extends Processor {
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

      pipeline.addProcessor(new OrderTrackingProcessor('First', order));
      pipeline.addProcessor(new OrderTrackingProcessor('Second', order));
      pipeline.addProcessor(new OrderTrackingProcessor('Third', order));

      await pipeline.processFile('test.jpg', mockGrayImage, mockColoredImage);

      expect(order).toEqual(['First', 'Second', 'Third']);
    });

    it('should handle empty pipeline', async () => {
      const pipeline = new ProcessingPipeline(mockTemplate);

      const result = await pipeline.processFile('test.jpg', mockGrayImage, mockColoredImage);

      // Should still return valid context
      expect(result.filePath).toBe('test.jpg');
      expect(result.omrResponse).toEqual({});
    });
  });

  describe('context flow', () => {
    it('should pass context through processors', async () => {
      const pipeline = new ProcessingPipeline(mockTemplate);

      class ContextModifyingProcessor extends Processor {
        constructor(private name: string, private key: string, private value: any) {
          super();
        }

        getName(): string {
          return this.name;
        }

        process(context: ProcessingContext): ProcessingContext {
          context.metadata[this.key] = this.value;
          return context;
        }
      }

      pipeline.addProcessor(new ContextModifyingProcessor('Proc1', 'key1', 'value1'));
      pipeline.addProcessor(new ContextModifyingProcessor('Proc2', 'key2', 'value2'));

      const result = await pipeline.processFile('test.jpg', mockGrayImage, mockColoredImage);

      expect(result.metadata['key1']).toBe('value1');
      expect(result.metadata['key2']).toBe('value2');
    });

    it('should accumulate changes from all processors', async () => {
      const pipeline = new ProcessingPipeline(mockTemplate);

      class AccumulatorProcessor extends Processor {
        constructor(private name: string, private value: number) {
          super();
        }

        getName(): string {
          return this.name;
        }

        process(context: ProcessingContext): ProcessingContext {
          const current = (context.metadata['sum'] as number) || 0;
          context.metadata['sum'] = current + this.value;
          return context;
        }
      }

      pipeline.addProcessor(new AccumulatorProcessor('Add10', 10));
      pipeline.addProcessor(new AccumulatorProcessor('Add20', 20));
      pipeline.addProcessor(new AccumulatorProcessor('Add30', 30));

      const result = await pipeline.processFile('test.jpg', mockGrayImage, mockColoredImage);

      expect(result.metadata['sum']).toBe(60);
    });
  });
});

