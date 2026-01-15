/**
 * Tests for ImageTemplatePreprocessor base class.
 *
 * Tests the base functionality for image preprocessing.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  ImageTemplatePreprocessor,
  ImagePreprocessorOptions,
  SaveImageOps,
} from '../base';
import { createProcessingContext } from '../../base';
import * as cv from '@techstark/opencv-js';

// Mock implementation for testing
class MockImagePreprocessor extends ImageTemplatePreprocessor {
  private filterCalled = false;

  protected getClassName(): string {
    return 'MockImagePreprocessor';
  }

  protected applyFilter(
    image: cv.Mat,
    coloredImage: cv.Mat,
    template: any,
    _filePath: string
  ): [cv.Mat, cv.Mat, any] {
    this.filterCalled = true;
    // Just return the same images for testing
    return [image, coloredImage, template];
  }

  public wasFilterCalled(): boolean {
    return this.filterCalled;
  }
}

describe('ImageTemplatePreprocessor', () => {
  let mockOptions: ImagePreprocessorOptions;
  let mockSaveImageOps: SaveImageOps;
  let defaultShape: [number, number];
  let mockGrayImage: cv.Mat;
  let mockColoredImage: cv.Mat;
  let mockTemplate: any;

  beforeEach(() => {
    // Create mock options
    mockOptions = {
      tuningOptions: {
        someOption: 'value',
      },
      processingImageShape: [800, 1200],
    };

    // Create mock save image ops
    mockSaveImageOps = {
      appendSaveImage: (_name: string, _image: cv.Mat) => {
        // Mock implementation
      },
      tuningConfig: {
        outputs: {
          colored_outputs_enabled: true,
        },
      },
    };

    defaultShape = [1000, 1500];

    // Create mock images (empty Mats for testing)
    mockGrayImage = new cv.Mat();
    mockColoredImage = new cv.Mat();
    mockTemplate = {
      name: 'test-template',
    };
  });

  describe('constructor', () => {
    it('should initialize with provided options', () => {
      const processor = new MockImagePreprocessor(
        mockOptions,
        'test/path',
        mockSaveImageOps,
        defaultShape
      );

      expect(processor).toBeDefined();
      expect(processor.getName()).toBe('MockImagePreprocessor');
    });

    it('should use default tuning options if not provided', () => {
      const optionsWithoutTuning = { ...mockOptions };
      delete optionsWithoutTuning.tuningOptions;

      const processor = new MockImagePreprocessor(
        optionsWithoutTuning,
        'test/path',
        mockSaveImageOps,
        defaultShape
      );

      expect(processor).toBeDefined();
    });

    it('should use default processing shape if not provided', () => {
      const optionsWithoutShape = { ...mockOptions };
      delete optionsWithoutShape.processingImageShape;

      const processor = new MockImagePreprocessor(
        optionsWithoutShape,
        'test/path',
        mockSaveImageOps,
        defaultShape
      );

      expect(processor).toBeDefined();
    });
  });

  describe('getName', () => {
    it('should return the class name', () => {
      const processor = new MockImagePreprocessor(
        mockOptions,
        'test/path',
        mockSaveImageOps,
        defaultShape
      );

      expect(processor.getName()).toBe('MockImagePreprocessor');
    });
  });

  describe('getRelativePath', () => {
    it('should join relative directory with path', () => {
      const processor = new MockImagePreprocessor(
        mockOptions,
        'test/dir',
        mockSaveImageOps,
        defaultShape
      );

      // Use type assertion to access protected method for testing
      const path = (processor as any).getRelativePath('file.txt');
      expect(path).toBe('test/dir/file.txt');
    });
  });

  describe('process', () => {
    it('should process context and return updated context', () => {
      const processor = new MockImagePreprocessor(
        mockOptions,
        'test/path',
        mockSaveImageOps,
        defaultShape
      );

      const context = createProcessingContext(
        'test.jpg',
        mockGrayImage,
        mockColoredImage,
        mockTemplate
      );

      const result = processor.process(context);

      expect(result).toBeDefined();
      expect(result.filePath).toBe('test.jpg');
      expect((processor as any).wasFilterCalled()).toBe(true);
    });

    it('should preserve context properties', () => {
      const processor = new MockImagePreprocessor(
        mockOptions,
        'test/path',
        mockSaveImageOps,
        defaultShape
      );

      const context = createProcessingContext(
        'test.jpg',
        mockGrayImage,
        mockColoredImage,
        mockTemplate
      );

      context.omrResponse = { Q1: 'A' };
      context.score = 85.5;

      const result = processor.process(context);

      expect(result.omrResponse).toEqual({ Q1: 'A' });
      expect(result.score).toBe(85.5);
    });

    it('should update gray and colored images', () => {
      const processor = new MockImagePreprocessor(
        mockOptions,
        'test/path',
        mockSaveImageOps,
        defaultShape
      );

      const context = createProcessingContext(
        'test.jpg',
        mockGrayImage,
        mockColoredImage,
        mockTemplate
      );

      const result = processor.process(context);

      // Images should be present (though they may be resized)
      expect(result.grayImage).toBeDefined();
      expect(result.coloredImage).toBeDefined();
    });
  });

  describe('excludeFiles', () => {
    it('should return empty array by default', () => {
      const processor = new MockImagePreprocessor(
        mockOptions,
        'test/path',
        mockSaveImageOps,
        defaultShape
      );

      const excluded = processor.excludeFiles();
      expect(excluded).toEqual([]);
    });
  });

  describe('inheritance', () => {
    it('should be instance of Processor', () => {
      const processor = new MockImagePreprocessor(
        mockOptions,
        'test/path',
        mockSaveImageOps,
        defaultShape
      );

      // Check that it implements the Processor interface
      expect(typeof processor.process).toBe('function');
      expect(typeof processor.getName).toBe('function');
    });
  });
});

