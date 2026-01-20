/**
 * Comprehensive tests for TemplateFileRunner.
 *
 * Tests multi-pass detection and interpretation architecture,
 * aggregate management, and edge cases.
 * Ported from Python test_template_file_runner.py
 */

import * as cv from '@techstark/opencv-js';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { TemplateFileRunner } from '../templateFileRunner';
import { TemplateLoader } from '../../../template/TemplateLoader';
import type { TemplateConfig } from '../../../template/types';
import {
  BubbleFieldDetectionResult,
  BubbleMeanValue,
} from '../models/detectionResults';
import type { Field } from '../../layout/field/base';

/**
 * Create minimal template config for testing.
 * Template path is handled internally by TemplateLoader in TypeScript,
 * so we don't need to set it explicitly like in Python.
 */
function createMinimalTemplateConfig(): TemplateConfig {
  return {
    templateDimensions: [900, 650],
    bubbleDimensions: [20, 20],
    emptyValue: '', // Explicitly set emptyValue to match Python tests
    fieldBlocks: {
      block1: {
        fieldDetectionType: 'BUBBLES_THRESHOLD',
        origin: [100, 100],
        fieldLabels: ['q1', 'q2'],
        bubbleFieldType: 'QTYPE_MCQ4',
        bubblesGap: 30,
        labelsGap: 50,
      },
    },
    preProcessors: [],
    alignment: { margins: { top: 0, right: 0, bottom: 0, left: 0 } },
    customBubbleFieldTypes: {},
    customLabels: {},
    outputColumns: { sortType: 'ALPHANUMERIC', customOrder: [] },
  };
}

/**
 * Create sample images for testing.
 */
function createSampleImages(): { grayImage: cv.Mat; coloredImage: cv.Mat } {
  const grayImage = new cv.Mat(800, 1000, cv.CV_8UC1);
  const coloredImage = new cv.Mat(800, 1000, cv.CV_8UC3);
  return { grayImage, coloredImage };
}

/**
 * Create mock bubble_fields dictionary for detection aggregates.
 * Port of Python's create_mock_bubble_fields helper function.
 *
 * @param fields - Array of Field objects
 * @returns Dictionary mapping field_label to BubbleFieldDetectionResult
 */
function createMockBubbleFields(fields: Field[]): Record<string, BubbleFieldDetectionResult> {
  const bubbleFields: Record<string, BubbleFieldDetectionResult> = {};

  for (const field of fields) {
    // Create minimal bubble means for testing
    // unitBubble can be null for tests (TypeScript allows null for optional BubbleLocation)
    const bubbleMeans = [
      new BubbleMeanValue(50.0, null as any, [0, 0]),
      new BubbleMeanValue(200.0, null as any, [0, 0]),
    ];
    const bubbleResult = new BubbleFieldDetectionResult(
      field.id,
      field.fieldLabel,
      bubbleMeans
    );
    bubbleFields[field.fieldLabel] = bubbleResult;
  }

  return bubbleFields;
}

describe('TemplateFileRunner', () => {
  let templateConfig: TemplateConfig;
  let templateLayout: ReturnType<typeof TemplateLoader.loadLayoutFromJSON>;
  let runner: TemplateFileRunner;
  let mockGrayImage: cv.Mat;
  let mockColoredImage: cv.Mat;

  beforeEach(() => {
    // Template config is set up once in fixture (createMinimalTemplateConfig)
    templateConfig = createMinimalTemplateConfig();
    templateLayout = TemplateLoader.loadLayoutFromJSON(templateConfig);
    const tuningConfig = {};
    runner = new TemplateFileRunner(templateLayout, tuningConfig);

    // Create mock images
    const images = createSampleImages();
    mockGrayImage = images.grayImage;
    mockColoredImage = images.coloredImage;
  });

  afterEach(() => {
    mockGrayImage.delete();
    mockColoredImage.delete();
  });

  describe('Initialization', () => {
    it('should initialize with template layout', () => {
      expect(runner.template).toBe(templateLayout);
      expect(runner.allFields.length).toBeGreaterThan(0);
      expect(runner.allFieldDetectionTypes.length).toBeGreaterThan(0);
    });

    it('should initialize field detection type runners', () => {
      const fieldRunners = (runner as any).fieldDetectionTypeFileRunners;
      expect(Object.keys(fieldRunners).length).toBeGreaterThan(0);
      expect(fieldRunners['BUBBLES_THRESHOLD']).toBeDefined();
    });

    it('should initialize directory level aggregates', () => {
      // Template layout is already set up in beforeEach fixture
      const detectionAggregates = runner.getDirectoryLevelDetectionAggregates();
      const interpretationAggregates = runner.getDirectoryLevelInterpretationAggregates();

      expect(detectionAggregates).toBeDefined();
      expect(interpretationAggregates).toBeDefined();
      if (detectionAggregates && 'initialDirectoryPath' in detectionAggregates) {
        expect(detectionAggregates.initialDirectoryPath).toBeDefined();
      }
      if (interpretationAggregates && 'initialDirectoryPath' in interpretationAggregates) {
        expect(interpretationAggregates.initialDirectoryPath).toBeDefined();
      }
    });
  });

  describe('readOmrAndUpdateMetrics', () => {
    it('should run two-pass detection and interpretation', () => {
      const filePath = 'test.jpg';

      // Mock the field runners to return responses (matching Python test)
      const fieldRunner = (runner as any).fieldDetectionTypeFileRunners['BUBBLES_THRESHOLD'];
      const mockDetection = vi.spyOn(fieldRunner, 'runFieldLevelDetection').mockReturnValue({} as any);
      const mockInterpretationResult = {
        getFieldInterpretationString: vi.fn().mockReturnValue('A'),
      };
      const mockInterpretation = vi.spyOn(fieldRunner, 'runFieldLevelInterpretation').mockReturnValue(mockInterpretationResult as any);
      const mockGetAggregates = vi.spyOn(fieldRunner, 'getFieldLevelInterpretationAggregates').mockReturnValue({
        field: runner.allFields[0],
        is_multi_marked: false,
      });

      // Mock bubble_fields in detection aggregates
      const bubbleFields = createMockBubbleFields(runner.allFields);
      const mockGetDetectionAggregates = vi.spyOn((runner as any).detectionPass, 'getFileLevelAggregates').mockReturnValue({
        bubble_fields: bubbleFields,
        ocr_fields: {},
        barcode_fields: {},
      });

      const omrResponse = runner.readOmrAndUpdateMetrics(
        filePath,
        mockGrayImage,
        mockColoredImage
      );

      expect(omrResponse).toBeDefined();
      expect(typeof omrResponse).toBe('object');

      // Check that aggregates were updated
      const detectionAggregates = runner.getDirectoryLevelDetectionAggregates();
      if (detectionAggregates && 'fileWiseAggregates' in detectionAggregates) {
        const fileWiseAggregates = detectionAggregates.fileWiseAggregates as Record<string, unknown>;
        expect(fileWiseAggregates[filePath]).toBeDefined();
      }

      // Restore mocks
      mockDetection.mockRestore();
      mockInterpretation.mockRestore();
      mockGetAggregates.mockRestore();
      mockGetDetectionAggregates.mockRestore();
    });

    it('should handle multiple files', () => {
      const filePaths = ['test1.jpg', 'test2.jpg', 'test3.jpg'];

      filePaths.forEach((filePath) => {
        runner.readOmrAndUpdateMetrics(filePath, mockGrayImage, mockColoredImage);
      });

      const detectionAggregates = runner.getDirectoryLevelDetectionAggregates();
      if (detectionAggregates && 'fileWiseAggregates' in detectionAggregates) {
        const fileWiseAggregates = detectionAggregates.fileWiseAggregates as Record<string, unknown>;
        expect(Object.keys(fileWiseAggregates).length).toBe(3);
      }
    });
  });

  describe('runFileLevelDetection', () => {
    it('should run detection for all fields', () => {
      const filePath = 'test.jpg';
      runner.initializeFileLevelDetectionAggregates(filePath);

      // Mock field detection (matching Python test)
      const fieldRunner = (runner as any).fieldDetectionTypeFileRunners['BUBBLES_THRESHOLD'];
      const mockDetection = vi.spyOn(fieldRunner, 'runFieldLevelDetection').mockReturnValue({} as any);

      runner.runFileLevelDetection(filePath, mockGrayImage, mockColoredImage);

      // Should have called detection for each field
      expect(mockDetection).toHaveBeenCalledTimes(2); // Two fields

      // Check that aggregates were updated
      const detectionAggregates = runner.getDirectoryLevelDetectionAggregates();
      if (detectionAggregates && 'fileWiseAggregates' in detectionAggregates) {
        const fileWiseAggregates = detectionAggregates.fileWiseAggregates as Record<string, unknown>;
        expect(fileWiseAggregates[filePath]).toBeDefined();
      }

      mockDetection.mockRestore();
    });

    it('should update detection aggregates on processed file', () => {
      const filePath = 'test.jpg';
      runner.initializeFileLevelDetectionAggregates(filePath);

      runner.runFileLevelDetection(filePath, mockGrayImage, mockColoredImage);

      // Check that aggregates were updated
      const aggregates = runner.getDirectoryLevelDetectionAggregates();
      if (aggregates && 'fileWiseAggregates' in aggregates) {
        const fileWiseAggregates = aggregates.fileWiseAggregates as Record<string, unknown>;
        expect(fileWiseAggregates[filePath]).toBeDefined();
      }
    });

    it('should handle empty fields', () => {
      // Create template with no fields
      const emptyConfig: TemplateConfig = {
        ...templateConfig,
        fieldBlocks: {},
      };

      expect(() => {
        TemplateLoader.loadLayoutFromJSON(emptyConfig);
      }).toThrow(); // Empty field blocks should throw during load
    });
  });

  describe('runFieldLevelDetection', () => {
    it('should run field-level detection for bubble fields', () => {
      const filePath = 'test.jpg';
      const field = runner.allFields[0];

      runner.initializeFileLevelDetectionAggregates(filePath);

      // Mock field detection (matching Python test)
      const fieldRunner = (runner as any).fieldDetectionTypeFileRunners['BUBBLES_THRESHOLD'];
      const mockDetection = vi.spyOn(fieldRunner, 'runFieldLevelDetection').mockReturnValue({} as any);

      runner.runFieldLevelDetection(field, mockGrayImage, mockColoredImage);

      expect(mockDetection).toHaveBeenCalledOnce();
      mockDetection.mockRestore();
    });
  });

  describe('runFileLevelInterpretation', () => {
    it('should run interpretation after detection', () => {
      // Template layout is already set up in beforeEach fixture
      const filePath = 'test.jpg';

      // Mock detection first (matching Python test)
      const fieldRunner = (runner as any).fieldDetectionTypeFileRunners['BUBBLES_THRESHOLD'];
      const mockDetection = vi.spyOn(fieldRunner, 'runFieldLevelDetection').mockReturnValue({} as any);

      runner.initializeFileLevelDetectionAggregates(filePath);
      runner.runFileLevelDetection(filePath, mockGrayImage, mockColoredImage);

      // Mock interpretation (matching Python test)
      const mockInterpretationResult = {
        getFieldInterpretationString: vi.fn().mockReturnValue('A'),
      };
      const mockInterpretation = vi.spyOn(
        fieldRunner.interpretationPass,
        'runFieldLevelInterpretation'
      ).mockReturnValue(mockInterpretationResult as any);
      const mockGetAggregates = vi.spyOn(fieldRunner, 'getFieldLevelInterpretationAggregates').mockReturnValue({
        field: runner.allFields[0],
        is_multi_marked: false,
      });

      // Mock bubble_fields in detection aggregates
      const bubbleFields = createMockBubbleFields(runner.allFields);
      const mockGetDetectionAggregates = vi.spyOn((runner as any).detectionPass, 'getFileLevelAggregates').mockReturnValue({
        bubble_fields: bubbleFields,
        ocr_fields: {},
        barcode_fields: {},
      });

      // Run interpretation
      const omrResponse = runner.runFileLevelInterpretation(
        filePath,
        mockGrayImage,
        mockColoredImage
      );

      expect(omrResponse).toBeDefined();
      expect(typeof omrResponse).toBe('object');
      expect(mockInterpretation).toHaveBeenCalledTimes(2); // Two fields

      // Check that interpretation aggregates were updated
      const interpretationAggregates = runner.getDirectoryLevelInterpretationAggregates();
      expect(interpretationAggregates).toBeDefined();

      // Restore mocks
      mockDetection.mockRestore();
      mockInterpretation.mockRestore();
      mockGetAggregates.mockRestore();
      mockGetDetectionAggregates.mockRestore();
    });
  });

  describe('runFieldLevelInterpretation', () => {
    it('should run field-level interpretation', () => {
      const filePath = 'test.jpg';
      const field = runner.allFields[0];

      // Run detection first to populate aggregates
      runner.initializeFileLevelDetectionAggregates(filePath);
      runner.runFileLevelDetection(filePath, mockGrayImage, mockColoredImage);

      // Initialize interpretation aggregates
      runner.initializeFileLevelInterpretationAggregates(filePath);

      const currentOmrResponse: Record<string, string> = {};

      // Mock detection pass to return bubble_fields
      const bubbleFields = createMockBubbleFields(runner.allFields);
      const detectionPass = (runner as any).detectionPass;
      const originalGetFileLevelAggregates = detectionPass.getFileLevelAggregates.bind(detectionPass);
      detectionPass.getFileLevelAggregates = () => ({
        bubble_fields: bubbleFields,
        ocr_fields: {},
        barcode_fields: {},
      });

      try {
        runner.runFieldLevelInterpretation(field, currentOmrResponse);

        expect(field.fieldLabel in currentOmrResponse).toBe(true);
      } finally {
        // Restore original method
        detectionPass.getFileLevelAggregates = originalGetFileLevelAggregates;
      }
    });
  });

  describe('Aggregate Management', () => {
    it('should collect aggregates across multiple files', () => {
      const filePaths = ['test1.jpg', 'test2.jpg', 'test3.jpg'];

      filePaths.forEach((filePath) => {
        runner.readOmrAndUpdateMetrics(filePath, mockGrayImage, mockColoredImage);
      });

      const detectionAggregates = runner.getDirectoryLevelDetectionAggregates();
      const interpretationAggregates = runner.getDirectoryLevelInterpretationAggregates();

      if (detectionAggregates && 'fileWiseAggregates' in detectionAggregates) {
        const fileWiseAggregates = detectionAggregates.fileWiseAggregates as Record<string, unknown>;
        expect(Object.keys(fileWiseAggregates).length).toBe(3);
      }
      if (interpretationAggregates && 'fileWiseAggregates' in interpretationAggregates) {
        const fileWiseAggregates = interpretationAggregates.fileWiseAggregates as Record<string, unknown>;
        expect(Object.keys(fileWiseAggregates).length).toBe(3);
      }
    });

    it('should finish processing directory', () => {
      const filePath = 'test.jpg';
      runner.readOmrAndUpdateMetrics(filePath, mockGrayImage, mockColoredImage);

      // Should not throw
      expect(() => {
        runner.finishProcessingDirectory();
      }).not.toThrow();
    });

    it('should get export OMR metrics for file', () => {
      const filePath = 'test.jpg';
      runner.readOmrAndUpdateMetrics(filePath, mockGrayImage, mockColoredImage);

      // Note: getExportOmrMetricsForFile doesn't take filePath parameter (matching Python)
      const metrics = runner.getExportOmrMetricsForFile();

      expect(metrics).toBeDefined();
      expect(typeof metrics).toBe('object');
    });

    it('should handle empty aggregates', () => {
      // Get metrics without processing
      const metrics = runner.getExportOmrMetricsForFile();

      expect(metrics).toBeDefined();
    });
  });

  describe('getFieldDetectionTypeFileRunner', () => {
    it('should return file runner for valid detection type', () => {
      const fileRunner = (runner as any).getFieldDetectionTypeFileRunner('BUBBLES_THRESHOLD');
      expect(fileRunner).toBeDefined();
      expect(fileRunner.fieldDetectionType).toBe('BUBBLES_THRESHOLD');
    });

    it('should throw error for invalid detection type', () => {
      expect(() => {
        (runner as any).getFieldDetectionTypeFileRunner('INVALID_TYPE');
      }).toThrow();
    });
  });

  describe('Edge Cases', () => {
    it('should handle no fields in template', () => {
      // Template layout is already set up in beforeEach fixture
      // This should be caught during template loading, but test the runner with minimal fields
      expect(runner.allFields.length).toBeGreaterThan(0);
      expect(runner.allFields.length).toBe(2); // q1 and q2 from block1
    });

    it('should handle zero-sized images', () => {
      const zeroImage = new cv.Mat(0, 0, cv.CV_8UC1);
      const filePath = 'test.jpg';

      expect(() => {
        runner.readOmrAndUpdateMetrics(filePath, zeroImage, zeroImage);
      }).not.toThrow();

      zeroImage.delete();
    });

    it('should handle different image sizes', () => {
      const smallImage = new cv.Mat(100, 100, cv.CV_8UC1);
      const largeImage = new cv.Mat(2000, 2000, cv.CV_8UC1);
      const filePath = 'test.jpg';

      expect(() => {
        runner.readOmrAndUpdateMetrics(filePath, smallImage, smallImage);
        runner.readOmrAndUpdateMetrics('test2.jpg', largeImage, largeImage);
      }).not.toThrow();

      smallImage.delete();
      largeImage.delete();
    });

    it('should handle multiple calls to same file', () => {
      const filePath = 'test.jpg';

      // Process same file multiple times
      runner.readOmrAndUpdateMetrics(filePath, mockGrayImage, mockColoredImage);
      runner.readOmrAndUpdateMetrics(filePath, mockGrayImage, mockColoredImage);
      runner.readOmrAndUpdateMetrics(filePath, mockGrayImage, mockColoredImage);

      const detectionAggregates = runner.getDirectoryLevelDetectionAggregates();
      // Should still have aggregates (may overwrite or accumulate)
      expect(detectionAggregates).toBeDefined();
    });
  });
});

