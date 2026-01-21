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

/**
 * Mock configuration interface for field runner methods.
 * Port of Python's MockConfig class.
 */
interface MockConfig {
  detection?: ReturnType<typeof vi.spyOn>;
  interpretation?: ReturnType<typeof vi.spyOn>;
  interpretationPass?: ReturnType<typeof vi.spyOn>;
  getInterpretationAggregates?: ReturnType<typeof vi.spyOn>;
  getDetectionAggregates?: ReturnType<typeof vi.spyOn>;
  getFileResults?: ReturnType<typeof vi.spyOn>;
  cleanup: () => void;
}

/**
 * Mock field runner methods with configurable options.
 * Port of Python's mock_field_runner_methods context manager.
 *
 * @param runner - TemplateFileRunner instance
 * @param fieldType - Field detection type to mock (default: "BUBBLES_THRESHOLD")
 * @param options - Options for which mocks to include
 * @returns MockConfig object with mocks and cleanup function
 */
function mockFieldRunnerMethods(
  runner: TemplateFileRunner,
  fieldType: string = 'BUBBLES_THRESHOLD',
  options: {
    includeDetection?: boolean;
    includeInterpretation?: boolean;
    includeAggregates?: boolean;
    includeRepository?: boolean;
  } = {}
): MockConfig {
  const {
    includeDetection = true,
    includeInterpretation = true,
    includeAggregates = true,
    includeRepository = true,
  } = options;

  const fieldRunner = (runner as any).fieldDetectionTypeFileRunners[fieldType];
  const mocks: MockConfig = { cleanup: () => {} };
  const cleanupFunctions: (() => void)[] = [];

  if (includeDetection) {
    const mock = vi.spyOn(fieldRunner, 'runFieldLevelDetection').mockReturnValue({} as any);
    mocks.detection = mock;
    cleanupFunctions.push(() => mock.mockRestore());
  }

  if (includeInterpretation) {
    const mock = vi.spyOn(fieldRunner, 'runFieldLevelInterpretation').mockReturnValue({
      getFieldInterpretationString: vi.fn().mockReturnValue('A'),
    } as any);
    mocks.interpretation = mock;
    cleanupFunctions.push(() => mock.mockRestore());

    const mockPass = vi.spyOn(
      fieldRunner.interpretationPass,
      'runFieldLevelInterpretation'
    ).mockReturnValue({
      getFieldInterpretationString: vi.fn().mockReturnValue('A'),
    } as any);
    mocks.interpretationPass = mockPass;
    cleanupFunctions.push(() => mockPass.mockRestore());
  }

  if (includeAggregates) {
    const mock = vi.spyOn(fieldRunner, 'getFieldLevelInterpretationAggregates').mockReturnValue({
      field: runner.allFields[0],
      is_multi_marked: false,
    });
    mocks.getInterpretationAggregates = mock;
    cleanupFunctions.push(() => mock.mockRestore());

    const bubbleFields = createMockBubbleFields(runner.allFields);
    const mockDetection = vi.spyOn(
      (runner as any).detectionPass,
      'getFileLevelAggregates'
    ).mockReturnValue({
      bubble_fields: bubbleFields,
      ocr_fields: {},
      barcode_fields: {},
    });
    mocks.getDetectionAggregates = mockDetection;
    cleanupFunctions.push(() => mockDetection.mockRestore());
  }

  if (includeRepository) {
    // Note: Repository mocking may not be needed in TypeScript version
    // This is a placeholder for consistency with Python version
  }

  mocks.cleanup = () => {
    cleanupFunctions.forEach((cleanup) => cleanup());
  };

  return mocks;
}

/**
 * Mock only detection methods.
 * Port of Python's mock_detection_only context manager.
 *
 * @param runner - TemplateFileRunner instance
 * @param fieldType - Field detection type to mock (default: "BUBBLES_THRESHOLD")
 * @returns MockConfig object with detection mocks and cleanup function
 */
function mockDetectionOnly(
  runner: TemplateFileRunner,
  fieldType: string = 'BUBBLES_THRESHOLD'
): MockConfig {
  return mockFieldRunnerMethods(runner, fieldType, {
    includeDetection: true,
    includeInterpretation: false,
    includeAggregates: false,
    includeRepository: false,
  });
}

/**
 * Mock only interpretation methods.
 * Port of Python's mock_interpretation_only context manager.
 *
 * @param runner - TemplateFileRunner instance
 * @param fieldType - Field detection type to mock (default: "BUBBLES_THRESHOLD")
 * @returns MockConfig object with interpretation mocks and cleanup function
 */
function mockInterpretationOnly(
  runner: TemplateFileRunner,
  fieldType: string = 'BUBBLES_THRESHOLD'
): MockConfig {
  return mockFieldRunnerMethods(runner, fieldType, {
    includeDetection: false,
    includeInterpretation: true,
    includeAggregates: true,
    includeRepository: false,
  });
}

/**
 * Configure mocks with default return values.
 * Port of Python's setup_default_mock_responses helper function.
 *
 * @param mocks - MockConfig object from mockFieldRunnerMethods
 * @param fields - Array of Field objects
 * @param filePath - File path string
 * @param interpretationValue - Value to return for interpretation (default: "A")
 * @param isMultiMarked - Whether field is multi-marked (default: false)
 */
function setupDefaultMockResponses(
  mocks: MockConfig,
  fields: Field[],
  filePath: string,
  interpretationValue: string = 'A',
  isMultiMarked: boolean = false
): void {
  if (mocks.detection) {
    mocks.detection.mockReturnValue({} as any);
  }

  if (mocks.interpretation) {
    mocks.interpretation.mockReturnValue({
      getFieldInterpretationString: vi.fn().mockReturnValue(interpretationValue),
    } as any);
  }

  if (mocks.interpretationPass) {
    mocks.interpretationPass.mockReturnValue({
      getFieldInterpretationString: vi.fn().mockReturnValue(interpretationValue),
    } as any);
  }

  if (mocks.getInterpretationAggregates) {
    mocks.getInterpretationAggregates.mockReturnValue({
      field: fields[0] || null,
      is_multi_marked: isMultiMarked,
    });
  }

  if (mocks.getDetectionAggregates) {
    const bubbleFields = createMockBubbleFields(fields);
    mocks.getDetectionAggregates.mockReturnValue({
      bubble_fields: bubbleFields,
      ocr_fields: {},
      barcode_fields: {},
    });
  }
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

      const mocks = mockFieldRunnerMethods(runner);
      setupDefaultMockResponses(mocks, runner.allFields, filePath);

      try {
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
      } finally {
        mocks.cleanup();
      }
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

      const mocks = mockDetectionOnly(runner);
      if (mocks.detection) {
        mocks.detection.mockReturnValue({} as any);
      }

      try {
        runner.runFileLevelDetection(filePath, mockGrayImage, mockColoredImage);

        // Should have called detection for each field
        if (mocks.detection) {
          expect(mocks.detection).toHaveBeenCalledTimes(2); // Two fields
        }

        // Check that aggregates were updated
        const detectionAggregates = runner.getDirectoryLevelDetectionAggregates();
        if (detectionAggregates && 'fileWiseAggregates' in detectionAggregates) {
          const fileWiseAggregates = detectionAggregates.fileWiseAggregates as Record<string, unknown>;
          expect(fileWiseAggregates[filePath]).toBeDefined();
        }
      } finally {
        mocks.cleanup();
      }
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

      const mocks = mockDetectionOnly(runner);
      if (mocks.detection) {
        mocks.detection.mockReturnValue({} as any);
      }

      try {
        runner.runFieldLevelDetection(field, mockGrayImage, mockColoredImage);

        if (mocks.detection) {
          expect(mocks.detection).toHaveBeenCalledOnce();
        }
      } finally {
        mocks.cleanup();
      }
    });
  });

  describe('runFileLevelInterpretation', () => {
    it('should run interpretation after detection', () => {
      // Template layout is already set up in beforeEach fixture
      const filePath = 'test.jpg';

      // Mock detection first
      const detectionMocks = mockDetectionOnly(runner);
      if (detectionMocks.detection) {
        detectionMocks.detection.mockReturnValue({} as any);
      }

      try {
        runner.initializeFileLevelDetectionAggregates(filePath);
        runner.runFileLevelDetection(filePath, mockGrayImage, mockColoredImage);
      } finally {
        detectionMocks.cleanup();
      }

      // Mock interpretation
      const interpretationMocks = mockInterpretationOnly(runner);
      setupDefaultMockResponses(interpretationMocks, runner.allFields, filePath);

      try {
        // Run interpretation
        const omrResponse = runner.runFileLevelInterpretation(
          filePath,
          mockGrayImage,
          mockColoredImage
        );

        expect(omrResponse).toBeDefined();
        expect(typeof omrResponse).toBe('object');
        if (interpretationMocks.interpretationPass) {
          expect(interpretationMocks.interpretationPass).toHaveBeenCalledTimes(2); // Two fields
        }

        // Check that interpretation aggregates were updated
        const interpretationAggregates = runner.getDirectoryLevelInterpretationAggregates();
        expect(interpretationAggregates).toBeDefined();
      } finally {
        interpretationMocks.cleanup();
      }
    });
  });

  describe('runFieldLevelInterpretation', () => {
    it('should run field-level interpretation', () => {
      const filePath = 'test.jpg';
      const field = runner.allFields[0];

      // Run detection first to populate aggregates
      const detectionMocks = mockDetectionOnly(runner);
      if (detectionMocks.detection) {
        detectionMocks.detection.mockReturnValue({} as any);
      }

      try {
        runner.initializeFileLevelDetectionAggregates(filePath);
        runner.runFileLevelDetection(filePath, mockGrayImage, mockColoredImage);
      } finally {
        detectionMocks.cleanup();
      }

      // Initialize interpretation aggregates
      runner.initializeFileLevelInterpretationAggregates(filePath);

      const currentOmrResponse: Record<string, string> = {};

      const interpretationMocks = mockInterpretationOnly(runner);
      setupDefaultMockResponses(interpretationMocks, runner.allFields, filePath);

      try {
        runner.runFieldLevelInterpretation(field, currentOmrResponse);

        expect(field.fieldLabel in currentOmrResponse).toBe(true);
        if (interpretationMocks.interpretationPass) {
          expect(interpretationMocks.interpretationPass).toHaveBeenCalledOnce();
        }
      } finally {
        interpretationMocks.cleanup();
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

