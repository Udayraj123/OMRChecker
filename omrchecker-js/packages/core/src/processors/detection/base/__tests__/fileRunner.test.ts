/**
 * Tests for file runner base classes.
 */

import * as cv from '@techstark/opencv-js';
import { FieldDetectionType } from '../../../constants';
import { BubbleField } from '../../../layout/field/bubbleField';
import { FieldBlock } from '../../../layout/fieldBlock/base';
import { FileLevelRunner, FieldTypeFileLevelRunner } from '../fileRunner';
import { FieldTypeDetectionPass, TemplateDetectionPass } from '../detectionPass';
import { FieldTypeInterpretationPass, TemplateInterpretationPass } from '../interpretationPass';
import type { FieldDetection } from '../detection';
import type { FieldInterpretation } from '../interpretation';
import type { Field } from '../../../layout/field/base';

// Mock passes for testing
class MockDetectionPass extends FieldTypeDetectionPass {
  getFieldDetection(_field: Field, _grayImage: cv.Mat, _coloredImage?: cv.Mat): FieldDetection {
    throw new Error('Not implemented in mock');
  }
}

class MockInterpretationPass extends FieldTypeInterpretationPass {
  getFieldInterpretation(
    _field: Field,
    _fileLevelDetectionAggregates: unknown,
    _fileLevelInterpretationAggregates: unknown
  ): FieldInterpretation {
    throw new Error('Not implemented in mock');
  }
}

describe('FileLevelRunner', () => {
  let tuningConfig: Record<string, unknown>;
  let detectionPass: TemplateDetectionPass;
  let interpretationPass: TemplateInterpretationPass;
  let runner: FileLevelRunner<TemplateDetectionPass, TemplateInterpretationPass>;

  beforeEach(() => {
    tuningConfig = {};
    detectionPass = new TemplateDetectionPass(tuningConfig);
    interpretationPass = new TemplateInterpretationPass(tuningConfig);
    runner = new FileLevelRunner(tuningConfig, detectionPass, interpretationPass);
  });

  describe('constructor', () => {
    it('should initialize with passes', () => {
      expect(runner).toBeInstanceOf(FileLevelRunner);
    });
  });

  describe('initializeDirectoryLevelAggregates', () => {
    it('should initialize both detection and interpretation aggregates', () => {
      runner.initializeDirectoryLevelAggregates('/test/path');

      const detectionAggs = runner.getDirectoryLevelDetectionAggregates();
      const interpretationAggs = runner.getDirectoryLevelInterpretationAggregates();

      expect(detectionAggs).toBeDefined();
      expect(interpretationAggs).toBeDefined();
    });
  });

  describe('file level detection aggregates', () => {
    it('should manage file level detection aggregates', () => {
      runner.initializeFileLevelDetectionAggregates('/test/file.jpg');
      const aggs = runner.getFileLevelDetectionAggregates();
      expect(aggs).toBeDefined();
    });

    it('should update detection aggregates on processed file', () => {
      runner.initializeFileLevelDetectionAggregates('/test/file.jpg');
      expect(() => {
        runner.updateDetectionAggregatesOnProcessedFile('/test/file.jpg');
      }).not.toThrow();
    });
  });

  describe('file level interpretation aggregates', () => {
    it('should manage file level interpretation aggregates', () => {
      runner.initializeFileLevelInterpretationAggregates('/test/file.jpg', {}, {});
      const aggs = runner.getFileLevelInterpretationAggregates();
      expect(aggs).toBeDefined();
    });

    it('should update interpretation aggregates on processed file', () => {
      runner.initializeFileLevelInterpretationAggregates('/test/file.jpg', {}, {});
      expect(() => {
        runner.updateInterpretationAggregatesOnProcessedFile('/test/file.jpg');
      }).not.toThrow();
    });
  });
});

describe('FieldTypeFileLevelRunner', () => {
  let tuningConfig: Record<string, unknown>;
  let detectionPass: MockDetectionPass;
  let interpretationPass: MockInterpretationPass;
  let runner: FieldTypeFileLevelRunner;
  let mockField: BubbleField;
  let mockGrayImage: cv.Mat;

  beforeEach(() => {
    tuningConfig = {};
    detectionPass = new MockDetectionPass(tuningConfig, FieldDetectionType.BUBBLES_THRESHOLD);
    interpretationPass = new MockInterpretationPass(tuningConfig, FieldDetectionType.BUBBLES_THRESHOLD);
    runner = new FieldTypeFileLevelRunner(
      tuningConfig,
      FieldDetectionType.BUBBLES_THRESHOLD,
      detectionPass,
      interpretationPass
    );

    // Create a mock field block
    const mockFieldBlock = {
      name: 'testBlock',
      shifts: [0, 0],
      bubbleDimensions: [10, 10],
      bubbleValues: ['A', 'B'],
      bubblesGap: 5,
      bubbleFieldType: 'QTYPE_MCQ4',
    } as FieldBlock;

    // Create a mock field
    mockField = new BubbleField(
      'horizontal',
      '',
      mockFieldBlock,
      'BUBBLES_THRESHOLD',
      'q1',
      [0, 0]
    );

    // Create a mock grayscale image
    mockGrayImage = new cv.Mat(100, 100, cv.CV_8UC1);
  });

  afterEach(() => {
    mockGrayImage.delete();
  });

  describe('constructor', () => {
    it('should initialize with field detection type', () => {
      expect(runner.fieldDetectionType).toBe(FieldDetectionType.BUBBLES_THRESHOLD);
    });
  });

  describe('runFieldLevelDetection', () => {
    it('should throw error if detection pass not properly implemented', () => {
      expect(() => {
        runner.runFieldLevelDetection(mockField, mockGrayImage);
      }).toThrow();
    });
  });

  describe('runFieldLevelInterpretation', () => {
    it('should throw error if interpretation pass not properly implemented', () => {
      runner.initializeFileLevelDetectionAggregates('/test/file.jpg');
      expect(() => {
        runner.runFieldLevelInterpretation(mockField);
      }).toThrow();
    });
  });
});

