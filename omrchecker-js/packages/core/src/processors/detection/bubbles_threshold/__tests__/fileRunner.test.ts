/**
 * Tests for BubblesThresholdFileRunner.
 */

import * as cv from '@techstark/opencv-js';
import { BubbleField } from '../../../layout/field/bubbleField';
import { FieldBlock } from '../../../layout/fieldBlock/base';
import { BubblesThresholdFileRunner } from '../fileRunner';

describe('BubblesThresholdFileRunner', () => {
  let tuningConfig: Record<string, unknown>;
  let runner: BubblesThresholdFileRunner;
  let mockField: BubbleField;
  let mockGrayImage: cv.Mat;

  beforeEach(() => {
    tuningConfig = {};
    runner = new BubblesThresholdFileRunner(tuningConfig);

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
    it('should initialize with BUBBLES_THRESHOLD detection type', () => {
      expect(runner.fieldDetectionType).toBe('BUBBLES_THRESHOLD');
    });
  });

  describe('runFieldLevelDetection', () => {
    it('should run field level detection', () => {
      runner.initializeFileLevelDetectionAggregates('/test/file.jpg');
      const detection = runner.runFieldLevelDetection(mockField, mockGrayImage);
      expect(detection).toBeDefined();
    });
  });

  describe('runFieldLevelInterpretation', () => {
    it('should run field level interpretation', () => {
      runner.initializeFileLevelDetectionAggregates('/test/file.jpg');
      runner.initializeFileLevelInterpretationAggregates('/test/file.jpg', {}, {});

      // Run detection first
      runner.runFieldLevelDetection(mockField, mockGrayImage);

      const interpretation = runner.runFieldLevelInterpretation(mockField);
      expect(interpretation).toBeDefined();
      expect(interpretation.getFieldInterpretationString).toBeDefined();
    });
  });
});

