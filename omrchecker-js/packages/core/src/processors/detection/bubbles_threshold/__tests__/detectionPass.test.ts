/**
 * Tests for BubblesThresholdDetectionPass.
 */

import * as cv from '@techstark/opencv-js';
import { BubbleField } from '../../../layout/field/bubbleField';
import { FieldBlock } from '../../../layout/fieldBlock/base';
import { BubblesThresholdDetectionPass } from '../detectionPass';
import { BubblesFieldDetection } from '../detection';

describe('BubblesThresholdDetectionPass', () => {
  let tuningConfig: Record<string, unknown>;
  let pass: BubblesThresholdDetectionPass;
  let mockField: BubbleField;
  let mockGrayImage: cv.Mat;

  beforeEach(() => {
    tuningConfig = {};
    pass = new BubblesThresholdDetectionPass(tuningConfig);

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
      expect(pass.fieldDetectionType).toBe('BUBBLES_THRESHOLD');
    });
  });

  describe('getFieldDetection', () => {
    it('should return BubblesFieldDetection instance', () => {
      const detection = pass.getFieldDetection(mockField, mockGrayImage);
      expect(detection).toBeInstanceOf(BubblesFieldDetection);
    });
  });

  describe('initializeDirectoryLevelAggregates', () => {
    it('should initialize with file_wise_thresholds', () => {
      pass.initializeDirectoryLevelAggregates('/test/path');
      const dirAgg = pass.getDirectoryLevelAggregates();
      expect(dirAgg).toBeDefined();
      expect((dirAgg as { file_wise_thresholds?: unknown }).file_wise_thresholds).toBeDefined();
    });
  });

  describe('initializeFileLevelAggregates', () => {
    it('should initialize with bubble means arrays', () => {
      pass.initializeFileLevelAggregates('/test/file.jpg');
      const fileAgg = pass.getFileLevelAggregates();
      expect(fileAgg).toBeDefined();
      expect((fileAgg as { all_field_bubble_means?: unknown[] }).all_field_bubble_means).toEqual([]);
      expect((fileAgg as { all_field_bubble_means_std?: unknown[] }).all_field_bubble_means_std).toEqual([]);
    });
  });

  describe('updateFieldLevelAggregatesOnProcessedFieldDetection', () => {
    it('should update field level aggregates', () => {
      pass.initializeFieldLevelAggregates(mockField);
      const detection = pass.getFieldDetection(mockField, mockGrayImage);

      expect(() => {
        pass.updateFieldLevelAggregatesOnProcessedFieldDetection(mockField, detection);
      }).not.toThrow();
    });
  });
});

