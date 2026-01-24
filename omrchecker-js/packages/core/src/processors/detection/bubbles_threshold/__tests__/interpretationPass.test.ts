/**
 * Tests for BubblesThresholdInterpretationPass.
 */

import { BubbleField } from '../../../layout/field/bubbleField';
import { FieldBlock } from '../../../layout/fieldBlock/base';
import { BubblesThresholdInterpretationPass } from '../interpretationPass';
import { BubblesFieldInterpretation } from '../interpretation';
import { BubbleMeanValue, BubbleFieldDetectionResult } from '../../models/detectionResults';
import { DetectionRepository } from '../../../repositories/DetectionRepository';
import { FieldDetectionType } from '../../../constants';

describe('BubblesThresholdInterpretationPass', () => {
  let tuningConfig: Record<string, unknown>;
  let pass: BubblesThresholdInterpretationPass;
  let mockField: BubbleField;
  let repository: DetectionRepository;

  beforeEach(() => {
    tuningConfig = {};
    repository = new DetectionRepository();
    pass = new BubblesThresholdInterpretationPass(
      tuningConfig,
      FieldDetectionType.BUBBLES_THRESHOLD,
      repository
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
  });

  describe('constructor', () => {
    it('should initialize with BUBBLES_THRESHOLD detection type', () => {
      expect(pass.fieldDetectionType).toBe('BUBBLES_THRESHOLD');
    });
  });

  describe('getFieldInterpretation', () => {
    it('should return BubblesFieldInterpretation instance', () => {
      // Set up proper tuning config with thresholding
      tuningConfig = {
        thresholding: {
          GLOBAL_PAGE_THRESHOLD: 180,
          MIN_JUMP: 10,
        },
      };
      pass = new BubblesThresholdInterpretationPass(
        tuningConfig,
        FieldDetectionType.BUBBLES_THRESHOLD,
        repository
      );

      // Provide proper mock detection aggregates using BubbleFieldDetectionResult
      const bubbleMeans = [
        new BubbleMeanValue(100, { x: 0, y: 0, width: 10, height: 10, label: 'A' }, [0, 0]),
        new BubbleMeanValue(200, { x: 10, y: 0, width: 10, height: 10, label: 'B' }, [10, 0]),
      ];
      const detectionResult = new BubbleFieldDetectionResult('q1', 'q1', bubbleMeans);
      
      const fileLevelDetectionAggregates = {
        bubble_fields: {
          q1: detectionResult,
        },
      };
      // Provide proper mock interpretation aggregates with fallback threshold
      const fileLevelInterpretationAggregates = {
        file_level_fallback_threshold: 180,
      };
      const interpretation = pass.getFieldInterpretation(
        mockField,
        fileLevelDetectionAggregates,
        fileLevelInterpretationAggregates
      );
      expect(interpretation).toBeInstanceOf(BubblesFieldInterpretation);
    });
  });

  describe('initializeFileLevelAggregates', () => {
    it('should initialize with threshold aggregates', () => {
      // Set up proper tuning config with thresholding
      tuningConfig = {
        thresholding: {
          MIN_JUMP_STD: 5.0,
          GLOBAL_PAGE_THRESHOLD_STD: 10.0,
          GLOBAL_PAGE_THRESHOLD: 180,
          MIN_JUMP: 10,
        },
      };
      pass = new BubblesThresholdInterpretationPass(
        tuningConfig,
        FieldDetectionType.BUBBLES_THRESHOLD,
        repository
      );

      // Initialize repository with some test data
      repository.initializeFile('/test/file.jpg');
      const bubbleMeans = [
        new BubbleMeanValue(100, { x: 0, y: 0, width: 10, height: 10, label: 'A' }, [0, 0]),
        new BubbleMeanValue(200, { x: 10, y: 0, width: 10, height: 10, label: 'B' }, [10, 0]),
      ];
      // Save to repository so getAllBubbleMeansForCurrentFile() returns data
      repository.saveBubbleField('q1', {
        fieldId: 'q1',
        fieldLabel: 'q1',
        bubbleMeans,
        stdDeviation: 5.0,
      });

      expect(() => {
        pass.initializeFileLevelAggregates('/test/file.jpg');
      }).not.toThrow();

      const fileAgg = pass.getFileLevelAggregates();
      expect(fileAgg).toBeDefined();
      expect((fileAgg as { file_level_fallback_threshold?: number }).file_level_fallback_threshold).toBeDefined();
    });
  });

  describe('getOutlierDeviationThreshold', () => {
    it('should calculate outlier deviation threshold', () => {
      // Set up proper tuning config with thresholding
      tuningConfig = {
        thresholding: {
          MIN_JUMP_STD: 5.0,
          GLOBAL_PAGE_THRESHOLD_STD: 10.0,
        },
      };
      pass = new BubblesThresholdInterpretationPass(
        tuningConfig,
        FieldDetectionType.BUBBLES_THRESHOLD,
        repository
      );

      const allOutlierDeviations = [5.0, 6.0, 7.0, 8.0];
      const threshold = pass.getOutlierDeviationThreshold(allOutlierDeviations);
      expect(typeof threshold).toBe('number');
      expect(threshold).toBeGreaterThan(0);
    });
  });

  describe('getFallbackThreshold', () => {
    it('should calculate fallback threshold', () => {
      // Set up proper tuning config with thresholding
      tuningConfig = {
        thresholding: {
          GLOBAL_PAGE_THRESHOLD: 180,
          MIN_JUMP: 10,
        },
      };
      pass = new BubblesThresholdInterpretationPass(
        tuningConfig,
        FieldDetectionType.BUBBLES_THRESHOLD,
        repository
      );

      const fieldWiseMeansAndRefs = [
        new BubbleMeanValue(100, { x: 0, y: 0, width: 10, height: 10, label: 'A' }, [0, 0]),
        new BubbleMeanValue(200, { x: 10, y: 0, width: 10, height: 10, label: 'B' }, [10, 0]),
      ];

      const result = pass.getFallbackThreshold(fieldWiseMeansAndRefs);
      expect(result.fileLevelFallbackThreshold).toBeDefined();
      expect(typeof result.fileLevelFallbackThreshold).toBe('number');
      expect(result.globalMaxJump).toBeDefined();
      expect(typeof result.globalMaxJump).toBe('number');
    });
  });
});

