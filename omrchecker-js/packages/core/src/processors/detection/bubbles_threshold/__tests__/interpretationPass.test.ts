/**
 * Tests for BubblesThresholdInterpretationPass.
 */

import { BubbleField } from '../../../layout/field/bubbleField';
import { FieldBlock } from '../../../layout/fieldBlock/base';
import { BubblesThresholdInterpretationPass } from '../interpretationPass';
import { BubblesFieldInterpretation } from '../interpretation';
import { BubbleMeanValue } from '../../models/detectionResults';

describe('BubblesThresholdInterpretationPass', () => {
  let tuningConfig: Record<string, unknown>;
  let pass: BubblesThresholdInterpretationPass;
  let mockField: BubbleField;

  beforeEach(() => {
    tuningConfig = {};
    pass = new BubblesThresholdInterpretationPass(tuningConfig);

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
      const interpretation = pass.getFieldInterpretation(mockField, {}, {});
      expect(interpretation).toBeInstanceOf(BubblesFieldInterpretation);
    });
  });

  describe('initializeFileLevelAggregates', () => {
    it('should initialize with threshold aggregates', () => {
      const fieldDetectionTypeWiseAggs = {
        BUBBLES_THRESHOLD: {
          all_field_bubble_means_std: [5.0, 6.0],
          all_field_bubble_means: [
            new BubbleMeanValue(100, { x: 0, y: 0, width: 10, height: 10, label: 'A' }, [0, 0]),
            new BubbleMeanValue(200, { x: 10, y: 0, width: 10, height: 10, label: 'B' }, [10, 0]),
          ],
        },
      };

      expect(() => {
        pass.initializeFileLevelAggregates(
          '/test/file.jpg',
          fieldDetectionTypeWiseAggs,
          {}
        );
      }).not.toThrow();

      const fileAgg = pass.getFileLevelAggregates();
      expect(fileAgg).toBeDefined();
      expect((fileAgg as { file_level_fallback_threshold?: number }).file_level_fallback_threshold).toBeDefined();
    });
  });

  describe('getOutlierDeviationThreshold', () => {
    it('should calculate outlier deviation threshold', () => {
      const allOutlierDeviations = [5.0, 6.0, 7.0, 8.0];
      const threshold = pass.getOutlierDeviationThreshold(allOutlierDeviations);
      expect(typeof threshold).toBe('number');
      expect(threshold).toBeGreaterThan(0);
    });
  });

  describe('getFallbackThreshold', () => {
    it('should calculate fallback threshold', () => {
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

