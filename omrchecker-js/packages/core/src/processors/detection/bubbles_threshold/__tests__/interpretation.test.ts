/**
 * Tests for BubblesFieldInterpretation and BubbleInterpretation.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { BubbleField } from '../../../layout/field/bubbleField';
import { FieldBlock } from '../../../layout/fieldBlock/base';
import { BubbleInterpretation, BubblesFieldInterpretation } from '../interpretation';
import { BubbleMeanValue, BubbleFieldDetectionResult, type BubbleLocation } from '../../models/detectionResults';
import type { BubblesScanBox } from '../../../layout/field/bubbleField';

// Helper to create mock BubbleLocation (with bubbleValue for interpretation)
function createMockBubbleLocation(
  x: number,
  y: number,
  bubbleValue: string = 'A',
  width: number = 10,
  height: number = 10
): BubbleLocation & { bubbleValue: string } {
  return {
    x,
    y,
    width,
    height,
    label: `bubble_${x}_${y}`,
    bubbleValue,
  };
}

// Helper to create mock field
function createMockField(fieldLabel: string = 'q1', emptyValue: string = ''): BubbleField {
  const mockFieldBlock = {
    name: 'testBlock',
    shifts: [0, 0],
    bubbleDimensions: [10, 10] as [number, number],
    bubbleValues: ['A', 'B', 'C'],
    bubblesGap: 5,
    bubbleFieldType: 'QTYPE_MCQ4',
  } as FieldBlock;

  return new BubbleField(
    'horizontal',
    emptyValue,
    mockFieldBlock,
    'BUBBLES_THRESHOLD',
    fieldLabel,
    [0, 0]
  );
}

// Helper to create mock tuning config
function createMockTuningConfig(showConfidenceMetrics: boolean = false): Record<string, unknown> {
  return {
    thresholding: {
      min_jump: 10,
      jump_delta: 5,
      min_gap_two_bubbles: 20,
      min_jump_surplus_for_global_fallback: 15,
      confident_jump_surplus_for_disparity: 10,
      global_threshold_margin: 5,
      global_page_threshold: 180,
    },
    outputs: {
      show_confidence_metrics: showConfidenceMetrics,
    },
  };
}

describe('BubbleInterpretation', () => {
  describe('constructor', () => {
    it('should create instance with marked bubble', () => {
      const bubble = createMockBubbleLocation(0, 0, 'A');
      const bubbleMean = new BubbleMeanValue(50.0, bubble, [0, 0]);
      const threshold = 100.0;

      const interpretation = new BubbleInterpretation(bubbleMean, threshold);

      expect(interpretation.bubbleMean).toBe(bubbleMean);
      expect(interpretation.threshold).toBe(threshold);
      expect(interpretation.meanValue).toBe(50.0);
      expect(interpretation.isAttempted).toBe(true);
      expect(interpretation.bubbleValue).toBe('A');
      expect(interpretation.itemReference).toBe(bubble);
    });

    it('should create instance with unmarked bubble', () => {
      const bubble = createMockBubbleLocation(0, 0, 'B');
      const bubbleMean = new BubbleMeanValue(200.0, bubble, [0, 0]);
      const threshold = 100.0;

      const interpretation = new BubbleInterpretation(bubbleMean, threshold);

      expect(interpretation.meanValue).toBe(200.0);
      expect(interpretation.isAttempted).toBe(false);
      expect(interpretation.bubbleValue).toBe('B');
      expect(interpretation.itemReference).toBe(bubble);
    });

    it('should extract bubble value from unitBubble', () => {
      const bubble = createMockBubbleLocation(0, 0, 'C');
      const bubbleMean = new BubbleMeanValue(80.0, bubble, [0, 0]);
      const threshold = 100.0;

      const interpretation = new BubbleInterpretation(bubbleMean, threshold);

      expect(interpretation.bubbleValue).toBe('C');
    });

    it('should handle missing bubbleValue gracefully', () => {
      const bubble: BubbleLocation = {
        x: 0,
        y: 0,
        width: 10,
        height: 10,
        label: 'bubble_0_0',
      };
      const bubbleMean = new BubbleMeanValue(80.0, bubble, [0, 0]);
      const threshold = 100.0;

      const interpretation = new BubbleInterpretation(bubbleMean, threshold);

      expect(interpretation.bubbleValue).toBe('');
    });
  });

  describe('getValue', () => {
    it('should return bubble value when marked', () => {
      const bubble = createMockBubbleLocation(0, 0, 'A');
      const bubbleMean = new BubbleMeanValue(50.0, bubble, [0, 0]);
      const threshold = 100.0;

      const interpretation = new BubbleInterpretation(bubbleMean, threshold);

      expect(interpretation.getValue()).toBe('A');
    });

    it('should return empty string when unmarked', () => {
      const bubble = createMockBubbleLocation(0, 0, 'B');
      const bubbleMean = new BubbleMeanValue(200.0, bubble, [0, 0]);
      const threshold = 100.0;

      const interpretation = new BubbleInterpretation(bubbleMean, threshold);

      expect(interpretation.getValue()).toBe('');
    });
  });
});

describe('BubblesFieldInterpretation', () => {
  let tuningConfig: Record<string, unknown>;
  let mockField: BubbleField;

  beforeEach(() => {
    tuningConfig = createMockTuningConfig();
    mockField = createMockField('q1', '');
  });

  describe('getFieldInterpretationString', () => {
    it('should return empty when no bubbles marked', () => {
      const bubble1 = createMockBubbleLocation(0, 0, 'A');
      const bubble2 = createMockBubbleLocation(10, 0, 'B');
      const bubbleMean1 = new BubbleMeanValue(200.0, bubble1, [0, 0]);  // Unmarked (high value)
      const bubbleMean2 = new BubbleMeanValue(210.0, bubble2, [10, 0]);  // Unmarked (high value)
      const detectionResult = new BubbleFieldDetectionResult('q1', 'q1', [bubbleMean1, bubbleMean2]);

      const fileLevelDetectionAggregates = {
        bubble_fields: {
          q1: detectionResult,
        },
      };
      const fileLevelInterpretationAggregates = {
        file_level_fallback_threshold: 180,
      };

      const interpretation = new BubblesFieldInterpretation(
        tuningConfig,
        mockField,
        fileLevelDetectionAggregates,
        fileLevelInterpretationAggregates
      );

      const result = interpretation.getFieldInterpretationString();
      expect(result).toBe('');
    });

    it('should return single value when one bubble marked', () => {
      const bubble1 = createMockBubbleLocation(0, 0, 'A');
      const bubble2 = createMockBubbleLocation(10, 0, 'B');
      const bubbleMean1 = new BubbleMeanValue(50.0, bubble1, [0, 0]); // Marked
      const bubbleMean2 = new BubbleMeanValue(200.0, bubble2, [10, 0]); // Unmarked
      const detectionResult = new BubbleFieldDetectionResult('q1', 'q1', [bubbleMean1, bubbleMean2]);

      const fileLevelDetectionAggregates = {
        bubble_fields: {
          q1: detectionResult,
        },
      };
      const fileLevelInterpretationAggregates = {
        file_level_fallback_threshold: 180,
      };

      const interpretation = new BubblesFieldInterpretation(
        tuningConfig,
        mockField,
        fileLevelDetectionAggregates,
        fileLevelInterpretationAggregates
      );

      const result = interpretation.getFieldInterpretationString();
      expect(result).toBe('A');
    });

    it('should return concatenated values when multiple marked', () => {
      const bubble1 = createMockBubbleLocation(0, 0, 'A');
      const bubble2 = createMockBubbleLocation(10, 0, 'B');
      const bubble3 = createMockBubbleLocation(20, 0, 'C');
      const bubbleMean1 = new BubbleMeanValue(50.0, bubble1, [0, 0]); // Marked
      const bubbleMean2 = new BubbleMeanValue(60.0, bubble2, [10, 0]); // Marked
      const bubbleMean3 = new BubbleMeanValue(200.0, bubble3, [20, 0]); // Unmarked
      const detectionResult = new BubbleFieldDetectionResult('q1', 'q1', [
        bubbleMean1,
        bubbleMean2,
        bubbleMean3,
      ]);

      const fileLevelDetectionAggregates = {
        bubble_fields: {
          q1: detectionResult,
        },
      };
      const fileLevelInterpretationAggregates = {
        file_level_fallback_threshold: 180,
      };

      const interpretation = new BubblesFieldInterpretation(
        tuningConfig,
        mockField,
        fileLevelDetectionAggregates,
        fileLevelInterpretationAggregates
      );

      const result = interpretation.getFieldInterpretationString();
      expect(result).toBe('AB');
    });

    it('should return empty when all bubbles marked', () => {
      const bubble1 = createMockBubbleLocation(0, 0, 'A');
      const bubble2 = createMockBubbleLocation(10, 0, 'B');
      const bubbleMean1 = new BubbleMeanValue(50.0, bubble1, [0, 0]); // Marked
      const bubbleMean2 = new BubbleMeanValue(60.0, bubble2, [10, 0]); // Marked
      const detectionResult = new BubbleFieldDetectionResult('q1', 'q1', [bubbleMean1, bubbleMean2]);

      const fileLevelDetectionAggregates = {
        bubble_fields: {
          q1: detectionResult,
        },
      };
      const fileLevelInterpretationAggregates = {
        file_level_fallback_threshold: 180,
      };

      const interpretation = new BubblesFieldInterpretation(
        tuningConfig,
        mockField,
        fileLevelDetectionAggregates,
        fileLevelInterpretationAggregates
      );

      // All bubbles marked should return empty (likely scanning issue)
      const result = interpretation.getFieldInterpretationString();
      expect(result).toBe('');
    });
  });

  describe('runInterpretation', () => {
    it('should extract detection result', () => {
      const bubble1 = createMockBubbleLocation(0, 0, 'A');
      const bubbleMean1 = new BubbleMeanValue(50.0, bubble1, [0, 0]);
      const detectionResult = new BubbleFieldDetectionResult('q1', 'q1', [bubbleMean1]);

      const fileLevelDetectionAggregates = {
        bubble_fields: {
          q1: detectionResult,
        },
      };
      const fileLevelInterpretationAggregates = {
        file_level_fallback_threshold: 180,
      };

      // Verify the field label matches
      expect(mockField.fieldLabel).toBe('q1');

      const interpretation = new BubblesFieldInterpretation(
        tuningConfig,
        mockField,
        fileLevelDetectionAggregates,
        fileLevelInterpretationAggregates
      );


      // Verify detection result was extracted and interpretation ran
      expect(interpretation.bubbleInterpretations.length).toBe(1);
      expect(interpretation.bubbleInterpretations[0].bubbleMean).toBe(bubbleMean1);
      expect(interpretation.localThresholdForField).toBeGreaterThan(0);
    });

    it('should calculate threshold', () => {
      const bubble1 = createMockBubbleLocation(0, 0, 'A');
      const bubble2 = createMockBubbleLocation(10, 0, 'B');
      const bubbleMean1 = new BubbleMeanValue(50.0, bubble1, [0, 0]);
      const bubbleMean2 = new BubbleMeanValue(200.0, bubble2, [10, 0]);
      const detectionResult = new BubbleFieldDetectionResult('q1', 'q1', [bubbleMean1, bubbleMean2]);

      const fileLevelDetectionAggregates = {
        bubble_fields: {
          q1: detectionResult,
        },
      };
      const fileLevelInterpretationAggregates = {
        file_level_fallback_threshold: 180,
      };

      const interpretation = new BubblesFieldInterpretation(
        tuningConfig,
        mockField,
        fileLevelDetectionAggregates,
        fileLevelInterpretationAggregates
      );

      // Verify threshold was calculated
      expect(interpretation.localThresholdForField).toBeGreaterThan(0);
      expect(interpretation.thresholdResult).not.toBeNull();
      expect(interpretation.thresholdResult?.thresholdValue).toBeGreaterThan(0);
    });

    it('should create bubble interpretations', () => {
      const bubble1 = createMockBubbleLocation(0, 0, 'A');
      const bubble2 = createMockBubbleLocation(10, 0, 'B');
      const bubbleMean1 = new BubbleMeanValue(50.0, bubble1, [0, 0]);
      const bubbleMean2 = new BubbleMeanValue(200.0, bubble2, [10, 0]);
      const detectionResult = new BubbleFieldDetectionResult('q1', 'q1', [bubbleMean1, bubbleMean2]);

      const fileLevelDetectionAggregates = {
        bubble_fields: {
          q1: detectionResult,
        },
      };
      const fileLevelInterpretationAggregates = {
        file_level_fallback_threshold: 180,
      };

      const interpretation = new BubblesFieldInterpretation(
        tuningConfig,
        mockField,
        fileLevelDetectionAggregates,
        fileLevelInterpretationAggregates
      );

      // Verify bubble interpretations were created
      expect(interpretation.bubbleInterpretations.length).toBe(2);
      expect(
        interpretation.bubbleInterpretations.every(
          (interp) => interp instanceof BubbleInterpretation
        )
      ).toBe(true);
    });

    it('should detect multi-marking', () => {
      const bubble1 = createMockBubbleLocation(0, 0, 'A');
      const bubble2 = createMockBubbleLocation(10, 0, 'B');
      const bubble3 = createMockBubbleLocation(20, 0, 'C');
      const bubbleMean1 = new BubbleMeanValue(50.0, bubble1, [0, 0]); // Marked
      const bubbleMean2 = new BubbleMeanValue(60.0, bubble2, [10, 0]); // Marked
      const bubbleMean3 = new BubbleMeanValue(200.0, bubble3, [20, 0]); // Unmarked
      const detectionResult = new BubbleFieldDetectionResult('q1', 'q1', [
        bubbleMean1,
        bubbleMean2,
        bubbleMean3,
      ]);

      const fileLevelDetectionAggregates = {
        bubble_fields: {
          q1: detectionResult,
        },
      };
      const fileLevelInterpretationAggregates = {
        file_level_fallback_threshold: 180,
      };

      const interpretation = new BubblesFieldInterpretation(
        tuningConfig,
        mockField,
        fileLevelDetectionAggregates,
        fileLevelInterpretationAggregates
      );

      expect(interpretation.isMultiMarked).toBe(true);
    });

    it('should calculate confidence metrics when enabled', () => {
      tuningConfig = createMockTuningConfig(true);
      const bubble1 = createMockBubbleLocation(0, 0, 'A');
      const bubble2 = createMockBubbleLocation(10, 0, 'B');
      const bubbleMean1 = new BubbleMeanValue(50.0, bubble1, [0, 0]);
      const bubbleMean2 = new BubbleMeanValue(200.0, bubble2, [10, 0]);
      const detectionResult = new BubbleFieldDetectionResult('q1', 'q1', [bubbleMean1, bubbleMean2]);

      const fileLevelDetectionAggregates = {
        bubble_fields: {
          q1: detectionResult,
        },
      };
      const fileLevelInterpretationAggregates = {
        file_level_fallback_threshold: 180,
      };

      const interpretation = new BubblesFieldInterpretation(
        tuningConfig,
        mockField,
        fileLevelDetectionAggregates,
        fileLevelInterpretationAggregates
      );

      // Verify confidence metrics were calculated
      const metrics = interpretation.getFieldLevelConfidenceMetrics();
      expect(metrics.overall_confidence_score).toBeDefined();
      expect(metrics.local_threshold).toBeDefined();
      expect(metrics.global_threshold).toBeDefined();
    });
  });

  describe('extractDetectionResult validation', () => {
    it('should throw error when bubble_fields is missing', () => {
      const fileLevelDetectionAggregates = {}; // Missing bubble_fields
      const fileLevelInterpretationAggregates = {
        file_level_fallback_threshold: 180,
      };

      expect(() => {
        new BubblesFieldInterpretation(
          tuningConfig,
          mockField,
          fileLevelDetectionAggregates,
          fileLevelInterpretationAggregates
        );
      }).toThrow(/No detection result for field 'q1'/);
    });

    it('should throw error when field not in bubble_fields', () => {
      const fileLevelDetectionAggregates = {
        bubble_fields: {
          q2: new BubbleFieldDetectionResult('q2', 'q2', [
            new BubbleMeanValue(100.0, createMockBubbleLocation(0, 0, 'A'), [0, 0]),
          ]),
        },
      };
      const fileLevelInterpretationAggregates = {
        file_level_fallback_threshold: 180,
      };

      expect(() => {
        new BubblesFieldInterpretation(
          tuningConfig,
          mockField, // field label is 'q1'
          fileLevelDetectionAggregates,
          fileLevelInterpretationAggregates
        );
      }).toThrow(/No detection result for field 'q1'/);
      expect(() => {
        new BubblesFieldInterpretation(
          tuningConfig,
          mockField,
          fileLevelDetectionAggregates,
          fileLevelInterpretationAggregates
        );
      }).toThrow(/Available: \[q2\]/);
    });

    it('should throw error when detection result has no bubble means', () => {
      const detectionResult = new BubbleFieldDetectionResult('q1', 'q1', []);
      const fileLevelDetectionAggregates = {
        bubble_fields: { q1: detectionResult },
      };
      const fileLevelInterpretationAggregates = {
        file_level_fallback_threshold: 180,
      };

      expect(() => {
        new BubblesFieldInterpretation(
          tuningConfig,
          mockField,
          fileLevelDetectionAggregates,
          fileLevelInterpretationAggregates
        );
      }).toThrow(/No bubble means in detection result for field 'q1'/);
    });
  });

  describe('checkMultiMarking', () => {
    it('should set isMultiMarked correctly', () => {
      // Test single mark
      const bubble1 = createMockBubbleLocation(0, 0, 'A');
      const bubble2 = createMockBubbleLocation(10, 0, 'B');
      const bubbleMean1 = new BubbleMeanValue(50.0, bubble1, [0, 0]); // Marked
      const bubbleMean2 = new BubbleMeanValue(200.0, bubble2, [10, 0]); // Unmarked
      const detectionResult = new BubbleFieldDetectionResult('q1', 'q1', [bubbleMean1, bubbleMean2]);

      const fileLevelDetectionAggregates = {
        bubble_fields: {
          q1: detectionResult,
        },
      };
      const fileLevelInterpretationAggregates = {
        file_level_fallback_threshold: 180,
      };

      const interpretation = new BubblesFieldInterpretation(
        tuningConfig,
        mockField,
        fileLevelDetectionAggregates,
        fileLevelInterpretationAggregates
      );

      expect(interpretation.isMultiMarked).toBe(false);

      // Test multiple marks
      const bubble3 = createMockBubbleLocation(20, 0, 'C');
      const bubbleMean3 = new BubbleMeanValue(60.0, bubble3, [20, 0]); // Marked
      const detectionResult2 = new BubbleFieldDetectionResult('q1', 'q1', [
        bubbleMean1,
        bubbleMean2,
        bubbleMean3,
      ]);

      const fileLevelDetectionAggregates2 = {
        bubble_fields: {
          q1: detectionResult2,
        },
      };

      const interpretation2 = new BubblesFieldInterpretation(
        tuningConfig,
        mockField,
        fileLevelDetectionAggregates2,
        fileLevelInterpretationAggregates
      );

      expect(interpretation2.isMultiMarked).toBe(true);
    });
  });

  describe('calculateOverallConfidenceScore', () => {
    it('should return valid score', () => {
      tuningConfig = createMockTuningConfig(true);
      // Create detection result with good separation (high confidence scenario)
      const bubble1 = createMockBubbleLocation(0, 0, 'A');
      const bubble2 = createMockBubbleLocation(10, 0, 'B');
      const bubbleMean1 = new BubbleMeanValue(50.0, bubble1, [0, 0]); // Clearly marked
      const bubbleMean2 = new BubbleMeanValue(200.0, bubble2, [10, 0]); // Clearly unmarked
      const detectionResult = new BubbleFieldDetectionResult('q1', 'q1', [bubbleMean1, bubbleMean2]);

      const fileLevelDetectionAggregates = {
        bubble_fields: {
          q1: detectionResult,
        },
      };
      const fileLevelInterpretationAggregates = {
        file_level_fallback_threshold: 180,
      };

      const interpretation = new BubblesFieldInterpretation(
        tuningConfig,
        mockField,
        fileLevelDetectionAggregates,
        fileLevelInterpretationAggregates
      );

      // Verify confidence score is calculated and in valid range
      const metrics = interpretation.getFieldLevelConfidenceMetrics();
      const confidenceScore = metrics.overall_confidence_score;
      expect(confidenceScore).toBeDefined();
      expect(confidenceScore).toBeGreaterThanOrEqual(0.0);
      expect(confidenceScore).toBeLessThanOrEqual(1.0);
    });
  });
});
