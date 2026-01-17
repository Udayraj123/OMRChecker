/**
 * Tests for interpretation pass base classes.
 */

import { FieldDetectionType } from '../../../constants';
import { BubbleField } from '../../../layout/field/bubbleField';
import { FieldBlock } from '../../../layout/fieldBlock/base';
import { FieldTypeInterpretationPass, TemplateInterpretationPass } from '../interpretationPass';
import { BaseInterpretation, FieldInterpretation } from '../interpretation';
import type { Field } from '../../../layout/field/base';
import { TextDetection } from '../detection';

// Mock FieldInterpretation for testing
class MockFieldInterpretation extends FieldInterpretation {
  getDrawingInstance() {
    return {};
  }

  runInterpretation(
    _field: Field,
    _fileLevelDetectionAggregates: unknown,
    _fileLevelInterpretationAggregates: unknown
  ): void {
    // Mock implementation
  }

  getFieldInterpretationString(): string {
    return 'mock_answer';
  }
}

// Mock FieldTypeInterpretationPass for testing
class MockFieldTypeInterpretationPass extends FieldTypeInterpretationPass {
  getFieldInterpretation(
    field: Field,
    _fileLevelDetectionAggregates: unknown,
    _fileLevelInterpretationAggregates: unknown
  ): FieldInterpretation {
    return new MockFieldInterpretation({}, field, {}, {});
  }
}

describe('BaseInterpretation', () => {
  describe('constructor', () => {
    it('should initialize with text detection', () => {
      const textDetection = new TextDetection('test', [0, 0, 10, 10], null, 0.9);
      const interpretation = new BaseInterpretation(textDetection);
      expect(interpretation.isAttempted).toBe(true);
      expect(interpretation.detectedText).toBe('test');
    });

    it('should handle null text detection', () => {
      const interpretation = new BaseInterpretation(null);
      expect(interpretation.isAttempted).toBe(false);
      expect(interpretation.detectedText).toBe('');
    });
  });

  describe('getValue', () => {
    it('should return detected text', () => {
      const textDetection = new TextDetection('answer', [0, 0, 10, 10], null, 0.9);
      const interpretation = new BaseInterpretation(textDetection);
      expect(interpretation.getValue()).toBe('answer');
    });
  });
});

describe('FieldTypeInterpretationPass', () => {
  let tuningConfig: Record<string, unknown>;
  let pass: MockFieldTypeInterpretationPass;
  let mockField: BubbleField;

  beforeEach(() => {
    tuningConfig = {};
    pass = new MockFieldTypeInterpretationPass(tuningConfig, FieldDetectionType.BUBBLES_THRESHOLD);

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
    it('should initialize with tuning config and field detection type', () => {
      expect(pass.fieldDetectionType).toBe(FieldDetectionType.BUBBLES_THRESHOLD);
    });
  });

  describe('getFieldInterpretation', () => {
    it('should return FieldInterpretation instance', () => {
      const interpretation = pass.getFieldInterpretation(mockField, {}, {});
      expect(interpretation).toBeInstanceOf(MockFieldInterpretation);
    });
  });

  describe('runFieldLevelInterpretation', () => {
    it('should run field level interpretation', () => {
      pass.initializeFileLevelAggregates('test', {}, {});
      const interpretation = pass.runFieldLevelInterpretation(mockField, {});
      expect(interpretation).toBeInstanceOf(MockFieldInterpretation);
    });
  });

  describe('updateAggregatesOnProcessedFieldInterpretation', () => {
    it('should update all aggregate levels', () => {
      pass.initializeFieldLevelAggregates(mockField);
      pass.initializeFileLevelAggregates('/test/file.jpg', {}, {});
      pass.initializeDirectoryLevelAggregates('/test/path');

      const interpretation = pass.getFieldInterpretation(mockField, {}, {});
      expect(() => {
        pass.updateAggregatesOnProcessedFieldInterpretation(mockField, interpretation);
      }).not.toThrow();
    });
  });
});

describe('TemplateInterpretationPass', () => {
  let tuningConfig: Record<string, unknown>;
  let pass: TemplateInterpretationPass;
  let mockField: BubbleField;

  beforeEach(() => {
    tuningConfig = {};
    pass = new TemplateInterpretationPass(tuningConfig);

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

  describe('initializeDirectoryLevelAggregates', () => {
    it('should initialize with field detection type tracking', () => {
      pass.initializeDirectoryLevelAggregates('/test/path', [
        FieldDetectionType.BUBBLES_THRESHOLD,
      ]);

      const dirAgg = pass.getDirectoryLevelAggregates();
      expect(dirAgg).toBeDefined();
      expect(dirAgg?.field_detection_type_wise_aggregates).toBeDefined();
    });
  });

  describe('initializeFileLevelAggregates', () => {
    it('should initialize with field detection type tracking', () => {
      pass.initializeFileLevelAggregates('/test/file.jpg', [
        FieldDetectionType.BUBBLES_THRESHOLD,
      ]);

      const fileAgg = pass.getFileLevelAggregates();
      expect(fileAgg).toBeDefined();
      expect(fileAgg?.field_detection_type_wise_aggregates).toBeDefined();
    });
  });

  describe('runFieldLevelInterpretation', () => {
    it('should run field level interpretation and update response', () => {
      pass.initializeFileLevelAggregates('/test/file.jpg', [
        FieldDetectionType.BUBBLES_THRESHOLD,
      ]);

      const interpretation = new MockFieldInterpretation({}, mockField, {}, {});
      const currentOmrResponse: Record<string, string> = {};

      pass.runFieldLevelInterpretation(mockField, interpretation, {}, currentOmrResponse);

      expect(currentOmrResponse[mockField.fieldLabel]).toBe('mock_answer');
    });
  });
});

