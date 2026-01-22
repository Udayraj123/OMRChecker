/**
 * Tests for detection pass base classes.
 */

const cv = global.cv;
import { FieldDetectionType } from '../../../constants';
import { BubbleField } from '../../../layout/field/bubbleField';
import { FieldBlock } from '../../../layout/fieldBlock/base';
import { FieldTypeDetectionPass, TemplateDetectionPass } from '../detectionPass';
import type { FieldDetection } from '../detection';
import type { Field } from '../../../layout/field/base';

// Mock FieldDetection for testing
class MockFieldDetection implements FieldDetection {
  constructor(
    public field: Field,
    public grayImage: cv.Mat,
    public coloredImage?: cv.Mat
  ) {}
}

// Mock FieldTypeDetectionPass for testing
class MockFieldTypeDetectionPass extends FieldTypeDetectionPass {
  getFieldDetection(
    field: Field,
    grayImage: cv.Mat,
    coloredImage?: cv.Mat
  ): FieldDetection {
    return new MockFieldDetection(field, grayImage, coloredImage);
  }
}

describe('FieldTypeDetectionPass', () => {
  let tuningConfig: Record<string, unknown>;
  let pass: MockFieldTypeDetectionPass;
  let mockField: BubbleField;
  let mockGrayImage: cv.Mat;

  beforeEach(() => {
    tuningConfig = {};
    pass = new MockFieldTypeDetectionPass(tuningConfig, FieldDetectionType.BUBBLES_THRESHOLD);

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
    it('should initialize with tuning config and field detection type', () => {
      expect(pass.fieldDetectionType).toBe(FieldDetectionType.BUBBLES_THRESHOLD);
    });
  });

  describe('getFieldDetection', () => {
    it('should return FieldDetection instance', () => {
      const detection = pass.getFieldDetection(mockField, mockGrayImage);
      expect(detection).toBeInstanceOf(MockFieldDetection);
      expect(detection.field).toBe(mockField);
    });
  });

  describe('updateAggregatesOnProcessedFieldDetection', () => {
    it('should update all aggregate levels', () => {
      pass.initializeFieldLevelAggregates(mockField);
      pass.initializeFileLevelAggregates('/test/file.jpg');
      pass.initializeDirectoryLevelAggregates('/test/path');

      const detection = pass.getFieldDetection(mockField, mockGrayImage);
      expect(() => {
        pass.updateAggregatesOnProcessedFieldDetection(mockField, detection);
      }).not.toThrow();
    });
  });
});

describe('TemplateDetectionPass', () => {
  let tuningConfig: Record<string, unknown>;
  let pass: TemplateDetectionPass;

  beforeEach(() => {
    tuningConfig = {};
    pass = new TemplateDetectionPass(tuningConfig);
  });

  describe('initializeDirectoryLevelAggregates', () => {
    it('should initialize with field detection type tracking', () => {
      pass.initializeDirectoryLevelAggregates('/test/path', [
        FieldDetectionType.BUBBLES_THRESHOLD,
      ]);

      const dirAgg = pass.getDirectoryLevelAggregates();
      expect(dirAgg).toBeDefined();
      expect(dirAgg?.files_by_label_count).toBeDefined();
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
});

