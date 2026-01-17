/**
 * Tests for FilePassAggregates base class.
 */

import { BubbleField } from '../../../layout/field/bubbleField';
import { FieldBlock } from '../../../layout/fieldBlock/base';
import { StatsByLabel } from '../../../../utils/stats';
import { FilePassAggregates, type TuningConfig } from '../commonPass';

 describe('FilePassAggregates', () => {
  let tuningConfig: TuningConfig;
  let aggregates: FilePassAggregates;
  let mockFieldBlock: FieldBlock;
  let mockField: BubbleField;

  beforeEach(() => {
    tuningConfig = {};
    aggregates = new FilePassAggregates(tuningConfig);

    // Create a mock field block
    mockFieldBlock = {
      name: 'testBlock',
      shifts: [0, 0],
      bubbleDimensions: [10, 10],
      bubbleValues: ['A', 'B', 'C'],
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
    it('should initialize with tuning config', () => {
      const agg = new FilePassAggregates(tuningConfig);
      expect(agg).toBeInstanceOf(FilePassAggregates);
    });
  });

  describe('directory level aggregates', () => {
    it('should initialize directory level aggregates', () => {
      aggregates.initializeDirectoryLevelAggregates('/test/path');
      const dirAgg = aggregates.getDirectoryLevelAggregates();

      expect(dirAgg).toBeDefined();
      expect(dirAgg?.initial_directory_path).toBe('/test/path');
      expect(dirAgg?.file_wise_aggregates).toEqual({});
      expect(dirAgg?.files_count).toBeInstanceOf(StatsByLabel);
    });

    it('should insert directory level aggregates', () => {
      aggregates.initializeDirectoryLevelAggregates('/test/path');
      aggregates.insertDirectoryLevelAggregates({
        custom_prop: 'value',
      });

      const dirAgg = aggregates.getDirectoryLevelAggregates();
      expect(dirAgg?.custom_prop).toBe('value');
      expect(dirAgg?.initial_directory_path).toBe('/test/path');
    });

    it('should throw error when inserting without initialization', () => {
      expect(() => {
        aggregates.insertDirectoryLevelAggregates({});
      }).toThrow('Directory level aggregates not initialized');
    });
  });

  describe('file level aggregates', () => {
    it('should initialize file level aggregates', () => {
      aggregates.initializeFileLevelAggregates('/test/file.jpg');
      const fileAgg = aggregates.getFileLevelAggregates();

      expect(fileAgg).toBeDefined();
      expect(fileAgg?.file_path).toBe('/test/file.jpg');
      expect(fileAgg?.fields_count).toBeInstanceOf(StatsByLabel);
      expect(fileAgg?.field_label_wise_aggregates).toEqual({});
    });

    it('should insert file level aggregates', () => {
      aggregates.initializeFileLevelAggregates('/test/file.jpg');
      aggregates.insertFileLevelAggregates({
        custom_prop: 'value',
      });

      const fileAgg = aggregates.getFileLevelAggregates();
      expect(fileAgg?.custom_prop).toBe('value');
      expect(fileAgg?.file_path).toBe('/test/file.jpg');
    });

    it('should throw error when inserting without initialization', () => {
      expect(() => {
        aggregates.insertFileLevelAggregates({});
      }).toThrow('File level aggregates not initialized');
    });
  });

  describe('field level aggregates', () => {
    it('should initialize field level aggregates', () => {
      aggregates.initializeFieldLevelAggregates(mockField);
      const fieldAgg = aggregates.getFieldLevelAggregates();

      expect(fieldAgg).toBeDefined();
      expect(fieldAgg?.field).toBe(mockField);
    });

    it('should insert field level aggregates', () => {
      aggregates.initializeFieldLevelAggregates(mockField);
      aggregates.insertFieldLevelAggregates({
        custom_prop: 'value',
      });

      const fieldAgg = aggregates.getFieldLevelAggregates();
      expect(fieldAgg?.custom_prop).toBe('value');
      expect(fieldAgg?.field).toBe(mockField);
    });

    it('should throw error when inserting without initialization', () => {
      expect(() => {
        aggregates.insertFieldLevelAggregates({});
      }).toThrow('Field level aggregates not initialized');
    });
  });

  describe('updateAggregatesOnProcessedFile', () => {
    it('should update directory aggregates with file aggregates', () => {
      aggregates.initializeDirectoryLevelAggregates('/test/path');
      aggregates.initializeFileLevelAggregates('/test/file.jpg');
      aggregates.updateAggregatesOnProcessedFile('/test/file.jpg');

      const dirAgg = aggregates.getDirectoryLevelAggregates();
      expect(dirAgg?.file_wise_aggregates['/test/file.jpg']).toBeDefined();
      expect(dirAgg?.files_count.labelCounts.processed).toBe(1);
    });

    it('should throw error when directory aggregates not initialized', () => {
      aggregates.initializeFileLevelAggregates('/test/file.jpg');
      expect(() => {
        aggregates.updateAggregatesOnProcessedFile('/test/file.jpg');
      }).toThrow('Directory or file level aggregates not initialized');
    });

    it('should throw error when file aggregates not initialized', () => {
      aggregates.initializeDirectoryLevelAggregates('/test/path');
      expect(() => {
        aggregates.updateAggregatesOnProcessedFile('/test/file.jpg');
      }).toThrow('Directory or file level aggregates not initialized');
    });
  });

  describe('updateFileLevelAggregatesOnProcessedField', () => {
    it('should update file aggregates with field aggregates', () => {
      aggregates.initializeFileLevelAggregates('/test/file.jpg');
      aggregates.initializeFieldLevelAggregates(mockField);

      const fieldAgg = aggregates.getFieldLevelAggregates()!;
      aggregates.updateFileLevelAggregatesOnProcessedField(mockField, fieldAgg);

      const fileAgg = aggregates.getFileLevelAggregates();
      expect(
        fileAgg?.field_label_wise_aggregates[mockField.fieldLabel]
      ).toBeDefined();
      expect(fileAgg?.fields_count.labelCounts.processed).toBe(1);
    });

    it('should throw error when file aggregates not initialized', () => {
      aggregates.initializeFieldLevelAggregates(mockField);
      const fieldAgg = aggregates.getFieldLevelAggregates()!;

      expect(() => {
        aggregates.updateFileLevelAggregatesOnProcessedField(mockField, fieldAgg);
      }).toThrow('File level aggregates not initialized');
    });
  });

  describe('updateFieldLevelAggregatesOnProcessedField', () => {
    it('should be callable without error (default implementation)', () => {
      aggregates.initializeFieldLevelAggregates(mockField);
      expect(() => {
        aggregates.updateFieldLevelAggregatesOnProcessedField(mockField);
      }).not.toThrow();
    });
  });

  describe('updateDirectoryLevelAggregatesOnProcessedField', () => {
    it('should be callable without error (default implementation)', () => {
      aggregates.initializeFieldLevelAggregates(mockField);
      const fieldAgg = aggregates.getFieldLevelAggregates()!;

      expect(() => {
        aggregates.updateDirectoryLevelAggregatesOnProcessedField(mockField, fieldAgg);
      }).not.toThrow();
    });
  });
});

