/**
 * Comprehensive tests for template validation.
 *
 * Tests schema validation, field block validation, custom label validation,
 * and output columns validation.
 * Ported from Python test_template_validations.py
 */

import { describe, it, expect } from 'vitest';
import { validateTemplate } from '../../schemas/templateSchema';
import { TemplateLoader } from '../TemplateLoader';
import { TemplateLayout } from '../TemplateLayout';
import { OMRCheckerError, FieldDefinitionError } from '../../core/exceptions';
import type { TemplateConfig } from '../types';

/**
 * Create minimal valid template config for testing.
 */
function createMinimalTemplateConfig(): TemplateConfig {
  return {
    templateDimensions: [300, 400],
    bubbleDimensions: [20, 20],
    emptyValue: '',
    fieldBlocksOffset: [0, 0],
    fieldBlocks: {
      MCQ_Block_1: {
        fieldDetectionType: 'BUBBLES_THRESHOLD',
        origin: [65, 60],
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

describe('Template Validations', () => {
  describe('Schema Validation', () => {
    it('should reject empty template', () => {
      const result = validateTemplate({});

      expect(result.valid).toBe(false);
      expect(result.errors.length).toBeGreaterThan(0);
    });

    it('should reject invalid bubble field type', () => {
      const config = createMinimalTemplateConfig();
      config.fieldBlocks = {
        MCQ_Block_1: {
          ...config.fieldBlocks.MCQ_Block_1,
          bubbleFieldType: 'X', // Invalid type
        },
      };

      const result = validateTemplate(config);

      expect(result.valid).toBe(false);
      expect(result.errors.length).toBeGreaterThan(0);
    });

    it('should reject invalid sort type', () => {
      const config = createMinimalTemplateConfig();
      config.outputColumns = {
        sortType: 'ABC', // Invalid sort type
        customOrder: [],
      };

      const result = validateTemplate(config);

      expect(result.valid).toBe(false);
      expect(result.errors.length).toBeGreaterThan(0);
    });

    it('should reject invalid sort order property', () => {
      const config = createMinimalTemplateConfig();
      config.outputColumns = {
        sortOrder: 'ABC', // Invalid property (should be sortType)
      } as any;

      const result = validateTemplate(config);

      expect(result.valid).toBe(false);
      expect(result.errors.length).toBeGreaterThan(0);
    });
  });

  describe('Field Block Validation', () => {
    it('should detect overflowing field blocks', () => {
      const config = createMinimalTemplateConfig();
      config.fieldBlocks = {
        MCQ_Block_1: {
          ...config.fieldBlocks.MCQ_Block_1,
          fieldLabels: ['q1..100'], // Will create many fields that overflow
        },
      };

      expect(() => {
        const layoutData = TemplateLoader.loadLayoutFromJSON(config);
        const layout = new TemplateLayout(layoutData, config);
      }).toThrow();
    });

    it('should accept safe dimensions', () => {
      const config = createMinimalTemplateConfig();
      config.templateDimensions = [255, 400]; // Larger template

      // Should not throw
      expect(() => {
        const layoutData = TemplateLoader.loadLayoutFromJSON(config);
        const layout = new TemplateLayout(layoutData, config);
      }).not.toThrow();
    });

    it('should detect overlapping field labels', () => {
      const config = createMinimalTemplateConfig();
      config.fieldBlocks = {
        ...config.fieldBlocks,
        New_Block: {
          ...config.fieldBlocks.MCQ_Block_1,
          fieldLabels: ['q1'], // Overlaps with MCQ_Block_1
        },
      };

      expect(() => {
        const layoutData = TemplateLoader.loadLayoutFromJSON(config);
        const layout = new TemplateLayout(layoutData, config);
      }).toThrow();
    });

    it('should detect missing labels gap', () => {
      const config = createMinimalTemplateConfig();
      config.fieldBlocks = {
        MCQ_Block_1: {
          ...config.fieldBlocks.MCQ_Block_1,
          fieldLabels: ['q1', 'q2', 'q3'], // Multiple labels
          labelsGap: undefined, // Missing labelsGap
        },
      };

      expect(() => {
        const layoutData = TemplateLoader.loadLayoutFromJSON(config);
        const layout = new TemplateLayout(layoutData, config);
      }).toThrow(OMRCheckerError);
    });
  });

  describe('Custom Label Validation', () => {
    it('should detect overlapping fields within same custom label', () => {
      const config = createMinimalTemplateConfig();
      config.customLabels = {
        label1: ['q1..2', 'q2..3'], // q2 overlaps
      };

      expect(() => {
        const layoutData = TemplateLoader.loadLayoutFromJSON(config);
        const layout = new TemplateLayout(layoutData, config);
      }).toThrow();
    });

    it('should detect overlapping fields between different custom labels', () => {
      const config = createMinimalTemplateConfig();
      config.customLabels = {
        label1: ['q1..2'],
        label2: ['q2..3'], // q2 overlaps with label1
      };

      expect(() => {
        const layoutData = TemplateLoader.loadLayoutFromJSON(config);
        const layout = new TemplateLayout(layoutData, config);
      }).toThrow();
    });

    it('should detect missing field block labels in custom labels', () => {
      const config = createMinimalTemplateConfig();
      config.customLabels = {
        Combined: ['qX', 'qY'], // qX and qY don't exist
      };

      expect(() => {
        const layoutData = TemplateLoader.loadLayoutFromJSON(config);
        const layout = new TemplateLayout(layoutData, config);
      }).toThrow();
    });

    it('should accept valid custom labels', () => {
      const config = createMinimalTemplateConfig();
      config.customLabels = {
        Combined: ['q1', 'q2'], // Valid labels
      };

      // Should not throw
      expect(() => {
        const layoutData = TemplateLoader.loadLayoutFromJSON(config);
        const layout = new TemplateLayout(layoutData, config);
      }).not.toThrow();
    });
  });

  describe('Output Columns Validation', () => {
    it('should detect missing output columns', () => {
      const config = createMinimalTemplateConfig();
      config.outputColumns = {
        sortType: 'CUSTOM',
        customOrder: ['qX', 'q1', 'q2'], // qX doesn't exist
      };

      expect(() => {
        const layoutData = TemplateLoader.loadLayoutFromJSON(config);
        const layout = new TemplateLayout(layoutData, config);
      }).toThrow(FieldDefinitionError);
    });

    it('should accept valid output columns', () => {
      const config = createMinimalTemplateConfig();
      config.outputColumns = {
        sortType: 'CUSTOM',
        customOrder: ['q1', 'q2'], // Valid columns
      };

      // Should not throw
      expect(() => {
        const layoutData = TemplateLoader.loadLayoutFromJSON(config);
        const layout = new TemplateLayout(layoutData, config);
      }).not.toThrow();
    });

    it('should accept range syntax in output columns', () => {
      const config = createMinimalTemplateConfig();
      config.fieldBlocks = {
        MCQ_Block_1: {
          ...config.fieldBlocks.MCQ_Block_1,
          fieldLabels: ['q1..4'], // Range syntax
        },
      };
      config.outputColumns = {
        sortType: 'CUSTOM',
        customOrder: ['q1..4'], // Range syntax
      };

      // Should not throw
      expect(() => {
        const layoutData = TemplateLoader.loadLayoutFromJSON(config);
        const layout = new TemplateLayout(layoutData, config);
      }).not.toThrow();
    });
  });

  describe('TemplateLoader Validation', () => {
    it('should reject missing template dimensions', () => {
      const config = createMinimalTemplateConfig();
      delete (config as any).templateDimensions;

      expect(() => {
        TemplateLoader.loadLayoutFromJSON(config);
      }).toThrow();
    });

    it('should reject missing bubble dimensions', () => {
      const config = createMinimalTemplateConfig();
      delete (config as any).bubbleDimensions;

      expect(() => {
        TemplateLoader.loadLayoutFromJSON(config);
      }).toThrow();
    });

    it('should reject empty field blocks', () => {
      const config = createMinimalTemplateConfig();
      config.fieldBlocks = {};

      expect(() => {
        TemplateLoader.loadLayoutFromJSON(config);
      }).toThrow();
    });

    it('should reject invalid field detection type', () => {
      const config = createMinimalTemplateConfig();
      config.fieldBlocks = {
        MCQ_Block_1: {
          ...config.fieldBlocks.MCQ_Block_1,
          fieldDetectionType: 'INVALID_TYPE',
        },
      };

      expect(() => {
        TemplateLoader.loadLayoutFromJSON(config);
      }).toThrow();
    });
  });

  describe('TemplateLayout Validation', () => {
    it('should validate field blocks with invalid bubble field type', () => {
      const config = createMinimalTemplateConfig();
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      const invalidFieldBlocks = {
        Block1: {
          fieldDetectionType: 'BUBBLES_THRESHOLD',
          origin: [100, 100],
          fieldLabels: ['q1'],
          bubbleFieldType: 'INVALID_TYPE', // Invalid type
          bubblesGap: 30,
          labelsGap: 50,
        },
      };

      expect(() => {
        layout.validateFieldBlocks(invalidFieldBlocks);
      }).toThrow(OMRCheckerError);
    });

    it('should validate template columns with missing columns', () => {
      const config = createMinimalTemplateConfig();
      const layoutData = TemplateLoader.loadLayoutFromJSON(config);
      const layout = new TemplateLayout(layoutData, config);

      // Set invalid output columns
      layout.outputColumns = ['qX', 'qY']; // Invalid columns

      expect(() => {
        layout.validateTemplateColumns(['q1', 'q2'], []);
      }).toThrow(FieldDefinitionError);
    });
  });

  describe('Edge Cases', () => {
    it('should handle zero template dimensions', () => {
      const config = createMinimalTemplateConfig();
      config.templateDimensions = [0, 0];

      expect(() => {
        TemplateLoader.loadLayoutFromJSON(config);
      }).toThrow();
    });

    it('should handle zero bubble dimensions', () => {
      const config = createMinimalTemplateConfig();
      config.bubbleDimensions = [0, 0];

      expect(() => {
        TemplateLoader.loadLayoutFromJSON(config);
      }).toThrow();
    });

    it('should handle negative template dimensions', () => {
      const config = createMinimalTemplateConfig();
      config.templateDimensions = [-100, -200];

      const result = validateTemplate(config);

      expect(result.valid).toBe(false);
    });

    it('should handle negative bubble dimensions', () => {
      const config = createMinimalTemplateConfig();
      config.bubbleDimensions = [-10, -10];

      const result = validateTemplate(config);

      expect(result.valid).toBe(false);
    });

    it('should handle empty field labels', () => {
      const config = createMinimalTemplateConfig();
      config.fieldBlocks = {
        MCQ_Block_1: {
          ...config.fieldBlocks.MCQ_Block_1,
          fieldLabels: [], // Empty labels
        },
      };

      expect(() => {
        TemplateLoader.loadLayoutFromJSON(config);
      }).toThrow();
    });

    it('should handle empty custom labels', () => {
      const config = createMinimalTemplateConfig();
      config.customLabels = {};

      // Should not throw (empty custom labels are valid)
      expect(() => {
        const layoutData = TemplateLoader.loadLayoutFromJSON(config);
        const layout = new TemplateLayout(layoutData, config);
      }).not.toThrow();
    });
  });
});

