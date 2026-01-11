/**
 * Unit tests for TemplateLoader.
 *
 * Tests template parsing, field label expansion, and bubble location calculation.
 */

import { describe, it, expect } from 'vitest';
import { TemplateLoader } from '../TemplateLoader';
import type { TemplateConfig } from '../types';

describe('TemplateLoader', () => {
  describe('Field Label Expansion', () => {
    it('should expand range syntax (q1..5)', () => {
      const template: TemplateConfig = {
        templateDimensions: [1000, 800],
        bubbleDimensions: [40, 40],
        fieldBlocks: {
          MCQ_Block: {
            origin: [100, 100],
            fieldLabels: ['q1..5'],
            fieldDetectionType: 'BUBBLES_THRESHOLD',
            bubbleFieldType: 'QTYPE_MCQ4',
            bubblesGap: 50,
            labelsGap: 60,
          },
        },
      };

      const parsed = TemplateLoader.loadFromJSON(template);

      expect(parsed.fields.size).toBe(5);
      expect(parsed.fields.has('q1')).toBe(true);
      expect(parsed.fields.has('q2')).toBe(true);
      expect(parsed.fields.has('q3')).toBe(true);
      expect(parsed.fields.has('q4')).toBe(true);
      expect(parsed.fields.has('q5')).toBe(true);
    });

    it('should handle simple labels without expansion', () => {
      const template: TemplateConfig = {
        templateDimensions: [1000, 800],
        bubbleDimensions: [40, 40],
        fieldBlocks: {
          MCQ_Block: {
            origin: [100, 100],
            fieldLabels: ['q1', 'q2', 'q3'],
            fieldDetectionType: 'BUBBLES_THRESHOLD',
            bubbleFieldType: 'QTYPE_MCQ4',
            bubblesGap: 50,
            labelsGap: 60,
          },
        },
      };

      const parsed = TemplateLoader.loadFromJSON(template);

      expect(parsed.fields.size).toBe(3);
      expect(parsed.fields.has('q1')).toBe(true);
      expect(parsed.fields.has('q2')).toBe(true);
      expect(parsed.fields.has('q3')).toBe(true);
    });

    it('should expand range with prefix (roll1..10)', () => {
      const template: TemplateConfig = {
        templateDimensions: [1000, 800],
        bubbleDimensions: [40, 40],
        fieldBlocks: {
          Roll_Block: {
            origin: [100, 100],
            fieldLabels: ['roll1..10'],
            fieldDetectionType: 'BUBBLES_THRESHOLD',
            bubbleFieldType: 'QTYPE_INT',
            bubblesGap: 50,
            labelsGap: 60,
          },
        },
      };

      const parsed = TemplateLoader.loadFromJSON(template);

      expect(parsed.fields.size).toBe(10);
      expect(parsed.fields.has('roll1')).toBe(true);
      expect(parsed.fields.has('roll5')).toBe(true);
      expect(parsed.fields.has('roll10')).toBe(true);
    });

    it('should throw error for invalid range syntax', () => {
      const template: TemplateConfig = {
        templateDimensions: [1000, 800],
        bubbleDimensions: [40, 40],
        fieldBlocks: {
          MCQ_Block: {
            origin: [100, 100],
            fieldLabels: ['q1..'],
            fieldDetectionType: 'BUBBLES_THRESHOLD',
            bubbleFieldType: 'QTYPE_MCQ4',
            bubblesGap: 50,
            labelsGap: 60,
          },
        },
      };

      expect(() => TemplateLoader.loadFromJSON(template)).toThrow('Invalid range syntax');
    });

    it('should throw error when start > end', () => {
      const template: TemplateConfig = {
        templateDimensions: [1000, 800],
        bubbleDimensions: [40, 40],
        fieldBlocks: {
          MCQ_Block: {
            origin: [100, 100],
            fieldLabels: ['q10..5'],
            fieldDetectionType: 'BUBBLES_THRESHOLD',
            bubbleFieldType: 'QTYPE_MCQ4',
            bubblesGap: 50,
            labelsGap: 60,
          },
        },
      };

      expect(() => TemplateLoader.loadFromJSON(template)).toThrow('Invalid range');
    });
  });

  describe('Bubble Location Calculation', () => {
    it('should calculate horizontal bubble positions (MCQ)', () => {
      const template: TemplateConfig = {
        templateDimensions: [1000, 800],
        bubbleDimensions: [40, 40],
        fieldBlocks: {
          MCQ_Block: {
            origin: [100, 200],
            fieldLabels: ['q1'],
            fieldDetectionType: 'BUBBLES_THRESHOLD',
            bubbleFieldType: 'QTYPE_MCQ4',
            bubblesGap: 50,
            labelsGap: 60,
          },
        },
      };

      const parsed = TemplateLoader.loadFromJSON(template);
      const q1 = parsed.fields.get('q1')!;

      expect(q1.bubbles).toHaveLength(4);
      expect(q1.bubbles[0]).toMatchObject({
        x: 100,
        y: 200,
        width: 40,
        height: 40,
        label: 'A',
      });
      expect(q1.bubbles[1]).toMatchObject({
        x: 150, // 100 + 50
        y: 200,
        label: 'B',
      });
      expect(q1.bubbles[2]).toMatchObject({
        x: 200, // 100 + 2*50
        y: 200,
        label: 'C',
      });
      expect(q1.bubbles[3]).toMatchObject({
        x: 250, // 100 + 3*50
        y: 200,
        label: 'D',
      });
    });

    it('should calculate vertical bubble positions (INT)', () => {
      const template: TemplateConfig = {
        templateDimensions: [1000, 800],
        bubbleDimensions: [40, 40],
        fieldBlocks: {
          Int_Block: {
            origin: [100, 200],
            fieldLabels: ['roll1'],
            fieldDetectionType: 'BUBBLES_THRESHOLD',
            bubbleFieldType: 'QTYPE_INT',
            bubblesGap: 45,
            labelsGap: 60,
          },
        },
      };

      const parsed = TemplateLoader.loadFromJSON(template);
      const roll1 = parsed.fields.get('roll1')!;

      expect(roll1.bubbles).toHaveLength(10);
      expect(roll1.bubbles[0]).toMatchObject({
        x: 100,
        y: 200,
        label: '0',
      });
      expect(roll1.bubbles[1]).toMatchObject({
        x: 100,
        y: 245, // 200 + 45
        label: '1',
      });
      expect(roll1.bubbles[9]).toMatchObject({
        x: 100,
        y: 605, // 200 + 9*45
        label: '9',
      });
    });

    it('should apply labelsGap for multiple fields', () => {
      const template: TemplateConfig = {
        templateDimensions: [1000, 800],
        bubbleDimensions: [40, 40],
        fieldBlocks: {
          MCQ_Block: {
            origin: [100, 200],
            fieldLabels: ['q1', 'q2', 'q3'],
            fieldDetectionType: 'BUBBLES_THRESHOLD',
            bubbleFieldType: 'QTYPE_MCQ4',
            bubblesGap: 50,
            labelsGap: 60,
          },
        },
      };

      const parsed = TemplateLoader.loadFromJSON(template);

      const q1 = parsed.fields.get('q1')!;
      const q2 = parsed.fields.get('q2')!;
      const q3 = parsed.fields.get('q3')!;

      // First bubble of each field
      expect(q1.bubbles[0].x).toBe(100); // origin
      expect(q2.bubbles[0].x).toBe(160); // 100 + 60
      expect(q3.bubbles[0].x).toBe(220); // 100 + 2*60
    });

    it('should apply fieldBlocksOffset', () => {
      const template: TemplateConfig = {
        templateDimensions: [1000, 800],
        bubbleDimensions: [40, 40],
        fieldBlocksOffset: [10, 20],
        fieldBlocks: {
          MCQ_Block: {
            origin: [100, 200],
            fieldLabels: ['q1'],
            fieldDetectionType: 'BUBBLES_THRESHOLD',
            bubbleFieldType: 'QTYPE_MCQ4',
            bubblesGap: 50,
            labelsGap: 60,
          },
        },
      };

      const parsed = TemplateLoader.loadFromJSON(template);
      const q1 = parsed.fields.get('q1')!;

      expect(q1.bubbles[0].x).toBe(110); // 100 + 10
      expect(q1.bubbles[0].y).toBe(220); // 200 + 20
    });
  });

  describe('Custom Bubble Field Types', () => {
    it('should use custom bubble field type', () => {
      const template: TemplateConfig = {
        templateDimensions: [1000, 800],
        bubbleDimensions: [40, 40],
        customBubbleFieldTypes: {
          CUSTOM_YESNO: {
            bubbleValues: ['Y', 'N'],
            direction: 'horizontal',
          },
        },
        fieldBlocks: {
          Custom_Block: {
            origin: [100, 200],
            fieldLabels: ['consent'],
            fieldDetectionType: 'BUBBLES_THRESHOLD',
            bubbleFieldType: 'CUSTOM_YESNO',
            bubblesGap: 50,
            labelsGap: 60,
          },
        },
      };

      const parsed = TemplateLoader.loadFromJSON(template);
      const consent = parsed.fields.get('consent')!;

      expect(consent.bubbles).toHaveLength(2);
      expect(consent.bubbles[0].label).toBe('Y');
      expect(consent.bubbles[1].label).toBe('N');
    });

    it('should throw error for unknown bubble field type', () => {
      const template: TemplateConfig = {
        templateDimensions: [1000, 800],
        bubbleDimensions: [40, 40],
        fieldBlocks: {
          MCQ_Block: {
            origin: [100, 200],
            fieldLabels: ['q1'],
            fieldDetectionType: 'BUBBLES_THRESHOLD',
            bubbleFieldType: 'UNKNOWN_TYPE',
            bubblesGap: 50,
            labelsGap: 60,
          },
        },
      };

      expect(() => TemplateLoader.loadFromJSON(template)).toThrow('Unknown bubbleFieldType');
    });
  });

  describe('Validation', () => {
    it('should throw error if templateDimensions is missing', () => {
      const template = {
        bubbleDimensions: [40, 40],
        fieldBlocks: {},
      } as unknown as TemplateConfig;

      expect(() => TemplateLoader.loadFromJSON(template)).toThrow('templateDimensions');
    });

    it('should throw error if bubbleDimensions is missing', () => {
      const template = {
        templateDimensions: [1000, 800],
        fieldBlocks: {},
      } as unknown as TemplateConfig;

      expect(() => TemplateLoader.loadFromJSON(template)).toThrow('bubbleDimensions');
    });

    it('should throw error if fieldBlocks is empty', () => {
      const template: TemplateConfig = {
        templateDimensions: [1000, 800],
        bubbleDimensions: [40, 40],
        fieldBlocks: {},
      };

      expect(() => TemplateLoader.loadFromJSON(template)).toThrow('fieldBlocks');
    });
  });

  describe('JSON String Loading', () => {
    it('should load from JSON string', () => {
      const jsonString = JSON.stringify({
        templateDimensions: [1000, 800],
        bubbleDimensions: [40, 40],
        fieldBlocks: {
          MCQ_Block: {
            origin: [100, 200],
            fieldLabels: ['q1'],
            fieldDetectionType: 'BUBBLES_THRESHOLD',
            bubbleFieldType: 'QTYPE_MCQ4',
            bubblesGap: 50,
            labelsGap: 60,
          },
        },
      });

      const parsed = TemplateLoader.loadFromJSONString(jsonString);

      expect(parsed.fields.size).toBe(1);
      expect(parsed.fields.has('q1')).toBe(true);
    });
  });

  describe('Helper Functions', () => {
    it('should get all bubbles flattened', () => {
      const template: TemplateConfig = {
        templateDimensions: [1000, 800],
        bubbleDimensions: [40, 40],
        fieldBlocks: {
          MCQ_Block: {
            origin: [100, 200],
            fieldLabels: ['q1', 'q2'],
            fieldDetectionType: 'BUBBLES_THRESHOLD',
            bubbleFieldType: 'QTYPE_MCQ4',
            bubblesGap: 50,
            labelsGap: 60,
          },
        },
      };

      const parsed = TemplateLoader.loadFromJSON(template);
      const allBubbles = TemplateLoader.getAllBubbles(parsed);

      expect(allBubbles).toHaveLength(8); // 2 fields * 4 bubbles each
    });

    it('should get sorted field IDs', () => {
      const template: TemplateConfig = {
        templateDimensions: [1000, 800],
        bubbleDimensions: [40, 40],
        fieldBlocks: {
          MCQ_Block: {
            origin: [100, 200],
            fieldLabels: ['q10', 'q2', 'q1'],
            fieldDetectionType: 'BUBBLES_THRESHOLD',
            bubbleFieldType: 'QTYPE_MCQ4',
            bubblesGap: 50,
            labelsGap: 60,
          },
        },
      };

      const parsed = TemplateLoader.loadFromJSON(template);
      const sortedIds = TemplateLoader.getSortedFieldIds(parsed);

      expect(sortedIds).toEqual(['q1', 'q2', 'q10']); // Natural sort
    });

    it('should respect custom column order', () => {
      const template: TemplateConfig = {
        templateDimensions: [1000, 800],
        bubbleDimensions: [40, 40],
        outputColumns: {
          customOrder: ['q3', 'q1'],
        },
        fieldBlocks: {
          MCQ_Block: {
            origin: [100, 200],
            fieldLabels: ['q1', 'q2', 'q3'],
            fieldDetectionType: 'BUBBLES_THRESHOLD',
            bubbleFieldType: 'QTYPE_MCQ4',
            bubblesGap: 50,
            labelsGap: 60,
          },
        },
      };

      const parsed = TemplateLoader.loadFromJSON(template);
      const sortedIds = TemplateLoader.getSortedFieldIds(parsed);

      expect(sortedIds).toEqual(['q3', 'q1', 'q2']); // Custom order first, then remaining
    });
  });

  describe('Multiple Field Blocks', () => {
    it('should handle multiple field blocks', () => {
      const template: TemplateConfig = {
        templateDimensions: [1000, 800],
        bubbleDimensions: [40, 40],
        fieldBlocks: {
          MCQ_Block: {
            origin: [100, 200],
            fieldLabels: ['q1..5'],
            fieldDetectionType: 'BUBBLES_THRESHOLD',
            bubbleFieldType: 'QTYPE_MCQ4',
            bubblesGap: 50,
            labelsGap: 60,
          },
          Int_Block: {
            origin: [500, 200],
            fieldLabels: ['roll1..3'],
            fieldDetectionType: 'BUBBLES_THRESHOLD',
            bubbleFieldType: 'QTYPE_INT',
            bubblesGap: 45,
            labelsGap: 55,
          },
        },
      };

      const parsed = TemplateLoader.loadFromJSON(template);

      expect(parsed.fields.size).toBe(8); // 5 MCQ + 3 INT
      expect(parsed.fields.get('q1')?.bubbles).toHaveLength(4); // MCQ4
      expect(parsed.fields.get('roll1')?.bubbles).toHaveLength(10); // INT
    });
  });

  describe('Block-Specific Overrides', () => {
    it('should use block-specific bubble dimensions', () => {
      const template: TemplateConfig = {
        templateDimensions: [1000, 800],
        bubbleDimensions: [40, 40], // Global default
        fieldBlocks: {
          MCQ_Block: {
            origin: [100, 200],
            fieldLabels: ['q1'],
            fieldDetectionType: 'BUBBLES_THRESHOLD',
            bubbleFieldType: 'QTYPE_MCQ4',
            bubblesGap: 50,
            labelsGap: 60,
            bubbleDimensions: [50, 50], // Block-specific override
          },
        },
      };

      const parsed = TemplateLoader.loadFromJSON(template);
      const q1 = parsed.fields.get('q1')!;

      expect(q1.bubbles[0].width).toBe(50);
      expect(q1.bubbles[0].height).toBe(50);
    });

    it('should use block-specific empty value', () => {
      const template: TemplateConfig = {
        templateDimensions: [1000, 800],
        bubbleDimensions: [40, 40],
        emptyValue: '-', // Global default
        fieldBlocks: {
          MCQ_Block: {
            origin: [100, 200],
            fieldLabels: ['q1'],
            fieldDetectionType: 'BUBBLES_THRESHOLD',
            bubbleFieldType: 'QTYPE_MCQ4',
            bubblesGap: 50,
            labelsGap: 60,
            emptyValue: 'N/A', // Block-specific override
          },
        },
      };

      const parsed = TemplateLoader.loadFromJSON(template);
      const q1 = parsed.fields.get('q1')!;

      expect(q1.emptyValue).toBe('N/A');
    });
  });
});

