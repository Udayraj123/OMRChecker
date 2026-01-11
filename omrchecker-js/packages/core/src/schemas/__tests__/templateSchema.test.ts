/**
 * Tests for template schema validation
 */

import { describe, it, expect } from 'vitest';
import { validateTemplate } from '../templateSchema';

describe('TemplateSchema', () => {
  it('should validate minimal valid template', () => {
    const template = {
      bubbleFieldDimensions: [30, 30],
      preProcessors: [],
      fieldBlocks: {},
    };

    const result = validateTemplate(template);
    expect(result.valid).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  it('should reject missing required fields', () => {
    const template = {
      bubbleFieldDimensions: [30, 30],
      // Missing preProcessors and fieldBlocks
    };

    const result = validateTemplate(template);
    expect(result.valid).toBe(false);
  });

  it('should validate preprocessors with options', () => {
    const template = {
      bubbleFieldDimensions: [30, 30],
      preProcessors: [
        {
          name: 'GaussianBlur',
          options: {
            kSize: [5, 5],
            sigmaX: 0,
          },
        },
        {
          name: 'MedianBlur',
          options: {
            kSize: 5,
          },
        },
      ],
      fieldBlocks: {},
    };

    const result = validateTemplate(template);
    expect(result.valid).toBe(true);
  });

  it('should reject invalid preprocessor names', () => {
    const template = {
      bubbleFieldDimensions: [30, 30],
      preProcessors: [
        {
          name: 'InvalidProcessor',
          options: {},
        },
      ],
      fieldBlocks: {},
    };

    const result = validateTemplate(template);
    expect(result.valid).toBe(false);
  });

  it('should validate field blocks', () => {
    const template = {
      bubbleFieldDimensions: [30, 30],
      preProcessors: [],
      fieldBlocks: {
        q1_10: {
          fieldType: 'QTYPE_MCQ4',
          fieldDetectionType: 'BUBBLES_THRESHOLD',
          origin: [100, 200],
          fieldLabels: ['q1', 'q2', 'q3'],
          emptyValue: '0',
        },
      },
    };

    const result = validateTemplate(template);
    expect(result.valid).toBe(true);
  });

  it('should validate builtin bubble field types', () => {
    const template = {
      bubbleFieldDimensions: [30, 30],
      preProcessors: [],
      fieldBlocks: {
        q1_10: {
          bubbleFieldType: 'QTYPE_MCQ4',
          fieldDetectionType: 'BUBBLES_THRESHOLD',
          origin: [100, 200],
          fieldLabels: ['q1', 'q2', 'q3'],
          emptyValue: '0',
        },
      },
    };

    const result = validateTemplate(template);
    expect(result.valid).toBe(true);
  });

  it('should validate marker zones with zone description', () => {
    const template = {
      bubbleFieldDimensions: [30, 30],
      preProcessors: [],
      fieldBlocks: {},
      markerZones: [
        {
          origin: [10, 10],
          dimensions: [50, 50],
          margins: { top: 5, right: 5, bottom: 5, left: 5 },
          scannerType: 'TEMPLATE_MATCH',
        },
      ],
    };

    const result = validateTemplate(template);
    expect(result.valid).toBe(true);
  });

  it('should validate page dimensions', () => {
    const template = {
      pageDimensions: [2480, 3508],
      bubbleFieldDimensions: [30, 30],
      preProcessors: [],
      fieldBlocks: {},
    };

    const result = validateTemplate(template);
    expect(result.valid).toBe(true);
  });
});

