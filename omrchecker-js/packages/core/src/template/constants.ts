/**
 * Template constants for OMRChecker TypeScript port.
 *
 * Migrated from: src/utils/constants.py (BUILTIN_BUBBLE_FIELD_TYPES, ZERO_MARGINS)
 */

export const ZERO_MARGINS = { top: 0, bottom: 0, left: 0, right: 0 } as const;

export interface BubbleFieldTypeData {
  bubble_values: string[];
  direction: 'vertical' | 'horizontal';
}

export const BUILTIN_BUBBLE_FIELD_TYPES: Record<string, BubbleFieldTypeData> = {
  QTYPE_INT: {
    bubble_values: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    direction: 'vertical',
  },
  QTYPE_INT_FROM_1: {
    bubble_values: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
    direction: 'vertical',
  },
  QTYPE_MCQ4: {
    bubble_values: ['A', 'B', 'C', 'D'],
    direction: 'horizontal',
  },
  QTYPE_MCQ5: {
    bubble_values: ['A', 'B', 'C', 'D', 'E'],
    direction: 'horizontal',
  },
};
