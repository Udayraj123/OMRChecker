/**
 * Constants and type definitions for OMRChecker
 *
 * TypeScript port of src/utils/constants.py
 * Uses factory functions to ensure DRY principles
 */

// Export types for reuse in other modules
// Note: OpenCV.js expects BGRA format (4 elements) for Scalar values
export type ColorTuple = [number, number, number, number];
export type BubbleFieldType = {
  readonly bubbleValues: readonly string[];
  readonly direction: 'horizontal' | 'vertical';
};

// DRY: Color constant factory to avoid repetition
// OpenCV uses BGR format, so we keep the same order as Python
// and add alpha channel (255 = fully opaque)
function createColor(b: number, g: number, r: number, a: number = 255): ColorTuple {
  return [b, g, r, a] as const;
}

// Filenames
export const TEMPLATE_FILENAME = 'template.json';
export const EVALUATION_FILENAME = 'evaluation.json';
export const CONFIG_FILENAME = 'config.json';

// Supported processor names
export const SUPPORTED_PROCESSOR_NAMES = [
  'AutoRotate',
  'Contrast',
  'CropOnMarkers',
  'CropPage',
  'FeatureBasedAlignment',
  'GaussianBlur',
  'Levels',
  'MedianBlur',
] as const;

// Field label number regex
export const FIELD_LABEL_NUMBER_REGEX = /([^\d]+)(\d*)/;

// Error codes
export const ERROR_CODES = {
  MULTI_BUBBLE_WARN: 1,
  NO_MARKER_ERR: 2,
} as const;

// Color constants using factory (BGR format + Alpha, matching Python's BGR tuples)
export const CLR_BLACK = createColor(0, 0, 0);
export const CLR_WHITE = createColor(255, 255, 255);
export const CLR_GRAY = createColor(130, 130, 130);
export const CLR_DARK_GRAY = createColor(100, 100, 100);
export const CLR_LIGHT_GRAY = createColor(200, 200, 200);
export const CLR_GREEN = createColor(100, 200, 100);
export const CLR_RED = createColor(20, 20, 255);
export const CLR_BLUE = createColor(255, 20, 20);
export const CLR_YELLOW = createColor(20, 255, 255);
export const CLR_DARK_GREEN = createColor(20, 255, 20);
export const CLR_DARK_RED = createColor(20, 20, 255);
export const CLR_DARK_BLUE = createColor(255, 20, 20);
export const CLR_NEAR_BLACK = createColor(20, 20, 10);

// Template transparency
export const MARKED_TEMPLATE_TRANSPARENCY = 0.65;

// Text size
export const TEXT_SIZE = 0.95;

// Paper thresholds
export const PAPER_VALUE_THRESHOLD = 180;
export const PAPER_SATURATION_THRESHOLD = 40;

// HSV white thresholds (as tuples for NumPy/OpenCV compatibility)
export const HSV_WHITE_LOW: readonly [number, number, number] = [
  0,
  0,
  PAPER_VALUE_THRESHOLD,
] as const;
export const HSV_WHITE_HIGH: readonly [number, number, number] = [
  180,
  PAPER_SATURATION_THRESHOLD,
  255,
] as const;

// Wait keys
export const WAIT_KEYS = {
  ENTER: 13,
  ESCAPE: 27,
  SPACE: 32,
} as const;

// Zero margins constant
export const ZERO_MARGINS = {
  top: 0,
  bottom: 0,
  left: 0,
  right: 0,
} as const;

// Output modes
export const OUTPUT_MODES = {
  SET_LAYOUT: 'setLayout',
  DEFAULT: 'default',
  MODERATION: 'moderation',
} as const;

// DRY: Field type builder to avoid structure repetition
function createFieldType(
  values: string[],
  direction: 'horizontal' | 'vertical'
): BubbleFieldType {
  return { bubbleValues: values, direction } as const;
}

// Built-in bubble field types using factory
export const BUILTIN_BUBBLE_FIELD_TYPES = {
  QTYPE_INT: createFieldType(
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    'vertical'
  ),
  QTYPE_INT_FROM_1: createFieldType(
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
    'vertical'
  ),
  QTYPE_MCQ4: createFieldType(['A', 'B', 'C', 'D'], 'horizontal'),
  QTYPE_MCQ5: createFieldType(['A', 'B', 'C', 'D', 'E'], 'horizontal'),
} as const;

// Custom bubble field type pattern
export const CUSTOM_BUBBLE_FIELD_TYPE_PATTERN = '^CUSTOM_.*$';

