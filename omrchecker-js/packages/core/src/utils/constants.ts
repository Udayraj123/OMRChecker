// Migrated from Python: src/utils/constants.py

// Filenames
export const TEMPLATE_FILENAME = "template.json";
export const EVALUATION_FILENAME = "evaluation.json";
export const CONFIG_FILENAME = "config.json";

// Supported processor names
export const SUPPORTED_PROCESSOR_NAMES = [
  "AutoRotate",
  "Contrast",
  "CropOnMarkers",
  "CropPage",
  "FeatureBasedAlignment",
  "GaussianBlur",
  "Levels",
  "MedianBlur",
  // "WarpOnPoints",
] as const;

export type SupportedProcessorName = typeof SUPPORTED_PROCESSOR_NAMES[number];

// Regex pattern for field label number parsing
export const FIELD_LABEL_NUMBER_REGEX = /([^\d]+)(\d*)/;

// Error codes
export const ERROR_CODES = {
  MULTI_BUBBLE_WARN: 1,
  NO_MARKER_ERR: 2,
} as const;

export type ErrorCode = typeof ERROR_CODES[keyof typeof ERROR_CODES];

// Bubble field type definition
export interface BubbleFieldType {
  bubble_values: string[];
  direction: "vertical" | "horizontal";
}

export const BUILTIN_BUBBLE_FIELD_TYPES: Record<string, BubbleFieldType> = {
  QTYPE_INT: {
    bubble_values: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    direction: "vertical",
  },
  QTYPE_INT_FROM_1: {
    bubble_values: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
    direction: "vertical",
  },
  QTYPE_MCQ4: {
    bubble_values: ["A", "B", "C", "D"],
    direction: "horizontal",
  },
  QTYPE_MCQ5: {
    bubble_values: ["A", "B", "C", "D", "E"],
    direction: "horizontal",
  },
  // Note: custom field types can be defined in template.json under "customBubbleFieldTypes".
};

export const CUSTOM_BUBBLE_FIELD_TYPE_PATTERN = /^CUSTOM_.*$/;

// Text rendering
export const TEXT_SIZE = 0.95;

// Colors (BGR tuples matching OpenCV convention)
export const CLR_BLACK: [number, number, number] = [0, 0, 0];
export const CLR_DARK_GRAY: [number, number, number] = [100, 100, 100];
export const CLR_DARK_BLUE: [number, number, number] = [255, 20, 20];
export const CLR_DARK_GREEN: [number, number, number] = [20, 255, 20];
export const CLR_DARK_RED: [number, number, number] = [20, 20, 255];
export const CLR_NEAR_BLACK: [number, number, number] = [20, 20, 10];
export const CLR_GRAY: [number, number, number] = [130, 130, 130];
export const CLR_LIGHT_GRAY: [number, number, number] = [200, 200, 200];
export const CLR_GREEN: [number, number, number] = [100, 200, 100];
export const CLR_WHITE: [number, number, number] = [255, 255, 255];

// Template rendering
export const MARKED_TEMPLATE_TRANSPARENCY = 0.65;

// Paper detection thresholds
// Note: hsv_white_low and hsv_white_high (numpy arrays) are omitted — browser-incompatible.
export const PAPER_VALUE_THRESHOLD = 180;
export const PAPER_SATURATION_THRESHOLD = 40;

// Layout / margin helpers
export const ZERO_MARGINS = { top: 0, bottom: 0, left: 0, right: 0 } as const;

// Output modes
export const OUTPUT_MODES = {
  SET_LAYOUT: "setLayout",
  DEFAULT: "default",
  MODERATION: "moderation",
} as const;

export type OutputMode = typeof OUTPUT_MODES[keyof typeof OUTPUT_MODES];

// Omitted (browser-incompatible):
//   hsv_white_low, hsv_white_high  — require numpy arrays (OpenCV-specific)
//   MATPLOTLIB_COLORS              — requires matplotlib, no browser equivalent
//   WAIT_KEYS                      — no interactive keyboard input in browser
