/**
 * TypeScript template schema types.
 *
 * Port of Python template models from src/schemas/models/template.py
 * Defines the structure of template.json files used for OMR sheet configuration.
 */

import type { BubbleLocation } from '../processors/detection';

/**
 * Bubble field type - defines the layout and values for a group of bubbles.
 *
 * Built-in types:
 * - QTYPE_MCQ4: Multiple choice with 4 options (A, B, C, D)
 * - QTYPE_MCQ5: Multiple choice with 5 options (A, B, C, D, E)
 * - QTYPE_INT: Integer bubbles (0-9)
 *
 * Custom types can be defined in customBubbleFieldTypes.
 */
export interface BubbleFieldType {
  /** Values for each bubble (e.g., ["A", "B", "C", "D"]) */
  bubbleValues: string[];
  /** Layout direction: "horizontal" or "vertical" */
  direction: 'horizontal' | 'vertical';
}

/**
 * Field block - defines a group of related fields (e.g., questions 1-10).
 */
export interface FieldBlock {
  /** Name of the field block */
  name?: string;
  /** Origin point [x, y] for the block */
  origin: [number, number];
  /** Labels for fields in this block (e.g., ["q1..10"] or ["q1", "q2", "q3"]) */
  fieldLabels: string[];
  /** Type of field detection (currently only "BUBBLES_THRESHOLD" supported) */
  fieldDetectionType: 'BUBBLES_THRESHOLD';
  /** Type of bubble field (e.g., "QTYPE_MCQ4", "QTYPE_INT", or custom) */
  bubbleFieldType: string;
  /** Gap between bubbles in pixels */
  bubblesGap: number;
  /** Gap between field labels in pixels */
  labelsGap: number;
  /** Optional: Override bubble dimensions [width, height] for this block */
  bubbleDimensions?: [number, number];
  /** Optional: Empty value for this block */
  emptyValue?: string;
  /** Optional: Alignment shifts computed during processing [x, y] */
  shifts?: [number, number];
  /** Optional: Bounding box origin [x, y] */
  boundingBoxOrigin?: [number, number];
  /** Optional: Bounding box dimensions [width, height] */
  boundingBoxDimensions?: [number, number];
  /** Optional: Alignment configuration for this field block */
  alignment?: {
    margins?: AlignmentMargins;
    maxDisplacement?: number;
    max_displacement?: number;
  };
  /** Optional: Fields in this block (populated during parsing) */
  fields?: any[];
}

/**
 * Alignment margins configuration.
 */
export interface AlignmentMargins {
  top?: number;
  bottom?: number;
  left?: number;
  right?: number;
}

/**
 * Alignment configuration for template matching.
 */
export interface AlignmentConfig {
  /** Path to reference image for alignment */
  referenceImage?: string;
  /** Maximum allowed displacement in pixels */
  maxDisplacement?: number;
  /** Maximum number of feature matches to use */
  maxMatchCount?: number;
  /** Margins to ignore during alignment */
  margins?: AlignmentMargins;
  /** Anchor window size [width, height] */
  anchorWindowSize?: [number, number];
}

/**
 * Pre-processor configuration (e.g., CropPage, GaussianBlur).
 */
export interface PreProcessorConfig {
  /** Name of the processor */
  name: string;
  /** Options for the processor */
  options?: Record<string, unknown>;
}

/**
 * Output columns configuration.
 */
export interface OutputColumnsConfig {
  /** Custom column order */
  customOrder?: string[];
  /** Sort type: "ALPHANUMERIC", "NUMERIC", or "CUSTOM" */
  sortType?: 'ALPHANUMERIC' | 'NUMERIC' | 'CUSTOM';
  /** Sort order: "ASC" or "DESC" */
  sortOrder?: 'ASC' | 'DESC';
}

/**
 * Sort files configuration.
 */
export interface SortFilesConfig {
  /** Whether to enable file sorting */
  enabled?: boolean;
}

/**
 * Main template configuration.
 *
 * This represents the structure of template.json files used for OMR sheet
 * layout definition and field detection.
 */
export interface TemplateConfig {
  /** Dimensions of the template image [width, height] */
  templateDimensions: [number, number];
  /** Default dimensions for bubbles [width, height] */
  bubbleDimensions: [number, number];
  /** Field blocks defining bubble locations and properties */
  fieldBlocks: Record<string, FieldBlock>;
  /** Pre-processors to apply before detection */
  preProcessors?: PreProcessorConfig[];
  /** Processing image shape [width, height] */
  processingImageShape?: [number, number];
  /** Alignment configuration */
  alignment?: AlignmentConfig;
  /** Custom labels for grouping fields */
  customLabels?: Record<string, string[]>;
  /** Custom bubble field type definitions */
  customBubbleFieldTypes?: Record<string, BubbleFieldType>;
  /** Global empty value (when no bubble is marked) */
  emptyValue?: string;
  /** Offset for field blocks [x, y] */
  fieldBlocksOffset?: [number, number];
  /** Output columns configuration */
  outputColumns?: OutputColumnsConfig;
  /** Sort files configuration */
  sortFiles?: SortFilesConfig;
  /** Conditional sets (not yet implemented) */
  conditionalSets?: unknown[];
}

/**
 * Built-in bubble field types.
 */
export const BUILTIN_BUBBLE_FIELD_TYPES: Record<string, BubbleFieldType> = {
  QTYPE_MCQ4: {
    bubbleValues: ['A', 'B', 'C', 'D'],
    direction: 'horizontal',
  },
  QTYPE_MCQ5: {
    bubbleValues: ['A', 'B', 'C', 'D', 'E'],
    direction: 'horizontal',
  },
  QTYPE_INT: {
    bubbleValues: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    direction: 'vertical',
  },
  QTYPE_MED: {
    bubbleValues: ['E', 'H'],
    direction: 'vertical',
  },
};

/**
 * Default template configuration values.
 */
export const DEFAULT_TEMPLATE_CONFIG: Partial<TemplateConfig> = {
  processingImageShape: [900, 650],
  fieldBlocksOffset: [0, 0],
  emptyValue: '',
  preProcessors: [],
  customLabels: {},
  customBubbleFieldTypes: {},
  alignment: {
    maxDisplacement: 10,
    margins: {
      top: 0,
      bottom: 0,
      left: 0,
      right: 0,
    },
  },
  outputColumns: {
    customOrder: [],
    sortType: 'ALPHANUMERIC',
    sortOrder: 'ASC',
  },
  sortFiles: {
    enabled: false,
  },
};

/**
 * Tuning configuration for processing algorithms.
 *
 * This controls various thresholds and parameters used during
 * bubble detection, alignment, and other processing steps.
 */
export interface TuningConfig {
  /** Thresholding parameters */
  thresholding?: {
    min_gap_two_bubbles?: number;
    min_jump?: number;
    min_jump_std?: number;
    global_threshold_margin?: number;
    [key: string]: any;
  };

  /** Output configuration */
  outputs?: {
    colored_outputs_enabled?: boolean;
    show_image_level?: number;
    save_detections?: boolean;
    [key: string]: any;
  };

  /** Alignment parameters */
  alignment?: {
    maxDisplacement?: number;
    margins?: AlignmentMargins;
    [key: string]: any;
  };

  /** Additional tuning parameters */
  [key: string]: any;
}

/**
 * Parsed field with expanded bubble locations.
 */
export interface ParsedField {
  /** Field ID (e.g., "q1", "q2") */
  fieldId: string;
  /** Block this field belongs to */
  blockName: string;
  /** Bubble locations for this field */
  bubbles: BubbleLocation[];
  /** Bubble field type */
  bubbleFieldType: BubbleFieldType;
  /** Empty value for this field */
  emptyValue: string;
}

/**
 * Fully parsed template with all bubble locations calculated.
 */
export interface ParsedTemplate {
  /** Original configuration */
  config: TemplateConfig;
  /** Template dimensions [width, height] */
  templateDimensions: [number, number];
  /** Bubble dimensions [width, height] */
  bubbleDimensions: [number, number];
  /** Field blocks (may have runtime data like shifts) */
  fieldBlocks: FieldBlock[] | Record<string, FieldBlock>;
  field_blocks?: FieldBlock[] | Record<string, FieldBlock>; // Python compatibility
  /** Map of field ID to parsed field */
  fields: Map<string, ParsedField>;
  /** Map of field ID to bubble locations (for BubblesFieldDetection) */
  fieldBubbles: Map<string, BubbleLocation[]>;
  /** Tuning configuration */
  tuningConfig?: TuningConfig;
  tuning_config?: TuningConfig; // Python compatibility
  /** Alignment configuration */
  alignment?: {
    grayAlignmentImage?: any;
    gray_alignment_image?: any;
    coloredAlignmentImage?: any;
    colored_alignment_image?: any;
    margins?: AlignmentMargins;
    maxDisplacement?: number;
    max_displacement?: number;
  };
}

