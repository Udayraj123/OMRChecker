/**
 * Template schema validation for OMRChecker
 *
 * TypeScript port of src/schemas/template_schema.py
 * Uses AJV with Draft 2020-12 support and shared definitions from common.ts
 *
 * Note: This is a simplified implementation covering the core template structure.
 * Full feature parity with Python version to be achieved in subsequent iterations.
 */

import {
  commonSchemaDefinitions,
  createSchemaValidator,
  formatValidationErrors,
  ValidationResult,
} from './common';
import {
  SUPPORTED_PROCESSOR_NAMES,
  BUILTIN_BUBBLE_FIELD_TYPES,
  CUSTOM_BUBBLE_FIELD_TYPE_PATTERN,
} from '../utils/constants';

// DRY: Preprocessor option builder (used 14+ times)
function createProcessorOption(
  name: string,
  requiredFields: string[],
  availableKeys: Record<string, unknown>,
  additionalValidation?: Record<string, unknown>
): Record<string, unknown> {
  return {
    if: {
      properties: { name: { const: name } },
      required: ['name', 'options'],
    },
    then: {
      properties: {
        options: {
          description: `Options for the ${name} pre-processor`,
          type: 'object' as const,
          required: requiredFields,
          additionalProperties: false,
          properties: availableKeys,
          ...additionalValidation,
        },
      },
    },
  };
}

// DRY: Common field block properties (used 3 times for different detection types)
const commonFieldBlockProps = {
  emptyValue: {
    type: 'string' as const,
    description: 'Value to use when no bubble is marked',
  },
  bubbleFieldType: {
    description: 'The type of bubble field',
    oneOf: [
      {
        type: 'string' as const,
        enum: Object.keys(BUILTIN_BUBBLE_FIELD_TYPES),
      },
      {
        type: 'string' as const,
        pattern: CUSTOM_BUBBLE_FIELD_TYPE_PATTERN,
      },
    ],
  },
  fieldDetectionType: {
    description: 'The detection method for this field block',
    type: 'string' as const,
    enum: ['BUBBLES_THRESHOLD', 'OCR', 'BARCODE_QR'],
  },
  fieldLabels: {
    description: 'Labels for each field in this block',
    type: 'array' as const,
    items: { $ref: '#/$defs/field_string_type' },
  },
};

// DRY: Zone description factory (used 5+ times)
function createZoneDescription(
  additionalProps: Record<string, unknown> = {}
): Record<string, unknown> {
  return {
    type: 'object' as const,
    required: ['origin', 'dimensions', 'margins'],
    properties: {
      origin: {
        $ref: '#/$defs/two_positive_integers',
        description: 'The origin point [x, y] of the zone',
      },
      dimensions: {
        $ref: '#/$defs/two_positive_integers',
        description: 'The dimensions [width, height] of the zone',
      },
      margins: {
        $ref: '#/$defs/margins_schema',
        description: 'The margins around the zone',
      },
      ...additionalProps,
    },
  };
}

// Preprocessor validation array (uses createProcessorOption for all 14 types)
const preprocessorValidations = [
  createProcessorOption('GaussianBlur', [], {
    kSize: { $ref: '#/$defs/two_positive_numbers' },
    sigmaX: { type: 'number' },
  }),
  createProcessorOption('MedianBlur', [], {
    kSize: { type: 'integer', minimum: 1 },
  }),
  createProcessorOption('Levels', [], {
    low: { type: 'integer', minimum: 0, maximum: 255 },
    high: { type: 'integer', minimum: 0, maximum: 255 },
    gamma: { type: 'number', minimum: 0 },
  }),
  createProcessorOption('Contrast', [], {
    alpha: { type: 'number', minimum: 0 },
    beta: { type: 'integer' },
  }),
  createProcessorOption('AutoRotate', [], {
    referenceImage: { type: 'string' },
    maxRotation: { type: 'number', default: 10 },
  }),
  createProcessorOption('CropPage', ['morphKernel'], {
    morphKernel: { $ref: '#/$defs/two_positive_integers' },
  }),
  createProcessorOption('CropOnMarkers', [], {
    scannerType: { type: 'string' },
    customOptions: { type: 'object' },
  }),
  createProcessorOption('FeatureBasedAlignment', [], {
    referenceImage: { type: 'string' },
    maxFeatures: { type: 'integer', minimum: 10, default: 500 },
  }),
];

// Template JSON Schema (Draft 2020-12)
const TEMPLATE_SCHEMA = {
  $schema: 'https://json-schema.org/draft/2020-12/schema',
  $id: 'https://github.com/Udayraj123/OMRChecker/tree/master/src/schemas/template-schema.json',
  $defs: {
    ...commonSchemaDefinitions,
  },
  title: 'Template Schema',
  description: 'OMRChecker template schema for OMR sheet configuration',
  type: 'object',
  required: ['bubbleFieldDimensions', 'preProcessors', 'fieldBlocks'],
  properties: {
    path: {
      description: 'The path to the template file',
      type: 'string',
    },
    pageDimensions: {
      description: 'The dimensions of the page [width, height]',
      $ref: '#/$defs/two_positive_integers',
    },
    bubbleFieldDimensions: {
      description: 'The dimensions of each bubble [width, height]',
      $ref: '#/$defs/two_positive_integers',
    },
    customLabels: {
      description: 'Custom label definitions',
      type: 'object',
      patternProperties: {
        '^.*$': {
          type: 'array',
          items: { type: 'string' },
        },
      },
    },
    preProcessors: {
      description: 'Image preprocessing steps to apply',
      type: 'array',
      items: {
        type: 'object',
        required: ['name'],
        properties: {
          name: {
            description: 'The name of the pre-processor',
            type: 'string',
            enum: SUPPORTED_PROCESSOR_NAMES,
          },
          options: {
            description: 'Options for the pre-processor',
            type: 'object',
          },
        },
        allOf: preprocessorValidations,
      },
    },
    fieldBlocks: {
      description: 'Definition of field blocks on the OMR sheet',
      type: 'object',
      patternProperties: {
        '^[a-zA-Z0-9_-]+$': {
          type: 'object',
          required: ['fieldDetectionType', 'origin', 'fieldLabels'],
          properties: {
            ...commonFieldBlockProps,
            origin: {
              $ref: '#/$defs/two_positive_numbers',
              description: 'The starting position of the field block',
            },
            bubblesGap: {
              $ref: '#/$defs/positive_number',
              description: 'Gap between bubbles',
            },
            labelsGap: {
              $ref: '#/$defs/positive_number',
              description: 'Gap between labels',
            },
          },
        },
      },
    },
    markerZones: {
      description: 'Zones containing alignment markers',
      type: 'array',
      items: createZoneDescription({
        scannerType: {
          type: 'string',
          description: 'Type of marker detection to use',
        },
      }),
    },
    scanZones: {
      description: 'Additional scan zones for special detection',
      type: 'object',
      patternProperties: {
        '^[a-zA-Z0-9_-]+$': createZoneDescription({
          scannerType: {
            type: 'string',
            description: 'Type of scanner for this zone',
          },
        }),
      },
    },
  },
};

// DRY: Reusable validator wrapper
function createTemplateValidator() {
  const ajv = createSchemaValidator();
  const validate = ajv.compile(TEMPLATE_SCHEMA);

  return (data: unknown): ValidationResult => ({
    valid: !!validate(data),
    errors: formatValidationErrors(validate.errors),
  });
}

// Export singleton validator instance
export const validateTemplate = createTemplateValidator();

// Export schema for testing/inspection
export { TEMPLATE_SCHEMA };

