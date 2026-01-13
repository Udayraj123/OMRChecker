/**
 * Common schema definitions and validation utilities
 *
 * Shared across configSchema and templateSchema to ensure DRY principles
 */

import Ajv, { type ErrorObject } from 'ajv';
import addFormats from 'ajv-formats';

// Common type definitions used in both schemas
export type TwoIntegers = [number, number];
export type TwoPositiveNumbers = [number, number];
export type Margins = {
  top: number;
  right: number;
  bottom: number;
  left: number;
};

// Validation result type (reused by all validators)
export interface ValidationResult {
  valid: boolean;
  errors: Array<{
    instancePath: string;
    message: string;
    params?: Record<string, unknown>;
  }>;
}

// Reusable schema fragments (matches Python's load_common_defs)
export const commonSchemaDefinitions = {
  positive_integer: {
    type: 'integer' as const,
    minimum: 1,
  },

  positive_number: {
    type: 'number' as const,
    exclusiveMinimum: 0, // AJV uses number, not boolean
  },

  two_integers: {
    type: 'array' as const,
    items: { type: 'integer' as const },
    minItems: 2,
    maxItems: 2,
  },

  two_positive_integers: {
    type: 'array' as const,
    items: { type: 'integer' as const, minimum: 1 },
    minItems: 2,
    maxItems: 2,
  },

  two_positive_numbers: {
    type: 'array' as const,
    items: { type: 'number' as const, minimum: 0 },
    minItems: 2,
    maxItems: 2,
  },

  two_odd_integers: {
    type: 'array' as const,
    items: {
      type: 'integer' as const,
      minimum: 1,
      // Note: AJV doesn't have built-in odd validation, but this ensures positive
    },
    minItems: 2,
    maxItems: 2,
  },

  margins_schema: {
    type: 'object' as const,
    required: ['top', 'right', 'bottom', 'left'],
    additionalProperties: false,
    properties: {
      top: { type: 'integer' as const, minimum: 0 },
      right: { type: 'integer' as const, minimum: 0 },
      bottom: { type: 'integer' as const, minimum: 0 },
      left: { type: 'integer' as const, minimum: 0 },
    },
  },

  field_string_type: {
    type: 'string' as const,
    pattern: '^[a-zA-Z0-9_-]+$',
  },

  array_of_strings: {
    type: 'array' as const,
    items: { type: 'string' as const },
  },

  zero_to_one_number: {
    type: 'number' as const,
    minimum: 0,
    maximum: 1,
  },

  matplotlib_color: {
    oneOf: [
      {
        type: 'string' as const,
        description: 'This should match with #rgb, #rgba, #rrggbb, and #rrggbbaa syntax',
        pattern: '^#(?:(?:[\\da-fA-F]{3}){1,2}|(?:[\\da-fA-F]{4}){1,2})$',
      },
      {
        type: 'string' as const,
        description: 'Named colors supported by matplotlib (simplified set)',
        enum: [
          'red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'black', 'white',
          'gray', 'orange', 'purple', 'pink', 'brown', 'lime', 'navy', 'teal'
        ],
      },
    ],
  },
};

/**
 * Shared AJV instance configuration
 * Configured for Draft 2020-12 support with lenient mode (matches Python jsonschema)
 */
export function createSchemaValidator(): Ajv {
  const ajv = new Ajv({
    allErrors: true,
    strict: false, // Python jsonschema is more lenient
    validateFormats: true,
    // Don't validate the $schema itself, just use the schema rules
    validateSchema: false,
  });
  addFormats(ajv);
  return ajv;
}

/**
 * Helper to format AJV errors into consistent ValidationResult
 */
export function formatValidationErrors(
  errors: ErrorObject[] | null | undefined
): ValidationResult['errors'] {
  if (!errors) return [];

  return errors.map((err) => ({
    instancePath: err.instancePath || '',
    message: err.message || 'Validation failed',
    params: err.params || {},
  }));
}

/**
 * Load common definitions by keys
 * Matches Python's load_common_defs function
 *
 * @param keys - Array of definition keys to load
 * @returns Object with the requested definitions
 */
export function loadCommonDefs(keys: string[]): Record<string, any> {
  const result: Record<string, any> = {};
  for (const key of keys) {
    if (key in commonSchemaDefinitions) {
      result[key] = commonSchemaDefinitions[key as keyof typeof commonSchemaDefinitions];
    } else {
      console.warn(`Common definition not found: ${key}`);
    }
  }
  return result;
}

