/**
 * Parsing utilities for OMRChecker
 *
 * TypeScript port of src/utils/parsing.py
 * Handles field label expansion, JSON serialization, and other parsing tasks
 */

import { OMRCheckerError } from '../core/exceptions';
import { FIELD_LABEL_NUMBER_REGEX } from './constants';

/**
 * Default JSON serialization helper.
 * Converts objects to JSON-serializable format.
 *
 * Port of Python's default_dump function.
 */
export function defaultDump(obj: unknown): unknown {
  if (obj === null || obj === undefined) {
    return null;
  }

  if (typeof obj === 'boolean' || typeof obj === 'string' || typeof obj === 'number') {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map(defaultDump);
  }

  if (typeof obj === 'object') {
    const result: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(obj)) {
      // Skip functions and undefined values
      if (typeof value !== 'function' && value !== undefined) {
        result[key] = defaultDump(value);
      }
    }
    return result;
  }

  // Fallback: convert to string
  return String(obj);
}

/**
 * Parse a single field string (e.g., "q1..10" → ["q1", "q2", ..., "q10"]).
 *
 * Port of Python's parse_field_string function.
 */
export function parseFieldString(fieldString: string): string[] {
  if (fieldString.includes('..')) {
    // Match pattern like "q1..10" or "field1..5" (2 or 3 dots)
    // Pattern: ([^.\d]+)(\d+)\.{2,3}(\d+)
    const match = fieldString.match(/^([^.\d]+)(\d+)\.{2,3}(\d+)$/);
    if (!match) {
      throw new OMRCheckerError(`Invalid field range format: ${fieldString}`, {
        fieldString,
      });
    }

    const [, prefix, startStr, endStr] = match;
    const start = parseInt(startStr, 10);
    const end = parseInt(endStr, 10);

    if (start >= end) {
      throw new OMRCheckerError(
        `Invalid range in field string: '${fieldString}', start: ${start} is not less than end: ${end}`,
        {
          fieldString,
          start,
          end,
        }
      );
    }

    const result: string[] = [];
    for (let i = start; i <= end; i++) {
      result.push(`${prefix}${i}`);
    }
    return result;
  }

  return [fieldString];
}

/**
 * Parse field labels with range expansion and duplicate detection.
 *
 * Port of Python's parse_fields function.
 * Expands ranges like "q1..10" and ensures no duplicates.
 */
export function parseFields(key: string, fields: string[]): string[] {
  const parsedFields: string[] = [];
  const fieldsSet = new Set<string>();

  for (const fieldString of fields) {
    const fieldsArray = parseFieldString(fieldString);
    const currentSet = new Set(fieldsArray);

    // Check for overlaps
    const overlap = Array.from(currentSet).filter((f) => fieldsSet.has(f));
    if (overlap.length > 0) {
      throw new OMRCheckerError(
        `Given field string '${fieldString}' has overlapping field(s) with other fields in '${key}': ${fields.join(', ')}`,
        {
          fieldString,
          key,
          overlappingFields: overlap,
        }
      );
    }

    // Add to sets
    fieldsArray.forEach((f) => fieldsSet.add(f));
    parsedFields.push(...fieldsArray);
  }

  return parsedFields;
}

/**
 * Alphanumerical sort key for field labels.
 * Extracts prefix and number for proper sorting.
 *
 * Port of Python's alphanumerical_sort_key function.
 */
export function alphanumericalSortKey(fieldLabel: string): [string, number, number] {
  const match = fieldLabel.match(FIELD_LABEL_NUMBER_REGEX);
  if (!match) {
    return [fieldLabel, 0, 0];
  }

  const [, labelPrefix, labelSuffix] = match;
  const number = labelSuffix.length > 0 ? parseInt(labelSuffix, 10) : 0;
  return [labelPrefix, number, 0];
}

