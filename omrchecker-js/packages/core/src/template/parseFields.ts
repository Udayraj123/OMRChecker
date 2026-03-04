/**
 * Field string parsing utilities for OMRChecker TypeScript port.
 *
 * Migrated from: src/utils/parsing.py (parse_field_string, parse_fields)
 */

import { OMRCheckerError } from '../utils/exceptions';

/**
 * Regex matching range syntax like "roll1..9" or "q10..13"
 * Groups: (prefix)(start)..(end)
 * Equivalent to Python's FIELD_STRING_REGEX_GROUPS = r"([^\.\d]+)(\d+)\.{2,3}(\d+)"
 */
const FIELD_STRING_REGEX = /^([^.\d]+)(\d+)\.{2,3}(\d+)$/;

/**
 * Parse a field string that may contain range syntax.
 *
 * Examples:
 *   "roll1..9"  → ["roll1", "roll2", ..., "roll9"]
 *   "q10..13"   → ["q10", "q11", "q12", "q13"]
 *   "Medium"    → ["Medium"]
 *
 * Ported from Python: src/utils/parsing.py::parse_field_string
 */
export function parseFieldString(fieldString: string): string[] {
  if (!fieldString.includes('.')) {
    return [fieldString];
  }

  const match = FIELD_STRING_REGEX.exec(fieldString);
  if (!match) {
    throw new OMRCheckerError(
      `Invalid field string format: '${fieldString}'. Expected format like 'q1..5'`,
      { field_string: fieldString },
    );
  }

  const [, fieldPrefix, startStr, endStr] = match;
  const start = parseInt(startStr, 10);
  const end = parseInt(endStr, 10);

  if (start >= end) {
    throw new OMRCheckerError(
      `Invalid range in fields string: '${fieldString}', start: ${start} is not less than end: ${end}`,
      { field_string: fieldString, start, end },
    );
  }

  const result: string[] = [];
  for (let i = start; i <= end; i++) {
    result.push(`${fieldPrefix}${i}`);
  }
  return result;
}

/**
 * Parse and validate a list of field strings (range or literal), checking for overlaps.
 *
 * Ported from Python: src/utils/parsing.py::parse_fields
 */
export function parseFields(key: string, fields: string[]): string[] {
  const parsedFields: string[] = [];
  const fieldsSet = new Set<string>();

  for (const fieldString of fields) {
    const fieldsArray = parseFieldString(fieldString);
    const currentSet = new Set(fieldsArray);

    // Check for overlap with already-parsed fields
    const overlap = [...currentSet].filter((f) => fieldsSet.has(f));
    if (overlap.length > 0) {
      throw new OMRCheckerError(
        `Given field string '${fieldString}' has overlapping field(s) with other fields in '${key}': ${fields}`,
        {
          field_string: fieldString,
          key,
          overlapping_fields: overlap,
        },
      );
    }

    for (const f of currentSet) {
      fieldsSet.add(f);
    }
    parsedFields.push(...fieldsArray);
  }

  return parsedFields;
}
