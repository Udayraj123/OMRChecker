/**
 * Generic serialization utilities for objects.
 *
 * TypeScript port of src/utils/serialization.py
 * Maintains 1:1 correspondence with Python implementation.
 *
 * This module provides generic serialization functions that handle nested
 * structures, Path objects (as strings), Enums, and collections.
 */

/**
 * Recursively convert an object to a dictionary.
 *
 * Handles nested objects, Path objects (as strings), Enums, lists, dicts, and other types.
 * This provides a generic serialization solution that automatically adapts to
 * object structure changes without requiring manual field mapping.
 *
 * Port of dataclass_to_dict() from Python.
 *
 * @param obj - The object to serialize (typically a class instance or plain object)
 * @returns Dictionary representation suitable for JSON serialization
 *
 * @example
 * ```typescript
 * class Config {
 *   path: string;
 *   value: number = 10;
 * }
 * const config = new Config();
 * config.path = "/tmp/test";
 * config.value = 42;
 * dataclassToDict(config);
 * // Returns: { path: '/tmp/test', value: 42 }
 * ```
 *
 * Supported Types:
 * - Primitive types: string, number, boolean, null, undefined
 * - Objects: Automatically serialized recursively
 * - Path objects (strings): Passed through as strings
 * - Enums: Converted to their values
 * - Collections: Arrays and objects are handled recursively
 * - Nested structures: Any combination of the above
 */
export function dataclassToDict(obj: any): any {
  // Handle null/undefined
  if (obj === null || obj === undefined) {
    return obj;
  }

  // Handle primitive types
  if (
    typeof obj === 'string' ||
    typeof obj === 'number' ||
    typeof obj === 'boolean'
  ) {
    return obj;
  }

  // Handle arrays
  if (Array.isArray(obj)) {
    return obj.map((item) => dataclassToDict(item));
  }

  // Handle Date objects
  if (obj instanceof Date) {
    return obj.toISOString();
  }

  // Handle objects with toDict method (custom serialization)
  if (typeof obj === 'object' && typeof obj.toDict === 'function') {
    return obj.toDict();
  }

  // Handle objects with toJSON method
  if (typeof obj === 'object' && typeof obj.toJSON === 'function') {
    return obj.toJSON();
  }

  // Handle plain objects
  if (typeof obj === 'object') {
    const result: Record<string, any> = {};

    // Handle Map objects
    if (obj instanceof Map) {
      const mapResult: Record<string, any> = {};
      for (const [key, value] of obj.entries()) {
        mapResult[String(key)] = dataclassToDict(value);
      }
      return mapResult;
    }

    // Handle Set objects
    if (obj instanceof Set) {
      return Array.from(obj).map((item) => dataclassToDict(item));
    }

    // Handle regular objects
    for (const key in obj) {
      if (Object.prototype.hasOwnProperty.call(obj, key)) {
        // Skip functions
        if (typeof obj[key] === 'function') {
          continue;
        }
        result[key] = dataclassToDict(obj[key]);
      }
    }

    return result;
  }

  // For other types, try to convert to string
  // This handles custom objects, etc.
  try {
    return String(obj);
  } catch {
    // If all else fails, return the object as-is
    return obj;
  }
}

