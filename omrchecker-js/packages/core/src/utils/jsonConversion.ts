/**
 * JSON key conversion utilities for camelCase ↔ snake_case conversion.
 *
 * This module provides utilities to convert between camelCase (used in JSON)
 * and snake_case (used in Python code), enabling a clean separation between
 * external API conventions and internal Python conventions.
 */

/**
 * Convert camelCase to snake_case.
 *
 * @param name - String in camelCase format
 * @returns String in snake_case format
 *
 * @example
 * ```typescript
 * camelToSnake("showImageLevel") // 'show_image_level'
 * camelToSnake("MLConfig") // 'ml_config'
 * camelToSnake("globalPageThreshold") // 'global_page_threshold'
 * ```
 */
export function camelToSnake(name: string): string {
  // Handle acronyms at the start (e.g., "MLConfig" -> "ml_config")
  name = name.replace(/([A-Z]+)([A-Z][a-z])/g, "$1_$2");
  // Insert underscore before uppercase letters (e.g., "camelCase" -> "camel_Case")
  name = name.replace(/([a-z\d])([A-Z])/g, "$1_$2");
  return name.toLowerCase();
}

/**
 * Convert snake_case to camelCase.
 *
 * @param name - String in snake_case format
 * @returns String in camelCase format
 *
 * @example
 * ```typescript
 * snakeToCamel("show_image_level") // 'showImageLevel'
 * snakeToCamel("ml_config") // 'mlConfig'
 * snakeToCamel("global_page_threshold") // 'globalPageThreshold'
 * ```
 */
export function snakeToCamel(name: string): string {
  const components = name.split("_");
  // Keep first component as-is, capitalize the rest
  return (
    components[0] +
    components
      .slice(1)
      .map((x) => x.charAt(0).toUpperCase() + x.slice(1))
      .join("")
  );
}

/**
 * Convert SCREAMING_SNAKE_CASE to camelCase.
 *
 * @param name - String in SCREAMING_SNAKE_CASE format
 * @returns String in camelCase format
 *
 * @example
 * ```typescript
 * screamingToCamel("GLOBAL_PAGE_THRESHOLD") // 'globalPageThreshold'
 * screamingToCamel("MIN_JUMP") // 'minJump'
 * ```
 */
export function screamingToCamel(name: string): string {
  return snakeToCamel(name.toLowerCase());
}

/**
 * Validate that a dictionary has no keys that would clash after case conversion.
 *
 * This checks if both camelCase and snake_case versions of the same logical key exist,
 * which would cause data loss or confusion during conversion.
 *
 * @param data - Dictionary to validate
 * @param path - Current path in nested structure (for error messages)
 * @throws Error if clashing keys are found
 *
 * @example
 * ```typescript
 * validateNoKeyClash({ userName: "Alice", user_name: "Bob" })
 * // Throws: Key clash detected: 'userName' and 'user_name' both convert to 'user_name'
 *
 * validateNoKeyClash({ userName: "Alice", email: "test@example.com" })
 * // No error - keys don't clash
 * ```
 */
export function validateNoKeyClash(
  data: Record<string, any>,
  path: string = ""
): void {
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    return;
  }

  // Build a mapping of converted keys to original keys
  const snakeToOriginal: Record<string, string> = {};

  for (const key of Object.keys(data)) {
    const snakeKey = camelToSnake(key);

    if (snakeKey in snakeToOriginal) {
      const originalKey = snakeToOriginal[snakeKey];
      if (originalKey !== key) {
        const prefix = path ? `at '${path}': ` : "";
        throw new Error(
          `${prefix}Key clash detected: '${originalKey}' and '${key}' ` +
            `both convert to '${snakeKey}'. Please use only one naming convention.`
        );
      }
    } else {
      snakeToOriginal[snakeKey] = key;
    }
  }

  // Recursively validate nested structures
  for (const [key, value] of Object.entries(data)) {
    const currentPath = path ? `${path}.${key}` : key;

    if (typeof value === "object" && value !== null && !Array.isArray(value)) {
      validateNoKeyClash(value, currentPath);
    } else if (Array.isArray(value)) {
      value.forEach((item, i) => {
        if (typeof item === "object" && item !== null && !Array.isArray(item)) {
          validateNoKeyClash(item, `${currentPath}[${i}]`);
        }
      });
    }
  }
}

/**
 * Recursively convert dictionary keys from camelCase to snake_case.
 *
 * @param data - Dictionary with camelCase keys
 * @returns Dictionary with snake_case keys
 */
export function convertDictKeysToSnake(
  data: Record<string, any>
): Record<string, any> {
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    return data;
  }

  const result: Record<string, any> = {};

  for (const [key, value] of Object.entries(data)) {
    // Convert the key
    const snakeKey = camelToSnake(key);

    // Recursively process the value
    if (typeof value === "object" && value !== null && !Array.isArray(value)) {
      result[snakeKey] = convertDictKeysToSnake(value);
    } else if (Array.isArray(value)) {
      result[snakeKey] = value.map((item) =>
        typeof item === "object" && item !== null && !Array.isArray(item)
          ? convertDictKeysToSnake(item)
          : item
      );
    } else {
      result[snakeKey] = value;
    }
  }

  return result;
}

/**
 * Recursively convert dictionary keys from snake_case to camelCase.
 *
 * @param data - Dictionary with snake_case keys
 * @returns Dictionary with camelCase keys
 */
export function convertDictKeysToCamel(
  data: Record<string, any>
): Record<string, any> {
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    return data;
  }

  const result: Record<string, any> = {};

  for (const [key, value] of Object.entries(data)) {
    // Convert the key
    const camelKey = snakeToCamel(key);

    // Recursively process the value
    if (typeof value === "object" && value !== null && !Array.isArray(value)) {
      result[camelKey] = convertDictKeysToCamel(value);
    } else if (Array.isArray(value)) {
      result[camelKey] = value.map((item) =>
        typeof item === "object" && item !== null && !Array.isArray(item)
          ? convertDictKeysToCamel(item)
          : item
      );
    } else {
      result[camelKey] = value;
    }
  }

  return result;
}
