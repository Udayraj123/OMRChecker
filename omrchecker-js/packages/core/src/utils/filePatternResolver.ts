/**
 * Utility for resolving file path patterns using OMR field data.
 *
 * TypeScript port of src/utils/file_pattern_resolver.py
 * Maintains 1:1 correspondence with Python implementation.
 */

import { Logger } from './logger';

const logger = new Logger('FilePatternResolver');

/**
 * Resolves file path/name patterns using field data.
 *
 * This utility can be used by any processor that needs to generate
 * dynamic file paths based on OMR detected values or other fields.
 *
 * Features:
 * - Format patterns with {field} placeholders
 * - Auto-preserve original file extensions if not specified
 * - Sanitize paths (remove invalid characters)
 * - Handle path collisions with configurable strategies
 *
 * Usage:
 * ```typescript
 * const resolver = new FilePatternResolver();
 * const path = resolver.resolvePattern(
 *   "booklet_{code}/{roll}_{score}",
 *   { code: "A", roll: "12345", score: "95" },
 *   "/path/to/image.jpg",
 *   "increment"
 * );
 * // Result: "booklet_A/12345_95.jpg"
 * ```
 */
export class FilePatternResolver {
  private baseDir?: string;
  private existsCheck?: (path: string) => boolean;

  /**
   * Initialize the pattern resolver.
   *
   * @param baseDir - Optional base directory for all resolved paths
   * @param existsCheck - Optional function to check if a path exists (for collision handling).
   *                     When not provided, collision handling is skipped and the resolved path is always returned.
   *                     In Node.js: pass (p) => require('fs').existsSync(p)
   */
  constructor(baseDir?: string, existsCheck?: (path: string) => boolean) {
    this.baseDir = baseDir;
    this.existsCheck = existsCheck;
  }

  /**
   * Resolve a file path pattern using provided fields.
   *
   * Port of resolve_pattern() from Python. Uses pattern.format(**fields) semantics:
   * - All {field} placeholders in pattern must exist in fields (otherwise logs and returns null).
   * - Preserves original file extension if not in pattern.
   *
   * @param pattern - Pattern string with {field} placeholders
   *                  e.g., "folder_{booklet}/{roll}_{score}"
   *                  If extension not in pattern, preserves original extension
   * @param fields - Dictionary of field values for substitution
   * @param originalPath - Original file path (for extension preservation)
   * @param collisionStrategy - How to handle existing files:
   *                - "skip": Return null if file exists
   *                - "increment": Append _001, _002, etc.
   *                - "overwrite": Allow overwriting
   * @returns Resolved path string, or null if collision with "skip" strategy or undefined field
   */
  resolvePattern(
    pattern: string,
    fields: Record<string, any>,
    originalPath?: string,
    collisionStrategy: 'skip' | 'increment' | 'overwrite' = 'skip'
  ): string | null {
    try {
      // Validate: every {key} in pattern must be in fields (matches Python KeyError on format(**fields))
      const placeholderRegex = /\{([a-zA-Z_][a-zA-Z0-9_]*)\}/g;
      let m: RegExpExecArray | null;
      while ((m = placeholderRegex.exec(pattern)) !== null) {
        const key = m[1];
        if (!(key in fields)) {
          logger.warn(`Pattern references undefined field: ${key}`);
          return null;
        }
      }

      // Format the pattern with fields (only {key} placeholders, no format specifiers)
      let formatted = pattern;
      for (const [key, value] of Object.entries(fields)) {
        const escaped = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        formatted = formatted.replace(new RegExp(`\\{${escaped}\\}`, 'g'), String(value));
      }

      // Sanitize the path (remove invalid characters)
      const sanitized = this._sanitizePath(formatted);

      // Handle extension preservation
      let resolvedPath = sanitized;
      if (originalPath && !this._hasExtension(resolvedPath)) {
        const originalExt = this._getExtension(originalPath);
        resolvedPath = resolvedPath + originalExt;
      }

      // Apply base directory if set
      if (this.baseDir) {
        resolvedPath = `${this.baseDir}/${resolvedPath}`;
      }

      return this._handleCollision(resolvedPath, collisionStrategy);
    } catch (error) {
      const errMsg = error instanceof Error ? error.message : String(error);
      logger.error(`Error resolving pattern '${pattern}': ${errMsg}`);
      return null;
    }
  }

  /**
   * Sanitize path string by removing invalid characters.
   *
   * @param pathStr - Path string to sanitize
   * @returns Sanitized path string
   */
  private _sanitizePath(pathStr: string): string {
    // Replace invalid filename characters with underscore
    // Invalid: < > : " / \ | ? *
    // Note: We need to preserve / for directory separators
    // Split by / to handle directory parts separately
    const parts = pathStr.split('/');
    const sanitizedParts: string[] = [];

    for (const part of parts) {
      // Sanitize each path component
      // Remove/replace invalid chars except forward slash
      let sanitized = part.replace(/[<>:"|?*\\]/g, '_');
      // Remove any double underscores
      sanitized = sanitized.replace(/_+/g, '_');
      // Strip leading/trailing underscores and spaces
      sanitized = sanitized.replace(/^[_ ]+|[_ ]+$/g, '');
      if (sanitized) {
        // Only add non-empty parts
        sanitizedParts.push(sanitized);
      }
    }

    return sanitizedParts.join('/');
  }

  /**
   * Handle file path collisions based on strategy.
   *
   * Port of _handle_collision() from Python. When existsCheck is not provided
   * (e.g. browser), always returns the path. When provided, implements skip/overwrite/increment.
   *
   * @param path - The path to check
   * @param strategy - Collision handling strategy
   * @returns Final path, or null if skipping collision
   */
  private _handleCollision(path: string, strategy: string): string | null {
    if (!this.existsCheck) {
      return path;
    }
    if (!this.existsCheck(path)) {
      return path;
    }
    if (strategy === 'skip') {
      logger.debug(`File exists, skipping: ${path.split('/').pop()}`);
      return null;
    }
    if (strategy === 'overwrite') {
      logger.debug(`File exists, will overwrite: ${path.split('/').pop()}`);
      return path;
    }
    if (strategy === 'increment') {
      const lastSlash = path.lastIndexOf('/');
      const dir = lastSlash >= 0 ? path.slice(0, lastSlash) : '';
      const file = lastSlash >= 0 ? path.slice(lastSlash + 1) : path;
      const dot = file.lastIndexOf('.');
      const stem = dot > 0 ? file.slice(0, dot) : file;
      const suffix = dot > 0 ? file.slice(dot) : '';
      let counter = 1;
      while (counter < 9999) {
        const newName = `${stem}_${String(counter).padStart(3, '0')}${suffix}`;
        const newPath = dir ? `${dir}/${newName}` : newName;
        if (!this.existsCheck(newPath)) {
          logger.debug(`File exists, using incremented name: ${newName}`);
          return newPath;
        }
        counter += 1;
        if (counter == 9999) {
          logger.error(`Too many collisions for ${stem}, giving up`);
          return null;
        }
      }
    }
    logger.warn(`Unknown collision strategy '${strategy}', skipping`);
    return null;
  }

  /**
   * Check if path has an extension.
   */
  private _hasExtension(path: string): boolean {
    const parts = path.split('/');
    const filename = parts[parts.length - 1];
    return filename.includes('.') && filename.lastIndexOf('.') > 0;
  }

  /**
   * Get extension from path.
   */
  private _getExtension(path: string): string {
    const parts = path.split('/');
    const filename = parts[parts.length - 1];
    const lastDot = filename.lastIndexOf('.');
    return lastDot > 0 ? filename.substring(lastDot) : '';
  }

  /**
   * Resolve multiple patterns in batch.
   *
   * @param patternsAndFields - List of [pattern, fields, originalPath] tuples
   * @param collisionStrategy - Collision handling strategy for all
   * @returns List of [resolvedPath, fields] tuples
   */
  resolveBatch(
    patternsAndFields: Array<[string, Record<string, any>, string?]>,
    collisionStrategy: 'skip' | 'increment' | 'overwrite' = 'skip'
  ): Array<[string | null, Record<string, any>]> {
    const results: Array<[string | null, Record<string, any>]> = [];
    for (const [pattern, fields, originalPath] of patternsAndFields) {
      const resolved = this.resolvePattern(
        pattern,
        fields,
        originalPath,
        collisionStrategy
      );
      results.push([resolved, fields]);
    }
    return results;
  }
}

