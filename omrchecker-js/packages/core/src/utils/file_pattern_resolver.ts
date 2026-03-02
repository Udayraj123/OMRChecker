/**
 * FilePatternResolver utility for OMRChecker.
 *
 * Migrated from src/utils/file_pattern_resolver.py.
 * Resolves file name/path patterns by substituting field values,
 * sanitizing the result, and handling path collisions.
 *
 * Uses dependency injection for the existence check so it works in
 * browser environments (no real file system).
 */

import { logger } from './logger';

export class FilePatternResolver {
  private baseDir?: string;
  private existsCheck: (path: string) => boolean;

  constructor(options?: { baseDir?: string; existsCheck?: (path: string) => boolean }) {
    this.baseDir = options?.baseDir;
    this.existsCheck = options?.existsCheck ?? (() => false);
  }

  resolvePattern(
    pattern: string,
    fields: Record<string, string>,
    options?: { originalPath?: string; collisionStrategy?: string }
  ): string | null {
    const { originalPath, collisionStrategy = 'skip' } = options ?? {};
    try {
      const formatted = this._formatPattern(pattern, fields);
      const sanitized = this._sanitizePath(formatted);
      let resolvedPath = sanitized;

      // Extension preservation: if resolved path has no extension, inherit from original
      if (originalPath && !this._getSuffix(resolvedPath)) {
        const ext = this._getSuffix(originalPath);
        if (ext) resolvedPath = resolvedPath + ext;
      }

      // Prepend base directory if set
      if (this.baseDir) {
        resolvedPath = this.baseDir + '/' + resolvedPath;
      }

      return this._handleCollision(resolvedPath, collisionStrategy);
    } catch (e: unknown) {
      if (e instanceof KeyError) {
        logger.warning(`Pattern references undefined field: ${e.key}`);
      } else {
        logger.error(`Error resolving pattern '${pattern}': ${e}`);
      }
      return null;
    }
  }

  private _formatPattern(pattern: string, fields: Record<string, string>): string {
    return pattern.replace(/\{(\w+)\}/g, (_match, key: string) => {
      if (!(key in fields)) {
        throw new KeyError(key);
      }
      return fields[key];
    });
  }

  private _sanitizePath(pathStr: string): string {
    const parts = pathStr.split('/');
    const sanitizedParts: string[] = [];
    for (const part of parts) {
      // Replace forbidden characters with underscore
      let sanitized = part.replace(/[<>:"|?*\\]/g, '_');
      // Collapse consecutive underscores
      sanitized = sanitized.replace(/_+/g, '_');
      // Strip leading/trailing underscores and spaces
      sanitized = sanitized.replace(/^[_ ]+|[_ ]+$/g, '');
      if (sanitized) {
        sanitizedParts.push(sanitized);
      }
    }
    return sanitizedParts.join('/');
  }

  /**
   * Returns the file extension including the leading dot (e.g. ".jpg"),
   * or an empty string if there is no extension.
   */
  private _getSuffix(pathStr: string): string {
    const name = pathStr.split('/').pop() ?? '';
    const dotIndex = name.lastIndexOf('.');
    if (dotIndex <= 0) return '';
    return name.slice(dotIndex);
  }

  /**
   * Returns the file name without extension (stem).
   */
  private _getStem(pathStr: string): string {
    const name = pathStr.split('/').pop() ?? '';
    const dotIndex = name.lastIndexOf('.');
    if (dotIndex <= 0) return name;
    return name.slice(0, dotIndex);
  }

  /**
   * Returns the parent directory path (everything before the last '/').
   */
  private _getParent(pathStr: string): string {
    const idx = pathStr.lastIndexOf('/');
    if (idx < 0) return '';
    return pathStr.slice(0, idx);
  }

  private _handleCollision(path: string, strategy: string): string | null {
    if (!this.existsCheck(path)) {
      return path;
    }
    if (strategy === 'skip') {
      return null;
    }
    if (strategy === 'overwrite') {
      return path;
    }
    if (strategy === 'increment') {
      const stem = this._getStem(path);
      const suffix = this._getSuffix(path);
      const parent = this._getParent(path);
      let counter = 1;
      while (counter < 9999) {
        const newName = `${stem}_${String(counter).padStart(3, '0')}${suffix}`;
        const newPath = parent ? parent + '/' + newName : newName;
        if (!this.existsCheck(newPath)) {
          return newPath;
        }
        counter++;
      }
      return null;
    }
    return null;
  }

  resolveBatch(
    patternsAndFields: Array<[string, Record<string, string>, string?]>,
    collisionStrategy = 'skip'
  ): Array<[string | null, Record<string, string>]> {
    const results: Array<[string | null, Record<string, string>]> = [];
    for (const [pattern, fields, originalPath] of patternsAndFields) {
      const resolved = this.resolvePattern(pattern, fields, { originalPath, collisionStrategy });
      results.push([resolved, fields]);
    }
    return results;
  }
}

/** Internal error class used to signal a missing field key. */
class KeyError extends Error {
  constructor(public readonly key: string) {
    super(`KeyError: '${key}'`);
    this.name = 'KeyError';
  }
}
