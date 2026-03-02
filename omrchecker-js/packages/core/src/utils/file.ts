/**
 * Migrated from Python: src/utils/file.py
 * Note: loadJson uses fetch() in browser. SaveImageOps omitted (requires OpenCV).
 * PathUtils.createOutputDirectories() omitted (no file system in browser).
 */
import { ConfigLoadError, InputFileNotFoundError } from './exceptions';

/**
 * Load and parse a JSON file from a URL or path.
 * Browser equivalent: uses fetch(). Throws InputFileNotFoundError (404) or ConfigLoadError (parse error).
 */
export async function loadJson(path: string): Promise<Record<string, unknown>> {
  let response: Response;
  try {
    response = await fetch(path);
  } catch {
    throw new InputFileNotFoundError(path, 'JSON');
  }
  if (!response.ok) {
    throw new InputFileNotFoundError(path, 'JSON');
  }
  const text = await response.text();
  try {
    return JSON.parse(text) as Record<string, unknown>;
  } catch (error) {
    throw new ConfigLoadError(path, `Invalid JSON format: ${error}`);
  }
}

/**
 * Synchronous JSON parse from a string (for testing and in-memory use).
 */
export function parseJsonString(jsonStr: string, sourcePath = '<inline>'): Record<string, unknown> {
  try {
    return JSON.parse(jsonStr) as Record<string, unknown>;
  } catch (error) {
    throw new ConfigLoadError(sourcePath, `Invalid JSON format: ${error}`);
  }
}

export class PathUtils {
  private static readonly PRINTABLE = new Set(
    Array.from({ length: 128 }, (_, i) => String.fromCharCode(i)).filter(
      c => c.trim() !== '' || c === ' '
    )
  );

  static removeNonUtfCharacters(pathString: string): string {
    return [...pathString].filter(c => PathUtils.PRINTABLE.has(c)).join('');
  }

  static sepBasedPosixPath(path: string): string {
    // Normalize backslashes to forward slashes
    const normalized = path.replace(/\\/g, '/');
    return PathUtils.removeNonUtfCharacters(normalized);
  }

  outputDir: string;
  saveMarkedDir: string;
  imageMetricsDir: string;
  resultsDir: string;
  manualDir: string;
  errorsDir: string;
  multiMarkedDir: string;
  evaluationsDir: string;
  debugDir: string;

  constructor(outputDir: string) {
    this.outputDir = outputDir;
    this.saveMarkedDir = `${outputDir}/CheckedOMRs`;
    this.imageMetricsDir = `${outputDir}/ImageMetrics`;
    this.resultsDir = `${outputDir}/Results`;
    this.manualDir = `${outputDir}/Manual`;
    this.errorsDir = `${this.manualDir}/ErrorFiles`;
    this.multiMarkedDir = `${this.manualDir}/MultiMarkedFiles`;
    this.evaluationsDir = `${outputDir}/Evaluations`;
    this.debugDir = `${outputDir}/Debug`;
  }
}
