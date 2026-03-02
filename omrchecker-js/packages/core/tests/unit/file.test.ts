/**
 * Translated from Python: src/tests/utils/__tests__/test_file.py
 * 7 tests covering loadJson (via parseJsonString), PathUtils static methods, and PathUtils constructor.
 */
import { describe, it, expect } from 'vitest';
import { loadJson, parseJsonString, PathUtils } from '../../src/utils/file';
import { InputFileNotFoundError, ConfigLoadError } from '../../src/utils/exceptions';

// ---------------------------------------------------------------------------
// loadJson / parseJsonString
// ---------------------------------------------------------------------------

describe('file utils', () => {
  // test_load_json_success
  // Python: reads JSON from a tempfile and asserts parsed dict.
  // TS: use parseJsonString (synchronous, in-memory equivalent).
  it('test_load_json_success: parseJsonString returns parsed object for valid JSON', () => {
    const result = parseJsonString('{"key":"value","number":42}');
    expect(result).toEqual({ key: 'value', number: 42 });
  });

  // test_load_json_file_not_found
  // Python: raises InputFileNotFoundError when path does not exist.
  // TS: loadJson with a path that fetch will reject (jsdom has no network).
  it('test_load_json_file_not_found: loadJson throws InputFileNotFoundError for nonexistent path', async () => {
    await expect(loadJson('/nonexistent.json')).rejects.toThrow(InputFileNotFoundError);
  });

  // test_load_json_invalid_json
  // Python: raises ConfigLoadError when file contains invalid JSON.
  // TS: parseJsonString with an invalid JSON string.
  it('test_load_json_invalid_json: parseJsonString throws ConfigLoadError for invalid JSON', () => {
    expect(() => parseJsonString('{invalid json}')).toThrow(ConfigLoadError);
  });

  // ---------------------------------------------------------------------------
  // PathUtils static methods
  // ---------------------------------------------------------------------------

  // test_remove_non_utf_characters
  // Python: filters non-printable chars from "test/path/file.txt".
  it('test_remove_non_utf_characters: returns same string when all chars are printable ASCII', () => {
    expect(PathUtils.removeNonUtfCharacters('test/path/file.txt')).toBe('test/path/file.txt');
  });

  // test_sep_based_posix_path
  // Python: normalizes path separators and removes non-printable chars.
  it('test_sep_based_posix_path: returns string with forward slashes', () => {
    const result = PathUtils.sepBasedPosixPath('test/path/file.txt');
    expect(result).not.toContain('\\');
    expect(result).toBe('test/path/file.txt');
  });

  // ---------------------------------------------------------------------------
  // PathUtils constructor / initialization
  // ---------------------------------------------------------------------------

  // test_path_utils_initialization
  // Python: asserts save_marked_dir, results_dir, manual_dir are correct Paths.
  it('test_path_utils_initialization: constructor sets all directory properties correctly', () => {
    const utils = new PathUtils('/tmp/output');
    expect(utils.outputDir).toBe('/tmp/output');
    expect(utils.saveMarkedDir).toBe('/tmp/output/CheckedOMRs');
    expect(utils.resultsDir).toBe('/tmp/output/Results');
    expect(utils.manualDir).toBe('/tmp/output/Manual');
  });

  // test_create_output_directories
  // Python: verifies directories are created on disk after create_output_directories().
  // TS: createOutputDirectories() is omitted (no filesystem in browser).
  //     Translated as: verify the PathUtils instance has all expected directory string properties.
  it('test_create_output_directories: PathUtils instance has all expected directory string properties', () => {
    const utils = new PathUtils('/tmp/output');
    expect(utils.saveMarkedDir).toBe('/tmp/output/CheckedOMRs');
    expect(utils.imageMetricsDir).toBe('/tmp/output/ImageMetrics');
    expect(utils.resultsDir).toBe('/tmp/output/Results');
    expect(utils.manualDir).toBe('/tmp/output/Manual');
    expect(utils.errorsDir).toBe('/tmp/output/Manual/ErrorFiles');
    expect(utils.multiMarkedDir).toBe('/tmp/output/Manual/MultiMarkedFiles');
    expect(utils.evaluationsDir).toBe('/tmp/output/Evaluations');
    expect(utils.debugDir).toBe('/tmp/output/Debug');
  });
});
