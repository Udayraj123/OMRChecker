/**
 * Tests for file utilities
 *
 * TypeScript port of src/tests/test_file.py
 */

import { describe, it, expect } from 'vitest';
import { promises as fs } from 'fs';
import * as path from 'path';
import * as os from 'os';
import { PathUtils, loadJson } from '../file';
import { InputFileNotFoundError, ConfigLoadError } from '../../core/exceptions';

describe('FileUtils', () => {
  describe('loadJson', () => {
    it('should load valid JSON file', async () => {
      const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'test-'));
      const tmpFile = path.join(tmpDir, 'test.json');
      const testData = { key: 'value', number: 42 };

      await fs.writeFile(tmpFile, JSON.stringify(testData));

      try {
        const result = await loadJson(tmpFile);
        expect(result).toEqual(testData);
      } finally {
        await fs.rm(tmpDir, { recursive: true });
      }
    });

    it('should throw InputFileNotFoundError for non-existent file', async () => {
      await expect(loadJson('nonexistent_file.json')).rejects.toThrow(
        InputFileNotFoundError
      );
    });

    it('should throw ConfigLoadError for invalid JSON', async () => {
      const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'test-'));
      const tmpFile = path.join(tmpDir, 'invalid.json');

      await fs.writeFile(tmpFile, '{invalid json content');

      try {
        await expect(loadJson(tmpFile)).rejects.toThrow(ConfigLoadError);
      } finally {
        await fs.rm(tmpDir, { recursive: true });
      }
    });
  });

  describe('PathUtils', () => {
    it('should initialize with output directory', () => {
      const outputDir = '/tmp/test_output';
      const pathUtils = new PathUtils(outputDir);

      expect(pathUtils.outputDir).toBe(outputDir);
      expect(pathUtils.saveMarkedDir).toBe(
        path.join(outputDir, 'CheckedOMRs')
      );
      expect(pathUtils.resultsDir).toBe(path.join(outputDir, 'Results'));
      expect(pathUtils.manualDir).toBe(path.join(outputDir, 'Manual'));
    });

    it('should create output directories', async () => {
      const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'test-'));
      const outputDir = path.join(tmpDir, 'test_output');

      try {
        const pathUtils = new PathUtils(outputDir);
        await pathUtils.createOutputDirectories();

        // Verify directories exist
        expect(await fs.stat(pathUtils.saveMarkedDir)).toBeDefined();
        expect(await fs.stat(pathUtils.resultsDir)).toBeDefined();
        expect(await fs.stat(pathUtils.manualDir)).toBeDefined();
      } finally {
        await fs.rm(tmpDir, { recursive: true });
      }
    });

    it('should remove non-utf characters', () => {
      const path = 'test/path/file.txt';
      const result = PathUtils.removeNonUtfCharacters(path);
      expect(result).toBe('test/path/file.txt');
    });

    it('should convert to posix path', () => {
      const testPath = 'test/path/file.txt';
      const result = PathUtils.sepBasedPosixPath(testPath);
      expect(result).toContain('/');
    });
  });
});

