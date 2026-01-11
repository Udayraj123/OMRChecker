/**
 * Tests for CSV utilities
 *
 * TypeScript port of src/tests/test_csv.py
 */

import { describe, it, expect } from 'vitest';
import { promises as fs } from 'fs';
import * as path from 'path';
import * as os from 'os';
import Papa from 'papaparse';
import { threadSafeCsvAppend } from '../csv';

describe('CSVUtils', () => {
  it('should append basic data to CSV', async () => {
    const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'test-'));
    const tmpFile = path.join(tmpDir, 'test.csv');

    try {
      const dataLine = ['Alice', '30', 'Engineer'];
      await threadSafeCsvAppend(tmpFile, dataLine);

      // Verify file exists
      await fs.access(tmpFile);

      // Read and verify content
      const content = await fs.readFile(tmpFile, 'utf-8');
      const parsed = Papa.parse(content, { header: false });

      expect(parsed.data.length).toBeGreaterThan(0);
      expect(parsed.data[0]).toEqual(['Alice', '30', 'Engineer']);
    } finally {
      await fs.rm(tmpDir, { recursive: true });
    }
  });

  it('should append multiple lines', async () => {
    const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'test-'));
    const tmpFile = path.join(tmpDir, 'test.csv');

    try {
      await threadSafeCsvAppend(tmpFile, ['Alice', '30', 'Engineer']);
      await threadSafeCsvAppend(tmpFile, ['Bob', '25', 'Designer']);

      const content = await fs.readFile(tmpFile, 'utf-8');
      const parsed = Papa.parse(content, { header: false });

      expect(parsed.data.length).toBeGreaterThan(1);
      expect(parsed.data[0]).toEqual(['Alice', '30', 'Engineer']);
      expect(parsed.data[1]).toEqual(['Bob', '25', 'Designer']);
    } finally {
      await fs.rm(tmpDir, { recursive: true });
    }
  });

  it('should append numeric data', async () => {
    const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'test-'));
    const tmpFile = path.join(tmpDir, 'test.csv');

    try {
      const dataLine = ['100', '200', '300'];
      await threadSafeCsvAppend(tmpFile, dataLine);

      const content = await fs.readFile(tmpFile, 'utf-8');
      const parsed = Papa.parse(content, { header: false });

      expect(parsed.data[0]).toEqual(['100', '200', '300']);
    } finally {
      await fs.rm(tmpDir, { recursive: true });
    }
  });

  it('should handle empty line', async () => {
    const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'test-'));
    const tmpFile = path.join(tmpDir, 'test.csv');

    try {
      await threadSafeCsvAppend(tmpFile, []);
      await fs.access(tmpFile);
    } finally {
      await fs.rm(tmpDir, { recursive: true });
    }
  });
});

