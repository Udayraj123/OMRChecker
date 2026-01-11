/**
 * File utility functions for OMRChecker
 *
 * TypeScript port of src/utils/file.py
 */

import { promises as fs } from 'fs';
import * as path from 'path';
import { InputFileNotFoundError, ConfigLoadError } from '../core/exceptions';
import { logger } from './logger';

/**
 * Load and parse a JSON file
 *
 * @param filePath - Path to the JSON file
 * @returns Parsed JSON object
 * @throws InputFileNotFoundError if file doesn't exist
 * @throws ConfigLoadError if JSON is invalid
 */
export async function loadJson(filePath: string): Promise<Record<string, unknown>> {
  try {
    await fs.access(filePath);
  } catch {
    throw new InputFileNotFoundError(filePath, 'JSON');
  }

  try {
    const content = await fs.readFile(filePath, 'utf-8');
    return JSON.parse(content);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    logger.fatal(`Error when loading json file at: '${filePath}'`, String(error));
    throw new ConfigLoadError(filePath, `Invalid JSON format: ${message}`);
  }
}

export class PathUtils {
  outputDir: string;
  saveMarkedDir: string;
  imageMetricsDir: string;
  resultsDir: string;
  manualDir: string;
  errorsDir: string;
  multiMarkedDir: string;
  evaluationsDir: string;
  debugDir: string;

  private static printableChars = new Set(
    ' !\\"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\\t\\n\\r\\x0b\\x0c'.split(
      ''
    )
  );

  constructor(outputDir: string) {
    this.outputDir = outputDir;
    this.saveMarkedDir = path.join(outputDir, 'CheckedOMRs');
    this.imageMetricsDir = path.join(outputDir, 'ImageMetrics');
    this.resultsDir = path.join(outputDir, 'Results');
    this.manualDir = path.join(outputDir, 'Manual');
    this.errorsDir = path.join(this.manualDir, 'ErrorFiles');
    this.multiMarkedDir = path.join(this.manualDir, 'MultiMarkedFiles');
    this.evaluationsDir = path.join(outputDir, 'Evaluations');
    this.debugDir = path.join(outputDir, 'Debug');
  }

  static removeNonUtfCharacters(pathString: string): string {
    return pathString
      .split('')
      .filter((x) => PathUtils.printableChars.has(x))
      .join('');
  }

  static sepBasedPosixPath(pathString: string): string {
    // Normalize path and convert to posix format
    let normalized = path.normalize(pathString);

    // Convert Windows paths to posix
    if (path.sep === '\\' || normalized.includes('\\')) {
      normalized = normalized.split(path.sep).join('/');
    }

    return PathUtils.removeNonUtfCharacters(normalized);
  }

  async createOutputDirectories(): Promise<void> {
    logger.info('Checking Directories...');

    // Create save marked directory
    await fs.mkdir(this.saveMarkedDir, { recursive: true });

    // Main output directories
    const mainOutputDirs = [
      path.join(this.saveMarkedDir, 'colored'),
      path.join(this.saveMarkedDir, 'stack'),
      path.join(this.saveMarkedDir, 'stack', 'colored'),
      path.join(this.saveMarkedDir, '_MULTI_'),
      path.join(this.saveMarkedDir, '_MULTI_', 'colored'),
    ];

    for (const dir of mainOutputDirs) {
      try {
        await fs.access(dir);
      } catch {
        await fs.mkdir(dir, { recursive: true });
        logger.info(`Created : ${dir}`);
      }
    }

    // Image buckets
    const imageBuckets = [this.manualDir, this.multiMarkedDir, this.errorsDir];

    for (const dir of imageBuckets) {
      try {
        await fs.access(dir);
      } catch {
        logger.info(`Created : ${dir}`);
        await fs.mkdir(dir, { recursive: true });
        await fs.mkdir(path.join(dir, 'colored'), { recursive: true });
      }
    }

    // Non-image directories
    const nonImageDirs = [
      this.resultsDir,
      this.imageMetricsDir,
      this.evaluationsDir,
    ];

    for (const dir of nonImageDirs) {
      try {
        await fs.access(dir);
      } catch {
        logger.info(`Created : ${dir}`);
        await fs.mkdir(dir, { recursive: true });
      }
    }
  }
}

