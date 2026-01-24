/**
 * Repository for managing detection results.
 *
 * TypeScript port of src/processors/repositories/detection_repository.py
 * Replaces nested dictionary-based aggregate management with clean repository pattern.
 */

import {
  BarcodeFieldDetectionResult,
  BubbleFieldDetectionResult,
  BubbleMeanValue,
  FileDetectionResults,
  OCRFieldDetectionResult,
} from '../detection/models';

/**
 * Repository for managing detection results at different levels.
 *
 * Replaces the complex nested dictionary structure in FilePassAggregates
 * with a clean, queryable interface.
 */
export class DetectionRepository {
  private currentFileResults: FileDetectionResults | null = null;
  private fileResults: Map<string, FileDetectionResults> = new Map();
  private directoryPath: string | null = null;

  // File-level operations
  /**
   * Initialize a new file for detection results.
   *
   * @param filePath - Path to the file being processed
   */
  initializeFile(filePath: string): void {
    // Normalize file_path to string for consistent storage
    const filePathStr = String(filePath);
    this.currentFileResults = new FileDetectionResults(filePathStr);
  }

  /**
   * Finalize current file and store results.
   */
  finalizeFile(): void {
    if (this.currentFileResults !== null) {
      this.fileResults.set(
        this.currentFileResults.filePath,
        this.currentFileResults
      );
    }
  }

  /**
   * Get results for current file being processed.
   *
   * @returns FileDetectionResults for current file
   * @throws RuntimeError if no file is currently being processed
   */
  getCurrentFileResults(): FileDetectionResults {
    if (this.currentFileResults === null) {
      throw new Error('No file currently being processed');
    }
    return this.currentFileResults;
  }

  /**
   * Get results for a specific file.
   *
   * @param filePath - Path to the file
   * @returns FileDetectionResults for the file
   * @throws Error if file has not been processed
   */
  getFileResults(filePath: string): FileDetectionResults {
    // Normalize file_path to string for consistent lookup
    const filePathStr = String(filePath);

    // Paths are normalized to string in initializeFile(), so exact match is sufficient
    const result = this.fileResults.get(filePathStr);
    if (!result) {
      // If not found, raise error with helpful message
      const availablePaths = Array.from(this.fileResults.keys());
      throw new Error(
        `No results found for file: ${filePathStr}. ` +
        `Available files: ${availablePaths.join(', ')}`
      );
    }
    return result;
  }

  // Field-level operations
  /**
   * Save bubble field detection result.
   *
   * @param fieldId - Unique field identifier
   * @param result - Detection result for the field
   */
  saveBubbleField(fieldId: string, result: BubbleFieldDetectionResult): void {
    const current = this.getCurrentFileResults();
    current.bubbleFields.set(fieldId, result);
  }

  /**
   * Save OCR field detection result.
   *
   * @param fieldId - Unique field identifier
   * @param result - Detection result for the field
   */
  saveOcrField(fieldId: string, result: OCRFieldDetectionResult): void {
    const current = this.getCurrentFileResults();
    current.ocrFields.set(fieldId, result);
  }

  /**
   * Save barcode field detection result.
   *
   * @param fieldId - Unique field identifier
   * @param result - Detection result for the field
   */
  saveBarcodeField(fieldId: string, result: BarcodeFieldDetectionResult): void {
    const current = this.getCurrentFileResults();
    current.barcodeFields.set(fieldId, result);
  }

  /**
   * Get bubble field result from current file.
   *
   * @param fieldId - Field identifier
   * @returns BubbleFieldDetectionResult
   * @throws KeyError if field not found
   */
  getBubbleField(fieldId: string): BubbleFieldDetectionResult {
    const current = this.getCurrentFileResults();
    const result = current.bubbleFields.get(fieldId);
    if (!result) {
      throw new Error(`Bubble field not found: ${fieldId}`);
    }
    return result;
  }

  /**
   * Get OCR field result from current file.
   *
   * @param fieldId - Field identifier
   * @returns OCRFieldDetectionResult
   * @throws KeyError if field not found
   */
  getOcrField(fieldId: string): OCRFieldDetectionResult {
    const current = this.getCurrentFileResults();
    const result = current.ocrFields.get(fieldId);
    if (!result) {
      throw new Error(`OCR field not found: ${fieldId}`);
    }
    return result;
  }

  /**
   * Get barcode field result from current file.
   *
   * @param fieldId - Field identifier
   * @returns BarcodeFieldDetectionResult
   * @throws KeyError if field not found
   */
  getBarcodeField(fieldId: string): BarcodeFieldDetectionResult {
    const current = this.getCurrentFileResults();
    const result = current.barcodeFields.get(fieldId);
    if (!result) {
      throw new Error(`Barcode field not found: ${fieldId}`);
    }
    return result;
  }

  // Query operations
  /**
   * Get all bubble means across all bubble fields in current file.
   *
   * @returns List of all BubbleMeanValue objects
   */
  getAllBubbleMeansForCurrentFile(): BubbleMeanValue[] {
    const current = this.getCurrentFileResults();
    return current.allBubbleMeans;
  }

  /**
   * Get all bubble mean values as floats for current file.
   *
   * @returns List of mean values
   */
  getAllBubbleMeanValuesForCurrentFile(): number[] {
    const current = this.getCurrentFileResults();
    return current.allBubbleMeanValues;
  }

  /**
   * Get all bubble field results for current file.
   *
   * @returns Dictionary mapping field_id to BubbleFieldDetectionResult
   */
  getAllBubbleFieldsForCurrentFile(): Map<string, BubbleFieldDetectionResult> {
    const current = this.getCurrentFileResults();
    return current.bubbleFields;
  }

  /**
   * Get all bubble field results for current file as a Record (dictionary-like object).
   *
   * @returns Record mapping field_id to BubbleFieldDetectionResult
   */
  getAllBubbleFieldsForCurrentFileAsRecord(): Record<string, BubbleFieldDetectionResult> {
    const current = this.getCurrentFileResults();
    const result: Record<string, BubbleFieldDetectionResult> = {};
    for (const [fieldId, fieldResult] of current.bubbleFields.entries()) {
      result[fieldId] = fieldResult;
    }
    return result;
  }

  /**
   * Get all bubble mean std deviations for current file.
   *
   * @returns List of std deviations
   */
  getAllBubbleMeanStdDeviationsForCurrentFile(): number[] {
    const current = this.getCurrentFileResults();
    const stdDeviations: number[] = [];
    for (const fieldResult of current.bubbleFields.values()) {
      stdDeviations.push(fieldResult.stdDeviation);
    }
    return stdDeviations;
  }

  // Directory-level operations
  /**
   * Initialize repository for a directory.
   *
   * @param directoryPath - Path to the directory being processed
   */
  initializeDirectory(directoryPath: string): void {
    this.directoryPath = directoryPath;
    this.fileResults.clear();
    this.currentFileResults = null;
  }

  /**
   * Get results for all processed files in directory.
   *
   * @returns Dictionary mapping file_path to FileDetectionResults
   */
  getAllFileResults(): Map<string, FileDetectionResults> {
    return new Map(this.fileResults);
  }

  /**
   * Clear all stored results.
   */
  clear(): void {
    this.currentFileResults = null;
    this.fileResults.clear();
    this.directoryPath = null;
  }

  // Statistics
  /**
   * Get total number of files processed.
   */
  getTotalFilesProcessed(): number {
    return this.fileResults.size;
  }

  /**
   * Get total number of fields in current file.
   */
  getTotalFieldsInCurrentFile(): number {
    const current = this.getCurrentFileResults();
    return current.numFields;
  }

  /**
   * Readable representation.
   */
  toString(): string {
    return (
      `DetectionRepository(` +
      `directory=${this.directoryPath}, ` +
      `files=${this.fileResults.size}, ` +
      `current_file=${this.currentFileResults?.filePath ?? null})`
    );
  }
}

