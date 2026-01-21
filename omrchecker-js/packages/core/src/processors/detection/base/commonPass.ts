/**
 * Common pass infrastructure for detection.
 *
 * TypeScript port of src/processors/detection/base/common_pass.py
 * Defines FilePassAggregates base class for managing aggregates at field, file, and directory levels.
 */

import type { Field } from '../../layout/field/base';
import { StatsByLabel } from '../../../utils/stats';

/**
 * Tuning configuration interface (minimal for now).
 */
export interface TuningConfig {
  [key: string]: unknown;
}

/**
 * Directory level aggregates structure.
 */
export interface DirectoryLevelAggregates {
  initial_directory_path: string;
  file_wise_aggregates: Record<string, FileLevelAggregates>;
  files_count: StatsByLabel;
  [key: string]: unknown;
}

/**
 * File level aggregates structure.
 */
export interface FileLevelAggregates {
  file_path: string;
  fields_count: StatsByLabel;
  field_label_wise_aggregates: Record<string, FieldLevelAggregates>;
  [key: string]: unknown;
}

/**
 * Field level aggregates structure.
 */
export interface FieldLevelAggregates {
  field: Field;
  [key: string]: unknown;
}

/**
 * Base class for managing aggregates at three levels:
 * - Field level: Per-field aggregates
 * - File level: Per-image aggregates
 * - Directory level: Cross-image aggregates
 *
 * This class provides the infrastructure for collecting and managing
 * statistics and data during multi-pass detection and interpretation.
 */
export class FilePassAggregates {
  protected tuningConfig: TuningConfig;
  protected directoryLevelAggregates?: DirectoryLevelAggregates;
  protected fileLevelAggregates?: FileLevelAggregates;
  protected fieldLevelAggregates?: FieldLevelAggregates;

  constructor(tuningConfig: TuningConfig) {
    this.tuningConfig = tuningConfig;
  }

  /**
   * Initialize directory-level aggregates.
   *
   * @param initialDirectoryPath - Path to the initial directory being processed
   * @param _additionalArgs - Additional arguments for subclasses (optional)
   */
  initializeDirectoryLevelAggregates(
    initialDirectoryPath: string,
    ..._additionalArgs: unknown[]
  ): void {
    this.directoryLevelAggregates = {
      initial_directory_path: initialDirectoryPath,
      file_wise_aggregates: {},
      files_count: new StatsByLabel('processed'),
    };
  }

  /**
   * Get directory-level aggregates.
   *
   * @returns Directory-level aggregates
   */
  getDirectoryLevelAggregates(): DirectoryLevelAggregates | undefined {
    return this.directoryLevelAggregates;
  }

  /**
   * Insert/merge additional directory-level aggregates.
   *
   * @param nextDirectoryLevelAggregates - Additional aggregates to merge
   */
  insertDirectoryLevelAggregates(
    nextDirectoryLevelAggregates: Partial<DirectoryLevelAggregates>
  ): void {
    // initializeDirectoryLevelAggregates() is always called before this in subclasses
    this.directoryLevelAggregates = {
      ...this.directoryLevelAggregates!,
      ...nextDirectoryLevelAggregates,
    };
  }

  /**
   * Initialize file-level aggregates.
   *
   * @param filePath - Path to the file being processed
   * @param _additionalArgs - Additional arguments for subclasses (optional)
   */
  initializeFileLevelAggregates(filePath: string, ..._additionalArgs: unknown[]): void {
    this.fileLevelAggregates = {
      file_path: filePath,
      fields_count: new StatsByLabel('processed'),
      field_label_wise_aggregates: {},
    };
  }

  /**
   * Get file-level aggregates.
   *
   * @returns File-level aggregates
   */
  getFileLevelAggregates(): FileLevelAggregates | undefined {
    return this.fileLevelAggregates;
  }

  /**
   * Insert/merge additional file-level aggregates.
   *
   * @param nextFileLevelAggregates - Additional aggregates to merge
   */
  insertFileLevelAggregates(
    nextFileLevelAggregates: Partial<FileLevelAggregates>
  ): void {
    // initializeFileLevelAggregates() is always called before this
    this.fileLevelAggregates = {
      ...this.fileLevelAggregates!,
      ...nextFileLevelAggregates,
    };
  }

  /**
   * Update aggregates when a file has been processed.
   *
   * @param filePath - Path to the processed file
   * @param _additionalArgs - Additional arguments for subclasses (optional)
   */
  updateAggregatesOnProcessedFile(filePath: string, ..._additionalArgs: unknown[]): void {
    // initializeDirectoryLevelAggregates() and initializeFileLevelAggregates() are always called before this
    this.directoryLevelAggregates!.file_wise_aggregates[filePath] =
      this.fileLevelAggregates!;
    this.directoryLevelAggregates!.files_count.push('processed');
  }

  /**
   * Initialize field-level aggregates.
   *
   * @param field - Field being processed
   */
  initializeFieldLevelAggregates(field: Field): void {
    this.fieldLevelAggregates = {
      field,
    };
  }

  /**
   * Get field-level aggregates.
   *
   * @returns Field-level aggregates
   */
  getFieldLevelAggregates(): FieldLevelAggregates | undefined {
    return this.fieldLevelAggregates;
  }

  /**
   * Insert/merge additional field-level aggregates.
   *
   * @param nextFieldLevelAggregates - Additional aggregates to merge
   */
  insertFieldLevelAggregates(
    nextFieldLevelAggregates: Partial<FieldLevelAggregates>
  ): void {
    // initializeFieldLevelAggregates() is always called before this
    this.fieldLevelAggregates = {
      ...this.fieldLevelAggregates!,
      ...nextFieldLevelAggregates,
    };
  }

  /**
   * Update field-level aggregates when a field has been processed.
   *
   * To be called by child classes as per consumer needs.
   *
   * @param field - Field that was processed
   */
  updateFieldLevelAggregatesOnProcessedField(_field: Field): void {
    // Default implementation does nothing
    // Child classes should override as needed
  }

  /**
   * Update file-level aggregates when a field has been processed.
   *
   * To be called by child classes as per consumer needs.
   *
   * @param field - Field that was processed
   * @param fieldLevelAggregates - Field-level aggregates for this field
   */
  updateFileLevelAggregatesOnProcessedField(
    field: Field,
    fieldLevelAggregates: FieldLevelAggregates
  ): void {
    // initializeFileLevelAggregates() is always called before field processing
    const fieldLabel = field.fieldLabel;
    // TODO: convert into field_id_wise_aggregates
    this.fileLevelAggregates!.field_label_wise_aggregates[fieldLabel] =
      fieldLevelAggregates;

    this.fileLevelAggregates!.fields_count.push('processed');
  }

  /**
   * Update directory-level aggregates when a field has been processed.
   *
   * To be called by child classes as per consumer needs.
   *
   * @param _field - Field that was processed
   * @param _fieldLevelAggregates - Field-level aggregates for this field
   */
  updateDirectoryLevelAggregatesOnProcessedField(
    _field: Field,
    _fieldLevelAggregates: FieldLevelAggregates
  ): void {
    // Default implementation does nothing
    // Child classes should override as needed
  }
}

