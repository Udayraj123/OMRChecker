/**
 * File runner base classes.
 *
 * TypeScript port of src/processors/detection/base/file_runner.py
 * Coordinates detection and interpretation passes at file level.
 */

import cv from '../../../utils/opencv';
import { FilePassAggregates, type DirectoryLevelAggregates, type FileLevelAggregates, type FieldLevelAggregates } from './commonPass';
import { FieldTypeDetectionPass } from './detectionPass';
import { FieldTypeInterpretationPass } from './interpretationPass';
import type { Field } from '../../layout/field/base';
import type { FieldDetection } from './detection';
import type { FieldInterpretation } from './interpretation';
import type { TuningConfig } from './commonPass';

/**
 * Generic file-level runner coordinating detection and interpretation passes.
 *
 * Manages aggregates at different levels (file, directory) for both
 * detection and interpretation processes.
 */
export class FileLevelRunner<
  DetectionPassT extends FilePassAggregates,
  InterpretationPassT extends FilePassAggregates
> {
  protected tuningConfig: TuningConfig;
  protected detectionPass: DetectionPassT;
  protected interpretationPass: InterpretationPassT;

  constructor(
    tuningConfig: TuningConfig,
    detectionPass: DetectionPassT,
    interpretationPass: InterpretationPassT
  ) {
    this.tuningConfig = tuningConfig;
    this.detectionPass = detectionPass;
    this.interpretationPass = interpretationPass;
  }

  /**
   * Initialize directory-level aggregates for both detection and interpretation.
   *
   * @param initialDirectoryPath - Path to the initial directory
   */
  initializeDirectoryLevelAggregates(initialDirectoryPath: string): void {
    this.detectionPass.initializeDirectoryLevelAggregates(initialDirectoryPath);
    this.interpretationPass.initializeDirectoryLevelAggregates(initialDirectoryPath);
  }

  // Detection: Field Level
  /**
   * Get field-level detection aggregates.
   *
   * @returns Field-level detection aggregates
   */
  getFieldLevelDetectionAggregates(): FieldLevelAggregates | undefined {
    return this.detectionPass.getFieldLevelAggregates();
  }

  // Detection: File Level
  /**
   * Initialize file-level detection aggregates.
   *
   * @param filePath - Path to the file being processed
   */
  initializeFileLevelDetectionAggregates(filePath: string): void {
    this.detectionPass.initializeFileLevelAggregates(filePath);
  }

  /**
   * Get file-level detection aggregates.
   *
   * @returns File-level detection aggregates
   */
  getFileLevelDetectionAggregates(): FileLevelAggregates | undefined {
    return this.detectionPass.getFileLevelAggregates();
  }

  /**
   * Update detection aggregates when a file has been processed.
   *
   * @param filePath - Path to the processed file
   */
  updateDetectionAggregatesOnProcessedFile(filePath: string): void {
    this.detectionPass.updateAggregatesOnProcessedFile(filePath);
  }

  // Detection: Directory Level
  /**
   * Get directory-level detection aggregates.
   *
   * @returns Directory-level detection aggregates
   */
  getDirectoryLevelDetectionAggregates(): DirectoryLevelAggregates | undefined {
    return this.detectionPass.getDirectoryLevelAggregates();
  }

  // Interpretation: Field Level
  /**
   * Get field-level interpretation aggregates.
   *
   * @returns Field-level interpretation aggregates
   */
  getFieldLevelInterpretationAggregates(): FieldLevelAggregates | undefined {
    return this.interpretationPass.getFieldLevelAggregates();
  }

  // Interpretation: File Level
  /**
   * Initialize file-level interpretation aggregates.
   *
   * @param filePath - Path to the file being processed
   */
  initializeFileLevelInterpretationAggregates(filePath: string): void {
    this.interpretationPass.initializeFileLevelAggregates(filePath);
  }

  /**
   * Get file-level interpretation aggregates.
   *
   * @returns File-level interpretation aggregates
   */
  getFileLevelInterpretationAggregates(): FileLevelAggregates | undefined {
    return this.interpretationPass.getFileLevelAggregates();
  }

  /**
   * Update interpretation aggregates when a file has been processed.
   *
   * @param filePath - Path to the processed file
   */
  updateInterpretationAggregatesOnProcessedFile(filePath: string): void {
    this.interpretationPass.updateAggregatesOnProcessedFile(filePath);
  }

  // Interpretation: Directory Level
  /**
   * Get directory-level interpretation aggregates.
   *
   * @returns Directory-level interpretation aggregates
   */
  getDirectoryLevelInterpretationAggregates(): DirectoryLevelAggregates | undefined {
    return this.interpretationPass.getDirectoryLevelAggregates();
  }
}

/**
 * Field type file-level runner.
 *
 * Specializes FileLevelRunner for specific field types.
 * Handles detection and interpretation for specific fields.
 *
 * It contains the external contract to be used by TemplateFileRunner
 * for each of the field_detection_types.
 * It is static per template instance. Instantiated once per field type
 * from the template.json
 */
export class FieldTypeFileLevelRunner extends FileLevelRunner<
  FieldTypeDetectionPass,
  FieldTypeInterpretationPass
> {
  public fieldDetectionType: string;

  constructor(
    tuningConfig: TuningConfig,
    fieldDetectionType: string,
    detectionPass: FieldTypeDetectionPass,
    interpretationPass: FieldTypeInterpretationPass
  ) {
    super(tuningConfig, detectionPass, interpretationPass);
    this.fieldDetectionType = fieldDetectionType;
  }

  /**
   * Run field-level detection.
   *
   * @param field - Field to detect
   * @param grayImage - Grayscale image
   * @param coloredImage - Colored image (optional)
   * @returns FieldDetection result
   */
  runFieldLevelDetection(
    field: Field,
    grayImage: cv.Mat,
    coloredImage?: cv.Mat
  ): FieldDetection {
    // initializeFieldLevelAggregates is now called automatically by runFieldLevelDetection
    return this.detectionPass.runFieldLevelDetection(
      field,
      grayImage,
      coloredImage
    );
  }

  /**
   * Run field-level interpretation.
   *
   * @param field - Field to interpret
   * @returns FieldInterpretation result
   */
  runFieldLevelInterpretation(field: Field): FieldInterpretation {
    // initializeFieldLevelAggregates is now called automatically by runFieldLevelInterpretation
    const fileLevelDetectionAggregates =
      this.detectionPass.getFileLevelAggregates();
    return this.interpretationPass.runFieldLevelInterpretation(
      field,
      fileLevelDetectionAggregates
    );
  }
}

