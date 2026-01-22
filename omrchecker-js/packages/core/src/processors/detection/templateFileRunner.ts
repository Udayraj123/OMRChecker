/**
 * Template file runner.
 *
 * TypeScript port of src/processors/detection/template_file_runner.py
 * Coordinates detection and interpretation at template level.
 */

import cv from '../../utils/opencv';
import { FieldDetectionType } from '../constants';
import { FileLevelRunner } from './base/fileRunner';
import { TemplateDetectionPass } from './base/detectionPass';
import { TemplateInterpretationPass } from './base/interpretationPass';
import { FieldTypeFileLevelRunner } from './base/fileRunner';
import { BubblesThresholdFileRunner } from './bubbles_threshold/fileRunner';
import type { Field } from '../layout/field/base';
import type { TuningConfig } from './base/commonPass';
import type { TemplateLayoutData } from '../../template/TemplateLoader';
import { DetectionRepository } from '../repositories/DetectionRepository';

/**
 * Mapping of field detection types to their file runner classes.
 */
const fieldDetectionTypeToRunner: Record<
  string,
  new (tuningConfig: TuningConfig, repository: DetectionRepository) => FieldTypeFileLevelRunner
> = {
  [FieldDetectionType.BUBBLES_THRESHOLD]: BubblesThresholdFileRunner,
  // TODO: Add OCR and Barcode runners when implemented
  // [FieldDetectionType.OCR]: OCRFileRunner,
  // [FieldDetectionType.BARCODE_QR]: BarcodeFileRunner,
};

/**
 * Template file runner.
 *
 * Responsible for running the file level detection and interpretation steps.
 * Maintains own template level runners as well as all the field detection type level runners.
 * One instance of TemplateFileRunner per Template - thus it is reused for all images
 * mapped to that Template.
 * Note: a Template may get reused for multiple directories (in nested case).
 */
export class TemplateFileRunner extends FileLevelRunner<
  TemplateDetectionPass,
  TemplateInterpretationPass
> {
  public template: TemplateLayoutData;
  public allFields: Field[] = [];
  public allFieldDetectionTypes: string[] = [];
  protected fieldDetectionTypeFileRunners: Record<
    string,
    FieldTypeFileLevelRunner
  > = {};
  private repository: DetectionRepository;

  constructor(template: TemplateLayoutData, tuningConfig: TuningConfig) {
    const detectionPass = new TemplateDetectionPass(tuningConfig);
    const interpretationPass = new TemplateInterpretationPass(tuningConfig);
    super(tuningConfig, detectionPass, interpretationPass);

    this.template = template;
    this.repository = new DetectionRepository();
    this.initializeFieldFileRunners(template);
    this.initializeDirectoryLevelAggregates(template);
  }

  /**
   * Initialize field file runners for all field detection types in the template.
   *
   * @param template - Template layout
   */
  initializeFieldFileRunners(template: TemplateLayoutData): void {
    this.allFields = template.allFields;
    this.allFieldDetectionTypes = template.allFieldDetectionTypes;

    // Create instances of all required field type processors
    this.fieldDetectionTypeFileRunners = {};
    for (const fieldDetectionType of this.allFieldDetectionTypes) {
      this.fieldDetectionTypeFileRunners[fieldDetectionType] =
        this.getFieldDetectionTypeFileRunner(fieldDetectionType);
    }
  }

  /**
   * Get field detection type file runner.
   *
   * @param fieldDetectionType - Field detection type
   * @returns Field type file runner instance
   */
  getFieldDetectionTypeFileRunner(
    fieldDetectionType: string
  ): FieldTypeFileLevelRunner {
    const FieldTypeProcessorClass = fieldDetectionTypeToRunner[fieldDetectionType];
    if (!FieldTypeProcessorClass) {
      throw new Error(
        `No file runner found for field detection type: ${fieldDetectionType}`
      );
    }
    return new FieldTypeProcessorClass(this.tuningConfig, this.repository);
  }

  /**
   * Read OMR and update metrics.
   *
   * Main entry point for processing an image. Runs detection pass first,
   * then interpretation pass, and returns the OMR response.
   *
   * @param filePath - Path to the file being processed
   * @param grayImage - Grayscale image
   * @param coloredImage - Colored image (optional)
   * @returns OMR response dictionary
   */
  readOmrAndUpdateMetrics(
    filePath: string,
    grayImage: cv.Mat,
    coloredImage?: cv.Mat
  ): Record<string, string> {
    // First pass to compute aggregates like global threshold
    // Populate detections
    this.runFileLevelDetection(filePath, grayImage, coloredImage);

    // Populate interpretations
    return this.runFileLevelInterpretation(filePath, grayImage, coloredImage);
  }

  /**
   * Run file-level detection.
   *
   * Performs detection step for each field and updates aggregates.
   *
   * @param filePath - Path to the file being processed
   * @param grayImage - Grayscale image
   * @param coloredImage - Colored image (optional)
   */
  runFileLevelDetection(
    filePath: string,
    grayImage: cv.Mat,
    coloredImage?: cv.Mat
  ): void {
    this.initializeFileLevelDetectionAggregates(filePath);

    // Perform detection step for each field
    for (const field of this.allFields) {
      this.runFieldLevelDetection(field, grayImage, coloredImage);
    }

    this.updateDetectionAggregatesOnProcessedFile(filePath);
  }

  /**
   * Run field-level detection.
   *
   * @param field - Field to detect
   * @param grayImage - Grayscale image
   * @param coloredImage - Colored image (optional)
   */
  runFieldLevelDetection(
    field: Field,
    grayImage: cv.Mat,
    coloredImage?: cv.Mat
  ): void {
    const fieldDetectionTypeFileRunner =
      this.fieldDetectionTypeFileRunners[field.fieldDetectionType];

    if (!fieldDetectionTypeFileRunner) {
      throw new Error(
        `No file runner found for field detection type: ${field.fieldDetectionType}`
      );
    }

    const fieldDetection = fieldDetectionTypeFileRunner.runFieldLevelDetection(
      field,
      grayImage,
      coloredImage
    );

    // initializeFieldLevelAggregates is now called automatically by runFieldLevelDetection
    this.detectionPass.runFieldLevelDetection(field, fieldDetection);
  }

  /**
   * Run file-level interpretation.
   *
   * Performs interpretation step for each field and returns OMR response.
   *
   * @param filePath - Path to the file being processed
   * @param _grayImage - Grayscale image (not used)
   * @param _coloredImage - Colored image (not used)
   * @returns OMR response dictionary
   */
  runFileLevelInterpretation(
    filePath: string,
    _grayImage: cv.Mat,
    _coloredImage?: cv.Mat
  ): Record<string, string> {
    this.initializeFileLevelInterpretationAggregates(filePath);

    const currentOmrResponse: Record<string, string> = {};

    // Perform interpretation step for each field
    for (const field of this.allFields) {
      // Intentional arg mutation
      this.runFieldLevelInterpretation(field, currentOmrResponse);
    }

    this.updateInterpretationAggregatesOnProcessedFile(filePath);

    return currentOmrResponse;
  }

  /**
   * Run field-level interpretation.
   *
   * @param field - Field to interpret
   * @param currentOmrResponse - Current OMR response (mutated)
   */
  runFieldLevelInterpretation(
    field: Field,
    currentOmrResponse: Record<string, string>
  ): void {
    const fieldDetectionTypeFileRunner =
      this.fieldDetectionTypeFileRunners[field.fieldDetectionType];

    if (!fieldDetectionTypeFileRunner) {
      throw new Error(
        `No file runner found for field detection type: ${field.fieldDetectionType}`
      );
    }

    // Run field-level interpretation with template-level aggregates
    // initializeFieldLevelAggregates is called automatically inside runFieldLevelInterpretation
    // The method internally gets file-level detection aggregates from its own detection pass
    const fieldInterpretation =
      fieldDetectionTypeFileRunner.runFieldLevelInterpretation(field);

    const fieldTypeRunnerFieldLevelAggregates =
      fieldDetectionTypeFileRunner.getFieldLevelInterpretationAggregates();
    // initializeFieldLevelAggregates is called automatically inside runFieldLevelInterpretation
    this.interpretationPass.runFieldLevelInterpretation(
      field,
      fieldInterpretation,
      fieldTypeRunnerFieldLevelAggregates,
      currentOmrResponse
    );

    currentOmrResponse[field.fieldLabel] =
      fieldInterpretation.getFieldInterpretationString();
  }

  /**
   * Initialize directory-level aggregates.
   *
   * Overrides parent to handle template-specific initialization.
   * Note: This method signature differs from the base class to accept TemplateLayout.
   *
   * @param templateOrPath - Template layout or directory path
   */
  initializeDirectoryLevelAggregates(
    templateOrPath: TemplateLayoutData | string
  ): void {
    // For browser environment, we don't have a file path, so use empty string
    const initialDirectoryPath =
      typeof templateOrPath === 'string' ? templateOrPath : '';

    this.repository.initializeDirectory(initialDirectoryPath);
    this.detectionPass.initializeDirectoryLevelAggregates(
      initialDirectoryPath,
      this.allFieldDetectionTypes
    );
    this.interpretationPass.initializeDirectoryLevelAggregates(
      initialDirectoryPath,
      this.allFieldDetectionTypes
    );

    for (const fieldDetectionTypeFileRunner of Object.values(
      this.fieldDetectionTypeFileRunners
    )) {
      fieldDetectionTypeFileRunner.initializeDirectoryLevelAggregates(
        initialDirectoryPath
      );
    }
  }

  /**
   * Initialize file-level detection aggregates.
   *
   * Overrides parent to handle template-specific initialization.
   *
   * @param filePath - Path to the file being processed
   */
  initializeFileLevelDetectionAggregates(filePath: string): void {
    this.detectionPass.initializeFileLevelAggregates(
      filePath,
      this.allFieldDetectionTypes
    );

    // Setup field type wise metrics
    for (const fieldDetectionTypeFileRunner of Object.values(
      this.fieldDetectionTypeFileRunners
    )) {
      fieldDetectionTypeFileRunner.initializeFileLevelDetectionAggregates(
        filePath
      );
    }
  }

  /**
   * Update detection aggregates when a file has been processed.
   *
   * Overrides parent to handle template-specific updates.
   *
   * @param filePath - Path to the processed file
   */
  updateDetectionAggregatesOnProcessedFile(filePath: string): void {
    for (const fieldDetectionTypeFileRunner of Object.values(
      this.fieldDetectionTypeFileRunners
    )) {
      fieldDetectionTypeFileRunner.updateDetectionAggregatesOnProcessedFile(
        filePath
      );
    }

    // Finalize file in repository
    const anyRunner = Object.values(this.fieldDetectionTypeFileRunners)[0] as {
      repository?: DetectionRepository;
    };
    if (anyRunner?.repository) {
      anyRunner.repository.finalizeFile();
    }

    // Note: we update file level after field levels are updated
    this.detectionPass.updateAggregatesOnProcessedFile(
      filePath,
      this.fieldDetectionTypeFileRunners
    );
  }

  /**
   * Initialize file-level interpretation aggregates.
   *
   * Overrides parent to handle template-specific initialization.
   * Interpretation passes now use repository directly, no need to pass aggregates.
   *
   * @param filePath - Path to the file being processed
   */
  initializeFileLevelInterpretationAggregates(filePath: string): void {
    // Interpretation passes now use repository directly, no need to pass aggregates
    this.interpretationPass.initializeFileLevelAggregates(
      filePath,
      this.allFieldDetectionTypes
    );

    // Setup field type wise metrics
    for (const fieldDetectionTypeFileRunner of Object.values(
      this.fieldDetectionTypeFileRunners
    )) {
      fieldDetectionTypeFileRunner.initializeFileLevelInterpretationAggregates(
        filePath
      );
    }
  }

  /**
   * Update interpretation aggregates when a file has been processed.
   *
   * Overrides parent to handle template-specific updates.
   *
   * @param filePath - Path to the processed file
   */
  updateInterpretationAggregatesOnProcessedFile(filePath: string): void {
    for (const fieldDetectionTypeFileRunner of Object.values(
      this.fieldDetectionTypeFileRunners
    )) {
      fieldDetectionTypeFileRunner.updateInterpretationAggregatesOnProcessedFile(
        filePath
      );
    }

    // Note: we update file level after field levels are updated
    this.interpretationPass.updateAggregatesOnProcessedFile(
      filePath,
      this.fieldDetectionTypeFileRunners
    );
  }

  /**
   * Finish processing directory.
   *
   * Called after all files in a directory have been processed.
   * Can be used to export directory-level statistics.
   */
  finishProcessingDirectory(): void {
    // TODO: get_directory_level_confidence_metrics()
    // TODO: update export directory level stats here
  }

  /**
   * Get export OMR metrics for file.
   *
   * @returns OMR metrics (placeholder for now)
   */
  getExportOmrMetricsForFile(): unknown {
    // TODO: Implement export metrics
    return {};
  }
}

