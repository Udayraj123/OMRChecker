/**
 * Detection pass base classes.
 *
 * TypeScript port of src/processors/detection/base/detection_pass.py
 * Defines abstract base classes for field type detection passes and template-level detection passes.
 */

import type { Field } from '../../layout/field/base';
import { StatsByLabel } from '../../../utils/stats';
import {
  FilePassAggregates,
  type FieldLevelAggregates,
  type TuningConfig,
} from './commonPass';
import type { FieldDetection } from './detection';
import { DetectionRepository } from '../../repositories/DetectionRepository';
import {
  BubbleFieldDetectionResult,
  OCRFieldDetectionResult,
  BarcodeFieldDetectionResult,
} from '../models';
import { Logger } from '../../../utils/logger';

/**
 * Abstract base class for field type detection passes.
 *
 * FieldTypeDetectionPass implements detection pass for specific field types,
 * managing the detection-related aggregates.
 *
 * It is responsible for executing detection logic on the image from the field information.
 * It does not determine the actual field values, that is left to the interpretation pass
 * which can make use of aggregate data collected during the detection pass.
 */
export abstract class FieldTypeDetectionPass extends FilePassAggregates {
  public fieldDetectionType: string;

  constructor(tuningConfig: TuningConfig, fieldDetectionType: string) {
    // Call super() first (proper initialization order)
    super(tuningConfig);
    // Set subclass-specific properties after super()
    this.fieldDetectionType = fieldDetectionType;
  }

  /**
   * Abstract method to get field detection.
   * Must be implemented by subclasses.
   *
   * @param field - Field to detect
   * @param grayImage - Grayscale image
   * @param coloredImage - Colored image (optional)
   * @returns FieldDetection result
   */
  abstract getFieldDetection(
    field: Field,
    grayImage: cv.Mat,
    coloredImage?: cv.Mat
  ): FieldDetection;

  /**
   * Run field-level detection with automatic initialization of aggregates.
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
    this.initializeFieldLevelAggregates(field);
    const fieldDetection = this.getFieldDetection(field, grayImage, coloredImage);
    this.updateAggregatesOnProcessedFieldDetection(field, fieldDetection);
    return fieldDetection;
  }

  /**
   * Update all aggregates when a field detection has been processed.
   *
   * @param field - Field that was processed
   * @param fieldDetection - Detection result
   */
  updateAggregatesOnProcessedFieldDetection(
    field: Field,
    fieldDetection: FieldDetection
  ): void {
    this.updateFieldLevelAggregatesOnProcessedFieldDetection(field, fieldDetection);

    const fieldLevelAggregates = this.getFieldLevelAggregates()!;

    this.updateFileLevelAggregatesOnProcessedFieldDetection(
      field,
      fieldDetection,
      fieldLevelAggregates
    );

    this.updateDirectoryLevelAggregatesOnProcessedFieldDetection(
      field,
      fieldDetection,
      fieldLevelAggregates
    );
  }

  /**
   * Update field-level aggregates when a field detection has been processed.
   *
   * @param field - Field that was processed
   * @param _fieldDetection - Detection result (not used in base implementation)
   */
  updateFieldLevelAggregatesOnProcessedFieldDetection(
    field: Field,
    _fieldDetection: FieldDetection
  ): void {
    this.updateFieldLevelAggregatesOnProcessedField(field);
  }

  /**
   * Update file-level aggregates when a field detection has been processed.
   *
   * @param field - Field that was processed
   * @param _fieldDetection - Detection result (not used in base implementation)
   * @param fieldLevelAggregates - Field-level aggregates
   */
  updateFileLevelAggregatesOnProcessedFieldDetection(
    field: Field,
    _fieldDetection: FieldDetection,
    fieldLevelAggregates: FieldLevelAggregates
  ): void {
    this.updateFileLevelAggregatesOnProcessedField(field, fieldLevelAggregates);
    // TODO: detection confidence metrics?
  }

  /**
   * Update directory-level aggregates when a field detection has been processed.
   *
   * @param field - Field that was processed
   * @param _fieldDetection - Detection result (not used in base implementation)
   * @param fieldLevelAggregates - Field-level aggregates
   */
  updateDirectoryLevelAggregatesOnProcessedFieldDetection(
    field: Field,
    _fieldDetection: FieldDetection,
    fieldLevelAggregates: FieldLevelAggregates
  ): void {
    this.updateDirectoryLevelAggregatesOnProcessedField(field, fieldLevelAggregates);
    // TODO: (if needed) update_directory_level_aggregates_on_processed_field
  }
}

/**
 * Template-level detection pass.
 *
 * Manages detection aggregates at the template level, coordinating
 * multiple field detection type passes.
 */
export class TemplateDetectionPass extends FilePassAggregates {
  /**
   * Initialize directory-level aggregates with field detection type tracking.
   *
   * @param initialDirectoryPath - Path to the initial directory
   * @param allFieldDetectionTypes - All field detection types in the template
   */
  initializeDirectoryLevelAggregates(
    initialDirectoryPath: string,
    allFieldDetectionTypes: string[]
  ): void {
    super.initializeDirectoryLevelAggregates(initialDirectoryPath);
    this.insertDirectoryLevelAggregates({
      files_by_label_count: new StatsByLabel('processed', 'multi_marked'),
      field_detection_type_wise_aggregates: Object.fromEntries(
        allFieldDetectionTypes.map((key) => [
          key,
          { fields_count: new StatsByLabel('processed') },
        ])
      ),
    });
  }

  /**
   * Initialize file-level aggregates with field detection type tracking.
   *
   * @param filePath - Path to the file being processed
   * @param allFieldDetectionTypes - All field detection types in the template
   */
  initializeFileLevelAggregates(
    filePath: string,
    allFieldDetectionTypes: string[]
  ): void {
    super.initializeFileLevelAggregates(filePath);
    this.insertFileLevelAggregates({
      field_detection_type_wise_aggregates: Object.fromEntries(
        allFieldDetectionTypes.map((key) => [
          key,
          { fields_count: new StatsByLabel('processed') },
        ])
      ),
    });
  }

  /**
   * Update all aggregates when a field detection has been processed.
   *
   * @param field - Field that was processed
   * @param fieldDetection - Detection result
   */
  updateAggregatesOnProcessedFieldDetection(
    field: Field,
    fieldDetection: FieldDetection
  ): void {
    this.updateFieldLevelAggregatesOnProcessedFieldDetection(field, fieldDetection);

    // TODO: can we move this getter to parent class?
    const fieldLevelAggregates = this.getFieldLevelAggregates()!;

    this.updateFileLevelAggregatesOnProcessedFieldDetection(
      field,
      fieldDetection,
      fieldLevelAggregates
    );
    this.updateDirectoryLevelAggregatesOnProcessedFieldDetection(
      field,
      fieldDetection,
      fieldLevelAggregates
    );
  }

  /**
   * Update field-level aggregates when a field detection has been processed.
   *
   * @param field - Field that was processed
   * @param _fieldDetection - Detection result (not used)
   */
  updateFieldLevelAggregatesOnProcessedFieldDetection(
    field: Field,
    _fieldDetection: FieldDetection
  ): void {
    this.updateFieldLevelAggregatesOnProcessedField(field);
  }

  /**
   * Update file-level aggregates when a field detection has been processed.
   *
   * @param field - Field that was processed
   * @param _fieldDetection - Detection result (not used)
   * @param fieldLevelAggregates - Field-level aggregates
   */
  updateFileLevelAggregatesOnProcessedFieldDetection(
    field: Field,
    _fieldDetection: FieldDetection,
    fieldLevelAggregates: FieldLevelAggregates
  ): void {
    this.updateFileLevelAggregatesOnProcessedField(field, fieldLevelAggregates);
    // TODO: detection confidence metrics?
  }

  /**
   * Update directory-level aggregates when a field detection has been processed.
   *
   * @param field - Field that was processed
   * @param _fieldDetection - Detection result (not used)
   * @param fieldLevelAggregates - Field-level aggregates
   */
  updateDirectoryLevelAggregatesOnProcessedFieldDetection(
    field: Field,
    _fieldDetection: FieldDetection,
    fieldLevelAggregates: FieldLevelAggregates
  ): void {
    this.updateDirectoryLevelAggregatesOnProcessedField(field, fieldLevelAggregates);
    const fieldDetectionType = field.fieldDetectionType;

    const directoryLevelAggregates = this.getDirectoryLevelAggregates()!;
    const fieldDetectionTypeWiseAggregates = (
      directoryLevelAggregates.field_detection_type_wise_aggregates as Record<
        string,
        { fields_count: StatsByLabel }
      >
    )[fieldDetectionType];
    // Update the processed field count for that runner
    fieldDetectionTypeWiseAggregates.fields_count.push('processed');
  }

  /**
   * Run field-level detection with automatic initialization of aggregates.
   *
   * Note: This method expects the field_detection to already be computed
   * by the field type runner. It initializes aggregates and updates them.
   *
   * @param field - Field that was processed
   * @param fieldDetection - Detection result (already computed)
   */
  runFieldLevelDetection(field: Field, fieldDetection: FieldDetection): void {
    this.initializeFieldLevelAggregates(field);
    this.updateAggregatesOnProcessedFieldDetection(field, fieldDetection);
  }

  /**
   * Update aggregates when a file has been processed.
   * Overrides base method to handle field detection type file runners.
   *
   * @param filePath - Path to the processed file
   * @param fieldDetectionTypeFileRunners - Map of field detection type to file runners
   */
  updateAggregatesOnProcessedFile(
    filePath: string,
    fieldDetectionTypeFileRunners?: Record<string, { getFileLevelDetectionAggregates(): unknown }>
  ): void {
    super.updateAggregatesOnProcessedFile(filePath);

    if (!fieldDetectionTypeFileRunners) {
      return;
    }

    const fileLevelAggregates = this.getFileLevelAggregates()!;
    const fieldDetectionTypeWiseAggregates = fileLevelAggregates.field_detection_type_wise_aggregates as Record<
      string,
      unknown
    >;

    for (const fieldDetectionTypeFileRunner of Object.values(fieldDetectionTypeFileRunners)) {
      const fieldDetectionType = (fieldDetectionTypeFileRunner as {
        fieldDetectionType: string;
        getFileLevelDetectionAggregates(): unknown;
      }).fieldDetectionType;
      fieldDetectionTypeWiseAggregates[fieldDetectionType] =
        fieldDetectionTypeFileRunner.getFileLevelDetectionAggregates();
    }

    // Populate typed field results from repository
    // Get repository from any field type runner (they all share the same repository)
    const anyRunner = Object.values(fieldDetectionTypeFileRunners)[0] as {
      repository?: DetectionRepository;
    };
    if (anyRunner?.repository) {
      try {
        const fileResults = anyRunner.repository.getFileResults(filePath);

        // Map all field types by field_label for interpretation access
        const bubbleFieldsByLabel: Record<string, BubbleFieldDetectionResult> = {};
        for (const result of fileResults.bubbleFields.values()) {
          bubbleFieldsByLabel[result.fieldLabel] = result;
        }

        const ocrFieldsByLabel: Record<string, OCRFieldDetectionResult> = {};
        for (const result of fileResults.ocrFields.values()) {
          ocrFieldsByLabel[result.fieldLabel] = result;
        }

        const barcodeFieldsByLabel: Record<string, BarcodeFieldDetectionResult> = {};
        for (const result of fileResults.barcodeFields.values()) {
          barcodeFieldsByLabel[result.fieldLabel] = result;
        }

        const fileLevelAggregates = this.getFileLevelAggregates()!;
        fileLevelAggregates.bubble_fields = bubbleFieldsByLabel;
        fileLevelAggregates.ocr_fields = ocrFieldsByLabel;
        fileLevelAggregates.barcode_fields = barcodeFieldsByLabel;
      } catch (error) {
        // Check if it's a KeyError-equivalent (file not found in repository)
        if (error instanceof Error && error.message.includes('No results found for file')) {
          // File not yet finalized in repository - this should not happen
          // but if it does, log and re-raise to surface the issue
          const logger = new Logger('TemplateDetectionPass');
          logger.error(
            `File ${filePath} not found in repository after finalize_file(). ` +
              `This indicates a bug in the repository lifecycle management. Error: ${error.message}`
          );
          // Re-raise to surface the issue during development
          throw error;
        } else {
          // Other unexpected errors
          const logger = new Logger('TemplateDetectionPass');
          logger.error(
            `Unexpected error populating field results from repository for ${filePath}: ${error}`
          );
          // Initialize empty dicts to prevent KeyError in interpretation
          const fileLevelAggregates = this.getFileLevelAggregates()!;
          if (!fileLevelAggregates.bubble_fields) {
            fileLevelAggregates.bubble_fields = {};
          }
          if (!fileLevelAggregates.ocr_fields) {
            fileLevelAggregates.ocr_fields = {};
          }
          if (!fileLevelAggregates.barcode_fields) {
            fileLevelAggregates.barcode_fields = {};
          }
        }
      }
    }
  }
}

