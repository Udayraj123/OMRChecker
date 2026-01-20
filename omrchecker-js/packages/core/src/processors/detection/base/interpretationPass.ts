/**
 * Interpretation pass base classes.
 *
 * TypeScript port of src/processors/detection/base/interpretation_pass.py
 * Defines abstract base classes for field type interpretation passes and template-level interpretation passes.
 */

import { StatsByLabel } from '../../../utils/stats';
import {
  FilePassAggregates,
  type FieldLevelAggregates,
  type TuningConfig,
} from './commonPass';
import type { Field } from '../../layout/field/base';
import type { FieldInterpretation } from './interpretation';

/**
 * Abstract base class for field type interpretation passes.
 *
 * FieldTypeInterpretationPass implements interpretation pass for specific field types,
 * managing the interpretation-related aggregates.
 *
 * It is responsible for executing interpretation logic on detection results,
 * determining the actual field values from detection data.
 */
export abstract class FieldTypeInterpretationPass extends FilePassAggregates {
  public fieldDetectionType: string;

  constructor(tuningConfig: TuningConfig, fieldDetectionType: string) {
    super(tuningConfig);
    this.fieldDetectionType = fieldDetectionType;
  }

  /**
   * Abstract method to get field interpretation.
   * Must be implemented by subclasses.
   *
   * @param field - Field to interpret
   * @param fileLevelDetectionAggregates - File-level detection aggregates
   * @param fileLevelInterpretationAggregates - File-level interpretation aggregates
   * @returns FieldInterpretation result
   */
  abstract getFieldInterpretation(
    field: Field,
    fileLevelDetectionAggregates: unknown,
    fileLevelInterpretationAggregates: unknown
  ): FieldInterpretation;

  /**
   * Initialize field-level aggregates for interpretation.
   *
   * @param field - Field being processed
   */
  initializeFieldLevelAggregates(field: Field): void {
    super.initializeFieldLevelAggregates(field);
    this.insertFieldLevelAggregates({
      field_level_confidence_metrics: {},
    });
  }

  /**
   * Initialize file-level aggregates for interpretation.
   *
   * @param filePath - Path to the file being processed
   */
  initializeFileLevelAggregates(filePath: string): void {
    super.initializeFileLevelAggregates(filePath);
    this.insertFileLevelAggregates({
      confidence_metrics_for_file: {},
      field_id_to_interpretation: {},
    });
  }

  /**
   * Run field-level interpretation.
   *
   * @param field - Field to interpret
   * @param fileLevelDetectionAggregates - File-level detection aggregates
   * @returns FieldInterpretation result
   */
  runFieldLevelInterpretation(
    field: Field,
    fileLevelDetectionAggregates: unknown
  ): FieldInterpretation {
    // Initialize field-level aggregates automatically
    this.initializeFieldLevelAggregates(field);
    const fileLevelInterpretationAggregates = this.getFileLevelAggregates();
    const fieldInterpretation = this.getFieldInterpretation(
      field,
      fileLevelDetectionAggregates,
      fileLevelInterpretationAggregates
    );

    // update_aggregates_on_processed_field_interpretation
    this.updateAggregatesOnProcessedFieldInterpretation(
      field,
      fieldInterpretation
    );

    return fieldInterpretation;
  }

  /**
   * Update all aggregates when a field interpretation has been processed.
   *
   * @param field - Field that was processed
   * @param fieldInterpretation - Interpretation result
   */
  updateAggregatesOnProcessedFieldInterpretation(
    field: Field,
    fieldInterpretation: FieldInterpretation
  ): void {
    this.updateFieldLevelAggregatesOnProcessedFieldInterpretation(
      field,
      fieldInterpretation
    );

    const fieldLevelAggregates = this.getFieldLevelAggregates();
    if (!fieldLevelAggregates) {
      throw new Error('Field level aggregates not initialized');
    }

    this.updateFileLevelAggregatesOnProcessedFieldInterpretation(
      field,
      fieldInterpretation,
      fieldLevelAggregates
    );

    this.updateDirectoryLevelAggregatesOnProcessedFieldInterpretation(
      field,
      fieldInterpretation,
      fieldLevelAggregates
    );
  }

  /**
   * Update field-level aggregates when a field interpretation has been processed.
   *
   * @param field - Field that was processed
   * @param fieldInterpretation - Interpretation result
   */
  updateFieldLevelAggregatesOnProcessedFieldInterpretation(
    field: Field,
    fieldInterpretation: FieldInterpretation
  ): void {
    this.insertFieldLevelAggregates({
      field_level_confidence_metrics: fieldInterpretation.getFieldLevelConfidenceMetrics(),
    });
    this.updateFieldLevelAggregatesOnProcessedField(field);
  }

  /**
   * Update file-level aggregates when a field interpretation has been processed.
   *
   * @param field - Field that was processed
   * @param fieldInterpretation - Interpretation result
   * @param fieldLevelAggregates - Field-level aggregates
   */
  updateFileLevelAggregatesOnProcessedFieldInterpretation(
    field: Field,
    fieldInterpretation: FieldInterpretation,
    fieldLevelAggregates: FieldLevelAggregates
  ): void {
    const fileAgg = this.getFileLevelAggregates();
    if (!fileAgg) {
      throw new Error('File level aggregates not initialized');
    }

    const confidenceMetricsForFile = fileAgg.confidence_metrics_for_file as Record<
      string,
      unknown
    >;
    const fieldIdToInterpretation = fileAgg.field_id_to_interpretation as Record<
      string,
      unknown
    >;

    confidenceMetricsForFile[field.fieldLabel] =
      fieldInterpretation.getFieldLevelConfidenceMetrics();
    fieldIdToInterpretation[field.id] = fieldInterpretation;

    this.updateFileLevelAggregatesOnProcessedField(field, fieldLevelAggregates);
  }

  /**
   * Update directory-level aggregates when a field interpretation has been processed.
   *
   * @param field - Field that was processed
   * @param _fieldInterpretation - Interpretation result (not used in base implementation)
   * @param fieldLevelAggregates - Field-level aggregates
   */
  updateDirectoryLevelAggregatesOnProcessedFieldInterpretation(
    field: Field,
    _fieldInterpretation: FieldInterpretation,
    fieldLevelAggregates: FieldLevelAggregates
  ): void {
    this.updateDirectoryLevelAggregatesOnProcessedField(field, fieldLevelAggregates);
  }
}

/**
 * Template-level interpretation pass.
 *
 * Manages interpretation aggregates at the template level, coordinating
 * multiple field detection type interpretation passes.
 */
export class TemplateInterpretationPass extends FilePassAggregates {
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
      confidence_metrics_for_file: {},
      field_detection_type_wise_aggregates: Object.fromEntries(
        allFieldDetectionTypes.map((key) => [
          key,
          { fields_count: new StatsByLabel('processed') },
        ])
      ),
    });
  }

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
      field_detection_type_wise_aggregates: Object.fromEntries(
        allFieldDetectionTypes.map((key) => [
          key,
          { fields_count: new StatsByLabel('processed') },
        ])
      ),
    });
  }

  /**
   * Run field-level interpretation.
   *
   * @param field - Field to interpret
   * @param fieldInterpretation - Interpretation result
   * @param _fieldTypeRunnerFieldLevelAggregates - Field-level aggregates from field type runner (not used)
   * @param currentOmrResponse - Current OMR response (optional)
   */
  runFieldLevelInterpretation(
    field: Field,
    fieldInterpretation: FieldInterpretation,
    _fieldTypeRunnerFieldLevelAggregates: unknown,
    currentOmrResponse?: Record<string, unknown>
  ): void {
    // Initialize field-level aggregates automatically
    this.initializeFieldLevelAggregates(field);
    // Update aggregates with interpretation result
    this.updateAggregatesOnProcessedFieldInterpretation(field, fieldInterpretation);

    // Update OMR response if provided
    if (currentOmrResponse) {
      const interpretationString = fieldInterpretation.getFieldInterpretationString();
      currentOmrResponse[field.fieldLabel] = interpretationString;
    }
  }

  /**
   * Update all aggregates when a field interpretation has been processed.
   *
   * @param field - Field that was processed
   * @param fieldInterpretation - Interpretation result
   */
  updateAggregatesOnProcessedFieldInterpretation(
    field: Field,
    fieldInterpretation: FieldInterpretation
  ): void {
    this.updateFieldLevelAggregatesOnProcessedField(field);

    const fieldLevelAggregates = this.getFieldLevelAggregates();
    if (!fieldLevelAggregates) {
      throw new Error('Field level aggregates not initialized');
    }

    // Store interpretation in file-level aggregates
    const fileAggForInterpretation = this.getFileLevelAggregates();
    if (fileAggForInterpretation) {
      const confidenceMetricsForFile = fileAggForInterpretation.confidence_metrics_for_file as Record<
        string,
        unknown
      >;
      const fieldIdToInterpretation = fileAggForInterpretation.field_id_to_interpretation as Record<
        string,
        unknown
      >;

      confidenceMetricsForFile[field.fieldLabel] =
        fieldInterpretation.getFieldLevelConfidenceMetrics();
      fieldIdToInterpretation[field.id] = fieldInterpretation;
    }

    this.updateFileLevelAggregatesOnProcessedField(field, fieldLevelAggregates);
    this.updateDirectoryLevelAggregatesOnProcessedField(field, fieldLevelAggregates);

    // Update field detection type counts
    const dirAgg = this.getDirectoryLevelAggregates();
    const fileAggForTypeCounts = this.getFileLevelAggregates();

    if (dirAgg && fileAggForTypeCounts) {
      const fieldDetectionType = field.fieldDetectionType;
      const dirTypeAggregates = (
        dirAgg.field_detection_type_wise_aggregates as Record<
          string,
          { fields_count: StatsByLabel }
        >
      )[fieldDetectionType];
      const fileTypeAggregates = (
        fileAggForTypeCounts.field_detection_type_wise_aggregates as Record<
          string,
          { fields_count: StatsByLabel }
        >
      )[fieldDetectionType];

      if (dirTypeAggregates) {
        dirTypeAggregates.fields_count.push('processed');
      }
      if (fileTypeAggregates) {
        fileTypeAggregates.fields_count.push('processed');
      }
    }
  }

  /**
   * Update aggregates when a file has been processed.
   *
   * @param filePath - Path to the processed file
   * @param fieldDetectionTypeFileRunners - Map of field detection type to file runners
   */
  updateAggregatesOnProcessedFile(
    filePath: string,
    fieldDetectionTypeFileRunners?: Record<string, { getFileLevelInterpretationAggregates(): unknown }>
  ): void {
    super.updateAggregatesOnProcessedFile(filePath);

    if (!fieldDetectionTypeFileRunners) {
      return;
    }

    const fileAgg = this.getFileLevelAggregates();
    if (!fileAgg) {
      throw new Error('File level aggregates not initialized');
    }

    const fieldDetectionTypeWiseAggregates = fileAgg.field_detection_type_wise_aggregates as Record<
      string,
      unknown
    >;

    for (const fieldDetectionTypeFileRunner of Object.values(fieldDetectionTypeFileRunners)) {
      const fieldDetectionType = (fieldDetectionTypeFileRunner as {
        fieldDetectionType: string;
        getFileLevelInterpretationAggregates(): unknown;
      }).fieldDetectionType;
      fieldDetectionTypeWiseAggregates[fieldDetectionType] =
        fieldDetectionTypeFileRunner.getFileLevelInterpretationAggregates();
    }
  }
}

