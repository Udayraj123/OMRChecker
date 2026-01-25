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

    // initializeFieldLevelAggregates() is always called before this (line 86)
    const fieldLevelAggregates = this.getFieldLevelAggregates()!;

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
    // initializeFileLevelAggregates() is always called before field processing
    const fileAgg = this.getFileLevelAggregates()!;

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
      field_id_to_interpretation: {},
      files_by_label_count: new StatsByLabel('processed', 'multi_marked'),
      read_response_flags: {
        is_multi_marked: false,
        multi_marked_fields: [],
        is_identifier_multi_marked: false,
      },
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
   * Run field-level interpretation.
   *
   * @param field - Field to interpret
   * @param fieldInterpretation - Interpretation result
   * @param fieldTypeRunnerFieldLevelAggregates - Field-level aggregates from field type runner
   * @param currentOmrResponse - Current OMR response
   */
  runFieldLevelInterpretation(
    field: Field,
    fieldInterpretation: FieldInterpretation,
    fieldTypeRunnerFieldLevelAggregates: Record<string, unknown>,
    currentOmrResponse: Record<string, unknown>
  ): void {
    // Initialize field-level aggregates automatically
    this.initializeFieldLevelAggregates(field);

    // update_aggregates_on_processed_field_interpretation
    // TODO: see if detection also needs this arg (field_type_runner_field_level_aggregates)
    this.updateFieldLevelAggregatesOnProcessedFieldInterpretation(
      currentOmrResponse,
      field,
      fieldInterpretation,
      fieldTypeRunnerFieldLevelAggregates
    );
    const templateFieldLevelAggregates = this.getFieldLevelAggregates()!;

    this.updateFileLevelAggregatesOnProcessedFieldInterpretation(
      field,
      fieldInterpretation,
      templateFieldLevelAggregates
    );
    this.updateDirectoryLevelAggregatesOnProcessedFieldInterpretation(
      field,
      fieldInterpretation,
      templateFieldLevelAggregates
    );
  }

  /**
   * Update field-level aggregates when a field interpretation has been processed.
   *
   * @param _currentOmrResponse - Current OMR response (not used in base implementation)
   * @param field - Field that was processed
   * @param _fieldInterpretation - Interpretation result (not used in base implementation)
   * @param fieldTypeRunnerFieldLevelAggregates - Field-level aggregates from field type runner
   */
  updateFieldLevelAggregatesOnProcessedFieldInterpretation(
    _currentOmrResponse: Record<string, unknown>,
    field: Field,
    _fieldInterpretation: FieldInterpretation,
    fieldTypeRunnerFieldLevelAggregates: Record<string, unknown>
  ): void {
    const fileLevelAggregates = this.getFileLevelAggregates()!;
    const readResponseFlags = fileLevelAggregates.read_response_flags as {
      is_multi_marked: boolean;
      multi_marked_fields: Field[];
      is_identifier_multi_marked: boolean;
    };

    if (fieldTypeRunnerFieldLevelAggregates.is_multi_marked) {
      readResponseFlags.is_multi_marked = true;
      readResponseFlags.multi_marked_fields.push(field);
      // TODO: define identifier_labels
      // if field.is_part_of_identifier():
      // if field_label in self.template.identifier_labels:
      //     readResponseFlags.is_identifier_multi_marked = true;
    }

    // TODO: is there a better way for this?
    this.insertFieldLevelAggregates({
      from_field_type_runner: fieldTypeRunnerFieldLevelAggregates,
    });
    this.updateFieldLevelAggregatesOnProcessedField(field);
    // TODO: support for more validations here?
  }

  /**
   * Update file-level aggregates when a field interpretation has been processed.
   *
   * @param field - Field that was processed
   * @param fieldInterpretation - Interpretation result
   * @param templateFieldLevelAggregates - Template field-level aggregates
   */
  updateFileLevelAggregatesOnProcessedFieldInterpretation(
    field: Field,
    fieldInterpretation: FieldInterpretation,
    templateFieldLevelAggregates: FieldLevelAggregates
  ): void {
    this.updateFileLevelAggregatesOnProcessedField(field, templateFieldLevelAggregates);

    const fileLevelAggregates = this.getFileLevelAggregates()!;
    const fieldIdToInterpretation = fileLevelAggregates.field_id_to_interpretation as Record<
      string,
      unknown
    >;
    const confidenceMetricsForFile = fileLevelAggregates.confidence_metrics_for_file as Record<
      string,
      unknown
    >;

    fieldIdToInterpretation[field.id] = fieldInterpretation;
    confidenceMetricsForFile[field.fieldLabel] =
      fieldInterpretation.getFieldLevelConfidenceMetrics();
  }

  /**
   * Update directory-level aggregates when a field interpretation has been processed.
   *
   * @param field - Field that was processed
   * @param _fieldInterpretation - Interpretation result (not used)
   * @param _templateFieldLevelAggregates - Template field-level aggregates (not used)
   */
  updateDirectoryLevelAggregatesOnProcessedFieldInterpretation(
    field: Field,
    _fieldInterpretation: FieldInterpretation,
    _templateFieldLevelAggregates: FieldLevelAggregates
  ): void {
    this.updateDirectoryLevelAggregatesOnProcessedField(field, _templateFieldLevelAggregates);
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

    const fileLevelAggregates = this.getFileLevelAggregates();
    if (!fileLevelAggregates) {
      return;
    }
    
    let fieldDetectionTypeWiseAggregates = fileLevelAggregates.field_detection_type_wise_aggregates as Record<
      string,
      unknown
    > | undefined;

    // Ensure field_detection_type_wise_aggregates exists
    if (!fieldDetectionTypeWiseAggregates) {
      fieldDetectionTypeWiseAggregates = {};
      this.insertFileLevelAggregates({
        field_detection_type_wise_aggregates: fieldDetectionTypeWiseAggregates,
      });
    }

    for (const fieldDetectionTypeFileRunner of Object.values(fieldDetectionTypeFileRunners)) {
      const fieldDetectionType = (fieldDetectionTypeFileRunner as {
        fieldDetectionType: string;
        getFileLevelInterpretationAggregates(): unknown;
      }).fieldDetectionType;
      fieldDetectionTypeWiseAggregates[fieldDetectionType] =
        fieldDetectionTypeFileRunner.getFileLevelInterpretationAggregates();
    }

    // Update read_response_flags
    const readResponseFlags = fileLevelAggregates.read_response_flags as {
      is_multi_marked: boolean;
      multi_marked_fields: Field[];
      is_identifier_multi_marked: boolean;
    };

    if (readResponseFlags.is_multi_marked) {
      // Thread-safe update to directory-level aggregates
      const directoryLevelAggregates = this.getDirectoryLevelAggregates()!;
      const filesByLabelCount = directoryLevelAggregates.files_by_label_count as StatsByLabel;
      filesByLabelCount.push('multi_marked');
    }
  }
}

