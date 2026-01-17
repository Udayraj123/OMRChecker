/**
 * Bubbles threshold interpretation pass.
 *
 * TypeScript port of src/processors/detection/bubbles_threshold/interpretation_pass.py
 * Extends FieldTypeInterpretationPass for bubble field interpretation.
 */

import { Logger } from '../../../utils/logger';
import { FieldDetectionType } from '../../constants';
import { FieldTypeInterpretationPass } from '../base/interpretationPass';
import type { Field } from '../../layout/field/base';
import { NumberAggregate } from '../../../utils/stats';
import { BubblesFieldInterpretation } from './interpretation';
import { GlobalThreshold, type ThresholdConfig } from '../../threshold/GlobalThreshold';
import type { TuningConfig, FieldLevelAggregates } from '../base/commonPass';
import type { BubbleMeanValue } from '../models/detectionResults';

const logger = new Logger('BubblesThresholdInterpretationPass');

/**
 * Interpretation pass for bubble fields.
 *
 * Manages file-level aggregates for threshold calculation and interpretation.
 * Uses BubblesFieldInterpretation for actual interpretation.
 */
export class BubblesThresholdInterpretationPass extends FieldTypeInterpretationPass {
  constructor(tuningConfig: TuningConfig) {
    super(tuningConfig, FieldDetectionType.BUBBLES_THRESHOLD);
  }

  /**
   * Get field interpretation for a bubble field.
   *
   * @param field - Field to interpret
   * @param fileLevelDetectionAggregates - File-level detection aggregates
   * @param fileLevelInterpretationAggregates - File-level interpretation aggregates
   * @returns BubblesFieldInterpretation result
   */
  getFieldInterpretation(
    field: Field,
    fileLevelDetectionAggregates: unknown,
    fileLevelInterpretationAggregates: unknown
  ): BubblesFieldInterpretation {
    return new BubblesFieldInterpretation(
      this.tuningConfig,
      field,
      fileLevelDetectionAggregates,
      fileLevelInterpretationAggregates
    );
  }

  /**
   * Initialize file-level aggregates for interpretation.
   *
   * @param filePath - Path to the file being processed
   * @param fieldDetectionTypeWiseDetectionAggregates - Detection aggregates by field detection type
   * @param fieldLabelWiseDetectionAggregates - Detection aggregates by field label
   */
  initializeFileLevelAggregates(
    filePath: string,
    fieldDetectionTypeWiseDetectionAggregates: unknown,
    fieldLabelWiseDetectionAggregates: unknown
  ): void {
    super.initializeFileLevelAggregates(
      filePath,
      fieldDetectionTypeWiseDetectionAggregates,
      fieldLabelWiseDetectionAggregates
    );

    // Get own file-level detection aggregates
    const typeWiseAggs = fieldDetectionTypeWiseDetectionAggregates as Record<
      string,
      {
        all_field_bubble_means_std: number[];
        all_field_bubble_means: BubbleMeanValue[];
      }
    >;

    const ownFileLevelDetectionAggregates =
      typeWiseAggs[this.fieldDetectionType];

    if (!ownFileLevelDetectionAggregates) {
      throw new Error(
        `Detection aggregates not found for ${this.fieldDetectionType}`
      );
    }

    const allOutlierDeviations =
      ownFileLevelDetectionAggregates.all_field_bubble_means_std;
    const outlierDeviationThresholdForFile =
      this.getOutlierDeviationThreshold(allOutlierDeviations);

    const fieldWiseMeansAndRefs =
      ownFileLevelDetectionAggregates.all_field_bubble_means;
    const { fileLevelFallbackThreshold, globalMaxJump } =
      this.getFallbackThreshold(fieldWiseMeansAndRefs);

    logger.debug(
      `Thresholding: file_level_fallback_threshold: ${fileLevelFallbackThreshold.toFixed(2)} global_std_THR: ${outlierDeviationThresholdForFile.toFixed(2)} ${
        fileLevelFallbackThreshold === 255 ? '(Looks like a Xeroxed OMR)' : ''
      }`
    );

    this.insertFileLevelAggregates({
      file_level_fallback_threshold: fileLevelFallbackThreshold,
      global_max_jump: globalMaxJump,
      outlier_deviation_threshold_for_file: outlierDeviationThresholdForFile,
      field_label_wise_local_thresholds: {},
      bubble_field_type_wise_thresholds: {},
      all_fields_local_thresholds: new NumberAggregate(),
      field_wise_confidence_metrics: {},
    });
  }

  /**
   * Get outlier deviation threshold.
   *
   * @param allOutlierDeviations - List of standard deviations
   * @returns Outlier deviation threshold
   */
  getOutlierDeviationThreshold(allOutlierDeviations: number[]): number {
    const config = this.tuningConfig.thresholding as {
      MIN_JUMP_STD?: number;
      GLOBAL_PAGE_THRESHOLD_STD?: number;
    };

    const minJumpStd = config.MIN_JUMP_STD || 5.0;
    const globalPageThresholdStd = config.GLOBAL_PAGE_THRESHOLD_STD || 10.0;

    // Use GlobalThresholdStrategy
    const strategy = new GlobalThreshold();
    const thresholdConfig: ThresholdConfig = {
      minJump: minJumpStd,
      defaultThreshold: globalPageThresholdStd,
    };

    // allOutlierDeviations is already a list of floats (std deviations)
    const result = strategy.calculateThreshold(allOutlierDeviations, thresholdConfig);
    return result.thresholdValue;
  }

  /**
   * Get fallback threshold.
   *
   * @param fieldWiseMeansAndRefs - List of BubbleMeanValue objects
   * @returns Object with fileLevelFallbackThreshold and globalMaxJump
   */
  getFallbackThreshold(fieldWiseMeansAndRefs: BubbleMeanValue[]): {
    fileLevelFallbackThreshold: number;
    globalMaxJump: number;
  } {
    const config = this.tuningConfig.thresholding as {
      GLOBAL_PAGE_THRESHOLD?: number;
      MIN_JUMP?: number;
    };

    const globalPageThreshold = config.GLOBAL_PAGE_THRESHOLD || 180;
    const minJump = config.MIN_JUMP || 10;

    // Use GlobalThresholdStrategy
    const strategy = new GlobalThreshold();
    const thresholdConfig: ThresholdConfig = {
      minJump,
      defaultThreshold: globalPageThreshold,
    };

    // fieldWiseMeansAndRefs is a list of BubbleMeanValue objects
    const bubbleValues = fieldWiseMeansAndRefs.map((item) => item.meanValue);

    const result = strategy.calculateThreshold(bubbleValues, thresholdConfig);

    // Approximate global_max_jump from result
    const globalMaxJump = result.maxJump;

    return {
      fileLevelFallbackThreshold: result.thresholdValue,
      globalMaxJump,
    };
  }

  /**
   * Update field-level aggregates after interpretation.
   *
   * @param field - Field that was processed
   * @param fieldInterpretation - Interpretation result
   */
  updateFieldLevelAggregatesOnProcessedFieldInterpretation(
    field: Field,
    fieldInterpretation: BubblesFieldInterpretation
  ): void {
    super.updateFieldLevelAggregatesOnProcessedFieldInterpretation(
      field,
      fieldInterpretation
    );

    this.insertFieldLevelAggregates({
      is_multi_marked: fieldInterpretation.isMultiMarked,
      local_threshold_for_field: fieldInterpretation.localThresholdForField,
      bubble_interpretations: fieldInterpretation.bubbleInterpretations,
    });
  }

  /**
   * Update file-level aggregates after field interpretation.
   *
   * @param field - Field that was processed
   * @param fieldInterpretation - Interpretation result
   * @param fieldLevelAggregates - Field-level aggregates
   */
  updateFileLevelAggregatesOnProcessedFieldInterpretation(
    field: Field,
    fieldInterpretation: BubblesFieldInterpretation,
    fieldLevelAggregates: FieldLevelAggregates
  ): void {
    super.updateFileLevelAggregatesOnProcessedFieldInterpretation(
      field,
      fieldInterpretation,
      fieldLevelAggregates
    );

    const fileAgg = this.getFileLevelAggregates();
    if (!fileAgg) {
      throw new Error('File level aggregates not initialized');
    }

    const allFieldsLocalThresholds = fileAgg.all_fields_local_thresholds as NumberAggregate;
    allFieldsLocalThresholds.push(fieldInterpretation.localThresholdForField, field.id);

    // TODO: update bubble_field_type_wise_thresholds
  }

  /**
   * Get file-level interpretation aggregates.
   * Used by TemplateInterpretationPass to collect aggregates from field type runners.
   *
   * @returns File-level aggregates
   */
  getFileLevelInterpretationAggregates(): unknown {
    return this.getFileLevelAggregates();
  }
}

