/**
 * Bubbles threshold detection pass.
 *
 * TypeScript port of src/processors/detection/bubbles_threshold/detection_pass.py
 * Extends FieldTypeDetectionPass for bubble field detection.
 */

import { FieldTypeDetectionPass } from '../base/detectionPass';
import type { Field } from '../../layout/field/base';
import { NumberAggregate } from '../../../utils/stats';
import { BubblesFieldDetection } from './detection';
import type { TuningConfig } from '../base/commonPass';
import { DetectionRepository } from '../../repositories/DetectionRepository';

/**
 * Detection pass for bubble fields.
 *
 * Manages file-level aggregates (global_max_jump, all_field_bubble_means, all_field_bubble_means_std).
 * Uses BubblesFieldDetection for actual detection.
 */
export class BubblesThresholdDetectionPass extends FieldTypeDetectionPass {
  private repository: DetectionRepository;

  constructor(
    tuningConfig: TuningConfig,
    fieldDetectionType: string,
    repository: DetectionRepository
  ) {
    super(tuningConfig, fieldDetectionType);
    this.repository = repository;
  }

  /**
   * Get field detection for a bubble field.
   *
   * @param field - Field to detect
   * @param grayImage - Grayscale image
   * @param coloredImage - Colored image (optional)
   * @returns BubblesFieldDetection result
   */
  getFieldDetection(
    field: Field,
    grayImage: cv.Mat,
    coloredImage?: cv.Mat
  ): BubblesFieldDetection {
    return new BubblesFieldDetection(field, grayImage, coloredImage);
  }

  /**
   * Initialize directory-level aggregates.
   *
   * @param initialDirectoryPath - Path to the initial directory
   */
  initializeDirectoryLevelAggregates(initialDirectoryPath: string): void {
    super.initializeDirectoryLevelAggregates(initialDirectoryPath);
    this.insertDirectoryLevelAggregates({
      file_wise_thresholds: new NumberAggregate(),
    });
    // Note: Repository is initialized by TemplateFileRunner before calling this method
  }

  /**
   * Initialize file-level aggregates.
   *
   * @param filePath - Path to the file being processed
   */
  initializeFileLevelAggregates(filePath: string): void {
    super.initializeFileLevelAggregates(filePath);
    this.insertFileLevelAggregates({
      global_max_jump: null as number | null,
      all_field_bubble_means: [] as number[],
      all_field_bubble_means_std: [] as number[],
    });
    // Note: Repository is initialized by TemplateFileRunner before calling this method
  }

  /**
   * Update field-level aggregates after detection.
   *
   * @param field - Field that was processed
   * @param fieldDetection - Detection result
   */
  updateFieldLevelAggregatesOnProcessedFieldDetection(
    field: Field,
    fieldDetection: BubblesFieldDetection
  ): void {
    super.updateFieldLevelAggregatesOnProcessedFieldDetection(field, fieldDetection);

    // Use result for aggregates
    if (!fieldDetection.result) {
      throw new Error('Field detection result not available');
    }

    // Save to repository
    this.repository.saveBubbleField(field.id, fieldDetection.result);

    const fieldBubbleMeans = fieldDetection.result.bubbleMeans.map((bm) => bm.meanValue);
    const stdDeviation = fieldDetection.result.stdDeviation;

    this.insertFieldLevelAggregates({
      field_bubble_means: fieldBubbleMeans,
      field_bubble_means_std: stdDeviation,
    });
  }

  /**
   * Update file-level aggregates after field detection.
   *
   * @param field - Field that was processed
   * @param _fieldDetection - Detection result (not used directly)
   * @param fieldLevelAggregates - Field-level aggregates
   */
  updateFileLevelAggregatesOnProcessedFieldDetection(
    _field: Field,
    _fieldDetection: BubblesFieldDetection,
    fieldLevelAggregates: {
      field?: Field;
      field_bubble_means?: number[];
      field_bubble_means_std?: number;
      [key: string]: unknown;
    }
  ): void {
    // When using repository, skip base class update_file_level_aggregates_on_processed_field
    // which populates field_label_wise_aggregates. Just update fields_count.
    const fileAgg = this.getFileLevelAggregates();
    if (!fileAgg) {
      throw new Error('File level aggregates not initialized');
    }

    // Update file-level aggregates with field data
    const fieldBubbleMeans = (fieldLevelAggregates.field_bubble_means as number[]) || [];
    const fieldBubbleMeansStd = (fieldLevelAggregates.field_bubble_means_std as number) || 0;

    const allFieldBubbleMeans = fileAgg.all_field_bubble_means as number[];
    const allFieldBubbleMeansStd = fileAgg.all_field_bubble_means_std as number[];

    allFieldBubbleMeans.push(...fieldBubbleMeans);
    allFieldBubbleMeansStd.push(fieldBubbleMeansStd);
  }

  /**
   * Get file-level detection aggregates.
   * Used by TemplateDetectionPass to collect aggregates from field type runners.
   *
   * @returns File-level aggregates
   */
  getFileLevelDetectionAggregates(): unknown {
    return this.getFileLevelAggregates();
  }
}

