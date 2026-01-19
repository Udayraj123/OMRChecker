/**
 * Bubbles threshold detection pass.
 *
 * TypeScript port of src/processors/detection/bubbles_threshold/detection_pass.py
 * Extends FieldTypeDetectionPass for bubble field detection.
 */

import * as cv from '@techstark/opencv-js';
import { FieldDetectionType } from '../../constants';
import { FieldTypeDetectionPass } from '../base/detectionPass';
import type { Field } from '../../layout/field/base';
import { NumberAggregate } from '../../../utils/stats';
import { BubblesFieldDetection } from './detection';
import type { TuningConfig } from '../base/commonPass';

/**
 * Detection pass for bubble fields.
 *
 * Manages file-level aggregates (global_max_jump, all_field_bubble_means, all_field_bubble_means_std).
 * Uses BubblesFieldDetection for actual detection.
 */
export class BubblesThresholdDetectionPass extends FieldTypeDetectionPass {
  constructor(tuningConfig: TuningConfig) {
    super(tuningConfig, FieldDetectionType.BUBBLES_THRESHOLD);
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
    // Extract bubble locations from field
    const bubbles = field.scanBoxes.map((scanBox) => ({
      x: scanBox.x,
      y: scanBox.y,
      width: scanBox.dimensions[0],
      height: scanBox.dimensions[1],
      label: (scanBox as { bubbleValue?: string }).bubbleValue || '',
    }));

    // Create detection instance
    // Note: BubblesFieldDetection constructor signature matches Python
    return new BubblesFieldDetection(
      field.id,
      field.fieldLabel,
      bubbles,
      grayImage,
      coloredImage
    );
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
    field: Field,
    _fieldDetection: BubblesFieldDetection,
    fieldLevelAggregates: {
      field?: Field;
      field_bubble_means?: number[];
      field_bubble_means_std?: number;
      [key: string]: unknown;
    }
  ): void {
    super.updateFileLevelAggregatesOnProcessedFieldDetection(
      field,
      _fieldDetection,
      fieldLevelAggregates as any // Type assertion needed due to base class signature
    );

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

