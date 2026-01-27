/**
 * Bubbles threshold interpretation.
 *
 * TypeScript port of src/processors/detection/bubbles_threshold/interpretation.py
 * Simplified bubble field interpretation using threshold strategies.
 */

import { Logger } from '../../../utils/logger';
import { FieldInterpretation, type InterpretationDrawing } from '../base/interpretation';
import type { Field } from '../../layout/field/base';
import {
  BubbleFieldDetectionResult,
  BubbleMeanValue,
} from '../models/detectionResults';
import { type ThresholdConfig, type ThresholdResult } from '../../threshold/GlobalThreshold';
import { LocalThreshold } from '../../threshold/LocalThreshold';
import type { BubblesScanBox } from '../../layout/field/bubbleField';
import { BubblesFieldInterpretationDrawing } from './interpretationDrawing';
import type { InterpretationThresholdConfig } from '../../../schemas/models/config';

const logger = new Logger('BubblesFieldInterpretation');

/**
 * Single bubble interpretation result.
 */
export class BubbleInterpretation {
  public bubbleMean: BubbleMeanValue;
  public threshold: number;
  public meanValue: number;
  public isAttempted: boolean;
  public bubbleValue: string;
  public itemReference: BubblesScanBox;

  constructor(bubbleMean: BubbleMeanValue, threshold: number) {
    this.bubbleMean = bubbleMean;
    this.threshold = threshold;
    this.meanValue = bubbleMean.meanValue;
    this.isAttempted = bubbleMean.meanValue < threshold;

    // Extract bubble value from unit_bubble if available
    // Note: unitBubble in BubbleMeanValue is BubbleLocation (interface), but in practice
    // it's a BubblesScanBox instance. We need to handle this properly.
    const unitBubble = bubbleMean.unitBubble as unknown as BubblesScanBox;
    this.bubbleValue = unitBubble?.bubbleValue || ''; // Ensure empty string if undefined/null

    // item_reference is used by the drawing code
    this.itemReference = unitBubble;
  }

  /**
   * Get bubble value if marked.
   *
   * @returns Bubble value or empty string
   */
  getValue(): string {
    return this.isAttempted ? this.bubbleValue : '';
  }
}

/**
 * Clean, simplified bubble field interpretation using strategies.
 *
 * Threshold calculation delegated to threshold strategy classes.
 */
export class BubblesFieldInterpretation extends FieldInterpretation {
  // Initialize with defaults. Parent constructor will call runInterpretation() which will
  // set proper values, but TypeScript properties declared without initializers are undefined
  // during parent constructor execution.
  public bubbleInterpretations: BubbleInterpretation[] = [];
  public isMultiMarked: boolean = false;
  public localThresholdForField: number = 0.0;
  public thresholdResult: ThresholdResult | null = null;

  constructor(
    tuningConfig: Record<string, unknown>,
    field: Field,
    fileLevelDetectionAggregates: unknown,
    fileLevelInterpretationAggregates: unknown
  ) {
    // Parent constructor calls runInterpretation(), but our property initializers
    // haven't run yet (they run AFTER super returns in TypeScript).
    // So runInterpretation sees undefined properties and initializes them.
    // Then when super() returns, our property initializers run and overwrite them.
    // Solution: Don't use property initializers, or re-run initialization after super().
    super(tuningConfig, field, fileLevelDetectionAggregates, fileLevelInterpretationAggregates);
    
    // Re-run interpretation after super() to ensure values are set after property initializers
    this.runInterpretation(field, fileLevelDetectionAggregates, fileLevelInterpretationAggregates);
  }

  /**
   * Get drawing instance for visualization.
   */
  getDrawingInstance(): InterpretationDrawing {
    return new BubblesFieldInterpretationDrawing(this);
  }

  /**
   * Get final interpretation string.
   *
   * Returns concatenated marked bubble values or empty value.
   * Special case: If ALL bubbles are marked, treat as unmarked (likely scanning issue).
   *
   * @returns Final interpretation string
   */
  getFieldInterpretationString(): string {
    const markedBubbles = this.bubbleInterpretations
      .filter((interp) => interp.isAttempted)
      .map((interp) => interp.bubbleValue);

    // If no bubbles marked, return empty value
    if (markedBubbles.length === 0) {
      return this.emptyValue;
    }

    // If ALL bubbles are marked, treat as unmarked (likely scanning/detection issue)
    const totalBubbles = this.bubbleInterpretations.length;
    if (markedBubbles.length === totalBubbles) {
      return this.emptyValue;
    }

    return markedBubbles.join('');
  }

  /**
   * Run interpretation using detection results and threshold strategies.
   *
   * @param field - Field to interpret
   * @param fileLevelDetectionAggregates - Detection results
   * @param fileLevelInterpretationAggregates - Interpretation aggregates
   */
  runInterpretation(
    field: Field,
    fileLevelDetectionAggregates: unknown,
    fileLevelInterpretationAggregates: unknown
  ): void {
    // Initialize properties first to ensure they're never undefined
    this.bubbleInterpretations = [];
    this.isMultiMarked = false;
    this.localThresholdForField = 0.0;
    this.thresholdResult = null;

    // Step 1: Extract detection result
    const detectionResult = this.extractDetectionResult(
      field,
      fileLevelDetectionAggregates
    );

    if (!detectionResult) {
      const aggregates = fileLevelDetectionAggregates as Record<string, unknown>;
      const bubbleFields = aggregates.bubble_fields as Record<string, unknown> | undefined;
      const availableFields = bubbleFields ? Object.keys(bubbleFields) : [];
      
      throw new Error(
        `No detection result for field '${field.fieldLabel}'. Available: [${availableFields.join(', ')}]`
      );
    }
    if (!detectionResult.bubbleMeans || detectionResult.bubbleMeans.length === 0) {
      throw new Error(
        `No bubble means in detection result for field '${field.fieldLabel}'`
      );
    }

    // Step 2: Calculate thresholds using strategies
    const thresholdConfig = this.createThresholdConfig(
      fileLevelInterpretationAggregates
    );

    const result = this.calculateThreshold(
      detectionResult,
      fileLevelInterpretationAggregates,
      thresholdConfig
    );

    if (!result) {
      logger.error('calculateThreshold returned null/undefined');
      return;
    }

    this.thresholdResult = result;
    this.localThresholdForField = result.thresholdValue;

    // Step 3: Interpret bubbles
    this.interpretBubbles(detectionResult);

    // Step 4: Check multi-marking
    this.checkMultiMarking();

    // Step 5: Calculate confidence metrics (if enabled)
    const outputs = this.tuningConfig.outputs as { show_confidence_metrics?: boolean } | undefined;
    if (outputs?.show_confidence_metrics) {
      this.calculateConfidenceMetrics(
        detectionResult,
        fileLevelInterpretationAggregates
      );
    }
  }

  /**
   * Extract detection result from aggregates.
   *
   * Can work with both new typed models and legacy dict format.
   */
  private extractDetectionResult(
    field: Field,
    fileLevelDetectionAggregates: unknown
  ): BubbleFieldDetectionResult | null {
    const fieldLabel = field.fieldLabel;
    const aggregates = fileLevelDetectionAggregates as Record<string, unknown>;

    // Try to get from new typed format first
    if ('bubble_fields' in aggregates) {
      const bubbleFields = aggregates.bubble_fields as Record<string, BubbleFieldDetectionResult>;
      if (bubbleFields[fieldLabel]) {
        return bubbleFields[fieldLabel];
      }
      logger.warn(`Field ${fieldLabel} not found in bubble_fields`);
      return null;
    }

    // Fallback to legacy dict format
    const fieldLabelWiseAggregates = aggregates.field_label_wise_aggregates as Record<
      string,
      { field_bubble_means: BubbleMeanValue[] }
    >;

    if (!fieldLabelWiseAggregates || !fieldLabelWiseAggregates[fieldLabel]) {
      logger.warn(
        `bubble_fields not found in file_level_detection_aggregates for field ${fieldLabel}`
      );
      return null;
    }

    const fieldLevelDetectionAggregates = fieldLabelWiseAggregates[fieldLabel];

    // Create typed result from legacy format
    return new BubbleFieldDetectionResult(
      field.id,
      fieldLabel,
      fieldLevelDetectionAggregates.field_bubble_means
    );
  }

  /**
   * Create threshold configuration from tuning config and file-level aggregates.
   */
  private createThresholdConfig(
    _fileLevelInterpretationAggregates: unknown
  ): ThresholdConfig {
    const thresholding = this.tuningConfig.thresholding as InterpretationThresholdConfig;

    return {
      defaultThreshold: thresholding?.global_page_threshold || 180,
      minJump: thresholding?.min_jump || 10,
      minGapTwoBubbles: thresholding?.min_gap_two_bubbles || 20,
      minJumpSurplusForGlobalFallback: thresholding?.min_jump_surplus_for_global_fallback || 10,
      // Note: TypeScript ThresholdConfig is simpler than Python's
      // Additional config options can be added if needed
    };
  }

  /**
   * Calculate threshold using strategies.
   *
   * Uses LocalThreshold with global fallback.
   */
  private calculateThreshold(
    detectionResult: BubbleFieldDetectionResult,
    fileLevelInterpretationAggregates: unknown,
    config: ThresholdConfig
  ): ThresholdResult {
    // Get global fallback threshold
    const fileLevelAggs = fileLevelInterpretationAggregates as Record<string, unknown>;
    const globalFallback =
      (fileLevelAggs.file_level_fallback_threshold as number) || config.defaultThreshold;

    // Use local strategy with global fallback
    const localThreshold = new LocalThreshold(globalFallback);
    const meanValues = detectionResult.meanValues;

    // Calculate threshold
    return localThreshold.calculateThreshold(meanValues, config);
  }

  /**
   * Interpret each bubble using calculated threshold.
   *
   * Creates interpretation for each bubble.
   */
  private interpretBubbles(detectionResult: BubbleFieldDetectionResult): void {
    if (!detectionResult || !detectionResult.bubbleMeans) {
      logger.warn('No bubble means in detection result');
      this.bubbleInterpretations = [];
      return;
    }
    this.bubbleInterpretations = detectionResult.bubbleMeans.map(
      (bubbleMean) =>
        new BubbleInterpretation(bubbleMean, this.localThresholdForField)
    );
  }

  /**
   * Check if multiple bubbles are marked.
   */
  private checkMultiMarking(): void {
    const markedCount = this.bubbleInterpretations.filter(
      (interp) => interp.isAttempted
    ).length;
    this.isMultiMarked = markedCount > 1;

    if (this.isMultiMarked) {
      logger.warn(
        `Multi-marking detected in field: ${this.field.fieldLabel}, marked bubbles: ${markedCount}`
      );
    }
  }

  /**
   * Calculate confidence metrics for the field.
   *
   * Simplified version - can be expanded based on threshold result.
   */
  private calculateConfidenceMetrics(
    detectionResult: BubbleFieldDetectionResult,
    fileLevelAggregates: unknown
  ): void {
    const aggregates = fileLevelAggregates as Record<string, unknown>;
    const globalThreshold =
      (aggregates.file_level_fallback_threshold as number) ||
      this.thresholdResult?.thresholdValue ||
      180;

    // Check for disparity between global and local thresholds
    const disparityBubbles: BubbleMeanValue[] = [];
    for (const bubbleMean of detectionResult.bubbleMeans) {
      const localMarked = bubbleMean.meanValue < this.localThresholdForField;
      const globalMarked = bubbleMean.meanValue < globalThreshold;

      if (localMarked !== globalMarked) {
        disparityBubbles.push(bubbleMean);
      }
    }

    // Calculate overall confidence score for ML training
    const confidenceScore = this.calculateOverallConfidenceScore(
      detectionResult,
      disparityBubbles
    );

    // Build confidence metrics
    this.insertFieldLevelConfidenceMetrics({
      local_threshold: this.localThresholdForField,
      global_threshold: globalThreshold,
      threshold_confidence: this.thresholdResult?.confidence || 0,
      threshold_method: this.thresholdResult?.methodUsed || 'unknown',
      max_jump: this.thresholdResult?.maxJump || 0,
      bubbles_in_doubt: {
        by_disparity: disparityBubbles,
      },
      is_local_jump_confident: (this.thresholdResult?.confidence || 0) > 0.7,
      field_label: this.field.fieldLabel,
      scan_quality: detectionResult.scanQuality,
      std_deviation: detectionResult.stdDeviation,
      overall_confidence_score: confidenceScore,
    });

    if (disparityBubbles.length > 0) {
      logger.warn(
        `Threshold disparity in field: ${this.field.fieldLabel}, bubbles in doubt: ${disparityBubbles.length}`
      );
    }
  }

  /**
   * Calculate overall confidence score for this field's detection.
   *
   * Score ranges from 0.0 to 1.0 based on:
   * - Threshold margin (how far marked bubbles are from threshold)
   * - Multi-mark probability
   * - Bubble intensity consistency within field
   * - Disparity with global threshold
   *
   * @returns Confidence score between 0.0 (low confidence) and 1.0 (high confidence)
   */
  private calculateOverallConfidenceScore(
    detectionResult: BubbleFieldDetectionResult,
    disparityBubbles: BubbleMeanValue[]
  ): number {
    if (detectionResult.bubbleMeans.length === 0) {
      return 0.0;
    }

    if (!this.thresholdResult) {
      return 0.0;
    }

    // Factor 1: Threshold confidence from strategy (0.0-1.0)
    const thresholdConfidence = this.thresholdResult.confidence;

    // Factor 2: Margin from threshold (how clearly marked/unmarked)
    const markedBubbles = detectionResult.bubbleMeans.filter(
      (b) => b.meanValue < this.localThresholdForField
    );

    let marginConfidence: number;
    if (markedBubbles.length > 0) {
      const avgMargin =
        markedBubbles.reduce(
          (sum, b) => sum + (this.localThresholdForField - b.meanValue),
          0
        ) / markedBubbles.length;
      // Normalize margin confidence (larger margin = higher confidence)
      // Assume 50 intensity units is very confident
      marginConfidence = Math.min(1.0, avgMargin / 50.0);
    } else {
      // No bubbles marked - check unmarked confidence
      const avgDistance =
        detectionResult.bubbleMeans.reduce(
          (sum, b) => sum + (b.meanValue - this.localThresholdForField),
          0
        ) / detectionResult.bubbleMeans.length;
      marginConfidence = Math.min(1.0, avgDistance / 50.0);
    }

    // Factor 3: Multi-mark penalty
    const markedCount = markedBubbles.length;
    let multiMarkPenalty: number;
    if (markedCount > 1) {
      multiMarkPenalty = 0.3; // Reduce by 30%
    } else if (markedCount === 0) {
      multiMarkPenalty = 0.1; // Slight penalty
    } else {
      multiMarkPenalty = 0.0; // Single mark - ideal
    }

    // Factor 4: Disparity penalty
    const disparityRatio =
      detectionResult.bubbleMeans.length > 0
        ? disparityBubbles.length / detectionResult.bubbleMeans.length
        : 0;
    const disparityPenalty = disparityRatio * 0.4; // Up to 40% penalty

    // Factor 5: Scan quality
    const scanQualityMap: Record<string, number> = {
      EXCELLENT: 1.0,
      GOOD: 0.9,
      MODERATE: 0.7,
      POOR: 0.5,
    };
    const scanQualityFactor =
      scanQualityMap[detectionResult.scanQuality] || 0.5;

    // Combine factors (weighted average)
    const confidenceScore =
      (thresholdConfidence * 0.35 +
        marginConfidence * 0.25 +
        scanQualityFactor * 0.2) *
      (1.0 - multiMarkPenalty - disparityPenalty);

    // Clamp to [0, 1]
    return Math.max(0.0, Math.min(1.0, confidenceScore));
  }
}

