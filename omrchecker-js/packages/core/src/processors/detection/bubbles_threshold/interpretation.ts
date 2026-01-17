/**
 * Bubbles threshold interpretation.
 *
 * TypeScript port of src/processors/detection/bubbles_threshold/interpretation.py
 * Simplified bubble field interpretation using threshold strategies.
 */

import { Logger } from '../../../utils/logger';
import { FieldInterpretation } from '../base/interpretation';
import type { Field } from '../../layout/field/base';
import {
  BubbleFieldDetectionResult,
  BubbleMeanValue,
} from '../models/detectionResults';
import { GlobalThreshold, type ThresholdConfig, type ThresholdResult } from '../../threshold/GlobalThreshold';
import { LocalThreshold } from '../../threshold/LocalThreshold';

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
  public itemReference: unknown; // BubblesScanBox

  constructor(bubbleMean: BubbleMeanValue, threshold: number) {
    this.bubbleMean = bubbleMean;
    this.threshold = threshold;
    this.meanValue = bubbleMean.meanValue;
    this.isAttempted = bubbleMean.meanValue < threshold;

    // Extract bubble value from unit_bubble if available
    const unitBubble = bubbleMean.unitBubble as { bubbleValue?: string };
    this.bubbleValue = unitBubble?.bubbleValue || '';

    // item_reference is used by the drawing code
    this.itemReference = bubbleMean.unitBubble;
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
 * Drawing instance interface (forward declaration).
 */
export interface BubblesFieldInterpretationDrawing {
  // Stub for now
}

/**
 * Clean, simplified bubble field interpretation using strategies.
 *
 * Threshold calculation delegated to threshold strategy classes.
 */
export class BubblesFieldInterpretation extends FieldInterpretation {
  public bubbleInterpretations: BubbleInterpretation[] = [];
  public isMultiMarked = false;
  public localThresholdForField = 0.0;
  public thresholdResult: ThresholdResult | null = null;

  constructor(
    tuningConfig: Record<string, unknown>,
    field: Field,
    fileLevelDetectionAggregates: unknown,
    fileLevelInterpretationAggregates: unknown
  ) {
    super(tuningConfig, field, fileLevelDetectionAggregates, fileLevelInterpretationAggregates);
  }

  /**
   * Get drawing instance for visualization.
   */
  getDrawingInstance(): BubblesFieldInterpretationDrawing {
    // Will be implemented when we port interpretation drawing
    return {} as BubblesFieldInterpretationDrawing;
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
    // Step 1: Extract detection result
    const detectionResult = this.extractDetectionResult(
      field,
      fileLevelDetectionAggregates
    );

    // Step 2: Calculate thresholds using strategies
    const thresholdConfig = this.createThresholdConfig(
      fileLevelInterpretationAggregates
    );
    this.thresholdResult = this.calculateThreshold(
      detectionResult,
      fileLevelInterpretationAggregates,
      thresholdConfig
    );

    this.localThresholdForField = this.thresholdResult.thresholdValue;

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
  ): BubbleFieldDetectionResult {
    const fieldLabel = field.fieldLabel;
    const aggregates = fileLevelDetectionAggregates as Record<string, unknown>;

    // Try to get from new typed format first
    if ('bubble_fields' in aggregates) {
      const bubbleFields = aggregates.bubble_fields as Record<string, BubbleFieldDetectionResult>;
      if (bubbleFields[fieldLabel]) {
        return bubbleFields[fieldLabel];
      }
    }

    // Fallback to legacy dict format
    const fieldLabelWiseAggregates = aggregates.field_label_wise_aggregates as Record<
      string,
      { field_bubble_means: BubbleMeanValue[] }
    >;

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
    const config = this.tuningConfig.thresholding as {
      MIN_JUMP?: number;
      JUMP_DELTA?: number;
      MIN_GAP_TWO_BUBBLES?: number;
      MIN_JUMP_SURPLUS_FOR_GLOBAL_FALLBACK?: number;
      CONFIDENT_JUMP_SURPLUS_FOR_DISPARITY?: number;
      GLOBAL_THRESHOLD_MARGIN?: number;
      GLOBAL_PAGE_THRESHOLD?: number;
    };

    return {
      defaultThreshold: config.GLOBAL_PAGE_THRESHOLD || 180,
      minJump: config.MIN_JUMP || 10,
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
    const localThreshold = new LocalThreshold();
    const meanValues = detectionResult.meanValues;

    // Calculate local threshold
    const localResult = localThreshold.calculateThreshold(meanValues, {
      ...config,
      defaultThreshold: globalFallback,
    });

    // If local threshold confidence is low, use global fallback
    if (localResult.confidence < 0.5) {
      const globalThreshold = new GlobalThreshold();
      return globalThreshold.calculateThreshold(meanValues, {
        ...config,
        defaultThreshold: globalFallback,
      });
    }

    return localResult;
  }

  /**
   * Interpret each bubble using calculated threshold.
   *
   * Creates interpretation for each bubble.
   */
  private interpretBubbles(detectionResult: BubbleFieldDetectionResult): void {
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

