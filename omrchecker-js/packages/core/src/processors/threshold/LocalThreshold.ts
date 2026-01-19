/**
 * Local threshold strategy for bubble detection.
 *
 * TypeScript port of src/processors/threshold/local_threshold.py
 * Maintains 1:1 correspondence with Python implementation.
 */

import { ThresholdConfig, ThresholdResult } from './GlobalThreshold';
import { Logger } from '../../utils/logger';

const logger = new Logger('LocalThreshold');

/**
 * Local threshold strategy - calculates threshold per question/group.
 *
 * Strategy using field-level statistics.
 * Based on the existing get_local_threshold logic.
 * Calculates threshold for individual field, with fallback to global.
 *
 * This method is useful when lighting conditions vary across the image.
 */
export class LocalThreshold {
  private globalFallback: number | null;

  /**
   * Initialize with optional global fallback threshold.
   *
   * @param globalFallback - Global threshold to use when local confidence is low
   */
  constructor(globalFallback: number | null = null) {
    this.globalFallback = globalFallback;
  }

  /**
   * Calculate local threshold with global fallback.
   *
   * @param bubbleMeanValues - List of bubble mean intensity values
   * @param config - Threshold configuration
   * @returns ThresholdResult with threshold and confidence information
   */
  calculateThreshold(
    bubbleMeanValues: number[],
    config: ThresholdConfig
  ): ThresholdResult {
    const fallbackThreshold =
      this.globalFallback ?? config.defaultThreshold;

    // Base case: empty or single bubble
    if (bubbleMeanValues.length < 2) {
      return {
        thresholdValue: fallbackThreshold,
        confidence: 0.0,
        maxJump: 0.0,
        methodUsed: 'local_single_bubble_fallback',
        fallbackUsed: true,
        metadata: {},
      };
    }

    const sortedValues = [...bubbleMeanValues].sort((a, b) => a - b);

    // Special case: exactly 2 bubbles
    if (sortedValues.length === 2) {
      const gap = sortedValues[1] - sortedValues[0];
      const minGapTwoBubbles = config.minGapTwoBubbles ?? 20.0;

      if (gap < minGapTwoBubbles) {
        return {
          thresholdValue: fallbackThreshold,
          confidence: 0.3,
          maxJump: gap,
          methodUsed: 'local_two_bubbles_small_gap_fallback',
          fallbackUsed: true,
          metadata: {},
        };
      }

      // Calculate mean of the two values
      const mean = (sortedValues[0] + sortedValues[1]) / 2;

      return {
        thresholdValue: mean,
        confidence: 0.7,
        maxJump: gap,
        methodUsed: 'local_two_bubbles_mean',
        fallbackUsed: false,
        metadata: {},
      };
    }

    // 3+ bubbles: find largest jump
    let maxJump = 0.0;
    let threshold = fallbackThreshold;

    for (let i = 1; i < sortedValues.length - 1; i++) {
      const jump = sortedValues[i + 1] - sortedValues[i - 1];
      if (jump > maxJump) {
        maxJump = jump;
        threshold = sortedValues[i - 1] + jump / 2;
      }
    }

    // Check if jump is confident
    const minJumpSurplusForGlobalFallback =
      config.minJumpSurplusForGlobalFallback ?? 10.0;
    const confidentJump = config.minJump + minJumpSurplusForGlobalFallback;

    if (maxJump < confidentJump) {
      // Low confidence - use global fallback
      return {
        thresholdValue: fallbackThreshold,
        confidence: 0.4,
        maxJump: maxJump,
        methodUsed: 'local_low_confidence_global_fallback',
        fallbackUsed: true,
        metadata: { localThreshold: threshold },
      };
    }

    // High confidence
    const confidence = Math.min(1.0, maxJump / (confidentJump * 2));

    logger.debug(
      `LocalThreshold: maxJump=${maxJump.toFixed(2)}, threshold=${threshold.toFixed(2)}, confidence=${confidence.toFixed(2)}`
    );

    return {
      thresholdValue: threshold,
      confidence: confidence,
      maxJump: maxJump,
      methodUsed: 'local_max_jump',
      fallbackUsed: false,
      metadata: { numBubbles: bubbleMeanValues.length },
    };
  }
}

