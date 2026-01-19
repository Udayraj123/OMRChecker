/**
 * Global threshold strategy for bubble detection.
 *
 * TypeScript port of src/processors/threshold/GlobalThreshold.py
 * Maintains 1:1 correspondence with Python implementation.
 */

import { Logger } from '../../utils/logger';

const logger = new Logger('GlobalThreshold');

/**
 * Configuration for threshold calculation
 */
export interface ThresholdConfig {
  defaultThreshold: number;
  minJump: number;
  jumpDelta?: number;
  minGapTwoBubbles?: number;
  minJumpSurplusForGlobalFallback?: number;
  confidentJumpSurplusForDisparity?: number;
  globalThresholdMargin?: number;
  outlierDeviationThreshold?: number;
}

/**
 * Result of threshold calculation
 */
export interface ThresholdResult {
  thresholdValue: number;
  confidence: number;
  maxJump: number;
  methodUsed: string;
  fallbackUsed: boolean;
  metadata: Record<string, any>;
}

/**
 * Global threshold strategy - finds the first large gap in sorted bubble values.
 *
 * This is the primary threshold detection method used in OMRChecker.
 */
export class GlobalThreshold {
  calculateThreshold(
    bubbleMeanValues: number[],
    config: ThresholdConfig
  ): ThresholdResult {
    if (bubbleMeanValues.length < 2) {
      return {
        thresholdValue: config.defaultThreshold,
        confidence: 0.0,
        maxJump: 0.0,
        methodUsed: 'global_default',
        fallbackUsed: true,
        metadata: {},
      };
    }

    const sortedValues = [...bubbleMeanValues].sort((a, b) => a - b);

    // Find the FIRST LARGE GAP using looseness parameter
    const looseness = 1;
    const ls = Math.floor((looseness + 1) / 2);

    let maxJump = 0;
    let threshold = config.defaultThreshold;
    let bestIndex = -1;

    for (let i = ls; i < sortedValues.length - ls; i++) {
      const jump = sortedValues[i] - sortedValues[i - 1];
      if (jump > maxJump) {
        maxJump = jump;
        bestIndex = i;
      }
    }

    if (bestIndex !== -1) {
      threshold = (sortedValues[bestIndex - 1] + sortedValues[bestIndex]) / 2;
    }

    // Calculate confidence based on jump size
    let confidence = 0.0;
    if (maxJump > config.minJump) {
      confidence = Math.min(1.0, maxJump / (config.minJump * 2));
    }

    logger.debug(
      `GlobalThreshold: maxJump=${maxJump.toFixed(2)}, threshold=${threshold.toFixed(2)}, confidence=${confidence.toFixed(2)}`
    );

    return {
      thresholdValue: threshold,
      confidence,
      maxJump,
      methodUsed: 'global_threshold',
      fallbackUsed: maxJump <= config.minJump,
      metadata: {
        sortedValues,
        bestIndex,
      },
    };
  }
}

