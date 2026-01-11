/**
 * Adaptive threshold strategy for bubble detection.
 *
 * TypeScript port of src/processors/threshold/AdaptiveThreshold.py
 * Maintains 1:1 correspondence with Python implementation.
 */

import { ThresholdConfig, ThresholdResult } from './GlobalThreshold';
import { Logger } from '../../utils/logger';

const logger = new Logger('AdaptiveThreshold');

/**
 * Adaptive threshold strategy - combines global and local approaches.
 *
 * Adapts threshold calculation based on the distribution of bubble values.
 */
export class AdaptiveThreshold {
  calculateThreshold(
    bubbleMeanValues: number[],
    config: ThresholdConfig
  ): ThresholdResult {
    if (bubbleMeanValues.length < 2) {
      return {
        thresholdValue: config.defaultThreshold,
        confidence: 0.0,
        maxJump: 0.0,
        methodUsed: 'adaptive_default',
        fallbackUsed: true,
        metadata: {},
      };
    }

    const sortedValues = [...bubbleMeanValues].sort((a, b) => a - b);

    // Calculate statistics
    const mean = sortedValues.reduce((a, b) => a + b, 0) / sortedValues.length;
    const variance =
      sortedValues.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) /
      sortedValues.length;
    const stdDev = Math.sqrt(variance);

    // Find all significant jumps
    const jumps: Array<{ index: number; jump: number }> = [];
    for (let i = 1; i < sortedValues.length; i++) {
      const jump = sortedValues[i] - sortedValues[i - 1];
      if (jump > config.minJump) {
        jumps.push({ index: i, jump });
      }
    }

    let threshold = config.defaultThreshold;
    let maxJump = 0;
    let methodUsed = 'adaptive_default';
    let bestIndex = -1;

    if (jumps.length > 0) {
      // Use the largest jump
      jumps.sort((a, b) => b.jump - a.jump);
      const largestJump = jumps[0];
      maxJump = largestJump.jump;
      bestIndex = largestJump.index;
      threshold = (sortedValues[bestIndex - 1] + sortedValues[bestIndex]) / 2;
      methodUsed = 'adaptive_largest_jump';
    }

    // Calculate confidence based on both jump size and distribution
    let confidence = 0.0;
    if (maxJump > config.minJump) {
      const jumpConfidence = Math.min(1.0, maxJump / (config.minJump * 2));
      const distributionConfidence = Math.min(1.0, stdDev / 50.0); // Normalize by expected std dev
      confidence = (jumpConfidence + distributionConfidence) / 2;
    }

    logger.debug(
      `AdaptiveThreshold: maxJump=${maxJump.toFixed(2)}, threshold=${threshold.toFixed(2)}, ` +
        `confidence=${confidence.toFixed(2)}, stdDev=${stdDev.toFixed(2)}`
    );

    return {
      thresholdValue: threshold,
      confidence,
      maxJump,
      methodUsed,
      fallbackUsed: maxJump <= config.minJump,
      metadata: {
        sortedValues,
        bestIndex,
        mean,
        stdDev,
        jumpsFound: jumps.length,
      },
    };
  }
}

