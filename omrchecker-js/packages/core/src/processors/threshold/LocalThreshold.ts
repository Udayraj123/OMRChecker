/**
 * Local threshold strategy for bubble detection.
 *
 * TypeScript port of src/processors/threshold/LocalThreshold.py
 * Maintains 1:1 correspondence with Python implementation.
 */

import { ThresholdConfig, ThresholdResult } from './GlobalThreshold';
import { Logger } from '../../utils/logger';

const logger = new Logger('LocalThreshold');

/**
 * Local threshold strategy - calculates threshold per question/group.
 *
 * This method is useful when lighting conditions vary across the image.
 */
export class LocalThreshold {
  calculateThreshold(
    bubbleMeanValues: number[],
    config: ThresholdConfig
  ): ThresholdResult {
    if (bubbleMeanValues.length < 2) {
      return {
        thresholdValue: config.defaultThreshold,
        confidence: 0.0,
        maxJump: 0.0,
        methodUsed: 'local_default',
        fallbackUsed: true,
        metadata: {},
      };
    }

    const sortedValues = [...bubbleMeanValues].sort((a, b) => a - b);

    // For local threshold, we look at the gap between min and next value
    const minValue = sortedValues[0];
    const secondValue = sortedValues[1];
    const maxJump = secondValue - minValue;

    // Threshold is midpoint between darkest and second-darkest bubble
    const threshold = (minValue + secondValue) / 2;

    // Calculate confidence
    let confidence = 0.0;
    if (maxJump > config.minJump) {
      confidence = Math.min(1.0, maxJump / (config.minJump * 2));
    }

    logger.debug(
      `LocalThreshold: maxJump=${maxJump.toFixed(2)}, threshold=${threshold.toFixed(2)}, confidence=${confidence.toFixed(2)}`
    );

    return {
      thresholdValue: threshold,
      confidence,
      maxJump,
      methodUsed: 'local_threshold',
      fallbackUsed: maxJump <= config.minJump,
      metadata: {
        minValue,
        secondValue,
        sortedValues,
      },
    };
  }
}

