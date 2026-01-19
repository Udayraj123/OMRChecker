/**
 * Adaptive threshold strategy for bubble detection.
 *
 * TypeScript port of src/processors/threshold/adaptive_threshold.py
 * Maintains 1:1 correspondence with Python implementation.
 *
 * Adaptive strategy that combines multiple strategies.
 * Uses weighted average based on confidence scores.
 */

import { GlobalThreshold } from './GlobalThreshold';
import { ThresholdConfig, ThresholdResult } from './GlobalThreshold';
import { LocalThreshold } from './LocalThreshold';
import { Logger } from '../../utils/logger';

const logger = new Logger('AdaptiveThreshold');

/**
 * Strategy interface for threshold calculation.
 */
export interface ThresholdStrategy {
  calculateThreshold(
    bubbleMeanValues: number[],
    config: ThresholdConfig
  ): ThresholdResult;
}

/**
 * Adaptive threshold strategy - combines multiple strategies.
 *
 * Uses weighted average based on confidence scores.
 */
export class AdaptiveThreshold implements ThresholdStrategy {
  private strategies: ThresholdStrategy[];
  private weights: number[];

  /**
   * Initialize with strategies and optional weights.
   *
   * @param strategies - List of threshold strategies to combine
   * @param weights - Optional weights for each strategy (default: equal weights)
   */
  constructor(
    strategies: ThresholdStrategy[],
    weights: number[] | null = null
  ) {
    this.strategies = strategies;
    this.weights = weights || strategies.map(() => 1.0);

    if (this.strategies.length !== this.weights.length) {
      throw new Error('Number of strategies must match number of weights');
    }
  }

  /**
   * Calculate threshold using weighted average of strategies.
   *
   * @param bubbleMeanValues - List of bubble mean intensity values
   * @param config - Threshold configuration
   * @returns ThresholdResult with threshold and confidence information
   */
  calculateThreshold(
    bubbleMeanValues: number[],
    config: ThresholdConfig
  ): ThresholdResult {
    // Get results from all strategies
    const results = this.strategies.map((strategy) =>
      strategy.calculateThreshold(bubbleMeanValues, config)
    );

    // Calculate weighted average based on confidence
    let totalWeight = 0;
    for (let i = 0; i < results.length; i++) {
      totalWeight += results[i].confidence * this.weights[i];
    }

    if (totalWeight === 0) {
      // All strategies have zero confidence
      return {
        thresholdValue: config.defaultThreshold,
        confidence: 0.0,
        maxJump: 0.0,
        methodUsed: 'adaptive_all_zero_confidence',
        fallbackUsed: true,
        metadata: {},
      };
    }

    // Calculate weighted threshold
    let weightedThreshold = 0;
    for (let i = 0; i < results.length; i++) {
      weightedThreshold +=
        results[i].thresholdValue *
        results[i].confidence *
        this.weights[i];
    }
    weightedThreshold /= totalWeight;

    // Use max confidence and max jump from all strategies
    const maxConfidence = Math.max(...results.map((r) => r.confidence));
    const maxJumpValue = Math.max(...results.map((r) => r.maxJump));

    // Build strategy results metadata
    const strategyResults = results.map((result, i) => ({
      method: result.methodUsed,
      threshold: result.thresholdValue,
      confidence: result.confidence,
      weight: this.weights[i],
    }));

    logger.debug(
      `AdaptiveThreshold: weighted_threshold=${weightedThreshold.toFixed(2)}, ` +
        `max_confidence=${maxConfidence.toFixed(2)}, max_jump=${maxJumpValue.toFixed(2)}`
    );

    return {
      thresholdValue: weightedThreshold,
      confidence: maxConfidence,
      maxJump: maxJumpValue,
      methodUsed: 'adaptive_weighted',
      fallbackUsed: results.some((r) => r.fallbackUsed),
      metadata: {
        strategyResults: strategyResults,
      },
    };
  }
}

/**
 * Factory function to create default threshold calculator.
 *
 * @param globalThreshold - Optional global threshold for local strategy fallback
 * @returns AdaptiveThreshold combining global and local strategies
 */
export function createDefaultThresholdCalculator(
  globalThreshold: number | null = null
): AdaptiveThreshold {
  return new AdaptiveThreshold(
    [
      new GlobalThreshold(),
      new LocalThreshold(globalThreshold),
    ],
    [0.4, 0.6] // Prefer local threshold
  );
}
