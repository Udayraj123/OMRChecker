/**
 * Unit tests for AdaptiveThreshold strategy.
 *
 * TypeScript port of tests for src/processors/threshold/adaptive_threshold.py
 */

import { describe, it, expect } from 'vitest';
import {
  AdaptiveThreshold,
  createDefaultThresholdCalculator,
} from '../AdaptiveThreshold';
import { GlobalThreshold } from '../GlobalThreshold';
import { LocalThreshold } from '../LocalThreshold';
import { ThresholdConfig } from '../GlobalThreshold';

describe('AdaptiveThreshold', () => {
  const defaultConfig: ThresholdConfig = {
    defaultThreshold: 200,
    minJump: 30,
  };

  it('should combine multiple strategies using weighted average', () => {
    const globalStrategy = new GlobalThreshold();
    const localStrategy = new LocalThreshold();
    const adaptive = new AdaptiveThreshold([globalStrategy, localStrategy], [0.5, 0.5]);

    const bubbleValues = [50, 55, 60, 150, 155, 160];

    const result = adaptive.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.methodUsed).toBe('adaptive_weighted');
    expect(result.thresholdValue).toBeGreaterThan(0);
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.metadata.strategyResults).toBeDefined();
    expect(result.metadata.strategyResults).toHaveLength(2);
  });

  it('should use different weights for strategies', () => {
    const globalStrategy = new GlobalThreshold();
    const localStrategy = new LocalThreshold();
    const adaptive = new AdaptiveThreshold([globalStrategy, localStrategy], [0.8, 0.2]);

    const bubbleValues = [50, 55, 60, 150, 155, 160];

    const result = adaptive.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.methodUsed).toBe('adaptive_weighted');
    expect(result.metadata.strategyResults[0].weight).toBe(0.8);
    expect(result.metadata.strategyResults[1].weight).toBe(0.2);
  });

  it('should return default when all strategies have zero confidence', () => {
    const globalStrategy = new GlobalThreshold();
    const localStrategy = new LocalThreshold();
    const adaptive = new AdaptiveThreshold([globalStrategy, localStrategy], [0.5, 0.5]);

    // Values with no clear separation
    const bubbleValues = [100, 101, 102, 103, 104];

    const result = adaptive.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.methodUsed).toBe('adaptive_all_zero_confidence');
    expect(result.thresholdValue).toBe(defaultConfig.defaultThreshold);
    expect(result.confidence).toBe(0.0);
    expect(result.fallbackUsed).toBe(true);
  });

  it('should use max confidence and max jump from all strategies', () => {
    const globalStrategy = new GlobalThreshold();
    const localStrategy = new LocalThreshold();
    const adaptive = new AdaptiveThreshold([globalStrategy, localStrategy], [0.5, 0.5]);

    const bubbleValues = [50, 55, 60, 150, 155, 160];

    const globalResult = globalStrategy.calculateThreshold(bubbleValues, defaultConfig);
    const localResult = localStrategy.calculateThreshold(bubbleValues, defaultConfig);
    const adaptiveResult = adaptive.calculateThreshold(bubbleValues, defaultConfig);

    expect(adaptiveResult.confidence).toBe(
      Math.max(globalResult.confidence, localResult.confidence)
    );
    expect(adaptiveResult.maxJump).toBe(
      Math.max(globalResult.maxJump, localResult.maxJump)
    );
  });

  it('should calculate weighted threshold correctly', () => {
    const globalStrategy = new GlobalThreshold();
    const localStrategy = new LocalThreshold();
    const adaptive = new AdaptiveThreshold([globalStrategy, localStrategy], [0.4, 0.6]);

    const bubbleValues = [50, 55, 60, 150, 155, 160];

    const result = adaptive.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.methodUsed).toBe('adaptive_weighted');
    expect(result.thresholdValue).toBeGreaterThan(0);
    expect(result.metadata.strategyResults).toHaveLength(2);
  });

  it('should handle single strategy', () => {
    const globalStrategy = new GlobalThreshold();
    const adaptive = new AdaptiveThreshold([globalStrategy], [1.0]);

    const bubbleValues = [50, 55, 60, 150, 155, 160];

    const result = adaptive.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.methodUsed).toBe('adaptive_weighted');
    expect(result.metadata.strategyResults).toHaveLength(1);
  });

  it('should throw error if strategies and weights length mismatch', () => {
    const globalStrategy = new GlobalThreshold();
    const localStrategy = new LocalThreshold();

    expect(() => {
      new AdaptiveThreshold([globalStrategy, localStrategy], [0.5]);
    }).toThrow('Number of strategies must match number of weights');
  });

  it('should use equal weights by default', () => {
    const globalStrategy = new GlobalThreshold();
    const localStrategy = new LocalThreshold();
    const adaptive = new AdaptiveThreshold([globalStrategy, localStrategy]);

    const bubbleValues = [50, 55, 60, 150, 155, 160];

    const result = adaptive.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.metadata.strategyResults[0].weight).toBe(1.0);
    expect(result.metadata.strategyResults[1].weight).toBe(1.0);
  });

  it('should include strategy results in metadata', () => {
    const globalStrategy = new GlobalThreshold();
    const localStrategy = new LocalThreshold();
    const adaptive = new AdaptiveThreshold([globalStrategy, localStrategy], [0.4, 0.6]);

    const bubbleValues = [50, 55, 60, 150, 155, 160];

    const result = adaptive.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.metadata.strategyResults).toBeDefined();
    expect(result.metadata.strategyResults).toHaveLength(2);
    expect(result.metadata.strategyResults[0]).toHaveProperty('method');
    expect(result.metadata.strategyResults[0]).toHaveProperty('threshold');
    expect(result.metadata.strategyResults[0]).toHaveProperty('confidence');
    expect(result.metadata.strategyResults[0]).toHaveProperty('weight');
  });

  it('should set fallbackUsed if any strategy uses fallback', () => {
    const globalStrategy = new GlobalThreshold();
    const localStrategy = new LocalThreshold();
    const adaptive = new AdaptiveThreshold([globalStrategy, localStrategy], [0.5, 0.5]);

    // Values with no clear separation - will trigger fallback
    const bubbleValues = [100, 101, 102, 103, 104];

    const result = adaptive.calculateThreshold(bubbleValues, defaultConfig);

    // If all strategies have zero confidence, fallback is used
    expect(result.fallbackUsed).toBe(true);
  });
});

describe('createDefaultThresholdCalculator', () => {
  const defaultConfig: ThresholdConfig = {
    defaultThreshold: 200,
    minJump: 30,
  };

  it('should create adaptive threshold with global and local strategies', () => {
    const calculator = createDefaultThresholdCalculator();

    const bubbleValues = [50, 55, 60, 150, 155, 160];

    const result = calculator.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.methodUsed).toBe('adaptive_weighted');
    expect(result.metadata.strategyResults).toHaveLength(2);
  });

  it('should use provided global threshold for local strategy fallback', () => {
    const globalThreshold = 180.0;
    const calculator = createDefaultThresholdCalculator(globalThreshold);

    const bubbleValues = [50, 55, 60, 150, 155, 160];

    const result = calculator.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.methodUsed).toBe('adaptive_weighted');
    expect(result.metadata.strategyResults).toHaveLength(2);
  });

  it('should use default weights (0.4 global, 0.6 local)', () => {
    const calculator = createDefaultThresholdCalculator();

    const bubbleValues = [50, 55, 60, 150, 155, 160];

    const result = calculator.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.metadata.strategyResults[0].weight).toBe(0.4);
    expect(result.metadata.strategyResults[1].weight).toBe(0.6);
  });
});
