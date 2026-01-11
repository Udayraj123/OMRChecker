/**
 * Unit tests for GlobalThreshold strategy.
 *
 * TypeScript port of tests for src/processors/threshold/GlobalThreshold.py
 */

import { describe, it, expect } from 'vitest';
import { GlobalThreshold, ThresholdConfig } from '../GlobalThreshold';

describe('GlobalThreshold', () => {
  const defaultConfig: ThresholdConfig = {
    defaultThreshold: 200,
    minJump: 30,
  };

  it('should calculate threshold for well-separated values', () => {
    const strategy = new GlobalThreshold();
    const bubbleValues = [50, 55, 60, 150, 155, 160]; // Clear gap at 60->150

    const result = strategy.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.thresholdValue).toBeCloseTo(105, 0); // Midpoint of gap
    expect(result.maxJump).toBeCloseTo(90, 0);
    expect(result.methodUsed).toBe('global_threshold');
    expect(result.fallbackUsed).toBe(false);
    expect(result.confidence).toBeGreaterThan(0.5);
  });

  it('should find first large gap, not necessarily the largest', () => {
    const strategy = new GlobalThreshold();
    const bubbleValues = [50, 100, 110, 200]; // Two gaps: 50->100 (50) and 110->200 (90)

    const result = strategy.calculateThreshold(bubbleValues, defaultConfig);

    // Should use the first significant gap (50->100)
    expect(result.maxJump).toBeCloseTo(50, 0);
    expect(result.thresholdValue).toBeCloseTo(75, 0);
  });

  it('should return default threshold for insufficient data', () => {
    const strategy = new GlobalThreshold();
    const bubbleValues = [100]; // Only one value

    const result = strategy.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.thresholdValue).toBe(200);
    expect(result.methodUsed).toBe('global_default');
    expect(result.fallbackUsed).toBe(true);
    expect(result.confidence).toBe(0.0);
  });

  it('should handle values with no significant gap', () => {
    const strategy = new GlobalThreshold();
    const bubbleValues = [100, 105, 110, 115, 120]; // All close together

    const result = strategy.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.maxJump).toBeLessThan(defaultConfig.minJump);
    expect(result.fallbackUsed).toBe(true);
    expect(result.confidence).toBeLessThan(0.5);
  });

  it('should calculate confidence based on jump size', () => {
    const strategy = new GlobalThreshold();

    // Large jump should give high confidence
    const largeJumpValues = [50, 55, 200, 205];
    const largeResult = strategy.calculateThreshold(largeJumpValues, defaultConfig);

    // Small jump should give low confidence
    const smallJumpValues = [100, 135, 140, 145];
    const smallResult = strategy.calculateThreshold(smallJumpValues, defaultConfig);

    expect(largeResult.confidence).toBeGreaterThan(smallResult.confidence);
  });

  it('should handle unsorted input correctly', () => {
    const strategy = new GlobalThreshold();
    const unsortedValues = [160, 55, 150, 50, 155, 60]; // Will be sorted internally

    const result = strategy.calculateThreshold(unsortedValues, defaultConfig);

    expect(result.thresholdValue).toBeCloseTo(105, 0); // Same as sorted version
    expect(result.maxJump).toBeCloseTo(90, 0);
  });

  it('should include metadata in result', () => {
    const strategy = new GlobalThreshold();
    const bubbleValues = [50, 55, 150, 155];

    const result = strategy.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.metadata).toBeDefined();
    expect(result.metadata.sortedValues).toBeDefined();
    expect(result.metadata.bestIndex).toBeGreaterThanOrEqual(0);
    expect(result.metadata.sortedValues).toHaveLength(4);
  });

  it('should handle edge case with two values', () => {
    const strategy = new GlobalThreshold();
    const bubbleValues = [50, 150];

    const result = strategy.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.thresholdValue).toBeCloseTo(100, 0);
    expect(result.maxJump).toBeCloseTo(100, 0);
    expect(result.fallbackUsed).toBe(false);
  });

  it('should use custom minJump in config', () => {
    const strategy = new GlobalThreshold();
    const bubbleValues = [50, 80, 100, 150]; // Max jump is 50

    const strictConfig: ThresholdConfig = {
      defaultThreshold: 200,
      minJump: 60, // Higher threshold for "significant" jump
    };

    const result = strategy.calculateThreshold(bubbleValues, strictConfig);

    expect(result.fallbackUsed).toBe(true); // Jump of 50 is below minJump of 60
  });
});

