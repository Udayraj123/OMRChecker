/**
 * Unit tests for AdaptiveThreshold strategy.
 *
 * TypeScript port of tests for src/processors/threshold/AdaptiveThreshold.py
 */

import { describe, it, expect } from 'vitest';
import { AdaptiveThreshold } from '../AdaptiveThreshold';
import { ThresholdConfig } from '../GlobalThreshold';

describe('AdaptiveThreshold', () => {
  const defaultConfig: ThresholdConfig = {
    defaultThreshold: 200,
    minJump: 30,
  };

  it('should calculate threshold using largest jump', () => {
    const strategy = new AdaptiveThreshold();
    const bubbleValues = [50, 55, 60, 150, 155, 160]; // Clear gap at 60->150

    const result = strategy.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.thresholdValue).toBeCloseTo(105, 0); // Midpoint of largest gap
    expect(result.maxJump).toBeCloseTo(90, 0);
    expect(result.methodUsed).toBe('adaptive_largest_jump');
    expect(result.fallbackUsed).toBe(false);
  });

  it('should return default threshold for insufficient data', () => {
    const strategy = new AdaptiveThreshold();
    const bubbleValues = [100]; // Only one value

    const result = strategy.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.thresholdValue).toBe(200);
    expect(result.methodUsed).toBe('adaptive_default');
    expect(result.fallbackUsed).toBe(true);
    expect(result.confidence).toBe(0.0);
  });

  it('should handle values with no significant gap', () => {
    const strategy = new AdaptiveThreshold();
    const bubbleValues = [100, 105, 110, 115, 120]; // All close together

    const result = strategy.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.maxJump).toBeLessThan(defaultConfig.minJump);
    expect(result.methodUsed).toBe('adaptive_default');
    expect(result.fallbackUsed).toBe(true);
  });

  it('should calculate confidence combining jump and distribution', () => {
    const strategy = new AdaptiveThreshold();

    // Large jump with high variance should give high confidence
    const highVarianceValues = [50, 55, 200, 205];
    const highResult = strategy.calculateThreshold(highVarianceValues, defaultConfig);

    // Small jump with low variance should give low confidence
    const lowVarianceValues = [100, 135, 140, 145];
    const lowResult = strategy.calculateThreshold(lowVarianceValues, defaultConfig);

    expect(highResult.confidence).toBeGreaterThan(lowResult.confidence);
  });

  it('should prefer largest jump over first jump', () => {
    const strategy = new AdaptiveThreshold();
    const bubbleValues = [50, 90, 95, 200]; // Jumps: 40, 5, 105

    const result = strategy.calculateThreshold(bubbleValues, defaultConfig);

    // Should use the largest jump (95->200, jump=105)
    expect(result.maxJump).toBeCloseTo(105, 0);
    expect(result.thresholdValue).toBeCloseTo(147.5, 0); // Midpoint
  });

  it('should include distribution statistics in metadata', () => {
    const strategy = new AdaptiveThreshold();
    const bubbleValues = [50, 55, 150, 155];

    const result = strategy.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.metadata).toBeDefined();
    expect(result.metadata.mean).toBeDefined();
    expect(result.metadata.stdDev).toBeDefined();
    expect(result.metadata.jumpsFound).toBeDefined();
    expect(result.metadata.sortedValues).toHaveLength(4);
  });

  it('should handle multiple significant jumps', () => {
    const strategy = new AdaptiveThreshold();
    // Two groups with significant jumps: [50-60] and [150-160]
    const bubbleValues = [50, 55, 60, 150, 155, 160];

    const result = strategy.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.metadata.jumpsFound).toBeGreaterThan(0);
    // Should choose the largest jump
    expect(result.maxJump).toBeGreaterThan(60);
  });

  it('should handle unsorted input correctly', () => {
    const strategy = new AdaptiveThreshold();
    const unsortedValues = [160, 55, 150, 50, 155, 60]; // Will be sorted internally

    const result = strategy.calculateThreshold(unsortedValues, defaultConfig);

    expect(result.thresholdValue).toBeCloseTo(105, 0);
    expect(result.maxJump).toBeCloseTo(90, 0);
  });

  it('should work with edge case of two values', () => {
    const strategy = new AdaptiveThreshold();
    const bubbleValues = [50, 150];

    const result = strategy.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.thresholdValue).toBeCloseTo(100, 0);
    expect(result.maxJump).toBeCloseTo(100, 0);
    expect(result.fallbackUsed).toBe(false);
  });

  it('should adapt to varying lighting conditions', () => {
    const strategy = new AdaptiveThreshold();
    // Simulating varying lighting: some dark marked (40-50),
    // some light marked (80-90), unmarked (150-200)
    const varyingValues = [40, 45, 50, 85, 90, 150, 160, 170, 180, 190, 200];

    const result = strategy.calculateThreshold(varyingValues, defaultConfig);

    // Should find the largest gap (90->150)
    expect(result.thresholdValue).toBeGreaterThan(90);
    expect(result.thresholdValue).toBeLessThan(150);
    expect(result.fallbackUsed).toBe(false);
  });

  it('should use custom minJump in config', () => {
    const strategy = new AdaptiveThreshold();
    const bubbleValues = [50, 80, 100, 150]; // Max jump is 50

    const strictConfig: ThresholdConfig = {
      defaultThreshold: 200,
      minJump: 60, // Higher threshold for "significant" jump
    };

    const result = strategy.calculateThreshold(bubbleValues, strictConfig);

    expect(result.fallbackUsed).toBe(true);
    expect(result.metadata.jumpsFound).toBe(0); // No jumps exceed minJump
  });

  it('should provide higher confidence than simple methods for complex distributions', () => {
    const strategy = new AdaptiveThreshold();
    // Complex distribution with clear separation
    const complexValues = [45, 50, 52, 55, 58, 175, 180, 185, 190, 195];

    const result = strategy.calculateThreshold(complexValues, defaultConfig);

    // Large jump and high variance should give good confidence
    expect(result.confidence).toBeGreaterThan(0.5);
    expect(result.thresholdValue).toBeGreaterThan(60);
    expect(result.thresholdValue).toBeLessThan(170);
  });
});

