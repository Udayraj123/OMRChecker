/**
 * Unit tests for LocalThreshold strategy.
 *
 * TypeScript port of tests for src/processors/threshold/LocalThreshold.py
 */

import { describe, it, expect } from 'vitest';
import { LocalThreshold } from '../LocalThreshold';
import { ThresholdConfig } from '../GlobalThreshold';

describe('LocalThreshold', () => {
  const defaultConfig: ThresholdConfig = {
    defaultThreshold: 200,
    minJump: 30,
  };

  it('should calculate threshold between darkest and second-darkest bubble', () => {
    const strategy = new LocalThreshold();
    const bubbleValues = [50, 100, 150, 160]; // Darkest is 50, second is 100

    const result = strategy.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.thresholdValue).toBeCloseTo(75, 0); // Midpoint of 50 and 100
    expect(result.maxJump).toBeCloseTo(50, 0);
    expect(result.methodUsed).toBe('local_threshold');
    expect(result.fallbackUsed).toBe(false);
  });

  it('should return default threshold for insufficient data', () => {
    const strategy = new LocalThreshold();
    const bubbleValues = [100]; // Only one value

    const result = strategy.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.thresholdValue).toBe(200);
    expect(result.methodUsed).toBe('local_default');
    expect(result.fallbackUsed).toBe(true);
    expect(result.confidence).toBe(0.0);
  });

  it('should handle small gap between darkest bubbles', () => {
    const strategy = new LocalThreshold();
    const bubbleValues = [100, 110, 150, 160]; // Gap is only 10

    const result = strategy.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.maxJump).toBeCloseTo(10, 0);
    expect(result.thresholdValue).toBeCloseTo(105, 0);
    expect(result.fallbackUsed).toBe(true); // Gap < minJump
    expect(result.confidence).toBeLessThan(0.5);
  });

  it('should calculate confidence based on gap size', () => {
    const strategy = new LocalThreshold();

    // Large gap should give high confidence
    const largeGapValues = [50, 150, 160, 170];
    const largeResult = strategy.calculateThreshold(largeGapValues, defaultConfig);

    // Small gap should give low confidence
    const smallGapValues = [100, 130, 140, 150];
    const smallResult = strategy.calculateThreshold(smallGapValues, defaultConfig);

    expect(largeResult.confidence).toBeGreaterThan(smallResult.confidence);
  });

  it('should handle unsorted input correctly', () => {
    const strategy = new LocalThreshold();
    const unsortedValues = [160, 100, 50, 150]; // Will be sorted internally

    const result = strategy.calculateThreshold(unsortedValues, defaultConfig);

    expect(result.thresholdValue).toBeCloseTo(75, 0); // Midpoint of 50 and 100
    expect(result.maxJump).toBeCloseTo(50, 0);
  });

  it('should include metadata with min values', () => {
    const strategy = new LocalThreshold();
    const bubbleValues = [50, 100, 150, 160];

    const result = strategy.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.metadata).toBeDefined();
    expect(result.metadata.minValue).toBe(50);
    expect(result.metadata.secondValue).toBe(100);
    expect(result.metadata.sortedValues).toHaveLength(4);
  });

  it('should work with just two values', () => {
    const strategy = new LocalThreshold();
    const bubbleValues = [50, 150];

    const result = strategy.calculateThreshold(bubbleValues, defaultConfig);

    expect(result.thresholdValue).toBeCloseTo(100, 0);
    expect(result.maxJump).toBeCloseTo(100, 0);
    expect(result.fallbackUsed).toBe(false);
  });

  it('should be effective for single-question thresholding', () => {
    const strategy = new LocalThreshold();
    // Simulating a single question where one bubble is marked (dark)
    // and others are unmarked (light)
    const singleQuestionValues = [45, 180, 185, 190]; // One marked, three unmarked

    const result = strategy.calculateThreshold(singleQuestionValues, defaultConfig);

    // Threshold should separate the marked bubble (45) from unmarked ones (180+)
    expect(result.thresholdValue).toBeGreaterThan(50);
    expect(result.thresholdValue).toBeLessThan(180);
    expect(result.fallbackUsed).toBe(false);
  });

  it('should use custom minJump in config', () => {
    const strategy = new LocalThreshold();
    const bubbleValues = [50, 90, 100, 150]; // Gap is 40

    const strictConfig: ThresholdConfig = {
      defaultThreshold: 200,
      minJump: 50, // Higher threshold for "significant" jump
    };

    const result = strategy.calculateThreshold(bubbleValues, strictConfig);

    expect(result.fallbackUsed).toBe(true); // Gap of 40 is below minJump of 50
  });
});

