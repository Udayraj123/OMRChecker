/**
 * Tests for detection result models
 * Port of Python's src/tests/test_refactored_detection.py
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  BubbleMeanValue,
  BubbleFieldDetectionResult,
  ScanQuality,
  type BubbleLocation,
} from '../models';

describe('BubbleMeanValue', () => {
  it('should create bubble mean value with all properties', () => {
    const bubble: BubbleLocation = {
      x: 10,
      y: 20,
      width: 30,
      height: 30,
      label: 'A',
    };

    const meanValue = new BubbleMeanValue(150.5, bubble, [10, 20]);

    expect(meanValue.meanValue).toBe(150.5);
    expect(meanValue.unitBubble).toBe(bubble);
    expect(meanValue.position).toEqual([10, 20]);
  });

  it('should handle different mean values', () => {
    const bubble: BubbleLocation = {
      x: 0,
      y: 0,
      width: 20,
      height: 20,
      label: 'B',
    };

    const darkBubble = new BubbleMeanValue(50, bubble, [0, 0]);
    const lightBubble = new BubbleMeanValue(200, bubble, [0, 0]);

    expect(darkBubble.meanValue).toBe(50);
    expect(lightBubble.meanValue).toBe(200);
  });
});

describe('BubbleFieldDetectionResult', () => {
  let testBubbles: BubbleLocation[];
  let testBubbleMeans: BubbleMeanValue[];

  beforeEach(() => {
    testBubbles = [
      { x: 0, y: 0, width: 20, height: 20, label: 'A' },
      { x: 30, y: 0, width: 20, height: 20, label: 'B' },
      { x: 60, y: 0, width: 20, height: 20, label: 'C' },
      { x: 90, y: 0, width: 20, height: 20, label: 'D' },
    ];

    testBubbleMeans = [
      new BubbleMeanValue(180, testBubbles[0], [0, 0]),   // Unmarked (light)
      new BubbleMeanValue(80, testBubbles[1], [30, 0]),   // Marked (dark)
      new BubbleMeanValue(190, testBubbles[2], [60, 0]),  // Unmarked (light)
      new BubbleMeanValue(185, testBubbles[3], [90, 0]),  // Unmarked (light)
    ];
  });

  describe('constructor', () => {
    it('should create result with basic properties', () => {
      const result = new BubbleFieldDetectionResult(
        'Q1',
        'Question 1',
        testBubbleMeans
      );

      expect(result.fieldId).toBe('Q1');
      expect(result.fieldLabel).toBe('Question 1');
      expect(result.bubbleMeans).toEqual(testBubbleMeans);
      expect(result.timestamp).toBeInstanceOf(Date);
    });

    it('should create result with custom timestamp', () => {
      const customDate = new Date('2026-01-15T12:00:00Z');
      const result = new BubbleFieldDetectionResult(
        'Q2',
        'Question 2',
        testBubbleMeans,
        customDate
      );

      expect(result.timestamp).toBe(customDate);
    });
  });

  describe('stdDeviation getter', () => {
    it('should calculate standard deviation correctly', () => {
      const result = new BubbleFieldDetectionResult(
        'Q1',
        'Question 1',
        testBubbleMeans
      );

      const stdDev = result.stdDeviation;

      // Expected: mean = (180 + 80 + 190 + 185) / 4 = 158.75
      // variance = ((180-158.75)^2 + (80-158.75)^2 + (190-158.75)^2 + (185-158.75)^2) / 4
      // variance = (451.5625 + 6201.5625 + 976.5625 + 689.0625) / 4 = 2079.6875
      // stdDev = sqrt(2079.6875) ≈ 45.6
      expect(stdDev).toBeCloseTo(45.6, 1);
    });

    it('should return 0 for empty bubble means', () => {
      const result = new BubbleFieldDetectionResult(
        'Q_EMPTY',
        'Empty Question',
        []
      );

      expect(result.stdDeviation).toBe(0);
    });

    it('should return 0 for single bubble', () => {
      const singleBubble = [testBubbleMeans[0]];
      const result = new BubbleFieldDetectionResult(
        'Q_SINGLE',
        'Single',
        singleBubble
      );

      expect(result.stdDeviation).toBe(0);
    });
  });

  describe('meanValues getter', () => {
    it('should extract mean values from bubble means', () => {
      const result = new BubbleFieldDetectionResult(
        'Q1',
        'Question 1',
        testBubbleMeans
      );

      expect(result.meanValues).toEqual([180, 80, 190, 185]);
    });

    it('should return empty array for no bubbles', () => {
      const result = new BubbleFieldDetectionResult(
        'Q_EMPTY',
        'Empty',
        []
      );

      expect(result.meanValues).toEqual([]);
    });
  });

  describe('jumps getter', () => {
    it('should calculate jumps between consecutive values', () => {
      const result = new BubbleFieldDetectionResult(
        'Q1',
        'Question 1',
        testBubbleMeans
      );

      const jumps = result.jumps;

      // Jumps: |180-80| = 100, |80-190| = 110, |190-185| = 5
      expect(jumps).toEqual([100, 110, 5]);
    });

    it('should return empty array for single or no bubbles', () => {
      const singleResult = new BubbleFieldDetectionResult(
        'Q_SINGLE',
        'Single',
        [testBubbleMeans[0]]
      );

      expect(singleResult.jumps).toEqual([]);

      const emptyResult = new BubbleFieldDetectionResult(
        'Q_EMPTY',
        'Empty',
        []
      );

      expect(emptyResult.jumps).toEqual([]);
    });
  });

  describe('scanQuality getter', () => {
    it('should return GOOD for high std deviation', () => {
      // High std deviation indicates clear distinction between marked/unmarked
      const result = new BubbleFieldDetectionResult(
        'Q_GOOD',
        'Good Quality',
        testBubbleMeans
      );

      expect(result.scanQuality).toBe(ScanQuality.GOOD);
    });

    it('should return MEDIOCRE for medium std deviation', () => {
      // Medium std deviation (20-40 range)
      const mediumBubbles = [
        new BubbleMeanValue(150, testBubbles[0], [0, 0]),
        new BubbleMeanValue(170, testBubbles[1], [30, 0]),
        new BubbleMeanValue(160, testBubbles[2], [60, 0]),
        new BubbleMeanValue(180, testBubbles[3], [90, 0]),
      ];

      const result = new BubbleFieldDetectionResult(
        'Q_MEDIOCRE',
        'Mediocre Quality',
        mediumBubbles
      );

      expect(result.scanQuality).toBe(ScanQuality.MEDIOCRE);
    });

    it('should return POOR for low std deviation', () => {
      // Low std deviation (< 20) indicates poor scan quality
      const poorBubbles = [
        new BubbleMeanValue(150, testBubbles[0], [0, 0]),
        new BubbleMeanValue(155, testBubbles[1], [30, 0]),
        new BubbleMeanValue(152, testBubbles[2], [60, 0]),
        new BubbleMeanValue(148, testBubbles[3], [90, 0]),
      ];

      const result = new BubbleFieldDetectionResult(
        'Q_POOR',
        'Poor Quality',
        poorBubbles
      );

      expect(result.scanQuality).toBe(ScanQuality.POOR);
    });
  });

  describe('edge cases', () => {
    it('should handle all same values (zero variance)', () => {
      const sameBubbles = testBubbles.map((b, i) =>
        new BubbleMeanValue(150, b, [i * 30, 0])
      );

      const result = new BubbleFieldDetectionResult(
        'Q_SAME',
        'Same Values',
        sameBubbles
      );

      expect(result.stdDeviation).toBe(0);
      expect(result.scanQuality).toBe(ScanQuality.POOR);
      expect(result.jumps).toEqual([0, 0, 0]);
    });

    it('should handle extreme values', () => {
      const extremeBubbles = [
        new BubbleMeanValue(0, testBubbles[0], [0, 0]),
        new BubbleMeanValue(255, testBubbles[1], [30, 0]),
      ];

      const result = new BubbleFieldDetectionResult(
        'Q_EXTREME',
        'Extreme',
        extremeBubbles
      );

      expect(result.stdDeviation).toBeGreaterThan(100);
      expect(result.scanQuality).toBe(ScanQuality.GOOD);
      expect(result.jumps).toEqual([255]);
    });
  });
});

