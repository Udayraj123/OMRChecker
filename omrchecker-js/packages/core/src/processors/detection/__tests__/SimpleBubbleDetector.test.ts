/**
 * Unit tests for SimpleBubbleDetector.
 *
 * Tests the core bubble detection logic using threshold strategies.
 */

import { describe, it, expect, beforeAll, afterEach } from 'vitest';
import * as cv from '@techstark/opencv-js';
import {
  SimpleBubbleDetector,
  type BubbleLocation,
  type FieldDetectionResult,
} from '../SimpleBubbleDetector';

describe('SimpleBubbleDetector', () => {
  let testImage: cv.Mat;

  beforeAll(async () => {
    // Wait for OpenCV to be ready
    if (cv.getBuildInformation) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  });

  afterEach(() => {
    if (testImage && !testImage.isDeleted()) {
      testImage.delete();
    }
  });

  describe('Basic Detection', () => {
    it('should detect a single marked bubble', () => {
      // Create test image: 200x200 white background
      testImage = new cv.Mat(200, 200, cv.CV_8UC1, new cv.Scalar(255));

      // Draw a dark circle (marked bubble) at position 50,50
      cv.circle(testImage, new cv.Point(50, 50), 15, new cv.Scalar(50), -1);

      // Draw light circles (unmarked bubbles)
      cv.circle(testImage, new cv.Point(100, 50), 15, new cv.Scalar(200), -1);
      cv.circle(testImage, new cv.Point(150, 50), 15, new cv.Scalar(200), -1);

      const bubbles: BubbleLocation[] = [
        { x: 35, y: 35, width: 30, height: 30, label: 'A' },
        { x: 85, y: 35, width: 30, height: 30, label: 'B' },
        { x: 135, y: 35, width: 30, height: 30, label: 'C' },
      ];

      const detector = new SimpleBubbleDetector();
      const result = detector.detectField(testImage, bubbles, 'Q1');

      expect(result.fieldId).toBe('Q1');
      expect(result.detectedAnswer).toBe('A');
      expect(result.isMultiMarked).toBe(false);
      expect(result.bubbles).toHaveLength(3);

      // Check that only A is marked
      expect(result.bubbles[0].isMarked).toBe(true);
      expect(result.bubbles[1].isMarked).toBe(false);
      expect(result.bubbles[2].isMarked).toBe(false);
    });

    it('should detect no answer when all bubbles are unmarked', () => {
      // Create test image with all light bubbles
      testImage = new cv.Mat(200, 200, cv.CV_8UC1, new cv.Scalar(255));

      cv.circle(testImage, new cv.Point(50, 50), 15, new cv.Scalar(200), -1);
      cv.circle(testImage, new cv.Point(100, 50), 15, new cv.Scalar(200), -1);

      const bubbles: BubbleLocation[] = [
        { x: 35, y: 35, width: 30, height: 30, label: 'A' },
        { x: 85, y: 35, width: 30, height: 30, label: 'B' },
      ];

      const detector = new SimpleBubbleDetector();
      const result = detector.detectField(testImage, bubbles, 'Q1');

      expect(result.detectedAnswer).toBeNull();
      expect(result.isMultiMarked).toBe(false);
      expect(result.bubbles.every((b) => !b.isMarked)).toBe(true);
    });

    it('should detect multi-marked and return darkest bubble', () => {
      // Create test image with two marked bubbles
      testImage = new cv.Mat(200, 200, cv.CV_8UC1, new cv.Scalar(255));

      cv.circle(testImage, new cv.Point(50, 50), 15, new cv.Scalar(60), -1); // Darker
      cv.circle(testImage, new cv.Point(100, 50), 15, new cv.Scalar(80), -1); // Lighter marked
      cv.circle(testImage, new cv.Point(150, 50), 15, new cv.Scalar(200), -1); // Unmarked

      const bubbles: BubbleLocation[] = [
        { x: 35, y: 35, width: 30, height: 30, label: 'A' },
        { x: 85, y: 35, width: 30, height: 30, label: 'B' },
        { x: 135, y: 35, width: 30, height: 30, label: 'C' },
      ];

      const detector = new SimpleBubbleDetector();
      const result = detector.detectField(testImage, bubbles, 'Q1');

      expect(result.isMultiMarked).toBe(true);
      expect(result.detectedAnswer).toBe('A'); // Darkest marked bubble
    });
  });

  describe('Edge Cases', () => {
    it('should handle bubbles at image boundaries', () => {
      testImage = new cv.Mat(100, 100, cv.CV_8UC1, new cv.Scalar(255));

      // Bubble at edge
      const bubbles: BubbleLocation[] = [
        { x: 0, y: 0, width: 20, height: 20, label: 'A' },
        { x: 80, y: 80, width: 20, height: 20, label: 'B' },
      ];

      const detector = new SimpleBubbleDetector();
      const result = detector.detectField(testImage, bubbles, 'Q1');

      expect(result.bubbles).toHaveLength(2);
      // Should not crash
    });

    it('should handle out-of-bounds bubbles gracefully', () => {
      testImage = new cv.Mat(100, 100, cv.CV_8UC1, new cv.Scalar(255));

      const bubbles: BubbleLocation[] = [
        { x: -10, y: -10, width: 20, height: 20, label: 'A' }, // Partially out of bounds
        { x: 150, y: 150, width: 20, height: 20, label: 'B' }, // Completely out of bounds
      ];

      const detector = new SimpleBubbleDetector();
      const result = detector.detectField(testImage, bubbles, 'Q1');

      expect(result.bubbles).toHaveLength(2);
      // Should not crash, returns white (unmarked) for invalid regions
    });

    it('should handle empty bubble list', () => {
      testImage = new cv.Mat(100, 100, cv.CV_8UC1, new cv.Scalar(255));

      const bubbles: BubbleLocation[] = [];

      const detector = new SimpleBubbleDetector();
      const result = detector.detectField(testImage, bubbles, 'Q1');

      expect(result.bubbles).toHaveLength(0);
      expect(result.detectedAnswer).toBeNull();
    });
  });

  describe('Multiple Fields Detection', () => {
    it('should detect multiple fields correctly', () => {
      testImage = new cv.Mat(400, 200, cv.CV_8UC1, new cv.Scalar(255));

      // Field 1: Answer A (top row)
      cv.circle(testImage, new cv.Point(50, 50), 15, new cv.Scalar(50), -1);
      cv.circle(testImage, new cv.Point(100, 50), 15, new cv.Scalar(200), -1);

      // Field 2: Answer B (bottom row)
      cv.circle(testImage, new cv.Point(50, 150), 15, new cv.Scalar(200), -1);
      cv.circle(testImage, new cv.Point(100, 150), 15, new cv.Scalar(50), -1);

      const fields = new Map<string, BubbleLocation[]>([
        [
          'Q1',
          [
            { x: 35, y: 35, width: 30, height: 30, label: 'A' },
            { x: 85, y: 35, width: 30, height: 30, label: 'B' },
          ],
        ],
        [
          'Q2',
          [
            { x: 35, y: 135, width: 30, height: 30, label: 'A' },
            { x: 85, y: 135, width: 30, height: 30, label: 'B' },
          ],
        ],
      ]);

      const detector = new SimpleBubbleDetector();
      const results = detector.detectMultipleFields(testImage, fields);

      expect(results.size).toBe(2);
      expect(results.get('Q1')?.detectedAnswer).toBe('A');
      expect(results.get('Q2')?.detectedAnswer).toBe('B');
    });
  });

  describe('Detection Statistics', () => {
    it('should calculate correct statistics', () => {
      testImage = new cv.Mat(400, 200, cv.CV_8UC1, new cv.Scalar(255));

      // Q1: Answered (A)
      cv.circle(testImage, new cv.Point(50, 50), 15, new cv.Scalar(50), -1);
      cv.circle(testImage, new cv.Point(100, 50), 15, new cv.Scalar(200), -1);

      // Q2: Unanswered
      cv.circle(testImage, new cv.Point(50, 150), 15, new cv.Scalar(200), -1);
      cv.circle(testImage, new cv.Point(100, 150), 15, new cv.Scalar(200), -1);

      // Q3: Multi-marked
      cv.circle(testImage, new cv.Point(50, 250), 15, new cv.Scalar(50), -1);
      cv.circle(testImage, new cv.Point(100, 250), 15, new cv.Scalar(60), -1);

      const fields = new Map<string, BubbleLocation[]>([
        [
          'Q1',
          [
            { x: 35, y: 35, width: 30, height: 30, label: 'A' },
            { x: 85, y: 35, width: 30, height: 30, label: 'B' },
          ],
        ],
        [
          'Q2',
          [
            { x: 35, y: 135, width: 30, height: 30, label: 'A' },
            { x: 85, y: 135, width: 30, height: 30, label: 'B' },
          ],
        ],
        [
          'Q3',
          [
            { x: 35, y: 235, width: 30, height: 30, label: 'A' },
            { x: 85, y: 235, width: 30, height: 30, label: 'B' },
          ],
        ],
      ]);

      const detector = new SimpleBubbleDetector();
      const results = detector.detectMultipleFields(testImage, fields);
      const stats = detector.getDetectionStats(results);

      expect(stats.totalFields).toBe(3);
      expect(stats.answeredFields).toBe(2); // Q1 and Q3
      expect(stats.multiMarkedFields).toBe(1); // Q3
      expect(stats.unansweredFields).toBe(1); // Q2
      expect(stats.avgConfidence).toBeGreaterThan(0);
    });
  });

  describe('Threshold Configuration', () => {
    it('should respect custom threshold config', () => {
      testImage = new cv.Mat(200, 200, cv.CV_8UC1, new cv.Scalar(255));

      // Bubble with medium darkness (150)
      cv.circle(testImage, new cv.Point(50, 50), 15, new cv.Scalar(150), -1);
      cv.circle(testImage, new cv.Point(100, 50), 15, new cv.Scalar(200), -1);

      const bubbles: BubbleLocation[] = [
        { x: 35, y: 35, width: 30, height: 30, label: 'A' },
        { x: 85, y: 35, width: 30, height: 30, label: 'B' },
      ];

      // With default threshold (200), should detect A
      const detector1 = new SimpleBubbleDetector();
      const result1 = detector1.detectField(testImage, bubbles, 'Q1');
      expect(result1.detectedAnswer).toBe('A');

      // With high threshold (100), should not detect anything
      const detector2 = new SimpleBubbleDetector({ defaultThreshold: 100 });
      const result2 = detector2.detectField(testImage, bubbles, 'Q1');
      expect(result2.detectedAnswer).toBeNull();
    });
  });

  describe('Confidence Scoring', () => {
    it('should calculate confidence based on threshold distance', () => {
      testImage = new cv.Mat(200, 200, cv.CV_8UC1, new cv.Scalar(255));

      // Very dark bubble (high confidence)
      cv.circle(testImage, new cv.Point(50, 50), 15, new cv.Scalar(30), -1);
      // Medium dark bubble (lower confidence)
      cv.circle(testImage, new cv.Point(100, 50), 15, new cv.Scalar(120), -1);
      // Light bubble (unmarked)
      cv.circle(testImage, new cv.Point(150, 50), 15, new cv.Scalar(200), -1);

      const bubbles: BubbleLocation[] = [
        { x: 35, y: 35, width: 30, height: 30, label: 'A' },
        { x: 85, y: 35, width: 30, height: 30, label: 'B' },
        { x: 135, y: 35, width: 30, height: 30, label: 'C' },
      ];

      const detector = new SimpleBubbleDetector();
      const result = detector.detectField(testImage, bubbles, 'Q1');

      const markedBubbles = result.bubbles.filter((b) => b.isMarked);
      expect(markedBubbles.length).toBeGreaterThan(0);

      // Very dark bubble should have higher confidence
      const bubbleA = result.bubbles.find((b) => b.label === 'A');
      expect(bubbleA?.confidence).toBeGreaterThan(0);
    });
  });
});

