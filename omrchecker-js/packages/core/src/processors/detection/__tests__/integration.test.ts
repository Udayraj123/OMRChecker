/**
 * Integration tests for detection + threshold strategies
 * Tests the complete pipeline from detection to answer determination
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { BubblesFieldDetection } from '../bubbles_threshold';
import { GlobalThreshold } from '../../threshold/GlobalThreshold';
import type { BubbleLocation } from '../models';
const cv = global.cv;

describe('Detection + Threshold Integration', () => {
  let testBubbles: BubbleLocation[];
  let thresholdStrategy: GlobalThreshold;

  beforeEach(() => {
    testBubbles = [
      { x: 10, y: 10, width: 20, height: 20, label: 'A' },
      { x: 40, y: 10, width: 20, height: 20, label: 'B' },
      { x: 70, y: 10, width: 20, height: 20, label: 'C' },
      { x: 100, y: 10, width: 20, height: 20, label: 'D' },
    ];

    thresholdStrategy = new GlobalThreshold();
  });

  describe('single marked bubble', () => {
    it('should correctly identify marked bubble with threshold', () => {
      // Create image with one marked bubble (B = dark)
      const image = new cv.Mat(150, 150, cv.CV_8UC1, new cv.Scalar(200));

      // Mark bubble B (second bubble)
      for (let y = 10; y < 30; y++) {
        for (let x = 40; x < 60; x++) {
          image.ucharPtr(y, x)[0] = 60; // Dark = marked
        }
      }

      // Step 1: Run detection
      const detection = new BubblesFieldDetection(
        'Q1',
        'Question 1',
        testBubbles,
        image
      );
      const result = detection.getResult();

      // Step 2: Calculate threshold
      const thresholdResult = thresholdStrategy.calculateThreshold(
        result.meanValues,
        { defaultThreshold: 130, minJump: 30 }
      );

      // Step 3: Determine marked bubbles
      const markedBubbles = result.bubbleMeans.filter(
        (bm) => bm.meanValue < thresholdResult.thresholdValue
      );

      expect(markedBubbles).toHaveLength(1);
      expect(markedBubbles[0].unitBubble.label).toBe('B');
      expect(result.stdDeviation).toBeGreaterThan(40); // Good quality

      image.delete();
    });
  });

  describe('multiple marked bubbles', () => {
    it('should detect multi-marked field', () => {
      // Create image with two marked bubbles (B and C)
      const image = new cv.Mat(150, 150, cv.CV_8UC1, new cv.Scalar(200));

      // Mark bubble B
      for (let y = 10; y < 30; y++) {
        for (let x = 40; x < 60; x++) {
          image.ucharPtr(y, x)[0] = 65;
        }
      }

      // Mark bubble C
      for (let y = 10; y < 30; y++) {
        for (let x = 70; x < 90; x++) {
          image.ucharPtr(y, x)[0] = 70;
        }
      }

      const detection = new BubblesFieldDetection(
        'Q2',
        'Question 2',
        testBubbles,
        image
      );
      const result = detection.getResult();

      const thresholdResult = thresholdStrategy.calculateThreshold(
        result.meanValues,
        { defaultThreshold: 130, minJump: 30 }
      );

      const markedBubbles = result.bubbleMeans.filter(
        (bm) => bm.meanValue < thresholdResult.thresholdValue
      );

      expect(markedBubbles).toHaveLength(2);
      expect(markedBubbles[0].unitBubble.label).toBe('B');
      expect(markedBubbles[1].unitBubble.label).toBe('C');

      image.delete();
    });
  });

  describe('no marked bubbles', () => {
    it('should detect empty field (no marks)', () => {
      // Create image with all light bubbles (none marked)
      const image = new cv.Mat(150, 150, cv.CV_8UC1, new cv.Scalar(200));

      const detection = new BubblesFieldDetection(
        'Q_EMPTY',
        'Empty Question',
        testBubbles,
        image
      );
      const result = detection.getResult();

      const thresholdResult = thresholdStrategy.calculateThreshold(
        result.meanValues,
        { defaultThreshold: 130, minJump: 30 }
      );

      const markedBubbles = result.bubbleMeans.filter(
        (bm) => bm.meanValue < thresholdResult.thresholdValue
      );

      expect(markedBubbles).toHaveLength(0);

      image.delete();
    });
  });

  describe('adaptive threshold behavior', () => {
    it('should use calculated threshold when jump is significant', () => {
      // Create image with clear distinction
      const image = new cv.Mat(150, 150, cv.CV_8UC1, new cv.Scalar(220));

      // Mark one bubble very dark
      for (let y = 10; y < 30; y++) {
        for (let x = 40; x < 60; x++) {
          image.ucharPtr(y, x)[0] = 50;
        }
      }

      const detection = new BubblesFieldDetection(
        'Q1',
        'Question 1',
        testBubbles,
        image
      );
      const result = detection.getResult();

      const thresholdResult = thresholdStrategy.calculateThreshold(
        result.meanValues,
        { defaultThreshold: 130, minJump: 30 }
      );

      // Should use calculated threshold (not default) due to large jump
      expect(thresholdResult.thresholdValue).not.toBe(130);
      expect(thresholdResult.thresholdValue).toBeGreaterThan(100);
      expect(thresholdResult.thresholdValue).toBeLessThan(150);

      image.delete();
    });

    it('should use default threshold when jumps are small', () => {
      // Create image with similar values (poor quality)
      const image = new cv.Mat(150, 150, cv.CV_8UC1);

      for (let y = 0; y < 150; y++) {
        for (let x = 0; x < 150; x++) {
          image.ucharPtr(y, x)[0] = 145 + (x % 15); // All similar values
        }
      }

      const detection = new BubblesFieldDetection(
        'Q_POOR',
        'Poor Quality',
        testBubbles,
        image
      );
      const result = detection.getResult();

      const thresholdResult = thresholdStrategy.calculateThreshold(
        result.meanValues,
        { defaultThreshold: 130, minJump: 30 }
      );

      // Should use default threshold due to small jumps
      expect(thresholdResult.thresholdValue).toBe(130);

      image.delete();
    });
  });

  describe('realistic scenarios', () => {
    it('should handle typical MCQ answer sheet (1 correct answer)', () => {
      // Simulate a typical MCQ with 4 options, one marked
      const image = new cv.Mat(150, 150, cv.CV_8UC1, new cv.Scalar(210));

      // Mark option C
      for (let y = 10; y < 30; y++) {
        for (let x = 70; x < 90; x++) {
          image.ucharPtr(y, x)[0] = 75;
        }
      }

      const detection = new BubblesFieldDetection(
        'Q15',
        'Question 15',
        testBubbles,
        image
      );
      const result = detection.getResult();

      const thresholdResult = thresholdStrategy.calculateThreshold(
        result.meanValues,
        { defaultThreshold: 140, minJump: 25 }
      );

      const markedBubbles = result.bubbleMeans.filter(
        (bm) => bm.meanValue < thresholdResult.thresholdValue
      );

      expect(markedBubbles).toHaveLength(1);
      expect(markedBubbles[0].unitBubble.label).toBe('C');
      expect(result.scanQuality).toBe('GOOD');

      image.delete();
    });

    it('should handle partially filled bubble (light mark)', () => {
      // Simulate a lightly marked bubble (student's light pencil mark)
      const image = new cv.Mat(150, 150, cv.CV_8UC1, new cv.Scalar(200));

      // Lightly mark bubble A
      for (let y = 10; y < 30; y++) {
        for (let x = 10; x < 30; x++) {
          image.ucharPtr(y, x)[0] = 140; // Not very dark
        }
      }

      const detection = new BubblesFieldDetection(
        'Q5',
        'Question 5',
        testBubbles,
        image
      );
      const result = detection.getResult();

      const thresholdResult = thresholdStrategy.calculateThreshold(
        result.meanValues,
        { defaultThreshold: 160, minJump: 20 }
      );

      const markedBubbles = result.bubbleMeans.filter(
        (bm) => bm.meanValue < thresholdResult.thresholdValue
      );

      // Should still detect the lightly marked bubble with appropriate threshold
      expect(markedBubbles).toHaveLength(1);
      expect(markedBubbles[0].unitBubble.label).toBe('A');

      image.delete();
    });
  });

  describe('scan quality impact', () => {
    it('should provide quality feedback for poor scans', () => {
      const poorImage = new cv.Mat(150, 150, cv.CV_8UC1);

      // Create poor quality scan (low contrast)
      for (let y = 0; y < 150; y++) {
        for (let x = 0; x < 150; x++) {
          poorImage.ucharPtr(y, x)[0] = 120 + (x % 30);
        }
      }

      const detection = new BubblesFieldDetection(
        'Q_POOR_SCAN',
        'Poor Scan',
        testBubbles,
        poorImage
      );
      const result = detection.getResult();

      expect(result.scanQuality).toBe('POOR');
      expect(result.stdDeviation).toBeLessThan(20);

      // In poor quality, threshold might fall back to default
      const thresholdResult = thresholdStrategy.calculateThreshold(
        result.meanValues,
        { defaultThreshold: 130, minJump: 30 }
      );

      expect(thresholdResult.thresholdValue).toBe(130); // Default used

      poorImage.delete();
    });

    it('should provide quality feedback for good scans', () => {
      const goodImage = new cv.Mat(150, 150, cv.CV_8UC1, new cv.Scalar(220));

      // Create good quality scan with clear marked bubble
      for (let y = 10; y < 30; y++) {
        for (let x = 40; x < 60; x++) {
          goodImage.ucharPtr(y, x)[0] = 50;
        }
      }

      const detection = new BubblesFieldDetection(
        'Q_GOOD_SCAN',
        'Good Scan',
        testBubbles,
        goodImage
      );
      const result = detection.getResult();

      expect(result.scanQuality).toBe('GOOD');
      expect(result.stdDeviation).toBeGreaterThan(40);

      goodImage.delete();
    });
  });
});

