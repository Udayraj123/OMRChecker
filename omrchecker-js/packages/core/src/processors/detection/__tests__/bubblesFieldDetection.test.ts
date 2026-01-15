/**
 * Tests for BubblesFieldDetection
 * Port of Python's src/tests/test_bubbles_threshold_detection.py
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { BubblesFieldDetection } from '../bubbles_threshold';
import {
  BubbleFieldDetectionResult,
  BubbleMeanValue,
  type BubbleLocation,
} from '../models';
import * as cv from '@techstark/opencv-js';

describe('BubblesFieldDetection', () => {
  let mockGrayImage: cv.Mat;
  let testBubbles: BubbleLocation[];

  beforeEach(() => {
    // Create a mock grayscale image (100x100, all pixels = 150)
    mockGrayImage = new cv.Mat(100, 100, cv.CV_8UC1, new cv.Scalar(150));

    testBubbles = [
      { x: 10, y: 10, width: 20, height: 20, label: 'A' },
      { x: 40, y: 10, width: 20, height: 20, label: 'B' },
      { x: 70, y: 10, width: 20, height: 20, label: 'C' },
    ];
  });

  afterEach(() => {
    mockGrayImage.delete();
  });

  describe('constructor and initialization', () => {
    it('should create detection and run detection automatically', () => {
      const detection = new BubblesFieldDetection(
        'Q1',
        'Question 1',
        testBubbles,
        mockGrayImage
      );

      expect(detection.result).not.toBeNull();
      expect(detection.result).toBeInstanceOf(BubbleFieldDetectionResult);
    });

    it('should store field information', () => {
      const detection = new BubblesFieldDetection(
        'Q2',
        'Question 2',
        testBubbles,
        mockGrayImage
      );

      const result = detection.getResult();
      expect(result.fieldId).toBe('Q2');
      expect(result.fieldLabel).toBe('Question 2');
    });

    it('should process all bubbles', () => {
      const detection = new BubblesFieldDetection(
        'Q1',
        'Question 1',
        testBubbles,
        mockGrayImage
      );

      const result = detection.getResult();
      expect(result.bubbleMeans).toHaveLength(3);
      expect(detection.fieldBubbleMeans).toHaveLength(3);
    });
  });

  describe('readBubbleMeanValue static method', () => {
    it('should calculate mean value for uniform region', () => {
      const bubble: BubbleLocation = {
        x: 10,
        y: 10,
        width: 20,
        height: 20,
        label: 'A',
      };

      const meanValue = BubblesFieldDetection.readBubbleMeanValue(
        bubble,
        mockGrayImage
      );

      expect(meanValue).toBeInstanceOf(BubbleMeanValue);
      expect(meanValue.meanValue).toBeCloseTo(150, 1);
      expect(meanValue.unitBubble).toBe(bubble);
      expect(meanValue.position).toEqual([10, 10]);
    });

    it('should handle different intensity regions', () => {
      // Create image with dark region (top half = 50) and light region (bottom half = 200)
      const testImage = new cv.Mat(100, 100, cv.CV_8UC1);

      // Fill top half with dark pixels
      for (let y = 0; y < 50; y++) {
        for (let x = 0; x < 100; x++) {
          testImage.ucharPtr(y, x)[0] = 50;
        }
      }

      // Fill bottom half with light pixels
      for (let y = 50; y < 100; y++) {
        for (let x = 0; x < 100; x++) {
          testImage.ucharPtr(y, x)[0] = 200;
        }
      }

      const darkBubble: BubbleLocation = {
        x: 10,
        y: 10,
        width: 20,
        height: 20,
        label: 'A',
      };

      const lightBubble: BubbleLocation = {
        x: 10,
        y: 60,
        width: 20,
        height: 20,
        label: 'B',
      };

      const darkMean = BubblesFieldDetection.readBubbleMeanValue(
        darkBubble,
        testImage
      );
      const lightMean = BubblesFieldDetection.readBubbleMeanValue(
        lightBubble,
        testImage
      );

      expect(darkMean.meanValue).toBeCloseTo(50, 1);
      expect(lightMean.meanValue).toBeCloseTo(200, 1);

      testImage.delete();
    });

    it('should calculate correct position', () => {
      const bubble: BubbleLocation = {
        x: 30,
        y: 40,
        width: 15,
        height: 15,
        label: 'C',
      };

      const meanValue = BubblesFieldDetection.readBubbleMeanValue(
        bubble,
        mockGrayImage
      );

      expect(meanValue.position).toEqual([30, 40]);
    });
  });

  describe('getResult method', () => {
    it('should return the detection result', () => {
      const detection = new BubblesFieldDetection(
        'Q1',
        'Question 1',
        testBubbles,
        mockGrayImage
      );

      const result = detection.getResult();

      expect(result).toBeInstanceOf(BubbleFieldDetectionResult);
      expect(result.fieldId).toBe('Q1');
      expect(result.bubbleMeans).toHaveLength(3);
    });

    it('should throw error if detection not run', () => {
      // Create a detection but prevent runDetection from running
      const detection = new BubblesFieldDetection(
        'Q1',
        'Question 1',
        testBubbles,
        mockGrayImage
      );

      // Manually set result to null to simulate detection not run
      detection.result = null;

      expect(() => detection.getResult()).toThrow('Detection not run');
    });
  });

  describe('integration with realistic data', () => {
    it('should detect marked vs unmarked bubbles correctly', () => {
      // Create image with one marked bubble (dark) and others unmarked (light)
      const realisticImage = new cv.Mat(100, 100, cv.CV_8UC1, new cv.Scalar(200));

      // Mark the second bubble region as dark (marked)
      for (let y = 10; y < 30; y++) {
        for (let x = 40; x < 60; x++) {
          realisticImage.ucharPtr(y, x)[0] = 60;
        }
      }

      const detection = new BubblesFieldDetection(
        'Q1',
        'Question 1',
        testBubbles,
        realisticImage
      );

      const result = detection.getResult();
      const meanValues = result.meanValues;

      // First bubble (unmarked) should have high mean (~200)
      expect(meanValues[0]).toBeGreaterThan(150);

      // Second bubble (marked) should have low mean (~60)
      expect(meanValues[1]).toBeLessThan(100);

      // Third bubble (unmarked) should have high mean (~200)
      expect(meanValues[2]).toBeGreaterThan(150);

      // Should have good scan quality due to high std deviation
      expect(result.scanQuality).toBe('GOOD');

      realisticImage.delete();
    });

    it('should handle poor quality scans (all similar values)', () => {
      // All bubbles have similar values (poor quality scan)
      const poorImage = new cv.Mat(100, 100, cv.CV_8UC1);

      // Fill with slightly varying values (140-160 range)
      for (let y = 0; y < 100; y++) {
        for (let x = 0; x < 100; x++) {
          poorImage.ucharPtr(y, x)[0] = 145 + (x % 10);
        }
      }

      const detection = new BubblesFieldDetection(
        'Q_POOR',
        'Poor Quality',
        testBubbles,
        poorImage
      );

      const result = detection.getResult();

      // Should have poor scan quality due to low std deviation
      expect(result.scanQuality).toBe('POOR');
      expect(result.stdDeviation).toBeLessThan(20);

      poorImage.delete();
    });
  });

  describe('edge cases', () => {
    it('should handle empty bubble array', () => {
      const detection = new BubblesFieldDetection(
        'Q_EMPTY',
        'Empty',
        [],
        mockGrayImage
      );

      const result = detection.getResult();
      expect(result.bubbleMeans).toHaveLength(0);
      expect(result.meanValues).toEqual([]);
      expect(result.stdDeviation).toBe(0);
    });

    it('should handle single bubble', () => {
      const singleBubble = [testBubbles[0]];

      const detection = new BubblesFieldDetection(
        'Q_SINGLE',
        'Single',
        singleBubble,
        mockGrayImage
      );

      const result = detection.getResult();
      expect(result.bubbleMeans).toHaveLength(1);
      expect(result.stdDeviation).toBe(0);
    });

    it('should handle large number of bubbles', () => {
      const manyBubbles: BubbleLocation[] = [];
      for (let i = 0; i < 50; i++) {
        manyBubbles.push({
          x: (i % 10) * 10,
          y: Math.floor(i / 10) * 10,
          width: 8,
          height: 8,
          label: String.fromCharCode(65 + (i % 26)), // A-Z cycling
        });
      }

      const detection = new BubblesFieldDetection(
        'Q_MANY',
        'Many Bubbles',
        manyBubbles,
        mockGrayImage
      );

      const result = detection.getResult();
      expect(result.bubbleMeans).toHaveLength(50);
    });
  });

  describe('backward compatibility', () => {
    it('should populate fieldBubbleMeans for backward compatibility', () => {
      const detection = new BubblesFieldDetection(
        'Q1',
        'Question 1',
        testBubbles,
        mockGrayImage
      );

      expect(detection.fieldBubbleMeans).toBeDefined();
      expect(detection.fieldBubbleMeans).toHaveLength(3);
      expect(detection.fieldBubbleMeans[0]).toBeInstanceOf(BubbleMeanValue);
    });
  });
});

