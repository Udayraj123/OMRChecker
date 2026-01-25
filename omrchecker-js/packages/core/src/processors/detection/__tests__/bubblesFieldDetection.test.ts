/**
 * Tests for BubblesFieldDetection
 * Port of Python's src/tests/test_bubbles_threshold_detection.py
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { BubblesFieldDetection } from '../bubbles_threshold';
import {
  BubbleFieldDetectionResult,
  BubbleMeanValue,
} from '../models';
import { BubbleField, BubblesScanBox } from '../../layout/field/bubbleField';
import { FieldBlock } from '../../layout/fieldBlock/base';
const cv = global.cv;

/**
 * Helper to create a mock BubbleField for testing.
 */
function createMockBubbleField(
  fieldLabel: string,
  bubbleValues: string[],
  origin: [number, number] = [0, 0],
  bubbleDimensions: [number, number] = [20, 20],
  bubblesGap: number = 30
): BubbleField {
  const fieldBlockConfig = {
    direction: 'horizontal' as const,
    emptyValue: '',
    fieldDetectionType: 'BUBBLES_THRESHOLD' as const,
    fieldLabels: [fieldLabel],
    labelsGap: 0,
    origin,
    bubbleDimensions,
    bubbleValues,
    bubblesGap,
    bubbleFieldType: 'STANDARD',
  };

  const fieldBlock = new FieldBlock('TestBlock', fieldBlockConfig, [0, 0]);

  // The FieldBlock constructor generates fields automatically
  // Return the first field
  return fieldBlock.fields[0] as BubbleField;
}

describe('BubblesFieldDetection', () => {
  let mockGrayImage: cv.Mat;
  let testField: BubbleField;

  beforeEach(() => {
    // Create a mock grayscale image (100x100, all pixels = 150)
    mockGrayImage = new cv.Mat(100, 100, cv.CV_8UC1, new cv.Scalar(150));

    // Create a test field with 3 bubbles at positions matching old test
    testField = createMockBubbleField('Q1', ['A', 'B', 'C'], [10, 10], [20, 20], 30);
  });

  afterEach(() => {
    mockGrayImage.delete();
  });

  describe('constructor and initialization', () => {
    it('should create detection and run detection automatically', () => {
      const detection = new BubblesFieldDetection(
        testField,
        mockGrayImage
      );

      expect(detection.result).not.toBeNull();
      expect(detection.result).toBeInstanceOf(BubbleFieldDetectionResult);
    });

    it('should store field information', () => {
      const field2 = createMockBubbleField('Q2', ['A', 'B', 'C'], [10, 10]);
      const detection = new BubblesFieldDetection(
        field2,
        mockGrayImage
      );

      const result = detection.getResult();
      expect(result.fieldLabel).toBe('Q2');
    });

    it('should process all bubbles', () => {
      const detection = new BubblesFieldDetection(
        testField,
        mockGrayImage
      );

      const result = detection.getResult();
      expect(result.bubbleMeans).toHaveLength(3);
      // Check backward compatibility property if it exists
      if ((detection as any).fieldBubbleMeans) {
        expect((detection as any).fieldBubbleMeans).toHaveLength(3);
      }
    });
  });

  describe('readBubbleMeanValue static method', () => {
    it('should calculate mean value for uniform region', () => {
      // Get the first bubble from our test field
      const bubble = testField.scanBoxes[0] as BubblesScanBox;

      const meanValue = BubblesFieldDetection.readBubbleMeanValue(
        bubble,
        mockGrayImage
      );

      expect(meanValue).toBeInstanceOf(BubbleMeanValue);
      expect(meanValue.meanValue).toBeCloseTo(150, 1);
      expect(meanValue.unitBubble.label).toBe('A');
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

      // Create fields with bubbles in dark and light regions
      const darkField = createMockBubbleField('Dark', ['A'], [10, 10], [20, 20], 30);
      const lightField = createMockBubbleField('Light', ['B'], [10, 60], [20, 20], 30);

      const darkBubble = darkField.scanBoxes[0] as BubblesScanBox;
      const lightBubble = lightField.scanBoxes[0] as BubblesScanBox;

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
      // Create field with bubble at specific position
      const field = createMockBubbleField('C', ['C'], [30, 40], [15, 15], 30);
      const bubble = field.scanBoxes[0] as BubblesScanBox;

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
        testField,
        mockGrayImage
      );

      const result = detection.getResult();

      expect(result).toBeInstanceOf(BubbleFieldDetectionResult);
      expect(result.fieldLabel).toBe('Q1');
      expect(result.bubbleMeans).toHaveLength(3);
    });

    it('should throw error if detection not run', () => {
      // Create a detection but prevent runDetection from running
      const detection = new BubblesFieldDetection(
        testField,
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
      // Second bubble is at x=40 (10 + 30), y=10, width=20, height=20
      for (let y = 10; y < 30; y++) {
        for (let x = 40; x < 60; x++) {
          realisticImage.ucharPtr(y, x)[0] = 60;
        }
      }

      const detection = new BubblesFieldDetection(
        testField,
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

      // Should have excellent scan quality due to high std deviation (marked vs unmarked)
      expect(result.scanQuality).toBe('excellent');

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
        testField,
        poorImage
      );

      const result = detection.getResult();

      // Should have poor scan quality due to low std deviation
      expect(result.scanQuality).toBe('poor');
      expect(result.stdDeviation).toBeLessThan(20);

      poorImage.delete();
    });
  });

  describe('edge cases', () => {
    it('should handle empty bubble array', () => {
      const emptyField = createMockBubbleField('Q_EMPTY', [], [0, 0]);
      const detection = new BubblesFieldDetection(
        emptyField,
        mockGrayImage
      );

      const result = detection.getResult();
      expect(result.bubbleMeans).toHaveLength(0);
      expect(result.meanValues).toEqual([]);
      expect(result.stdDeviation).toBe(0);
    });

    it('should handle single bubble', () => {
      const singleField = createMockBubbleField('Q_SINGLE', ['A'], [10, 10]);

      const detection = new BubblesFieldDetection(
        singleField,
        mockGrayImage
      );

      const result = detection.getResult();
      expect(result.bubbleMeans).toHaveLength(1);
      expect(result.stdDeviation).toBe(0);
    });

    it('should handle large number of bubbles', () => {
      // Create a larger image to fit all bubbles
      const largeImage = new cv.Mat(200, 200, cv.CV_8UC1, new cv.Scalar(150));
      
      // Create 20 bubbles that fit within 200x200 image
      // With gap=10 and size=8, 20 bubbles need: 20*10 = 200 pixels
      const manyBubbleValues: string[] = [];
      for (let i = 0; i < 20; i++) {
        manyBubbleValues.push(String.fromCharCode(65 + (i % 26))); // A-Z cycling
      }

      const manyField = createMockBubbleField(
        'Q_MANY',
        manyBubbleValues,
        [0, 0],
        [8, 8],
        10
      );

      const detection = new BubblesFieldDetection(
        manyField,
        largeImage
      );

      const result = detection.getResult();
      expect(result.bubbleMeans).toHaveLength(20);
      
      largeImage.delete();
    });
  });

  describe('backward compatibility', () => {
    it('should populate result bubbleMeans correctly', () => {
      const detection = new BubblesFieldDetection(
        testField,
        mockGrayImage
      );

      expect(detection.result).toBeDefined();
      expect(detection.result!.bubbleMeans).toBeDefined();
      expect(detection.result!.bubbleMeans).toHaveLength(3);
      expect(detection.result!.bubbleMeans[0]).toBeInstanceOf(BubbleMeanValue);
    });
  });
});

