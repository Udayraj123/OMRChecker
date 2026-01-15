/**
 * Tests for base detection classes
 * Port of Python's src/tests/test_detection_base.py concepts
 */

import { describe, it, expect } from 'vitest';
import { FieldDetection, TextDetection } from '../base';

describe('TextDetection', () => {
  it('should create text detection with all properties', () => {
    const detection = new TextDetection(
      'Hello World',
      [10, 20, 100, 50],
      { angle: 0, center: [55, 35] },
      0.95
    );

    expect(detection.detectedText).toBe('Hello World');
    expect(detection.boundingBox).toEqual([10, 20, 100, 50]);
    expect(detection.rotatedRectangle).toEqual({ angle: 0, center: [55, 35] });
    expect(detection.confidentScore).toBe(0.95);
  });

  it('should detect null text', () => {
    const nullDetection = new TextDetection(
      null,
      [0, 0, 0, 0],
      null,
      0
    );

    expect(nullDetection.isNull()).toBe(true);
  });

  it('should detect non-null text', () => {
    const validDetection = new TextDetection(
      'ABC',
      [10, 20, 30, 40],
      null,
      0.8
    );

    expect(validDetection.isNull()).toBe(false);
  });
});

describe('FieldDetection', () => {
  // Create a concrete implementation for testing the abstract class
  class TestFieldDetection extends FieldDetection {
    public detectionRan = false;

    protected runDetection(_field: any, _grayImage: any, _coloredImage: any): void {
      this.detectionRan = true;
    }
  }

  it('should call runDetection in constructor', () => {
    const mockField = { id: 'Q1', label: 'Question 1' };
    const mockGrayImage = { rows: 100, cols: 100 };
    const mockColoredImage = { rows: 100, cols: 100 };

    const detection = new TestFieldDetection(mockField, mockGrayImage, mockColoredImage);

    expect(detection.detectionRan).toBe(true);
  });

  it('should store field, grayImage, and coloredImage', () => {
    const mockField = { id: 'Q2', label: 'Question 2' };
    const mockGrayImage = { rows: 200, cols: 200 };
    const mockColoredImage = { rows: 200, cols: 200 };

    const detection = new TestFieldDetection(mockField, mockGrayImage, mockColoredImage);

    expect(detection['field']).toBe(mockField);
    expect(detection['grayImage']).toBe(mockGrayImage);
    expect(detection['coloredImage']).toBe(mockColoredImage);
  });

  it('should require runDetection implementation', () => {
    // This test verifies that FieldDetection is abstract
    // TypeScript will enforce this at compile time
    expect(() => {
      // @ts-expect-error - Cannot create instance of abstract class
      new FieldDetection({}, {}, {});
    }).toThrow();
  });
});

