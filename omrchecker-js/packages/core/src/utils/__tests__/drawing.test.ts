/**
 * Tests for drawing utilities.
 *
 * Tests visualization functions for OMR results.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { DrawingUtils, CLR_GREEN, CLR_RED, TEXT_SIZE } from '../drawing';
const cv = global.cv;

describe('DrawingUtils', () => {
  let testImage: cv.Mat;

  beforeEach(() => {
    // Create a test image (100x100, grayscale)
    testImage = new cv.Mat(100, 100, cv.CV_8UC3);
    testImage.setTo([255, 255, 255, 255]); // White background
  });

  afterEach(() => {
    testImage.delete();
  });

  describe('drawBox', () => {
    it('should draw a hollow box', () => {
      const position: [number, number] = [10, 10];
      const dimensions: [number, number] = [30, 30];

      const [pos, posDiag] = DrawingUtils.drawBox(
        testImage,
        position,
        dimensions,
        CLR_GREEN,
        'BOX_HOLLOW'
      );

      expect(pos).toBeDefined();
      expect(posDiag).toBeDefined();
      expect(pos[0]).toBeGreaterThan(position[0]);
      expect(pos[1]).toBeGreaterThan(position[1]);
    });

    it('should draw a filled box', () => {
      const position: [number, number] = [20, 20];
      const dimensions: [number, number] = [20, 20];

      const [pos, posDiag] = DrawingUtils.drawBox(
        testImage,
        position,
        dimensions,
        CLR_RED,
        'BOX_FILLED'
      );

      expect(pos).toBeDefined();
      expect(posDiag).toBeDefined();
    });

    it('should handle centered box', () => {
      const position: [number, number] = [50, 50];
      const dimensions: [number, number] = [20, 20];

      const [pos, posDiag] = DrawingUtils.drawBox(
        testImage,
        position,
        dimensions,
        undefined,
        'BOX_HOLLOW',
        1 / 12,
        3,
        true // centered
      );

      expect(pos).toBeDefined();
      expect(posDiag).toBeDefined();
    });
  });

  describe('drawLine', () => {
    it('should draw a line between two points', () => {
      const start: [number, number] = [10, 10];
      const end: [number, number] = [50, 50];

      DrawingUtils.drawLine(testImage, start, end, CLR_GREEN, 2);

      // If no errors, drawing succeeded
      expect(testImage.rows).toBe(100);
      expect(testImage.cols).toBe(100);
    });
  });

  describe('drawText', () => {
    it('should draw text at position', () => {
      const text = 'Test';
      const position: [number, number] = [30, 30];

      DrawingUtils.drawText(testImage, text, position);

      // If no errors, drawing succeeded
      expect(testImage.rows).toBe(100);
    });

    it('should draw centered text', () => {
      const text = 'Center';
      const position: [number, number] = [50, 50];

      DrawingUtils.drawText(testImage, text, position, TEXT_SIZE, 2, true);

      expect(testImage.rows).toBe(100);
    });

    it('should handle position calculator function', () => {
      const text = 'Calc';
      const positionFn = (sizeX: number, sizeY: number): [number, number] => [
        50 - sizeX / 2,
        50 + sizeY / 2,
      ];

      DrawingUtils.drawText(testImage, text, positionFn);

      expect(testImage.rows).toBe(100);
    });
  });

  describe('drawPolygon', () => {
    it('should draw a closed polygon', () => {
      const points = [
        [20, 20],
        [40, 20],
        [40, 40],
        [20, 40],
      ];

      DrawingUtils.drawPolygon(testImage, points, CLR_GREEN, 2, true);

      expect(testImage.rows).toBe(100);
    });

    it('should draw an open polygon', () => {
      const points = [
        [60, 20],
        [80, 20],
        [80, 40],
      ];

      DrawingUtils.drawPolygon(testImage, points, CLR_RED, 2, false);

      expect(testImage.rows).toBe(100);
    });
  });

  describe('drawSymbol', () => {
    it('should draw a symbol centered between points', () => {
      const symbol = '✓';
      const position: [number, number] = [20, 20];
      const positionDiagonal: [number, number] = [40, 40];

      DrawingUtils.drawSymbol(testImage, symbol, position, positionDiagonal, CLR_GREEN);

      expect(testImage.rows).toBe(100);
    });
  });

  describe('drawGroup', () => {
    it('should draw group indicator on TOP edge', () => {
      const start: [number, number] = [10, 10];
      const dimensions: [number, number] = [20, 20];

      DrawingUtils.drawGroup(testImage, start, dimensions, 'TOP', CLR_GREEN);

      expect(testImage.rows).toBe(100);
    });

    it('should draw group indicator on each edge', () => {
      const edges: ('TOP' | 'RIGHT' | 'BOTTOM' | 'LEFT')[] = ['TOP', 'RIGHT', 'BOTTOM', 'LEFT'];

      edges.forEach((edge, index) => {
        const start: [number, number] = [10 + index * 25, 10];
        const dimensions: [number, number] = [20, 20];

        DrawingUtils.drawGroup(testImage, start, dimensions, edge, CLR_GREEN);
      });

      expect(testImage.rows).toBe(100);
    });
  });
});

