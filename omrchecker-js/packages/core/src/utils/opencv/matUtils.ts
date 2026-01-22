/**
 * OpenCV Mat utilities for DRY code patterns
 *
 * Provides common OpenCV.js operations to avoid repetitive type conversions
 * Used by image.ts and drawing.ts
 */

import cv from './index';
import type { ColorTuple } from '../constants';

/**
 * Utility class for common OpenCV Mat operations
 * Follows DRY principle by centralizing repetitive conversions
 */
export class MatUtils {
  /**
   * DRY: Color tuple to cv.Scalar conversion (used 30+ times)
   */
  static toScalar(color: ColorTuple): cv.Scalar {
    return new cv.Scalar(...color);
  }

  /**
   * DRY: Point tuple to cv.Point (used 20+ times)
   */
  static toPoint(point: readonly [number, number] | [number, number]): cv.Point {
    return new cv.Point(Math.floor(point[0]), Math.floor(point[1]));
  }

  /**
   * DRY: Size tuple to cv.Size (used 15+ times)
   */
  static toSize(dimensions: readonly [number, number] | [number, number]): cv.Size {
    return new cv.Size(Math.floor(dimensions[0]), Math.floor(dimensions[1]));
  }

  /**
   * DRY: Rect tuple to cv.Rect
   */
  static toRect(
    x: number,
    y: number,
    width: number,
    height: number
  ): cv.Rect {
    return new cv.Rect(
      Math.floor(x),
      Math.floor(y),
      Math.floor(width),
      Math.floor(height)
    );
  }

  /**
   * DRY: Safe Mat deletion (used everywhere)
   */
  static delete(...mats: (cv.Mat | null | undefined)[]): void {
    mats.forEach((mat) => {
      if (mat && !mat.isDeleted()) {
        mat.delete();
      }
    });
  }

  /**
   * DRY: Clone or return original based on condition
   */
  static cloneIfNeeded(mat: cv.Mat, condition: boolean): cv.Mat {
    return condition ? mat.clone() : mat;
  }

  /**
   * DRY: Check if dimensions match (used 5+ times)
   */
  static dimensionsMatch(
    mat: cv.Mat,
    width: number,
    height: number
  ): boolean {
    return mat.cols === width && mat.rows === height;
  }

  /**
   * DRY: Safe Mat empty check
   */
  static isEmpty(mat: cv.Mat | null | undefined): boolean {
    return !mat || mat.empty();
  }

  /**
   * DRY: Get Mat dimensions as tuple
   */
  static getDimensions(mat: cv.Mat): [number, number] {
    return [mat.cols, mat.rows];
  }

  /**
   * DRY: Get Mat shape as tuple [height, width]
   */
  static getShape(mat: cv.Mat): [number, number] {
    return [mat.rows, mat.cols];
  }

  /**
   * DRY: Create Mat from dimensions
   */
  static createMat(
    width: number,
    height: number,
    type: number = cv.CV_8UC3
  ): cv.Mat {
    return new cv.Mat(height, width, type);
  }

  /**
   * DRY: Check if Mat is valid (non-null, not deleted, not empty)
   */
  static isValid(mat: cv.Mat | null | undefined): mat is cv.Mat {
    return !!mat && !mat.isDeleted() && !mat.empty();
  }

  /**
   * DRY: Convert array of points to cv.Mat for contour operations
   */
  static pointsToMat(points: Array<readonly [number, number]>): cv.Mat {
    const matVector = new cv.MatVector();
    const ptsMat = cv.matFromArray(
      points.length,
      1,
      cv.CV_32SC2,
      points.flat()
    );
    matVector.push_back(ptsMat);
    return ptsMat;
  }

  /**
   * DRY: Safe try-finally for Mat operations
   */
  static withMats<T>(
    mats: cv.Mat[],
    fn: () => T
  ): T {
    try {
      return fn();
    } finally {
      MatUtils.delete(...mats);
    }
  }
}

