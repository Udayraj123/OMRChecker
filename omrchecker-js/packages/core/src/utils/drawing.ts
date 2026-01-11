/**
 * Drawing utilities for OMRChecker
 *
 * TypeScript port of src/utils/drawing.py
 * Uses DRY patterns with shared position/text calculation helpers
 * All OpenCV conversions use matUtils for consistency
 */

import cv from '@techstark/opencv-js';
import { MatUtils } from './opencv/matUtils';
import {
  CLR_BLACK,
  CLR_GRAY,
  CLR_DARK_GRAY,
  CLR_GREEN,
  TEXT_SIZE,
  type ColorTuple,
} from './constants';

/**
 * Drawing utility class with DRY patterns
 * Shared calculation methods are private and reused by public methods
 */
export class DrawingUtils {
  // ==================== BOX DRAWING (DRY Pattern) ====================

  /**
   * DRY: Calculate box positions (used by drawBox and drawSymbol)
   * @private
   */
  private static calculateBoxPositions(
    position: [number, number],
    dimensions: [number, number],
    thicknessFactor: number,
    centered: boolean
  ): { pos: [number, number]; posDiag: [number, number] } {
    const [x, y] = position;
    const [boxW, boxH] = dimensions;

    let pos: [number, number] = [
      Math.floor(x + boxW * thicknessFactor),
      Math.floor(y + boxH * thicknessFactor),
    ];
    let posDiag: [number, number] = [
      Math.floor(x + boxW - boxW * thicknessFactor),
      Math.floor(y + boxH - boxH * thicknessFactor),
    ];

    if (centered) {
      pos = [
        Math.floor((3 * pos[0] - posDiag[0]) / 2),
        Math.floor((3 * pos[1] - posDiag[1]) / 2),
      ];
      posDiag = [
        Math.floor((pos[0] + posDiag[0]) / 2),
        Math.floor((pos[1] + posDiag[1]) / 2),
      ];
    }

    return { pos, posDiag };
  }

  /**
   * Draw a rectangle using diagonal corners
   */
  static drawBoxDiagonal(
    mat: cv.Mat,
    position: [number, number],
    positionDiagonal: [number, number],
    color: ColorTuple = CLR_DARK_GRAY,
    border = 3
  ): void {
    cv.rectangle(
      mat,
      MatUtils.toPoint(position),
      MatUtils.toPoint(positionDiagonal),
      MatUtils.toScalar(color),
      border
    );
  }

  /**
   * Draw a box with styling options
   */
  static drawBox(
    mat: cv.Mat,
    position: [number, number],
    dimensions: [number, number],
    color?: ColorTuple,
    style: 'BOX_HOLLOW' | 'BOX_FILLED' = 'BOX_HOLLOW',
    thicknessFactor = 1 / 12,
    border = 3,
    centered = false
  ): { pos: [number, number]; posDiag: [number, number] } {
    const { pos, posDiag } = this.calculateBoxPositions(
      position,
      dimensions,
      thicknessFactor,
      centered
    );

    const finalColor =
      color || (style === 'BOX_HOLLOW' ? CLR_GRAY : CLR_DARK_GRAY);
    const finalBorder = style === 'BOX_FILLED' ? -1 : border;

    this.drawBoxDiagonal(mat, pos, posDiag, finalColor, finalBorder);

    return { pos, posDiag };
  }

  // ==================== TEXT DRAWING (DRY Pattern) ====================

  /**
   * DRY: Calculate text position (used by 3 text methods)
   * @private
   */
  private static calculateTextPosition(
    text: string,
    basePosition: [number, number],
    textSize: number,
    thickness: number,
    fontFace: number,
    centered: boolean
  ): [number, number] {
    const baseline = [0]; // Output parameter for baseline
    const textDims = cv.getTextSize(text, fontFace, textSize, thickness, baseline);
    const sizeX = textDims.width;
    const sizeY = textDims.height;

    if (centered) {
      return [
        basePosition[0] - Math.floor(sizeX / 2),
        basePosition[1] + Math.floor(sizeY / 2),
      ];
    }

    return basePosition;
  }

  /**
   * Draw text on image
   */
  static drawText(
    mat: cv.Mat,
    text: string,
    position: [number, number],
    textSize = TEXT_SIZE,
    thickness = 2,
    centered = false,
    color: ColorTuple = CLR_BLACK,
    lineType = cv.LINE_AA,
    fontFace = cv.FONT_HERSHEY_SIMPLEX
  ): void {
    const finalPos = this.calculateTextPosition(
      text,
      position,
      textSize,
      thickness,
      fontFace,
      centered
    );

    cv.putText(
      mat,
      text,
      MatUtils.toPoint(finalPos),
      fontFace,
      textSize,
      MatUtils.toScalar(color),
      thickness,
      lineType
    );
  }

  /**
   * Draw text that stays within image boundaries
   */
  static drawTextResponsive(
    mat: cv.Mat,
    text: string,
    position: [number, number],
    textSize = TEXT_SIZE,
    thickness = 2,
    color: ColorTuple = CLR_BLACK,
    lineType = cv.LINE_AA,
    fontFace = cv.FONT_HERSHEY_SIMPLEX
  ): void {
    const [h, w] = [mat.rows, mat.cols];
    const baseline = [0];
    const textDims = cv.getTextSize(text, fontFace, textSize, thickness, baseline);
    const sizeX = textDims.width;
    const sizeY = textDims.height;

    // Adjust position to stay within bounds
    const adjustedPos: [number, number] = [
      position[0] - Math.max(0, position[0] + sizeX - w),
      position[1] - Math.max(0, position[1] + sizeY - h),
    ];

    cv.putText(
      mat,
      text,
      MatUtils.toPoint(adjustedPos),
      fontFace,
      textSize,
      MatUtils.toScalar(color),
      thickness,
      lineType
    );
  }

  // ==================== SHAPE DRAWING (DRY Pattern) ====================

  /**
   * DRY: Convert array of points (used by 4 methods)
   * @private
   */
  private static convertPoints(
    points: Array<[number, number]>
  ): cv.Point[] {
    return points.map((p) => MatUtils.toPoint(p));
  }

  /**
   * Draw a line between two points
   */
  static drawLine(
    mat: cv.Mat,
    start: [number, number],
    end: [number, number],
    color: ColorTuple = CLR_BLACK,
    thickness = 1,
    lineType = cv.LINE_AA
  ): void {
    cv.line(
      mat,
      MatUtils.toPoint(start),
      MatUtils.toPoint(end),
      MatUtils.toScalar(color),
      thickness,
      lineType
    );
  }

  /**
   * Draw a circle
   */
  static drawCircle(
    mat: cv.Mat,
    center: [number, number],
    radius: number,
    color: ColorTuple = CLR_BLACK,
    thickness = 1,
    lineType = cv.LINE_AA
  ): void {
    cv.circle(
      mat,
      MatUtils.toPoint(center),
      Math.floor(radius),
      MatUtils.toScalar(color),
      thickness,
      lineType
    );
  }

  /**
   * Draw a polygon (closed or open)
   */
  static drawPolygon(
    mat: cv.Mat,
    points: Array<[number, number]>,
    color: ColorTuple = CLR_BLACK,
    thickness = 1,
    closed = true
  ): void {
    const n = points.length;

    for (let i = 0; i < n; i++) {
      if (!closed && i === n - 1) continue;
      this.drawLine(mat, points[i], points[(i + 1) % n], color, thickness);
    }
  }

  /**
   * Draw contour on image
   */
  static drawContour(
    mat: cv.Mat,
    contour: Array<[number, number]>,
    color: ColorTuple = CLR_GREEN,
    thickness = 2
  ): void {
    // Create contour as MatVector
    const points = new cv.MatVector();
    const contourMat = cv.matFromArray(
      contour.length,
      1,
      cv.CV_32SC2,
      contour.flat()
    );
    points.push_back(contourMat);

    cv.drawContours(
      mat,
      points,
      0, // contourIdx
      MatUtils.toScalar(color),
      thickness
    );

    // Clean up
    contourMat.delete();
    points.delete();
  }

  /**
   * Draw arrows between points
   */
  static drawArrows(
    mat: cv.Mat,
    startPoints: Array<[number, number]>,
    endPoints: Array<[number, number]>,
    color: ColorTuple = CLR_GREEN,
    thickness = 2,
    lineType = cv.LINE_AA,
    tipLength = 0.1
  ): void {
    const starts = this.convertPoints(startPoints);
    const ends = this.convertPoints(endPoints);

    for (let i = 0; i < Math.min(starts.length, ends.length); i++) {
      cv.arrowedLine(
        mat,
        starts[i],
        ends[i],
        MatUtils.toScalar(color),
        thickness,
        lineType,
        0, // shift
        tipLength
      );
    }
  }

  /**
   * Draw matches between two images (for alignment visualization)
   */
  static drawMatches(
    image1: cv.Mat,
    fromPoints: Array<[number, number]>,
    image2: cv.Mat,
    toPoints: Array<[number, number]>,
    matchColor: ColorTuple = CLR_GREEN,
    thickness = 3
  ): cv.Mat {
    // Stack images horizontally
    const matVector = new cv.MatVector();
    matVector.push_back(image1);
    matVector.push_back(image2);
    const stacked = new cv.Mat();
    cv.hconcat(matVector, stacked);
    matVector.delete();

    // Draw lines between matched points
    const w = image1.cols;
    const adjustedToPoints = toPoints.map(
      ([x, y]) => [w + x, y] as [number, number]
    );

    for (let i = 0; i < Math.min(fromPoints.length, toPoints.length); i++) {
      this.drawLine(
        stacked,
        fromPoints[i],
        adjustedToPoints[i],
        matchColor,
        thickness
      );
    }

    return stacked;
  }

  // ==================== SYMBOL DRAWING ====================

  /**
   * Draw a symbol (cross, tick, etc.) in a box
   */
  static drawSymbol(
    mat: cv.Mat,
    symbol: 'CROSS' | 'TICK' | 'DOT',
    position: [number, number],
    dimensions: [number, number],
    color: ColorTuple = CLR_BLACK,
    thickness = 2
  ): void {
    const { pos, posDiag } = this.calculateBoxPositions(
      position,
      dimensions,
      1 / 12,
      false
    );

    const centerX = Math.floor((pos[0] + posDiag[0]) / 2);
    const centerY = Math.floor((pos[1] + posDiag[1]) / 2);

    switch (symbol) {
      case 'CROSS':
        // Draw X
        this.drawLine(mat, pos, posDiag, color, thickness);
        this.drawLine(mat, [posDiag[0], pos[1]], [pos[0], posDiag[1]], color, thickness);
        break;
      case 'TICK':
        // Draw checkmark
        const midX = Math.floor((pos[0] + centerX) / 2);
        const midY = Math.floor((centerY + posDiag[1]) / 2);
        this.drawLine(mat, [pos[0], centerY], [midX, midY], color, thickness);
        this.drawLine(mat, [midX, midY], [posDiag[0], pos[1]], color, thickness);
        break;
      case 'DOT':
        // Draw filled circle
        const radius = Math.floor(Math.min(dimensions[0], dimensions[1]) / 6);
        this.drawCircle(mat, [centerX, centerY], radius, color, -1);
        break;
    }
  }
}

