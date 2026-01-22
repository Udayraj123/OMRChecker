/**
 * Drawing utilities for OMR visualization.
 *
 * TypeScript port of src/utils/drawing.py
 * Maintains 1:1 correspondence with Python implementation.
 *
 * Provides functions for drawing boxes, text, contours, arrows, and other
 * visual elements on OpenCV Mat images for OMR result visualization.
 */

import cv from './opencv';
import { ImageUtils } from './ImageUtils';
import { MathUtils } from './math';
import {
  CLR_BLACK,
  CLR_WHITE,
  CLR_GRAY,
  CLR_DARK_GRAY,
  CLR_GREEN,
  CLR_RED,
  CLR_BLUE,
  CLR_YELLOW,
  CLR_DARK_BLUE,
  CLR_DARK_GREEN,
  CLR_DARK_RED,
  CLR_NEAR_BLACK,
  CLR_LIGHT_GRAY,
  type ColorTuple,
} from './constants';

// Re-export color constants for convenience
export {
  CLR_BLACK,
  CLR_WHITE,
  CLR_GRAY,
  CLR_DARK_GRAY,
  CLR_GREEN,
  CLR_RED,
  CLR_BLUE,
  CLR_YELLOW,
  CLR_DARK_BLUE,
  CLR_DARK_GREEN,
  CLR_DARK_RED,
  CLR_NEAR_BLACK,
  CLR_LIGHT_GRAY,
};

// Text constants
export const TEXT_SIZE = 0.6;

/**
 * Box drawing styles
 */
export type BoxStyle = 'BOX_HOLLOW' | 'BOX_FILLED';

/**
 * Box edge positions for group drawing
 */
export type BoxEdge = 'TOP' | 'RIGHT' | 'BOTTOM' | 'LEFT';

/**
 * Drawing utilities class providing static methods for visualization.
 */
export class DrawingUtils {
  /**
   * Draw matching lines between two images side by side.
   *
   * @param image - First image
   * @param fromPoints - Points in the first image
   * @param warpedImage - Second image (warped)
   * @param toPoints - Corresponding points in the second image
   * @returns Horizontal stack with lines drawn
   */
  static drawMatches(
    image: cv.Mat,
    fromPoints: number[][]  | [number, number][],
    warpedImage: cv.Mat,
    toPoints: number[][] | [number, number][]
  ): cv.Mat {
    const horizontalStack = ImageUtils.getPaddedHstack([image, warpedImage]);
    const w = image.cols;

    const fromTuples = MathUtils.getTuplePoints(fromPoints as any);
    const toTuples = MathUtils.getTuplePoints(toPoints as any);

    for (let i = 0; i < fromTuples.length; i++) {
      cv.line(
        horizontalStack,
        new cv.Point(fromTuples[i][0], fromTuples[i][1]),
        new cv.Point(w + toTuples[i][0], toTuples[i][1]),
        CLR_GREEN,
        3
      );
    }

    return horizontalStack;
  }

  /**
   * Draw a rectangle using diagonal points.
   *
   * @param image - Image to draw on
   * @param position - Top-left corner
   * @param positionDiagonal - Bottom-right corner
   * @param color - Rectangle color
   * @param border - Border thickness (-1 for filled)
   */
  static drawBoxDiagonal(
    image: cv.Mat,
    position: [number, number],
    positionDiagonal: [number, number],
    color: ColorTuple = CLR_DARK_GRAY,
    border: number = 3
  ): void {
    cv.rectangle(
      image,
      new cv.Point(Math.floor(position[0]), Math.floor(position[1])),
      new cv.Point(Math.floor(positionDiagonal[0]), Math.floor(positionDiagonal[1])),
      color,
      border
    );
  }

  /**
   * Draw a contour on the image.
   *
   * @param image - Image to draw on
   * @param contour - Contour points
   * @param color - Contour color
   * @param thickness - Line thickness
   */
  static drawContour(
    image: cv.Mat,
    contour: cv.Mat | number[][],
    color: ColorTuple = CLR_GREEN,
    thickness: number = 2
  ): void {
    // Convert to MatVector if needed
    const contours = new cv.MatVector();

    if (Array.isArray(contour)) {
      const mat = cv.matFromArray(contour.length, 1, cv.CV_32SC2, contour.flat());
      contours.push_back(mat);
      cv.drawContours(image, contours, 0, color, thickness);
      mat.delete();
    } else {
      contours.push_back(contour);
      cv.drawContours(image, contours, 0, color, thickness);
    }

    contours.delete();
  }

  /**
   * Draw a convex hull around points.
   *
   * @param image - Image to draw on
   * @param points - Points to compute convex hull for
   * @param color - Hull color
   * @param thickness - Line thickness
   */
  static drawConvexHull(
    image: cv.Mat,
    points: cv.Mat | number[][],
    color: ColorTuple = CLR_BLUE,
    thickness: number = 2
  ): void {
    const hull = new cv.Mat();

    if (Array.isArray(points)) {
      const pointsMat = cv.matFromArray(points.length, 1, cv.CV_32SC2, points.flat());
      cv.convexHull(pointsMat, hull, false, true);
      pointsMat.delete();
    } else {
      cv.convexHull(points, hull, false, true);
    }

    this.drawContour(image, hull, color, thickness);
    hull.delete();
  }

  /**
   * Draw a box (rectangle) with various styles.
   *
   * @param image - Image to draw on
   * @param position - Top-left position
   * @param boxDimensions - Box dimensions [width, height]
   * @param color - Box color (optional)
   * @param style - Box style (hollow or filled)
   * @param thicknessFactor - Inset factor for the box
   * @param border - Border thickness
   * @param centered - Whether to center the box
   * @returns Tuple of [position, positionDiagonal]
   */
  static drawBox(
    image: cv.Mat,
    position: [number, number],
    boxDimensions: [number, number],
    color?: ColorTuple,
    style: BoxStyle = 'BOX_HOLLOW',
    thicknessFactor: number = 1 / 12,
    border: number = 3,
    centered: boolean = false
  ): [[number, number], [number, number]] {
    const [x, y] = position;
    const [boxW, boxH] = boxDimensions;

    let pos: [number, number] = [
      Math.floor(x + boxW * thicknessFactor),
      Math.floor(y + boxH * thicknessFactor),
    ];

    let posDiag: [number, number] = [
      Math.floor(x + boxW - boxW * thicknessFactor),
      Math.floor(y + boxH - boxH * thicknessFactor),
    ];

    if (centered) {
      const centeredPos: [number, number] = [
        Math.floor((3 * pos[0] - posDiag[0]) / 2),
        Math.floor((3 * pos[1] - posDiag[1]) / 2),
      ];
      const centeredDiag: [number, number] = [
        Math.floor((pos[0] + posDiag[0]) / 2),
        Math.floor((pos[1] + posDiag[1]) / 2),
      ];
      pos = centeredPos;
      posDiag = centeredDiag;
    }

    let finalColor = color;
    let finalBorder = border;

    if (style === 'BOX_HOLLOW') {
      if (!finalColor) finalColor = CLR_GRAY;
    } else if (style === 'BOX_FILLED') {
      if (!finalColor) finalColor = CLR_DARK_GRAY;
      finalBorder = -1;
    }

    this.drawBoxDiagonal(image, pos, posDiag, finalColor!, finalBorder);
    return [pos, posDiag];
  }

  /**
   * Draw arrows from start points to end points.
   *
   * @param image - Image to draw on
   * @param startPoints - Array of start points
   * @param endPoints - Array of end points
   * @param color - Arrow color
   * @param thickness - Line thickness
   * @param lineType - OpenCV line type
   * @param tipLength - Arrow tip length ratio
   * @returns Modified image
   */
  static drawArrows(
    image: cv.Mat,
    startPoints: number[][] | [number, number][],
    endPoints: number[][] | [number, number][],
    color: ColorTuple = CLR_GREEN,
    thickness: number = 2,
    lineType: number = cv.LINE_AA,
    tipLength: number = 0.1
  ): cv.Mat {
    const startTuples = MathUtils.getTuplePoints(startPoints as any);
    const endTuples = MathUtils.getTuplePoints(endPoints as any);

    for (let i = 0; i < startTuples.length; i++) {
      cv.arrowedLine(
        image,
        new cv.Point(startTuples[i][0], startTuples[i][1]),
        new cv.Point(endTuples[i][0], endTuples[i][1]),
        color,
        thickness,
        lineType,
        0,
        tipLength
      );
    }

    return image;
  }

  /**
   * Draw text that adjusts position to stay within image bounds.
   *
   * @param image - Image to draw on
   * @param text - Text to draw
   * @param position - Desired position
   * @param textSize - Font size
   * @param thickness - Text thickness
   * @param centered - Whether to center text
   * @param color - Text color
   * @param lineType - OpenCV line type
   */
  static drawTextResponsive(
    image: cv.Mat,
    text: string,
    position: [number, number],
    textSize: number = TEXT_SIZE,
    thickness: number = 2,
    centered: boolean = false,
    color: ColorTuple = CLR_BLACK,
    lineType: number = cv.LINE_AA
  ): void {
    const h = image.rows;
    const w = image.cols;

    const textPosition = (sizeX: number, sizeY: number): [number, number] => [
      position[0] - Math.max(0, position[0] + sizeX - w),
      position[1] - Math.max(0, position[1] + sizeY - h),
    ];

    this.drawText(image, text, textPosition, textSize, thickness, centered, color, lineType);
  }

  /**
   * Draw text on the image.
   *
   * @param image - Image to draw on
   * @param textValue - Text to draw
   * @param position - Position or position calculator function
   * @param textSize - Font size
   * @param thickness - Text thickness
   * @param centered - Whether to center text
   * @param color - Text color
   * @param lineType - OpenCV line type
   * @param fontFace - OpenCV font face
   */
  static drawText(
    image: cv.Mat,
    textValue: string,
    position: [number, number] | ((sizeX: number, sizeY: number) => [number, number]),
    textSize: number = TEXT_SIZE,
    thickness: number = 2,
    centered: boolean = false,
    color: ColorTuple = CLR_BLACK,
    lineType: number = cv.LINE_AA,
    fontFace: number = cv.FONT_HERSHEY_SIMPLEX
  ): void {
    let finalPosition: [number, number];

    if (centered && !(typeof position === 'function')) {
      const textPosition = position;
      const textSizeResult = cv.getTextSize(textValue, fontFace, textSize, thickness, 0);
      const sizeX = textSizeResult.width;
      const sizeY = textSizeResult.height;
      finalPosition = [
        textPosition[0] - Math.floor(sizeX / 2),
        textPosition[1] + Math.floor(sizeY / 2),
      ];
    } else if (typeof position === 'function') {
      const textSizeResult = cv.getTextSize(textValue, fontFace, textSize, thickness, 0);
      const sizeX = textSizeResult.width;
      const sizeY = textSizeResult.height;
      finalPosition = position(sizeX, sizeY);
    } else {
      finalPosition = position;
    }

    cv.putText(
      image,
      textValue,
      new cv.Point(Math.floor(finalPosition[0]), Math.floor(finalPosition[1])),
      fontFace,
      textSize,
      color,
      thickness,
      lineType
    );
  }

  /**
   * Draw a symbol centered between two points.
   *
   * @param image - Image to draw on
   * @param symbol - Symbol text to draw
   * @param position - Top-left position
   * @param positionDiagonal - Bottom-right position
   * @param color - Symbol color
   */
  static drawSymbol(
    image: cv.Mat,
    symbol: string,
    position: [number, number],
    positionDiagonal: [number, number],
    color: ColorTuple = CLR_BLACK
  ): void {
    const centerPosition = (sizeX: number, sizeY: number): [number, number] => [
      Math.floor((position[0] + positionDiagonal[0] - sizeX) / 2),
      Math.floor((position[1] + positionDiagonal[1] + sizeY) / 2),
    ];

    this.drawText(image, symbol, centerPosition, TEXT_SIZE, 2, false, color);
  }

  /**
   * Draw a line between two points.
   *
   * @param image - Image to draw on
   * @param start - Start point
   * @param end - End point
   * @param color - Line color
   * @param thickness - Line thickness
   */
  static drawLine(
    image: cv.Mat,
    start: [number, number],
    end: [number, number],
    color: ColorTuple = CLR_BLACK,
    thickness: number = 3
  ): void {
    cv.line(
      image,
      new cv.Point(start[0], start[1]),
      new cv.Point(end[0], end[1]),
      color,
      thickness
    );
  }

  /**
   * Draw a polygon connecting multiple points.
   *
   * @param image - Image to draw on
   * @param points - Array of points
   * @param color - Line color
   * @param thickness - Line thickness
   * @param closed - Whether to close the polygon
   */
  static drawPolygon(
    image: cv.Mat,
    points: number[][],
    color: ColorTuple = CLR_BLACK,
    thickness: number = 1,
    closed: boolean = true
  ): void {
    const n = points.length;
    for (let i = 0; i < n; i++) {
      if (!closed && i === n - 1) continue;

      this.drawLine(
        image,
        [points[i][0], points[i][1]],
        [points[(i + 1) % n][0], points[(i + 1) % n][1]],
        color,
        thickness
      );
    }
  }

  /**
   * Draw a group indicator on a specific edge of a bubble.
   *
   * @param image - Image to draw on
   * @param start - Start position
   * @param bubbleDimensions - Bubble dimensions [width, height]
   * @param boxEdge - Which edge to draw on
   * @param color - Line color
   * @param thickness - Line thickness
   * @param thicknessFactor - Length factor for the indicator
   */
  static drawGroup(
    image: cv.Mat,
    start: [number, number],
    bubbleDimensions: [number, number],
    boxEdge: BoxEdge,
    color: ColorTuple,
    thickness: number = 3,
    thicknessFactor: number = 7 / 10
  ): void {
    const [startX, startY] = start;
    const [boxW, boxH] = bubbleDimensions;

    let lineStart: [number, number];
    let lineEnd: [number, number];

    switch (boxEdge) {
      case 'TOP':
        lineEnd = [startX + Math.floor(boxW * thicknessFactor), startY];
        lineStart = [startX + Math.floor(boxW * (1 - thicknessFactor)), startY];
        break;
      case 'RIGHT':
        lineStart = [startX + boxW, startY + Math.floor(boxH * (1 - thicknessFactor))];
        lineEnd = [startX + boxW, startY + Math.floor(boxH * thicknessFactor)];
        break;
      case 'BOTTOM':
        lineStart = [startX + Math.floor(boxW * (1 - thicknessFactor)), startY + boxH];
        lineEnd = [startX + Math.floor(boxW * thicknessFactor), startY + boxH];
        break;
      case 'LEFT':
        lineStart = [startX, startY + Math.floor(boxH * (1 - thicknessFactor))];
        lineEnd = [startX, startY + Math.floor(boxH * thicknessFactor)];
        break;
    }

    this.drawLine(image, lineStart, lineEnd, color, thickness);
  }
}
