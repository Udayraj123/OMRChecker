import cv from '@techstark/opencv-js';
import { MathUtils, Point } from './math';
import { ImageProcessingError } from './exceptions';
import { CLR_BLACK, CLR_DARK_GRAY, CLR_GRAY, CLR_GREEN, TEXT_SIZE } from './constants';

export { CLR_BLACK, CLR_DARK_GRAY, CLR_GRAY, CLR_GREEN, TEXT_SIZE };

type BoxStyle = 'BOX_HOLLOW' | 'BOX_FILLED';

interface TextPositionFn {
  (sizeX: number, sizeY: number): Point;
}

export class DrawingUtils {
  static drawMatches(
    image: cv.Mat,
    fromPoints: Point[],
    warpedImage: cv.Mat,
    toPoints: Point[]
  ): cv.Mat {
    // Create horizontal stack - simplified for browser (no padding helper)
    const result = new cv.Mat();
    const imageMats = new cv.MatVector();
    imageMats.push_back(image);
    imageMats.push_back(warpedImage);
    cv.hconcat(imageMats, result);
    imageMats.delete();

    const w = image.cols;
    const fromTuples = MathUtils.getTuplePoints(fromPoints);
    const toTuples = MathUtils.getTuplePoints(toPoints);

    for (let i = 0; i < Math.min(fromTuples.length, toTuples.length); i++) {
      const fromPoint = new cv.Point(fromTuples[i][0], fromTuples[i][1]);
      const toPoint = new cv.Point(w + toTuples[i][0], toTuples[i][1]);
      cv.line(result, fromPoint, toPoint, CLR_GREEN, 3);
    }

    return result;
  }

  static drawBoxDiagonal(
    image: cv.Mat,
    position: Point,
    positionDiagonal: Point,
    color: [number, number, number] = CLR_DARK_GRAY,
    border: number = 3
  ): void {
    const pt1 = new cv.Point(Math.floor(position[0]), Math.floor(position[1]));
    const pt2 = new cv.Point(Math.floor(positionDiagonal[0]), Math.floor(positionDiagonal[1]));
    cv.rectangle(image, pt1, pt2, color, border);
  }

  static drawContour(
    image: cv.Mat,
    contour: Point[],
    color: [number, number, number] = CLR_GREEN,
    thickness: number = 2
  ): void {
    if (contour.some((pt) => pt === null || pt === undefined)) {
      throw new ImageProcessingError('Invalid contour provided', {
        contour: JSON.stringify(contour),
      });
    }

    const contourMat = cv.matFromArray(contour.length, 1, cv.CV_32SC2, contour.flat());
    const contours = new cv.MatVector();
    contours.push_back(contourMat);
    cv.drawContours(image, contours, -1, color, thickness);
    
    contours.delete();
    contourMat.delete();
  }

  static drawBox(
    image: cv.Mat,
    position: Point,
    boxDimensions: [number, number],
    color?: [number, number, number],
    style: BoxStyle = 'BOX_HOLLOW',
    thicknessFactor: number = 1 / 12,
    border: number = 3,
    centered: boolean = false
  ): { position: Point; positionDiagonal: Point } {
    const [x, y] = position;
    const [boxW, boxH] = boxDimensions;

    let pos: Point = [
      Math.floor(x + boxW * thicknessFactor),
      Math.floor(y + boxH * thicknessFactor),
    ];
    let posDiag: Point = [
      Math.floor(x + boxW - boxW * thicknessFactor),
      Math.floor(y + boxH - boxH * thicknessFactor),
    ];

    if (centered) {
      const centeredPosition: Point = [
        Math.floor((3 * pos[0] - posDiag[0]) / 2),
        Math.floor((3 * pos[1] - posDiag[1]) / 2),
      ];
      const centeredDiagonal: Point = [
        Math.floor((pos[0] + posDiag[0]) / 2),
        Math.floor((pos[1] + posDiag[1]) / 2),
      ];
      pos = centeredPosition;
      posDiag = centeredDiagonal;
    }

    let finalColor = color;
    let finalBorder = border;

    if (style === 'BOX_HOLLOW') {
      finalColor = color ?? CLR_GRAY;
    } else if (style === 'BOX_FILLED') {
      finalColor = color ?? CLR_DARK_GRAY;
      finalBorder = -1;
    }

    this.drawBoxDiagonal(image, pos, posDiag, finalColor!, finalBorder);
    return { position: pos, positionDiagonal: posDiag };
  }

  static drawArrows(
    image: cv.Mat,
    startPoints: Point[],
    endPoints: Point[],
    color: [number, number, number] = CLR_GREEN,
    thickness: number = 2,
    lineType: number = cv.LINE_AA,
    tipLength: number = 0.1
  ): cv.Mat {
    const startTuples = MathUtils.getTuplePoints(startPoints);
    const endTuples = MathUtils.getTuplePoints(endPoints);

    for (let i = 0; i < Math.min(startTuples.length, endTuples.length); i++) {
      const start = new cv.Point(startTuples[i][0], startTuples[i][1]);
      const end = new cv.Point(endTuples[i][0], endTuples[i][1]);
      cv.arrowedLine(image, start, end, color, thickness, lineType, 0, tipLength);
    }

    return image;
  }

  static drawTextResponsive(
    image: cv.Mat,
    text: string,
    position: Point,
    textSize: number = TEXT_SIZE,
    thickness: number = 2,
    centered: boolean = false,
    color: [number, number, number] = CLR_BLACK,
    lineType: number = cv.LINE_AA,
    fontFace: number = cv.FONT_HERSHEY_SIMPLEX
  ): void {
    const h = image.rows;
    const w = image.cols;

    const textPosition: TextPositionFn = (sizeX: number, sizeY: number): Point => [
      position[0] - Math.max(0, position[0] + sizeX - w),
      position[1] - Math.max(0, position[1] + sizeY - h),
    ];

    this.drawText(image, text, textPosition, textSize, thickness, centered, color, lineType, fontFace);
  }

  static drawText(
    image: cv.Mat,
    textValue: string,
    position: Point | TextPositionFn,
    textSize: number = TEXT_SIZE,
    thickness: number = 2,
    centered: boolean = false,
    color: [number, number, number] = CLR_BLACK,
    lineType: number = cv.LINE_AA,
    fontFace: number = cv.FONT_HERSHEY_SIMPLEX
  ): void {
    let finalPosition: Point | TextPositionFn = position;

    if (centered) {
      if (typeof position === 'function') {
        throw new ImageProcessingError(`centered=${centered} but position is a callable`, {
          centered,
          position: position.toString(),
        });
      }
      const textPosition = position;

      finalPosition = (sizeX: number, sizeY: number): Point => [
        textPosition[0] - Math.floor(sizeX / 2),
        textPosition[1] + Math.floor(sizeY / 2),
      ];
    }

    if (typeof finalPosition === 'function') {
      const textSizeResult = cv.getTextSize(textValue, fontFace, textSize, thickness);
      finalPosition = finalPosition(textSizeResult.size.width, textSizeResult.size.height);
    }

    const pt = new cv.Point(Math.floor(finalPosition[0]), Math.floor(finalPosition[1]));
    cv.putText(image, textValue, pt, fontFace, textSize, color, thickness, lineType);
  }

  static drawSymbol(
    image: cv.Mat,
    symbol: string,
    position: Point,
    positionDiagonal: Point,
    color: [number, number, number] = CLR_BLACK
  ): void {
    const centerPosition: TextPositionFn = (sizeX: number, sizeY: number): Point => [
      Math.floor((position[0] + positionDiagonal[0] - sizeX) / 2),
      Math.floor((position[1] + positionDiagonal[1] + sizeY) / 2),
    ];

    this.drawText(image, symbol, centerPosition, TEXT_SIZE, 2, false, color);
  }

  static drawLine(
    image: cv.Mat,
    start: Point,
    end: Point,
    color: [number, number, number] = CLR_BLACK,
    thickness: number = 3
  ): void {
    const pt1 = new cv.Point(start[0], start[1]);
    const pt2 = new cv.Point(end[0], end[1]);
    cv.line(image, pt1, pt2, color, thickness);
  }

  static drawPolygon(
    image: cv.Mat,
    points: Point[],
    color: [number, number, number] = CLR_BLACK,
    thickness: number = 1,
    closed: boolean = true
  ): void {
    const n = points.length;
    for (let i = 0; i < n; i++) {
      if (!closed && i === n - 1) {
        continue;
      }
      this.drawLine(image, points[i % n], points[(i + 1) % n], color, thickness);
    }
  }

  static drawGroup(
    image: cv.Mat,
    start: Point,
    bubbleDimensions: [number, number],
    boxEdge: 'TOP' | 'RIGHT' | 'BOTTOM' | 'LEFT',
    color: [number, number, number],
    thickness: number = 3,
    thicknessFactor: number = 7 / 10
  ): void {
    const [startX, startY] = start;
    const [boxW, boxH] = bubbleDimensions;

    let startPos: Point;
    let endPos: Point;

    if (boxEdge === 'TOP') {
      endPos = [startX + Math.floor(boxW * thicknessFactor), startY];
      startPos = [startX + Math.floor(boxW * (1 - thicknessFactor)), startY];
      this.drawLine(image, startPos, endPos, color, thickness);
    } else if (boxEdge === 'RIGHT') {
      startPos = [startX + boxW, startY];
      endPos = [startX, Math.floor(startY + boxH * thicknessFactor)];
      startPos = [startX, Math.floor(startY + boxH * (1 - thicknessFactor))];
      this.drawLine(image, startPos, endPos, color, thickness);
    } else if (boxEdge === 'BOTTOM') {
      startPos = [startX, startY + boxH];
      endPos = [Math.floor(startX + boxW * thicknessFactor), startY];
      startPos = [Math.floor(startX + boxW * (1 - thicknessFactor)), startY];
      this.drawLine(image, startPos, endPos, color, thickness);
    } else if (boxEdge === 'LEFT') {
      endPos = [startX, Math.floor(startY + boxH * thicknessFactor)];
      startPos = [startX, Math.floor(startY + boxH * (1 - thicknessFactor))];
      this.drawLine(image, startPos, endPos, color, thickness);
    }
  }
}
