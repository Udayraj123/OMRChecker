/**
 * Field drawing classes.
 *
 * TypeScript port of src/processors/layout/field/field_drawing.py
 * Handles visualization of fields and scan boxes.
 */

import cv from '../../../utils/opencv';
import { CLR_BLACK } from '../../../utils/constants';
import { DrawingUtils } from '../../../utils/drawing';
import type { Field } from './base';

/**
 * Base class for field drawing.
 */
export class FieldDrawing {
  protected field: Field;

  constructor(field: Field) {
    this.field = field;
  }

  /**
   * Draw all scan boxes for this field.
   */
  drawScanBoxes(
    markedImage: cv.Mat,
    shifts: [number, number],
    thicknessFactor: number,
    border: cv.Vec3 | number
  ): void {
    FieldDrawing.drawScanBoxesUtil(
      this.field,
      markedImage,
      shifts,
      thicknessFactor,
      border
    );
  }

  /**
   * Static utility to draw scan boxes for any field.
   */
  static drawScanBoxesUtil(
    field: Field,
    markedImage: cv.Mat,
    shifts: [number, number],
    thicknessFactor: number,
    border: cv.Vec3 | number
  ): void {
    const scanBoxes = field.scanBoxes;
    for (const unitBubble of scanBoxes) {
      const shiftedPosition = unitBubble.getShiftedPosition(shifts);
      const dimensions = unitBubble.dimensions;
      // Convert border from Vec3 to ColorTuple, or use default color if number
      const color: [number, number, number] =
        typeof border === 'number'
          ? CLR_BLACK
          : ([border[0], border[1], border[2]] as [number, number, number]);
      const borderThickness = typeof border === 'number' ? border : 3;
      DrawingUtils.drawBox(
        markedImage,
        shiftedPosition,
        dimensions,
        color,
        'BOX_HOLLOW',
        thicknessFactor,
        borderThickness
      );
    }
  }
}

/**
 * Drawing class for bubble fields.
 */
export class BubbleFieldDrawing extends FieldDrawing {
  // Inherits all functionality from FieldDrawing
  // Can be extended with bubble-specific drawing if needed
}

/**
 * Drawing class for OCR fields.
 * TODO: Implement custom drawing of the layout for OCR fields
 */
export class OCRFieldDrawing extends FieldDrawing {
  // TODO: Implement custom drawing
}

/**
 * Drawing class for barcode fields.
 * TODO: Implement custom drawing of the layout for barcode fields
 */
export class BarcodeFieldDrawing extends FieldDrawing {
  // TODO: Implement custom drawing
}

