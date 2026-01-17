/**
 * Field drawing classes.
 *
 * TypeScript port of src/processors/layout/field/field_drawing.py
 * Handles visualization of fields and scan boxes.
 */

import * as cv from '@techstark/opencv-js';
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
    border: cv.Vec3
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
    border: cv.Vec3
  ): void {
    const scanBoxes = field.scanBoxes;
    for (const unitBubble of scanBoxes) {
      const shiftedPosition = unitBubble.getShiftedPosition(shifts);
      const dimensions = unitBubble.dimensions;
      // Convert border from Vec3 to ColorTuple
      const color: [number, number, number] = [border[0], border[1], border[2]];
      DrawingUtils.drawBox(
        markedImage,
        shiftedPosition,
        dimensions,
        color,
        'BOX_HOLLOW',
        thicknessFactor,
        3 // border thickness
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

