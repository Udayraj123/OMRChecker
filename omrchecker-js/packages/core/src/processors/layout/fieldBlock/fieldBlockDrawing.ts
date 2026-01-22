/**
 * Field block drawing class.
 *
 * TypeScript port of src/processors/layout/field_block/field_block_drawing.py
 * Handles visualization of field blocks.
 */

import cv from '../../../utils/opencv';
import { CLR_BLACK } from '../../../utils/constants';
import { DrawingUtils } from '../../../utils/drawing';
import { FieldBlock } from './base';

/**
 * Drawing class for field blocks.
 */
export class FieldBlockDrawing {
  protected fieldBlock: FieldBlock;

  constructor(fieldBlock: FieldBlock) {
    this.fieldBlock = fieldBlock;
  }

  /**
   * Draw field block on marked image.
   *
   * @param markedImage - Image to draw on
   * @param shifted - Whether to use shifted positions
   * @param thickness - Line thickness for text
   * @param border - Border thickness
   */
  drawFieldBlock(
    markedImage: cv.Mat,
    shifted: boolean = true,
    thickness: number = 3,
    border: number = 3
  ): void {
    const fieldBlock = this.fieldBlock;
    // TODO: get this field block using a bounding box of all bubbles instead. (remove shift at field block level)
    FieldBlockDrawing.drawBoundingRectangle(
      fieldBlock,
      markedImage,
      shifted,
      border
    );

    const thicknessFactor = 1 / 10;
    for (const field of fieldBlock.fields) {
      field.drawing.drawScanBoxes(
        markedImage,
        fieldBlock.shifts,
        thicknessFactor,
        border // Pass as number, FieldDrawing handles both Vec3 and number
      );
    }

    FieldBlockDrawing.drawFieldBlockLabel(
      fieldBlock,
      markedImage,
      shifted,
      thickness
    );
  }

  /**
   * Draw bounding rectangle for field block.
   *
   * @param fieldBlock - Field block to draw
   * @param markedImage - Image to draw on
   * @param shifted - Whether to use shifted positions
   * @param border - Border thickness
   */
  static drawBoundingRectangle(
    fieldBlock: FieldBlock,
    markedImage: cv.Mat,
    shifted: boolean,
    border: number
  ): void {
    const boundingBoxOrigin = fieldBlock.boundingBoxOrigin;
    const boundingBoxDimensions = fieldBlock.boundingBoxDimensions;

    if (!boundingBoxOrigin || !boundingBoxDimensions) {
      return;
    }

    const blockPosition = shifted
      ? fieldBlock.getShiftedOrigin()
      : boundingBoxOrigin;

    if (!shifted) {
      DrawingUtils.drawBox(
        markedImage,
        blockPosition,
        boundingBoxDimensions,
        CLR_BLACK,
        'BOX_HOLLOW',
        0,
        border
      );
    }
  }

  /**
   * Draw field block label.
   *
   * @param fieldBlock - Field block to draw
   * @param markedImage - Image to draw on
   * @param shifted - Whether to use shifted positions
   * @param thickness - Text thickness
   */
  static drawFieldBlockLabel(
    fieldBlock: FieldBlock,
    markedImage: cv.Mat,
    shifted: boolean,
    thickness: number
  ): void {
    const fieldBlockName = fieldBlock.name;
    const boundingBoxOrigin = fieldBlock.boundingBoxOrigin;
    const boundingBoxDimensions = fieldBlock.boundingBoxDimensions;

    if (!boundingBoxOrigin || !boundingBoxDimensions) {
      return;
    }

    const blockPosition = shifted
      ? fieldBlock.getShiftedOrigin()
      : boundingBoxOrigin;

    const textPosition = (sizeX: number, sizeY: number): [number, number] => {
      return [
        Math.floor(blockPosition[0] + boundingBoxDimensions[0] - sizeX),
        Math.floor(blockPosition[1] - sizeY),
      ];
    };

    let text = fieldBlockName;
    if (shifted) {
      text = `(${fieldBlock.shifts[0]},${fieldBlock.shifts[1]})${fieldBlockName}`;
    }

    DrawingUtils.drawText(
      markedImage,
      text,
      textPosition,
      undefined, // textSize - use default
      thickness,
      false, // centered
      CLR_BLACK
    );
  }
}

