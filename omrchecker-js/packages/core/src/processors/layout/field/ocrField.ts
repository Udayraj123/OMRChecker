/**
 * OCR field class (stub for future implementation).
 *
 * TypeScript port of src/processors/layout/field/ocr_field.py
 * Stub implementation - full port deferred to future phase.
 */

import { defaultDump } from '../../../utils/parsing';
import type { FieldBlock } from '../fieldBlock/base';
import { Field, ScanBox } from './base';
import { OCRFieldDrawing, type FieldDrawing } from './fieldDrawing';

/**
 * OCR scan box (stub).
 */
export class OCRScanBox extends ScanBox {
  constructor(
    fieldIndex: number,
    field: OCRField,
    origin: [number, number],
    _scanZone: unknown
  ) {
    // Stub implementation
    super(fieldIndex, field, origin, [0, 0], { top: 0, bottom: 0, left: 0, right: 0 });
    // TODO: Implement OCR scan box setup
  }

  toJSON(): Record<string, unknown> {
    return {
      field_label: this.fieldLabel,
      name: this.name,
      x: this.x,
      y: this.y,
      origin: defaultDump(this.origin),
    };
  }
}

/**
 * OCR field class (stub).
 */
export class OCRField extends Field {
  public scanZone?: unknown;

  constructor(
    direction: 'horizontal' | 'vertical',
    emptyValue: string,
    fieldBlock: FieldBlock,
    fieldDetectionType: 'OCR',
    fieldLabel: string,
    origin: [number, number]
  ) {
    super(direction, emptyValue, fieldBlock, fieldDetectionType, fieldLabel, origin);
    this.scanZone = fieldBlock.scanZone;
  }

  setupScanBoxes(fieldBlock: FieldBlock): void {
    const scanZone = fieldBlock.scanZone;
    const origin = fieldBlock.origin;
    // TODO: support for multiple scan zones per field (grid structure)
    const fieldIndex = 0;
    const scanBox = new OCRScanBox(fieldIndex, this, origin, scanZone);
    this.scanBoxes = [scanBox];
  }

  getDrawingInstance(): FieldDrawing {
    return new OCRFieldDrawing(this);
  }

  toJSON(): Record<string, unknown> {
    return {
      field_label: this.fieldLabel,
      direction: this.direction,
      scan_boxes: this.scanBoxes.map((box) => box.toJSON()),
    };
  }
}

