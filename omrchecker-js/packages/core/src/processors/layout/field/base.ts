/**
 * Base classes for field layout.
 *
 * TypeScript port of src/processors/layout/field/base.py
 * Defines Field and ScanBox base classes used throughout the detection system.
 */

import type { FieldDetectionTypeValue } from '../../constants';
import { defaultDump } from '../../../utils/parsing';

/**
 * Base class for scan boxes (bubbles, OCR zones, etc.).
 *
 * The smallest unit in the template layout.
 */
export class ScanBox {
  public fieldIndex: number;
  public dimensions: [number, number];
  public margins: { top: number; bottom: number; left: number; right: number };
  public origin: [number, number];
  public x: number;
  public y: number;
  public field: Field;
  public fieldLabel: string;
  public fieldDetectionType: FieldDetectionTypeValue;
  public name: string;
  public shifts: [number, number] = [0, 0];

  constructor(
    fieldIndex: number,
    field: Field,
    origin: [number, number],
    dimensions: [number, number],
    margins: { top: number; bottom: number; left: number; right: number }
  ) {
    this.fieldIndex = fieldIndex;
    this.dimensions = dimensions;
    this.margins = margins;
    this.origin = origin;
    this.x = Math.round(origin[0]);
    this.y = Math.round(origin[1]);
    this.field = field;
    this.fieldLabel = field.fieldLabel;
    this.fieldDetectionType = field.fieldDetectionType;
    this.name = `${this.fieldLabel}_${this.fieldIndex}`;
  }

  /**
   * Reset shifts to [0, 0].
   */
  resetShifts(): void {
    this.shifts = [0, 0];
  }

  /**
   * Get position with shifts applied.
   *
   * @param shifts - Optional shifts to apply. If not provided, uses field_block.shifts.
   * @returns [x, y] position with shifts applied
   */
  getShiftedPosition(shifts?: [number, number]): [number, number] {
    if (shifts === undefined) {
      shifts = this.field.fieldBlock.shifts;
    }
    return [this.x + this.shifts[0] + shifts[0], this.y + this.shifts[1] + shifts[1]];
  }

  /**
   * Serialize to JSON.
   */
  toJSON(): Record<string, unknown> {
    return {
      field_label: this.fieldLabel,
      field_detection_type: this.fieldDetectionType,
      name: this.name,
      x: this.x,
      y: this.y,
      origin: defaultDump(this.origin),
    };
  }

  toString(): string {
    return this.name;
  }
}

/**
 * Field block interface (forward declaration).
 * Defined in fieldBlock/base.ts
 */
export interface FieldBlock {
  name: string;
  shifts: [number, number];
  bubbleDimensions?: [number, number];
  bubbleValues?: string[];
  bubblesGap?: number;
  bubbleFieldType?: string;
}

/**
 * Drawing instance interface (forward declaration).
 * Defined in field/fieldDrawing.ts
 */
export interface FieldDrawing {
  drawScanBoxes(
    markedImage: unknown,
    shifts: [number, number],
    thicknessFactor: number,
    border: unknown
  ): void;
}

/**
 * Abstract base class for all field types.
 *
 * Container for a Field on the OMR i.e. a group of ScanBoxes with a collective field_label.
 */
export abstract class Field {
  public direction: 'horizontal' | 'vertical';
  public emptyValue: string;
  public fieldBlock: FieldBlock;
  public fieldDetectionType: FieldDetectionTypeValue;
  public fieldLabel: string;
  public id: string;
  public name: string;
  public plotBinName: string;
  public origin: [number, number];
  public scanBoxes: ScanBox[] = [];
  public drawing: FieldDrawing;

  constructor(
    direction: 'horizontal' | 'vertical',
    emptyValue: string,
    fieldBlock: FieldBlock,
    fieldDetectionType: FieldDetectionTypeValue,
    fieldLabel: string,
    origin: [number, number]
  ) {
    this.direction = direction;
    this.emptyValue = emptyValue;
    this.fieldBlock = fieldBlock;
    this.fieldDetectionType = fieldDetectionType;
    this.fieldLabel = fieldLabel;
    this.id = `${fieldBlock.name}::${fieldLabel}`;
    this.name = fieldLabel;
    this.plotBinName = fieldLabel;
    this.origin = origin;

    // Child class will populate scan_boxes
    this.setupScanBoxes(fieldBlock);
    this.drawing = this.getDrawingInstance();
  }

  /**
   * Abstract method to populate scan boxes.
   * Must be implemented by subclasses.
   */
  abstract setupScanBoxes(fieldBlock: FieldBlock): void;

  /**
   * Abstract method to get drawing instance.
   * Must be implemented by subclasses.
   */
  abstract getDrawingInstance(): FieldDrawing;

  /**
   * Reset all shifts for all scan boxes.
   */
  resetAllShifts(): void {
    for (const scanBox of this.scanBoxes) {
      scanBox.resetShifts();
    }
  }

  /**
   * Serialize to JSON.
   */
  toJSON(): Record<string, unknown> {
    return {
      id: this.id,
      field_label: this.fieldLabel,
      field_detection_type: this.fieldDetectionType,
      direction: this.direction,
      scan_boxes: this.scanBoxes.map((box) => box.toJSON()),
    };
  }

  toString(): string {
    return this.id;
  }
}

