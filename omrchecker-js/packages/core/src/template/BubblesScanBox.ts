/**
 * ScanBox and BubblesScanBox classes for OMRChecker TypeScript port.
 *
 * Migrated from:
 *   src/processors/layout/field/base.py (ScanBox)
 *   src/processors/layout/field/bubble_field.py (BubblesScanBox)
 */

export interface Margins {
  top: number;
  bottom: number;
  left: number;
  right: number;
}

/**
 * The smallest unit in the template layout.
 * Represents a single scan area at a specific (x, y) coordinate.
 *
 * Ported from Python: src/processors/layout/field/base.py::ScanBox
 */
export class ScanBox {
  fieldIndex: number;
  dimensions: [number, number];
  margins: Margins;
  origin: [number, number];
  x: number;
  y: number;
  fieldLabel: string;
  name: string;
  shifts: [number, number];

  constructor(
    fieldIndex: number,
    fieldLabel: string,
    origin: [number, number],
    dimensions: [number, number],
    margins: Margins,
  ) {
    this.fieldIndex = fieldIndex;
    this.dimensions = dimensions;
    this.margins = margins;
    this.origin = origin;
    this.x = Math.round(origin[0]);
    this.y = Math.round(origin[1]);
    this.fieldLabel = fieldLabel;
    this.name = `${fieldLabel}_${fieldIndex}`;
    this.shifts = [0, 0];
  }

  resetShifts(): void {
    this.shifts = [0, 0];
  }

  getShiftedPosition(extraShifts: [number, number] = [0, 0]): [number, number] {
    return [
      this.x + this.shifts[0] + extraShifts[0],
      this.y + this.shifts[1] + extraShifts[1],
    ];
  }
}

/**
 * A scan box specifically for bubble detection.
 * Overrides `name` to use the bubble value instead of the field index.
 *
 * Ported from Python: src/processors/layout/field/bubble_field.py::BubblesScanBox
 */
export class BubblesScanBox extends ScanBox {
  bubbleValue: string;
  bubbleDimensions: [number, number];
  bubbleFieldType: string;

  constructor(
    fieldIndex: number,
    fieldLabel: string,
    origin: [number, number],
    dimensions: [number, number],
    margins: Margins,
    bubbleValue: string,
    bubbleFieldType: string,
  ) {
    super(fieldIndex, fieldLabel, origin, dimensions, margins);
    this.bubbleValue = bubbleValue;
    this.bubbleDimensions = dimensions;
    this.bubbleFieldType = bubbleFieldType;
    // Override name: use bubble value instead of field index
    this.name = `${fieldLabel}_${bubbleValue}`;
  }
}
