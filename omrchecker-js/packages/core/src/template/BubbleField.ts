/**
 * BubbleField class for OMRChecker TypeScript port.
 *
 * Migrated from:
 *   src/processors/layout/field/base.py (Field)
 *   src/processors/layout/field/bubble_field.py (BubbleField)
 */

import { ZERO_MARGINS } from './constants';
import { BubblesScanBox } from './BubblesScanBox';

/**
 * A field representing a group of bubble scan boxes sharing a common field label.
 *
 * Ported from Python:
 *   src/processors/layout/field/base.py::Field
 *   src/processors/layout/field/bubble_field.py::BubbleField
 */
export class BubbleField {
  direction: string;
  emptyValue: string;
  fieldLabel: string;
  id: string;
  name: string;
  origin: [number, number];
  scanBoxes: BubblesScanBox[];
  bubbleDimensions: [number, number];
  bubbleValues: string[];
  bubblesGap: number;
  bubbleFieldType: string;

  constructor(
    direction: string,
    emptyValue: string,
    fieldBlockName: string,
    bubbleDimensions: [number, number],
    bubbleValues: string[],
    bubblesGap: number,
    bubbleFieldType: string,
    fieldLabel: string,
    origin: [number, number],
  ) {
    this.direction = direction;
    this.emptyValue = emptyValue;
    this.fieldLabel = fieldLabel;
    this.id = `${fieldBlockName}::${fieldLabel}`;
    this.name = fieldLabel;
    this.origin = [origin[0], origin[1]];

    this.bubbleDimensions = bubbleDimensions;
    this.bubbleValues = bubbleValues;
    this.bubblesGap = bubblesGap;
    this.bubbleFieldType = bubbleFieldType;

    this.scanBoxes = [];
    this.setupScanBoxes();
  }

  /**
   * Generate scan boxes for each bubble value.
   *
   * Direction determines which axis the bubbles extend along:
   *   vertical   → h=1 → Y axis increments (bubbles stack top-to-bottom)
   *   horizontal → h=0 → X axis increments (bubbles go left-to-right)
   *
   * Ported from Python: src/processors/layout/field/bubble_field.py::BubbleField.setup_scan_boxes
   */
  private setupScanBoxes(): void {
    const { bubbleValues, bubbleDimensions, bubblesGap, bubbleFieldType, direction } = this;

    if (!bubbleValues || bubbleValues.length === 0) {
      throw new Error('bubbleValues is required and must not be empty');
    }

    // h=1 means Y-axis gap (vertical direction), h=0 means X-axis gap (horizontal direction)
    const h = direction === 'vertical' ? 1 : 0;

    const bubblePoint: [number, number] = [this.origin[0], this.origin[1]];

    for (let fieldIndex = 0; fieldIndex < bubbleValues.length; fieldIndex++) {
      const bubbleValue = bubbleValues[fieldIndex];
      const bubbleOrigin: [number, number] = [bubblePoint[0], bubblePoint[1]];

      const scanBox = new BubblesScanBox(
        fieldIndex,
        this.fieldLabel,
        bubbleOrigin,
        bubbleDimensions,
        { ...ZERO_MARGINS },
        bubbleValue,
        bubbleFieldType,
      );
      this.scanBoxes.push(scanBox);
      bubblePoint[h] += bubblesGap;
    }
  }

  resetAllShifts(): void {
    for (const scanBox of this.scanBoxes) {
      scanBox.resetShifts();
    }
  }

  toString(): string {
    return this.id;
  }
}
