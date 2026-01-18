/**
 * Bubble field classes.
 *
 * TypeScript port of src/processors/layout/field/bubble_field.py
 * Defines BubbleField and BubblesScanBox for bubble-based OMR fields.
 */

import { ZERO_MARGINS } from '../../../utils/constants';
import { defaultDump } from '../../../utils/parsing';
import { FieldBlock } from '../fieldBlock/base';
import { Field, ScanBox } from './base';
import { BubbleFieldDrawing, type FieldDrawing } from './fieldDrawing';

/**
 * Scan box for individual bubbles.
 *
 * Container for a Point Box on the OMR.
 * field_label is the point's property - field to which this point belongs to.
 * It can be used as a roll number column as well. (eg roll1)
 * It can also correspond to a single digit of integer type Q (eg q5d1).
 */
export class BubblesScanBox extends ScanBox {
  public bubbleFieldType: string;
  public bubbleDimensions: [number, number];
  public bubbleValue: string;

  constructor(
    fieldIndex: number,
    field: BubbleField,
    origin: [number, number],
    bubbleValue: string
  ) {
    const dimensions = field.bubbleDimensions;
    const margins = ZERO_MARGINS;
    super(fieldIndex, field, origin, dimensions, margins);

    this.bubbleFieldType = field.bubbleFieldType;
    this.bubbleDimensions = field.bubbleDimensions;
    this.bubbleValue = bubbleValue;
    this.name = `${this.fieldLabel}_${this.bubbleValue}`;
  }

  /**
   * Serialize to JSON.
   */
  toJSON(): Record<string, unknown> {
    return {
      field_label: this.fieldLabel,
      bubble_value: this.bubbleValue,
      name: this.name,
      x: this.x,
      y: this.y,
      origin: defaultDump(this.origin),
    };
  }
}

/**
 * Bubble field class.
 *
 * Extends Field for bubble-based fields (MCQ, integer, etc.).
 */
export class BubbleField extends Field {
  public bubbleDimensions: [number, number];
  public bubbleValues: string[];
  public bubblesGap: number;
  public bubbleFieldType: string;

  constructor(
    direction: 'horizontal' | 'vertical',
    emptyValue: string,
    fieldBlock: FieldBlock,
    fieldDetectionType: 'BUBBLES_THRESHOLD',
    fieldLabel: string,
    origin: [number, number]
  ) {
    // Must call super first
    super(direction, emptyValue, fieldBlock, fieldDetectionType, fieldLabel, origin);

    if (
      !fieldBlock.bubbleDimensions ||
      !fieldBlock.bubbleValues ||
      fieldBlock.bubblesGap === undefined ||
      !fieldBlock.bubbleFieldType
    ) {
      throw new Error('FieldBlock missing bubble properties');
    }

    this.bubbleDimensions = fieldBlock.bubbleDimensions;
    this.bubbleValues = fieldBlock.bubbleValues;
    this.bubblesGap = fieldBlock.bubblesGap;
    this.bubbleFieldType = fieldBlock.bubbleFieldType;
  }

  /**
   * Setup scan boxes for this bubble field.
   * Creates a BubblesScanBox for each bubble value.
   */
  setupScanBoxes(_fieldBlock: FieldBlock): void {
    // Determine direction index (0 for vertical, 1 for horizontal)
    const h = this.direction === 'vertical' ? 1 : 0;

    const bubblePoint: [number, number] = [...this.origin];
    this.scanBoxes = [] as BubblesScanBox[];

    for (let fieldIndex = 0; fieldIndex < this.bubbleValues.length; fieldIndex++) {
      const bubbleValue = this.bubbleValues[fieldIndex];
      const bubbleOrigin: [number, number] = [...bubblePoint];
      const scanBox = new BubblesScanBox(fieldIndex, this, bubbleOrigin, bubbleValue);
      this.scanBoxes.push(scanBox);
      bubblePoint[h] += this.bubblesGap;
    }

  }

  /**
   * Get drawing instance for bubble fields.
   */
  getDrawingInstance(): FieldDrawing {
    return new BubbleFieldDrawing(this);
  }
}

