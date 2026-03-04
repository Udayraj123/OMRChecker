/**
 * FieldBlock class for OMRChecker TypeScript port.
 *
 * Migrated from: src/processors/layout/field_block/base.py (FieldBlock)
 * Scope: BUBBLES_THRESHOLD field detection only.
 */

import { OMRCheckerError } from '../utils/exceptions';
import { parseFields } from './parseFields';
import { BubbleField } from './BubbleField';

/**
 * Raw JSON shape of a field block entry (camelCase from template.json).
 * The Python code converts these to snake_case internally; in TS we keep camelCase.
 */
export interface FieldBlockJson {
  origin: [number, number];
  fieldLabels: string[];
  fieldDetectionType: string;
  labelsGap?: number;
  labels_gap?: number;
  direction?: string;
  emptyValue?: string;
  empty_value?: string;
  // BUBBLES_THRESHOLD fields
  bubbleFieldType?: string;
  bubble_field_type?: string;
  bubbleDimensions?: [number, number];
  bubble_dimensions?: [number, number];
  bubbleValues?: string[];
  bubble_values?: string[];
  bubblesGap?: number;
  bubbles_gap?: number;
}

/**
 * A block of fields sharing common layout properties.
 * Each block contains multiple BubbleField instances.
 *
 * Ported from Python: src/processors/layout/field_block/base.py::FieldBlock
 */
export class FieldBlock {
  name: string;
  direction: string;
  emptyValue: string;
  labelsGap: number;
  origin: [number, number];
  bubbleDimensions: [number, number];
  bubbleValues: string[];
  bubblesGap: number;
  bubbleFieldType: string;
  shifts: [number, number];
  parsedFieldLabels: string[];
  fields: BubbleField[];
  boundingBoxOrigin: [number, number];
  boundingBoxDimensions: [number, number];

  constructor(
    blockName: string,
    fieldBlockObject: FieldBlockJson & Record<string, any>,
    fieldBlocksOffset: [number, number],
  ) {
    this.name = blockName;
    this.shifts = [0, 0];

    this.setupFieldBlock(fieldBlockObject, fieldBlocksOffset);
    this.generateFields();
  }

  private setupFieldBlock(
    obj: FieldBlockJson & Record<string, any>,
    fieldBlocksOffset: [number, number],
  ): void {
    const fieldDetectionType: string =
      obj.fieldDetectionType ?? obj.field_detection_type ?? 'BUBBLES_THRESHOLD';

    if (fieldDetectionType !== 'BUBBLES_THRESHOLD') {
      throw new OMRCheckerError(
        `Unsupported field detection type: ${fieldDetectionType}`,
        { field_detection_type: fieldDetectionType, block_name: this.name },
      );
    }

    this.direction = obj.direction ?? 'horizontal';
    this.emptyValue = obj.emptyValue ?? obj.empty_value ?? '';
    this.labelsGap = obj.labelsGap ?? obj.labels_gap ?? 0;

    const [ox, oy] = obj.origin;
    const [offsetX, offsetY] = fieldBlocksOffset;
    this.origin = [ox + offsetX, oy + offsetY];

    const fieldLabels: string[] = obj.fieldLabels ?? obj.field_labels ?? [];
    this.parsedFieldLabels = parseFields(`Field Block Labels: ${this.name}`, fieldLabels);

    // BUBBLES_THRESHOLD-specific properties
    this.bubbleDimensions = obj.bubbleDimensions ?? obj.bubble_dimensions ?? [10, 10];
    this.bubbleValues = obj.bubbleValues ?? obj.bubble_values ?? [];
    this.bubblesGap = obj.bubblesGap ?? obj.bubbles_gap ?? 0;
    this.bubbleFieldType = obj.bubbleFieldType ?? obj.bubble_field_type ?? '';
  }

  /**
   * Generate BubbleField instances, one per parsed label.
   *
   * Direction determines which axis labels extend along:
   *   vertical   → v=0 → X axis increments (labels stack left-to-right)
   *   horizontal → v=1 → Y axis increments (labels go top-to-bottom)
   *
   * Ported from Python: src/processors/layout/field_block/base.py::FieldBlock.generate_fields
   */
  generateFields(): void {
    const { direction, emptyValue, labelsGap, parsedFieldLabels } = this;
    const { bubbleDimensions, bubbleValues, bubblesGap, bubbleFieldType } = this;

    // v=0 means X-axis gap (vertical direction), v=1 means Y-axis gap (horizontal direction)
    const v = direction === 'vertical' ? 0 : 1;

    this.fields = [];
    const leadPoint: [number, number] = [
      parseFloat(String(this.origin[0])),
      parseFloat(String(this.origin[1])),
    ];

    for (const fieldLabel of parsedFieldLabels) {
      const origin: [number, number] = [leadPoint[0], leadPoint[1]];
      const field = new BubbleField(
        direction,
        emptyValue,
        this.name,
        bubbleDimensions,
        bubbleValues,
        bubblesGap,
        bubbleFieldType,
        fieldLabel,
        origin,
      );
      this.fields.push(field);
      leadPoint[v] += labelsGap;
    }

    this.updateBoundingBox();
  }

  /**
   * Update the bounding box that covers all scan boxes in this field block.
   *
   * Ported from Python: src/processors/layout/field_block/base.py::FieldBlock.update_bounding_box
   */
  updateBoundingBox(): void {
    const allScanBoxes = this.fields.flatMap((field) => field.scanBoxes);

    if (allScanBoxes.length === 0) {
      this.boundingBoxOrigin = [this.origin[0], this.origin[1]];
      this.boundingBoxDimensions = [0, 0];
      return;
    }

    const minX = Math.min(...allScanBoxes.map((sb) => sb.origin[0]));
    const minY = Math.min(...allScanBoxes.map((sb) => sb.origin[1]));

    const maxX = Math.max(...allScanBoxes.map((sb) => sb.origin[0] + sb.dimensions[0]));
    const maxY = Math.max(...allScanBoxes.map((sb) => sb.origin[1] + sb.dimensions[1]));

    this.boundingBoxOrigin = [minX, minY];
    this.boundingBoxDimensions = [
      Math.round((maxX - minX) * 100) / 100,
      Math.round((maxY - minY) * 100) / 100,
    ];
  }

  getShiftedOrigin(): [number, number] {
    return [
      this.origin[0] + this.shifts[0],
      this.origin[1] + this.shifts[1],
    ];
  }

  resetAllShifts(): void {
    this.shifts = [0, 0];
    for (const field of this.fields) {
      field.resetAllShifts();
    }
  }
}
