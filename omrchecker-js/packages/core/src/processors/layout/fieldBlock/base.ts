/**
 * Field block base class.
 *
 * TypeScript port of src/processors/layout/field_block/base.py
 * Defines FieldBlock which contains a group of related fields.
 */

import { FieldDefinitionError } from '../../../core/exceptions';
import {
  FieldDetectionType,
  type FieldDetectionTypeValue,
} from '../../constants';
import { defaultDump, parseFields } from '../../../utils/parsing';
import type { Field, ScanBox } from '../field/base';
import { BarcodeField } from '../field/barcodeField';
import { BubbleField } from '../field/bubbleField';
import { OCRField } from '../field/ocrField';
import { FieldBlockDrawing } from './fieldBlockDrawing';

/**
 * Field block configuration from template JSON.
 */
export interface FieldBlockConfig {
  direction: 'horizontal' | 'vertical';
  emptyValue?: string;
  fieldDetectionType: FieldDetectionTypeValue;
  fieldLabels: string[];
  labelsGap: number;
  origin: [number, number];
  // Bubble-specific
  bubbleDimensions?: [number, number];
  bubbleValues?: string[];
  bubblesGap?: number;
  bubbleFieldType?: string;
  alignment?: {
    margins?: { top?: number; bottom?: number; left?: number; right?: number };
    maxDisplacement?: number;
    max_displacement?: number;
  };
  // OCR/Barcode-specific
  scanZone?: unknown;
}

/**
 * Field block class.
 *
 * Container for a group of related fields (e.g., questions 1-10).
 * Manages field generation, bounding box calculation, and shifts.
 */
export class FieldBlock {
  public static readonly fieldDetectionTypeToFieldClass = {
    [FieldDetectionType.BUBBLES_THRESHOLD]: BubbleField,
    [FieldDetectionType.OCR]: OCRField,
    [FieldDetectionType.BARCODE_QR]: BarcodeField,
  };

  public name: string;
  public plotBinName: string;
  public shifts: [number, number] = [0, 0];
  public direction!: 'horizontal' | 'vertical';
  public emptyValue!: string;
  public fieldDetectionType!: FieldDetectionTypeValue;
  public labelsGap!: number;
  public origin!: [number, number];
  public parsedFieldLabels: string[] = [];
  public fields: Field[] = [];
  public boundingBoxOrigin?: [number, number];
  public boundingBoxDimensions?: [number, number];
  public drawing: FieldBlockDrawing;

  // Bubble-specific properties
  public bubbleDimensions?: [number, number];
  public bubbleValues?: string[];
  public bubblesGap?: number;
  public bubbleFieldType?: string;
  public alignment?: {
    margins?: { top?: number; bottom?: number; left?: number; right?: number };
    maxDisplacement?: number;
    max_displacement?: number;
  };

  // OCR/Barcode-specific properties
  public scanZone?: unknown;

  constructor(
    blockName: string,
    fieldBlockObject: FieldBlockConfig,
    fieldBlocksOffset: [number, number]
  ) {
    this.name = blockName;
    this.plotBinName = blockName;
    this.shifts = [0, 0];

    this.setupFieldBlock(fieldBlockObject, fieldBlocksOffset);
    this.generateFields();
    this.drawing = this.getDrawingInstance();
  }

  /**
   * Setup field block from configuration object.
   */
  setupFieldBlock(
    fieldBlockObject: FieldBlockConfig,
    fieldBlocksOffset: [number, number]
  ): void {
    const {
      direction,
      emptyValue = '',
      fieldDetectionType,
      fieldLabels,
      labelsGap,
      origin,
    } = fieldBlockObject;

    this.direction = direction;
    this.emptyValue = emptyValue;
    this.fieldDetectionType = fieldDetectionType;
    this.labelsGap = labelsGap;
    const [offsetX, offsetY] = fieldBlocksOffset;
    this.origin = [origin[0] + offsetX, origin[1] + offsetY];

    this.parsedFieldLabels = parseFields(
      `Field Block Labels: ${this.name}`,
      fieldLabels
    );

    // Conditionally setup based on field detection type
    if (fieldDetectionType === FieldDetectionType.BUBBLES_THRESHOLD) {
      this.setupBubblesFieldBlock(fieldBlockObject);
    } else if (fieldDetectionType === FieldDetectionType.OCR) {
      this.setupOcrFieldBlock(fieldBlockObject);
    } else if (fieldDetectionType === FieldDetectionType.BARCODE_QR) {
      this.setupBarcodeQrFieldBlock(fieldBlockObject);
    } else {
      throw new FieldDefinitionError(
        `Unsupported field detection type: ${fieldDetectionType}`,
        { field_detection_type: fieldDetectionType }
      );
    }
  }

  /**
   * Setup bubble-specific field block properties.
   */
  setupBubblesFieldBlock(fieldBlockObject: FieldBlockConfig): void {
    const {
      alignment,
      bubbleDimensions,
      bubbleValues,
      bubblesGap,
      bubbleFieldType,
    } = fieldBlockObject;

    if (!bubbleDimensions || !bubbleValues || bubblesGap === undefined || !bubbleFieldType) {
      throw new FieldDefinitionError(
        'Missing required bubble field block properties',
        { fieldBlockObject }
      );
    }

    this.bubbleDimensions = bubbleDimensions;
    this.bubbleValues = bubbleValues;
    this.bubblesGap = bubblesGap;
    this.bubbleFieldType = bubbleFieldType;

    // Setup alignment
    this.alignment = {};
    if (alignment) {
      this.alignment = { ...alignment };
    }
  }

  /**
   * Setup OCR-specific field block properties.
   */
  setupOcrFieldBlock(fieldBlockObject: FieldBlockConfig): void {
    const { scanZone } = fieldBlockObject;
    this.scanZone = scanZone;
    // TODO: compute scan zone?
  }

  /**
   * Setup barcode/QR-specific field block properties.
   */
  setupBarcodeQrFieldBlock(fieldBlockObject: FieldBlockConfig): void {
    const { scanZone } = fieldBlockObject;
    this.scanZone = scanZone;
  }

  /**
   * Generate Field instances from parsed labels.
   */
  generateFields(): void {
    const { direction, emptyValue, fieldDetectionType, labelsGap } = this;

    // Determine direction index (0 for vertical, 1 for horizontal)
    const v = direction === 'vertical' ? 0 : 1;
    this.fields = [];

    // Generate the field grid
    const leadPoint: [number, number] = [this.origin[0], this.origin[1]];

    const FieldClass = FieldBlock.fieldDetectionTypeToFieldClass[fieldDetectionType];
    if (!FieldClass) {
      throw new FieldDefinitionError(
        `No field class found for detection type: ${fieldDetectionType}`,
        { fieldDetectionType }
      );
    }

    for (const fieldLabel of this.parsedFieldLabels) {
      const origin: [number, number] = [leadPoint[0], leadPoint[1]];
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const field = new (FieldClass as any)(
        direction,
        emptyValue,
        this,
        fieldDetectionType,
        fieldLabel,
        origin
      );
      this.fields.push(field);
      leadPoint[v] += labelsGap;
    }

    // TODO: validate for field block overflow outside template dimensions
    this.updateBoundingBox();
  }

  /**
   * Update bounding box from all scan boxes.
   */
  updateBoundingBox(): void {
    const allScanBoxes: ScanBox[] = [];
    for (const field of this.fields) {
      allScanBoxes.push(...field.scanBoxes);
    }

    if (allScanBoxes.length === 0) {
      this.boundingBoxOrigin = [0, 0];
      this.boundingBoxDimensions = [0, 0];
      return;
    }

    // Calculate bounding box origin (min x, min y)
    this.boundingBoxOrigin = [
      Math.min(...allScanBoxes.map((box) => box.origin[0])),
      Math.min(...allScanBoxes.map((box) => box.origin[1])),
    ];

    // Calculate bounding box dimensions
    // Note: we ignore the margins of the scan boxes when visualizing
    const maxX = Math.max(
      ...allScanBoxes.map((box) => box.origin[0] + box.dimensions[0])
    );
    const maxY = Math.max(
      ...allScanBoxes.map((box) => box.origin[1] + box.dimensions[1])
    );

    this.boundingBoxDimensions = [
      Math.round(maxX - this.boundingBoxOrigin[0]),
      Math.round(maxY - this.boundingBoxOrigin[1]),
    ];
  }

  /**
   * Create shallow copy for shifting operations.
   */
  getCopyForShifting(): FieldBlock {
    // Shallow copy is sufficient since we reset shifts anyway
    return Object.assign(Object.create(Object.getPrototypeOf(this)), this);
  }

  /**
   * Reset all shifts for block and all fields.
   */
  resetAllShifts(): void {
    this.shifts = [0, 0];
    for (const field of this.fields) {
      field.resetAllShifts();
    }
  }

  /**
   * Get origin with shifts applied.
   */
  getShiftedOrigin(): [number, number] {
    const [originX, originY] = this.origin;
    const [shiftX, shiftY] = this.shifts;
    return [originX + shiftX, originY + shiftY];
  }

  /**
   * Get drawing instance.
   */
  getDrawingInstance(): FieldBlockDrawing {
    return new FieldBlockDrawing(this);
  }

  /**
   * Serialize to JSON.
   */
  toJSON(): Record<string, unknown> {
    return {
      bubble_dimensions: defaultDump(this.bubbleDimensions),
      bounding_box_dimensions: defaultDump(this.boundingBoxDimensions),
      empty_value: this.emptyValue,
      fields: this.fields.map((field) => field.toJSON()),
      name: this.name,
      bounding_box_origin: defaultDump(this.boundingBoxOrigin),
    };
  }
}

