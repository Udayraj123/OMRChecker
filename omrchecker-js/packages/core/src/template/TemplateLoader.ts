/**
 * Template loader and parser.
 *
 * Port of Python template loading logic (src/processors/layout/template_layout.py).
 * Loads and validates template.json files, creating FieldBlock and Field instances.
 */

import { OMRCheckerError } from '../core/exceptions';
import { FieldDetectionType } from '../processors/constants';
import { FieldBlock, type FieldBlockConfig } from '../processors/layout/fieldBlock/base';
import type { Field } from '../processors/layout/field/base';
import { BUILTIN_BUBBLE_FIELD_TYPES } from '../utils/constants';
import { Logger } from '../utils/logger';
import type { SaveImageOps } from '../utils/SaveImageOps';
// parseFields is used in FieldBlock, not here
import {
  type TemplateConfig,
  type BubbleFieldType,
  type ParsedTemplate,
  type ParsedField,
  DEFAULT_TEMPLATE_CONFIG,
} from './types';
import { type BubbleLocation } from '../processors/detection';

const logger = new Logger('TemplateLoader');

export type { ParsedTemplate, ParsedField };

/**
 * Template layout data structure (matches Python TemplateLayout data).
 * This is the data structure returned by loadLayoutFromJSON.
 * For the class with methods, see TemplateLayout class.
 */
export interface TemplateLayoutData {
  templateDimensions: [number, number];
  bubbleDimensions: [number, number];
  fieldBlocks: FieldBlock[];
  allFields: Field[];
  allFieldDetectionTypes: string[];
  allParsedLabels: Set<string>;
  fieldBlocksOffset: [number, number];
  globalEmptyValue: string;
  bubbleFieldTypesData: Record<string, BubbleFieldType>;
  saveImageOps?: SaveImageOps; // Optional SaveImageOps for debug image saving
}

/**
 * Template loader and parser.
 */
export class TemplateLoader {
  /**
   * Load template layout from JSON (creates FieldBlock instances).
   *
   * Port of Python TemplateLayout.setup_layout().
   *
   * @param json - Template JSON object
   * @returns Template layout with FieldBlock instances
   */
  static loadLayoutFromJSON(json: TemplateConfig): TemplateLayoutData {
    logger.info('Loading template layout from JSON');

    // Apply defaults
    const config: TemplateConfig = {
      ...DEFAULT_TEMPLATE_CONFIG,
      ...json,
      alignment: { ...DEFAULT_TEMPLATE_CONFIG.alignment, ...json.alignment },
      outputColumns: { ...DEFAULT_TEMPLATE_CONFIG.outputColumns, ...json.outputColumns },
      sortFiles: { ...DEFAULT_TEMPLATE_CONFIG.sortFiles, ...json.sortFiles },
    };

    // Validate required fields
    if (!config.templateDimensions || config.templateDimensions.length !== 2) {
      throw new Error('templateDimensions is required and must be [width, height]');
    }
    if (!config.bubbleDimensions || config.bubbleDimensions.length !== 2) {
      throw new Error('bubbleDimensions is required and must be [width, height]');
    }
    if (!config.fieldBlocks || Object.keys(config.fieldBlocks).length === 0) {
      throw new Error('fieldBlocks is required and must contain at least one block');
    }

    // Merge custom bubble field types with built-in types
    const bubbleFieldTypesData: Record<string, BubbleFieldType> = {
      ...Object.fromEntries(
        Object.entries(BUILTIN_BUBBLE_FIELD_TYPES).map(([k, v]) => [
          k,
          { bubbleValues: [...v.bubbleValues], direction: v.direction },
        ])
      ),
      ...config.customBubbleFieldTypes,
    };

    const fieldBlocks: FieldBlock[] = [];
    const allFields: Field[] = [];
    const allFieldDetectionTypes = new Set<string>();
    const allParsedLabels = new Set<string>();

    const fieldBlocksOffset = config.fieldBlocksOffset || [0, 0];
    const globalEmptyValue = config.emptyValue || '';

    // Validate and create field blocks
    for (const [blockName, fieldBlockObject] of Object.entries(config.fieldBlocks)) {
      logger.debug(`Parsing field block: ${blockName}`);

      // Validate field block
      this.validateFieldBlock(blockName, fieldBlockObject, bubbleFieldTypesData);

      // Prefill field block with defaults
      const filledFieldBlock = this.prefillFieldBlock(
        fieldBlockObject,
        config.bubbleDimensions,
        globalEmptyValue,
        bubbleFieldTypesData
      );

      // Create FieldBlock instance
      const blockInstance = new FieldBlock(blockName, filledFieldBlock, fieldBlocksOffset);

      // Validate parsed field block
      this.validateParsedFieldBlock(blockName, blockInstance, allParsedLabels, config.templateDimensions);

      fieldBlocks.push(blockInstance);
      allFields.push(...blockInstance.fields);
      allFieldDetectionTypes.add(blockInstance.fieldDetectionType);

      // Update parsed labels
      for (const label of blockInstance.parsedFieldLabels) {
        allParsedLabels.add(label);
      }
    }

    logger.info(`Template layout loaded: ${fieldBlocks.length} blocks, ${allFields.length} fields`);

    return {
      templateDimensions: config.templateDimensions,
      bubbleDimensions: config.bubbleDimensions,
      fieldBlocks,
      allFields,
      allFieldDetectionTypes: Array.from(allFieldDetectionTypes),
      allParsedLabels,
      fieldBlocksOffset,
      globalEmptyValue,
      bubbleFieldTypesData,
    };
  }

  /**
   * Load and parse a template from JSON (legacy method - maintains backward compatibility).
   *
   * @param json - Template JSON object
   * @returns Parsed template with bubble locations
   */
  static loadFromJSON(json: TemplateConfig): ParsedTemplate {
    logger.info('Loading template from JSON');

    // Apply defaults
    const config: TemplateConfig = {
      ...DEFAULT_TEMPLATE_CONFIG,
      ...json,
      alignment: { ...DEFAULT_TEMPLATE_CONFIG.alignment, ...json.alignment },
      outputColumns: { ...DEFAULT_TEMPLATE_CONFIG.outputColumns, ...json.outputColumns },
      sortFiles: { ...DEFAULT_TEMPLATE_CONFIG.sortFiles, ...json.sortFiles },
    };

    // Validate required fields
    if (!config.templateDimensions || config.templateDimensions.length !== 2) {
      throw new Error('templateDimensions is required and must be [width, height]');
    }
    if (!config.bubbleDimensions || config.bubbleDimensions.length !== 2) {
      throw new Error('bubbleDimensions is required and must be [width, height]');
    }
    if (!config.fieldBlocks || Object.keys(config.fieldBlocks).length === 0) {
      throw new Error('fieldBlocks is required and must contain at least one block');
    }

    // Merge custom bubble field types with built-in types
    const allBubbleFieldTypes: Record<string, BubbleFieldType> = {
      ...Object.fromEntries(
        Object.entries(BUILTIN_BUBBLE_FIELD_TYPES).map(([k, v]) => [
          k,
          { bubbleValues: [...v.bubbleValues], direction: v.direction },
        ])
      ),
      ...config.customBubbleFieldTypes,
    };

    // Parse field blocks
    const fields = new Map<string, ParsedField>();
    const fieldBubbles = new Map<string, BubbleLocation[]>();

    const [offsetX, offsetY] = config.fieldBlocksOffset || [0, 0];

    for (const [blockName, fieldBlock] of Object.entries(config.fieldBlocks)) {
      logger.debug(`Parsing field block: ${blockName}`);

      // Get bubble field type
      const bubbleFieldType = allBubbleFieldTypes[fieldBlock.bubbleFieldType];
      if (!bubbleFieldType) {
        throw new Error(
          `Unknown bubbleFieldType: ${fieldBlock.bubbleFieldType} in block ${blockName}`
        );
      }

      // Expand field labels (e.g., "q1..10" → ["q1", "q2", ..., "q10"])
      const expandedLabels = this.expandFieldLabels(fieldBlock.fieldLabels);

      // Get bubble dimensions (block-specific or global)
      const bubbleDims = fieldBlock.bubbleDimensions || config.bubbleDimensions;
      const [bubbleWidth, bubbleHeight] = bubbleDims;

      // Get empty value (block-specific or global)
      const emptyValue = fieldBlock.emptyValue ?? config.emptyValue ?? '';

      // Calculate bubble locations for each field
      const [originX, originY] = fieldBlock.origin;
      const adjustedOriginX = originX + offsetX;
      const adjustedOriginY = originY + offsetY;

      for (let fieldIdx = 0; fieldIdx < expandedLabels.length; fieldIdx++) {
        const fieldId = expandedLabels[fieldIdx];
        const bubbles: BubbleLocation[] = [];

        // Calculate field origin (with labelsGap)
        const fieldOriginX = adjustedOriginX + fieldIdx * fieldBlock.labelsGap;
        const fieldOriginY = adjustedOriginY;

        // Create bubbles for this field
        for (let bubbleIdx = 0; bubbleIdx < bubbleFieldType.bubbleValues.length; bubbleIdx++) {
          const bubbleValue = bubbleFieldType.bubbleValues[bubbleIdx];

          // Calculate bubble position based on direction
          let bubbleX: number;
          let bubbleY: number;

          if (bubbleFieldType.direction === 'horizontal') {
            bubbleX = fieldOriginX + bubbleIdx * fieldBlock.bubblesGap;
            bubbleY = fieldOriginY;
          } else {
            // vertical
            bubbleX = fieldOriginX;
            bubbleY = fieldOriginY + bubbleIdx * fieldBlock.bubblesGap;
          }

          bubbles.push({
            x: bubbleX,
            y: bubbleY,
            width: bubbleWidth,
            height: bubbleHeight,
            label: bubbleValue,
          });
        }

        // Store parsed field
        fields.set(fieldId, {
          fieldId,
          blockName,
          bubbles,
          bubbleFieldType,
          emptyValue,
        });

        fieldBubbles.set(fieldId, bubbles);

        logger.debug(
          `  Field ${fieldId}: ${bubbles.length} bubbles at (${fieldOriginX}, ${fieldOriginY})`
        );
      }
    }

    logger.info(`Template loaded: ${fields.size} fields, ${fieldBubbles.size} bubble groups`);

    return {
      config,
      templateDimensions: config.templateDimensions,
      bubbleDimensions: config.bubbleDimensions,
      fieldBlocks: config.fieldBlocks,
      fields,
      fieldBubbles,
    };
  }

  /**
   * Load template from JSON string.
   *
   * @param jsonString - Template JSON as string
   * @returns Parsed template
   */
  static loadFromJSONString(jsonString: string): ParsedTemplate {
    const json = JSON.parse(jsonString) as TemplateConfig;
    return this.loadFromJSON(json);
  }

  /**
   * Expand field labels with range syntax.
   *
   * Examples:
   * - ["q1..5"] → ["q1", "q2", "q3", "q4", "q5"]
   * - ["q1", "q2", "q3"] → ["q1", "q2", "q3"] (no change)
   * - ["roll1..10"] → ["roll1", "roll2", ..., "roll10"]
   *
   * @param labels - Field labels (may contain ".." range syntax)
   * @returns Expanded labels
   */
  private static expandFieldLabels(labels: string[]): string[] {
    const expanded: string[] = [];

    for (const label of labels) {
      if (label.includes('..')) {
        // Range syntax: "q1..10"
        const match = label.match(/^(.+?)(\d+)\.\.(\d+)$/);
        if (!match) {
          throw new Error(`Invalid range syntax: ${label}`);
        }

        const [, prefix, startStr, endStr] = match;
        const start = parseInt(startStr, 10);
        const end = parseInt(endStr, 10);

        if (start > end) {
          throw new Error(`Invalid range: ${label} (start > end)`);
        }

        for (let i = start; i <= end; i++) {
          expanded.push(`${prefix}${i}`);
        }
      } else {
        // Simple label
        expanded.push(label);
      }
    }

    return expanded;
  }

  /**
   * Get all bubble locations flattened (for visualization).
   *
   * @param parsedTemplate - Parsed template
   * @returns Array of all bubble locations
   */
  static getAllBubbles(parsedTemplate: ParsedTemplate): BubbleLocation[] {
    const allBubbles: BubbleLocation[] = [];

    for (const field of parsedTemplate.fields.values()) {
      allBubbles.push(...field.bubbles);
    }

    return allBubbles;
  }

  /**
   * Get field IDs in sorted order (for CSV export).
   *
   * @param parsedTemplate - Parsed template
   * @returns Sorted field IDs
   */
  static getSortedFieldIds(parsedTemplate: ParsedTemplate): string[] {
    const { config, fields } = parsedTemplate;
    const fieldIds = Array.from(fields.keys());

    // Apply custom order if specified
    if (config.outputColumns?.customOrder && config.outputColumns.customOrder.length > 0) {
      const customOrder = config.outputColumns.customOrder;
      const remaining = fieldIds.filter((id) => !customOrder.includes(id));
      return [...customOrder, ...remaining];
    }

    // Sort alphanumerically
    return fieldIds.sort((a: string, b: string) => {
      // Natural sort (q1 < q2 < q10)
      return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
    });
  }

  /**
   * Validate field block configuration.
   * Port of Python TemplateLayout.validate_field_blocks().
   */
  private static validateFieldBlock(
    blockName: string,
    fieldBlockObject: any,
    bubbleFieldTypesData: Record<string, BubbleFieldType>
  ): void {
    if (fieldBlockObject.fieldDetectionType === FieldDetectionType.BUBBLES_THRESHOLD) {
      const bubbleFieldType = fieldBlockObject.bubbleFieldType;
      if (!bubbleFieldType || !bubbleFieldTypesData[bubbleFieldType]) {
        throw new OMRCheckerError(
          `Invalid bubble field type: ${bubbleFieldType} in block ${blockName}. Have you defined customBubbleFieldTypes?`,
          {
            bubble_field_type: bubbleFieldType,
            block_name: blockName,
          }
        );
      }
    }

    const fieldLabels = fieldBlockObject.fieldLabels;
    if (fieldLabels.length > 1 && !fieldBlockObject.labelsGap) {
      throw new OMRCheckerError(
        `More than one fieldLabels provided, but labelsGap not present for block ${blockName}`,
        {
          field_labels: fieldLabels,
          block_name: blockName,
        }
      );
    }
  }

  /**
   * Prefill field block with defaults.
   * Port of Python TemplateLayout.prefill_field_block().
   */
  private static prefillFieldBlock(
    fieldBlockObject: any,
    bubbleDimensions: [number, number],
    globalEmptyValue: string,
    bubbleFieldTypesData: Record<string, BubbleFieldType>
  ): FieldBlockConfig {
    const filled: FieldBlockConfig = { ...fieldBlockObject };

    if (fieldBlockObject.fieldDetectionType === FieldDetectionType.BUBBLES_THRESHOLD) {
      const bubbleFieldType = fieldBlockObject.bubbleFieldType;
      const fieldTypeData = bubbleFieldTypesData[bubbleFieldType];

      filled.bubbleFieldType = bubbleFieldType;
      filled.emptyValue = globalEmptyValue;
      filled.bubbleDimensions = bubbleDimensions;
      filled.direction = fieldTypeData.direction;
      filled.bubbleValues = fieldTypeData.bubbleValues;
    } else if (
      fieldBlockObject.fieldDetectionType === FieldDetectionType.OCR ||
      fieldBlockObject.fieldDetectionType === FieldDetectionType.BARCODE_QR
    ) {
      filled.emptyValue = globalEmptyValue;
      filled.labelsGap = 0;
    }

    return filled;
  }

  /**
   * Validate parsed field block.
   * Port of Python TemplateLayout.validate_parsed_field_block().
   */
  private static validateParsedFieldBlock(
    blockName: string,
    blockInstance: FieldBlock,
    allParsedLabels: Set<string>,
    templateDimensions: [number, number]
  ): void {
    const parsedFieldLabels = blockInstance.parsedFieldLabels;
    const fieldLabelsSet = new Set(parsedFieldLabels);

    // Check for overlapping labels
    const overlap = Array.from(fieldLabelsSet).filter((label) => allParsedLabels.has(label));
    if (overlap.length > 0) {
      throw new OMRCheckerError(
        `The field strings for field block ${blockName} overlap with other existing fields: ${overlap.join(', ')}`,
        {
          block_name: blockName,
          field_labels: parsedFieldLabels,
          overlap,
        }
      );
    }

    // Check for bounding box overflow
    if (!blockInstance.boundingBoxOrigin || !blockInstance.boundingBoxDimensions) {
      return; // Skip if bounding box not calculated
    }

    const [pageWidth, pageHeight] = templateDimensions;
    const [blockWidth, blockHeight] = blockInstance.boundingBoxDimensions;
    const [blockStartX, blockStartY] = blockInstance.boundingBoxOrigin;
    const blockEndX = blockStartX + blockWidth;
    const blockEndY = blockStartY + blockHeight;

    if (
      blockEndX >= pageWidth ||
      blockEndY >= pageHeight ||
      blockStartX < 0 ||
      blockStartY < 0
    ) {
      throw new OMRCheckerError(
        `Overflowing field block '${blockName}' with origin [${blockStartX}, ${blockStartY}] and dimensions [${blockWidth}, ${blockHeight}] in template with dimensions [${pageWidth}, ${pageHeight}]`,
        {
          block_name: blockName,
          bounding_box_origin: blockInstance.boundingBoxOrigin,
          bounding_box_dimensions: blockInstance.boundingBoxDimensions,
          template_dimensions: templateDimensions,
        }
      );
    }
  }
}

