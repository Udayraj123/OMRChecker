/**
 * Template loader and parser.
 *
 * Port of Python template loading logic.
 * Loads and validates template.json files, expanding field labels,
 * and generating bubble locations for detection.
 */

import {
  type TemplateConfig,
  type BubbleFieldType,
  BUILTIN_BUBBLE_FIELD_TYPES,
  DEFAULT_TEMPLATE_CONFIG,
} from './types';
import { type BubbleLocation } from '../processors/detection/SimpleBubbleDetector';
import { Logger } from '../utils/logger';

const logger = new Logger('TemplateLoader');

/**
 * Parsed field with expanded bubble locations.
 */
export interface ParsedField {
  /** Field ID (e.g., "q1", "q2") */
  fieldId: string;
  /** Block this field belongs to */
  blockName: string;
  /** Bubble locations for this field */
  bubbles: BubbleLocation[];
  /** Bubble field type */
  bubbleFieldType: BubbleFieldType;
  /** Empty value for this field */
  emptyValue: string;
}

/**
 * Fully parsed template with all bubble locations calculated.
 */
export interface ParsedTemplate {
  /** Original configuration */
  config: TemplateConfig;
  /** Map of field ID to parsed field */
  fields: Map<string, ParsedField>;
  /** Map of field ID to bubble locations (for SimpleBubbleDetector) */
  fieldBubbles: Map<string, BubbleLocation[]>;
}

/**
 * Template loader and parser.
 */
export class TemplateLoader {
  /**
   * Load and parse a template from JSON.
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
      ...BUILTIN_BUBBLE_FIELD_TYPES,
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
    return fieldIds.sort((a, b) => {
      // Natural sort (q1 < q2 < q10)
      return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
    });
  }
}

