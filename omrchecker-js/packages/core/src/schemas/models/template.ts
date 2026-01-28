/**
 * Typed models for template configuration.
 *
 * TypeScript port of src/schemas/models/template.py
 * Maintains 1:1 correspondence with Python implementation.
 */

import type {
  AlignmentConfig,
  OutputColumnsConfig,
  SortFilesConfig,
  TemplateConfig,
} from '../../template/types';
import {
  validateNoKeyClash,
  convertDictKeysToSnake,
} from '../../utils/jsonConversion';

/**
 * Create TemplateConfig from dictionary (typically from JSON).
 *
 * Converts camelCase keys from JSON to snake_case for internal use.
 * Preserves user-defined names in customBubbleFieldTypes and fieldBlocks.
 *
 * Port of TemplateConfig.from_dict() from Python.
 *
 * @param data - Dictionary containing template configuration data (with camelCase keys)
 * @returns TemplateConfig instance with nested configs
 * @throws Error if the template JSON has clashing keys (both camelCase and snake_case)
 */
export function templateConfigFromDict(data: Record<string, any>): TemplateConfig {
  // Validate no key clashes before conversion
  try {
    validateNoKeyClash(data);
  } catch (e) {
    throw new Error(`Invalid template JSON: ${e instanceof Error ? e.message : String(e)}`);
  }

  // Preserve and convert customBubbleFieldTypes
  // Keep type names (keys), but convert keys within each type definition
  const customBubbleFieldTypesRaw = data.customBubbleFieldTypes || {};
  const customBubbleFieldTypesConverted: Record<string, any> = {};
  for (const [typeName, typeData] of Object.entries(customBubbleFieldTypesRaw)) {
    // Preserve type name, convert keys within type definition
    customBubbleFieldTypesConverted[typeName] = convertDictKeysToSnake(
      typeData as Record<string, any>
    );
  }

  // Preserve and convert fieldBlocks
  // Keep block names (keys), but convert keys within each block
  const fieldBlocksRaw = data.fieldBlocks || {};
  const fieldBlocksConverted: Record<string, any> = {};
  for (const [blockName, blockData] of Object.entries(fieldBlocksRaw)) {
    // Preserve block name, convert keys within block
    fieldBlocksConverted[blockName] = convertDictKeysToSnake(blockData as Record<string, any>);
  }

  // Convert all other top-level keys from camelCase to snake_case
  const dataConverted = convertDictKeysToSnake(data);

  // Parse alignment if present
  const alignmentData = dataConverted.alignment || {};
  const alignment: AlignmentConfig = {
    margins: {
      top: 0,
      bottom: 0,
      left: 0,
      right: 0,
      ...alignmentData.margins,
    },
    maxDisplacement: alignmentData.max_displacement ?? 10,
    ...alignmentData,
  };

  // Parse outputColumns if present
  const outputColumnsData = dataConverted.output_columns || {};
  const outputColumns: OutputColumnsConfig = {
    customOrder: outputColumnsData.custom_order || [],
    sortType: outputColumnsData.sort_type || 'ALPHANUMERIC',
    sortOrder: outputColumnsData.sort_order || 'ASC',
  };

  // Parse sortFiles if present
  const sortFilesData = dataConverted.sort_files || {};
  const sortFiles: SortFilesConfig = {
    enabled: sortFilesData.enabled ?? false,
  };

  return {
    templateDimensions: dataConverted.template_dimensions || [1200, 1600],
    bubbleDimensions: dataConverted.bubble_dimensions || [10, 10],
    alignment,
    conditionalSets: dataConverted.conditional_sets || [],
    customLabels: dataConverted.custom_labels || {},
    // Use converted custom bubble field types (type names preserved, keys converted)
    customBubbleFieldTypes: customBubbleFieldTypesConverted,
    emptyValue: dataConverted.empty_value || '',
    // Use converted field blocks (block names preserved, keys converted)
    fieldBlocks: fieldBlocksConverted,
    fieldBlocksOffset: dataConverted.field_blocks_offset || [0, 0],
    outputColumns,
    preProcessors: dataConverted.pre_processors || [],
    processingImageShape: dataConverted.processing_image_shape || [900, 650],
    sortFiles,
  };
}

/**
 * Convert TemplateConfig to dictionary for JSON serialization.
 *
 * Port of TemplateConfig.to_dict() from Python.
 *
 * @param config - TemplateConfig instance
 * @returns Dictionary representation of the template config
 */
export function templateConfigToDict(config: TemplateConfig): Record<string, any> {
  return {
    templateDimensions: config.templateDimensions,
    bubbleDimensions: config.bubbleDimensions,
    alignment: config.alignment,
    conditionalSets: config.conditionalSets,
    customLabels: config.customLabels,
    customBubbleFieldTypes: config.customBubbleFieldTypes,
    emptyValue: config.emptyValue,
    fieldBlocks: config.fieldBlocks,
    fieldBlocksOffset: config.fieldBlocksOffset,
    outputColumns: config.outputColumns,
    preProcessors: config.preProcessors,
    processingImageShape: config.processingImageShape,
    sortFiles: config.sortFiles,
  };
}

