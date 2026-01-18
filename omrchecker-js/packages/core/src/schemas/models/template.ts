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

/**
 * Create TemplateConfig from dictionary (typically from JSON).
 *
 * Port of TemplateConfig.from_dict() from Python.
 *
 * @param data - Dictionary containing template configuration data
 * @returns TemplateConfig instance with nested configs
 */
export function templateConfigFromDict(data: Record<string, any>): TemplateConfig {
  // Parse alignment if present
  const alignmentData = data.alignment || {};
  const alignment: AlignmentConfig = {
    margins: {
      top: 0,
      bottom: 0,
      left: 0,
      right: 0,
      ...alignmentData.margins,
    },
    maxDisplacement: alignmentData.maxDisplacement ?? 10,
    ...alignmentData,
  };

  // Parse outputColumns if present
  const outputColumnsData = data.outputColumns || {};
  const outputColumns: OutputColumnsConfig = {
    customOrder: outputColumnsData.customOrder || [],
    sortType: outputColumnsData.sortType || 'ALPHANUMERIC',
    sortOrder: outputColumnsData.sortOrder || 'ASC',
  };

  // Parse sortFiles if present
  const sortFilesData = data.sortFiles || {};
  const sortFiles: SortFilesConfig = {
    enabled: sortFilesData.enabled ?? false,
  };

  return {
    templateDimensions: data.templateDimensions,
    bubbleDimensions: data.bubbleDimensions,
    alignment,
    conditionalSets: data.conditionalSets || [],
    customLabels: data.customLabels || {},
    customBubbleFieldTypes: data.customBubbleFieldTypes || {},
    emptyValue: data.emptyValue || '',
    fieldBlocks: data.fieldBlocks || {},
    fieldBlocksOffset: data.fieldBlocksOffset || [0, 0],
    outputColumns,
    preProcessors: data.preProcessors || [],
    processingImageShape: data.processingImageShape || [900, 650],
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

