/**
 * Template configuration defaults.
 *
 * TypeScript port of src/schemas/defaults/template.py
 * Maintains 1:1 correspondence with Python implementation.
 */

import type {
  AlignmentConfig,
  OutputColumnsConfig,
  SortFilesConfig,
  TemplateConfig,
} from '../../template/types';

/**
 * Create default template config instance.
 *
 * Port of TEMPLATE_DEFAULTS from Python.
 * Note: templateDimensions and bubbleDimensions are required and not included in defaults.
 */
export const TEMPLATE_DEFAULTS: Partial<TemplateConfig> = {
  alignment: {
    margins: {
      top: 0,
      bottom: 0,
      left: 0,
      right: 0,
    },
    maxDisplacement: 10,
  } as AlignmentConfig,
  conditionalSets: [],
  customLabels: {},
  customBubbleFieldTypes: {},
  emptyValue: '',
  fieldBlocks: {},
  fieldBlocksOffset: [0, 0],
  outputColumns: {
    customOrder: [],
    sortType: 'ALPHANUMERIC',
    sortOrder: 'ASC',
  } as OutputColumnsConfig,
  preProcessors: [],
  processingImageShape: [900, 650],
  sortFiles: {
    enabled: false,
  } as SortFilesConfig,
};

