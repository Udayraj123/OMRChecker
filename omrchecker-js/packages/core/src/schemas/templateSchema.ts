/**
 * Template schema validation for OMRChecker TypeScript port
 *
 * Migrated from: src/schemas/template_schema.py
 */

import {
  L_MARKER_ZONE_TYPES_IN_ORDER,
  SCANNER_TYPES_IN_ORDER,
  SELECTOR_TYPES_IN_ORDER,
  ScannerTypeValue,
  SelectorTypeValue,
  ZonePresetValue,
} from '../processors/constants';

// Crop-on-marker types supported by the CropOnMarkers pre-processor
export const CROP_ON_MARKER_TYPES = [
  'FOUR_MARKERS',
  'L_MARKERS',
  'ONE_LINE_TWO_DOTS',
  'TWO_DOTS_ONE_LINE',
  'TWO_LINES',
  'FOUR_DOTS',
] as const;

export type CropOnMarkerType = typeof CROP_ON_MARKER_TYPES[number];

// Tuning options for the CropOnLMarkers pre-processor
export interface CropOnLMarkersTuningOptions {
  morphKernelSize?: [number, number];   // default [5, 5]
  morphIterations?: number;             // default 2
  minMarkerArea?: number;               // default 500
  maxMarkerArea?: number;               // default 50000
}

// L-marker zone preset names (exported for use in validation)
export const L_MARKER_ZONE_PRESETS: readonly ZonePresetValue[] = L_MARKER_ZONE_TYPES_IN_ORDER;

// Scanner types available in scan zones
export const VALID_SCANNER_TYPES: readonly ScannerTypeValue[] = SCANNER_TYPES_IN_ORDER;

// Selector types available in zone descriptions
export const VALID_SELECTOR_TYPES: readonly SelectorTypeValue[] = SELECTOR_TYPES_IN_ORDER;
