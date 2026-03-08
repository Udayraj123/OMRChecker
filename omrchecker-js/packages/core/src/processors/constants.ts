/**
 * Processor constants for OMRChecker TypeScript port
 *
 * Migrated from: src/processors/constants.py
 */

// Edge types for rectangle sides
export const EdgeType = {
  TOP: 'TOP',
  RIGHT: 'RIGHT',
  BOTTOM: 'BOTTOM',
  LEFT: 'LEFT',
} as const;
export type EdgeTypeValue = typeof EdgeType[keyof typeof EdgeType];

// Edge types in clockwise order: TOP, RIGHT, BOTTOM, LEFT
export const EDGE_TYPES_IN_ORDER: EdgeTypeValue[] = [
  EdgeType.TOP,
  EdgeType.RIGHT,
  EdgeType.BOTTOM,
  EdgeType.LEFT,
];

// Scanner types for patch detection
export const ScannerType = {
  PATCH_DOT: 'PATCH_DOT',
  PATCH_LINE: 'PATCH_LINE',
  TEMPLATE_MATCH: 'TEMPLATE_MATCH',
  PATCH_L_MARKER: 'PATCH_L_MARKER',
} as const;
export type ScannerTypeValue = typeof ScannerType[keyof typeof ScannerType];

export const SCANNER_TYPES_IN_ORDER: ScannerTypeValue[] = [
  ScannerType.PATCH_DOT,
  ScannerType.PATCH_LINE,
  ScannerType.TEMPLATE_MATCH,
  ScannerType.PATCH_L_MARKER,
];

// Warp methods for image transformation
export const WarpMethod = {
  PERSPECTIVE_TRANSFORM: 'PERSPECTIVE_TRANSFORM',
  HOMOGRAPHY: 'HOMOGRAPHY',
  REMAP_GRIDDATA: 'REMAP_GRIDDATA',
  DOC_REFINE: 'DOC_REFINE',
  WARP_AFFINE: 'WARP_AFFINE',
} as const;
export type WarpMethodValue = typeof WarpMethod[keyof typeof WarpMethod];

// Warp interpolation flags (string → cv constant mapping)
export const WarpMethodFlags = {
  INTER_LINEAR: 'INTER_LINEAR',   // cv.INTER_LINEAR = 1
  INTER_CUBIC: 'INTER_CUBIC',     // cv.INTER_CUBIC = 2
  INTER_NEAREST: 'INTER_NEAREST', // cv.INTER_NEAREST = 0
} as const;
export type WarpMethodFlagsValue = typeof WarpMethodFlags[keyof typeof WarpMethodFlags];

// Interpolation flag name → cv constant numeric value
export const WARP_METHOD_FLAG_VALUES: Record<string, number> = {
  INTER_LINEAR: 1,
  INTER_CUBIC: 2,
  INTER_NEAREST: 0,
};

// Zone preset names for alignment anchors
export const ZonePreset = {
  topLeftDot: 'topLeftDot',
  topRightDot: 'topRightDot',
  bottomRightDot: 'bottomRightDot',
  bottomLeftDot: 'bottomLeftDot',
  topLeftMarker: 'topLeftMarker',
  topRightMarker: 'topRightMarker',
  bottomRightMarker: 'bottomRightMarker',
  bottomLeftMarker: 'bottomLeftMarker',
  topLeftLMarker: 'topLeftLMarker',
  topRightLMarker: 'topRightLMarker',
  bottomRightLMarker: 'bottomRightLMarker',
  bottomLeftLMarker: 'bottomLeftLMarker',
  topLine: 'topLine',
  leftLine: 'leftLine',
  bottomLine: 'bottomLine',
  rightLine: 'rightLine',
} as const;
export type ZonePresetValue = typeof ZonePreset[keyof typeof ZonePreset];

export const L_MARKER_ZONE_TYPES_IN_ORDER: ZonePresetValue[] = [
  ZonePreset.topLeftLMarker,
  ZonePreset.topRightLMarker,
  ZonePreset.bottomRightLMarker,
  ZonePreset.bottomLeftLMarker,
];

// Selector types for point extraction from zones
export const SelectorType = {
  SELECT_TOP_LEFT: 'SELECT_TOP_LEFT',
  SELECT_TOP_RIGHT: 'SELECT_TOP_RIGHT',
  SELECT_BOTTOM_RIGHT: 'SELECT_BOTTOM_RIGHT',
  SELECT_BOTTOM_LEFT: 'SELECT_BOTTOM_LEFT',
  SELECT_CENTER: 'SELECT_CENTER',
  LINE_INNER_EDGE: 'LINE_INNER_EDGE',
  LINE_OUTER_EDGE: 'LINE_OUTER_EDGE',
  L_INNER_CORNER: 'L_INNER_CORNER',
} as const;
export type SelectorTypeValue = typeof SelectorType[keyof typeof SelectorType];

export const SELECTOR_TYPES_IN_ORDER: SelectorTypeValue[] = [
  SelectorType.SELECT_TOP_LEFT,
  SelectorType.SELECT_TOP_RIGHT,
  SelectorType.SELECT_BOTTOM_RIGHT,
  SelectorType.SELECT_BOTTOM_LEFT,
  SelectorType.SELECT_CENTER,
  SelectorType.LINE_INNER_EDGE,
  SelectorType.LINE_OUTER_EDGE,
  SelectorType.L_INNER_CORNER,
];
