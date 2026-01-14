/**
 * Processor constants and enums.
 *
 * TypeScript port of src/processors/constants.py
 * Maintains 1:1 correspondence with Python implementation.
 */

// Edge types for rectangle/contour operations
export const EdgeType = {
  TOP: 'TOP',
  RIGHT: 'RIGHT',
  BOTTOM: 'BOTTOM',
  LEFT: 'LEFT',
} as const;

export type EdgeTypeValue = (typeof EdgeType)[keyof typeof EdgeType];

export const EDGE_TYPES_IN_ORDER: EdgeTypeValue[] = [
  EdgeType.TOP,
  EdgeType.RIGHT,
  EdgeType.BOTTOM,
  EdgeType.LEFT,
];

// Zone presets for markers, dots, and lines
export const ZonePreset = {
  topLeftDot: 'topLeftDot',
  topRightDot: 'topRightDot',
  bottomRightDot: 'bottomRightDot',
  bottomLeftDot: 'bottomLeftDot',
  topLeftMarker: 'topLeftMarker',
  topRightMarker: 'topRightMarker',
  bottomRightMarker: 'bottomRightMarker',
  bottomLeftMarker: 'bottomLeftMarker',
  topLine: 'topLine',
  leftLine: 'leftLine',
  bottomLine: 'bottomLine',
  rightLine: 'rightLine',
} as const;

export type ZonePresetValue = (typeof ZonePreset)[keyof typeof ZonePreset];

export const DOT_ZONE_TYPES_IN_ORDER: ZonePresetValue[] = [
  ZonePreset.topLeftDot,
  ZonePreset.topRightDot,
  ZonePreset.bottomRightDot,
  ZonePreset.bottomLeftDot,
];

export const MARKER_ZONE_TYPES_IN_ORDER: ZonePresetValue[] = [
  ZonePreset.topLeftMarker,
  ZonePreset.topRightMarker,
  ZonePreset.bottomRightMarker,
  ZonePreset.bottomLeftMarker,
];

export const LINE_ZONE_TYPES_IN_ORDER: ZonePresetValue[] = [
  ZonePreset.topLine,
  ZonePreset.rightLine,
  ZonePreset.bottomLine,
  ZonePreset.leftLine,
];

// Scanner types
export const ScannerType = {
  PATCH_DOT: 'PATCH_DOT',
  PATCH_LINE: 'PATCH_LINE',
  TEMPLATE_MATCH: 'TEMPLATE_MATCH',
} as const;

export type ScannerTypeValue = (typeof ScannerType)[keyof typeof ScannerType];

// Selector types for point selection
export const SelectorType = {
  SELECT_CENTER: 'SELECT_CENTER',
  SELECT_TOP_LEFT: 'SELECT_TOP_LEFT',
  SELECT_TOP_RIGHT: 'SELECT_TOP_RIGHT',
  SELECT_BOTTOM_LEFT: 'SELECT_BOTTOM_LEFT',
  SELECT_BOTTOM_RIGHT: 'SELECT_BOTTOM_RIGHT',
  LINE_INNER_EDGE: 'LINE_INNER_EDGE',
  LINE_OUTER_EDGE: 'LINE_OUTER_EDGE',
} as const;

export type SelectorTypeValue = (typeof SelectorType)[keyof typeof SelectorType];

// Warp methods
export const WarpMethod = {
  HOMOGRAPHY: 'HOMOGRAPHY',
  PERSPECTIVE_TRANSFORM: 'PERSPECTIVE_TRANSFORM',
  DOC_REFINE: 'DOC_REFINE',
  REMAP: 'REMAP',
} as const;

export type WarpMethodValue = (typeof WarpMethod)[keyof typeof WarpMethod];

// Target edges for lines (used in alignment)
export const TARGET_EDGE_FOR_LINE: Record<ZonePresetValue, EdgeTypeValue> = {
  [ZonePreset.topLine]: EdgeType.TOP,
  [ZonePreset.rightLine]: EdgeType.RIGHT,
  [ZonePreset.bottomLine]: EdgeType.BOTTOM,
  [ZonePreset.leftLine]: EdgeType.LEFT,
  // Dots and markers don't have target edges
  [ZonePreset.topLeftDot]: EdgeType.TOP,
  [ZonePreset.topRightDot]: EdgeType.TOP,
  [ZonePreset.bottomRightDot]: EdgeType.BOTTOM,
  [ZonePreset.bottomLeftDot]: EdgeType.BOTTOM,
  [ZonePreset.topLeftMarker]: EdgeType.TOP,
  [ZonePreset.topRightMarker]: EdgeType.TOP,
  [ZonePreset.bottomRightMarker]: EdgeType.BOTTOM,
  [ZonePreset.bottomLeftMarker]: EdgeType.BOTTOM,
};

// Target endpoints for edges (used in warping)
// Maps each edge to zone presets and the index of the point to use from that zone
// "ALL" means use all points from that zone contour
export const TARGET_ENDPOINTS_FOR_EDGES: Record<EdgeTypeValue, [ZonePresetValue, number | 'ALL'][]> = {
  [EdgeType.TOP]: [
    [ZonePreset.topLeftDot, 0],
    [ZonePreset.topLeftMarker, 0],
    [ZonePreset.leftLine, -1],
    [ZonePreset.topLine, 'ALL'],
    [ZonePreset.rightLine, 0],
    [ZonePreset.topRightDot, 0],
    [ZonePreset.topRightMarker, 0],
  ],
  [EdgeType.RIGHT]: [
    [ZonePreset.topRightDot, 0],
    [ZonePreset.topRightMarker, 0],
    [ZonePreset.topLine, -1],
    [ZonePreset.rightLine, 'ALL'],
    [ZonePreset.bottomLine, 0],
    [ZonePreset.bottomRightDot, 0],
    [ZonePreset.bottomRightMarker, 0],
  ],
  [EdgeType.LEFT]: [
    [ZonePreset.bottomLeftDot, 0],
    [ZonePreset.bottomLeftMarker, 0],
    [ZonePreset.bottomLine, -1],
    [ZonePreset.leftLine, 'ALL'],
    [ZonePreset.topLine, 0],
    [ZonePreset.topLeftDot, 0],
    [ZonePreset.topLeftMarker, 0],
  ],
  [EdgeType.BOTTOM]: [
    [ZonePreset.bottomRightDot, 0],
    [ZonePreset.bottomRightMarker, 0],
    [ZonePreset.rightLine, -1],
    [ZonePreset.bottomLine, 'ALL'],
    [ZonePreset.leftLine, 0],
    [ZonePreset.bottomLeftDot, 0],
    [ZonePreset.bottomLeftMarker, 0],
  ],
};

