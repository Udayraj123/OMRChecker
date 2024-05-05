from dotmap import DotMap

EdgeType = DotMap(
    {
        "TOP": "TOP",
        "RIGHT": "RIGHT",
        "BOTTOM": "BOTTOM",
        "LEFT": "LEFT",
    },
    _dynamic=False,
)

EDGE_TYPES_IN_ORDER = [EdgeType.TOP, EdgeType.RIGHT, EdgeType.BOTTOM, EdgeType.LEFT]


AreaPreset = DotMap(
    {
        "topLeftDot": "topLeftDot",
        "topRightDot": "topRightDot",
        "bottomRightDot": "bottomRightDot",
        "bottomLeftDot": "bottomLeftDot",
        "topLeftMarker": "topLeftMarker",
        "topRightMarker": "topRightMarker",
        "bottomRightMarker": "bottomRightMarker",
        "bottomLeftMarker": "bottomLeftMarker",
        "topLine": "topLine",
        "leftLine": "leftLine",
        "bottomLine": "bottomLine",
        "rightLine": "rightLine",
    },
    _dynamic=False,
)

DOT_AREA_TYPES_IN_ORDER = [
    AreaPreset.topLeftDot,
    AreaPreset.topRightDot,
    AreaPreset.bottomRightDot,
    AreaPreset.bottomLeftDot,
]

MARKER_AREA_TYPES_IN_ORDER = [
    AreaPreset.topLeftMarker,
    AreaPreset.topRightMarker,
    AreaPreset.bottomRightMarker,
    AreaPreset.bottomLeftMarker,
]

LINE_AREA_TYPES_IN_ORDER = [
    AreaPreset.topLine,
    AreaPreset.rightLine,
    AreaPreset.bottomLine,
    AreaPreset.leftLine,
]

TARGET_EDGE_FOR_LINE = {
    AreaPreset.topLine: EdgeType.TOP,
    AreaPreset.rightLine: EdgeType.RIGHT,
    AreaPreset.bottomLine: EdgeType.BOTTOM,
    AreaPreset.leftLine: EdgeType.LEFT,
}

# This defines the precedence for composing ordered points in the edge_contours_map
TARGET_ENDPOINTS_FOR_EDGES = {
    EdgeType.TOP: [
        [AreaPreset.topLeftDot, 0],
        [AreaPreset.topLeftMarker, 0],
        [AreaPreset.leftLine, -1],
        [AreaPreset.topLine, "ALL"],
        [AreaPreset.rightLine, 0],
        [AreaPreset.topRightDot, 0],
        [AreaPreset.topRightMarker, 0],
    ],
    EdgeType.RIGHT: [
        [AreaPreset.topRightDot, 0],
        [AreaPreset.topRightMarker, 0],
        [AreaPreset.topLine, -1],
        [AreaPreset.rightLine, "ALL"],
        [AreaPreset.bottomLine, 0],
        [AreaPreset.bottomRightDot, 0],
        [AreaPreset.bottomRightMarker, 0],
    ],
    EdgeType.LEFT: [
        [AreaPreset.bottomLeftDot, 0],
        [AreaPreset.bottomLeftMarker, 0],
        [AreaPreset.bottomLine, -1],
        [AreaPreset.leftLine, "ALL"],
        [AreaPreset.topLine, 0],
        [AreaPreset.topLeftDot, 0],
        [AreaPreset.topLeftMarker, 0],
    ],
    EdgeType.BOTTOM: [
        [AreaPreset.bottomRightDot, 0],
        [AreaPreset.bottomRightMarker, 0],
        [AreaPreset.rightLine, -1],
        [AreaPreset.bottomLine, "ALL"],
        [AreaPreset.leftLine, 0],
        [AreaPreset.bottomLeftDot, 0],
        [AreaPreset.bottomLeftMarker, 0],
    ],
}


ScannerType = DotMap(
    {
        "PATCH_DOT": "PATCH_DOT",
        "PATCH_LINE": "PATCH_LINE",
        "TEMPLATE_MATCH": "TEMPLATE_MATCH",
    },
    _dynamic=False,
)

SCANNER_TYPES_IN_ORDER = [
    ScannerType.PATCH_DOT,
    ScannerType.PATCH_LINE,
    ScannerType.TEMPLATE_MATCH,
]


SelectorType = DotMap(
    {
        "SELECT_TOP_LEFT": "SELECT_TOP_LEFT",
        "SELECT_TOP_RIGHT": "SELECT_TOP_RIGHT",
        "SELECT_BOTTOM_RIGHT": "SELECT_BOTTOM_RIGHT",
        "SELECT_BOTTOM_LEFT": "SELECT_BOTTOM_LEFT",
        "SELECT_CENTER": "SELECT_CENTER",
        "LINE_INNER_EDGE": "LINE_INNER_EDGE",
        "LINE_OUTER_EDGE": "LINE_OUTER_EDGE",
    },
    _dynamic=False,
)
SELECTOR_TYPES_IN_ORDER = [
    SelectorType.SELECT_TOP_LEFT,
    SelectorType.SELECT_TOP_RIGHT,
    SelectorType.SELECT_BOTTOM_RIGHT,
    SelectorType.SELECT_BOTTOM_LEFT,
    SelectorType.SELECT_CENTER,
    SelectorType.LINE_INNER_EDGE,
    SelectorType.LINE_OUTER_EDGE,
]
WarpMethodFlags = DotMap(
    {
        "INTER_LINEAR": "INTER_LINEAR",
        "INTER_CUBIC": "INTER_CUBIC",
        "INTER_NEAREST": "INTER_NEAREST",
    },
    _dynamic=False,
)

WarpMethod = DotMap(
    {
        "PERSPECTIVE_TRANSFORM": "PERSPECTIVE_TRANSFORM",
        "HOMOGRAPHY": "HOMOGRAPHY",
        "REMAP_GRIDDATA": "REMAP_GRIDDATA",
        "DOC_REFINE": "DOC_REFINE",
        "WARP_AFFINE": "WARP_AFFINE",
    },
    _dynamic=False,
)
