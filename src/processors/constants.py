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


AreaTemplate = DotMap(
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
    AreaTemplate.topLeftDot,
    AreaTemplate.topRightDot,
    AreaTemplate.bottomRightDot,
    AreaTemplate.bottomLeftDot,
]

MARKER_AREA_TYPES_IN_ORDER = [
    AreaTemplate.topLeftMarker,
    AreaTemplate.topRightMarker,
    AreaTemplate.bottomRightMarker,
    AreaTemplate.bottomLeftMarker,
]

LINE_AREA_TYPES_IN_ORDER = [
    AreaTemplate.topLine,
    AreaTemplate.rightLine,
    AreaTemplate.bottomLine,
    AreaTemplate.leftLine,
]

TARGET_EDGE_FOR_LINE = {
    AreaTemplate.topLine: EdgeType.TOP,
    AreaTemplate.rightLine: EdgeType.RIGHT,
    AreaTemplate.bottomLine: EdgeType.BOTTOM,
    AreaTemplate.leftLine: EdgeType.LEFT,
}

# This defines the precedence for composing ordered points in the edge_contours_map
TARGET_ENDPOINTS_FOR_EDGES = {
    EdgeType.TOP: [
        [AreaTemplate.topLeftDot, 0],
        [AreaTemplate.topLeftMarker, 0],
        [AreaTemplate.leftLine, -1],
        [AreaTemplate.topLine, "ALL"],
        [AreaTemplate.rightLine, 0],
        [AreaTemplate.topRightDot, 0],
        [AreaTemplate.topRightMarker, 0],
    ],
    EdgeType.RIGHT: [
        [AreaTemplate.topRightDot, 0],
        [AreaTemplate.topRightMarker, 0],
        [AreaTemplate.topLine, -1],
        [AreaTemplate.rightLine, "ALL"],
        [AreaTemplate.bottomLine, 0],
        [AreaTemplate.bottomRightDot, 0],
        [AreaTemplate.bottomRightMarker, 0],
    ],
    EdgeType.LEFT: [
        [AreaTemplate.bottomLeftDot, 0],
        [AreaTemplate.bottomLeftMarker, 0],
        [AreaTemplate.bottomLine, -1],
        [AreaTemplate.leftLine, "ALL"],
        [AreaTemplate.topLine, 0],
        [AreaTemplate.topLeftDot, 0],
        [AreaTemplate.topLeftMarker, 0],
    ],
    EdgeType.BOTTOM: [
        [AreaTemplate.bottomRightDot, 0],
        [AreaTemplate.bottomRightMarker, 0],
        [AreaTemplate.rightLine, -1],
        [AreaTemplate.bottomLine, "ALL"],
        [AreaTemplate.leftLine, 0],
        [AreaTemplate.bottomLeftDot, 0],
        [AreaTemplate.bottomLeftMarker, 0],
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
