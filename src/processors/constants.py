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


ZonePreset = DotMap(
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

DOT_ZONE_TYPES_IN_ORDER = [
    ZonePreset.topLeftDot,
    ZonePreset.topRightDot,
    ZonePreset.bottomRightDot,
    ZonePreset.bottomLeftDot,
]

MARKER_ZONE_TYPES_IN_ORDER = [
    ZonePreset.topLeftMarker,
    ZonePreset.topRightMarker,
    ZonePreset.bottomRightMarker,
    ZonePreset.bottomLeftMarker,
]

LINE_ZONE_TYPES_IN_ORDER = [
    ZonePreset.topLine,
    ZonePreset.rightLine,
    ZonePreset.bottomLine,
    ZonePreset.leftLine,
]

TARGET_EDGE_FOR_LINE = {
    ZonePreset.topLine: EdgeType.TOP,
    ZonePreset.rightLine: EdgeType.RIGHT,
    ZonePreset.bottomLine: EdgeType.BOTTOM,
    ZonePreset.leftLine: EdgeType.LEFT,
}

# This defines the precedence for composing ordered points in the edge_contours_map
TARGET_ENDPOINTS_FOR_EDGES = {
    EdgeType.TOP: [
        [ZonePreset.topLeftDot, 0],
        [ZonePreset.topLeftMarker, 0],
        [ZonePreset.leftLine, -1],
        [ZonePreset.topLine, "ALL"],
        [ZonePreset.rightLine, 0],
        [ZonePreset.topRightDot, 0],
        [ZonePreset.topRightMarker, 0],
    ],
    EdgeType.RIGHT: [
        [ZonePreset.topRightDot, 0],
        [ZonePreset.topRightMarker, 0],
        [ZonePreset.topLine, -1],
        [ZonePreset.rightLine, "ALL"],
        [ZonePreset.bottomLine, 0],
        [ZonePreset.bottomRightDot, 0],
        [ZonePreset.bottomRightMarker, 0],
    ],
    EdgeType.LEFT: [
        [ZonePreset.bottomLeftDot, 0],
        [ZonePreset.bottomLeftMarker, 0],
        [ZonePreset.bottomLine, -1],
        [ZonePreset.leftLine, "ALL"],
        [ZonePreset.topLine, 0],
        [ZonePreset.topLeftDot, 0],
        [ZonePreset.topLeftMarker, 0],
    ],
    EdgeType.BOTTOM: [
        [ZonePreset.bottomRightDot, 0],
        [ZonePreset.bottomRightMarker, 0],
        [ZonePreset.rightLine, -1],
        [ZonePreset.bottomLine, "ALL"],
        [ZonePreset.leftLine, 0],
        [ZonePreset.bottomLeftDot, 0],
        [ZonePreset.bottomLeftMarker, 0],
    ],
}


FieldDetectionType = DotMap(
    {
        "BUBBLES_THRESHOLD": "BUBBLES_THRESHOLD",
        "BUBBLES_BLOB": "BUBBLES_BLOB",
        "OCR": "OCR",
        # "PHOTO_BLOB": "PHOTO_BLOB",
        "BARCODE_QR": "BARCODE_QR",
    },
    _dynamic=False,
)
FIELD_DETECTION_TYPES_IN_ORDER = [
    FieldDetectionType.BUBBLES_THRESHOLD,
    FieldDetectionType.OCR,
    # FieldDetectionType.BUBBLES_BLOB,
    # FieldDetectionType.PHOTO_BLOB,
    # FieldDetectionType.BARCODE_QR,
]
ScannerType = DotMap(
    {
        "PATCH_DOT": "PATCH_DOT",
        "PATCH_LINE": "PATCH_LINE",
        "TEMPLATE_MATCH": "TEMPLATE_MATCH",
        # TODO: OCR, QR
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
