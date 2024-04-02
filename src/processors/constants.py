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
        "leftLine": "leftLine",
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
    # TODO: support for all kinds of lines
    # AreaTemplate.topLine,
    AreaTemplate.rightLine,
    AreaTemplate.leftLine,
    # AreaTemplate.bottomLine,
]
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
