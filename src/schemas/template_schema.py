from src.processors.constants import (
    SCANNER_TYPES_IN_ORDER,
    SELECTOR_TYPES_IN_ORDER,
    AreaTemplate,
    WarpMethod,
    WarpMethodFlags,
)
from src.schemas.constants import (
    ARRAY_OF_STRINGS,
    FIELD_STRING_TYPE,
    positive_integer,
    positive_number,
    two_positive_integers,
    two_positive_numbers,
    zero_to_one_number,
)
from src.utils.constants import FIELD_TYPES

margins_schema = {
    "type": "object",
    "required": ["top", "right", "bottom", "left"],
    "additionalProperties": False,
    "properties": {
        "top": positive_integer,
        "right": positive_integer,
        "bottom": positive_integer,
        "left": positive_integer,
    },
}
# TODO: deprecate in favor of scan_area_description
patch_area_description = {
    "type": "object",
    "required": ["origin", "dimensions", "margins"],
    "additionalProperties": False,
    "properties": {
        "origin": two_positive_integers,
        "dimensions": two_positive_integers,
        "margins": margins_schema,
        "selector": {
            "type": "string",
            "enum": [*SELECTOR_TYPES_IN_ORDER],
        },
        # TODO: give customOptions via merging the two
    },
}

scan_area_template = {
    "type": "string",
    "enum": ["CUSTOM", *AreaTemplate.values()],
}

scan_area_description = {
    **patch_area_description,
    # TODO: "required": [...],
    "additionalProperties": False,
    "properties": {
        **patch_area_description["properties"],
        "label": {
            "type": "string",
        },
        "selectorMargins": margins_schema,
        "scannerType": {
            "type": "string",
            "enum": SCANNER_TYPES_IN_ORDER,
        },
        "maxPoints": positive_integer,
    },
}
scan_areas_object = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["areaTemplate"],
        "additionalProperties": False,
        "properties": {
            "areaTemplate": scan_area_template,
            "areaDescription": scan_area_description,
            "customOptions": {
                "type": "object"
                # TODO: add conditional properties here like maxPoints and excludeFromCropping here based on scannerType
            },
        },
    },
}

default_points_selector_types = [
    "CENTERS",
    "INNER_WIDTHS",
    "INNER_HEIGHTS",
    "INNER_CORNERS",
    "OUTER_CORNERS",
]

# TODO: deprecate crop_on_marker_types
crop_on_marker_types = [
    "FOUR_MARKERS",
    "ONE_LINE_TWO_DOTS",
    "TWO_DOTS_ONE_LINE",
    "TWO_LINES",
    # TODO: support for "TWO_LINES_HORIZONTAL"
    "FOUR_DOTS",
]

points_layout_types = [
    *crop_on_marker_types,
    "CUSTOM",
]

# if_required_attrs help in suppressing redundant errors from 'allOf'
pre_processor_if_required_attrs = {
    "required": ["name", "options"],
}
crop_on_markers_options_if_required_attrs = {
    "required": ["type"],
}
warp_on_points_options_if_required_attrs = {
    "required": ["scanAreas"],
}
pre_processor_options_available_keys = {"processingImageShape": True}

crop_on_markers_tuning_options_available_keys = {
    "dotThreshold": True,
    "dotKernel": True,
    "lineThreshold": True,
    "dotBlur": True,
    "lineKernel": True,
    "apply_erode_subtract": True,
    "marker_rescale_range": True,
    "marker_rescale_steps": True,
    "min_matching_threshold": True,
}
crop_on_markers_options_available_keys = {
    **pre_processor_options_available_keys,
    "scanAreas": True,
    "defaultSelector": True,
    "tuningOptions": True,
    "type": True,
}

warp_on_points_tuning_options = {
    "warpMethod": {
        "type": "string",
        "enum": [*WarpMethod.values()],
    },
    "warpMethodFlag": {
        "type": "string",
        "enum": [*WarpMethodFlags.values()],
    },
}


warp_on_points_options_available_keys = {
    **pre_processor_options_available_keys,
    "enableCropping": True,
    "defaultSelector": True,
    "scanAreas": True,
    "tuningOptions": True,
}

crop_on_dot_lines_tuning_options = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        **crop_on_markers_tuning_options_available_keys,
        **warp_on_points_tuning_options,
        "dotBlur": positive_integer,
        "dotKernel": two_positive_integers,
        "lineKernel": two_positive_integers,
        "dotThreshold": positive_number,
        "lineThreshold": positive_number,
    },
}
crop_on_four_markers_tuning_options = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        **crop_on_markers_tuning_options_available_keys,
        **warp_on_points_tuning_options,
        "apply_erode_subtract": {"type": "boolean"},
        # Range of rescaling in percentage -
        "marker_rescale_range": two_positive_integers,
        "marker_rescale_steps": positive_integer,
        "min_matching_threshold": {"type": "number"},
    },
}

TEMPLATE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/Udayraj123/OMRChecker/tree/master/src/schemas/template-schema.json",
    "title": "Template Validation Schema",
    "description": "OMRChecker input template schema",
    "type": "object",
    "required": [
        "bubbleDimensions",
        "templateDimensions",
        "preProcessors",
        "fieldBlocks",
    ],
    "additionalProperties": False,
    "properties": {
        "bubbleDimensions": {
            **two_positive_integers,
            "description": "The dimensions of the overlay bubble area: [width, height]",
        },
        "customLabels": {
            "description": "The customLabels contain fields that need to be joined together before generating the results sheet",
            "type": "object",
            "patternProperties": {
                "^.*$": {"type": "array", "items": FIELD_STRING_TYPE}
            },
        },
        "emptyValue": {
            "description": "The value to be used in case of empty bubble detected at global level.",
            "type": "string",
        },
        "outputColumns": {
            "type": "array",
            "items": FIELD_STRING_TYPE,
            "description": "The ordered list of columns to be contained in the output csv(default order: alphabetical)",
        },
        "templateDimensions": {
            **two_positive_integers,
            "description": "The dimensions(width, height) to which the page will be resized to before applying template",
        },
        "outputImageShape": two_positive_integers,
        "processingImageShape": two_positive_integers,
        "preProcessors": {
            "description": "Custom configuration values to use in the template's directory",
            "type": "array",
            "items": {
                "type": "object",
                **pre_processor_if_required_attrs,
                "additionalProperties": True,
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": [
                            "CropOnMarkers",
                            # TODO: "WarpOnPoints",
                            "CropPage",
                            "FeatureBasedAlignment",
                            "GaussianBlur",
                            "Levels",
                            "MedianBlur",
                        ],
                    },
                    "options": {
                        "type": "object",
                        "additionalProperties": True,
                        "properties": {
                            # Note: common properties across all preprocessors items can stay here
                            "processingImageShape": two_positive_integers,
                        },
                    },
                },
                "allOf": [
                    {
                        "if": {
                            "properties": {"name": {"const": "CropPage"}},
                            **pre_processor_if_required_attrs,
                        },
                        "then": {
                            "properties": {
                                "options": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        **pre_processor_options_available_keys,
                                        "tuningOptions": False,
                                        "morphKernel": two_positive_integers,
                                        "maxPointsPerEdge": positive_integer,
                                    },
                                },
                            }
                        },
                    },
                    {
                        "if": {
                            "properties": {"name": {"const": "FeatureBasedAlignment"}},
                            **pre_processor_if_required_attrs,
                        },
                        "then": {
                            "properties": {
                                "options": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        **pre_processor_options_available_keys,
                                        "2d": {"type": "boolean"},
                                        "goodMatchPercent": {"type": "number"},
                                        "maxFeatures": {"type": "integer"},
                                        "reference": {"type": "string"},
                                        "matcherType": {
                                            "type": "string",
                                            "enum": [
                                                "DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING",
                                                "NORM_HAMMING",
                                            ],
                                        },
                                    },
                                    "required": ["reference"],
                                }
                            }
                        },
                    },
                    {
                        "if": {
                            "properties": {"name": {"const": "GaussianBlur"}},
                            **pre_processor_if_required_attrs,
                        },
                        "then": {
                            "properties": {
                                "options": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        **pre_processor_options_available_keys,
                                        "kSize": two_positive_integers,
                                        "sigmaX": {"type": "number"},
                                    },
                                }
                            }
                        },
                    },
                    {
                        "if": {
                            "properties": {"name": {"const": "Levels"}},
                            **pre_processor_if_required_attrs,
                        },
                        "then": {
                            "properties": {
                                "options": {
                                    **pre_processor_options_available_keys,
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "gamma": zero_to_one_number,
                                        "high": zero_to_one_number,
                                        "low": zero_to_one_number,
                                    },
                                }
                            }
                        },
                    },
                    {
                        "if": {
                            "properties": {"name": {"const": "MedianBlur"}},
                            **pre_processor_if_required_attrs,
                        },
                        "then": {
                            "properties": {
                                "options": {
                                    **pre_processor_options_available_keys,
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {"kSize": {"type": "integer"}},
                                }
                            }
                        },
                    },
                    {
                        "if": {
                            "properties": {"name": {"const": "WarpOnPoints"}},
                            **pre_processor_if_required_attrs,
                        },
                        "then": {
                            "properties": {
                                "options": {
                                    **warp_on_points_options_if_required_attrs,
                                    "type": "object",
                                    "required": [],
                                    "additionalProperties": False,
                                    "properties": {
                                        **warp_on_points_options_available_keys,
                                        "pointsLayout": {
                                            "type": "string",
                                            "enum": points_layout_types,
                                        },
                                        "defaultSelector": {
                                            "type": "string",
                                            "enum": default_points_selector_types,
                                        },
                                        "enableCropping": {"type": "boolean"},
                                        "tuningOptions": {
                                            "type": "object",
                                            "required": [],
                                            "additionalProperties": False,
                                            "properties": {
                                                **crop_on_dot_lines_tuning_options[
                                                    "properties"
                                                ],
                                            },
                                        },
                                        "scanAreas": scan_areas_object,
                                    },
                                }
                            }
                        },
                    },
                    {
                        "if": {
                            "properties": {"name": {"const": "CropOnMarkers"}},
                            **pre_processor_if_required_attrs,
                        },
                        "then": {
                            "properties": {
                                "options": {
                                    "type": "object",
                                    # Note: "required" key is retrieved from crop_on_markers_options_if_required_attrs
                                    **crop_on_markers_options_if_required_attrs,
                                    "additionalProperties": True,
                                    "properties": {
                                        # Note: the keys need to match with crop_on_markers_options_available_keys
                                        **crop_on_markers_options_available_keys,
                                        "scanAreas": scan_areas_object,
                                        "defaultSelector": {
                                            "type": "string",
                                            "enum": default_points_selector_types,
                                        },
                                        "type": {
                                            "type": "string",
                                            "enum": crop_on_marker_types,
                                        },
                                    },
                                    "allOf": [
                                        {
                                            "if": {
                                                **crop_on_markers_options_if_required_attrs,
                                                "properties": {
                                                    "type": {"const": "FOUR_MARKERS"}
                                                },
                                            },
                                            "then": {
                                                "required": [
                                                    "relativePath",
                                                    "dimensions",
                                                ],
                                                "additionalProperties": False,
                                                "properties": {
                                                    **crop_on_markers_options_available_keys,
                                                    "dimensions": two_positive_integers,
                                                    "relativePath": {"type": "string"},
                                                    "topRightMarker": patch_area_description,
                                                    "bottomRightMarker": patch_area_description,
                                                    "topLeftMarker": patch_area_description,
                                                    "bottomLeftMarker": patch_area_description,
                                                    "tuningOptions": crop_on_four_markers_tuning_options,
                                                },
                                            },
                                        },
                                        {
                                            "if": {
                                                **crop_on_markers_options_if_required_attrs,
                                                "properties": {
                                                    "type": {
                                                        "const": "ONE_LINE_TWO_DOTS"
                                                    }
                                                },
                                            },
                                            "then": {
                                                # TODO: check that "topLeftDot": False, etc here is not passable
                                                "required": [
                                                    "leftLine",
                                                    "topRightDot",
                                                    "bottomRightDot",
                                                ],
                                                "additionalProperties": False,
                                                "properties": {
                                                    **crop_on_markers_options_available_keys,
                                                    "tuningOptions": crop_on_dot_lines_tuning_options,
                                                    "leftLine": patch_area_description,
                                                    "topRightDot": patch_area_description,
                                                    "bottomRightDot": patch_area_description,
                                                },
                                            },
                                        },
                                        {
                                            "if": {
                                                **crop_on_markers_options_if_required_attrs,
                                                "properties": {
                                                    "type": {
                                                        "const": "TWO_DOTS_ONE_LINE"
                                                    }
                                                },
                                            },
                                            "then": {
                                                "required": [
                                                    "rightLine",
                                                    "topLeftDot",
                                                    "bottomLeftDot",
                                                ],
                                                "additionalProperties": False,
                                                "properties": {
                                                    **crop_on_markers_options_available_keys,
                                                    "tuningOptions": crop_on_dot_lines_tuning_options,
                                                    "rightLine": patch_area_description,
                                                    "topLeftDot": patch_area_description,
                                                    "bottomLeftDot": patch_area_description,
                                                },
                                            },
                                        },
                                        {
                                            "if": {
                                                **crop_on_markers_options_if_required_attrs,
                                                "properties": {
                                                    "type": {"const": "TWO_LINES"}
                                                },
                                            },
                                            "then": {
                                                "required": ["leftLine", "rightLine"],
                                                "additionalProperties": False,
                                                "properties": {
                                                    **crop_on_markers_options_available_keys,
                                                    "tuningOptions": crop_on_dot_lines_tuning_options,
                                                    "leftLine": patch_area_description,
                                                    "rightLine": patch_area_description,
                                                },
                                            },
                                        },
                                        {
                                            "if": {
                                                **crop_on_markers_options_if_required_attrs,
                                                "properties": {
                                                    "type": {"const": "FOUR_DOTS"}
                                                },
                                            },
                                            "then": {
                                                "required": [
                                                    "topRightDot",
                                                    "bottomRightDot",
                                                    "topLeftDot",
                                                    "bottomLeftDot",
                                                ],
                                                "additionalProperties": False,
                                                "properties": {
                                                    **crop_on_markers_options_available_keys,
                                                    "tuningOptions": crop_on_dot_lines_tuning_options,
                                                    "topRightDot": patch_area_description,
                                                    "bottomRightDot": patch_area_description,
                                                    "topLeftDot": patch_area_description,
                                                    "bottomLeftDot": patch_area_description,
                                                },
                                            },
                                        },
                                    ],
                                }
                            }
                        },
                    },
                ],
            },
        },
        "fieldBlocks": {
            "description": "The fieldBlocks denote small groups of adjacent fields",
            "type": "object",
            "patternProperties": {
                "^.*$": {
                    "type": "object",
                    "required": [
                        "origin",
                        "bubblesGap",
                        "labelsGap",
                        "fieldLabels",
                    ],
                    "allOf": [
                        {"required": ["fieldType"]},
                        {
                            "if": {
                                "properties": {"fieldType": {"const": "CUSTOM"}},
                            },
                            "then": {
                                "required": ["bubbleValues", "direction", "fieldType"]
                            },
                        },
                    ],
                    "additionalProperties": False,
                    "properties": {
                        "bubbleDimensions": two_positive_numbers,
                        "bubblesGap": positive_number,
                        "bubbleValues": ARRAY_OF_STRINGS,
                        "direction": {
                            "type": "string",
                            "enum": ["horizontal", "vertical"],
                        },
                        "emptyValue": {"type": "string"},
                        "fieldLabels": {"type": "array", "items": FIELD_STRING_TYPE},
                        "labelsGap": positive_number,
                        "origin": two_positive_integers,
                        "fieldType": {
                            "type": "string",
                            "enum": [*list(FIELD_TYPES.keys()), "CUSTOM"],
                        },
                    },
                }
            },
        },
    },
}
