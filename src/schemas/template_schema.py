from src.processors.constants import (
    MARKER_AREA_TYPES_IN_ORDER,
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
from src.utils.constants import BUILTIN_FIELD_TYPES

margins_schema = {
    "description": "The margins to use around a box",
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
box_area_description = {
    "description": "The description of a box area on the image",
    "type": "object",
    "required": ["origin", "dimensions", "margins"],
    "additionalProperties": False,
    "properties": {
        "origin": two_positive_integers,
        "dimensions": two_positive_integers,
        "margins": margins_schema,
    },
}

scan_area_template = {
    "description": "The ready-made template to use to prefill description of a scanArea",
    "type": "string",
    "enum": ["CUSTOM", *AreaTemplate.values()],
}

custom_marker_options = {
    "markerDimensions": {
        **two_positive_integers,
        "description": "The dimensions of the omr marker",
    },
    "referenceImage": {
        "description": "The relative path to reference image of the omr marker",
        "type": "string",
    },
}

crop_on_marker_custom_options_schema = {
    "description": "Custom options for the scannerType TEMPLATE_MATCH",
    "type": "object",
    "additionalProperties": False,
    "properties": {**custom_marker_options},
}

# TODO: actually this one needs to be specific to processor.type
common_custom_options_schema = {
    "description": "Custom options based on the scannerType",
    "type": "object",
    # TODO: add conditional properties here like maxPoints and excludeFromCropping here based on scannerType
    # expand conditionally: crop_on_marker_custom_options_schema
}


point_selector_patch_area_description = {
    **box_area_description,
    "description": "The detailed description of a patch area with an optional point selector",
    "additionalProperties": False,
    "properties": {
        **box_area_description["properties"],
        "selector": {
            "type": "string",
            "enum": [*SELECTOR_TYPES_IN_ORDER],
        },
    },
}


marker_area_description = {
    **point_selector_patch_area_description,
    "description": "The detailed description of a patch area for a custom marker",
    "additionalProperties": False,
    "properties": {
        **point_selector_patch_area_description["properties"],
        "selector": {
            "type": "string",
            "enum": [*SELECTOR_TYPES_IN_ORDER],
        },
        "customOptions": crop_on_marker_custom_options_schema,
    },
}


scan_area_description = {
    **point_selector_patch_area_description,
    "description": "The detailed description of a scanArea's coordinates and purpose",
    # TODO: "required": [...],
    "additionalProperties": False,
    "properties": {
        **point_selector_patch_area_description["properties"],
        "label": {
            "description": "The label to use for the scanArea",
            "type": "string",
        },
        "selectorMargins": {
            **margins_schema,
            "description": "The margins around the scanArea's box at provided origin",
        },
        "scannerType": {
            "description": "The scanner type to use in the given scan area",
            "type": "string",
            "enum": SCANNER_TYPES_IN_ORDER,
        },
        "maxPoints": {
            **positive_integer,
            "description": "The maximum points to pick from the given scanArea",
        },
    },
}
scan_areas_object = {
    "description": "The schema of a scanArea",
    "type": "array",
    "items": {
        "type": "object",
        "required": ["areaTemplate"],
        "additionalProperties": False,
        "properties": {
            "areaTemplate": scan_area_template,
            "areaDescription": scan_area_description,
            "customOptions": common_custom_options_schema,
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
    "dotBlurKernel": True,
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
    "description": "Custom tuning options for the CropOnDotLines pre-processor",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        **crop_on_markers_tuning_options_available_keys,
        **warp_on_points_tuning_options,
        "dotBlurKernel": {
            **two_positive_integers,
            "description": "The size of the kernel to use for blurring in each dot's scanArea",
        },
        "dotKernel": {
            **two_positive_integers,
            "description": "The size of the morph kernel to use for smudging each dot",
        },
        "lineKernel": {
            **two_positive_integers,
            "description": "The size of the morph kernel to use for smudging each line",
        },
        "dotThreshold": {
            **positive_number,
            "description": "The threshold to apply for clearing out the noise near a dot after smudging",
        },
        "lineThreshold": {
            **positive_number,
            "description": "The threshold to apply for clearing out the noise near a line after smudging",
        },
    },
}
crop_on_four_markers_tuning_options = {
    "description": "Custom tuning options for the CropOnCustomMarkers pre-processor",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        **crop_on_markers_tuning_options_available_keys,
        **warp_on_points_tuning_options,
        "apply_erode_subtract": {
            "description": "A boolean to enable erosion for (sometimes) better marker detection",
            "type": "boolean",
        },
        "marker_rescale_range": {
            **two_positive_integers,
            "description": "The range of rescaling in percentage",
        },
        "marker_rescale_steps": {
            **positive_integer,
            "description": "The number of rescaling steps",
        },
        "min_matching_threshold": {
            "description": "The threshold for template matching",
            "type": "number",
        },
    },
}

common_field_block_properties = {
    # Common properties here
    "emptyValue": {
        "description": "The custom empty bubble value to use in this field block",
        "type": "string",
    },
    "fieldType": {
        "description": "The field type to use from a list of ready-made types as well as the custom type",
        "type": "string",
        "enum": [*list(BUILTIN_FIELD_TYPES.keys()), "CUSTOM", "BARCODE"],
    },
}
traditional_field_block_properties = {
    **common_field_block_properties,
    "bubbleDimensions": {
        **two_positive_numbers,
        "description": "The custom dimensions for the bubbles in the current field block: [width, height]",
    },
    "bubblesGap": {
        **positive_number,
        "description": "The gap between two bubbles(top-left to top-left) in the current field block",
    },
    "bubbleValues": {
        **ARRAY_OF_STRINGS,
        "description": "The ordered array of values to use for given bubbles per field in this field block",
    },
    "direction": {
        "description": "The direction of expanding the bubbles layout in this field block",
        "type": "string",
        "enum": ["horizontal", "vertical"],
    },
    "fieldLabels": {
        "description": "The ordered array of labels to use for given fields in this field block",
        "type": "array",
        "items": FIELD_STRING_TYPE,
    },
    "labelsGap": {
        **positive_number,
        "description": "The gap between two labels(top-left to top-left) in the current field block",
    },
    "origin": {
        **two_positive_integers,
        "description": "The top left point of the first bubble in this field block",
    },
}

many_field_blocks_description = {
    "description": "Each fieldBlock denotes a small group of adjacent fields",
    "type": "object",
    "patternProperties": {
        "^.*$": {
            "description": "The key is a unique name for the field block",
            "type": "object",
            "required": [
                "origin",
                "bubblesGap",
                "labelsGap",
                "fieldLabels",
            ],
            "allOf": [
                {
                    "if": {
                        "properties": {
                            "fieldType": {
                                "enum": [
                                    *list(BUILTIN_FIELD_TYPES.keys()),
                                ]
                            }
                        },
                    },
                    "then": {
                        "required": ["fieldType"],
                        "additionalProperties": False,
                        "properties": traditional_field_block_properties,
                    },
                },
                {
                    "if": {
                        "properties": {"fieldType": {"const": "CUSTOM"}},
                    },
                    "then": {
                        "required": [
                            "bubbleValues",
                            "direction",
                            "fieldType",
                        ],
                        "additionalProperties": False,
                        "properties": traditional_field_block_properties,
                    },
                },
                {
                    "if": {
                        "properties": {"fieldType": {"const": "BARCODE"}},
                    },
                    "then": {
                        # TODO: move barcode specific properties into this if-else
                        "required": [
                            "scanArea",
                            "fieldType",
                            "fieldLabel",
                            # TODO: "failIfNotFound"
                            # "emptyValue",
                        ],
                        "additionalProperties": False,
                        "properties": {
                            **common_field_block_properties,
                            "scanArea": box_area_description,
                            "fieldLabel": {"type": "string"},
                        },
                    },
                },
                # TODO: support for PHOTO_BLOB, OCR custom fields here
            ],
            "properties": common_field_block_properties,
        }
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
            "description": "The default dimensions for the bubbles in the template overlay: [width, height]",
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
            "description": "The ordered list of columns to be contained in the output csv(default order: alphabetical)",
            "type": "array",
            "items": FIELD_STRING_TYPE,
        },
        "templateDimensions": {
            **two_positive_integers,
            "description": "The dimensions(width, height) to which the page will be resized to before applying template",
        },
        "processingImageShape": {
            **two_positive_integers,
            "description": "Shape of the processing image after all the pre-processors are applied: [height, width]",
        },
        "outputImageShape": {
            **two_positive_integers,
            "description": "Shape of the final output image: [height, width]",
        },
        "preProcessors": {
            "description": "Custom configuration values to use in the template's directory",
            "type": "array",
            "items": {
                "type": "object",
                **pre_processor_if_required_attrs,
                "additionalProperties": True,
                "properties": {
                    # Common properties to be used here
                    "name": {
                        "description": "The name of the pre-processor to use",
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
                        "description": "The options to pass to the pre-processor",
                        "type": "object",
                        "additionalProperties": True,
                        "properties": {
                            # Note: common properties across all preprocessors items can stay here
                            "processingImageShape": {
                                **two_positive_integers,
                                "description": "Shape of the processing image for the current pre-processors: [height, width]",
                            },
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
                                    "description": "Options for the CropPage pre-processor",
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        **pre_processor_options_available_keys,
                                        # TODO: support DOC_REFINE warpMethod
                                        "tuningOptions": True,
                                        "morphKernel": {
                                            **two_positive_integers,
                                            "description": "The size of the morph kernel used for smudging the page",
                                        },
                                        # TODO: support for maxPointsPerEdge
                                        "maxPointsPerEdge": {
                                            **two_positive_integers,
                                            "description": "Max number of control points to use in one edge",
                                        },
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
                                    "description": "Options for the FeatureBasedAlignment pre-processor",
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        **pre_processor_options_available_keys,
                                        "2d": {
                                            "description": "Uses warpAffine if True, otherwise uses warpPerspective",
                                            "type": "boolean",
                                        },
                                        "goodMatchPercent": {
                                            "description": "Threshold for the match percentage",
                                            "type": "number",
                                        },
                                        "maxFeatures": {
                                            "description": "Maximum number of matched features to consider",
                                            "type": "integer",
                                        },
                                        "reference": {
                                            "description": "Relative path to the reference image",
                                            "type": "string",
                                        },
                                        "matcherType": {
                                            "description": "Type of the matcher to use",
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
                                    "description": "Options for the GaussianBlur pre-processor",
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        **pre_processor_options_available_keys,
                                        "kSize": {
                                            **two_positive_integers,
                                            "description": "Size of the kernel",
                                        },
                                        "sigmaX": {
                                            "description": "Value of sigmaX in fraction",
                                            "type": "number",
                                        },
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
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        **pre_processor_options_available_keys,
                                        "gamma": {
                                            **zero_to_one_number,
                                            "description": "The value for gamma parameter",
                                        },
                                        "high": {
                                            **zero_to_one_number,
                                            "description": "The value for high parameter",
                                        },
                                        "low": {
                                            **zero_to_one_number,
                                            "description": "The value for low parameter",
                                        },
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
                                    "description": "Options for the MedianBlur pre-processor",
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        **pre_processor_options_available_keys,
                                        "kSize": {"type": "integer"},
                                    },
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
                                    "description": "Options for the WarpOnPoints pre-processor",
                                    **warp_on_points_options_if_required_attrs,
                                    "type": "object",
                                    "required": [],
                                    "additionalProperties": False,
                                    "properties": {
                                        **warp_on_points_options_available_keys,
                                        "pointsLayout": {
                                            "description": "The type of layout of the scanAreas for finding anchor points",
                                            "type": "string",
                                            "enum": points_layout_types,
                                        },
                                        "defaultSelector": {
                                            "description": "The default points selector for the given scanAreas",
                                            "type": "string",
                                            "enum": default_points_selector_types,
                                        },
                                        "enableCropping": {
                                            "description": "Whether to crop the image to a bounding box of the given anchor points",
                                            "type": "boolean",
                                        },
                                        "tuningOptions": {
                                            "description": "Custom tuning options for the WarpOnPoints pre-processor",
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
                                    "description": "Options for the CropOnMarkers pre-processor",
                                    "type": "object",
                                    # Note: "required" key is retrieved from crop_on_markers_options_if_required_attrs
                                    **crop_on_markers_options_if_required_attrs,
                                    "additionalProperties": True,
                                    "properties": {
                                        # Note: the keys need to match with crop_on_markers_options_available_keys
                                        **crop_on_markers_options_available_keys,
                                        "scanAreas": scan_areas_object,
                                        "defaultSelector": {
                                            "description": "The default points selector for the given scanAreas",
                                            "type": "string",
                                            "enum": default_points_selector_types,
                                        },
                                        "type": {
                                            "description": "The type of the Cropping instance to use",
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
                                                    "referenceImage",
                                                    "markerDimensions",
                                                    # *MARKER_AREA_TYPES_IN_ORDER,
                                                ],
                                                "additionalProperties": False,
                                                "properties": {
                                                    **crop_on_markers_options_available_keys,
                                                    **custom_marker_options,
                                                    **{
                                                        area_template: marker_area_description
                                                        for area_template in MARKER_AREA_TYPES_IN_ORDER
                                                    },
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
                                                    "leftLine": point_selector_patch_area_description,
                                                    "topRightDot": point_selector_patch_area_description,
                                                    "bottomRightDot": point_selector_patch_area_description,
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
                                                    "rightLine": point_selector_patch_area_description,
                                                    "topLeftDot": point_selector_patch_area_description,
                                                    "bottomLeftDot": point_selector_patch_area_description,
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
                                                "required": [
                                                    "leftLine",
                                                    "rightLine",
                                                ],
                                                "additionalProperties": False,
                                                "properties": {
                                                    **crop_on_markers_options_available_keys,
                                                    "tuningOptions": crop_on_dot_lines_tuning_options,
                                                    "leftLine": point_selector_patch_area_description,
                                                    "rightLine": point_selector_patch_area_description,
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
                                                    "topRightDot": point_selector_patch_area_description,
                                                    "bottomRightDot": point_selector_patch_area_description,
                                                    "topLeftDot": point_selector_patch_area_description,
                                                    "bottomLeftDot": point_selector_patch_area_description,
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
            **many_field_blocks_description,
            "description": "The default field block to apply and read before applying any matcher on the fields response.",
        },
        "conditionalSets": {
            "description": "An array of field block sets with their conditions. These will override the default values in case of any conflict",
            "type": "array",
            "items": {
                "description": "Each item represents a conditional layout of field blocks",
                "type": "object",
                "required": ["name", "matcher", "fieldBlocks"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "matcher": {
                        "description": "Mapping response fields from default layout to the set name",
                        "type": "object",
                        "required": ["formatString", "matchRegex"],
                        "additionalProperties": False,
                        "properties": {
                            "formatString": {
                                "description": "Format string composed of the response variables to apply the regex on e.g. '{roll}-{barcode}'",
                                "type": "string",
                            },
                            # Example: match last four characters ".*-SET1"
                            "matchRegex": {
                                "description": "Mapping to use on the composed field string",
                                "type": "string",
                                "format": "regex",
                            },
                        },
                    },
                    "fieldBlocks": {
                        **many_field_blocks_description,
                        "description": "The custom field blocks layout to apply if given matcher is satisfied",
                    },
                },
            },
        },
        "sortFiles": {
            "description": "Configuration to sort images/files based on field responses, QR codes, barcodes, etc and a regex mapping",
            "type": "object",
            "allOf": [
                {"required": ["enabled"]},
                {
                    "if": {
                        "properties": {"enabled": {"const": True}},
                    },
                    "then": {
                        "required": [
                            "enabled",
                            "sortMode",
                            "outputDirectory",
                            "fileMappings",
                        ]
                    },
                },
            ],
            "additionalProperties": False,
            "properties": {
                "enabled": {
                    "description": "Whether to enable sorting. Note that file copies/movements are irreversible once enabled",
                    "type": "boolean",
                },
                "sortMode": {
                    "description": "Whether to copy files or move files",
                    "type": "string",
                    "enum": [
                        "COPY",
                        "MOVE",
                    ],
                },
                # TODO: ignore outputDirectory when resolved during template reading
                "outputDirectory": {
                    "description": "Relative path of the directory to use to sort the files",
                    "type": "string",
                },
                "fileMapping": {
                    "description": "A mapping from regex to the relative file path to use",
                    "type": "object",
                    "required": ["formatString"],
                    "additionalProperties": False,
                    "properties": {
                        # "{barcode}", "{roll}-{barcode}"
                        "formatString": {
                            "description": "Format string composed of the response variables to apply the regex on e.g. '{roll}-{barcode}'",
                            "type": "string",
                        },
                        # Example: extract first four characters "(\w{4}).*"
                        "extractRegex": {
                            "description": "Mapping to use on the composed field string",
                            "type": "string",
                            "format": "regex",
                        },
                        # TODO: support for "abortIfNotMatched"?
                        "capturedString": {
                            "description": "The captured groups string to use for replacement",
                            "type": "string",
                        },
                    },
                },
            },
        },
    },
}
