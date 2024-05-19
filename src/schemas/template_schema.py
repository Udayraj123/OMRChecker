from src.processors.constants import (
    MARKER_AREA_TYPES_IN_ORDER,
    SCANNER_TYPES_IN_ORDER,
    SELECTOR_TYPES_IN_ORDER,
    AreaTemplate,
    WarpMethod,
    WarpMethodFlags,
)
from src.schemas.constants import load_common_defs
from src.utils.constants import BUILTIN_FIELD_TYPES

margins_schema_def = {
    "description": "The margins to use around a box",
    "type": "object",
    "required": ["top", "right", "bottom", "left"],
    "additionalProperties": False,
    "properties": {
        "top": {
            "$ref": "#/$def/positive_integer",
        },
        "right": {
            "$ref": "#/$def/positive_integer",
        },
        "bottom": {
            "$ref": "#/$def/positive_integer",
        },
        "left": {
            "$ref": "#/$def/positive_integer",
        },
    },
}
# TODO: deprecate in favor of _scan_area_description (support barcode as scannerType)
_box_area_description = {
    "description": "The description of a box area on the image",
    "type": "object",
    "required": ["origin", "dimensions", "margins"],
    "additionalProperties": False,
    "properties": {
        "origin": {
            "$ref": "#/$def/two_positive_integers",
        },
        "dimensions": {
            "$ref": "#/$def/two_positive_integers",
        },
        "margins": {"$ref": "#/$def/margins_schema"},
    },
}

_scan_area_template = {
    "description": "The ready-made template to use to prefill description of a scanArea",
    "type": "string",
    "enum": ["CUSTOM", *AreaTemplate.values()],
}

_custom_marker_options = {
    "markerDimensions": {
        "$ref": "#/$def/two_positive_numbers",
        "description": "The dimensions of the omr marker",
    },
    "referenceImage": {
        "description": "The relative path to reference image of the omr marker",
        "type": "string",
    },
}

_crop_on_marker_custom_options_schema = {
    "description": "Custom options for the scannerType TEMPLATE_MATCH",
    "type": "object",
    "additionalProperties": False,
    "properties": {**_custom_marker_options},
}

# TODO: actually this one needs to be specific to processor.type
_common_custom_options_schema = {
    "description": "Custom options based on the scannerType",
    "type": "object",
    # TODO: add conditional properties here like maxPoints and excludeFromCropping here based on scannerType
    # expand conditionally: _crop_on_marker_custom_options_schema
}


point_selector_patch_area_def = {
    **_box_area_description,
    "description": "The detailed description of a patch area with an optional point selector",
    "additionalProperties": False,
    "properties": {
        **_box_area_description["properties"],
        "selector": {
            "type": "string",
            "enum": [*SELECTOR_TYPES_IN_ORDER],
        },
    },
}


marker_area_description_def = {
    **point_selector_patch_area_def,
    "required": ["origin", "margins"],
    "description": "The detailed description of a patch area for a custom marker",
    "additionalProperties": False,
    "properties": {
        **point_selector_patch_area_def["properties"],
        "selector": {
            "type": "string",
            "enum": [*SELECTOR_TYPES_IN_ORDER],
        },
        "customOptions": _crop_on_marker_custom_options_schema,
    },
}


_scan_area_description = {
    **point_selector_patch_area_def,
    "description": "The detailed description of a scanArea's coordinates and purpose",
    # TODO: "required": [...],
    "additionalProperties": False,
    "properties": {
        **point_selector_patch_area_def["properties"],
        "label": {
            "description": "The label to use for the scanArea",
            "type": "string",
        },
        "selectorMargins": {
            "$ref": "#/$def/margins_schema",
            "description": "The margins around the scanArea's box at provided origin",
        },
        "scannerType": {
            "description": "The scanner type to use in the given scan area",
            "type": "string",
            "enum": SCANNER_TYPES_IN_ORDER,
        },
        "maxPoints": {
            "$ref": "#/$def/positive_integer",
            "description": "The maximum points to pick from the given scanArea",
        },
    },
}
scan_areas_array_def = {
    "description": "The schema of a scanArea",
    "type": "array",
    "items": {
        "type": "object",
        "required": ["areaTemplate"],
        "additionalProperties": False,
        "properties": {
            "areaTemplate": _scan_area_template,
            "areaDescription": _scan_area_description,
            "customOptions": _common_custom_options_schema,
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

crop_on_dot_lines_tuning_options_def = {
    "description": "Custom tuning options for the CropOnDotLines pre-processor",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        **crop_on_markers_tuning_options_available_keys,
        **warp_on_points_tuning_options,
        "dotBlurKernel": {
            "$ref": "#/$def/two_positive_numbers",
            "description": "The size of the kernel to use for blurring in each dot's scanArea",
        },
        "dotKernel": {
            "$ref": "#/$def/two_positive_numbers",
            "description": "The size of the morph kernel to use for smudging each dot",
        },
        "lineKernel": {
            "$ref": "#/$def/two_positive_numbers",
            "description": "The size of the morph kernel to use for smudging each line",
        },
        "dotThreshold": {
            "$ref": "#/$def/positive_number",
            "description": "The threshold to apply for clearing out the noise near a dot after smudging",
        },
        "lineThreshold": {
            "$ref": "#/$def/positive_number",
            "description": "The threshold to apply for clearing out the noise near a line after smudging",
        },
    },
}
crop_on_four_markers_tuning_options_def = {
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
            "$ref": "#/$def/two_positive_numbers",
            "description": "The range of rescaling in percentage",
        },
        "marker_rescale_steps": {
            "$ref": "#/$def/positive_integer",
            "description": "The number of rescaling steps",
        },
        "min_matching_threshold": {
            "description": "The threshold for template matching",
            "type": "number",
        },
    },
}

_common_field_block_properties = {
    # Common properties here
    "emptyValue": {
        "description": "The custom empty bubble value to use in this field block",
        "type": "string",
    },
    "fieldType": {
        "description": "The field type to use from a list of ready-made types as well as the custom type",
        "type": "string",
        "enum": [*list(BUILTIN_FIELD_TYPES.keys()), "CUSTOM_BUBBLES", "BARCODE", "OCR"],
    },
}

_traditional_field_block_properties = {
    **_common_field_block_properties,
    "bubbleDimensions": {
        "$ref": "#/$def/two_positive_numbers",
        "description": "The custom dimensions for the bubbles in the current field block: [width, height]",
    },
    "bubblesGap": {
        "$ref": "#/$def/positive_number",
        "description": "The gap between two bubbles(top-left to top-left) in the current field block",
    },
    "bubbleValues": {
        "$ref": "#/$def/array_of_strings",
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
        "items": {
            "$ref": "#/$def/field_string_type",
        },
    },
    "labelsGap": {
        "$ref": "#/$def/positive_number",
        "description": "The gap between two labels(top-left to top-left) in the current field block",
    },
    "origin": {
        "$ref": "#/$def/two_positive_numbers",
        "description": "The top left point of the first bubble in this field block",
    },
}

many_field_blocks_description_def = {
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
            "properties": _common_field_block_properties,
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
                        "required": ["fieldType"],
                    },
                    "then": {
                        "additionalProperties": False,
                        "properties": _traditional_field_block_properties,
                    },
                },
                {
                    "if": {
                        "properties": {"fieldType": {"const": "CUSTOM_BUBBLES"}},
                        "required": [
                            "bubbleValues",
                            "direction",
                            "fieldType",
                        ],
                    },
                    "then": {
                        "additionalProperties": False,
                        "properties": _traditional_field_block_properties,
                    },
                },
                {
                    "if": {
                        "properties": {"fieldType": {"const": "BARCODE"}},
                        # TODO: move barcode specific properties into this if-else
                        "required": [
                            "scanArea",
                            "fieldType",
                            "fieldLabel",
                            # TODO: "failIfNotFound"
                            # "emptyValue",
                        ],
                    },
                    "then": {
                        "additionalProperties": False,
                        "properties": {
                            **_common_field_block_properties,
                            "scanArea": _box_area_description,
                            "fieldLabel": {"type": "string"},
                        },
                    },
                },
                # TODO: support for PHOTO_BLOB, OCR custom fields here
            ],
        }
    },
}


TEMPLATE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/Udayraj123/OMRChecker/tree/master/src/schemas/template-schema.json",
    "$def": {
        # The common definitions go here
        **load_common_defs(
            [
                "array_of_strings",
                "field_string_type",
                "positive_integer",
                "positive_number",
                "two_positive_integers",
                "two_positive_numbers",
                "zero_to_one_number",
            ]
        ),
        "marker_area_description": marker_area_description_def,
        "crop_on_four_markers_tuning_options": crop_on_four_markers_tuning_options_def,
        "scan_areas_array": scan_areas_array_def,
        "margins_schema": margins_schema_def,
        "point_selector_patch_area": point_selector_patch_area_def,
        "crop_on_dot_lines_tuning_options": crop_on_dot_lines_tuning_options_def,
        "many_field_blocks_description": many_field_blocks_description_def,
    },
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
            "$ref": "#/$def/two_positive_numbers",
            "description": "The default dimensions for the bubbles in the template overlay: [width, height]",
        },
        "customLabels": {
            "description": "The customLabels contain fields that need to be joined together before generating the results sheet",
            "type": "object",
            "patternProperties": {
                "^.*$": {
                    "type": "array",
                    "items": {
                        "$ref": "#/$def/field_string_type",
                    },
                }
            },
        },
        "emptyValue": {
            "description": "The value to be used in case of empty bubble detected at global level.",
            "type": "string",
        },
        "outputColumns": {
            "description": "The ordered list of columns to be contained in the output csv(default order: alphabetical)",
            "type": "array",
            "items": {
                "$ref": "#/$def/field_string_type",
            },
        },
        "templateDimensions": {
            "$ref": "#/$def/two_positive_numbers",
            "description": "The dimensions(width, height) to which the page will be resized to before applying template",
        },
        "processingImageShape": {
            "$ref": "#/$def/two_positive_numbers",
            "description": "Shape of the processing image after all the pre-processors are applied: [height, width]",
        },
        "outputImageShape": {
            "$ref": "#/$def/two_positive_numbers",
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
                                "$ref": "#/$def/two_positive_numbers",
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
                                            "$ref": "#/$def/two_positive_numbers",
                                            "description": "The size of the morph kernel used for smudging the page",
                                        },
                                        # TODO: support for maxPointsPerEdge
                                        "maxPointsPerEdge": {
                                            "$ref": "#/$def/two_positive_numbers",
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
                                            "$ref": "#/$def/two_positive_numbers",
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
                                            "$ref": "#/$def/zero_to_one_number",
                                            "description": "The value for gamma parameter",
                                        },
                                        "high": {
                                            "$ref": "#/$def/zero_to_one_number",
                                            "description": "The value for high parameter",
                                        },
                                        "low": {
                                            "$ref": "#/$def/zero_to_one_number",
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
                                                **crop_on_dot_lines_tuning_options_def[
                                                    "properties"
                                                ],
                                            },
                                        },
                                        "scanAreas": {
                                            "$ref": "#/$def/scan_areas_array"
                                        },
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
                                        "scanAreas": {
                                            "$ref": "#/$def/scan_areas_array"
                                        },
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
                                                    **_custom_marker_options,
                                                    **{
                                                        area_template: {
                                                            "$ref": "#/$def/marker_area_description"
                                                        }
                                                        for area_template in MARKER_AREA_TYPES_IN_ORDER
                                                    },
                                                    "tuningOptions": {
                                                        "$ref": "#/$def/crop_on_four_markers_tuning_options"
                                                    },
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
                                                    "tuningOptions": {
                                                        "$ref": "#/$def/crop_on_dot_lines_tuning_options"
                                                    },
                                                    "leftLine": {
                                                        "$ref": "#/$def/point_selector_patch_area"
                                                    },
                                                    "topRightDot": {
                                                        "$ref": "#/$def/point_selector_patch_area"
                                                    },
                                                    "bottomRightDot": {
                                                        "$ref": "#/$def/point_selector_patch_area"
                                                    },
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
                                                    "tuningOptions": {
                                                        "$ref": "#/$def/crop_on_dot_lines_tuning_options"
                                                    },
                                                    "rightLine": {
                                                        "$ref": "#/$def/point_selector_patch_area"
                                                    },
                                                    "topLeftDot": {
                                                        "$ref": "#/$def/point_selector_patch_area"
                                                    },
                                                    "bottomLeftDot": {
                                                        "$ref": "#/$def/point_selector_patch_area"
                                                    },
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
                                                    "tuningOptions": {
                                                        "$ref": "#/$def/crop_on_dot_lines_tuning_options"
                                                    },
                                                    "leftLine": {
                                                        "$ref": "#/$def/point_selector_patch_area"
                                                    },
                                                    "rightLine": {
                                                        "$ref": "#/$def/point_selector_patch_area"
                                                    },
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
                                                    "tuningOptions": {
                                                        "$ref": "#/$def/crop_on_dot_lines_tuning_options"
                                                    },
                                                    "topRightDot": {
                                                        "$ref": "#/$def/point_selector_patch_area"
                                                    },
                                                    "bottomRightDot": {
                                                        "$ref": "#/$def/point_selector_patch_area"
                                                    },
                                                    "topLeftDot": {
                                                        "$ref": "#/$def/point_selector_patch_area"
                                                    },
                                                    "bottomLeftDot": {
                                                        "$ref": "#/$def/point_selector_patch_area"
                                                    },
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
            "$ref": "#/$def/many_field_blocks_description",
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
                            # Example: "SET_{booklet_No}", "{filename}"
                            "formatString": {
                                "description": "Format string composed of the response variables to apply the regex on e.g. '{roll}-{barcode}'",
                                "type": "string",
                            },
                            # Examples:
                            # Direct string: "SET_A", "B"
                            # Match last four characters: ".*-SET1"
                            # For multi-page: "*_1.(jpg|png)"
                            "matchRegex": {
                                "description": "Mapping to use on the composed field string",
                                "type": "string",
                                "format": "regex",
                            },
                        },
                    },
                    "fieldBlocks": {
                        "$ref": "#/$def/many_field_blocks_description",
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
