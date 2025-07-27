```js
TEMPLATE_SCHEMA = {
    "$def": {
        "marker_zone_description": {
            "description": "The detailed description of a patch zone for a custom marker",
            "required": [
                "origin",
                "margins"
            ],
            "properties": {
                "origin": {
                    "$ref": "#/$def/two_positive_integers"
                },
                "dimensions": {
                    "$ref": "#/$def/two_positive_integers"
                },
                "margins": {
                    "$ref": "#/$def/margins_schema"
                },
                "selector": {
                    "enum": [
                        "SELECT_TOP_LEFT",
                        "SELECT_TOP_RIGHT",
                        "SELECT_BOTTOM_RIGHT",
                        "SELECT_BOTTOM_LEFT",
                        "SELECT_CENTER",
                        "LINE_INNER_EDGE",
                        "LINE_OUTER_EDGE"
                    ]
                },
                "customOptions": {
                    "description": "Custom options for the scannerType TEMPLATE_MATCH",
                    "properties": {
                        "markerDimensions": {
                            "$ref": "#/$def/two_positive_numbers",
                            "description": "The dimensions of the omr marker"
                        },
                        "referenceImage": {
                            "description": "The relative path to reference image of the omr marker",
                        }
                    }
                }
            }
        },
        "crop_on_four_markers_tuning_options": {
            "description": "Custom tuning options for the CropOnCustomMarkers pre-processor",
            "properties": {
                "apply_erode_subtract": {
                    "description": "A boolean to enable erosion for (sometimes) better marker detection",
                    "type": "boolean"
                },
                "marker_rescale_range": {
                    "$ref": "#/$def/two_positive_numbers",
                    "description": "The range of rescaling in percentage"
                },
                "marker_rescale_steps": {
                    "$ref": "#/$def/positive_integer",
                    "description": "The number of rescaling steps"
                },
                "min_matching_threshold": {
                    "description": "The threshold for template matching",
                    "type": "number"
                },
                "warpMethod": {
                    "enum": [
                        "PERSPECTIVE_TRANSFORM",
                        "HOMOGRAPHY",
                        "REMAP_GRIDDATA",
                        "DOC_REFINE",
                        "WARP_AFFINE"
                    ]
                },
                "warpMethodFlag": {
                    "enum": [
                        "INTER_LINEAR",
                        "INTER_CUBIC",
                        "INTER_NEAREST"
                    ]
                }
            }
        },
        "scan_zones_array": {
            "description": "The schema of a scanZone",
            "items": {
                "required": [
                    "zonePreset"
                ],
                "properties": {
                    "zonePreset": {
                        "description": "The ready-made template to use to prefill description of a scanZone",
                        "enum": [
                            "CUSTOM",
                            "topLeftDot",
                            "topRightDot",
                            "bottomRightDot",
                            "bottomLeftDot",
                            "topLeftMarker",
                            "topRightMarker",
                            "bottomRightMarker",
                            "bottomLeftMarker",
                            "topLine",
                            "leftLine",
                            "bottomLine",
                            "rightLine"
                        ]
                    },
                    "zoneDescription": {
                        "description": "The detailed description of a scanZone's coordinates and purpose",
                        "required": [
                            "origin",
                            "dimensions",
                            "margins"
                        ],
                        "properties": {
                            "origin": {
                                "$ref": "#/$def/two_positive_integers"
                            },
                            "dimensions": {
                                "$ref": "#/$def/two_positive_integers"
                            },
                            "margins": {
                                "$ref": "#/$def/margins_schema"
                            },
                            "selector": {
                                "enum": [
                                    "SELECT_TOP_LEFT",
                                    "SELECT_TOP_RIGHT",
                                    "SELECT_BOTTOM_RIGHT",
                                    "SELECT_BOTTOM_LEFT",
                                    "SELECT_CENTER",
                                    "LINE_INNER_EDGE",
                                    "LINE_OUTER_EDGE"
                                ]
                            },
                            "label": {
                                "description": "The label to use for the scanZone",
                            },
                            "selectorMargins": {
                                "$ref": "#/$def/margins_schema",
                                "description": "The margins around the scanZone's box at provided origin"
                            },
                            "scannerType": {
                                "description": "The scanner type to use in the given scan zone",
                                "enum": [
                                    "PATCH_DOT",
                                    "PATCH_LINE",
                                    "TEMPLATE_MATCH"
                                ]
                            },
                            "maxPoints": {
                                "$ref": "#/$def/positive_integer",
                                "description": "The maximum points to pick from the given scanZone"
                            }
                        }
                    },
                    "customOptions": {
                        "description": "Custom options based on the scannerType",
                        "type": "object"
                    }
                }
            }
        },
        "margins_schema": {
            "description": "The margins to use around a box",
            "required": [
                "top",
                "right",
                "bottom",
                "left"
            ],
            "properties": {
                "top": {
                    "$ref": "#/$def/positive_integer"
                },
                "right": {
                    "$ref": "#/$def/positive_integer"
                },
                "bottom": {
                    "$ref": "#/$def/positive_integer"
                },
                "left": {
                    "$ref": "#/$def/positive_integer"
                }
            }
        },
        "point_selector_patch_zone": {
            "description": "The detailed description of a patch zone with an optional point selector",
            "required": [
                "origin",
                "dimensions",
                "margins"
            ],
            "properties": {
                "origin": {
                    "$ref": "#/$def/two_positive_integers"
                },
                "dimensions": {
                    "$ref": "#/$def/two_positive_integers"
                },
                "margins": {
                    "$ref": "#/$def/margins_schema"
                },
                "selector": {
                    "enum": [
                        "SELECT_TOP_LEFT",
                        "SELECT_TOP_RIGHT",
                        "SELECT_BOTTOM_RIGHT",
                        "SELECT_BOTTOM_LEFT",
                        "SELECT_CENTER",
                        "LINE_INNER_EDGE",
                        "LINE_OUTER_EDGE"
                    ]
                }
            }
        },
        "crop_on_dot_lines_tuning_options": {
            "description": "Custom tuning options for the CropOnDotLines pre-processor",
            "properties": {
                "dotThreshold": {
                    "$ref": "#/$def/positive_number",
                    "description": "The threshold to apply for clearing out the noise near a dot after smudging"
                },
                "dotKernel": {
                    "$ref": "#/$def/two_positive_integers",
                    "description": "The size of the morph kernel to use for smudging each dot"
                },
                "lineThreshold": {
                    "$ref": "#/$def/positive_number",
                    "description": "The threshold to apply for clearing out the noise near a line after smudging"
                },
                "dotBlurKernel": {
                    "$ref": "#/$def/two_odd_integers",
                    "description": "The size of the kernel to use for blurring in each dot's scanZone"
                },
                "lineBlurKernel": {
                    "$ref": "#/$def/two_odd_integers",
                    "description": "The size of the kernel to use for blurring in each line's scanZone"
                },
                "lineKernel": {
                    "$ref": "#/$def/two_positive_integers",
                    "description": "The size of the morph kernel to use for smudging each line"
                },
                "warpMethod": {
                    "enum": [
                        "PERSPECTIVE_TRANSFORM",
                        "HOMOGRAPHY",
                        "REMAP_GRIDDATA",
                        "DOC_REFINE",
                        "WARP_AFFINE"
                    ]
                },
                "warpMethodFlag": {
                    "enum": [
                        "INTER_LINEAR",
                        "INTER_CUBIC",
                        "INTER_NEAREST"
                    ]
                }
            }
        },
        "many_field_blocks_description": {
            "description": "Each fieldBlock denotes a small group of adjacent fields",
            "patternProperties": {
                "^.*$": {
                    "description": "The key is a unique name for the field block",
                    "required": [
                        "fieldDetectionType"
                    ],
                    "properties": {
                        "emptyValue": {
                            "description": "The custom empty bubble value to use in this field block",
                        },
                        "bubbleFieldType": {
                            "description": "The bubble field type to use from a list of ready-made types as well as the custom type",
                        },
                        "fieldDetectionType": {
                            "description": "The detection type to use for reading the fields in the block",
                            "enum": [
                                "BUBBLES_THRESHOLD",
                                "OCR",
                                "BARCODE_QR"
                            ]
                        },
                        "alignment": {
                            "properties": {
                                "margins": {
                                    "$ref": "#/$def/margins_schema"
                                },
                                "maxDisplacement": {
                                    "$ref": "#/$def/positive_integer"
                                },
                                "maxMatchCount": {
                                    "$ref": "#/$def/positive_integer"
                                }
                            }
                        },
                        "fieldLabels": {
                            "description": "The ordered array of labels to use for given fields in this field block",
                            "items": {
                                "$ref": "#/$def/field_string_type"
                            }
                        },
                        "labelsGap": {
                            "$ref": "#/$def/positive_number",
                            "description": "The gap between two labels(top-left to top-left) in the current field block"
                        },
                        "origin": {
                            "$ref": "#/$def/two_positive_numbers",
                            "description": "The top left point of the first bubble in this field block"
                        }
                    },
                    "allOf": [
                        {
                            "if": {
                                "properties": {
                                    "fieldDetectionType": {
                                        "const": "BUBBLES_THRESHOLD"
                                    }
                                }
                            },
                            "then": {
                                "required": [
                                    "origin",
                                    "bubbleFieldType",
                                    "bubblesGap",
                                    "labelsGap",
                                    "fieldLabels"
                                ],
                                "properties": {
                                    "emptyValue": {
                                        "description": "The custom empty bubble value to use in this field block",
                                    },
                                    "bubbleFieldType": {
                                        "oneOf": [
                                            {
                                                "enum": [
                                                    "QTYPE_INT",
                                                    "QTYPE_INT_FROM_1",
                                                    "QTYPE_MCQ4",
                                                    "QTYPE_MCQ5"
                                                ]
                                            },
                                            {
                                                "pattern": "^CUSTOM_.*$"
                                            }
                                        ]
                                    },
                                    "fieldDetectionType": {
                                        "description": "The detection type to use for reading the fields in the block",
                                        "enum": [
                                            "BUBBLES_THRESHOLD",
                                            "OCR",
                                            "BARCODE_QR"
                                        ]
                                    },
                                    "alignment": {
                                        "properties": {
                                            "margins": {
                                                "$ref": "#/$def/margins_schema"
                                            },
                                            "maxDisplacement": {
                                                "$ref": "#/$def/positive_integer"
                                            },
                                            "maxMatchCount": {
                                                "$ref": "#/$def/positive_integer"
                                            }
                                        }
                                    },
                                    "fieldLabels": {
                                        "description": "The ordered array of labels to use for given fields in this field block",
                                        "items": {
                                            "$ref": "#/$def/field_string_type"
                                        }
                                    },
                                    "labelsGap": {
                                        "$ref": "#/$def/positive_number",
                                        "description": "The gap between two labels(top-left to top-left) in the current field block"
                                    },
                                    "origin": {
                                        "$ref": "#/$def/two_positive_numbers",
                                        "description": "The top left point of the first bubble in this field block"
                                    },
                                    "bubbleValues": {
                                        "$ref": "#/$def/array_of_strings",
                                        "description": "The ordered array of values to use for given bubbles per field in this field block"
                                    },
                                    "direction": {
                                        "description": "The direction of expanding the bubbles layout in this field block",
                                        "enum": [
                                            "horizontal",
                                            "vertical"
                                        ]
                                    },
                                    "bubbleDimensions": {
                                        "$ref": "#/$def/two_positive_numbers",
                                        "description": "The custom dimensions for the bubbles in the current field block: [width, height]"
                                    },
                                    "bubblesGap": {
                                        "$ref": "#/$def/positive_number",
                                        "description": "The gap between two bubbles(top-left to top-left) in the current field block"
                                    }
                                }
                            }
                        },
                        {
                            "if": {
                                "properties": {
                                    "fieldDetectionType": {
                                        "const": "OCR"
                                    }
                                },
                                "required": [
                                    "origin",
                                    "scanZone",
                                    "fieldLabels"
                                ]
                            },
                            "then": {
                                "properties": {
                                    "emptyValue": {
                                        "description": "The custom empty bubble value to use in this field block",
                                    },
                                    "bubbleFieldType": {
                                        "description": "The bubble field type to use from a list of ready-made types as well as the custom type",
                                    },
                                    "fieldDetectionType": {
                                        "description": "The detection type to use for reading the fields in the block",
                                        "enum": [
                                            "BUBBLES_THRESHOLD",
                                            "OCR",
                                            "BARCODE_QR"
                                        ]
                                    },
                                    "alignment": {
                                        "properties": {
                                            "margins": {
                                                "$ref": "#/$def/margins_schema"
                                            },
                                            "maxDisplacement": {
                                                "$ref": "#/$def/positive_integer"
                                            },
                                            "maxMatchCount": {
                                                "$ref": "#/$def/positive_integer"
                                            }
                                        }
                                    },
                                    "fieldLabels": {
                                        "description": "The ordered array of labels to use for given fields in this field block",
                                        "items": {
                                            "$ref": "#/$def/field_string_type"
                                        }
                                    },
                                    "labelsGap": {
                                        "$ref": "#/$def/positive_number",
                                        "description": "The gap between two labels(top-left to top-left) in the current field block"
                                    },
                                    "origin": {
                                        "$ref": "#/$def/two_positive_numbers",
                                        "description": "The top left point of the first bubble in this field block"
                                    },
                                    "scanZone": {
                                        "description": "The description of a box zone on the image",
                                        "required": [
                                            "dimensions",
                                            "margins"
                                        ],
                                        "properties": {
                                            "origin": {
                                                "$ref": "#/$def/two_positive_integers"
                                            },
                                            "dimensions": {
                                                "$ref": "#/$def/two_positive_integers"
                                            },
                                            "margins": {
                                                "$ref": "#/$def/margins_schema"
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        {
                            "if": {
                                "properties": {
                                    "fieldDetectionType": {
                                        "const": "BARCODE_QR"
                                    }
                                },
                                "required": [
                                    "origin",
                                    "scanZone",
                                    "fieldLabels"
                                ]
                            },
                            "then": {
                                "properties": {
                                    "emptyValue": {
                                        "description": "The custom empty bubble value to use in this field block",
                                    },
                                    "bubbleFieldType": {
                                        "description": "The bubble field type to use from a list of ready-made types as well as the custom type",
                                    },
                                    "fieldDetectionType": {
                                        "description": "The detection type to use for reading the fields in the block",
                                        "enum": [
                                            "BUBBLES_THRESHOLD",
                                            "OCR",
                                            "BARCODE_QR"
                                        ]
                                    },
                                    "alignment": {
                                        "properties": {
                                            "margins": {
                                                "$ref": "#/$def/margins_schema"
                                            },
                                            "maxDisplacement": {
                                                "$ref": "#/$def/positive_integer"
                                            },
                                            "maxMatchCount": {
                                                "$ref": "#/$def/positive_integer"
                                            }
                                        }
                                    },
                                    "fieldLabels": {
                                        "description": "The ordered array of labels to use for given fields in this field block",
                                        "items": {
                                            "$ref": "#/$def/field_string_type"
                                        }
                                    },
                                    "labelsGap": {
                                        "$ref": "#/$def/positive_number",
                                        "description": "The gap between two labels(top-left to top-left) in the current field block"
                                    },
                                    "origin": {
                                        "$ref": "#/$def/two_positive_numbers",
                                        "description": "The top left point of the first bubble in this field block"
                                    },
                                    "scanZone": {
                                        "description": "The description of a box zone on the image",
                                        "required": [
                                            "dimensions",
                                            "margins"
                                        ],
                                        "properties": {
                                            "origin": {
                                                "$ref": "#/$def/two_positive_integers"
                                            },
                                            "dimensions": {
                                                "$ref": "#/$def/two_positive_integers"
                                            },
                                            "margins": {
                                                "$ref": "#/$def/margins_schema"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        }
    },
    "title": "Template Validation Schema",
    "description": "OMRChecker input template schema",
    "required": [
        "bubbleDimensions",
        "templateDimensions",
        "processingImageShape",
        "preProcessors",
        "fieldBlocks"
    ],
    "properties": {
        "output": {
            "type": "boolean"
        },
        "alignment": {
            "properties": {
                "referenceImage": {
                    "description": "The relative path to reference image",
                },
                "margins": {
                    "$ref": "#/$def/margins_schema"
                },
                "maxDisplacement": {
                    "$ref": "#/$def/positive_integer"
                },
                "maxMatchCount": {
                    "$ref": "#/$def/positive_integer"
                },
                "anchorWindowSize": {
                    "$ref": "#/$def/two_positive_numbers",
                    "description": "The size of the anchor window for picking best local anchor points"
                }
            }
        },
        "bubbleDimensions": {
            "$ref": "#/$def/two_positive_numbers",
            "description": "The default dimensions for the bubbles in the template overlay: [width, height]"
        },
        "customLabels": {
            "description": "The customLabels contain fields that need to be joined together before generating the results sheet",
            "patternProperties": {
                "^.*$": {
                    "items": {
                        "$ref": "#/$def/field_string_type"
                    }
                }
            }
        },
        "emptyValue": {
            "description": "The value to be used in case of empty bubble detected at global level.",
        },
        "outputColumns": {
            "description": "The ordered list of columns to be contained in the output csv(default order: alphabetical)",
            "items": {
                "$ref": "#/$def/field_string_type"
            }
        },
        "templateDimensions": {
            "$ref": "#/$def/two_positive_numbers",
            "description": "The dimensions(width, height) to which the page will be resized to before applying template"
        },
        "processingImageShape": {
            "$ref": "#/$def/two_positive_numbers",
            "description": "Shape of the processing image after all the pre-processors are applied: [height, width]"
        },
        "outputImageShape": {
            "$ref": "#/$def/two_positive_numbers",
            "description": "Shape of the final output image: [height, width]"
        },
        "preProcessors": {
            "description": "Custom configuration values to use in the template's directory",
            "items": {
                "required": [
                    "name",
                    "options"
                ],
                "properties": {
                    "name": {
                        "description": "The name of the pre-processor to use",
                        "enum": [
                            "AutoRotate",
                            "Contrast",
                            "CropOnMarkers",
                            "CropPage",
                            "FeatureBasedAlignment",
                            "GaussianBlur",
                            "Levels",
                            "MedianBlur"
                        ]
                    },
                    "options": {
                        "description": "The options to pass to the pre-processor",
                        "properties": {
                            "processingImageShape": {
                                "$ref": "#/$def/two_positive_numbers",
                                "description": "Shape of the processing image for the current pre-processors: [height, width]"
                            }
                        }
                    }
                },
                "allOf": [
                    {
                        "if": {
                            "properties": {
                                "name": {
                                    "const": "CropPage"
                                }
                            },
                            "required": [
                                "name",
                                "options"
                            ]
                        },
                        "then": {
                            "properties": {
                                "options": {
                                    "description": "Options for the CropPage pre-processor",
                                    "properties": {
                                        "morphKernel": {
                                            "$ref": "#/$def/two_positive_numbers",
                                            "description": "The size of the morph kernel used for smudging the page"
                                        },
                                        "useColoredCanny": {
                                            "description": "Whether to separate 'white' from other colors during page detection. Requires config.colored_outputs_enabled == True",
                                            "type": "boolean"
                                        },
                                        "maxPointsPerEdge": {
                                            "$ref": "#/$def/two_positive_numbers",
                                            "description": "Max number of control points to use in one edge"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    {
                        "if": {
                            "properties": {
                                "name": {
                                    "const": "FeatureBasedAlignment"
                                }
                            },
                            "required": [
                                "name",
                                "options"
                            ]
                        },
                        "then": {
                            "properties": {
                                "options": {
                                    "description": "Options for the FeatureBasedAlignment pre-processor",
                                    "properties": {
                                        "2d": {
                                            "description": "Uses warpAffine if True, otherwise uses warpPerspective",
                                            "type": "boolean"
                                        },
                                        "goodMatchPercent": {
                                            "description": "Threshold for the match percentage",
                                            "type": "number"
                                        },
                                        "maxFeatures": {
                                            "description": "Maximum number of matched features to consider",
                                            "type": "integer"
                                        },
                                        "reference": {
                                            "description": "Relative path to the reference image",
                                        },
                                        "matcherType": {
                                            "description": "Type of the matcher to use",
                                            "enum": [
                                                "BRUTEFORCE_HAMMING",
                                                "NORM_HAMMING"
                                            ]
                                        }
                                    },
                                    "required": [
                                        "reference"
                                    ]
                                }
                            }
                        }
                    },
                    {
                        "if": {
                            "properties": {
                                "name": {
                                    "const": "GaussianBlur"
                                }
                            },
                            "required": [
                                "name",
                                "options"
                            ]
                        },
                        "then": {
                            "properties": {
                                "options": {
                                    "description": "Options for the GaussianBlur pre-processor",
                                    "properties": {
                                        "kSize": {
                                            "$ref": "#/$def/two_positive_numbers",
                                            "description": "Size of the kernel"
                                        },
                                        "sigmaX": {
                                            "description": "Value of sigmaX in fraction",
                                            "type": "number"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    {
                        "if": {
                            "properties": {
                                "name": {
                                    "const": "Levels"
                                }
                            },
                            "required": [
                                "name",
                                "options"
                            ]
                        },
                        "then": {
                            "properties": {
                                "options": {
                                    "properties": {
                                        "gamma": {
                                            "$ref": "#/$def/zero_to_one_number",
                                            "description": "The value for gamma parameter"
                                        },
                                        "high": {
                                            "$ref": "#/$def/zero_to_one_number",
                                            "description": "The value for high parameter"
                                        },
                                        "low": {
                                            "$ref": "#/$def/zero_to_one_number",
                                            "description": "The value for low parameter"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    {
                        "if": {
                            "properties": {
                                "name": {
                                    "const": "MedianBlur"
                                }
                            },
                            "required": [
                                "name",
                                "options"
                            ]
                        },
                        "then": {
                            "properties": {
                                "options": {
                                    "description": "Options for the MedianBlur pre-processor",
                                    "properties": {
                                        "kSize": {
                                            "type": "integer"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    {
                        "if": {
                            "properties": {
                                "name": {
                                    "const": "AutoRotate"
                                }
                            },
                            "required": [
                                "name",
                                "options"
                            ]
                        },
                        "then": {
                            "properties": {
                                "options": {
                                    "description": "Options for the AutoRotate pre-processor",
                                    "required": [
                                        "referenceImage"
                                    ],
                                    "properties": {
                                        "referenceImage": {
                                            "description": "The relative path to reference image",
                                        },
                                        "markerDimensions": {
                                            "description": "Dimensions of the reference image",
                                            "$ref": "#/$def/two_positive_numbers"
                                        },
                                        "threshold": {
                                            "description": "Threshold for the match score below it will throw error/warning",
                                            "required": [
                                                "value",
                                                "passthrough"
                                            ],
                                            "properties": {
                                                "value": {
                                                    "type": "number"
                                                },
                                                "passthrough": {
                                                    "type": "boolean"
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    {
                        "if": {
                            "properties": {
                                "name": {
                                    "const": "Contrast"
                                }
                            },
                            "required": [
                                "name",
                                "options"
                            ]
                        },
                        "then": {
                            "properties": {
                                "options": {
                                    "description": "Options for the Contrast pre-processor",
                                    "properties": {
                                        "mode": {
                                            "enum": [
                                                "manual",
                                                "auto"
                                            ]
                                        },
                                        "alpha": {
                                            "type": "number"
                                        },
                                        "beta": {
                                            "type": "number"
                                        },
                                        "clipPercentage": {
                                            "type": "number"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    {
                        "if": {
                            "properties": {
                                "name": {
                                    "const": "WarpOnPoints"
                                }
                            },
                            "required": [
                                "name",
                                "options"
                            ]
                        },
                        "then": {
                            "properties": {
                                "options": {
                                    "description": "Options for the WarpOnPoints pre-processor",
                                    "required": [],
                                    "properties": {
                                        "enableCropping": {
                                            "description": "Whether to crop the image to a bounding box of the given anchor points",
                                            "type": "boolean"
                                        },
                                        "defaultSelector": {
                                            "description": "The default points selector for the given scanZones",
                                            "enum": [
                                                "CENTERS",
                                                "INNER_WIDTHS",
                                                "INNER_HEIGHTS",
                                                "INNER_CORNERS",
                                                "OUTER_CORNERS"
                                            ]
                                        },
                                        "scanZones": {
                                            "$ref": "#/$def/scan_zones_array"
                                        },
                                        "tuningOptions": {
                                            "description": "Custom tuning options for the WarpOnPoints pre-processor",
                                            "required": [],
                                            "properties": {
                                                "dotThreshold": {
                                                    "$ref": "#/$def/positive_number",
                                                    "description": "The threshold to apply for clearing out the noise near a dot after smudging"
                                                },
                                                "dotKernel": {
                                                    "$ref": "#/$def/two_positive_integers",
                                                    "description": "The size of the morph kernel to use for smudging each dot"
                                                },
                                                "lineThreshold": {
                                                    "$ref": "#/$def/positive_number",
                                                    "description": "The threshold to apply for clearing out the noise near a line after smudging"
                                                },
                                                "dotBlurKernel": {
                                                    "$ref": "#/$def/two_odd_integers",
                                                    "description": "The size of the kernel to use for blurring in each dot's scanZone"
                                                },
                                                "lineBlurKernel": {
                                                    "$ref": "#/$def/two_odd_integers",
                                                    "description": "The size of the kernel to use for blurring in each line's scanZone"
                                                },
                                                "lineKernel": {
                                                    "$ref": "#/$def/two_positive_integers",
                                                    "description": "The size of the morph kernel to use for smudging each line"
                                                },
                                                "warpMethod": {
                                                    "enum": [
                                                        "PERSPECTIVE_TRANSFORM",
                                                        "HOMOGRAPHY",
                                                        "REMAP_GRIDDATA",
                                                        "DOC_REFINE",
                                                        "WARP_AFFINE"
                                                    ]
                                                },
                                                "warpMethodFlag": {
                                                    "enum": [
                                                        "INTER_LINEAR",
                                                        "INTER_CUBIC",
                                                        "INTER_NEAREST"
                                                    ]
                                                }
                                            }
                                        },
                                        "pointsLayout": {
                                            "description": "The type of layout of the scanZones for finding anchor points",
                                            "enum": [
                                                "FOUR_MARKERS",
                                                "ONE_LINE_TWO_DOTS",
                                                "TWO_DOTS_ONE_LINE",
                                                "TWO_LINES",
                                                "FOUR_DOTS",
                                                "CUSTOM"
                                            ]
                                        }
                                    }
                                }
                            }
                        }
                    },
                    {
                        "if": {
                            "properties": {
                                "name": {
                                    "const": "CropOnMarkers"
                                }
                            },
                            "required": [
                                "name",
                                "options"
                            ]
                        },
                        "then": {
                            "properties": {
                                "options": {
                                    "description": "Options for the CropOnMarkers pre-processor",
                                    "required": [
                                        "type",
                                        "processingImageShape"
                                    ],
                                    "properties": {
                                        "enableCropping": {
                                            "description": "Whether to crop the image to a bounding box of the given anchor points",
                                            "type": "boolean"
                                        },
                                        "scanZones": {
                                            "$ref": "#/$def/scan_zones_array"
                                        },
                                        "defaultSelector": {
                                            "description": "The default points selector for the given scanZones",
                                            "enum": [
                                                "CENTERS",
                                                "INNER_WIDTHS",
                                                "INNER_HEIGHTS",
                                                "INNER_CORNERS",
                                                "OUTER_CORNERS"
                                            ]
                                        },
                                        "type": {
                                            "description": "The type of the Cropping instance to use",
                                            "enum": [
                                                "FOUR_MARKERS",
                                                "ONE_LINE_TWO_DOTS",
                                                "TWO_DOTS_ONE_LINE",
                                                "TWO_LINES",
                                                "FOUR_DOTS"
                                            ]
                                        }
                                    },
                                    "allOf": [
                                        {
                                            "if": {
                                                "required": [
                                                    "type",
                                                    "processingImageShape"
                                                ],
                                                "properties": {
                                                    "type": {
                                                        "const": "FOUR_MARKERS"
                                                    }
                                                }
                                            },
                                            "then": {
                                                "required": [
                                                    "referenceImage",
                                                    "markerDimensions"
                                                ],
                                                "properties": {
                                                    "tuningOptions": {
                                                        "$ref": "#/$def/crop_on_four_markers_tuning_options"
                                                    },
                                                    "markerDimensions": {
                                                        "$ref": "#/$def/two_positive_numbers",
                                                        "description": "The dimensions of the omr marker"
                                                    },
                                                    "referenceImage": {
                                                        "description": "The relative path to reference image of the omr marker",
                                                    },
                                                    "topLeftMarker": {
                                                        "$ref": "#/$def/marker_zone_description"
                                                    },
                                                    "topRightMarker": {
                                                        "$ref": "#/$def/marker_zone_description"
                                                    },
                                                    "bottomRightMarker": {
                                                        "$ref": "#/$def/marker_zone_description"
                                                    },
                                                    "bottomLeftMarker": {
                                                        "$ref": "#/$def/marker_zone_description"
                                                    }
                                                }
                                            }
                                        },
                                        {
                                            "if": {
                                                "required": [
                                                    "type",
                                                    "processingImageShape"
                                                ],
                                                "properties": {
                                                    "type": {
                                                        "const": "ONE_LINE_TWO_DOTS"
                                                    }
                                                }
                                            },
                                            "then": {
                                                "oneOf": [
                                                    {
                                                        "required": [
                                                            "scanZones"
                                                        ]
                                                    },
                                                    {
                                                        "required": [
                                                            "leftLine",
                                                            "topRightDot",
                                                            "bottomRightDot"
                                                        ]
                                                    }
                                                ],
                                                "properties": {
                                                    "tuningOptions": {
                                                        "$ref": "#/$def/crop_on_dot_lines_tuning_options"
                                                    },
                                                    "leftLine": {
                                                        "$ref": "#/$def/point_selector_patch_zone"
                                                    },
                                                    "topRightDot": {
                                                        "$ref": "#/$def/point_selector_patch_zone"
                                                    },
                                                    "bottomRightDot": {
                                                        "$ref": "#/$def/point_selector_patch_zone"
                                                    }
                                                }
                                            }
                                        },
                                        {
                                            "if": {
                                                "required": [
                                                    "type",
                                                    "processingImageShape"
                                                ],
                                                "properties": {
                                                    "type": {
                                                        "const": "TWO_DOTS_ONE_LINE"
                                                    }
                                                }
                                            },
                                            "then": {
                                                "oneOf": [
                                                    {
                                                        "required": [
                                                            "scanZones"
                                                        ]
                                                    },
                                                    {
                                                        "required": [
                                                            "rightLine",
                                                            "topLeftDot",
                                                            "bottomLeftDot"
                                                        ]
                                                    }
                                                ],
                                                "properties": {
                                                    "tuningOptions": {
                                                        "$ref": "#/$def/crop_on_dot_lines_tuning_options"
                                                    },
                                                    "rightLine": {
                                                        "$ref": "#/$def/point_selector_patch_zone"
                                                    },
                                                    "topLeftDot": {
                                                        "$ref": "#/$def/point_selector_patch_zone"
                                                    },
                                                    "bottomLeftDot": {
                                                        "$ref": "#/$def/point_selector_patch_zone"
                                                    }
                                                }
                                            }
                                        },
                                        {
                                            "if": {
                                                "required": [
                                                    "type",
                                                    "processingImageShape"
                                                ],
                                                "properties": {
                                                    "type": {
                                                        "const": "TWO_LINES"
                                                    }
                                                }
                                            },
                                            "then": {
                                                "oneOf": [
                                                    {
                                                        "required": [
                                                            "scanZones"
                                                        ]
                                                    },
                                                    {
                                                        "required": [
                                                            "leftLine",
                                                            "rightLine"
                                                        ]
                                                    }
                                                ],
                                                "properties": {
                                                    "tuningOptions": {
                                                        "$ref": "#/$def/crop_on_dot_lines_tuning_options"
                                                    },
                                                    "leftLine": {
                                                        "$ref": "#/$def/point_selector_patch_zone"
                                                    },
                                                    "rightLine": {
                                                        "$ref": "#/$def/point_selector_patch_zone"
                                                    }
                                                }
                                            }
                                        },
                                        {
                                            "if": {
                                                "required": [
                                                    "type",
                                                    "processingImageShape"
                                                ],
                                                "properties": {
                                                    "type": {
                                                        "const": "FOUR_DOTS"
                                                    }
                                                }
                                            },
                                            "then": {
                                                "oneOf": [
                                                    {
                                                        "required": [
                                                            "scanZones"
                                                        ]
                                                    },
                                                    {
                                                        "required": [
                                                            "topRightDot",
                                                            "bottomRightDot",
                                                            "topLeftDot",
                                                            "bottomLeftDot"
                                                        ]
                                                    }
                                                ],
                                                "properties": {
                                                    "tuningOptions": {
                                                        "$ref": "#/$def/crop_on_dot_lines_tuning_options"
                                                    },
                                                    "topRightDot": {
                                                        "$ref": "#/$def/point_selector_patch_zone"
                                                    },
                                                    "bottomRightDot": {
                                                        "$ref": "#/$def/point_selector_patch_zone"
                                                    },
                                                    "topLeftDot": {
                                                        "$ref": "#/$def/point_selector_patch_zone"
                                                    },
                                                    "bottomLeftDot": {
                                                        "$ref": "#/$def/point_selector_patch_zone"
                                                    }
                                                }
                                            }
                                        }
                                    ]
                                }
                            }
                        }
                    }
                ]
            }
        },
        "customBubbleFieldTypes": {
            "patternProperties": {
                "^CUSTOM_.*$": {
                    "description": "The key is a unique name for the custom bubble field type. It can override built-in field types as well.",
                    "required": [
                        "bubbleValues",
                        "direction"
                    ],
                    "properties": {
                        "bubbleValues": {
                            "$ref": "#/$def/array_of_strings",
                            "description": "The ordered array of values to use for given bubbles per field in this field block"
                        },
                        "direction": {
                            "description": "The direction of expanding the bubbles layout in this field block",
                            "enum": [
                                "horizontal",
                                "vertical"
                            ]
                        }
                    }
                }
            }
        },
        "fieldBlocksOffset": {
            "$ref": "#/$def/two_integers",
            "description": "The offset to apply for each field block. This makes changing pointsSelectors convenient on a ready template"
        },
        "fieldBlocks": {
            "$ref": "#/$def/many_field_blocks_description",
            "description": "The default field block to apply and read before applying any matcher on the fields response."
        },
        "conditionalSets": {
            "description": "An array of field block sets with their conditions. These will override the default values in case of any conflict",
            "items": {
                "description": "Each item represents a conditional layout of field blocks",
                "required": [
                    "name",
                    "matcher",
                    "fieldBlocks"
                ],
                "properties": {
                    "name": {
                    },
                    "matcher": {
                        "description": "Mapping response fields from default layout to the set name",
                        "required": [
                            "formatString",
                            "matchRegex"
                        ],
                        "properties": {
                            "formatString": {
                                "description": "Format string composed of the response variables to apply the regex on e.g. '{roll}-{barcode}'",
                            },
                            "matchRegex": {
                                "description": "Mapping to use on the composed field string",
                                "format": "regex"
                            }
                        }
                    },
                    "fieldBlocks": {
                        "$ref": "#/$def/many_field_blocks_description",
                        "description": "The custom field blocks layout to apply if given matcher is satisfied"
                    }
                }
            }
        },
        "sortFiles": {
            "description": "Configuration to sort images/files based on field responses, QR codes, barcodes, etc and a regex mapping",
            "allOf": [
                {
                    "required": [
                        "enabled"
                    ]
                },
                {
                    "if": {
                        "properties": {
                            "enabled": {
                                "const": true
                            }
                        }
                    },
                    "then": {
                        "required": [
                            "enabled",
                            "sortMode",
                            "outputDirectory",
                            "fileMappings"
                        ]
                    }
                }
            ],
            "properties": {
                "enabled": {
                    "description": "Whether to enable sorting. Note that file copies/movements are irreversible once enabled",
                    "type": "boolean"
                },
                "sortMode": {
                    "description": "Whether to copy files or move files",
                    "enum": [
                        "COPY",
                        "MOVE"
                    ]
                },
                "outputDirectory": {
                    "description": "Relative path of the directory to use to sort the files",
                },
                "fileMapping": {
                    "description": "A mapping from regex to the relative file path to use",
                    "required": [
                        "formatString"
                    ],
                    "properties": {
                        "formatString": {
                            "description": "Format string composed of the response variables to apply the regex on e.g. '{roll}-{barcode}'",
                        },
                        "extractRegex": {
                            "description": "Mapping to use on the composed field string",
                            "format": "regex"
                        },
                        "capturedString": {
                            "description": "The captured groups string to use for replacement",
                        }
                    }
                }
            }
        }
    }
}
```