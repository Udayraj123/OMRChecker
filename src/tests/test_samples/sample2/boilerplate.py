TEMPLATE_BOILERPLATE = {
    "templateDimensions": [2550, 3300],
    "bubbleDimensions": [32, 32],
    "processingImageShape": [1640, 1332],
    "preProcessors": [
        {
            "name": "CropOnMarkers",
            "options": {
                "type": "FOUR_MARKERS",
                "relativePath": "omr_marker.jpg",
                "dimensions": [40, 40],
                "tuningOptions": {"marker_rescale_range": [80, 120]},
            },
        }
    ],
    "fieldBlockGroups": {
        "DEFAULT": {
            "MCQBlock1a1": {
                "fieldType": "QTYPE_MCQ4",
                "origin": [197, 300],
                "bubblesGap": 92,
                "labelsGap": 59.6,
                "fieldLabels": ["q1..17"],
            },
            "MCQBlock1a11": {
                "fieldType": "QTYPE_MCQ4",
                "origin": [1770, 1310],
                "bubblesGap": 92,
                "labelsGap": 59.6,
                "fieldLabels": ["q168..184"],
            },
        }
    },
}

CONFIG_BOILERPLATE = {
    "outputs": {
        "show_image_level": 0,
        "filter_out_multimarked_files": True,
        "display_image_dimensions": [960, 1280],
    },
}
