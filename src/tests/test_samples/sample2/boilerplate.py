TEMPLATE_BOILERPLATE = {
    "pageDimensions": [2550, 3300],
    "bubbleDimensions": [32, 32],
    "preProcessors": [
        {
            "name": "CropOnMarkers",
            "options": {
                "relativePath": "omr_marker.jpg",
                "sheetToMarkerWidthRatio": 17,
            },
        }
    ],
    "fieldBlocks": {
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
    },
}

CONFIG_BOILERPLATE = {
    "dimensions": {
        "display_height": 960,
        "display_width": 1280,
        "processing_height": 1640,
        "processing_width": 1332,
    },
    "outputs": {"show_image_level": 0, "filter_out_multimarked_files": True},
}
