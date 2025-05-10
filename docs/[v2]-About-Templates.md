# wip

## V1 to V2 Migration
    - Renamed pageDimensions -> templateDimensions
    - Support for a top level processingImageShape as well as pre-processor level processingImageShape
    - Added conditionalSets in the schema.
    - Support for outputImageShape

<!-- - TODO: add Links to Readmes inside individual samples -->

# Template Documentation (WIP - ai generated)

The template file defines how OMRChecker should process and interpret OMR sheets. It includes configurations for image preprocessing, field layouts, bubble arrangements, and output formatting.

## Core Components

### templateDimensions
The dimensions to which each page will be resized before applying the template.
```json
"templateDimensions": [816, 1056] // [width, height]
```

### processingImageShape
The shape of the processing image when applying pre-processors.
```json
"processingImageShape": [900, 650] // [height, width]
```

### bubbleDimensions
The default dimensions for bubbles in the template overlay.
```json
"bubbleDimensions": [20, 20] // [width, height]
```

### fieldBlocks
Groups of adjacent fields that represent questions/answers.
```json
"fieldBlocks": {
  "roll_number": {
    "fieldLabels": ["roll_1", "roll_2"],
    "bubbleValues": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "direction": "vertical",
    "origin": [50, 100],
    "fieldDetectionType": "BUBBLES_THRESHOLD"
  }
}
```

## Pre-processors

Pre-processors are image processing steps applied in sequence. Common pre-processors include:

### CropPage
Crops the input image to the page boundaries.
```json
{
  "name": "CropPage",
  "options": {
    "processingImageShape": [900, 650]
  }
}
```

### CropOnMarkers
Crops and aligns the image using reference markers.
```json
{
  "name": "CropOnMarkers",
  "options": {
    "type": "FOUR_MARKERS",
    "referenceImage": "path/to/marker.jpg",
    "markerDimensions": [50, 50]
  }
}
```

### FeatureBasedAlignment
Aligns image using feature matching.
```json
{
  "name": "FeatureBasedAlignment",
  "options": {
    "reference": "path/to/reference.jpg",
    "goodMatchPercent": 0.15
  }
}
```

## Field Detection Types

### BUBBLES_THRESHOLD
Standard bubble detection using thresholding.
```json
{
  "fieldDetectionType": "BUBBLES_THRESHOLD",
  "bubbleFieldType": "NUMERIC",
  "bubbleValues": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
  "direction": "vertical"
}
```

### OCR
Text recognition from specified zones.
```json
{
  "fieldDetectionType": "OCR",
  "scanZone": {
    "dimensions": [100, 30],
    "margins": {"top": 0, "right": 0, "bottom": 0, "left": 0}
  }
}
```

### BARCODE_QR
Barcode/QR code scanning from specified zones.
```json
{
  "fieldDetectionType": "BARCODE_QR",
  "scanZone": {
    "dimensions": [200, 200],
    "margins": {"top": 5, "right": 5, "bottom": 5, "left": 5}
  }
}
```

## Conditional Sets

Allows different template configurations based on matching conditions.
```json
{
  "conditionalSets": [{
    "name": "Set A",
    "matcher": {
      "formatString": "{Roll}",
      "matchRegex": ".*-SET1"
    },
    "fieldBlocks": {
      // Custom field blocks for this set
    }
  }]
}
```

## Output Configuration

### sortFiles
Configures how processed files should be organized.
```json
{
  "sortFiles": {
    "enabled": true,
    "sortMode": "COPY",
    "outputDirectory": "sorted_outputs",
    "fileMapping": {
      "formatString": "{roll}-{section}",
      "extractRegex": "(\\w{4}).*"
    }
  }
}
```
