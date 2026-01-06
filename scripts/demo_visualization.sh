#!/bin/bash

# Workflow Visualization Tool - Quick Demo Script
# This script demonstrates the workflow visualization tool with sample data

set -e  # Exit on error

echo "=========================================="
echo "OMR Workflow Visualization Tool - Demo"
echo "=========================================="
echo ""

# Check if sample data exists
if [ ! -d "inputs/sample1" ] && [ ! -d "samples/1-mobile-camera" ]; then
    echo "❌ Error: No sample data found."
    echo "Please ensure you have sample data in 'inputs/' or 'samples/' directory."
    exit 1
fi

# Determine sample directory
SAMPLE_DIR="inputs/sample1"
if [ ! -d "$SAMPLE_DIR" ]; then
    SAMPLE_DIR="samples/1-mobile-camera"
fi

echo "Using sample directory: $SAMPLE_DIR"
echo ""

# Check if there are any image files
IMAGE_FILE=$(find "$SAMPLE_DIR" -type f \( -name "*.jpg" -o -name "*.png" \) | head -n 1)
if [ -z "$IMAGE_FILE" ]; then
    echo "❌ Error: No image files found in $SAMPLE_DIR"
    exit 1
fi

TEMPLATE_FILE="$SAMPLE_DIR/template.json"
CONFIG_FILE="$SAMPLE_DIR/config.json"

if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "❌ Error: Template file not found: $TEMPLATE_FILE"
    exit 1
fi

echo "Configuration:"
echo "  Image: $IMAGE_FILE"
echo "  Template: $TEMPLATE_FILE"
echo "  Config: ${CONFIG_FILE:-Default}"
echo ""

# Create output directory
OUTPUT_DIR="outputs/visualization_demo"
mkdir -p "$OUTPUT_DIR"

# Run visualization
echo "Running workflow visualization..."
echo ""

if [ -f "$CONFIG_FILE" ]; then
    uv run python -m src.utils.visualization_runner \
        --input "$IMAGE_FILE" \
        --template "$TEMPLATE_FILE" \
        --config "$CONFIG_FILE" \
        --output "$OUTPUT_DIR" \
        --title "OMR Workflow Demo" \
        --capture-processors "all"
else
    uv run python -m src.utils.visualization_runner \
        --input "$IMAGE_FILE" \
        --template "$TEMPLATE_FILE" \
        --output "$OUTPUT_DIR" \
        --title "OMR Workflow Demo" \
        --capture-processors "all"
fi

echo ""
echo "=========================================="
echo "✅ Demo completed successfully!"
echo "=========================================="
echo ""
echo "The visualization has been saved to:"
echo "  $OUTPUT_DIR/"
echo ""
echo "Check your browser - the visualization should have opened automatically."
echo "If not, open the HTML file manually from the output directory."
echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Try with specific processors only:"
echo "   uv run python -m src.utils.visualization_runner \\"
echo "       --input \"$IMAGE_FILE\" \\"
echo "       --template \"$TEMPLATE_FILE\" \\"
echo "       --capture-processors \"AutoRotate,CropOnMarkers,ReadOMR\" \\"
echo "       --output outputs/viz_custom"
echo ""
echo "2. Adjust image quality:"
echo "   uv run python -m src.utils.visualization_runner \\"
echo "       --input \"$IMAGE_FILE\" \\"
echo "       --template \"$TEMPLATE_FILE\" \\"
echo "       --max-width 1200 \\"
echo "       --quality 95 \\"
echo "       --output outputs/viz_hq"
echo ""
echo "3. Run the example scripts:"
echo "   uv run python examples/workflow_visualization_examples.py"
echo ""
echo "For more information, see:"
echo "  - docs/workflow-visualization.md"
echo "  - src/processors/visualization/README.md"
echo ""

