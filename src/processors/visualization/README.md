# Workflow Visualization Module

This module provides comprehensive visualization and debugging capabilities for the OMR processing pipeline.

## Quick Start

```bash
# Visualize an OMR workflow
uv run python -m src.utils.visualization_runner \
    --input inputs/sample1/sample1.jpg \
    --template inputs/sample1/template.json \
    --output outputs/visualization
```

## Components

### Core Classes

- **`WorkflowSession`**: Data model storing complete workflow execution data
- **`ProcessorState`**: Represents state of a processor at execution time
- **`WorkflowGraph`**: Graph structure of the workflow
- **`WorkflowTracker`**: Tracks and captures processor states during execution
- **`HTMLExporter`**: Generates interactive HTML visualizations
- **`ImageEncoder`**: Utility for encoding/decoding images to base64

### High-Level Functions

- **`track_workflow()`**: Track a complete workflow execution
- **`export_to_html()`**: Export session to HTML visualization
- **`replay_from_json()`**: Replay visualization from saved JSON

## File Structure

```
src/processors/visualization/
├── __init__.py              # Module exports
├── workflow_session.py      # Data models
├── workflow_tracker.py      # Tracking logic
├── html_exporter.py         # HTML generation
└── templates/
    └── viewer.html          # HTML template with vis.js
```

## Usage Examples

See `/examples/workflow_visualization_examples.py` for comprehensive examples.

### Basic Usage

```python
from src.processors.visualization import track_workflow, export_to_html

# Track a workflow
session = track_workflow(
    file_path="inputs/sample1/sample1.jpg",
    template_path="inputs/sample1/template.json"
)

# Export to HTML
html_path = export_to_html(session, "outputs/visualization")
```

### Capture Specific Processors

```python
session = track_workflow(
    file_path="inputs/sample1/sample1.jpg",
    template_path="inputs/sample1/template.json",
    capture_processors=["AutoRotate", "CropOnMarkers", "ReadOMR"]
)
```

### Custom Image Settings

```python
session = track_workflow(
    file_path="inputs/sample1/sample1.jpg",
    template_path="inputs/sample1/template.json",
    max_image_width=1200,
    image_quality=95,
    include_colored=True
)
```

## Documentation

Full documentation available at: `/docs/workflow-visualization.md`

## Testing

Run tests with:

```bash
pytest src/tests/test_workflow_visualization.py -v
```

## Features

- ✅ Interactive flowchart visualization
- ✅ Step-by-step image viewer
- ✅ Playback controls with adjustable speed
- ✅ Configurable processor capture
- ✅ Export to standalone HTML
- ✅ JSON export for replay
- ✅ Grayscale and colored image support
- ✅ Custom metadata capture
- ✅ Error state tracking

## Dependencies

- **vis-network**: For flowchart rendering (loaded from CDN)
- **OpenCV (cv2)**: Image processing
- **numpy**: Array operations
- Standard library: json, base64, dataclasses

## Performance

- Image capture adds ~50-100ms per processor (negligible)
- Memory usage: ~2-5MB per captured image
- Typical HTML file size: 10-30MB (all images embedded)
- Configurable quality/size tradeoffs available

## License

Same as OMRChecker main project (MIT License)

