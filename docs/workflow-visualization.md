# Workflow Visualization Guide

## Overview

The OMR Workflow Visualization tool allows you to visualize and debug the image processing pipeline by capturing intermediate images and generating interactive HTML reports. This is useful for:

- **Debugging**: See exactly how each processor transforms the image
- **Documentation**: Create shareable visualizations of your pipeline
- **Training**: Help users understand the processing flow
- **Optimization**: Identify bottlenecks and unnecessary steps

## Features

- 📊 **Interactive Flowchart**: Visual graph showing processor flow
- 🖼️ **Image Viewer**: View output from each processor
- ▶️ **Playback Controls**: Step through or animate the workflow
- 💾 **Export**: Save as standalone HTML or JSON for replay
- ⚙️ **Configurable**: Capture all or specific processors
- 🎨 **Dual Mode**: View grayscale or colored images

## Quick Start

### Using the CLI Tool

```bash
# Visualize a single OMR file
uv run python -m src.utils.visualization_runner \
    --input inputs/sample1/sample1.jpg \
    --template inputs/sample1/template.json \
    --output outputs/visualization
```

This will:
1. Process the OMR file through the pipeline
2. Capture images from each processor
3. Generate an interactive HTML visualization
4. Open it in your browser automatically

### Using Python API

```python
from src.processors.visualization import track_workflow, export_to_html

# Track a workflow
session = track_workflow(
    file_path="inputs/sample1/sample1.jpg",
    template_path="inputs/sample1/template.json",
    capture_processors=["AutoRotate", "CropOnMarkers", "ReadOMR"]
)

# Export to HTML
html_path = export_to_html(
    session,
    output_dir="outputs/visualization",
    title="My OMR Workflow"
)

print(f"Visualization saved to: {html_path}")
```

## CLI Options

### Basic Options

- `-i, --input`: Path to input OMR image (required)
- `-t, --template`: Path to template JSON (required)
- `-c, --config`: Path to config JSON (optional)
- `-o, --output`: Output directory (default: `outputs/visualization`)

### Capture Options

- `--capture-processors`: Comma-separated processor names to capture
  - Default: `"all"` (captures all processors)
  - Example: `"AutoRotate,CropOnMarkers,ReadOMR"`

### Image Options

- `--max-width`: Maximum width for captured images in pixels (default: 800)
- `--quality`: JPEG quality 1-100 (default: 85)
- `--no-colored`: Only capture grayscale images

### Display Options

- `--no-browser`: Don't open visualization in browser automatically
- `--no-json`: Don't export JSON data file
- `--title`: Custom title for the visualization

## Examples

### Example 1: Capture Specific Processors

Only capture key stages of the pipeline:

```bash
uv run python -m src.utils.visualization_runner \
    --input inputs/sample1/sample1.jpg \
    --template inputs/sample1/template.json \
    --capture-processors "AutoRotate,CropOnMarkers,FeatureBasedAlignment,ReadOMR" \
    --output outputs/viz_key_stages
```

### Example 2: High-Quality Images

Capture larger, higher quality images:

```bash
uv run python -m src.utils.visualization_runner \
    --input inputs/sample1/sample1.jpg \
    --template inputs/sample1/template.json \
    --max-width 1600 \
    --quality 95 \
    --output outputs/viz_high_quality
```

### Example 3: Grayscale Only

Reduce file size by only capturing grayscale:

```bash
uv run python -m src.utils.visualization_runner \
    --input inputs/sample1/sample1.jpg \
    --template inputs/sample1/template.json \
    --no-colored \
    --output outputs/viz_grayscale
```

### Example 4: Python API with Custom Processing

```python
from pathlib import Path
from src.processors.visualization import WorkflowTracker, export_to_html
from src.processors.pipeline import ProcessingPipeline
from src.processors.template.template import Template
from src.schemas.models.config import Config
from src.utils.image import ImageUtils

# Load configuration
config = Config.from_file(Path("inputs/sample1/config.json"))
template = Template(Path("inputs/sample1/template.json"), config)

# Load image
gray_image, colored_image = ImageUtils.read_image_util(
    Path("inputs/sample1/sample1.jpg"),
    config
)

# Create tracker
tracker = WorkflowTracker(
    file_path="inputs/sample1/sample1.jpg",
    template_name=template.template_name,
    capture_processors=["all"],
    max_image_width=800,
    include_colored=True
)

# Create and execute pipeline
pipeline = ProcessingPipeline(template)
tracker.build_graph(pipeline.get_processor_names())

# Track each processor
context = ProcessingContext(
    file_path="inputs/sample1/sample1.jpg",
    gray_image=gray_image,
    colored_image=colored_image,
    template=template
)

for processor in pipeline.processors:
    processor_name = processor.get_name()
    tracker.start_processor(processor_name)

    try:
        context = processor.process(context)
        tracker.capture_state(processor_name, context, success=True)
    except Exception as e:
        tracker.capture_state(
            processor_name,
            context,
            success=False,
            error_message=str(e)
        )
        raise

# Finalize and export
session = tracker.finalize()
html_path = export_to_html(session, "outputs/visualization")
```

## Configuration File

You can enable visualization in your `config.json`:

```json
{
  "visualization": {
    "enabled": true,
    "capture_processors": ["all"],
    "capture_frequency": "on_change",
    "include_colored": true,
    "max_image_width": 800,
    "embed_images": true,
    "export_format": "html",
    "output_dir": "outputs/visualization",
    "auto_open_browser": true
  }
}
```

### Configuration Options

- `enabled`: Enable/disable visualization (default: false)
- `capture_processors`: List of processor names or `["all"]`
- `capture_frequency`: `"always"` or `"on_change"` (default: "on_change")
- `include_colored`: Capture colored images (default: true)
- `max_image_width`: Maximum width in pixels (default: 800)
- `embed_images`: Embed images in HTML (default: true)
- `export_format`: `"html"` or `"json"` (default: "html")
- `output_dir`: Output directory path (default: "outputs/visualization")
- `auto_open_browser`: Open in browser automatically (default: true)

## Understanding the Visualization

### Layout

The HTML visualization has three main areas:

1. **Left Panel (Graph)**: Shows the workflow as a flowchart
   - Blue nodes: Input
   - Gray nodes: Processors
   - Green nodes: Output
   - Click nodes to view their output

2. **Right Panel (Image Viewer)**: Displays the current image
   - Shows output from selected processor
   - Toggle between grayscale/colored if available
   - Displays metadata (duration, dimensions, status)

3. **Bottom Bar (Controls)**: Playback and timeline
   - Play/Pause: Animate through processors
   - Previous/Next: Step through manually
   - Timeline: Scrub to any processor
   - Speed: Adjust playback speed (0.5x to 4x)
   - Export: Download session as JSON

### Metadata Panel

Below the graph, you'll see:
- Execution order
- Timestamp
- Any error messages
- Custom processor metadata

## Replaying Saved Sessions

You can replay a previously saved visualization:

```python
from src.processors.visualization import replay_from_json

html_path = replay_from_json(
    "outputs/visualization/sessions/session_20240106_123456_abcd1234.json",
    output_dir="outputs/visualization",
    open_browser=True
)
```

Or using the CLI:

```bash
uv run python -c "
from src.processors.visualization import replay_from_json
replay_from_json('outputs/visualization/sessions/session_20240106_123456.json')
"
```

## Performance Tips

1. **Limit Processors**: Only capture processors you need to debug
2. **Reduce Image Size**: Use `--max-width 400` for faster processing
3. **Lower Quality**: Use `--quality 70` to reduce file size
4. **Grayscale Only**: Use `--no-colored` to save space
5. **External Images**: Set `embed_images: false` in config (requires serving)

## File Outputs

When you run visualization, these files are created:

```
outputs/visualization/
├── session_20240106_123456_abcd1234.html  # Standalone HTML
└── sessions/
    └── session_20240106_123456_abcd1234.json  # Session data
```

### HTML File

- **Standalone**: Contains all images and code
- **Portable**: Share via email or web hosting
- **Size**: 10-30MB typical (depends on settings)

### JSON File

- **Replayable**: Generate new HTML anytime
- **Editable**: Can modify metadata programmatically
- **Compact**: Smaller if you reduce image quality

## Troubleshooting

### Issue: "Template file not found"

**Solution**: Ensure the template path is correct:

```bash
ls -la inputs/sample1/template.json
```

### Issue: "Large HTML files"

**Solutions**:
- Use `--max-width 600` to reduce image size
- Use `--quality 70` to reduce JPEG quality
- Use `--no-colored` to skip colored images
- Capture fewer processors with `--capture-processors`

### Issue: "Browser doesn't open"

**Solutions**:
- The HTML file path is printed in the output
- Open it manually: `file:///path/to/visualization.html`
- Or use `--no-browser` and open manually

### Issue: "Memory error with large images"

**Solutions**:
- Use `--max-width 800` (or lower) to resize images
- Process one file at a time
- Increase system memory if possible

## Advanced Usage

### Custom Processors

You can track custom processors:

```python
from src.processors.base import Processor, ProcessingContext
from src.processors.visualization import WorkflowTracker

class MyCustomProcessor(Processor):
    def process(self, context: ProcessingContext) -> ProcessingContext:
        # Your processing logic
        context.gray_image = my_transform(context.gray_image)
        return context

    def get_name(self) -> str:
        return "MyCustomProcessor"

# Track it
tracker = WorkflowTracker("input.jpg")
tracker.start_processor("MyCustomProcessor")
# ... process ...
tracker.capture_state("MyCustomProcessor", context)
```

### Adding Custom Metadata

```python
tracker.capture_state(
    "MyProcessor",
    context,
    metadata={
        "threshold_used": 180,
        "bubbles_detected": 42,
        "confidence_score": 0.95,
        "custom_flag": True
    }
)
```

This metadata will appear in the visualization's metadata panel.

### Conditional Capture

```python
def should_capture(processor_name: str, context: ProcessingContext) -> bool:
    # Only capture if image changed significantly
    if hasattr(context, 'previous_image'):
        diff = cv2.absdiff(context.gray_image, context.previous_image)
        change = np.sum(diff) / diff.size
        return change > 10  # Threshold
    return True

# Use in tracking
if should_capture(processor_name, context):
    tracker.capture_state(processor_name, context)
```

## API Reference

### `track_workflow()`

High-level function to track a complete workflow.

```python
def track_workflow(
    file_path: Path | str,
    template_path: Path | str,
    config_path: Path | str | None = None,
    capture_processors: list[str] | None = None,
    max_image_width: int = 800,
    include_colored: bool = True,
    image_quality: int = 85
) -> WorkflowSession
```

### `export_to_html()`

Export a session to HTML visualization.

```python
def export_to_html(
    session: WorkflowSession,
    output_dir: Path | str,
    title: str | None = None,
    open_browser: bool = True,
    export_json: bool = True
) -> Path
```

### `replay_from_json()`

Replay a visualization from saved JSON.

```python
def replay_from_json(
    json_path: Path | str,
    output_dir: Path | str | None = None,
    title: str | None = None,
    open_browser: bool = True
) -> Path
```

### `WorkflowTracker`

Low-level tracker for custom workflows.

```python
tracker = WorkflowTracker(
    file_path: Path | str,
    template_name: str = "Unknown",
    capture_processors: list[str] | None = None,
    max_image_width: int = 800,
    include_colored: bool = True,
    image_quality: int = 85,
    config: dict[str, Any] | None = None
)

tracker.start_processor(name: str)
tracker.capture_state(name: str, context: ProcessingContext, ...)
tracker.build_graph(processor_names: list[str])
session = tracker.finalize()
```

## Contributing

Found a bug or have a feature request? Please open an issue on GitHub!

Ideas for future enhancements:
- Side-by-side image comparison
- Diff highlighting between consecutive images
- Processor parameter editing and re-run
- Export to video (MP4) format
- Batch processing multiple files
- Live preview during processing

---

**Need Help?** Join our [Discord community](https://discord.gg/qFv2Vqf) or check the [main documentation](../README.md).

