# HTML Exporter Flows

**Module**: Domain - Visualization - HTML Export
**Python Reference**: `src/processors/visualization/html_exporter.py`
**Last Updated**: 2026-02-21

---

## Overview

HTML Exporter generates standalone, interactive HTML visualizations from WorkflowSession data. It embeds all images, flowcharts, and session metadata into a single shareable file that can be opened in any browser.

**Use Case**: Create portable workflow visualizations for debugging, sharing, and documentation.

---

## Main Export Flow

### HTMLExporter.export()

```
START: HTMLExporter.export(session, output_path, title, open_browser)
│
├─► STEP 1: Initialize Output Path
│   │
│   output_path = Path(output_path)
│   │
│   ├─ Create parent directories if needed
│   │  output_path.parent.mkdir(parents=True, exist_ok=True)
│   │
│   └─ Example:
│      Input:  "outputs/viz/report.html"
│      Creates: outputs/viz/ (if doesn't exist)
│
├─► STEP 2: Generate Title
│   │
│   if title is None:
│   │   title = f"OMR Workflow Visualization - {session.session_id}"
│   │
│   │ Example:
│   │ session_id: "session_20240106_123456_abcd1234"
│   │ title: "OMR Workflow Visualization - session_20240106_123456_abcd1234"
│
├─► STEP 3: Read HTML Template
│   │
│   template_content = self.template_path.read_text()
│   │
│   │ Default template: src/processors/visualization/templates/viewer.html
│   │ Size: ~25 KB (HTML + CSS + JavaScript)
│   │
│   │ Template contains placeholders:
│   │ - {{ title }}
│   │ - {{ session_id }}
│   │ - {{ file_path }}
│   │ - {{ total_duration_ms }}
│   │ - {{ session_data }}
│
├─► STEP 4: Prepare Session Data
│   │
│   session_json = json.dumps(session.to_dict())
│   │
│   │ session.to_dict() returns:
│   │ {
│   │   "session_id": str,
│   │   "file_path": str,
│   │   "template_name": str,
│   │   "start_time": str (ISO format),
│   │   "end_time": str (ISO format),
│   │   "total_duration_ms": float,
│   │   "processor_states": [
│   │     {
│   │       "name": str,
│   │       "order": int,
│   │       "timestamp": str,
│   │       "duration_ms": float,
│   │       "image_shape": [h, w, c],
│   │       "gray_image_base64": str (JPEG base64),
│   │       "colored_image_base64": str (JPEG base64, optional),
│   │       "metadata": dict,
│   │       "success": bool,
│   │       "error_message": str (optional)
│   │     },
│   │     ...
│   │   ],
│   │   "graph": {
│   │     "nodes": [
│   │       {
│   │         "id": str,
│   │         "label": str,
│   │         "metadata": {"type": "input"|"processor"|"output", "order": int}
│   │       },
│   │       ...
│   │     ],
│   │     "edges": [
│   │       {"from": str, "to": str},
│   │       ...
│   │     ]
│   │   },
│   │   "config": dict,
│   │   "metadata": dict
│   │ }
│   │
│   │ Typical size: 500 KB - 5 MB (depends on image count/quality)
│
├─► STEP 5: Format Duration String
│   │
│   if session.total_duration_ms is not None:
│   │   duration_str = f"{session.total_duration_ms:.2f}"
│   else:
│   │   duration_str = "N/A"
│   │
│   │ Example:
│   │ 1234.5678 → "1234.57"
│
├─► STEP 6: Substitute Template Variables
│   │
│   html_content = template_content.replace("{{ title }}", title)
│   html_content = html_content.replace("{{ session_id }}", session.session_id)
│   html_content = html_content.replace("{{ file_path }}", session.file_path)
│   html_content = html_content.replace("{{ total_duration_ms }}", duration_str)
│   html_content = html_content.replace("{{ session_data }}", session_json)
│   │
│   │ Note: Simple string replacement, not a full template engine
│   │ {{ session_data }} is embedded as JavaScript variable
│
├─► STEP 7: Write Output File
│   │
│   output_path.write_text(html_content)
│   logger.info(f"Exported workflow visualization to: {output_path}")
│   │
│   │ File size: 500 KB - 10 MB depending on:
│   │ - Number of processor states
│   │ - Image resolution (max_width=800 by default)
│   │ - JPEG quality (85 by default)
│
├─► STEP 8: Open in Browser (Optional)
│   │
│   if open_browser:
│   │   webbrowser.open(f"file://{output_path.absolute()}")
│   │   logger.info("Opened visualization in browser")
│   │
│   │ Opens default system browser with file:// URL
│   │ Works on Windows/Mac/Linux
│
└─► STEP 9: Return Path
    │
    return output_path  # Path to generated HTML file
    END
```

---

## Export with JSON Flow

### HTMLExporter.export_with_json()

```
START: export_with_json(session, output_dir, title, open_browser)
│
├─► STEP 1: Initialize Output Directory
│   │
│   output_dir = Path(output_dir)
│   output_dir.mkdir(parents=True, exist_ok=True)
│   │
│   │ Creates output directory if doesn't exist
│
├─► STEP 2: Generate File Names
│   │
│   base_name = session.session_id
│   html_path = output_dir / f"{base_name}.html"
│   json_path = output_dir / "sessions" / f"{base_name}.json"
│   │
│   │ Example:
│   │ output_dir: outputs/visualization/
│   │ base_name: session_20240106_123456_abcd1234
│   │ html_path: outputs/visualization/session_20240106_123456_abcd1234.html
│   │ json_path: outputs/visualization/sessions/session_20240106_123456_abcd1234.json
│
├─► STEP 3: Export HTML
│   │
│   html_path = self.export(session, html_path, title, open_browser)
│   │
│   │ Calls main export() method (see above)
│
├─► STEP 4: Export JSON
│   │
│   session.save_to_file(json_path)
│   │
│   │ Creates sessions/ subdirectory if needed
│   │ Writes JSON file with session data
│   │ JSON is formatted with indent=2 for readability
│   │
│   logger.info(f"Exported session JSON to: {json_path}")
│
└─► STEP 5: Return Paths
    │
    return (html_path, json_path)
    END
```

---

## High-Level Export Function

### export_to_html()

```
START: export_to_html(session, output_dir, title, open_browser, export_json)
│
├─► STEP 1: Create Exporter
│   │
│   exporter = HTMLExporter()
│   │
│   │ Uses default template (viewer.html)
│
├─► STEP 2: Choose Export Path
│   │
│   if export_json:
│   │   ├─ Call export_with_json()
│   │   └─ Returns both HTML and JSON paths
│   else:
│   │   ├─ Create output directory
│   │   │  output_dir.mkdir(parents=True, exist_ok=True)
│   │   ├─ Generate HTML path
│   │   │  html_path = output_dir / f"{session.session_id}.html"
│   │   └─ Call export()
│
└─► STEP 3: Return HTML Path
    │
    return html_path
    END
```

**Usage Example**:
```python
from src.processors.visualization.workflow_tracker import track_workflow
from src.processors.visualization.html_exporter import export_to_html

# Track workflow execution
session = track_workflow(
    file_path="inputs/sample1/sample1.jpg",
    template_path="inputs/sample1/template.json"
)

# Export to HTML (with JSON)
html_path = export_to_html(session, "outputs/visualization")
print(f"Visualization saved to: {html_path}")
# Opens in browser automatically by default
```

---

## Replay from JSON Flow

### replay_from_json()

```
START: replay_from_json(json_path, output_dir, title, open_browser)
│
├─► STEP 1: Validate JSON Path
│   │
│   json_path = Path(json_path)
│   │
│   if not json_path.exists():
│   │   raise FileNotFoundError(f"JSON file not found: {json_path}")
│
├─► STEP 2: Load Session
│   │
│   session = WorkflowSession.load_from_file(json_path)
│   logger.info(f"Loaded session from: {json_path}")
│   │
│   │ Deserializes JSON to WorkflowSession object
│   │ Reconstructs all processor states, graph, metadata
│
├─► STEP 3: Determine Output Directory
│   │
│   if output_dir is None:
│   │   output_dir = json_path.parent.parent
│   │   # Go up from 'sessions' directory
│   │
│   │ Example:
│   │ json_path: outputs/visualization/sessions/session_xxx.json
│   │ output_dir: outputs/visualization/
│
└─► STEP 4: Export to HTML
    │
    return export_to_html(
        session,
        output_dir,
        title=title,
        open_browser=open_browser,
        export_json=False  # Don't re-export JSON
    )
    END
```

**Usage Example**:
```python
from src.processors.visualization.html_exporter import replay_from_json

# Replay visualization from saved JSON
html_path = replay_from_json(
    "outputs/visualization/sessions/session_20240106_123456_abcd1234.json"
)
# Regenerates HTML from JSON and opens in browser
```

---

## Template Substitution Details

### Template Variables

The HTML template (`viewer.html`) contains 5 placeholders:

```html
<title>{{ title }}</title>
<!-- Appears in browser tab -->

<h1>{{ title }}</h1>
<!-- Appears in page header -->

Session: {{ session_id }} | File: {{ file_path }} | Duration: {{ total_duration_ms }}ms
<!-- Appears in subtitle -->

<script>
    const sessionData = {{ session_data }};
    // Embedded as JavaScript object
</script>
```

### Substitution Order

All substitutions happen in sequence (not simultaneously):

```python
html_content = template_content.replace("{{ title }}", title)
html_content = html_content.replace("{{ session_id }}", session.session_id)
html_content = html_content.replace("{{ file_path }}", session.file_path)
html_content = html_content.replace("{{ total_duration_ms }}", duration_str)
html_content = html_content.replace("{{ session_data }}", session_json)
```

**No Escaping**: Values are inserted directly without HTML/JavaScript escaping.

**Potential Issue**: If `session.file_path` contains `{{ title }}`, it will NOT be re-substituted (already processed).

**Browser Safety**: Session data is embedded in `<script>` tag. If `session_data` contains `</script>`, it could break JavaScript. This is mitigated by JSON encoding (no raw `</script>` should appear).

---

## Visualization Features

### Interactive Flowchart (vis.js)

The HTML template uses `vis-network@9.1.6` for graph visualization:

```javascript
// Nodes
const nodes = new vis.DataSet(sessionData.graph.nodes.map(node => ({
    id: node.id,
    label: node.label,
    shape: node.metadata.type === 'input' ? 'ellipse' :
           node.metadata.type === 'output' ? 'ellipse' : 'box',
    color: node.metadata.type === 'input' ? '#3498db' :
           node.metadata.type === 'output' ? '#27ae60' : '#ecf0f1',
    metadata: node.metadata
})));

// Edges
const edges = new vis.DataSet(sessionData.graph.edges.map(edge => ({
    from: edge.from,
    to: edge.to,
    arrows: 'to',
    color: { color: '#bdc3c7' }
})));

// Layout: Hierarchical left-to-right
const options = {
    layout: {
        hierarchical: {
            direction: 'LR',
            sortMethod: 'directed',
            nodeSpacing: 150,
            levelSeparation: 200
        }
    },
    physics: false
};
```

**Node Colors**:
- Input: Blue (`#3498db`)
- Processor: Light gray (`#ecf0f1`)
- Output: Green (`#27ae60`)

### Image Viewer

Images are displayed with toggle between grayscale and colored:

```javascript
const hasGray = state.gray_image_base64;
const hasColored = state.colored_image_base64;
const imageData = showColored && hasColored
    ? state.colored_image_base64
    : state.gray_image_base64;

// Display as data URI
img.src = 'data:image/jpeg;base64,' + imageData;
```

**Toggle Button**: Only shown if `colored_image_base64` exists.

### Playback Controls

```
Controls:
├─ Play/Pause button
├─ Previous step button
├─ Next step button
├─ Timeline slider (clickable)
├─ Speed control (0.5x, 1x, 2x, 4x)
└─ Export JSON button
```

**Timeline Slider**:
```javascript
// Click on timeline to jump to step
document.getElementById('timeline-slider').addEventListener('click', function(e) {
    const rect = this.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percent = x / rect.width;
    currentStep = Math.floor(percent * sessionData.processor_states.length);
    updateUI();
});
```

**Speed Control**:
```html
<select id="speed-select">
    <option value="2000">0.5x</option>
    <option value="1000" selected>1x</option>
    <option value="500">2x</option>
    <option value="250">4x</option>
</select>
```

Values are milliseconds between steps.

### Export JSON from Browser

Users can re-export session data from the browser:

```javascript
function exportJSON() {
    const dataStr = JSON.stringify(sessionData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = sessionData.session_id + '.json';
    link.click();
    URL.revokeObjectURL(url);
}
```

Triggers browser download of session JSON.

---

## File Size Optimization

### Image Encoding Parameters

From `workflow_session.py`:

```python
class ImageEncoder:
    @staticmethod
    def encode_image(image, max_width=800, quality=85):
        # Resize if needed
        if max_width is not None and w > max_width:
            scale = max_width / w
            new_width = max_width
            new_height = int(h * scale)
            image = cv2.resize(image, (new_width, new_height))

        # Encode to JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode(".jpg", image, encode_param)

        # Convert to base64
        return base64.b64encode(buffer).decode("utf-8")
```

**Default Settings**:
- `max_width`: 800px (maintains aspect ratio)
- `quality`: 85 (JPEG quality, 0-100)

**Typical Sizes**:
```
Original image: 2000×1500 px = 3 MB
After resize: 800×600 px
After JPEG 85: ~150 KB
After base64: ~200 KB (33% overhead)

Per processor state (gray + colored): ~400 KB
10 processor states: ~4 MB
Total HTML file: ~4.5 MB (includes template + JSON structure)
```

### Reducing File Size

**Option 1: Lower max_width**
```python
# Create custom session with smaller images
encoder = ImageEncoder()
for state in session.processor_states:
    state.gray_image_base64 = encoder.encode_image(image, max_width=400, quality=85)
```

**Option 2: Lower quality**
```python
state.gray_image_base64 = encoder.encode_image(image, max_width=800, quality=70)
```

**Option 3: Skip colored images**
```python
# Only store grayscale
state.colored_image_base64 = None
```

**Option 4: Exclude some processor states**
```python
# Only keep key states (input, alignment, final)
session.processor_states = [
    session.processor_states[0],   # Input
    session.processor_states[5],   # Alignment
    session.processor_states[-1]   # Final
]
```

---

## Error Handling

### Template Not Found

```python
def __init__(self, template_path=None):
    if template_path is None:
        template_path = Path(__file__).parent / "templates" / "viewer.html"

    self.template_path = Path(template_path)

    if not self.template_path.exists():
        raise FileNotFoundError(f"Template file not found: {self.template_path}")
```

**Raised**: `FileNotFoundError` if template missing.

### JSON Not Found (Replay)

```python
def replay_from_json(json_path, ...):
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
```

**Raised**: `FileNotFoundError` if JSON missing.

### Browser Loading Errors

The HTML template has error handling:

```javascript
try {
    console.log('Starting initialization...');
    init();
    console.log('Initialization complete');
} catch (error) {
    console.error('Fatal error during initialization:', error);
    alert('Failed to initialize visualization: ' + error.message);
}
```

```javascript
function initGraph() {
    try {
        // Check if vis is loaded
        if (typeof vis === 'undefined') {
            container.innerHTML = '<div style="padding: 20px; color: red;">Error: Visualization library failed to load. Check your internet connection.</div>';
            return;
        }

        // Initialize graph...
    } catch (error) {
        console.error('Error initializing graph:', error);
        container.innerHTML = '<div style="padding: 20px; color: red;">Error initializing graph: ' + error.message + '</div>';
    }
}
```

**Common Browser Errors**:
1. **vis.js CDN unavailable**: Shows error message, requires internet
2. **Invalid JSON**: JavaScript parse error, shows in console
3. **Missing images**: Broken image icons in viewer

---

## Browser Migration

### Python to JavaScript Mapping

| Python | JavaScript | Notes |
|--------|-----------|-------|
| `Path(output_path)` | `File` API | Use File/Blob for output |
| `pathlib.Path.write_text()` | Download via `<a>` | See constraints.md |
| `webbrowser.open()` | Automatic (already in browser) | N/A |
| `json.dumps()` | `JSON.stringify()` | Same API |
| `base64.b64encode()` | `btoa()` | Browser built-in |

### File Writing (Download)

In browser, "write file" becomes "download file":

```javascript
function exportHTML(sessionData, filename) {
    // 1. Read template (fetch or inline)
    const templateHTML = await fetch('templates/viewer.html').then(r => r.text());

    // 2. Prepare session JSON
    const sessionJSON = JSON.stringify(sessionData);

    // 3. Substitute variables
    let html = templateHTML;
    html = html.replace('{{ title }}', `OMR Workflow - ${sessionData.session_id}`);
    html = html.replace('{{ session_id }}', sessionData.session_id);
    html = html.replace('{{ file_path }}', sessionData.file_path);
    html = html.replace('{{ total_duration_ms }}', sessionData.total_duration_ms.toFixed(2));
    html = html.replace('{{ session_data }}', sessionJSON);

    // 4. Create Blob
    const blob = new Blob([html], { type: 'text/html' });

    // 5. Download
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
}
```

### Opening in Browser

Since code is already running in browser, "open in browser" is automatic:

```javascript
// Python: webbrowser.open(path)
// Browser: Already open! Just navigate

// Option 1: Open in new tab
window.open(url, '_blank');

// Option 2: Replace current page
window.location.href = url;

// Option 3: Open in iframe
document.getElementById('viewer-frame').src = url;
```

### Template Loading

**Python**: Template loaded from file system
**Browser**: Three options:

1. **Inline Template** (recommended):
   ```javascript
   const TEMPLATE = `
   <!DOCTYPE html>
   <html>...entire template...</html>
   `;
   ```

2. **Fetch Template**:
   ```javascript
   const template = await fetch('/templates/viewer.html').then(r => r.text());
   ```

3. **Bundle Template**:
   ```javascript
   import viewerTemplate from './templates/viewer.html?raw';
   // Using Vite/Webpack raw loader
   ```

**Recommendation**: Inline template (ensures offline functionality).

---

## Related Documentation

- **Workflow Session**: `../workflow-session/flows.md` (data model)
- **Workflow Tracker**: `../workflow-tracker/flows.md` (session creation)
- **Browser File I/O**: `../../../integration/file-io/file-io.md`

---

## Summary

HTML Exporter:

1. **Reads WorkflowSession** with processor states and images
2. **Loads HTML template** with placeholders
3. **Serializes session to JSON** and embeds in template
4. **Substitutes variables** (title, session_id, file_path, duration, session_data)
5. **Writes standalone HTML file** (500 KB - 10 MB)
6. **Optionally opens in browser** using `webbrowser.open()`
7. **Supports JSON export** for later replay

**Browser Visualization Features**:
- Interactive flowchart (vis.js)
- Image viewer with grayscale/colored toggle
- Playback controls (play, step, speed)
- Timeline slider
- Metadata display
- Export JSON from browser

**Best For**: Debugging workflows, sharing results, documentation
**File Format**: Single HTML file (portable, no external dependencies except vis.js CDN)
