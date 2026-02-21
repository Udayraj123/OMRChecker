# Workflow Tracker - Constraints

**Python Reference**: `src/processors/visualization/workflow_tracker.py`
**Related**: `src/processors/visualization/workflow_session.py`

## Performance Constraints

### 1. Memory Usage

**Image Storage**:
```python
# Default settings
max_image_width = 800      # Resize to 800px width
image_quality = 85         # JPEG quality (0-100)
include_colored = True     # Store both gray and colored
```

**Memory Calculation**:
```
Per Processor State:
- Gray image (800x600): ~50-100 KB (JPEG compressed)
- Colored image (800x600): ~150-300 KB (JPEG compressed)
- Metadata: ~1-5 KB

Total per state: ~200-400 KB

For 10 processors: ~2-4 MB
For 100 images (batch): ~200-400 MB
```

**Python Constraints**:
- No hard limit in Python (system memory only)
- Base64 encoding adds ~33% overhead
- Session can grow large with many processors

**Browser Constraints**:
- **Memory Limit**: 2-4 GB per tab (browser-dependent)
- **Base64 Overhead**: 33% increase in size
- **IndexedDB Limit**:
  - Chrome/Edge: 60% of available disk space
  - Firefox: 50% of available disk space
  - Safari: 1 GB per origin
- **Recommendation**: Process in batches, clear after export

---

### 2. Image Encoding Performance

**Python (OpenCV + base64)**:
```python
def encode_image(image: MatLike, max_width: int = 800, quality: int = 85) -> str:
    # Resize: ~5-10ms for typical image
    if w > max_width:
        image = cv2.resize(image, (new_width, new_height))

    # Encode to JPEG: ~10-30ms
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])

    # Base64 encode: ~5-10ms
    return base64.b64encode(buffer).decode("utf-8")

# Total: ~20-50ms per image
```

**Browser (OpenCV.js + Canvas)**:
```typescript
async encodeImage(mat: cv.Mat, maxWidth: number = 800, quality: number = 85): Promise<string> {
  // Resize: ~10-20ms (slightly slower than Python)
  const resized = resizeIfNeeded(mat, maxWidth);

  // Encode to JPEG: ~20-50ms (Canvas API)
  const canvas = matToCanvas(resized);
  const blob = await canvasToBlob(canvas, 'image/jpeg', quality / 100);

  // Base64 encode: ~5-15ms
  const base64 = await blobToBase64(blob);

  // Total: ~35-85ms per image (slower than Python)
}
```

**Impact on Pipeline**:
```
Without tracking: ~500ms for 10 processors
With tracking (10 images * 50ms): ~1000ms total
Overhead: ~100% increase in processing time

Browser overhead: ~150-200% (due to slower encoding)
```

**Optimization**:
- Reduce `max_image_width` (e.g., 600px or 400px)
- Lower `image_quality` (e.g., 70-75)
- Disable `include_colored` if only gray needed
- Use Web Workers for parallel encoding (browser)

---

### 3. Selective Capture Impact

**Capture All (Default)**:
```python
tracker = WorkflowTracker(
    file_path="input.jpg",
    template_name="Sample",
    capture_processors=None  # All processors
)

# 10 processors × 50ms encoding = 500ms overhead
```

**Capture Specific Processors**:
```python
tracker = WorkflowTracker(
    file_path="input.jpg",
    template_name="Sample",
    capture_processors=["AutoRotate", "CropOnMarkers", "ReadOMR"]
)

# 3 processors × 50ms encoding = 150ms overhead
# 70% reduction in overhead
```

**Recommendation**:
- **Debug mode**: Capture all for full visibility
- **Production**: Capture only critical processors
- **Performance testing**: Disable tracking entirely

---

### 4. Session File Size

**Example Session**:
```json
{
  "session_id": "session_20240220_103045_a1b2c3d4",
  "file_path": "inputs/sample1.jpg",
  "template_name": "Sample",
  "start_time": "2024-02-20T10:30:45.123Z",
  "end_time": "2024-02-20T10:30:47.456Z",
  "total_duration_ms": 2333.0,
  "processor_states": [...],  // 10 states × 300 KB = 3 MB
  "graph": {...},              // ~5 KB
  "config": {...},             // ~2 KB
  "metadata": {...}            // ~1 KB
}

// Total: ~3 MB per session
```

**File Size Constraints**:
- **Python**: No limit (disk space only)
- **Browser Download**:
  - No size limit for downloads
  - Large files (>10 MB) may be slow
- **IndexedDB Storage**:
  - Chrome/Edge: Up to 60% of disk space
  - Firefox: Up to 50% of disk space
  - Safari: 1 GB per origin
  - Recommendation: Store only recent sessions

---

## Browser Visualization Constraints

### 1. Rendering Encoded Images

**Decoding Base64 to Image**:

```typescript
// Method 1: Canvas API (faster)
function base64ToCanvas(base64: string): HTMLCanvasElement {
  const img = new Image();
  img.src = `data:image/jpeg;base64,${base64}`;

  await img.decode(); // Wait for decode (~10-30ms)

  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;

  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0);

  return canvas;
}

// Method 2: OpenCV.js (for processing)
function base64ToMat(base64: string): cv.Mat {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return cv.imdecode(cv.matFromArray(bytes.length, 1, cv.CV_8U, bytes), cv.IMREAD_UNCHANGED);
}
```

**Performance**:
- **Decode time**: 10-30ms per image
- **Canvas rendering**: 5-10ms
- **Total**: 15-40ms per state display

**For 10 processor states**:
- Sequential: 150-400ms
- With lazy loading: Only visible images

---

### 2. Step-by-Step Visualization UI

**Typical UI Components**:

```typescript
interface WorkflowVisualizerProps {
  session: WorkflowSession;
}

function WorkflowVisualizer({ session }: WorkflowVisualizerProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const currentState = session.processorStates[currentStep];

  return (
    <div className="workflow-visualizer">
      {/* Timeline */}
      <Timeline
        states={session.processorStates}
        currentStep={currentStep}
        onStepChange={setCurrentStep}
      />

      {/* Image Display */}
      <ImageDisplay
        grayImageBase64={currentState.grayImageBase64}
        coloredImageBase64={currentState.coloredImageBase64}
      />

      {/* Metadata Panel */}
      <MetadataPanel
        name={currentState.name}
        duration={currentState.durationMs}
        metadata={currentState.metadata}
        success={currentState.success}
        errorMessage={currentState.errorMessage}
      />

      {/* Graph View */}
      <GraphView
        graph={session.graph}
        currentNode={`processor_${currentStep}`}
      />
    </div>
  );
}
```

**Performance Considerations**:
- **Lazy Image Loading**: Only decode visible images
- **Virtual Scrolling**: For long processor lists
- **Canvas Reuse**: Reuse canvas elements
- **Debounced Updates**: Throttle step changes

---

### 3. Graph Visualization

**Libraries**:

```typescript
// Option 1: D3.js (flexible, powerful)
import * as d3 from 'd3';

function renderGraph(graph: WorkflowGraph) {
  const svg = d3.select('#graph-container')
    .append('svg')
    .attr('width', 800)
    .attr('height', 600);

  // Force-directed layout
  const simulation = d3.forceSimulation(graph.nodes)
    .force('link', d3.forceLink(graph.edges).id(d => d.id))
    .force('charge', d3.forceManyBody().strength(-100))
    .force('center', d3.forceCenter(400, 300));

  // Render nodes and edges
  // ...
}

// Option 2: React Flow (React-based, simpler)
import ReactFlow, { Node, Edge } from 'react-flow-renderer';

function GraphView({ graph }: { graph: WorkflowGraph }) {
  const nodes: Node[] = graph.nodes.map((node, i) => ({
    id: node.id,
    data: { label: node.label },
    position: { x: i * 150, y: 100 },
  }));

  const edges: Edge[] = graph.edges.map((edge, i) => ({
    id: `edge-${i}`,
    source: edge.from,
    target: edge.to,
    label: edge.label,
  }));

  return <ReactFlow nodes={nodes} edges={edges} />;
}

// Option 3: Mermaid (declarative, markdown-like)
import mermaid from 'mermaid';

function generateMermaidDiagram(graph: WorkflowGraph): string {
  let diagram = 'graph TD\n';

  graph.edges.forEach(edge => {
    diagram += `  ${edge.from}[${getLabelForNode(edge.from)}] --> ${edge.to}[${getLabelForNode(edge.to)}]\n`;
  });

  return diagram;
}
```

**Performance**:
- **D3.js**: Best for complex graphs, ~50-100ms render
- **React Flow**: Best for React apps, ~30-50ms render
- **Mermaid**: Simplest, ~20-40ms render
- **Recommendation**: Use Mermaid for simple pipelines, React Flow for interactive UI

---

### 4. Timeline Navigation

**Scrubbing Performance**:

```typescript
function Timeline({ states, currentStep, onStepChange }) {
  const [isPlaying, setIsPlaying] = useState(false);

  // Auto-play with interval
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      onStepChange(step => (step + 1) % states.length);
    }, 1000); // 1 second per step

    return () => clearInterval(interval);
  }, [isPlaying, states.length]);

  return (
    <div className="timeline">
      {/* Scrubber */}
      <input
        type="range"
        min={0}
        max={states.length - 1}
        value={currentStep}
        onChange={(e) => onStepChange(parseInt(e.target.value))}
      />

      {/* Play/Pause */}
      <button onClick={() => setIsPlaying(!isPlaying)}>
        {isPlaying ? 'Pause' : 'Play'}
      </button>

      {/* Step indicators */}
      <div className="steps">
        {states.map((state, i) => (
          <div
            key={i}
            className={`step ${i === currentStep ? 'active' : ''} ${state.success ? 'success' : 'error'}`}
            onClick={() => onStepChange(i)}
          >
            {state.name}
          </div>
        ))}
      </div>
    </div>
  );
}
```

**Constraints**:
- **Smooth Scrubbing**: Decode images in background
- **Play Speed**: 500ms-2000ms per step (configurable)
- **Large Sessions**: Virtual scrolling for 100+ states

---

### 5. Export Options

**Browser Export Formats**:

```typescript
// 1. JSON Download (most common)
async function downloadSessionJSON(session: WorkflowSession) {
  const json = JSON.stringify(session, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = `${session.sessionId}.json`;
  a.click();

  URL.revokeObjectURL(url);
}

// 2. HTML Report (standalone visualization)
async function exportHTMLReport(session: WorkflowSession) {
  const html = generateHTMLReport(session);
  const blob = new Blob([html], { type: 'text/html' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = `${session.sessionId}_report.html`;
  a.click();

  URL.revokeObjectURL(url);
}

// 3. PDF Report (using jsPDF + html2canvas)
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

async function exportPDFReport(session: WorkflowSession) {
  const element = document.getElementById('workflow-visualizer')!;
  const canvas = await html2canvas(element);

  const pdf = new jsPDF('p', 'mm', 'a4');
  const imgData = canvas.toDataURL('image/png');
  pdf.addImage(imgData, 'PNG', 10, 10, 190, 0);
  pdf.save(`${session.sessionId}_report.pdf`);
}

// 4. IndexedDB Storage (for later viewing)
async function saveToIndexedDB(session: WorkflowSession) {
  const db = await openDB('omrchecker-sessions', 1);
  await db.put('sessions', session, session.sessionId);
}

async function loadFromIndexedDB(sessionId: string): Promise<WorkflowSession> {
  const db = await openDB('omrchecker-sessions', 1);
  return await db.get('sessions', sessionId);
}
```

**File Size Limits**:
- **JSON**: No limit (download)
- **HTML**: Inline base64 can be large (5-50 MB)
- **PDF**: Limited by jsPDF (~100 MB max)
- **IndexedDB**: 1 GB per origin (Safari), 60% disk space (Chrome)

---

## Optimization Strategies

### 1. Reduce Image Size

```python
# Python
tracker = WorkflowTracker(
    file_path="input.jpg",
    template_name="Sample",
    max_image_width=600,      # Smaller width
    image_quality=70,         # Lower quality
    include_colored=False,    # Only gray images
)

# Savings:
# - 600px vs 800px: ~44% size reduction
# - Quality 70 vs 85: ~30% size reduction
# - Gray only: ~66% size reduction
# Combined: ~80% size reduction
```

### 2. Lazy Loading in Browser

```typescript
function ImageDisplay({ grayImageBase64 }: { grayImageBase64: string | null }) {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!grayImageBase64) return;

    setIsLoading(true);

    // Decode in background
    const img = new Image();
    img.src = `data:image/jpeg;base64,${grayImageBase64}`;

    img.onload = () => {
      setImageUrl(img.src);
      setIsLoading(false);
    };
  }, [grayImageBase64]);

  if (isLoading) return <div>Loading...</div>;
  if (!imageUrl) return <div>No image</div>;

  return <img src={imageUrl} alt="Processor output" />;
}
```

### 3. Compression (Advanced)

```typescript
// Use pako for gzip compression
import pako from 'pako';

function compressSession(session: WorkflowSession): Uint8Array {
  const json = JSON.stringify(session);
  const compressed = pako.gzip(json);
  return compressed;
}

function decompressSession(compressed: Uint8Array): WorkflowSession {
  const decompressed = pako.ungzip(compressed, { to: 'string' });
  return JSON.parse(decompressed);
}

// Savings: ~50-70% size reduction
// Trade-off: ~50-100ms compression time
```

### 4. Streaming Large Sessions

```typescript
// For very large sessions, stream processor states
async function* streamProcessorStates(sessionId: string) {
  const db = await openDB('omrchecker-sessions', 1);
  const session = await db.get('sessions', sessionId);

  for (const state of session.processorStates) {
    yield state;
  }
}

// Usage
for await (const state of streamProcessorStates(sessionId)) {
  renderProcessorState(state);
}
```

---

## Key Constraints Summary

### Python Constraints
1. **Memory**: Limited by system RAM (~GB scale)
2. **Encoding**: 20-50ms per image (fast)
3. **File Size**: 2-4 MB per session (10 processors)
4. **No Hard Limits**: Can process hundreds of sessions

### Browser Constraints
1. **Memory**: 2-4 GB per tab (strict limit)
2. **Encoding**: 35-85ms per image (slower)
3. **File Size**: Same as Python (2-4 MB)
4. **Storage**: IndexedDB (1 GB Safari, 60% disk Chrome)
5. **Rendering**: 15-40ms per state display
6. **UI Performance**: Use lazy loading, virtual scrolling

### Recommendations
1. **Development**: Capture all, high quality
2. **Production**: Capture selectively, optimize size
3. **Browser**: Reduce image size, use Web Workers
4. **Large Batches**: Process in chunks, clear after export
5. **Visualization**: Lazy load images, virtual scroll timeline
6. **Storage**: IndexedDB for recent sessions, download for archive
