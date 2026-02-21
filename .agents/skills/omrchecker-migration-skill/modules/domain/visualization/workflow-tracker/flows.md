# Workflow Tracker - Flows

**Python Reference**: `src/processors/visualization/workflow_tracker.py`
**Related**: `src/processors/visualization/workflow_session.py`

## Overview

The Workflow Tracker provides comprehensive tracking and visualization capabilities for the OMR processing pipeline. It captures the state of images as they flow through each processor, enabling step-by-step visualization, debugging, and replay of the entire workflow.

---

## Core Classes

### 1. WorkflowTracker

Main tracker class that wraps pipeline execution and captures intermediate states.

```python
class WorkflowTracker:
    """Tracks workflow execution for visualization purposes.

    Captures:
    - Processor execution order and timing
    - Input/output images at each stage
    - Processor metadata and success/failure states
    - Workflow graph structure
    """

    def __init__(
        self,
        file_path: Path | str,
        template_name: str = "Unknown",
        capture_processors: list[str] | None = None,
        max_image_width: int = 800,
        include_colored: bool = True,
        image_quality: int = 85,
        config: dict[str, Any] | None = None,
    )
```

**Key Attributes**:
- `session`: WorkflowSession containing all captured data
- `capture_processors`: Filter which processors to capture (["all"] for all)
- `max_image_width`: Resize images for storage optimization
- `include_colored`: Whether to capture colored images
- `image_quality`: JPEG quality (0-100)
- `_start_times`: Track start times for duration calculation
- `_execution_order`: Sequential order counter

---

## Tracking Flow

### Step 1: Initialization

```python
# Create unique session ID with timestamp
session_id = f"session_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

# Initialize session
self.session = WorkflowSession(
    session_id=session_id,
    file_path=str(file_path),
    template_name=template_name,
    start_time=datetime.now(UTC).isoformat(),
    config=config or {},
)
```

**What happens**:
1. Generate unique session ID with timestamp and UUID
2. Create WorkflowSession with metadata
3. Initialize tracking state (start times, execution order)

---

### Step 2: Build Workflow Graph

```python
def build_graph(self, processor_names: list[str]) -> None:
    """Build the workflow graph structure from processor names."""

    # Add input node
    self.session.graph.add_node(
        node_id="input",
        label="Input Image",
        metadata={"type": "input", "file_path": self.session.file_path}
    )

    # Add processor nodes
    for i, name in enumerate(processor_names):
        self.session.graph.add_node(
            node_id=f"processor_{i}",
            label=name,
            metadata={"type": "processor", "order": i}
        )

    # Add output node
    self.session.graph.add_node(
        node_id="output",
        label="Output",
        metadata={"type": "output"}
    )

    # Connect nodes with edges
    # input -> processor_0 -> processor_1 -> ... -> output
```

**Graph Structure**:
```
[Input Image]
    ↓
[PreprocessingCoordinator] (processor_0)
    ↓
[AlignmentProcessor] (processor_1)
    ↓
[ReadOMRProcessor] (processor_2)
    ↓
[Output]
```

---

### Step 3: Track Processor Execution

For each processor in the pipeline:

#### 3a. Mark Start

```python
def start_processor(self, processor_name: str) -> None:
    """Mark the start of a processor execution."""
    if self.should_capture(processor_name):
        self._start_times[processor_name] = time.time()
        logger.debug(f"[Tracker] Started tracking: {processor_name}")
```

#### 3b. Execute Processor

```python
# Pipeline executes the processor
context = processor.process(context)
```

#### 3c. Capture State

```python
def capture_state(
    self,
    processor_name: str,
    context: ProcessingContext,
    metadata: dict[str, Any] | None = None,
    success: bool = True,
    error_message: str | None = None,
) -> None:
    """Capture the current state after a processor execution."""

    # 1. Calculate duration
    start_time = self._start_times.get(processor_name)
    duration_ms = 0.0
    if start_time is not None:
        duration_ms = (time.time() - start_time) * 1000

    # 2. Encode images to base64 JPEG
    gray_image_base64 = None
    colored_image_base64 = None
    image_shape = (0, 0)

    if context.gray_image is not None:
        gray_image_base64 = ImageEncoder.encode_image(
            context.gray_image,
            max_width=self.max_image_width,
            quality=self.image_quality,
        )
        image_shape = context.gray_image.shape

    if self.include_colored and context.colored_image is not None:
        colored_image_base64 = ImageEncoder.encode_image(
            context.colored_image,
            max_width=self.max_image_width,
            quality=self.image_quality,
        )

    # 3. Create processor state
    state = ProcessorState(
        name=processor_name,
        order=self._execution_order,
        timestamp=datetime.now(UTC).isoformat(),
        duration_ms=duration_ms,
        image_shape=image_shape,
        gray_image_base64=gray_image_base64,
        colored_image_base64=colored_image_base64,
        metadata=metadata or {},
        success=success,
        error_message=error_message,
    )

    # 4. Add to session
    self.session.add_processor_state(state)
    self._execution_order += 1
```

**Image Encoding Flow**:
1. Resize image if width > max_image_width (maintains aspect ratio)
2. Encode to JPEG with specified quality
3. Convert to base64 string for JSON serialization
4. Store in ProcessorState

---

### Step 4: Handle Errors

```python
try:
    tracker.start_processor(processor_name)
    context = processor.process(context)
    tracker.capture_state(
        processor_name,
        context,
        metadata={"stage": "processing"},
        success=True,
    )
except Exception as e:
    logger.error(f"Error in processor {processor_name}: {e}")
    tracker.capture_state(
        processor_name,
        context,
        metadata={"stage": "processing"},
        success=False,
        error_message=str(e),
    )
    raise
```

**Error Capture**:
- Processor failures are captured with `success=False`
- Error message is stored
- Context state at time of failure is preserved
- Exception is re-raised for pipeline handling

---

### Step 5: Finalize Session

```python
def finalize(self) -> WorkflowSession:
    """Finalize the tracking session and return the complete session data."""

    # Calculate total duration
    end_time = datetime.now(UTC).isoformat()
    start_dt = datetime.fromisoformat(self.session.start_time)
    end_dt = datetime.fromisoformat(end_time)
    total_duration_ms = (end_dt - start_dt).total_seconds() * 1000

    # Finalize session
    self.session.finalize(end_time, total_duration_ms)

    logger.info(
        f"[Tracker] Finalized session: {self.session.session_id} "
        f"(duration={total_duration_ms:.2f}ms, "
        f"processors={len(self.session.processor_states)})"
    )

    return self.session
```

**Finalization**:
1. Record end time
2. Calculate total duration
3. Update session with final metadata
4. Return complete WorkflowSession for export

---

## High-Level API: track_workflow()

Convenience function that runs the entire pipeline with tracking:

```python
def track_workflow(
    file_path: Path | str,
    template_path: Path | str,
    config_path: Path | str | None = None,
    capture_processors: list[str] | None = None,
    max_image_width: int = 800,
    include_colored: bool = True,
    image_quality: int = 85,
) -> WorkflowSession:
    """High-level function to track a complete workflow execution."""
```

### Complete Flow

```
1. Load Configuration
   ↓
2. Load Template
   ↓
3. Load Input Image
   ↓
4. Initialize WorkflowTracker
   ↓
5. Create ProcessingPipeline
   ↓
6. Build Workflow Graph
   ↓
7. Capture Initial State
   ↓
8. Execute Pipeline with Tracking
   - For each processor:
     - start_processor()
     - processor.process()
     - capture_state()
   ↓
9. Finalize Session
   ↓
10. Return WorkflowSession
```

---

## Selective Capture

### Capture All Processors (Default)

```python
tracker = WorkflowTracker(
    file_path="input.jpg",
    template_name="Sample",
    capture_processors=None  # or ["all"]
)
```

### Capture Specific Processors

```python
tracker = WorkflowTracker(
    file_path="input.jpg",
    template_name="Sample",
    capture_processors=["AutoRotate", "CropOnMarkers", "ReadOMR"]
)
```

**Filter Logic**:
```python
def should_capture(self, processor_name: str) -> bool:
    """Check if a processor should be captured."""
    if not self.capture_processors or "all" in self.capture_processors:
        return True
    return processor_name in self.capture_processors
```

**Use Cases**:
- **Debug specific stage**: Only capture problematic processors
- **Performance**: Reduce overhead by skipping unneeded captures
- **Storage**: Minimize session file size

---

## Data Captured Per Processor

### ProcessorState Structure

```python
@dataclass
class ProcessorState:
    name: str                              # "AutoRotate"
    order: int                             # 0, 1, 2, ...
    timestamp: str                         # "2024-02-20T10:30:45.123Z"
    duration_ms: float                     # 123.45
    image_shape: tuple[int, ...]           # (1200, 1600, 3)
    gray_image_base64: str | None          # Base64 JPEG
    colored_image_base64: str | None       # Base64 JPEG (optional)
    metadata: dict[str, Any]               # Processor-specific data
    success: bool                          # True/False
    error_message: str | None              # Error details if failed
```

### Example Metadata

**AutoRotate**:
```python
metadata = {
    "stage": "processing",
    "rotation_angle": 0,
    "match_score": 0.98,
}
```

**CropOnMarkers**:
```python
metadata = {
    "stage": "processing",
    "marker_type": "FOUR_DOTS",
    "markers_detected": 4,
    "corners": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
}
```

**ReadOMR**:
```python
metadata = {
    "stage": "processing",
    "fields_processed": 50,
    "bubbles_detected": 200,
    "detection_method": "BUBBLES_THRESHOLD",
}
```

---

## Complete Example

```python
from src.processors.visualization.workflow_tracker import track_workflow

# Track a complete workflow
session = track_workflow(
    file_path="inputs/sample1/sample1.jpg",
    template_path="inputs/sample1/template.json",
    config_path="inputs/sample1/config.json",
    capture_processors=None,  # Capture all
    max_image_width=800,
    include_colored=True,
    image_quality=85,
)

# Session now contains:
print(f"Session ID: {session.session_id}")
print(f"Total Duration: {session.total_duration_ms}ms")
print(f"Processors Captured: {len(session.processor_states)}")

# Export to JSON
session.save_to_file("outputs/workflow_session.json")

# Access individual states
for state in session.processor_states:
    print(f"{state.name}: {state.duration_ms:.2f}ms - Success: {state.success}")

# Access graph structure
print(f"Graph Nodes: {len(session.graph.nodes)}")
print(f"Graph Edges: {len(session.graph.edges)}")
```

---

## Integration with Pipeline

### Manual Integration

```python
from src.processors.pipeline import ProcessingPipeline
from src.processors.visualization.workflow_tracker import WorkflowTracker

# Initialize tracker
tracker = WorkflowTracker(
    file_path="input.jpg",
    template_name="Sample"
)

# Create pipeline
pipeline = ProcessingPipeline(template)

# Build graph
processor_names = pipeline.get_processor_names()
tracker.build_graph(processor_names)

# Track initial state
tracker.capture_state("Input", initial_context)

# Execute with tracking
for processor in pipeline.processors:
    name = processor.get_name()

    tracker.start_processor(name)
    context = processor.process(context)
    tracker.capture_state(name, context, success=True)

# Finalize
session = tracker.finalize()
```

---

## Browser Migration

### TypeScript Implementation

```typescript
interface WorkflowTracker {
  session: WorkflowSession;
  captureProcessors: string[];
  maxImageWidth: number;
  includeColored: boolean;
  imageQuality: number;

  startProcessor(name: string): void;
  captureState(
    name: string,
    context: ProcessingContext,
    metadata?: Record<string, any>,
    success?: boolean,
    errorMessage?: string
  ): void;
  buildGraph(processorNames: string[]): void;
  finalize(): WorkflowSession;
}

class BrowserWorkflowTracker implements WorkflowTracker {
  private startTimes: Map<string, number> = new Map();
  private executionOrder: number = 0;

  constructor(
    filePath: string,
    templateName: string,
    options: {
      captureProcessors?: string[];
      maxImageWidth?: number;
      includeColored?: boolean;
      imageQuality?: number;
      config?: Record<string, any>;
    } = {}
  ) {
    const sessionId = `session_${new Date().toISOString().replace(/[:.]/g, '_')}_${crypto.randomUUID().slice(0, 8)}`;

    this.session = {
      sessionId,
      filePath,
      templateName,
      startTime: new Date().toISOString(),
      processorStates: [],
      graph: { nodes: [], edges: [] },
      config: options.config || {},
      metadata: {},
    };

    this.captureProcessors = options.captureProcessors || ['all'];
    this.maxImageWidth = options.maxImageWidth || 800;
    this.includeColored = options.includeColored ?? true;
    this.imageQuality = options.imageQuality || 85;
  }

  startProcessor(name: string): void {
    if (this.shouldCapture(name)) {
      this.startTimes.set(name, performance.now());
    }
  }

  async captureState(
    name: string,
    context: ProcessingContext,
    metadata: Record<string, any> = {},
    success: boolean = true,
    errorMessage?: string
  ): Promise<void> {
    if (!this.shouldCapture(name)) return;

    // Calculate duration
    const startTime = this.startTimes.get(name);
    const durationMs = startTime ? performance.now() - startTime : 0;

    // Encode images
    const grayImageBase64 = context.grayImage
      ? await ImageEncoder.encodeImage(context.grayImage, this.maxImageWidth, this.imageQuality)
      : null;

    const coloredImageBase64 = this.includeColored && context.coloredImage
      ? await ImageEncoder.encodeImage(context.coloredImage, this.maxImageWidth, this.imageQuality)
      : null;

    // Create state
    const state: ProcessorState = {
      name,
      order: this.executionOrder++,
      timestamp: new Date().toISOString(),
      durationMs,
      imageShape: context.grayImage ? [context.grayImage.rows, context.grayImage.cols] : [0, 0],
      grayImageBase64,
      coloredImageBase64,
      metadata,
      success,
      errorMessage,
    };

    this.session.processorStates.push(state);
  }

  buildGraph(processorNames: string[]): void {
    // Add input node
    this.session.graph.nodes.push({
      id: 'input',
      label: 'Input Image',
      metadata: { type: 'input', filePath: this.session.filePath },
    });

    // Add processor nodes
    processorNames.forEach((name, i) => {
      this.session.graph.nodes.push({
        id: `processor_${i}`,
        label: name,
        metadata: { type: 'processor', order: i },
      });
    });

    // Add output node
    this.session.graph.nodes.push({
      id: 'output',
      label: 'Output',
      metadata: { type: 'output' },
    });

    // Add edges
    if (processorNames.length > 0) {
      this.session.graph.edges.push({ from: 'input', to: 'processor_0' });

      for (let i = 0; i < processorNames.length - 1; i++) {
        this.session.graph.edges.push({
          from: `processor_${i}`,
          to: `processor_${i + 1}`,
        });
      }

      this.session.graph.edges.push({
        from: `processor_${processorNames.length - 1}`,
        to: 'output',
      });
    } else {
      this.session.graph.edges.push({ from: 'input', to: 'output' });
    }
  }

  finalize(): WorkflowSession {
    const endTime = new Date().toISOString();
    const startDt = new Date(this.session.startTime);
    const endDt = new Date(endTime);
    const totalDurationMs = endDt.getTime() - startDt.getTime();

    this.session.endTime = endTime;
    this.session.totalDurationMs = totalDurationMs;

    return this.session;
  }

  private shouldCapture(name: string): boolean {
    return !this.captureProcessors.length ||
           this.captureProcessors.includes('all') ||
           this.captureProcessors.includes(name);
  }
}
```

### Image Encoding in Browser

```typescript
class ImageEncoder {
  static async encodeImage(
    mat: cv.Mat,
    maxWidth: number = 800,
    quality: number = 85
  ): Promise<string | null> {
    if (!mat || mat.empty()) return null;

    let resized = mat;

    // Resize if needed
    if (mat.cols > maxWidth) {
      const scale = maxWidth / mat.cols;
      const newSize = new cv.Size(maxWidth, Math.floor(mat.rows * scale));
      resized = new cv.Mat();
      cv.resize(mat, resized, newSize, 0, 0, cv.INTER_LINEAR);
    }

    // Encode to JPEG
    const encoded = cv.imencode('.jpg', resized, [cv.IMWRITE_JPEG_QUALITY, quality]);

    // Convert to base64
    const base64 = btoa(String.fromCharCode(...encoded.data));

    // Cleanup
    if (resized !== mat) resized.delete();
    encoded.delete();

    return base64;
  }

  static decodeImage(base64: string): cv.Mat {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return cv.imdecode(cv.matFromArray(bytes.length, 1, cv.CV_8U, bytes), cv.IMREAD_UNCHANGED);
  }
}
```

### Usage in Browser

```typescript
import { BrowserWorkflowTracker } from './workflow-tracker';
import { ProcessingPipeline } from './pipeline';

async function trackWorkflow(
  imageFile: File,
  template: Template,
  config: Config
): Promise<WorkflowSession> {
  // Initialize tracker
  const tracker = new BrowserWorkflowTracker(
    imageFile.name,
    template.name,
    {
      captureProcessors: ['all'],
      maxImageWidth: 800,
      includeColored: true,
      imageQuality: 85,
    }
  );

  // Create pipeline
  const pipeline = new ProcessingPipeline(template);

  // Build graph
  const processorNames = pipeline.getProcessorNames();
  tracker.buildGraph(processorNames);

  // Load initial image
  const context = await loadInitialContext(imageFile, template);
  await tracker.captureState('Input', context);

  // Execute pipeline with tracking
  for (const processor of pipeline.processors) {
    const name = processor.getName();

    try {
      tracker.startProcessor(name);
      await processor.process(context);
      await tracker.captureState(name, context, {}, true);
    } catch (error) {
      await tracker.captureState(name, context, {}, false, error.message);
      throw error;
    }
  }

  // Finalize and return
  return tracker.finalize();
}

// Save session to IndexedDB or download as JSON
async function saveSession(session: WorkflowSession) {
  const json = JSON.stringify(session, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = `${session.sessionId}.json`;
  a.click();

  URL.revokeObjectURL(url);
}
```

---

## Key Points

1. **Comprehensive Tracking**: Captures processor execution, timing, images, metadata, and errors
2. **Selective Capture**: Filter processors to reduce overhead and storage
3. **Image Optimization**: Resize and compress images for efficient storage
4. **Graph Structure**: Visual representation of processor flow
5. **Error Handling**: Captures failures with context for debugging
6. **Session Management**: Complete workflow state in serializable format
7. **Browser Support**: Full TypeScript implementation with async image encoding
8. **Performance Monitoring**: Track duration of each processor for optimization
9. **Replay Capability**: Session data enables step-by-step visualization
10. **Export Options**: Save to JSON file or IndexedDB storage
