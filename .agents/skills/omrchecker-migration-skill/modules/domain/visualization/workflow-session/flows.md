# Workflow Session Flows

**Module**: Domain - Visualization - Workflow Session
**Python Reference**: `src/processors/visualization/workflow_session.py`
**Last Updated**: 2026-02-21

---

## Overview

Workflow Session is the core data model for capturing and replaying entire OMR processing workflows. It encapsulates all state needed for visualization, debugging, and analysis including processor execution states, timing information, images at each stage, and workflow graph structure.

**Use Case**: Enable step-by-step visualization, debugging, and replay of OMR processing pipelines.

---

## Core Data Models

### ProcessorState

Represents the state of a single processor at a specific execution point.

```
ProcessorState Entity
│
├─► name: str
│   Human-readable processor name (e.g., "AutoRotate", "CropOnMarkers")
│
├─► order: int
│   0-indexed execution order in pipeline
│
├─► timestamp: str
│   ISO format timestamp when processor executed
│   Format: "2024-01-06T12:34:56.789Z"
│
├─► duration_ms: float
│   Execution time in milliseconds
│
├─► image_shape: tuple[int, ...]
│   Shape of output image (height, width, channels)
│   Example: (1200, 800, 3) or (1200, 800) for grayscale
│
├─► gray_image_base64: str | None
│   Base64-encoded JPEG of grayscale output
│   Data URI ready for HTML embedding
│
├─► colored_image_base64: str | None
│   Base64-encoded JPEG of colored output (optional)
│   None if colored output not captured
│
├─► metadata: dict[str, Any]
│   Processor-specific metadata
│   Examples:
│   - {"stage": "preprocessing", "rotation_angle": 90}
│   - {"stage": "detection", "bubbles_detected": 40}
│
├─► success: bool
│   Whether processor executed successfully
│   True = success, False = failure
│
└─► error_message: str | None
    Error message if processor failed
    None if success=True
```

### WorkflowGraph

Represents the processor workflow as a directed graph.

```
WorkflowGraph Entity
│
├─► nodes: list[dict[str, Any]]
│   List of node definitions
│   Each node: {
│       "id": str,           # Unique node ID (e.g., "processor_0")
│       "label": str,        # Display label (e.g., "AutoRotate")
│       "metadata": dict     # Additional node data
│   }
│
└─► edges: list[dict[str, Any]]
    List of edge definitions connecting nodes
    Each edge: {
        "from": str,        # Source node ID
        "to": str,          # Target node ID
        "label": str | None # Optional edge label
    }
```

### WorkflowSession

Complete workflow execution session data.

```
WorkflowSession Entity
│
├─► session_id: str
│   Unique identifier for session
│   Format: "session_YYYYMMDD_HHMMSS_<8-char-uuid>"
│   Example: "session_20240106_123456_abcd1234"
│
├─► file_path: str
│   Path to input file being processed
│
├─► template_name: str
│   Name of template used
│
├─► start_time: str
│   ISO format timestamp when session started
│
├─► end_time: str | None
│   ISO format timestamp when session ended
│   None until finalized
│
├─► total_duration_ms: float | None
│   Total execution time in milliseconds
│   None until finalized
│
├─► processor_states: list[ProcessorState]
│   List of processor states in execution order
│
├─► graph: WorkflowGraph
│   Workflow graph structure
│
├─► config: dict[str, Any]
│   Configuration used for this session
│   Contains: template_path, config_path, etc.
│
└─► metadata: dict[str, Any]
    Additional session metadata
```

---

## Session Lifecycle Flow

### 1. Session Creation

```
START: Create WorkflowSession
│
├─► STEP 1: Generate Session ID
│   │
│   session_id = f"session_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
│   │
│   │ Example: "session_20240106_123456_abcd1234"
│   │
│   │ Components:
│   │ - "session_" prefix
│   │ - Date: YYYYMMDD (20240106)
│   │ - Time: HHMMSS (123456)
│   │ - UUID: 8 hex chars (abcd1234)
│
├─► STEP 2: Capture Start Time
│   │
│   start_time = datetime.now(UTC).isoformat()
│   │
│   │ Example: "2024-01-06T12:34:56.789012+00:00"
│   │ Format: ISO 8601 with UTC timezone
│
├─► STEP 3: Initialize Session
│   │
│   session = WorkflowSession(
│       session_id=session_id,
│       file_path=str(file_path),
│       template_name=template_name,
│       start_time=start_time,
│       end_time=None,              # Not finalized yet
│       total_duration_ms=None,     # Not finalized yet
│       processor_states=[],        # Empty initially
│       graph=WorkflowGraph(),      # Empty graph
│       config=config or {},
│       metadata={}
│   )
│
└─► RETURN session
    Session ready for tracking

END
```

### 2. State Capture Flow

```
START: Capture Processor State
│
├─► INPUT: processor_name, context, metadata, success, error_message
│
├─► STEP 1: Calculate Duration
│   │
│   │ If start_time was tracked:
│   duration_ms = (time.time() - start_time) * 1000
│   │
│   │ Else:
│   duration_ms = 0.0
│
├─► STEP 2: Encode Images
│   │
│   ├─► Encode Grayscale Image
│   │   │
│   │   If context.gray_image is not None:
│   │   │
│   │   ├─► Resize if needed (max_width)
│   │   │   │
│   │   │   h, w = image.shape[:2]
│   │   │   if w > max_width:
│   │   │       scale = max_width / w
│   │   │       new_width = max_width
│   │   │       new_height = int(h * scale)
│   │   │       image = cv2.resize(image, (new_width, new_height))
│   │   │
│   │   ├─► Encode to JPEG
│   │   │   │
│   │   │   encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
│   │   │   _, buffer = cv2.imencode('.jpg', image, encode_param)
│   │   │
│   │   └─► Convert to Base64
│   │       │
│   │       gray_image_base64 = base64.b64encode(buffer).decode('utf-8')
│   │       image_shape = context.gray_image.shape
│   │
│   └─► Encode Colored Image (if include_colored=True)
│       │
│       Same process as grayscale
│       colored_image_base64 = base64.b64encode(buffer).decode('utf-8')
│
├─► STEP 3: Create ProcessorState
│   │
│   state = ProcessorState(
│       name=processor_name,
│       order=execution_order,
│       timestamp=datetime.now(UTC).isoformat(),
│       duration_ms=duration_ms,
│       image_shape=image_shape,
│       gray_image_base64=gray_image_base64,
│       colored_image_base64=colored_image_base64,
│       metadata=metadata or {},
│       success=success,
│       error_message=error_message
│   )
│
├─► STEP 4: Add to Session
│   │
│   session.processor_states.append(state)
│   execution_order += 1
│
└─► END
    State captured and stored
```

### 3. Graph Building Flow

```
START: Build Workflow Graph
│
├─► INPUT: processor_names (list of processor names in order)
│
├─► STEP 1: Add Input Node
│   │
│   graph.add_node(
│       node_id="input",
│       label="Input Image",
│       metadata={
│           "type": "input",
│           "file_path": session.file_path
│       }
│   )
│
├─► STEP 2: Add Processor Nodes
│   │
│   For i, name in enumerate(processor_names):
│   │
│   │   node_id = f"processor_{i}"
│   │   │
│   │   graph.add_node(
│   │       node_id=node_id,
│   │       label=name,
│   │       metadata={
│   │           "type": "processor",
│   │           "order": i
│   │       }
│   │   )
│   │
│   │ Example nodes:
│   │ - processor_0: "AutoRotate"
│   │ - processor_1: "CropOnMarkers"
│   │ - processor_2: "ReadOMR"
│
├─► STEP 3: Add Output Node
│   │
│   graph.add_node(
│       node_id="output",
│       label="Output",
│       metadata={"type": "output"}
│   )
│
├─► STEP 4: Add Edges
│   │
│   If len(processor_names) > 0:
│   │
│   ├─► Input to First Processor
│   │   │
│   │   graph.add_edge("input", "processor_0")
│   │
│   ├─► Between Processors
│   │   │
│   │   For i in range(len(processor_names) - 1):
│   │       graph.add_edge(f"processor_{i}", f"processor_{i+1}")
│   │   │
│   │   │ Creates linear chain:
│   │   │ processor_0 → processor_1 → processor_2 → ...
│   │
│   └─► Last Processor to Output
│       │
│       graph.add_edge(f"processor_{len(processor_names)-1}", "output")
│   │
│   Else (no processors):
│       │
│       graph.add_edge("input", "output")
│       │ Direct connection if pipeline is empty
│
└─► END
    Graph structure complete

Example Graph:
┌─────────────┐
│ Input Image │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ AutoRotate  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│CropOnMarkers│
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   ReadOMR   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Output    │
└─────────────┘
```

### 4. Session Finalization Flow

```
START: Finalize Session
│
├─► STEP 1: Capture End Time
│   │
│   end_time = datetime.now(UTC).isoformat()
│
├─► STEP 2: Calculate Total Duration
│   │
│   start_dt = datetime.fromisoformat(session.start_time)
│   end_dt = datetime.fromisoformat(end_time)
│   │
│   total_duration_ms = (end_dt - start_dt).total_seconds() * 1000
│   │
│   │ Example:
│   │ start: 2024-01-06T12:34:56.000
│   │ end:   2024-01-06T12:35:01.250
│   │ duration: 5.25 seconds = 5250 ms
│
├─► STEP 3: Update Session
│   │
│   session.end_time = end_time
│   session.total_duration_ms = total_duration_ms
│
├─► STEP 4: Log Summary
│   │
│   logger.info(
│       f"Finalized session: {session.session_id} "
│       f"(duration={total_duration_ms:.2f}ms, "
│       f"processors={len(session.processor_states)})"
│   )
│
└─► RETURN session
    Session complete and ready for export

END
```

---

## Serialization Flows

### 1. Session to Dictionary

```
START: session.to_dict()
│
├─► STEP 1: Convert Processor States
│   │
│   processor_states_dict = [state.to_dict() for state in processor_states]
│   │
│   │ Each ProcessorState converted using asdict()
│
├─► STEP 2: Convert Graph
│   │
│   graph_dict = {
│       "nodes": graph.nodes,  # Already list of dicts
│       "edges": graph.edges   # Already list of dicts
│   }
│
├─► STEP 3: Build Complete Dictionary
│   │
│   return {
│       "session_id": session_id,
│       "file_path": file_path,
│       "template_name": template_name,
│       "start_time": start_time,
│       "end_time": end_time,
│       "total_duration_ms": total_duration_ms,
│       "processor_states": processor_states_dict,
│       "graph": graph_dict,
│       "config": config,
│       "metadata": metadata
│   }
│
└─► END
    Dictionary ready for JSON serialization
```

### 2. Session to JSON

```
START: session.to_json(indent=2)
│
├─► STEP 1: Convert to Dictionary
│   │
│   data = session.to_dict()
│
├─► STEP 2: Serialize to JSON
│   │
│   json_str = json.dumps(data, indent=2)
│   │
│   │ Pretty-printed with 2-space indentation
│
└─► RETURN json_str
    JSON string ready for file or network

END
```

### 3. Save to File

```
START: session.save_to_file(file_path)
│
├─► STEP 1: Convert to JSON
│   │
│   json_str = session.to_json()
│
├─► STEP 2: Ensure Directory Exists
│   │
│   file_path = Path(file_path)
│   file_path.parent.mkdir(parents=True, exist_ok=True)
│
├─► STEP 3: Write to File
│   │
│   file_path.write_text(json_str)
│
└─► END
    Session saved to disk

Example file structure:
outputs/visualization/sessions/session_20240106_123456_abcd1234.json
```

### 4. Load from File

```
START: WorkflowSession.load_from_file(file_path)
│
├─► STEP 1: Read File
│   │
│   file_path = Path(file_path)
│   json_str = file_path.read_text()
│
├─► STEP 2: Parse JSON
│   │
│   data = json.loads(json_str)
│
├─► STEP 3: Reconstruct Processor States
│   │
│   processor_states = [
│       ProcessorState(**state_data)
│       for state_data in data.get("processor_states", [])
│   ]
│
├─► STEP 4: Reconstruct Graph
│   │
│   graph_data = data.get("graph", {})
│   graph = WorkflowGraph(
│       nodes=graph_data.get("nodes", []),
│       edges=graph_data.get("edges", [])
│   )
│
├─► STEP 5: Create WorkflowSession
│   │
│   session = WorkflowSession(
│       session_id=data["session_id"],
│       file_path=data["file_path"],
│       template_name=data["template_name"],
│       start_time=data["start_time"],
│       end_time=data.get("end_time"),
│       total_duration_ms=data.get("total_duration_ms"),
│       processor_states=processor_states,
│       graph=graph,
│       config=data.get("config", {}),
│       metadata=data.get("metadata", {})
│   )
│
└─► RETURN session
    Session reconstructed from file

END
```

---

## Image Encoding Flow

### ImageEncoder.encode_image()

```
START: encode_image(image, max_width=800, quality=85)
│
├─► INPUT VALIDATION
│   │
│   If image is None:
│       RETURN None
│
├─► STEP 1: Resize Image (if needed)
│   │
│   If max_width is not None:
│   │
│   ├─► Get dimensions
│   │   h, w = image.shape[:2]
│   │
│   ├─► Check if resize needed
│   │   │
│   │   If w > max_width:
│   │   │
│   │   │   ├─► Calculate scale
│   │   │   │   scale = max_width / w
│   │   │   │
│   │   │   ├─► Calculate new dimensions
│   │   │   │   new_width = max_width
│   │   │   │   new_height = int(h * scale)
│   │   │   │
│   │   │   └─► Resize
│   │   │       image = cv2.resize(image, (new_width, new_height))
│   │   │
│   │   │   Example:
│   │   │   Original: 1600 × 1200
│   │   │   max_width: 800
│   │   │   scale: 800 / 1600 = 0.5
│   │   │   new_height: 1200 * 0.5 = 600
│   │   │   Resized: 800 × 600
│
├─► STEP 2: Encode to JPEG
│   │
│   encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
│   _, buffer = cv2.imencode('.jpg', image, encode_param)
│   │
│   │ cv2.imencode returns (success, buffer)
│   │ buffer is numpy array of bytes
│
├─► STEP 3: Convert to Base64
│   │
│   base64_str = base64.b64encode(buffer).decode('utf-8')
│   │
│   │ Example output:
│   │ "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIB..."
│   │ (long base64 string)
│
└─► RETURN base64_str
    Base64-encoded JPEG ready for embedding

END
```

### ImageEncoder.decode_image()

```
START: decode_image(base64_str)
│
├─► STEP 1: Decode Base64
│   │
│   img_data = base64.b64decode(base64_str)
│   │
│   │ Converts base64 string back to bytes
│
├─► STEP 2: Convert to NumPy Array
│   │
│   nparr = np.frombuffer(img_data, np.uint8)
│   │
│   │ Creates 1D array of uint8 values
│
├─► STEP 3: Decode JPEG
│   │
│   image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
│   │
│   │ cv2.imdecode reconstructs image from buffer
│   │ IMREAD_UNCHANGED preserves original format
│
└─► RETURN image
    Image as NumPy array (MatLike)

END
```

### ImageEncoder.get_data_uri()

```
START: get_data_uri(base64_str, mime_type="image/jpeg")
│
├─► Build Data URI
│   │
│   data_uri = f"data:{mime_type};base64,{base64_str}"
│   │
│   │ Example:
│   │ "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
│   │
│   │ Format: data:[<mime type>][;base64],<data>
│
└─► RETURN data_uri
    Data URI ready for HTML <img src="...">

END
```

---

## Complete Workflow Example

### End-to-End Session Tracking

```
START: Track Complete Workflow
│
├─► STEP 1: Initialize WorkflowTracker
│   │
│   tracker = WorkflowTracker(
│       file_path="inputs/sample1.jpg",
│       template_name="sample1",
│       capture_processors=["all"],  # Capture all processors
│       max_image_width=800,
│       include_colored=True,
│       image_quality=85,
│       config={"template_path": "...", "config_path": "..."}
│   )
│   │
│   │ Creates WorkflowSession with unique session_id
│
├─► STEP 2: Build Graph Structure
│   │
│   processor_names = ["AutoRotate", "CropOnMarkers", "ReadOMR"]
│   tracker.build_graph(processor_names)
│   │
│   │ Creates nodes: input → processor_0 → processor_1 → processor_2 → output
│
├─► STEP 3: Capture Initial State
│   │
│   tracker.capture_state("Input", initial_context, metadata={"stage": "initial"})
│   │
│   │ Captures input image before any processing
│
├─► STEP 4: Track Each Processor
│   │
│   For each processor in pipeline:
│   │
│   ├─► Mark Start
│   │   tracker.start_processor(processor_name)
│   │   # Records start time
│   │
│   ├─► Execute Processor
│   │   context = processor.process(context)
│   │
│   ├─► Capture State
│   │   tracker.capture_state(
│   │       processor_name,
│   │       context,
│   │       metadata={"stage": "processing"},
│   │       success=True
│   │   )
│   │   │
│   │   │ Captures:
│   │   │ - Output images (gray + colored)
│   │   │ - Execution time
│   │   │ - Image shape
│   │   │ - Metadata
│   │
│   └─► Handle Errors
│       │
│       If processor fails:
│           tracker.capture_state(
│               processor_name,
│               context,
│               metadata={"stage": "processing"},
│               success=False,
│               error_message=str(error)
│           )
│
├─► STEP 5: Finalize Session
│   │
│   session = tracker.finalize()
│   │
│   │ Sets end_time and total_duration_ms
│   │ Returns complete WorkflowSession
│
├─► STEP 6: Export to Files
│   │
│   ├─► Save JSON
│   │   session.save_to_file("outputs/sessions/session_xyz.json")
│   │
│   └─► Export HTML
│       exporter = HTMLExporter()
│       exporter.export(session, "outputs/session_xyz.html")
│
└─► END
    Complete workflow tracked and exported

Final Output:
- outputs/sessions/session_20240106_123456_abcd1234.json
- outputs/session_20240106_123456_abcd1234.html
```

---

## Browser Migration

### TypeScript Interfaces

```typescript
// ProcessorState interface
interface ProcessorState {
  name: string;
  order: number;
  timestamp: string;  // ISO 8601
  duration_ms: number;
  image_shape: number[];
  gray_image_base64: string | null;
  colored_image_base64: string | null;
  metadata: Record<string, any>;
  success: boolean;
  error_message: string | null;
}

// WorkflowGraph interface
interface WorkflowGraph {
  nodes: Array<{
    id: string;
    label: string;
    metadata: Record<string, any>;
  }>;
  edges: Array<{
    from: string;
    to: string;
    label?: string;
  }>;
}

// WorkflowSession interface
interface WorkflowSession {
  session_id: string;
  file_path: string;
  template_name: string;
  start_time: string;
  end_time: string | null;
  total_duration_ms: number | null;
  processor_states: ProcessorState[];
  graph: WorkflowGraph;
  config: Record<string, any>;
  metadata: Record<string, any>;
}
```

### Browser Storage Strategies

```javascript
// 1. In-Memory Storage (during active session)
class WorkflowSessionManager {
  private currentSession: WorkflowSession | null = null;

  createSession(filePath, templateName, config) {
    const sessionId = `session_${this.generateSessionId()}`;
    const startTime = new Date().toISOString();

    this.currentSession = {
      session_id: sessionId,
      file_path: filePath,
      template_name: templateName,
      start_time: startTime,
      end_time: null,
      total_duration_ms: null,
      processor_states: [],
      graph: { nodes: [], edges: [] },
      config: config,
      metadata: {}
    };

    return this.currentSession;
  }

  addProcessorState(state: ProcessorState) {
    if (this.currentSession) {
      this.currentSession.processor_states.push(state);
    }
  }

  finalize() {
    if (this.currentSession) {
      const endTime = new Date().toISOString();
      const startDt = new Date(this.currentSession.start_time);
      const endDt = new Date(endTime);

      this.currentSession.end_time = endTime;
      this.currentSession.total_duration_ms = endDt.getTime() - startDt.getTime();
    }
    return this.currentSession;
  }

  private generateSessionId(): string {
    const now = new Date();
    const dateStr = now.toISOString().slice(0, 19).replace(/[-:]/g, '').replace('T', '_');
    const uuid = this.generateUUID().slice(0, 8);
    return `${dateStr}_${uuid}`;
  }

  private generateUUID(): string {
    return crypto.randomUUID();
  }
}

// 2. IndexedDB Storage (persistent)
class SessionDatabase {
  private db: IDBDatabase | null = null;

  async init() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('OMRCheckerDB', 1);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve(this.db);
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;

        // Create sessions store
        if (!db.objectStoreNames.contains('sessions')) {
          const store = db.createObjectStore('sessions', { keyPath: 'session_id' });
          store.createIndex('timestamp', 'start_time', { unique: false });
        }
      };
    });
  }

  async saveSession(session: WorkflowSession) {
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['sessions'], 'readwrite');
      const store = transaction.objectStore('sessions');
      const request = store.put(session);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async loadSession(sessionId: string): Promise<WorkflowSession> {
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['sessions'], 'readonly');
      const store = transaction.objectStore('sessions');
      const request = store.get(sessionId);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async listSessions(limit = 10): Promise<WorkflowSession[]> {
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['sessions'], 'readonly');
      const store = transaction.objectStore('sessions');
      const index = store.index('timestamp');
      const request = index.openCursor(null, 'prev');  // Newest first

      const sessions: WorkflowSession[] = [];

      request.onsuccess = (event) => {
        const cursor = (event.target as IDBRequest).result;
        if (cursor && sessions.length < limit) {
          sessions.push(cursor.value);
          cursor.continue();
        } else {
          resolve(sessions);
        }
      };

      request.onerror = () => reject(request.error);
    });
  }

  async deleteSession(sessionId: string) {
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['sessions'], 'readwrite');
      const store = transaction.objectStore('sessions');
      const request = store.delete(sessionId);

      request.onsuccess = () => resolve(undefined);
      request.onerror = () => reject(request.error);
    });
  }
}

// 3. Download as JSON (export)
function downloadSession(session: WorkflowSession) {
  const json = JSON.stringify(session, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = `${session.session_id}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// 4. Load from File (import)
async function loadSessionFromFile(file: File): Promise<WorkflowSession> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = (e) => {
      try {
        const json = e.target!.result as string;
        const session = JSON.parse(json) as WorkflowSession;
        resolve(session);
      } catch (error) {
        reject(error);
      }
    };

    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}
```

### Image Encoding in Browser

```javascript
class ImageEncoder {
  // Encode Canvas to Base64
  static async encodeCanvas(
    canvas: HTMLCanvasElement,
    maxWidth: number = 800,
    quality: number = 0.85
  ): Promise<string | null> {
    if (!canvas) return null;

    // Resize if needed
    if (canvas.width > maxWidth) {
      const scale = maxWidth / canvas.width;
      const newWidth = maxWidth;
      const newHeight = Math.floor(canvas.height * scale);

      const resizedCanvas = document.createElement('canvas');
      resizedCanvas.width = newWidth;
      resizedCanvas.height = newHeight;

      const ctx = resizedCanvas.getContext('2d')!;
      ctx.drawImage(canvas, 0, 0, newWidth, newHeight);

      canvas = resizedCanvas;
    }

    // Convert to JPEG base64
    const dataUrl = canvas.toDataURL('image/jpeg', quality);

    // Extract base64 portion (remove "data:image/jpeg;base64," prefix)
    const base64 = dataUrl.split(',')[1];

    return base64;
  }

  // Encode cv.Mat to Base64 (OpenCV.js)
  static async encodeMat(
    mat: any,  // cv.Mat
    maxWidth: number = 800,
    quality: number = 0.85
  ): Promise<string | null> {
    if (!mat || mat.empty()) return null;

    // Resize if needed
    let processedMat = mat;
    if (mat.cols > maxWidth) {
      const scale = maxWidth / mat.cols;
      const newWidth = maxWidth;
      const newHeight = Math.floor(mat.rows * scale);

      processedMat = new cv.Mat();
      const dsize = new cv.Size(newWidth, newHeight);
      cv.resize(mat, processedMat, dsize, 0, 0, cv.INTER_LINEAR);
    }

    // Encode to JPEG
    const jpegMat = new cv.Mat();
    const params = new cv.IntVector();
    params.push_back(cv.IMWRITE_JPEG_QUALITY);
    params.push_back(Math.floor(quality * 100));

    cv.imencode('.jpg', processedMat, jpegMat, params);

    // Convert to base64
    const buffer = jpegMat.data;
    const base64 = btoa(String.fromCharCode(...buffer));

    // Cleanup
    if (processedMat !== mat) processedMat.delete();
    jpegMat.delete();
    params.delete();

    return base64;
  }

  // Decode Base64 to Canvas
  static async decodeToCanvas(base64: string): Promise<HTMLCanvasElement> {
    return new Promise((resolve, reject) => {
      const img = new Image();

      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;

        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(img, 0, 0);

        resolve(canvas);
      };

      img.onerror = () => reject(new Error('Failed to decode image'));
      img.src = `data:image/jpeg;base64,${base64}`;
    });
  }

  // Get Data URI
  static getDataURI(base64: string, mimeType: string = 'image/jpeg'): string {
    return `data:${mimeType};base64,${base64}`;
  }
}
```

---

## Related Documentation

- **Workflow Tracker**: `../workflow-tracker/flows.md`
- **HTML Exporter**: `../html-export/flows.md`
- **Processing Context**: `../../processing-context/concept.md`
- **Pipeline Flow**: `../../pipeline/concept.md`

---

## Summary

Workflow Session provides:

1. **Complete State Capture**: All processor states, images, timing, and metadata
2. **Graph Structure**: Visual representation of workflow as directed graph
3. **Serialization**: JSON export/import for persistence and sharing
4. **Image Encoding**: Base64-encoded JPEGs for embedding in HTML/JSON
5. **Browser Support**: IndexedDB storage, download/upload, in-memory management
6. **Replay Capability**: Reconstruct and visualize past executions

**Best For**: Debugging, visualization, analysis, sharing results
**Typical Size**: 1-10 MB per session (depending on image count and quality)
**Performance**: < 100ms overhead per processor capture
