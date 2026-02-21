# Workflow Session Constraints

**Module**: Domain - Visualization - Workflow Session
**Python Reference**: `src/processors/visualization/workflow_session.py`
**Last Updated**: 2026-02-21

---

## ProcessorState Constraints

### name

**Type**: `str`
**Purpose**: Human-readable processor name

**Valid Values**:
```python
# Preprocessing processors
"AutoRotate", "CropOnMarkers", "CropPage", "GaussianBlur", "Contrast", "Levels"

# Alignment processors
"Alignment", "SIFTAlignment", "PhaseCorrelation"

# Detection processors
"ReadOMR", "MLFieldBlockDetector", "ShiftDetection"

# Special states
"Input"  # Initial state before any processing
```

**Constraints**:
- Non-empty string
- No minimum/maximum length
- Should match actual processor class name
- Used for display in visualization

---

### order

**Type**: `int`
**Purpose**: 0-indexed execution order

**Valid Range**: 0 to unlimited

**Constraints**:
```python
# First state (Input)
order = 0

# Processors
order = 1, 2, 3, ..., N

# Must be sequential
assert states[i].order == i  # For all i
```

**Example**:
```python
[
  {"name": "Input", "order": 0},
  {"name": "AutoRotate", "order": 1},
  {"name": "CropOnMarkers", "order": 2},
  {"name": "ReadOMR", "order": 3}
]
```

---

### timestamp

**Type**: `str`
**Format**: ISO 8601 with timezone

**Valid Format**:
```python
# Full ISO format with UTC
"2024-01-06T12:34:56.789012+00:00"

# Components:
# YYYY-MM-DD: Date
# T: Separator
# HH:MM:SS.mmmmmm: Time with microseconds
# +00:00: UTC timezone
```

**Generation**:
```python
timestamp = datetime.now(UTC).isoformat()
```

**Constraints**:
- Must be valid ISO 8601 format
- Should include timezone (preferably UTC)
- Timestamps should be monotonically increasing (later states have later timestamps)
- Resolution: microseconds

**Parsing in Browser**:
```javascript
const date = new Date(timestamp);  // JavaScript Date object
const millis = date.getTime();     // Milliseconds since epoch
```

---

### duration_ms

**Type**: `float`
**Purpose**: Execution time in milliseconds

**Valid Range**: 0.0 to unlimited

**Typical Values**:
```
Fast processors (Contrast, Levels):     1-10 ms
Medium processors (AutoRotate, Blur):   10-100 ms
Slow processors (CropOnMarkers, SIFT):  100-500 ms
Very slow (ReadOMR with many bubbles):  500-2000 ms
```

**Constraints**:
```python
assert duration_ms >= 0.0  # Non-negative
assert duration_ms < 60000  # Sanity check: < 1 minute per processor

# Can be 0.0 if timing wasn't captured
if not tracked:
    duration_ms = 0.0
```

**Precision**: Milliseconds (0.001 seconds)

**Calculation**:
```python
start_time = time.time()
# ... execute processor ...
duration_ms = (time.time() - start_time) * 1000
```

---

### image_shape

**Type**: `tuple[int, ...]`
**Purpose**: Shape of output image

**Valid Formats**:
```python
# Grayscale image
(height, width)           # e.g., (1200, 800)

# Color image (BGR)
(height, width, channels) # e.g., (1200, 800, 3)

# Color image (BGRA)
(height, width, 4)        # e.g., (1200, 800, 4)

# Empty (no image)
(0, 0)                    # Special case
```

**Constraints**:
```python
# Tuple length: 2 or 3
assert len(image_shape) in [2, 3]

# Dimensions must be positive (or zero)
assert all(dim >= 0 for dim in image_shape)

# Channels (if present) must be 1, 3, or 4
if len(image_shape) == 3:
    assert image_shape[2] in [1, 3, 4]

# Typical OMR sheet dimensions
assert 100 <= image_shape[0] <= 10000  # Height: 100-10000 pixels
assert 100 <= image_shape[1] <= 10000  # Width: 100-10000 pixels
```

**Common Shapes**:
```python
# A4 at 150 DPI: ~1240 × 1754
(1754, 1240, 3)

# A4 at 200 DPI: ~1654 × 2339
(2339, 1654, 3)

# Resized for visualization (max_width=800)
(1068, 800, 3)  # Maintains aspect ratio
```

---

### gray_image_base64

**Type**: `str | None`
**Purpose**: Base64-encoded JPEG of grayscale output

**Format**:
```python
# Base64 string (no prefix)
"/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIB..."

# NOT a data URI (no "data:image/jpeg;base64," prefix)
# Prefix added by get_data_uri() when needed
```

**Constraints**:
```python
# Can be None if no image
gray_image_base64 = None  # Valid

# If present, must be non-empty
if gray_image_base64 is not None:
    assert len(gray_image_base64) > 0

# Valid base64 characters: A-Z, a-z, 0-9, +, /, =
import base64
try:
    base64.b64decode(gray_image_base64)
except Exception:
    raise ValueError("Invalid base64")
```

**Size Constraints**:
```python
# Typical sizes (after compression)
# Small (300×200, quality=85): ~5-10 KB base64 (~7-14 KB encoded)
# Medium (800×600, quality=85): ~30-50 KB base64 (~40-70 KB encoded)
# Large (1200×900, quality=85): ~60-100 KB base64 (~80-140 KB encoded)

# Rule: base64 is ~33% larger than raw bytes
base64_size ≈ jpeg_size * 1.33

# Recommendation: Keep under 200 KB per image
assert len(gray_image_base64) < 200000  # ~150 KB JPEG
```

**Encoding Parameters**:
```python
max_width = 800        # Resize to max 800px wide
quality = 85           # JPEG quality (0-100)
format = '.jpg'        # Always JPEG
```

---

### colored_image_base64

**Type**: `str | None`
**Purpose**: Base64-encoded JPEG of colored output

**Same Constraints as gray_image_base64**

**Optional Nature**:
```python
# Can be None even if gray_image exists
if not include_colored:
    colored_image_base64 = None

# Typically:
# - gray_image_base64: Always present (if image exists)
# - colored_image_base64: Optional (only if include_colored=True)
```

**Use Cases**:
- Gray: Essential for threshold visualization
- Colored: Useful for marker detection, debugging

---

### metadata

**Type**: `dict[str, Any]`
**Purpose**: Processor-specific metadata

**Valid Structure**:
```python
# Empty dict is valid
metadata = {}

# Common keys
metadata = {
    "stage": "preprocessing" | "alignment" | "detection" | "initial",
    "rotation_angle": float,      # For AutoRotate
    "bubbles_detected": int,      # For ReadOMR
    "alignment_method": str,      # For Alignment
    "confidence": float,          # For various processors
    # ... any processor-specific data
}
```

**Constraints**:
```python
# Must be JSON-serializable
import json
json.dumps(metadata)  # Should not raise exception

# Recommended size: < 1 KB
# Avoid storing large data (use separate fields)

# Common metadata fields:
# - stage: str (processing stage)
# - Any processor-specific metrics
# - NOT raw image data (use image fields)
```

**Examples**:
```python
# AutoRotate
{"stage": "preprocessing", "rotation_angle": 90, "match_score": 0.95}

# CropOnMarkers
{"stage": "preprocessing", "markers_found": 4, "method": "FOUR_DOTS"}

# ReadOMR
{"stage": "detection", "bubbles_detected": 40, "fields_processed": 10}

# Initial state
{"stage": "initial"}
```

---

### success

**Type**: `bool`
**Purpose**: Whether processor executed successfully

**Valid Values**: `True` or `False`

**Semantics**:
```python
# Success
success = True
error_message = None

# Failure
success = False
error_message = "Error description"
```

**Usage**:
```python
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
```

---

### error_message

**Type**: `str | None`
**Purpose**: Error message if processor failed

**Valid Values**:
```python
# Success case
error_message = None

# Failure case
error_message = "MarkerNotFoundError: Could not detect 4 markers"
error_message = "ValueError: Invalid template configuration"
```

**Constraints**:
```python
# Must be None if success=True
if success:
    assert error_message is None

# Should be non-empty if success=False
if not success:
    assert error_message is not None
    assert len(error_message) > 0

# Recommended max length: 500 characters
assert len(error_message) < 500

# Should be human-readable
# Include exception type and message
```

---

## WorkflowGraph Constraints

### nodes

**Type**: `list[dict[str, Any]]`
**Purpose**: List of node definitions

**Node Structure**:
```python
{
    "id": str,              # Unique node ID
    "label": str,           # Display label
    "metadata": dict        # Additional node data
}
```

**Node ID Constraints**:
```python
# Special nodes
"input"                     # Input node (always present)
"output"                    # Output node (always present)

# Processor nodes
"processor_0", "processor_1", ..., "processor_N"

# Must be unique
node_ids = [node["id"] for node in nodes]
assert len(node_ids) == len(set(node_ids))  # No duplicates
```

**Node Metadata**:
```python
# Input node
{
    "id": "input",
    "label": "Input Image",
    "metadata": {
        "type": "input",
        "file_path": "inputs/sample1.jpg"
    }
}

# Processor node
{
    "id": "processor_0",
    "label": "AutoRotate",
    "metadata": {
        "type": "processor",
        "order": 0
    }
}

# Output node
{
    "id": "output",
    "label": "Output",
    "metadata": {
        "type": "output"
    }
}
```

**Typical Node Count**:
```python
# Minimum: 2 (input + output)
len(nodes) >= 2

# Typical: 5-15 nodes
# input + 3-13 processors + output
```

---

### edges

**Type**: `list[dict[str, Any]]`
**Purpose**: List of edge definitions

**Edge Structure**:
```python
{
    "from": str,           # Source node ID
    "to": str,             # Target node ID
    "label": str | None    # Optional edge label
}
```

**Edge Constraints**:
```python
# from and to must reference valid nodes
for edge in edges:
    assert edge["from"] in node_ids
    assert edge["to"] in node_ids

# Typical pattern: linear chain
edges = [
    {"from": "input", "to": "processor_0"},
    {"from": "processor_0", "to": "processor_1"},
    {"from": "processor_1", "to": "processor_2"},
    {"from": "processor_2", "to": "output"}
]

# Edge count = node count - 1 (for linear chain)
assert len(edges) == len(nodes) - 1
```

**Graph Topology**:
```python
# Currently: Always linear (no branches)
# Future: Could support conditional processors

# Directed Acyclic Graph (DAG)
# - No cycles
# - Single source (input)
# - Single sink (output)
```

---

## WorkflowSession Constraints

### session_id

**Type**: `str`
**Format**: `session_YYYYMMDD_HHMMSS_<uuid8>`

**Generation**:
```python
session_id = f"session_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
```

**Example**: `"session_20240106_123456_abcd1234"`

**Constraints**:
```python
# Must be unique across all sessions
# Length: ~40 characters
assert len(session_id) == 40  # "session_" (8) + date (8) + "_" (1) + time (6) + "_" (1) + uuid (8)

# Must be valid filename (no special chars)
import re
assert re.match(r'^session_\d{8}_\d{6}_[0-9a-f]{8}$', session_id)
```

---

### file_path

**Type**: `str`
**Purpose**: Path to input file

**Constraints**:
```python
# Can be absolute or relative path
file_path = "/absolute/path/to/file.jpg"
file_path = "relative/path/to/file.jpg"

# Should be valid path (but file may not exist after processing)
# Used for display/reference only

# Typical length: < 500 characters
assert len(file_path) < 500
```

---

### template_name

**Type**: `str`
**Purpose**: Name of template used

**Constraints**:
```python
# Non-empty string
assert len(template_name) > 0

# Typical: filename without extension
template_name = "sample1"
template_name = "exam_template_v2"

# Max length: 200 characters
assert len(template_name) < 200
```

---

### start_time / end_time

**Type**: `str`
**Format**: ISO 8601

**Constraints**:
```python
# start_time: Always present
assert start_time is not None

# end_time: Present only after finalization
if finalized:
    assert end_time is not None
else:
    assert end_time is None

# end_time must be after start_time
if end_time is not None:
    start_dt = datetime.fromisoformat(start_time)
    end_dt = datetime.fromisoformat(end_time)
    assert end_dt >= start_dt
```

---

### total_duration_ms

**Type**: `float | None`
**Purpose**: Total execution time

**Valid Range**: 0.0 to unlimited

**Constraints**:
```python
# None until finalized
if not finalized:
    assert total_duration_ms is None

# Must match start/end time difference
if total_duration_ms is not None:
    start_dt = datetime.fromisoformat(start_time)
    end_dt = datetime.fromisoformat(end_time)
    calculated = (end_dt - start_dt).total_seconds() * 1000
    assert abs(total_duration_ms - calculated) < 1.0  # Within 1ms

# Typical range: 500ms - 30 seconds
# Sanity check: < 5 minutes
assert total_duration_ms < 300000  # 5 minutes
```

**Relationship with Processor Durations**:
```python
# Sum of processor durations may be less than total_duration_ms
# (overhead from tracking, serialization, etc.)

sum_processor_durations = sum(state.duration_ms for state in processor_states)
assert sum_processor_durations <= total_duration_ms
```

---

### processor_states

**Type**: `list[ProcessorState]`
**Purpose**: List of processor states

**Constraints**:
```python
# Can be empty (no processing done)
processor_states = []  # Valid

# Typically 1-20 states
assert 0 <= len(processor_states) <= 100

# States must be ordered by 'order' field
for i, state in enumerate(processor_states):
    assert state.order == i

# First state often "Input"
if len(processor_states) > 0:
    assert processor_states[0].name in ["Input", "input"]
```

---

### config

**Type**: `dict[str, Any]`
**Purpose**: Configuration used for session

**Typical Structure**:
```python
{
    "template_path": str,
    "config_path": str,
    # ... other config values
}
```

**Constraints**:
```python
# Must be JSON-serializable
json.dumps(config)  # Should not raise

# Recommended size: < 10 KB
# Should not include large objects
```

---

### metadata

**Type**: `dict[str, Any]`
**Purpose**: Additional session metadata

**Constraints**:
```python
# Can be empty
metadata = {}

# Must be JSON-serializable
json.dumps(metadata)

# Recommended keys:
{
    "version": "1.0.0",
    "platform": "Python 3.11",
    "opencv_version": "4.8.0",
    # ... any session-level data
}
```

---

## Image Encoding Constraints

### max_width

**Type**: `int | None`
**Default**: 800
**Purpose**: Maximum width for resizing

**Valid Range**:
```python
# None: No resizing
max_width = None

# Typical values: 400-1600
max_width = 400   # Small, faster loading
max_width = 800   # Default, good balance
max_width = 1200  # Larger, more detail
max_width = 1600  # Very large, best quality

# Minimum: 100 (sanity check)
assert max_width is None or max_width >= 100

# Maximum: 4000 (sanity check)
assert max_width is None or max_width <= 4000
```

**Impact on File Size**:
```
max_width | Approx JPEG size | Base64 size | Total session (10 images)
----------|------------------|-------------|---------------------------
400       | 10-20 KB         | 13-27 KB    | 130-270 KB
800       | 30-50 KB         | 40-67 KB    | 400-670 KB
1200      | 60-100 KB        | 80-133 KB   | 800 KB - 1.3 MB
1600      | 100-150 KB       | 133-200 KB  | 1.3-2.0 MB
None      | 200-500 KB       | 267-667 KB  | 2.7-6.7 MB
```

---

### quality

**Type**: `int`
**Default**: 85
**Purpose**: JPEG compression quality

**Valid Range**: 0-100

**Constraints**:
```python
assert 0 <= quality <= 100

# Recommended range: 75-95
quality = 75   # Smaller files, slight quality loss
quality = 85   # Default, good balance
quality = 95   # Larger files, minimal quality loss
```

**Impact on File Size**:
```
quality | File size | Visual quality
--------|-----------|----------------
50      | 50%       | Noticeable artifacts
75      | 75%       | Minor artifacts
85      | 100%      | Excellent (baseline)
95      | 150%      | Near-perfect
100     | 200%      | Perfect (but huge)
```

---

### include_colored

**Type**: `bool`
**Default**: True
**Purpose**: Whether to capture colored images

**Constraints**:
```python
# If True: Both gray and colored images captured
if include_colored:
    assert gray_image_base64 is not None
    assert colored_image_base64 is not None  # (if image exists)

# If False: Only gray images captured
if not include_colored:
    assert gray_image_base64 is not None     # (if image exists)
    assert colored_image_base64 is None

# Impact on session size:
# include_colored=True: ~2x file size
# include_colored=False: ~1x file size (only gray)
```

---

## Performance Constraints

### Time Overhead per Capture

**Breakdown**:
```
Image resizing:       5-20 ms  (depends on size)
JPEG encoding:        10-30 ms (depends on size and quality)
Base64 encoding:      1-5 ms
Metadata creation:    < 1 ms
State append:         < 1 ms
-----------------------------------
Total per capture:    20-60 ms
```

**Total Session Overhead**:
```
10 processors × 50 ms average = 500 ms
Plus finalization overhead: ~10 ms
Total tracking overhead: ~510 ms

Typical workflow time: 2-5 seconds
Overhead percentage: 10-25%
```

**Recommendation**: Acceptable overhead for debugging/visualization

---

### Memory Constraints

**Session Size in Memory**:
```
Component                  | Size
---------------------------|------------------
Session metadata           | ~1 KB
Graph (10 nodes, 9 edges)  | ~2 KB
Config + metadata          | ~5 KB
ProcessorState × 10        | ~10 KB
Gray images × 10           | 400-670 KB (base64)
Colored images × 10        | 400-670 KB (base64)
---------------------------|------------------
Total:                     | ~800 KB - 1.3 MB

With max_width=800, quality=85, include_colored=True
```

**Browser Memory Limits**:
```
Chrome:  ~2 GB per tab (depends on system)
Firefox: ~2 GB per tab
Safari:  ~1 GB per tab (more conservative)

Maximum sessions in memory: ~1500-2500 sessions (before hitting limits)
Practical limit: 10-50 sessions (for responsive UI)
```

**Recommendation**: Store sessions in IndexedDB, load on demand

---

### Storage Constraints

**JSON File Size**:
```
Session with:
- 10 processors
- max_width=800
- quality=85
- include_colored=True

JSON file size: ~1.2 MB (pretty-printed)
JSON file size: ~1.0 MB (minified)
```

**IndexedDB Limits**:
```
Chrome:  ~60% of disk free space (up to several GB)
Firefox: ~50% of disk free space (up to several GB)
Safari:  ~1 GB (more conservative)

Maximum sessions: ~500-5000 sessions (depends on browser and disk)
```

**Recommendation**: Implement quota management, allow user to delete old sessions

---

## Browser-Specific Constraints

### Data URI Length

**Maximum Length**:
```
Chrome:  ~2 MB per data URI
Firefox: ~100 MB per data URI
Safari:  ~2 MB per data URI
Edge:    ~2 MB per data URI

Our images (800px, quality=85): ~50 KB each
Well within limits for all browsers
```

### Base64 Encoding

**JavaScript Implementation**:
```javascript
// Modern browsers: btoa() for binary to base64
const base64 = btoa(binaryString);

// Alternative: Use Uint8Array and manual encoding
const base64 = arrayBufferToBase64(buffer);

// Decoding: atob() for base64 to binary
const binaryString = atob(base64);
```

### JSON.stringify Limits

**Maximum JSON Size**:
```
Chrome:  ~500 MB
Firefox: ~500 MB
Safari:  ~500 MB

Our sessions: ~1 MB each
Well within limits
```

### Canvas toDataURL Limits

**Maximum Canvas Size**:
```
Chrome:  16384 × 16384
Firefox: 11180 × 11180
Safari:  4096 × 4096

Our resized images: 800 × ~1200
Well within limits
```

---

## Migration-Specific Constraints

### localStorage vs IndexedDB

**localStorage**:
```
Limit: ~5-10 MB total
Not suitable for sessions (too large)
Use for: Small config, recent session IDs
```

**IndexedDB**:
```
Limit: Several GB (depends on browser)
Suitable for: Session storage
Asynchronous API (use promises)
```

### Web Workers

**Constraints**:
```
Cannot access DOM or Canvas directly
Can encode images via OffscreenCanvas (modern browsers)
Can process session data (JSON parsing, filtering)
Good for: Background session export, compression
```

### Memory Management

**Recommendations**:
```javascript
// Cleanup after use
function cleanupSession(session) {
  // Clear large base64 strings
  for (const state of session.processor_states) {
    state.gray_image_base64 = null;
    state.colored_image_base64 = null;
  }
}

// Load images on demand
async function loadImage(sessionId, stateOrder) {
  const session = await db.loadSession(sessionId);
  const state = session.processor_states[stateOrder];
  return state.gray_image_base64;
}

// Limit concurrent sessions in memory
const MAX_SESSIONS_IN_MEMORY = 10;
```

---

## Validation Constraints

### Required Fields

**ProcessorState**:
```python
# Always required
assert state.name is not None
assert state.order is not None
assert state.timestamp is not None
assert state.duration_ms is not None
assert state.image_shape is not None
assert state.success is not None

# Conditional
if state.success:
    assert state.error_message is None
else:
    assert state.error_message is not None
```

**WorkflowSession**:
```python
# Always required
assert session.session_id is not None
assert session.file_path is not None
assert session.template_name is not None
assert session.start_time is not None
assert session.processor_states is not None
assert session.graph is not None
assert session.config is not None
assert session.metadata is not None

# Required after finalization
if finalized:
    assert session.end_time is not None
    assert session.total_duration_ms is not None
```

---

## Error Handling Constraints

### Serialization Errors

**Handle Non-Serializable Data**:
```python
# Problem: NumPy types not JSON-serializable
metadata = {"mean": np.float64(123.45)}  # TypeError

# Solution: Convert to native Python types
metadata = {"mean": float(np.float64(123.45))}  # OK
```

### Memory Errors

**Handle Large Images**:
```python
try:
    base64_str = ImageEncoder.encode_image(image, max_width=800, quality=85)
except MemoryError:
    # Fallback: Reduce quality or size
    base64_str = ImageEncoder.encode_image(image, max_width=400, quality=75)
```

### Storage Errors

**Handle Quota Exceeded**:
```javascript
try {
  await db.saveSession(session);
} catch (error) {
  if (error.name === 'QuotaExceededError') {
    // Delete old sessions to free space
    await db.deleteOldestSessions(5);
    await db.saveSession(session);
  }
}
```

---

## Summary of Critical Constraints

| Constraint | Value/Rule | Impact |
|------------|-----------|---------|
| session_id format | `session_YYYYMMDD_HHMMSS_<uuid8>` | Uniqueness, sorting |
| Timestamp format | ISO 8601 | Parsing, timezone handling |
| Image max_width | 800 (default) | File size, quality trade-off |
| Image quality | 85 (default) | File size, visual quality |
| Base64 encoding | No data URI prefix | Consistency, size |
| Session file size | ~1 MB typical | Storage, transfer time |
| Memory overhead | 20-60 ms per capture | Performance impact |
| IndexedDB storage | Several GB limit | Maximum sessions |
| Processor state order | Sequential (0, 1, 2, ...) | Replay correctness |
| Graph topology | Linear DAG | Visualization simplicity |

---

## Related Constraints

- **Workflow Tracker**: `../workflow-tracker/constraints.md`
- **HTML Exporter**: `../html-export/constraints.md`
- **Processing Context**: `../../processing-context/concept.md`
- **OpenCV Operations**: `../../../technical/opencv/opencv-operations.md`
