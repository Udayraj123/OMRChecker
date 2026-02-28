# Python ↔ TypeScript Migration Differences

**Created**: 2026-02-28  
**Purpose**: Document intentional differences between Python and TypeScript implementations

---

## Overview

This document tracks deliberate differences between the Python (CLI) and TypeScript (browser) versions of OMRChecker. Not all Python features can or should be migrated to the browser environment.

---

## Architecture Differences

### 1. Entry Points

**Python**:
- CLI-based: `main.py` → `src/entry.py`
- Batch processing with file system access
- Terminal interaction for setup/debugging

**TypeScript**:
- Browser-based: `OMRProcessor.ts`
- Single-image or FileList processing
- Web UI for interaction

**Reason**: Different execution environments

---

### 2. File I/O

**Python**:
- Direct file system access via `pathlib`, `os`
- Read/write files anywhere
- Directory traversal

**TypeScript**:
- Browser File API (user uploads only)
- No direct file system access
- Memory-based processing

**Reason**: Browser security sandbox

---

### 3. External Dependencies

**Python**:
- NumPy for array operations
- Matplotlib for visualization
- Rich for terminal UI
- Pillow for image I/O

**TypeScript**:
- Native TypedArrays (no NumPy)
- Canvas API for visualization
- No terminal (browser console only)
- HTMLImageElement / Canvas for images

**Reason**: Browser APIs vs system libraries

---

## Migration Differences by Module

### utils/math.py → utils/math.ts

| Feature | Python | TypeScript | Notes |
|---------|--------|------------|-------|
| **Color conversion** | `matplotlib.colors.to_rgb()` | Canvas `fillStyle` parsing | Browser-native approach |
| **Point types** | Lists `[x, y]` | Tuples `[number, number]` | TypeScript type safety |
| **NumPy arrays** | `np.array()`, `.sum()`, `.argmin()` | Native arrays with `.reduce()`, `Math.min()` | No NumPy in browser |
| **Type hints** | Minimal (11% coverage) | Full (100% coverage) | TypeScript enforces types |

**Key Difference**: TypeScript version is fully typed with explicit Point/Rectangle types.

---

### utils/checksum.py → utils/checksum.ts

| Feature | Python | TypeScript | Notes |
|---------|--------|------------|-------|
| **Hashing** | `hashlib` (md5, sha1, sha256, sha512) | Web Crypto API (SHA-1, SHA-256, SHA-384, SHA-512) | No MD5 in Web Crypto (security) |
| **File input** | `Path` object, file path string | `ArrayBuffer` or `Blob` | Browser file handling |
| **Sync/async** | Synchronous | Asynchronous (Promises) | Browser APIs are async |
| **Error suppression** | `contextlib.suppress()` | `try/catch` with empty catch | Same behavior, different syntax |

**Key Difference**: Async API, no MD5 support, different input types.

---

## Excluded Features

### ML Training & Models
**Not migrated**:
- `src/processors/experimental/training/`
- `src/processors/detection/ml_bubble_detector.py`
- `src/processors/detection/models/stn_module.py`
- All training scripts in `scripts/ai-generated/`

**Reason**: Training happens offline, only inference needed in browser

---

### CLI-Specific Features
**Not migrated**:
- `src/entry.py` - CLI entry point
- `main.py` - CLI entry point  
- `src/utils/interaction.py` - Terminal prompts
- `src/utils/env.py` - Environment variables

**Reason**: No terminal in browser

---

### Experimental Features
**Not migrated**:
- `src/processors/experimental/organization/` - File organization
- Advanced alignment (K-nearest, piecewise affine)

**Reason**: Not production-ready or not needed for browser

---

## Type System Differences

### Python Type Hints (Gradual)

```python
def distance(point1, point2):  # No types
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])
```

**Coverage**: ~11% in utils/math.py

### TypeScript Types (Strict)

```typescript
type Point = [number, number];

static distance(point1: Point, point2: Point): number {
  const dx = point1[0] - point2[0];
  const dy = point1[1] - point2[1];
  return Math.hypot(dx, dy);
}
```

**Coverage**: 100% (TypeScript enforced)

**Difference**: TypeScript version is more strictly typed with domain-specific types (Point, Rectangle, Line).

---

## OpenCV Differences

### Python OpenCV (cv2)

```python
import cv2
result = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
```

- Direct native library
- Automatic memory management
- Synchronous operations

### TypeScript OpenCV.js

```typescript
import cv from '@techstark/opencv-js';

const result = new cv.Mat();
try {
  cv.resize(src, result, new cv.Size(width, height), 0, 0, cv.INTER_LINEAR);
  return result.clone();
} finally {
  result.delete(); // MUST manually free memory!
}
```

- WebAssembly wrapper
- **Manual memory management required**
- All operations synchronous (but load is async)

**Critical Difference**: TypeScript MUST call `.delete()` on all cv.Mat objects to prevent memory leaks.

---

## Naming Conventions

### Python (snake_case)

```python
def shift_points_from_origin(new_origin, list_of_points):
    return [add_points(new_origin, point) for point in list_of_points]
```

### TypeScript (camelCase)

```typescript
static shiftPointsFromOrigin(newOrigin: Point, listOfPoints: Point[]): Point[] {
  return listOfPoints.map(point => MathUtils.addPoints(newOrigin, point));
}
```

**Conversion**: Automated by migration scripts

---

## Error Handling

### Python Exceptions

```python
if not file_path.exists():
    msg = f"File not found: {file_path}"
    raise FileNotFoundError(msg)
```

### TypeScript Errors

```typescript
if (!fileExists) {
  throw new Error(`File not found: ${filePath}`);
}
```

**Difference**: TypeScript uses generic `Error` class, Python has specific exception types.

---

## Configuration Format

### Python (snake_case JSON keys)

```json
{
  "field_blocks": [...],
  "pre_processors": [...],
  "bubble_threshold": 127
}
```

### TypeScript (camelCase JSON keys)

```json
{
  "fieldBlocks": [...],
  "preProcessors": [...],
  "bubbleThreshold": 127
}
```

**Conversion**: Handled by schema validation and migration scripts

---

## Browser-Specific Adaptations

### 1. No File System Access
**Python**: Read/write files anywhere  
**TypeScript**: User must upload files via `<input type="file">`

### 2. Async Everything
**Python**: Mostly synchronous  
**TypeScript**: File operations, image loading = async

### 3. Memory Constraints
**Python**: Can process large batches  
**TypeScript**: Limited by browser memory, process one at a time

### 4. No Terminal Interaction
**Python**: `InteractionUtils.show()` for debugging  
**TypeScript**: Canvas rendering or console.log only

---

## Testing Differences

### Python Tests (pytest)

```python
def test_distance():
    p1 = [0, 0]
    p2 = [3, 4]
    assert MathUtils.distance(p1, p2) == 5.0
```

### TypeScript Tests (Vitest/Jest)

```typescript
describe('MathUtils', () => {
  it('calculates distance correctly', () => {
    const p1: Point = [0, 0];
    const p2: Point = [3, 4];
    expect(MathUtils.distance(p1, p2)).toBe(5.0);
  });
});
```

**Difference**: Different test frameworks, same logic

---

## Migration Quality Metrics

### Target Thresholds

- **Type Safety**: ≥95% (< 5 'any' types per file)
- **Validation Score**: ≥80%
- **Compilation**: 0 errors
- **Function Coverage**: 100% of public API

### Actual Results (so far)

| File | Python Lines | TS Lines | Type Safety | Validation | Notes |
|------|-------------|----------|-------------|------------|-------|
| checksum.py | 50 | 66 | 100% (0 any) | 80% | Test migration |
| math.py | 179 | 332 | 100% (0 any) | 60%* | *Validator issue |

*Validation score doesn't account for enhanced documentation and type safety

---

## Decision Log

### Why Not Migrate Feature X?

1. **ML Training** - Runs offline, models used in browser
2. **CLI Tools** - No terminal in browser
3. **Interaction.show()** - No CV2 window in browser
4. **pytest fixtures** - Different test framework
5. **pathlib** - Browser has no file system

### Why Adapt Feature Y?

1. **Color conversion** - Use Canvas API (browser-native)
2. **Async operations** - Browser APIs are promise-based
3. **Memory management** - OpenCV.js requires manual `.delete()`
4. **Type definitions** - TypeScript enforces stricter types

---

## References

- **Exclusion List**: `.ts-migration-exclude`
- **Migration Skill**: `.agents/skills/python-to-typescript-migration/SKILL.md`
- **Subagent Tasks**: `.agents/SUBAGENT_TASKS.md`
- **FILE_MAPPING.json**: Tracks migration status

---

**Last Updated**: 2026-02-28  
**Maintained By**: Migration Team
