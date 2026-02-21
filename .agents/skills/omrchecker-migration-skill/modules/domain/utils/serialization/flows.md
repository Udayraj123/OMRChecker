# Serialization Utils Flows

**Module**: Domain - Utils - Serialization
**Python Reference**: `src/utils/serialization.py`
**Last Updated**: 2026-02-21

---

## Overview

Serialization utilities provide generic object-to-dictionary conversion for dataclasses and other objects. The system handles nested structures, Path objects, Enums, and collections without circular dependencies.

**Use Case**: Convert complex Python objects (dataclasses, Paths, Enums) to JSON-serializable dictionaries for storage, transmission, or API responses.

---

## Main Function: dataclass_to_dict

### Function Signature

```python
def dataclass_to_dict(obj: Any) -> dict | list | Any:
    """Recursively convert a dataclass instance to a dictionary.

    Args:
        obj: The object to serialize (typically a dataclass instance)

    Returns:
        Dictionary representation suitable for JSON serialization
    """
```

---

## Serialization Flow

### Main Conversion Flow

```
START: dataclass_to_dict(obj)
│
├─► TYPE CHECK 1: Is Dataclass?
│   │
│   ├─ Check: is_dataclass(obj) and not isinstance(obj, type)
│   │   │
│   │   │ Note: isinstance(obj, type) check excludes dataclass CLASSES
│   │   │       We only want to serialize dataclass INSTANCES
│   │   │
│   │   ├─ If TRUE (obj is a dataclass instance):
│   │   │   │
│   │   │   ├─► Convert to dict using asdict()
│   │   │   │   dataclass_dict = asdict(obj)
│   │   │   │
│   │   │   └─► Recursively process each field
│   │   │       return {
│   │   │           key: dataclass_to_dict(value)
│   │   │           for key, value in dataclass_dict.items()
│   │   │       }
│   │   │       END
│   │   │
│   │   └─ If FALSE: Continue to next type check
│
├─► TYPE CHECK 2: Is Path?
│   │
│   ├─ Check: isinstance(obj, Path)
│   │   │
│   │   ├─ If TRUE (obj is Path):
│   │   │   │
│   │   │   └─► Convert to string
│   │   │       return str(obj)
│   │   │       │
│   │   │       │ Examples:
│   │   │       │ Path("/tmp/test") → "/tmp/test"
│   │   │       │ Path("relative/path") → "relative/path"
│   │   │       │ Path.home() → "/Users/username"
│   │   │       │
│   │   │       END
│   │   │
│   │   └─ If FALSE: Continue to next type check
│
├─► TYPE CHECK 3: Is Enum?
│   │
│   ├─ Check: isinstance(obj, Enum)
│   │   │
│   │   ├─ If TRUE (obj is Enum):
│   │   │   │
│   │   │   └─► Extract value
│   │   │       return obj.value
│   │   │       │
│   │   │       │ Examples:
│   │   │       │ Color.RED (value=1) → 1
│   │   │       │ Status.ACTIVE (value="active") → "active"
│   │   │       │ Priority.HIGH (value=3) → 3
│   │   │       │
│   │   │       END
│   │   │
│   │   └─ If FALSE: Continue to next type check
│
├─► TYPE CHECK 4: Is Dictionary?
│   │
│   ├─ Check: isinstance(obj, dict)
│   │   │
│   │   ├─ If TRUE (obj is dict):
│   │   │   │
│   │   │   └─► Recursively process values
│   │   │       return {
│   │   │           key: dataclass_to_dict(value)
│   │   │           for key, value in obj.items()
│   │   │       }
│   │   │       │
│   │   │       │ Note: Keys are NOT processed, only values
│   │   │       │ Keys must already be JSON-serializable (str, int)
│   │   │       │
│   │   │       END
│   │   │
│   │   └─ If FALSE: Continue to next type check
│
├─► TYPE CHECK 5: Is List or Tuple?
│   │
│   ├─ Check: isinstance(obj, (list, tuple))
│   │   │
│   │   ├─ If TRUE (obj is list/tuple):
│   │   │   │
│   │   │   └─► Recursively process items
│   │   │       return [
│   │   │           dataclass_to_dict(item)
│   │   │           for item in obj
│   │   │       ]
│   │   │       │
│   │   │       │ Note: Always returns list (even if input is tuple)
│   │   │       │ JSON doesn't distinguish between list/tuple
│   │   │       │
│   │   │       END
│   │   │
│   │   └─ If FALSE: Continue to next type check
│
├─► TYPE CHECK 6: Is Primitive?
│   │
│   ├─ Check: isinstance(obj, (str, int, float, bool, type(None)))
│   │   │
│   │   ├─ If TRUE (obj is primitive):
│   │   │   │
│   │   │   └─► Return as-is
│   │   │       return obj
│   │   │       │
│   │   │       │ These types are already JSON-serializable
│   │   │       │ No conversion needed
│   │   │       │
│   │   │       END
│   │   │
│   │   └─ If FALSE: Continue to fallback
│
└─► FALLBACK: Unknown Type
    │
    ├─► Try to convert to string
    │   try:
    │       return str(obj)
    │   except Exception:
    │       return obj
    │   │
    │   │ Handles:
    │   │ - datetime objects → ISO string
    │   │ - Custom objects with __str__ → string representation
    │   │ - Objects without __str__ → default repr
    │   │
    │   │ If str() fails (rare), return object as-is
    │   │ Caller may get JSON serialization error later
    │
    END
```

---

## Detailed Examples

### Example 1: Simple Dataclass

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    path: Path
    value: int = 10

config = Config(path=Path("/tmp/test"), value=42)

# Serialization flow:
dataclass_to_dict(config)
│
├─► TYPE CHECK 1: is_dataclass(config) → True
│   │
│   ├─► asdict(config) → {"path": Path("/tmp/test"), "value": 42}
│   │
│   └─► Process each field:
│       │
│       ├─► key="path", value=Path("/tmp/test")
│       │   │
│       │   └─► dataclass_to_dict(Path("/tmp/test"))
│       │       │
│       │       ├─► TYPE CHECK 2: isinstance(Path("/tmp/test"), Path) → True
│       │       │
│       │       └─► return "/tmp/test"
│       │
│       ├─► key="value", value=42
│       │   │
│       │   └─► dataclass_to_dict(42)
│       │       │
│       │       ├─► TYPE CHECK 6: isinstance(42, int) → True
│       │       │
│       │       └─► return 42
│       │
│       └─► Final result: {"path": "/tmp/test", "value": 42}

# Result:
{"path": "/tmp/test", "value": 42}
```

### Example 2: Nested Dataclasses

```python
@dataclass
class Point:
    x: int
    y: int

@dataclass
class Shape:
    name: str
    center: Point

shape = Shape(name="circle", center=Point(x=10, y=20))

# Serialization flow:
dataclass_to_dict(shape)
│
├─► TYPE CHECK 1: is_dataclass(shape) → True
│   │
│   ├─► asdict(shape) → {"name": "circle", "center": Point(10, 20)}
│   │
│   └─► Process each field:
│       │
│       ├─► key="name", value="circle"
│       │   └─► dataclass_to_dict("circle") → "circle" (primitive)
│       │
│       ├─► key="center", value=Point(10, 20)
│       │   │
│       │   └─► dataclass_to_dict(Point(10, 20))
│       │       │
│       │       ├─► TYPE CHECK 1: is_dataclass(Point(10, 20)) → True
│       │       │
│       │       ├─► asdict(Point(10, 20)) → {"x": 10, "y": 20}
│       │       │
│       │       └─► Process nested fields:
│       │           │
│       │           ├─► key="x", value=10 → 10
│       │           └─► key="y", value=20 → 20
│       │           │
│       │           └─► return {"x": 10, "y": 20}
│       │
│       └─► Final result: {"name": "circle", "center": {"x": 10, "y": 20}}

# Result:
{
    "name": "circle",
    "center": {
        "x": 10,
        "y": 20
    }
}
```

### Example 3: Enum Serialization

```python
from enum import Enum

class Status(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETE = "complete"

@dataclass
class Task:
    name: str
    status: Status

task = Task(name="Review", status=Status.ACTIVE)

# Serialization flow:
dataclass_to_dict(task)
│
├─► TYPE CHECK 1: is_dataclass(task) → True
│   │
│   ├─► asdict(task) → {"name": "Review", "status": Status.ACTIVE}
│   │
│   └─► Process each field:
│       │
│       ├─► key="name", value="Review" → "Review"
│       │
│       ├─► key="status", value=Status.ACTIVE
│       │   │
│       │   └─► dataclass_to_dict(Status.ACTIVE)
│       │       │
│       │       ├─► TYPE CHECK 3: isinstance(Status.ACTIVE, Enum) → True
│       │       │
│       │       └─► return Status.ACTIVE.value → "active"
│       │
│       └─► Final result: {"name": "Review", "status": "active"}

# Result:
{"name": "Review", "status": "active"}
```

### Example 4: Collections (Lists, Dicts)

```python
@dataclass
class Config:
    paths: list[Path]
    settings: dict[str, int]

config = Config(
    paths=[Path("/tmp/a"), Path("/tmp/b")],
    settings={"timeout": 30, "retries": 3}
)

# Serialization flow:
dataclass_to_dict(config)
│
├─► TYPE CHECK 1: is_dataclass(config) → True
│   │
│   ├─► asdict(config) → {
│   │       "paths": [Path("/tmp/a"), Path("/tmp/b")],
│   │       "settings": {"timeout": 30, "retries": 3}
│   │   }
│   │
│   └─► Process each field:
│       │
│       ├─► key="paths", value=[Path("/tmp/a"), Path("/tmp/b")]
│       │   │
│       │   └─► dataclass_to_dict([Path("/tmp/a"), Path("/tmp/b")])
│       │       │
│       │       ├─► TYPE CHECK 5: isinstance([...], list) → True
│       │       │
│       │       └─► Process each item:
│       │           │
│       │           ├─► dataclass_to_dict(Path("/tmp/a")) → "/tmp/a"
│       │           ├─► dataclass_to_dict(Path("/tmp/b")) → "/tmp/b"
│       │           │
│       │           └─► return ["/tmp/a", "/tmp/b"]
│       │
│       ├─► key="settings", value={"timeout": 30, "retries": 3}
│       │   │
│       │   └─► dataclass_to_dict({"timeout": 30, "retries": 3})
│       │       │
│       │       ├─► TYPE CHECK 4: isinstance({...}, dict) → True
│       │       │
│       │       └─► Process each value:
│       │           │
│       │           ├─► key="timeout", value=30 → 30
│       │           ├─► key="retries", value=3 → 3
│       │           │
│       │           └─► return {"timeout": 30, "retries": 3}
│       │
│       └─► Final result: {
│               "paths": ["/tmp/a", "/tmp/b"],
│               "settings": {"timeout": 30, "retries": 3}
│           }

# Result:
{
    "paths": ["/tmp/a", "/tmp/b"],
    "settings": {"timeout": 30, "retries": 3}
}
```

### Example 5: Fallback for Unknown Types

```python
from datetime import datetime

@dataclass
class Event:
    name: str
    timestamp: datetime

event = Event(name="Login", timestamp=datetime(2024, 1, 15, 10, 30))

# Serialization flow:
dataclass_to_dict(event)
│
├─► TYPE CHECK 1: is_dataclass(event) → True
│   │
│   ├─► asdict(event) → {
│   │       "name": "Login",
│   │       "timestamp": datetime(2024, 1, 15, 10, 30)
│   │   }
│   │
│   └─► Process each field:
│       │
│       ├─► key="name", value="Login" → "Login"
│       │
│       ├─► key="timestamp", value=datetime(2024, 1, 15, 10, 30)
│       │   │
│       │   └─► dataclass_to_dict(datetime(2024, 1, 15, 10, 30))
│       │       │
│       │       ├─► TYPE CHECK 1-6: All fail (datetime is none of these)
│       │       │
│       │       └─► FALLBACK: Try str()
│       │           │
│       │           ├─► str(datetime(2024, 1, 15, 10, 30))
│       │           │   → "2024-01-15 10:30:00"
│       │           │
│       │           └─► return "2024-01-15 10:30:00"
│       │
│       └─► Final result: {
│               "name": "Login",
│               "timestamp": "2024-01-15 10:30:00"
│           }

# Result:
{
    "name": "Login",
    "timestamp": "2024-01-15 10:30:00"
}

# Note: datetime is converted to string, not ISO format
# For ISO format, use datetime.isoformat() before serialization
```

---

## Edge Cases

### Edge Case 1: Dataclass Class (Not Instance)

```python
@dataclass
class Config:
    value: int

# Pass the CLASS, not an instance
result = dataclass_to_dict(Config)

# Flow:
├─► TYPE CHECK 1: is_dataclass(Config) → True
│   │           but isinstance(Config, type) → True (it's a class!)
│   │
│   └─► Check fails: is_dataclass(obj) and not isinstance(obj, type) → False
│
├─► TYPE CHECK 2-6: All fail (class is not Path, Enum, dict, list, primitive)
│
└─► FALLBACK: str(Config)
    → "<class '__main__.Config'>"

# Result:
"<class '__main__.Config'>"

# This prevents accidentally serializing class definitions instead of instances
```

### Edge Case 2: Circular References

```python
@dataclass
class Node:
    value: int
    next: 'Node' = None

# Create circular reference
node1 = Node(value=1)
node2 = Node(value=2)
node1.next = node2
node2.next = node1  # Circular!

# Attempt serialization:
dataclass_to_dict(node1)

# Flow:
├─► asdict(node1)  # Python's asdict() detects circular reference
│   └─► Raises RecursionError or ValueError
│
# EXCEPTION: RecursionError: maximum recursion depth exceeded

# Note: Current implementation does NOT handle circular references
# asdict() will raise an error before dataclass_to_dict can handle it
```

### Edge Case 3: None Values

```python
@dataclass
class Config:
    path: Path | None
    value: int | None

config = Config(path=None, value=None)

# Serialization flow:
dataclass_to_dict(config)
│
├─► asdict(config) → {"path": None, "value": None}
│
├─► Process fields:
│   │
│   ├─► key="path", value=None
│   │   │
│   │   └─► dataclass_to_dict(None)
│   │       │
│   │       ├─► TYPE CHECK 6: isinstance(None, type(None)) → True
│   │       │
│   │       └─► return None
│   │
│   └─► key="value", value=None → None
│
└─► Final result: {"path": null, "value": null}

# Result:
{"path": null, "value": null}

# Note: None is correctly serialized to JSON null
```

### Edge Case 4: Empty Collections

```python
@dataclass
class Config:
    items: list[str]
    mapping: dict[str, int]

config = Config(items=[], mapping={})

# Serialization flow:
dataclass_to_dict(config)
│
├─► Process fields:
│   │
│   ├─► key="items", value=[]
│   │   │
│   │   └─► dataclass_to_dict([])
│   │       │
│   │       ├─► TYPE CHECK 5: isinstance([], list) → True
│   │       │
│   │       └─► return [] (empty list comprehension)
│   │
│   └─► key="mapping", value={}
│       │
│       └─► dataclass_to_dict({})
│           │
│           ├─► TYPE CHECK 4: isinstance({}, dict) → True
│           │
│           └─► return {} (empty dict comprehension)
│
└─► Final result: {"items": [], "mapping": {}}

# Result:
{"items": [], "mapping": {}}

# Note: Empty collections are preserved
```

### Edge Case 5: Tuple Conversion

```python
@dataclass
class Config:
    coordinates: tuple[int, int, int]

config = Config(coordinates=(10, 20, 30))

# Serialization flow:
dataclass_to_dict(config)
│
├─► Process field "coordinates":
│   │
│   └─► dataclass_to_dict((10, 20, 30))
│       │
│       ├─► TYPE CHECK 5: isinstance((10, 20, 30), tuple) → True
│       │
│       └─► return [10, 20, 30]  # Returns LIST, not tuple

# Result:
{"coordinates": [10, 20, 30]}

# Note: Tuples are converted to lists (JSON doesn't have tuples)
# Information about tuple vs list is lost
```

---

## Usage Patterns

### Pattern 1: Template Serialization

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TemplateConfig:
    template_path: Path
    dimensions: tuple[int, int]
    markers: list[str]

template_config = TemplateConfig(
    template_path=Path("/templates/form.json"),
    dimensions=(1200, 1600),
    markers=["FOUR_DOTS", "TWO_LINES"]
)

# Serialize for JSON export
json_data = dataclass_to_dict(template_config)

# Use with json.dumps()
import json
json_string = json.dumps(json_data, indent=2)

# Result:
# {
#   "template_path": "/templates/form.json",
#   "dimensions": [1200, 1600],
#   "markers": ["FOUR_DOTS", "TWO_LINES"]
# }
```

### Pattern 2: Config Export

```python
@dataclass
class ThresholdConfig:
    min_jump: float
    default_threshold: float

@dataclass
class TuningConfig:
    threshold: ThresholdConfig
    outputs_dir: Path

config = TuningConfig(
    threshold=ThresholdConfig(min_jump=30.0, default_threshold=127.5),
    outputs_dir=Path("/outputs")
)

# Export to JSON file
import json

serialized = dataclass_to_dict(config)
with open("/config/tuning.json", "w") as f:
    json.dump(serialized, f, indent=2)
```

### Pattern 3: API Response

```python
from enum import Enum

class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"

@dataclass
class ProcessingResult:
    file_path: Path
    status: ProcessingStatus
    errors: list[str]
    timestamp: datetime

result = ProcessingResult(
    file_path=Path("/scans/sheet001.jpg"),
    status=ProcessingStatus.COMPLETE,
    errors=[],
    timestamp=datetime.now()
)

# Serialize for API response
response_data = dataclass_to_dict(result)

# Returns:
# {
#     "file_path": "/scans/sheet001.jpg",
#     "status": "complete",
#     "errors": [],
#     "timestamp": "2024-01-15 10:30:00"
# }
```

---

## Browser Migration

### JavaScript Implementation

```javascript
/**
 * Convert an object to a JSON-serializable dictionary
 * Handles nested objects, arrays, and special types
 */
function dataclassToDict(obj) {
    // Check if obj is null or undefined
    if (obj === null || obj === undefined) {
        return obj;
    }

    // Check for primitive types
    const primitiveTypes = ['string', 'number', 'boolean'];
    if (primitiveTypes.includes(typeof obj)) {
        return obj;
    }

    // Check for Date objects (equivalent to datetime)
    if (obj instanceof Date) {
        return obj.toISOString();  // Better than str() - ISO format
    }

    // Check for Array (equivalent to list/tuple)
    if (Array.isArray(obj)) {
        return obj.map(item => dataclassToDict(item));
    }

    // Check for plain objects (equivalent to dict or dataclass)
    if (typeof obj === 'object' && obj.constructor === Object) {
        const result = {};
        for (const [key, value] of Object.entries(obj)) {
            result[key] = dataclassToDict(value);
        }
        return result;
    }

    // Check for Map (alternative to dict)
    if (obj instanceof Map) {
        const result = {};
        for (const [key, value] of obj.entries()) {
            result[key] = dataclassToDict(value);
        }
        return result;
    }

    // Check for Set (convert to array)
    if (obj instanceof Set) {
        return Array.from(obj).map(item => dataclassToDict(item));
    }

    // Fallback: try to convert to string
    try {
        // Check if object has toJSON method
        if (typeof obj.toJSON === 'function') {
            return obj.toJSON();
        }
        // Convert to string
        return String(obj);
    } catch (e) {
        // If all else fails, return as-is
        return obj;
    }
}
```

### TypeScript Implementation with Types

```typescript
type Primitive = string | number | boolean | null | undefined;

interface SerializableObject {
    [key: string]: any;
}

function dataclassToDict(obj: any): Primitive | SerializableObject | any[] {
    // Null/undefined
    if (obj === null || obj === undefined) {
        return obj;
    }

    // Primitives
    if (
        typeof obj === 'string' ||
        typeof obj === 'number' ||
        typeof obj === 'boolean'
    ) {
        return obj;
    }

    // Date → ISO string
    if (obj instanceof Date) {
        return obj.toISOString();
    }

    // Array → recursive map
    if (Array.isArray(obj)) {
        return obj.map((item) => dataclassToDict(item));
    }

    // Plain object → recursive process
    if (typeof obj === 'object' && obj.constructor === Object) {
        const result: SerializableObject = {};
        for (const [key, value] of Object.entries(obj)) {
            result[key] = dataclassToDict(value);
        }
        return result;
    }

    // Map → object
    if (obj instanceof Map) {
        const result: SerializableObject = {};
        for (const [key, value] of obj.entries()) {
            result[String(key)] = dataclassToDict(value);
        }
        return result;
    }

    // Set → array
    if (obj instanceof Set) {
        return Array.from(obj).map((item) => dataclassToDict(item));
    }

    // Custom toJSON
    if (typeof obj.toJSON === 'function') {
        return obj.toJSON();
    }

    // Fallback: stringify
    try {
        return String(obj);
    } catch (e) {
        return obj;
    }
}
```

### Key Differences from Python

1. **No Dataclass Type**:
   ```javascript
   // Python: Check if dataclass
   if is_dataclass(obj) and not isinstance(obj, type):
       return {key: dataclass_to_dict(value) for key, value in asdict(obj).items()}

   // JavaScript: Plain objects are already dict-like
   if (typeof obj === 'object' && obj.constructor === Object) {
       // Process as plain object
   }
   ```

2. **No Path Type**:
   ```javascript
   // Python: Path objects converted to string
   if isinstance(obj, Path):
       return str(obj)

   // JavaScript: No Path type in browser
   // File paths are already strings
   // For File objects:
   if (obj instanceof File) {
       return obj.name;  // Or obj.webkitRelativePath
   }
   ```

3. **No Enum Type (native)**:
   ```javascript
   // Python: Extract enum value
   if isinstance(obj, Enum):
       return obj.value

   // JavaScript: No native Enum
   // Use TypeScript enums or plain objects
   // TypeScript enum values are already primitives
   ```

4. **Date Handling**:
   ```javascript
   // Python: datetime → str() (not ideal)
   // "2024-01-15 10:30:00"

   // JavaScript: Date → toISOString() (better!)
   // "2024-01-15T10:30:00.000Z"
   ```

5. **JSON.stringify Integration**:
   ```javascript
   // In browser, can use native JSON.stringify
   const obj = {
       name: "test",
       timestamp: new Date(),
       nested: { value: 42 }
   };

   // Option 1: Use dataclassToDict first
   const serialized = dataclassToDict(obj);
   const json = JSON.stringify(serialized);

   // Option 2: Use toJSON method
   obj.toJSON = function() {
       return dataclassToDict(this);
   };
   const json = JSON.stringify(obj);
   ```

---

## Related Documentation

- **Parsing Utils**: `../parsing/flows.md` (for reverse: JSON → objects)
- **Config Management**: `../../../foundation/configuration.md`
- **Template Entity**: `../../template/concept.md`
- **File Utils**: `../file/flows.md`

---

## Summary

Serialization utilities provide:

1. **Generic object-to-dict conversion** for any Python object
2. **Recursive processing** of nested structures
3. **Special handling** for Path, Enum, datetime
4. **Fallback strategy** for unknown types
5. **JSON-ready output** for storage/transmission

**Type Support**:
- Dataclasses → dictionaries
- Path objects → strings
- Enums → values
- Collections → recursive conversion
- Primitives → pass-through
- Unknown types → string fallback

**Limitations**:
- No circular reference handling
- Tuple/list distinction lost
- datetime → str() (not ISO format)
- No validation of output

**Browser Equivalent**: Use plain objects with recursive processing, Date.toISOString(), and JSON.stringify()
