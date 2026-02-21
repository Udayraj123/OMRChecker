# Serialization Utils Constraints

**Module**: Domain - Utils - Serialization
**Python Reference**: `src/utils/serialization.py`
**Last Updated**: 2026-02-21

---

## Input Constraints

### obj Parameter

**Type**: `Any`
**Description**: The object to serialize (typically a dataclass instance)

**Valid Inputs**:
```python
# Primitives
None                          # Returns None
True, False                   # Returns bool
42                            # Returns int
3.14                          # Returns float
"hello"                       # Returns str

# Collections
[]                            # Returns []
[1, 2, 3]                     # Returns [1, 2, 3]
(1, 2, 3)                     # Returns [1, 2, 3] (tuple → list)
{}                            # Returns {}
{"key": "value"}              # Returns {"key": "value"}

# Special types
Path("/tmp/test")             # Returns "/tmp/test"
Status.ACTIVE                 # Returns enum value (e.g., "active")

# Dataclasses
@dataclass
class Config:
    value: int
Config(value=42)              # Returns {"value": 42}

# Unknown types
datetime(2024, 1, 15)         # Returns "2024-01-15 00:00:00" (str fallback)
```

**Invalid Inputs** (cause undefined behavior):
```python
# Circular references
node1 = Node(value=1)
node2 = Node(value=2)
node1.next = node2
node2.next = node1
# RecursionError from asdict()

# Non-serializable objects (rare)
obj = SomeClass()  # If str() fails
# May return unpredictable string representation
```

**Constraints**:
- No type restrictions (accepts `Any`)
- No size limit (memory permitting)
- Circular references will cause RecursionError
- Non-finite numbers (NaN, Inf) pass through (may break JSON serialization later)

---

## Type Handling Constraints

### Dataclass Constraints

**Detection**:
```python
is_dataclass(obj) and not isinstance(obj, type)
```

**Why the `not isinstance(obj, type)` check?**
```python
@dataclass
class Config:
    value: int

# Config is a dataclass CLASS
is_dataclass(Config)  # True
isinstance(Config, type)  # True
# Result: Check fails, Config is NOT serialized as dataclass

# Config(42) is a dataclass INSTANCE
is_dataclass(Config(42))  # True
isinstance(Config(42), type)  # False
# Result: Check succeeds, serialized as {"value": 42}
```

**Constraint**: Only dataclass instances are serialized, not dataclass classes.

**asdict() Behavior**:
```python
from dataclasses import asdict

@dataclass
class Config:
    value: int
    nested: 'OtherDataclass'

# asdict() recursively converts nested dataclasses
# But returns Python objects (Path, Enum, etc.) unchanged
# That's why we need to recursively call dataclass_to_dict
```

**Constraint**: `asdict()` returns dict with potentially non-JSON-serializable values. Recursive processing required.

### Path Constraints

**Type**: `pathlib.Path`

**Conversion**:
```python
str(Path("/tmp/test"))  # "/tmp/test"
str(Path("relative"))   # "relative"
str(Path.home())        # "/Users/username"
```

**Constraints**:
- Always converts to string
- Preserves relative vs absolute path
- Platform-dependent separators (/ on Unix, \ on Windows)
- Empty Path: `str(Path())` → "."

**Edge Cases**:
```python
Path()              # → "."
Path("/")           # → "/"
Path("//share")     # → "//share" (UNC path on Windows)
```

### Enum Constraints

**Type**: `enum.Enum`

**Conversion**:
```python
class Status(Enum):
    ACTIVE = "active"
    PENDING = 1

Status.ACTIVE.value  # "active"
Status.PENDING.value  # 1
```

**Constraints**:
- Always extracts `.value` attribute
- Value type is NOT constrained (can be str, int, tuple, etc.)
- Assumes enum value is JSON-serializable (not enforced)

**Edge Cases**:
```python
# Enum with complex value
class Color(Enum):
    RED = (255, 0, 0)

Color.RED.value  # (255, 0, 0) → tuple
# Later converted to [255, 0, 0] by list/tuple handler

# Enum with non-serializable value
class Mode(Enum):
    PROCESS = SomeClass()

Mode.PROCESS.value  # SomeClass instance
# Falls through to str() fallback
```

### Dictionary Constraints

**Type**: `dict`

**Key Constraints**:
```python
# Keys are NOT processed, only values
{
    "string_key": 1,      # ✓ Valid
    123: "int_key",       # ✓ Valid (JSON.stringify will convert to "123")
    (1, 2): "tuple_key"   # ✗ TypeError in JSON serialization
}
```

**Constraint**: Dictionary keys must already be JSON-serializable (str, int). Complex keys (tuples, objects) will cause JSON serialization error later.

**Value Constraints**:
- Values are recursively processed
- No restriction on value types

**Empty Dict**:
```python
dataclass_to_dict({})  # Returns {}
```

### List/Tuple Constraints

**Type**: `list` or `tuple`

**Conversion**:
```python
dataclass_to_dict([1, 2, 3])  # [1, 2, 3]
dataclass_to_dict((1, 2, 3))  # [1, 2, 3] (tuple → list)
```

**Constraint**: Both list and tuple return list. Tuple information is lost.

**Nested Collections**:
```python
[(1, 2), (3, 4)]  # → [[1, 2], [3, 4]]
[[1, 2], [3, 4]]  # → [[1, 2], [3, 4]]
```

**Empty Collection**:
```python
dataclass_to_dict([])   # []
dataclass_to_dict(())   # []
```

### Primitive Constraints

**Types**: `str`, `int`, `float`, `bool`, `None`

**Constraint**: Pass through unchanged (already JSON-serializable)

**Edge Cases**:
```python
# Special float values
float('inf')   # Returns inf (JSON serialization may fail!)
float('-inf')  # Returns -inf
float('nan')   # Returns nan (JSON may convert to null)

# Very large int
10**100        # Returns 10000...000 (no limit in Python 3)

# Empty string
""             # Returns ""

# Boolean
True           # Returns True (JSON: true)
False          # Returns False (JSON: false)
```

**Constraint**: NaN, Inf, -Inf pass through. JSON serialization behavior:
```python
import json
json.dumps(float('inf'))   # Raises ValueError (unless allow_nan=True)
json.dumps(float('nan'))   # Raises ValueError (unless allow_nan=True)
```

---

## Output Constraints

### Return Type

**Type**: `dict | list | Any`

**Possible Returns**:
```python
dict         # From dataclass or dict input
list         # From list/tuple input
str          # From str, Path, or str() fallback
int          # From int input
float        # From float input (including NaN, Inf)
bool         # From bool input
None         # From None input
```

**Constraint**: Return type depends on input type. No single guaranteed output type.

### Dictionary Output

**Structure**:
```python
{
    "key1": value1,  # Recursively serialized
    "key2": value2,
    ...
}
```

**Constraints**:
- Keys are strings (from dataclass fields or dict keys)
- Values are recursively serialized
- Order preserves Python 3.7+ dict order
- No duplicate keys (dict property)

**Size**:
```python
# Minimum: empty dict
{}

# Maximum: Limited by memory
# Large dataclass with many fields (100+) is fine
```

### List Output

**Structure**:
```python
[item1, item2, ...]  # Recursively serialized items
```

**Constraints**:
- Items are recursively serialized
- Order preserved from input
- Can contain mixed types
- No size limit (memory permitting)

---

## Performance Constraints

### Time Complexity

**Overall**: O(n) where n = total number of objects in tree

**Breakdown by Type**:
```python
# Primitives, Path, Enum: O(1)
str, int, float, bool, None  # Immediate return
Path("/tmp")                 # O(1) string conversion
Status.ACTIVE                # O(1) value extraction

# Collections: O(n) where n = number of items
list, tuple, dict            # O(n) iteration + recursive processing

# Dataclass: O(f) where f = number of fields
@dataclass
class Config:
    field1: int  # O(1)
    field2: str  # O(1)
    # Total: O(f)
```

**Nested Structures**:
```python
# Tree with depth d and branching factor b
# Total nodes: 1 + b + b^2 + ... + b^d ≈ O(b^d)
# Time: O(b^d)

# Example: Dataclass with 10 fields, each with 10 nested dataclasses, 3 levels deep
# Nodes: 10^3 = 1000
# Time: O(1000) ≈ 1ms
```

**Typical Performance**:
```
Object Size      | Time (estimated)
-----------------|------------------
Small dataclass  | < 0.01 ms
(5 fields)       |
                 |
Medium dataclass | < 0.1 ms
(20 fields)      |
                 |
Large nested     | < 1 ms
(100+ fields)    |
                 |
Very large tree  | 1-10 ms
(1000+ objects)  |
```

**Constraint**: Must complete in < 100ms for typical use cases (templates, configs)

### Space Complexity

**Overall**: O(n) where n = total number of objects

**Memory Usage**:
```python
# Primitive: ~28-50 bytes (Python object overhead)
42  # int: ~28 bytes

# String: ~50 + len(str) bytes
"hello"  # ~55 bytes

# List: ~64 + 8*len bytes
[1, 2, 3]  # ~88 bytes + item sizes

# Dict: ~240 + ~100*len bytes (approximate)
{"key": "value"}  # ~340 bytes + item sizes

# Dataclass: Sum of field sizes + overhead
@dataclass
class Config:
    value: int  # ~28 bytes
# Config(42): ~200 bytes total
```

**Recursive Memory**:
```python
# For nested structures, memory accumulates
# Each recursion level adds stack frame (~1KB)

# Maximum recursion depth: sys.getrecursionlimit() ≈ 1000
# Stack size: ~1000 * 1KB = 1MB

# Constraint: Nested structures deeper than ~1000 will cause RecursionError
```

**Typical Memory**:
```
Object Size      | Memory (estimated)
-----------------|-------------------
Small config     | < 1 KB
Medium template  | 1-10 KB
Large template   | 10-100 KB
Very large tree  | 100KB - 1MB
```

**Constraint**: Must use < 10MB memory for typical use cases

### Recursion Depth Constraint

**Python Limit**: `sys.getrecursionlimit()` (default ~1000)

**Recursion Sources**:
```python
# 1. Nested dataclasses
@dataclass
class Node:
    child: 'Node'

# 100 levels deep: 100 recursion calls

# 2. Nested lists
[[[[[[...]]]]]]  # Each level adds recursion

# 3. Nested dicts
{"a": {"b": {"c": {...}}}}  # Each level adds recursion
```

**Constraint**: Maximum nesting depth ~500-800 (safe margin before hitting limit)

**Edge Case**:
```python
# Pathological case: Very deep nesting
deep_list = []
current = deep_list
for _ in range(1500):
    new_list = []
    current.append(new_list)
    current = new_list

dataclass_to_dict(deep_list)
# RecursionError: maximum recursion depth exceeded
```

**Recommendation**: Typical OMR templates have nesting depth < 10. No concern for practical use.

---

## Error Handling Constraints

### No Explicit Exceptions

**Guarantee**: `dataclass_to_dict()` does NOT raise exceptions for most inputs

**Exception Sources**:
```python
# 1. Circular references (from asdict())
# RecursionError

# 2. Very deep nesting (from recursion)
# RecursionError

# 3. Memory exhaustion (from very large objects)
# MemoryError
```

**Constraint**: Function does not validate input or raise custom exceptions. Errors come from Python internals.

### Fallback Behavior

**Unknown Type Handling**:
```python
# Unknown type → try str()
try:
    return str(obj)
except Exception:
    return obj  # Return as-is
```

**Constraint**: Even if `str()` fails (rare), function returns original object. May cause downstream JSON serialization error.

**Example**:
```python
class BadClass:
    def __str__(self):
        raise RuntimeError("Can't stringify!")

obj = BadClass()
result = dataclass_to_dict(obj)  # Returns obj as-is (no exception)

# Later:
json.dumps(result)  # TypeError: Object of type BadClass is not JSON serializable
```

### JSON Serialization Constraints

**dataclass_to_dict() Output → JSON**:

```python
import json

# Valid outputs (JSON serializable)
json.dumps(42)                # "42"
json.dumps("hello")           # "\"hello\""
json.dumps([1, 2, 3])         # "[1, 2, 3]"
json.dumps({"key": "value"})  # "{\"key\": \"value\"}"
json.dumps(None)              # "null"
json.dumps(True)              # "true"

# Invalid outputs (not JSON serializable)
json.dumps(float('inf'))      # ValueError (unless allow_nan=True)
json.dumps(float('nan'))      # ValueError (unless allow_nan=True)
json.dumps(SomeClass())       # TypeError
```

**Constraint**: `dataclass_to_dict()` does NOT guarantee JSON-serializable output. Caller must handle JSON errors.

**Recommendation**:
```python
# Safe JSON serialization
try:
    serialized = dataclass_to_dict(obj)
    json_string = json.dumps(serialized)
except (ValueError, TypeError) as e:
    # Handle serialization error
    logger.error(f"JSON serialization failed: {e}")
```

---

## Browser Migration Constraints

### JavaScript Number Limits

**Python float vs JavaScript Number**:
```python
# Python: arbitrary precision integers
10**100  # 100000000000...000 (exact)

# JavaScript: IEEE 754 double precision
Number.MAX_SAFE_INTEGER  # 2^53 - 1 = 9007199254740991

# Integers larger than MAX_SAFE_INTEGER lose precision
10**100 in JS  # 1e+100 (approximation)
```

**Constraint**: Very large integers (> 2^53) lose precision in JavaScript.

**Solution**:
```javascript
// Use BigInt for large integers
const bigNum = BigInt("10000000000000000000");

// Or convert to string
const largeInt = "10000000000000000000";
```

### JavaScript Type Differences

**No Path Type**:
```javascript
// Python: Path objects
from pathlib import Path
Path("/tmp/test")  # Path object → serialized to "/tmp/test"

// JavaScript: Strings already
const filePath = "/tmp/test";  // Already a string, no conversion needed

// For File API:
const file = new File([], "test.txt");
file.name  // "test.txt" (already string)
```

**No Enum Type (native)**:
```javascript
// Python: Enum
class Status(Enum):
    ACTIVE = "active"

Status.ACTIVE.value  # "active"

// JavaScript: Use objects or TypeScript enums
const Status = {
    ACTIVE: "active",
    PENDING: "pending"
};

// Or TypeScript enum
enum Status {
    ACTIVE = "active",
    PENDING = "pending"
}

// Access
Status.ACTIVE  // "active" (already the value)
```

**No Dataclass Type**:
```javascript
// Python: Dataclasses with asdict()
@dataclass
class Config:
    value: int

asdict(Config(42))  # {"value": 42}

// JavaScript: Plain objects already dict-like
const config = { value: 42 };  // Already serializable

// Or TypeScript interfaces
interface Config {
    value: number;
}

const config: Config = { value: 42 };
```

### JSON API Differences

**Python json module**:
```python
import json

# Serialization
json.dumps(obj)  # → str
json.dump(obj, file)  # → write to file

# Options
json.dumps(obj, indent=2)  # Pretty print
json.dumps(obj, allow_nan=True)  # Allow NaN/Inf
```

**JavaScript JSON object**:
```javascript
// Serialization
JSON.stringify(obj);  // → string
JSON.stringify(obj, null, 2);  // Pretty print (indent=2)

// Custom serialization
JSON.stringify(obj, (key, value) => {
    // Custom replacer function
    if (value instanceof Date) {
        return value.toISOString();
    }
    return value;
});

// Or use toJSON method
obj.toJSON = function() {
    return dataclassToDict(this);
};
JSON.stringify(obj);  // Calls toJSON automatically
```

**Constraint**: JavaScript JSON.stringify() does NOT have `allow_nan` option. NaN/Inf automatically convert to `null`.

### Performance Comparison

**Python**:
```python
# Typical performance
100 objects: ~0.1ms
1000 objects: ~1ms
```

**JavaScript (Browser)**:
```javascript
// Typical performance (similar to Python)
100 objects: ~0.1ms
1000 objects: ~1ms

// V8 optimization helps
// Recursive function calls are fast
```

**Constraint**: Performance is comparable. No significant bottleneck in browser.

---

## Validation Constraints

### No Input Validation

**Current Implementation**: No type checking, no validation

```python
# No check for:
dataclass_to_dict(float('nan'))  # NaN passes through
dataclass_to_dict(float('inf'))  # Inf passes through
dataclass_to_dict(circular_ref)  # Causes RecursionError (not caught)
```

**Constraint**: Caller must ensure input is valid.

**Recommendation for Browser**:
```typescript
// Add TypeScript type guards
function isSerializable(obj: any): boolean {
    if (obj === null || obj === undefined) return true;
    if (typeof obj === 'string') return true;
    if (typeof obj === 'number') {
        return isFinite(obj);  // Reject NaN, Inf
    }
    if (typeof obj === 'boolean') return true;
    if (Array.isArray(obj)) {
        return obj.every(isSerializable);
    }
    if (typeof obj === 'object') {
        return Object.values(obj).every(isSerializable);
    }
    return false;
}

// Use before serialization
if (!isSerializable(obj)) {
    throw new Error('Object contains non-serializable values');
}
```

### No Output Validation

**Current Implementation**: Returns result without validation

```python
result = dataclass_to_dict(obj)
# No check if result is JSON-serializable
```

**Constraint**: Caller must validate before JSON serialization.

**Recommendation**:
```python
# Add validation step
import json

def safe_dataclass_to_dict(obj):
    result = dataclass_to_dict(obj)
    # Validate by attempting JSON serialization
    try:
        json.dumps(result)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Serialization produced non-JSON-serializable output: {e}")
    return result
```

---

## Concurrency Constraints

### Thread Safety

**Status**: Fully thread-safe

**Reasoning**:
```python
def dataclass_to_dict(obj: Any) -> dict | list | Any:
    # No global state
    # No instance variables (if used as function)
    # No mutation of input
    # All data is local to function call
    pass
```

**Constraint**: Multiple threads can call `dataclass_to_dict()` concurrently without issues.

**Browser Implication**:
```javascript
// Can use in Web Workers
// Each worker can serialize independently

// Main thread
const worker = new Worker('serializer.js');
worker.postMessage({ type: 'serialize', data: obj });

// Worker thread (serializer.js)
self.onmessage = (e) => {
    if (e.data.type === 'serialize') {
        const result = dataclassToDict(e.data.data);
        self.postMessage({ type: 'result', data: result });
    }
};
```

---

## Summary of Critical Constraints

| Constraint | Value/Rule | Impact |
|------------|-----------|---------|
| Input type | `Any` | No restrictions, may fail later |
| Circular references | Not handled | RecursionError |
| Max recursion depth | ~500-800 | Very deep nesting fails |
| Output type | `dict \| list \| Any` | Depends on input |
| JSON compatibility | Not guaranteed | Caller must validate |
| NaN/Inf handling | Pass through | May break JSON serialization |
| Time complexity | O(n) | Linear in object count |
| Space complexity | O(n) | Linear in object count |
| Thread safety | Yes | No shared state |
| Exceptions | Minimal | Only RecursionError/MemoryError |
| Browser equivalent | Plain objects | Use recursive function |
| TypeScript support | Recommended | Add type guards |

---

## Related Constraints

- **Parsing Utils**: `../parsing/constraints.md` (reverse operation)
- **Config Management**: `../../../foundation/configuration.md`
- **Template Entity**: `../../template/concept.md`
- **File Utils**: `../file/constraints.md`
