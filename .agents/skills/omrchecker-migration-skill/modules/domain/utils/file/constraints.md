# File Utils Constraints

**Module**: Domain - Utils - File
**Python Reference**: `src/utils/file.py`
**Last Updated**: 2026-02-21

---

## Component 1: JSON Loading (load_json)

### Input Constraints

#### path Parameter

**Type**: `str` or `Path`
**Valid Values**: Any string representing a file path

**Format Requirements**:
```python
# Absolute path (recommended)
"/Users/name/OMRChecker/template.json"
"C:\\Users\\name\\OMRChecker\\template.json"

# Relative path (relative to cwd)
"template.json"
"../templates/template.json"

# Path object
Path("template.json")
```

**Constraints**:
- Must point to existing file
- Must have read permissions
- File must contain valid UTF-8 text (default encoding)
- No maximum path length enforced (OS-dependent)

**Platform Differences**:
```python
# Windows
path = "C:\\OMRChecker\\template.json"  # Valid
path = "C:/OMRChecker/template.json"    # Also valid (Python handles it)

# Unix/Linux/Mac
path = "/home/user/OMRChecker/template.json"  # Valid

# Both
path = Path("template.json")  # Platform-independent
```

#### rest Parameter (Keyword Arguments)

**Type**: `dict` (passed to `json.load()`)
**Valid Keys**: Any valid json.load() parameter

**Common Options**:
```python
# Custom object hook
load_json("data.json", object_hook=custom_decoder)

# Custom float parser
load_json("data.json", parse_float=decimal.Decimal)

# Strict mode (default True)
load_json("data.json", strict=False)  # Allow control chars

# Custom integer parser
load_json("data.json", parse_int=custom_int_parser)
```

**Constraints**:
- Must be valid json.load() parameters
- No validation performed on these parameters

### Output Constraints

#### Return Value

**Type**: `dict[str, Any]`
**Guarantee**: Always returns a dict (or raises exception)

**Constraints**:
```python
# JSON must have object at root
{"key": "value"}  # Valid
[1, 2, 3]         # Invalid (array at root)
"string"          # Invalid (string at root)
123               # Invalid (number at root)
```

**If Root is Not Object**:
```python
# JSON file: [1, 2, 3]
result = load_json("array.json")
# Returns: [1, 2, 3]
# Type: list, not dict
# Type hint is incorrect for this case!
```

**Actual Behavior**: Returns whatever JSON root is (dict, list, str, etc.)
**Type Hint Says**: `dict[str, Any]` (not enforced at runtime)

**Recommendation**: Always ensure JSON files have object at root for OMRChecker

### Exception Constraints

#### InputFileNotFoundError

**Raised When**: File doesn't exist
**Parameters**:
- `path`: Path object
- `file_type`: "JSON"

**Example**:
```python
try:
    load_json("missing.json")
except InputFileNotFoundError as e:
    # e.path = Path("missing.json")
    # str(e) = "JSON file not found at: missing.json"
    pass
```

#### ConfigLoadError

**Raised When**: JSON parsing fails
**Parameters**:
- `path`: Path object
- `message`: Error details

**Common Causes**:
```python
# Syntax error
{"key": value}  # Missing quotes → JSONDecodeError

# Trailing comma
{"key": "value",}  # Invalid JSON (valid in some parsers)

# Comments
{"key": "value"}  // comment  # JSON doesn't support comments

# Single quotes
{'key': 'value'}  # JSON requires double quotes
```

**Example**:
```python
try:
    load_json("malformed.json")
except ConfigLoadError as e:
    # e.path = Path("malformed.json")
    # e.message = "Invalid JSON format: Expecting value: line 1 column 10"
    pass
```

#### UnicodeDecodeError (Potential)

**Raised When**: File encoding is not UTF-8
**Current Handling**: Not explicitly caught (propagates up)

**TODO**: Code comment mentions need for non-UTF character handling

**Example**:
```python
# File encoded in Latin-1 with special chars
try:
    load_json("latin1.json")
except UnicodeDecodeError as e:
    # Not caught by load_json, caller must handle
    pass
```

### Performance Constraints

#### Time Complexity

**Complexity**: O(n) where n = file size in bytes
**Breakdown**:
```python
Path(path).exists()     # O(1) - filesystem check
Path.open(path)         # O(1) - open file descriptor
json.load(f)            # O(n) - parse entire file
```

**Typical Performance**:
```
File Size | Parse Time
----------|------------
1 KB      | < 1 ms
10 KB     | ~2 ms
100 KB    | ~15 ms
1 MB      | ~150 ms
10 MB     | ~1.5 s
```

**Large File Warning**: JSON parsing is memory-intensive
- Entire file loaded into memory
- Parsed structure also in memory
- Peak memory: ~3-5x file size

#### Memory Constraints

**Memory Usage**: O(n) where n = file size

**Breakdown**:
```python
# File on disk: 1 MB
# Read into string: 1 MB
# Parsed structure: 2-4 MB (depends on nesting/duplication)
# Peak memory: ~5 MB
```

**Constraints**:
- No streaming JSON parser (loads entire file)
- No size limit enforced
- Python dict overhead: ~240 bytes per dict + keys/values

**Large File Handling**:
```python
# For very large JSON files (> 100 MB), consider:
# 1. Use ijson for streaming (not implemented)
# 2. Split JSON into smaller files
# 3. Use alternative format (msgpack, pickle)
```

### Thread Safety

**Status**: Thread-safe
**Reasoning**:
- No shared state
- Each call operates on independent file handle
- json.load() is thread-safe

**Concurrent Usage**:
```python
# Multiple threads can call load_json() concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(load_json, f"template_{i}.json")
        for i in range(10)
    ]
    results = [f.result() for f in futures]
```

**Constraint**: Same file can be read by multiple threads (read-only)

---

## Component 2: PathUtils Static Methods

### remove_non_utf_characters()

#### Input Constraints

**Type**: `str`
**Length**: 0 to unlimited
**Characters**: Any Unicode characters

**Valid Inputs**:
```python
""                           # Empty string - valid
"normal_path.txt"           # ASCII - valid
"path/with\x00null.txt"     # Contains null byte - valid
"file\u2019.txt"            # Unicode apostrophe - valid
"C:\\Windows\\file.txt"     # Backslashes - valid
```

**No Validation**: Accepts any string input

#### Printable Characters Set

**Definition**: `string.printable`
**Size**: 100 characters
**Contents**:
```python
import string
string.printable = (
    string.digits +      # '0123456789'
    string.ascii_letters +  # 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    string.punctuation + # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    string.whitespace    # ' \t\n\r\x0b\x0c'
)
```

**Included**:
- ASCII letters: a-z, A-Z
- Digits: 0-9
- Punctuation: All ASCII punctuation
- Whitespace: space, tab, newline, carriage return, vertical tab, form feed

**Excluded**:
- Non-ASCII characters (é, ñ, 中文, emoji, etc.)
- Control characters beyond whitespace
- Zero-width characters
- Unicode variants

**Examples**:
```python
# Kept
"file_01.txt" → "file_01.txt"
"C:\\Users\\name" → "C:\\Users\\name"

# Removed
"file\x00.txt" → "file.txt"  # Null byte removed
"café.txt" → "caf.txt"        # é removed
"file\u2019.txt" → "file.txt" # Smart quote removed
"文件.txt" → ".txt"             # Chinese chars removed
```

#### Output Constraints

**Type**: `str`
**Length**: 0 to len(input)
**Guarantee**: Only contains printable ASCII

**Constraints**:
```python
output_len <= input_len  # Can only remove, never add
all(c in string.printable for c in output)  # All chars printable
```

**Edge Cases**:
```python
# Empty input
remove_non_utf_characters("") → ""

# All non-printable
remove_non_utf_characters("\x00\x01\x02") → ""

# Mixed
remove_non_utf_characters("file\x00name.txt") → "filename.txt"
```

#### Performance Constraints

**Time Complexity**: O(n) where n = string length
**Space Complexity**: O(n) for new string

**Breakdown**:
```python
"".join(x for x in path_string if x in PathUtils.printable_chars)
# For each char (n iterations):
#   - Check membership in set: O(1)
#   - Join: O(n)
# Total: O(n)
```

**Typical Performance**:
```
String Length | Time
--------------|--------
10 chars      | < 1 μs
100 chars     | < 5 μs
1000 chars    | < 50 μs
```

**Constraint**: Should complete in < 1 ms for typical paths (< 500 chars)

### sep_based_posix_path()

#### Input Constraints

**Type**: `str`
**Format**: Any path string (Unix or Windows style)

**Valid Inputs**:
```python
# Unix paths
"/home/user/file.txt"
"../relative/path.txt"
"./current/file.txt"

# Windows paths
"C:\\Users\\name\\file.txt"
"D:/mixed/slashes\\file.txt"

# UNC paths
"\\\\server\\share\\file.txt"

# Mixed
"folder\\subfolder/file.txt"
```

**Constraints**:
- No length limit
- No validation of path validity
- Accepts any string (not required to be valid path)

#### Normalization Behavior

**os.path.normpath()**:

```python
# Collapse redundant separators
"a//b//c" → "a/b/c" (Unix)
"a\\\\b\\\\c" → "a\\b\\c" (Windows)

# Resolve up-level references
"a/b/../c" → "a/c"
"a/./b" → "a/b"

# Remove trailing slashes
"a/b/c/" → "a/b/c"

# Platform-specific
# Unix: uses /
# Windows: uses \\
```

**Constraints**:
- Output depends on OS (uses `os.path.sep`)
- Absolute/relative preserved
- Cannot resolve symlinks (use `os.path.realpath` for that)

#### POSIX Conversion

**PureWindowsPath.as_posix()**:

```python
# Converts backslashes to forward slashes
"C:\\Users\\name" → "C:/Users/name"

# Preserves drive letters
"D:\\folder\\file.txt" → "D:/folder/file.txt"

# UNC paths
"\\\\server\\share" → "//server/share"
```

**Trigger Conditions**:
```python
if os.path.sep == "\\" or "\\" in path:
    # Convert to POSIX
```

**Constraint**: Conversion happens if:
1. Running on Windows (`os.path.sep == "\\"`) OR
2. Path contains backslashes (even on Unix)

**Edge Case**: On Unix with backslash in filename:
```python
# Unix allows backslash in filenames
path = "folder/file\\with\\backslash.txt"
# Will be converted: "folder/file/with/backslash.txt"
# This changes the meaning! Backslash was part of filename, not separator
```

#### Output Constraints

**Type**: `str`
**Format**: POSIX-style path (forward slashes only)
**Character Set**: Printable ASCII only

**Guarantees**:
```python
# No backslashes
assert "\\" not in output

# Only printable ASCII
assert all(c in string.printable for c in output)

# Normalized
assert "//" not in output  # No double slashes (after normpath)
```

**Examples**:
```python
# Windows → POSIX
"C:\\OMRChecker\\output\\file.jpg"
→ "C:/OMRChecker/output/file.jpg"

# Mixed separators
"folder\\subfolder/file.txt"
→ "folder/subfolder/file.txt"

# With non-UTF chars
"folder\x00\\file\u2019.txt"
→ "folder/file.txt"

# Already POSIX (Unix)
"/home/user/file.txt"
→ "/home/user/file.txt"
```

#### Performance Constraints

**Time Complexity**: O(n) where n = path length

**Breakdown**:
```python
os.path.normpath(path)                    # O(n)
PureWindowsPath(path).as_posix()         # O(n)
remove_non_utf_characters(path)           # O(n)
# Total: O(n)
```

**Typical Performance**:
```
Path Length | Time
------------|--------
50 chars    | < 10 μs
200 chars   | < 30 μs
500 chars   | < 70 μs
```

**Constraint**: Must complete in < 100 μs for typical paths

---

## Component 3: PathUtils Directory Management

### Initialization Constraints

#### output_dir Parameter

**Type**: `Path`
**Requirement**: Must be Path object (not string)

**Valid Inputs**:
```python
Path("/absolute/path")
Path("relative/path")
Path(".")  # Current directory
```

**Constraints**:
- Does not validate existence
- Does not create directory
- Does not check permissions
- Just stores reference

**Directory Structure**:
```python
# All subdirectories are defined relative to output_dir
# Total: 8 top-level attributes
self.output_dir          # Base
self.save_marked_dir     # CheckedOMRs/
self.image_metrics_dir   # ImageMetrics/
self.results_dir         # Results/
self.manual_dir          # Manual/
self.errors_dir          # Manual/ErrorFiles/
self.multi_marked_dir    # Manual/MultiMarkedFiles/
self.evaluations_dir     # Evaluations/
self.debug_dir           # Debug/
```

**Constraint**: All paths are relative to `output_dir`

### create_output_directories() Constraints

#### Behavior Constraints

**Idempotency**: Can be called multiple times safely
```python
path_utils = PathUtils(output_dir)
path_utils.create_output_directories()  # Creates directories
path_utils.create_output_directories()  # No-op (already exist)
path_utils.create_output_directories()  # Still safe
```

**No Cleanup**: Never deletes existing directories or files

**Directory Creation Order**:
```
1. save_marked_dir (with parents=True)
2. save_marked_dir subdirectories (without parents)
3. Image bucket directories (with parents=True)
4. Non-image directories (with parents=True)
```

**Constraint**: Uses `parents=True` selectively
- Main directories: `parents=True` (creates intermediate)
- Subdirectories: No `parents` (assumes parent exists)

#### Total Directories Created

**Count**: Up to 15 directories

**List**:
```
1. CheckedOMRs/
2. CheckedOMRs/colored/
3. CheckedOMRs/stack/
4. CheckedOMRs/stack/colored/
5. CheckedOMRs/_MULTI_/
6. CheckedOMRs/_MULTI_/colored/
7. Manual/
8. Manual/ErrorFiles/
9. Manual/ErrorFiles/colored/
10. Manual/MultiMarkedFiles/
11. Manual/MultiMarkedFiles/colored/
12. Results/
13. ImageMetrics/
14. Evaluations/
15. Debug/ (defined but not created by this method!)
```

**Note**: `debug_dir` is defined in `__init__` but NOT created by `create_output_directories()`

#### Permission Constraints

**Required Permissions**:
- Write permission on `output_dir`
- Ability to create subdirectories

**Failure Modes**:
```python
# No write permission
# OSError: [Errno 13] Permission denied

# Disk full
# OSError: [Errno 28] No space left on device

# Invalid path (e.g., file exists with same name)
# FileExistsError: [Errno 17] File exists (if name conflicts with file)
```

**No Error Handling**: Exceptions propagate to caller

#### Performance Constraints

**Time Complexity**: O(d) where d = number of directories to create
**Typical Time**: 10-50 ms (depends on filesystem)

**Breakdown**:
```python
# Each mkdir: ~1-5 ms (filesystem dependent)
# 15 directories: ~15-75 ms
# exists() checks: ~0.1 ms each
```

**Constraint**: Should complete in < 100 ms on typical systems

**Optimization**: Uses `exists()` to skip creation
- Faster when directories already exist
- First run: slower (creates all)
- Subsequent runs: very fast (all exist checks return True)

---

## Component 4: SaveImageOps Constraints

### Initialization Constraints

#### tuning_config Parameter

**Type**: TuningConfig object
**Required Attributes**:
```python
tuning_config.outputs.save_image_level  # int (0-6)
tuning_config.outputs.display_image_dimensions  # tuple[int, int]
```

**save_image_level**:
- **Type**: `int`
- **Range**: 0 (no debug images) to 6 (all debug images)
- **Default**: Typically 2 or 3
- **Constraint**: Must be non-negative integer

**display_image_dimensions**:
- **Type**: `tuple[int, int]`
- **Format**: `(height, width)` in pixels
- **Typical**: `(400, 400)` or `(600, 600)`
- **Constraint**: Both values must be positive integers

#### Storage Initialization

**Data Structure**:
```python
self.gray_images = defaultdict(list)
# Type: defaultdict[int, list[list[str, np.ndarray]]]
# Keys: 1-6 (save levels)
# Values: [[title1, image1], [title2, image2], ...]
```

**Constraint**: Uses defaultdict to auto-create empty lists
- No need to pre-initialize keys
- Accessing non-existent key returns `[]`

**Memory**: Initially ~1 KB (empty defaultdicts)

### append_save_image() Constraints

#### title Parameter

**Type**: `str` (strictly enforced)
**Validation**: Type check with exception

```python
if not isinstance(title, str):
    raise TypeError(f"title={title} is not a string")
```

**Valid Titles**:
```python
"Preprocessed Image"         # Valid
"After Threshold"            # Valid
"Bubble Detection - Pass 1"  # Valid
""                           # Valid (empty string allowed)
```

**Invalid Titles**:
```python
None                  # TypeError
123                   # TypeError
["title"]             # TypeError
b"bytes"              # TypeError
```

**Constraint**: Must be string type, content not validated

#### keys Parameter

**Type**: `int` or `list[int]`
**Auto-Conversion**: Single int converted to list

```python
keys = 2       # Converted to [2]
keys = [2, 3]  # Already list, used as-is
```

**Valid Values**:
```python
keys = 1           # Valid
keys = [1, 2, 3]   # Valid
keys = [1, 1, 1]   # Valid (duplicates allowed, causes duplicate storage)
keys = []          # Valid (no-op, nothing stored)
```

**Constraint**: Values should be 1-6, but not validated
- Values > `save_image_level` are skipped
- Values < 1 or > 6 are stored but never used

#### Image Parameters

**Type**: `np.ndarray` or `None`
**Formats**:
- **gray_image**: 2D array `(H, W)` or 3D array `(H, W, 1)`
- **colored_image**: 3D array `(H, W, 3)` in BGR format

**Constraints**:
- No shape validation
- No dtype validation
- No format validation (assumed to be compatible with cv2)

**Image Copying**:
```python
if gray_image is not None:
    gray_image_copy = gray_image.copy()
```

**Purpose**: Prevent mutations
- Original image can be modified without affecting stored copy
- Adds memory overhead (~image_size bytes per copy)

**Memory Constraint**: Each image copy takes full memory
- 640×480 grayscale: ~300 KB
- 640×480 color: ~900 KB
- 1920×1080 color: ~6 MB

**Storage Decision**:
```python
for key in keys:
    if int(key) > self.save_image_level:
        continue  # Skip this key
```

**Constraint**: Only stores if `key <= save_image_level`
- Allows specifying higher levels for future use
- No error if key is too high, just skipped

### save_image_stacks() Constraints

#### file_path Parameter

**Type**: `Path` object
**Required Attribute**: `.stem` (filename without extension)

**Example**:
```python
file_path = Path("sheet_01.jpg")
stem = file_path.stem  # "sheet_01"
```

**Constraint**: Must be Path object with valid stem

#### save_marked_dir Parameter

**Type**: `Path` or `str`
**Usage**: String formatted into save path

**Constraint**: Not validated, assumed to exist or be creatable

#### key Parameter

**Type**: `int` or `None`
**Default**: `self.save_image_level` if None

**Constraint**: Should be 1-6, not validated

#### images_per_row Parameter

**Type**: `int`
**Default**: 4
**Constraint**: Must be positive (not validated)

**Effect on Layout**:
```python
images_per_row = 3
# 7 images →
# Row 1: [img1, img2, img3]
# Row 2: [img4, img5, img6]
# Row 3: [img7]
```

**Recommendation**: 2-5 for readability

#### File Saving Constraints

**File Paths Generated**:
```python
# Gray stack
f"{save_marked_dir}/stack/{stem}_{key}_stack.jpg"

# Colored stack
f"{save_marked_dir}/stack/colored/{stem}_{key}_stack.jpg"
```

**Constraint**: Directories must exist
- `save_marked_dir/stack/`
- `save_marked_dir/stack/colored/`

**No Error Handling**: If directories don't exist, ImageUtils.save_img() will fail

**Overwrite Behavior**: Overwrites existing files without warning

### get_result_hstack() Constraints

#### Input Constraints

**titles_and_images**:
- **Type**: `list[list[str, np.ndarray]]`
- **Length**: 1 to unlimited
- **Constraint**: Must have at least 1 image

**images_per_row**:
- **Type**: `int`
- **Range**: 1 to unlimited
- **Constraint**: Must be positive

#### Processing Constraints

**Resize Operations**:
```python
images = ImageUtils.resize_multiple([image for _, image in ...], display_width)
```

**Constraint**: All images resized to same width
- Height adjusted to maintain aspect ratio
- May result in rows of different heights

**Grid Chunking**:
```python
grid_images = MathUtils.chunks(images, images_per_row)
```

**Constraint**: Last row may have fewer images
```
images_per_row = 4, total = 7
→ [[img1-4], [img5-7]]  # Second row has 3 images
```

**Vertical Stacking**:
```python
result = ImageUtils.get_vstack_image_grid(grid_images)
```

**Constraint**: Depends on ImageUtils implementation
- Assumes all images in same row are same width
- May add padding for uneven rows

**Final Resize**:
```python
final_width = min(
    len(titles_and_images) * display_width // 3,
    int(display_width * 2.5)
)
```

**Calculation**:
- **Option 1**: `num_images * display_width / 3`
  - More images → wider output
- **Option 2**: `display_width * 2.5` (cap)
  - Maximum width limit

**Constraint**: Final width is capped
- Prevents excessively wide images
- Typical range: 200-1000 pixels

#### Output Constraints

**Type**: `np.ndarray`
**Shape**: `(H, W, C)` where:
- H: Variable (depends on number of rows and image heights)
- W: `final_width`
- C: 3 (BGR) or 1 (grayscale)

**Memory**: Depends on final dimensions
```python
# Example: 1000 × 800 × 3 (colored)
memory = 1000 * 800 * 3 = 2.4 MB
```

### reset_all_save_img() Constraints

**Behavior**: Clears all stored images

**Levels Cleared**: 1 through 7
```python
for i in range(7):
    self.gray_images[i + 1] = []   # Clears levels 1-7
    self.colored_images[i + 1] = []
```

**Note**: Clears level 7 even though max level is 6
- Defensive programming
- Ensures complete cleanup

**Memory**: Releases references to all stored image arrays
- Python garbage collector will free memory
- May take time for large collections

**Constraint**: Must be called between processing different files
- Prevents accumulation of images
- Prevents mixing images from different files in same stack

---

## Browser Migration Constraints

### File API Constraints

**No Filesystem Access**:
- Cannot use `Path` library
- Cannot create directories
- Cannot check file existence before open

**File Object Constraints**:
```javascript
// File from <input type="file">
file.name      // Filename only, no path
file.size      // Size in bytes
file.type      // MIME type
file.text()    // Async method, returns Promise<string>
```

**Security Restrictions**:
- No access to file paths outside user selection
- No write access to filesystem
- Must use downloads or IndexedDB for output

### JSON Loading in Browser

**Async Requirement**: All file operations are async
```javascript
// Python: synchronous
data = load_json("file.json")

// Browser: async
const data = await loadJson(file);
```

**Error Handling**:
```javascript
try {
  const data = await loadJson(file);
} catch (error) {
  if (error instanceof InputFileNotFoundError) {
    // File selection cancelled
  } else if (error instanceof ConfigLoadError) {
    // Invalid JSON
  }
}
```

### Path Handling in Browser

**No Real Paths**: All paths are virtual strings
```javascript
// Not actual filesystem paths, just organizational strings
const virtualPath = "CheckedOMRs/stack/sheet_01_2_stack.jpg";
```

**Download Constraints**:
```javascript
// Browser can only download with flat filenames
// Virtual path must be flattened or zipped
downloadFile("CheckedOMRs_stack_sheet_01_2_stack.jpg", blob);

// Or use ZIP to preserve structure
zip.file("CheckedOMRs/stack/sheet_01_2_stack.jpg", blob);
```

### Image Storage Constraints

**ImageData vs NumPy**:
```javascript
// Python: numpy array
image = np.array([[0, 128, 255], ...])  # Direct memory

// Browser: ImageData
imageData = new ImageData(width, height);
// imageData.data is Uint8ClampedArray (RGBA only, no grayscale)
```

**Format Differences**:
- Python: Supports grayscale (H, W) and BGR (H, W, 3)
- Browser: Only RGBA (width, height, 4 channels)
- Conversion required for grayscale

**Memory Constraints**:
```javascript
// Browser has stricter memory limits
// Typical: 1-2 GB for entire page
// Must be careful with image accumulation

// Python: Can handle 10+ GB easily
// Browser: Must release ImageData explicitly
```

### Performance Constraints in Browser

**Slower Operations**:
- Canvas operations: 2-10× slower than OpenCV/NumPy
- JSON parsing: Similar performance
- File I/O: Async overhead

**Recommended Optimizations**:
1. Use Web Workers for heavy processing
2. Process images in batches
3. Release memory explicitly (set to null)
4. Use OffscreenCanvas when possible

---

## Summary of Critical Constraints

| Component | Constraint | Impact |
|-----------|-----------|---------|
| load_json | File must exist | Raises InputFileNotFoundError |
| load_json | Must be valid JSON | Raises ConfigLoadError |
| load_json | No streaming | Large files load entirely into memory |
| PathUtils.printable_chars | Only ASCII printable | Non-ASCII chars removed from paths |
| PathUtils.posix_path | Converts backslashes | Changes Windows paths |
| create_directories | No error handling | Exceptions propagate |
| SaveImageOps.title | Must be string | TypeError if not |
| SaveImageOps.keys | Auto-converts int | Single int → list |
| SaveImageOps images | Copied | Memory overhead |
| save_image_level | Filters storage | Only stores if key <= level |
| reset_all_save_img | Clears 1-7 | Level 7 cleared (even though unused) |
| Browser | All async | Must use await/Promises |
| Browser | No filesystem | Must use downloads/IndexedDB |
| Browser | RGBA only | No native grayscale support |

---

## Related Constraints

- **Image Utils**: `../image/constraints.md`
- **Error Handling**: `../../../foundation/error-handling.md`
- **File System Patterns**: `../../../technical/filesystem/filesystem-operations.md`
- **Browser File API**: `../../../migration/browser-adaptations.md`
