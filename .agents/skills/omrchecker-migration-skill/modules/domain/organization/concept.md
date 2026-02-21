# File Organization - Core Concepts

## Overview

The **File Organization System** is a powerful post-processing feature that automatically organizes processed OMR files into custom directory structures using dynamic patterns. It runs AFTER all detection and evaluation is complete, grouping files based on detected OMR responses (roll numbers, booklet codes, scores, etc.) without modifying the original `CheckedOMRs/` output.

## What is File Organization?

File Organization is the process of creating an organized view of processed OMR files by:
- Creating symbolic links or copies from `CheckedOMRs/` to custom folders
- Using dynamic patterns with field placeholders (e.g., `booklet_{code}/roll_{roll}.jpg`)
- Matching files to rules based on regex patterns and priorities
- Handling collisions with configurable strategies

**Key Insight**: The system operates on **processed files**, using detected OMR values to determine file placement. It's a **post-processing step** that runs sequentially after all parallel processing completes.

## Why is File Organization Needed?

### Common Use Cases
- **Booklet-based Sorting**: Group answer sheets by question paper code (A/B/C/D)
- **Score-based Categorization**: Separate high/medium/low scoring sheets
- **Multi-marked Detection**: Isolate sheets with multiple marks for manual review
- **Roll Number Organization**: Create hierarchical structures like `batch_1/section_A/roll_12345.jpg`
- **Quality Control**: Group files by confidence metrics or detection errors

### Without File Organization
```
outputs/CheckedOMRs/
  ├── image1_CHECKED.jpg
  ├── image2_CHECKED.jpg
  ├── image3_CHECKED.jpg
  └── image4_CHECKED.jpg
```

### With File Organization
```
outputs/
  ├── CheckedOMRs/              # Original files (preserved)
  │   ├── image1_CHECKED.jpg
  │   ├── image2_CHECKED.jpg
  │   ├── image3_CHECKED.jpg
  │   └── image4_CHECKED.jpg
  └── organized/                # Organized view (symlinks/copies)
      ├── booklet_A/
      │   ├── roll_12345.jpg → ../../CheckedOMRs/image1_CHECKED.jpg
      │   └── roll_12346.jpg → ../../CheckedOMRs/image3_CHECKED.jpg
      └── booklet_B/
          ├── roll_12347.jpg → ../../CheckedOMRs/image2_CHECKED.jpg
          └── roll_12348.jpg → ../../CheckedOMRs/image4_CHECKED.jpg
```

## Core Architecture

### Entry Point
```python
# src/processors/organization/processor.py
class FileOrganizerProcessor(Processor):
    """Processor that organizes files into folders with dynamic patterns.

    CONCURRENCY-SAFE: Runs AFTER all parallel processing completes.
    Creates symlinks/copies with dynamic names using FilePatternResolver.
    """
```

### Primary Components

1. **FileOrganizerProcessor** - Main orchestrator
2. **FilePatternResolver** - Pattern formatting and collision handling
3. **GroupingRule** - Individual organization rule
4. **FileGroupingConfig** - Configuration container

### Two-Phase Operation

```
Phase 1: Collection (Parallel Processing)
    ↓ For each file (thread-safe):
    process() → Collect context + results

Phase 2: Organization (Sequential)
    ↓ After all processing:
    finish_processing_directory() → Organize all files
```

## Configuration Schema

### Template-Level Config
```json
{
  "outputs": {
    "fileGrouping": {
      "enabled": true,
      "defaultPattern": "ungrouped/{original_name}",
      "rules": [
        {
          "name": "Sort by Booklet",
          "priority": 1,
          "destinationPattern": "booklet_{code}/{roll}",
          "matcher": {
            "formatString": "{code}",
            "matchRegex": "^[A-D]$"
          },
          "action": "symlink",
          "collisionStrategy": "increment"
        }
      ]
    }
  }
}
```

### Rule Structure
```python
@dataclass
class GroupingRule:
    name: str                    # Human-readable rule name
    priority: int                # Lower number = higher priority
    destination_pattern: str     # Path pattern with {field} placeholders
    matcher: dict                # { "formatString": "...", "matchRegex": "..." }
    action: str = "symlink"      # "symlink" or "copy"
    collision_strategy: str = "skip"  # "skip", "increment", or "overwrite"
```

## Key Concepts

### 1. Dynamic Patterns

Patterns support placeholders that get replaced with OMR detected values:

```python
# Built-in fields (always available)
{file_path}       # Full path to original input file
{file_name}       # Input filename with extension
{file_stem}       # Input filename without extension
{original_name}   # Output filename (from CheckedOMRs/)
{is_multi_marked} # Boolean string ("True"/"False")

# OMR fields (from template)
{roll}            # Any field defined in template
{code}            # Any field defined in template
{batch}           # Any field defined in template

# Evaluation fields (requires evaluation.json)
{score}           # Total score from evaluation
```

**Example Pattern Resolution**:
```python
Pattern:  "booklet_{code}/roll_{roll}"
Fields:   {"code": "A", "roll": "12345"}
Result:   "booklet_A/roll_12345.jpg"
```

### 2. Rule Matching with Priority

Rules are evaluated in **priority order** (lower number first):

```python
# Priority 1 (evaluated first)
Rule: Match rolls starting with "123"
Pattern: "batch_morning/{roll}"

# Priority 2 (evaluated second)
Rule: Match all rolls
Pattern: "all_students/{roll}"
```

The **first matching rule** is used. If no rules match, `defaultPattern` is applied.

### 3. Matcher System

Each rule has a matcher that determines if it applies:

```json
{
  "formatString": "{score}",    // Format a string from fields
  "matchRegex": "^(9[0-9]|100)$"  // Match scores 90-100
}
```

**Matching Process**:
1. Format `formatString` using available fields
2. Apply regex pattern to formatted string
3. If match succeeds, use this rule

### 4. Collision Strategies

When destination file already exists:

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| `skip` | Don't create file, log warning | Avoid duplicates |
| `increment` | Append `_001`, `_002`, etc. | Keep all versions |
| `overwrite` | Replace existing file | Update with latest |

### 5. Action Types

| Action | Behavior | OS Compatibility |
|--------|----------|------------------|
| `symlink` | Create symbolic link (relative) | Unix/Linux/Mac ✓, Windows partial |
| `copy` | Copy file contents | All platforms ✓ |

**Note**: On Windows, `symlink` automatically falls back to `copy` if symlink creation fails.

### 6. Extension Preservation

If pattern doesn't specify an extension, the original extension is preserved:

```python
Pattern:  "booklet_{code}/{roll}"
Original: "image_CHECKED.jpg"
Result:   "booklet_A/12345.jpg"  # .jpg preserved
```

## Data Flow

### Input (Collection Phase)
```python
context.file_path           # Original input file path
context.omr_response        # Detected OMR values {"roll": "12345", "code": "A"}
context.score               # Evaluation score (if evaluation enabled)
context.is_multi_marked     # Multi-mark detection flag
context.metadata["output_path"]  # Path to file in CheckedOMRs/
```

### Processing (Organization Phase)
```python
# For each collected result:
1. Build formatting_fields dictionary (all available fields)
2. Find matching rule by priority
3. Resolve pattern using FilePatternResolver
4. Handle collisions based on strategy
5. Create symlink or copy
6. Track operation for summary
```

### Output
```python
# File operations tracked
file_operations = [
    {
        "source": "/path/to/CheckedOMRs/image_CHECKED.jpg",
        "destination": "/path/to/organized/booklet_A/12345.jpg",
        "action": "symlink",
        "rule": "Sort by Booklet"
    }
]

# Summary printed
Total files processed: 10
  Sort by Booklet: 8 files
  default: 2 files
Organized files location: /path/to/organized/
```

## Thread Safety

### Why Thread-Safe Collection?

FileOrganizerProcessor runs **during** parallel processing but organizes **after**:

```python
# Thread-safe result collection
def process(self, context: ProcessingContext) -> ProcessingContext:
    with self._lock:
        self.results.append({...})  # Thread-safe append
    return context
```

### Why Sequential Organization?

```python
# Sequential organization (no race conditions)
def finish_processing_directory(self):
    for result in self.results:
        self._organize_single_file(result)  # Safe to access filesystem
```

**Key Design**: Collection is parallel-safe, organization is sequential. This eliminates:
- File system race conditions
- Collision handling conflicts
- Directory creation conflicts

## Edge Cases & Constraints

### 1. Disabled Configuration
```python
if not self.config.enabled:
    return context  # Skip collection entirely
```

### 2. Missing Output File
```python
if not output_path or not Path(output_path).exists():
    logger.warning(f"Output file not found, skipping: {output_path}")
    return  # Skip this file
```

### 3. Missing Fields in Pattern
```python
pattern = "booklet_{code}/{roll}"
fields = {"roll": "12345"}  # Missing 'code'
# Result: Pattern resolution fails, file skipped with warning
```

### 4. Invalid Characters in Paths
```python
# FilePatternResolver sanitizes paths automatically
field_value = "Code: A/B?"
# Sanitized to: "Code_A_B"
```

### 5. Duplicate Rule Priorities
```python
# Validation catches this in config.validate()
priorities = [1, 1, 2]  # Error: Duplicate priority 1
```

### 6. Windows Symlink Limitations
```python
try:
    dest_path.symlink_to(source_path)
except (OSError, NotImplementedError):
    if os.name == "nt":
        shutil.copy2(source_path, dest_path)  # Fallback to copy
```

## Dependencies

### Internal Modules
- `FilePatternResolver` - Pattern formatting and collision handling
- `FileGroupingConfig` / `GroupingRule` - Configuration dataclasses
- `ProcessingContext` - Context flow from pipeline
- `Processor` - Base class interface

### Python Standard Library
- `pathlib.Path` - Path manipulation
- `shutil.copy2` - File copying
- `os.path.relpath` - Relative symlink creation
- `threading.Lock` - Thread-safe collection
- `re.search` - Regex matching

## Performance Considerations

### Computational Cost
1. **Collection Phase**: O(1) per file (just append to list)
2. **Organization Phase**: O(n) where n = number of files
3. **Rule Matching**: O(r) where r = number of rules (stop at first match)
4. **Collision Handling**: O(1) for skip/overwrite, O(k) for increment

### Optimization Strategies
- Lazy initialization (only create organizer if enabled)
- Single-pass organization (no re-scanning)
- Early exit on rule match (priority-based)
- Relative symlinks (faster than copies)

## Browser Migration Notes

### Critical Challenges

#### 1. Symbolic Links Not Supported
**Python**: `path.symlink_to()` creates filesystem symlinks
**Browser**:
- File API does NOT support symlinks
- **Solution**: Always use `action: "copy"` in browser

#### 2. File System Access
**Python**: Direct filesystem read/write
**Browser**:
- Use **File System Access API** (Chrome/Edge only)
- Or **IndexedDB** for storing organized file references
- Or **Download API** to download organized files as ZIP

#### 3. Concurrent Processing
**Python**: ThreadPoolExecutor + Lock
**Browser**:
- Use **Web Workers** for parallel processing
- Use **SharedArrayBuffer** + **Atomics** for thread-safe collection
- Or **sequential organization** with async/await

### Recommended Browser Strategy

```typescript
// Option 1: IndexedDB References
interface OrganizedFile {
  sourcePath: string;
  destinationPath: string;
  blob: Blob;
}

// Store organized files in IndexedDB
await db.organizedFiles.put({
  sourcePath: "CheckedOMRs/image1.jpg",
  destinationPath: "booklet_A/roll_12345.jpg",
  blob: imageBlob
});

// Option 2: ZIP Download
import JSZip from 'jszip';

const zip = new JSZip();
for (const file of organizedFiles) {
  zip.file(file.destinationPath, file.blob);
}
const zipBlob = await zip.generateAsync({type: "blob"});
downloadBlob(zipBlob, "organized_files.zip");

// Option 3: File System Access API (Chrome/Edge)
const dirHandle = await window.showDirectoryPicker();
const organizedHandle = await dirHandle.getDirectoryHandle("organized", {create: true});
// Create nested folders and write files
```

#### 4. Path Sanitization
**Python**: OS handles invalid characters
**Browser**:
- Must sanitize manually (no `/` in filenames for ZIP)
- Different rules for File System Access API vs IndexedDB keys

```typescript
function sanitizeForZip(path: string): string {
  // ZIP uses forward slashes for directories
  return path.split('/').map(part =>
    part.replace(/[<>:"|?*\\]/g, '_')
  ).join('/');
}
```

#### 5. Pattern Resolution
**Python**: `str.format()` with dict
**Browser**:
- Use template literals or string replacement

```typescript
function resolvePattern(pattern: string, fields: Record<string, string>): string {
  return pattern.replace(/\{(\w+)\}/g, (match, key) => {
    return fields[key] || match;
  });
}
```

## Integration Points

### 1. Pipeline Integration
```python
# Added dynamically in entry.py if enabled
if file_grouping_config.enabled:
    organizer = FileOrganizerProcessor(file_grouping_config, output_dir)
    template.pipeline.add_processor(organizer)
```

### 2. Lifecycle Hooks
```python
# Collection: Called during parallel processing
organizer.process(context)

# Organization: Called after directory processing completes
organizer.finish_processing_directory()
```

### 3. Config Validation
```python
# Validates patterns against template fields
validation_errors = file_grouping_config.validate(
    template=template,
    has_evaluation=(evaluation_config is not None)
)
```

## Related Modules

- **FilePatternResolver** (`utils/file_pattern_resolver.py`) - Pattern formatting engine
- **Pipeline** (`processors/pipeline.py`) - Processor orchestration
- **Config** (`schemas/models/config.py`) - Configuration dataclasses
- **CSV Writer** (`utils/csv.py`) - Similar thread-safe append pattern

## Next Steps

1. Read `flows.md` for detailed execution flow and algorithms
2. Read `decisions.md` for design choices and trade-offs
3. Read `constraints.md` for edge cases and validation rules
4. Read `integration.md` for pipeline integration and lifecycle
