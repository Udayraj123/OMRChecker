# File Organization - Detailed Flows

## Overview

This document details the execution flows for the File Organization system, from configuration loading through pattern resolution to final file placement.

## Flow 1: Processor Initialization

### Trigger
```python
# In entry.py during directory processing
if file_grouping_config.enabled:
    organizer = FileOrganizerProcessor(file_grouping_config, output_dir)
    template.pipeline.add_processor(organizer)
```

### Initialization Flow

```
FileOrganizerProcessor.__init__()
    ↓
1. Store config and output_dir
    ↓
2. Create organized_dir path (output_dir / "organized")
    ↓
3. Initialize collections
   - results = []           # Collected processing contexts
   - file_operations = []   # Track all file operations
   - _lock = Lock()         # Thread-safe collection
    ↓
4. Initialize FilePatternResolver(base_dir=organized_dir)
    ↓
5. Sort rules by priority (lower number = higher priority)
   config.rules.sort(key=lambda r: r.priority)
```

**Code Reference**:
```python
def __init__(self, file_grouping_config, output_dir: Path) -> None:
    self.config = file_grouping_config
    self.output_dir = output_dir
    self.organized_dir = output_dir / "organized"
    self.results = []
    self.file_operations = []
    self._lock = Lock()

    # Initialize pattern resolver
    self.pattern_resolver = FilePatternResolver(base_dir=self.organized_dir)

    # Sort rules by priority
    if self.config.enabled and self.config.rules:
        self.config.rules.sort(key=lambda r: r.priority)
```

## Flow 2: Result Collection (Parallel Phase)

### Trigger
```python
# During parallel file processing
context = processor.process(context)  # For each processor
```

### Collection Flow

```
FileOrganizerProcessor.process(context)
    ↓
1. Check if enabled
   if not config.enabled → return context unchanged
    ↓
2. Acquire thread lock
   with self._lock:
    ↓
3. Extract relevant data from context
   - output_path from metadata
   - score (if available)
   - omr_response (copy to avoid mutation)
   - is_multi_marked
    ↓
4. Append to results list (thread-safe)
   results.append({
       "context": context,
       "output_path": output_path,
       "score": score,
       "omr_response": omr_response.copy(),
       "is_multi_marked": is_multi_marked
   })
    ↓
5. Release lock and return context
```

**Code Reference**:
```python
def process(self, context: ProcessingContext) -> ProcessingContext:
    if not self.config.enabled:
        return context

    # Thread-safe collection
    with self._lock:
        self.results.append({
            "context": context,
            "output_path": context.metadata.get("output_path"),
            "score": context.score if hasattr(context, "score") else 0,
            "omr_response": context.omr_response.copy(),
            "is_multi_marked": context.is_multi_marked,
        })

    return context
```

**Thread Safety**: The `_lock` ensures that multiple worker threads can safely append to `results` without race conditions.

## Flow 3: Directory Organization (Sequential Phase)

### Trigger
```python
# In entry.py after all files processed
template.finish_processing_directory()
```

### Organization Flow

```
FileOrganizerProcessor.finish_processing_directory()
    ↓
1. Check prerequisites
   if not enabled OR no results → return early
    ↓
2. Print organization header
   logger.info("Starting file organization...")
    ↓
3. Create organized base directory
   organized_dir.mkdir(parents=True, exist_ok=True)
    ↓
4. Process each result SEQUENTIALLY
   for result in results:
       _organize_single_file(result)
    ↓
5. Print summary report
   _print_summary()
```

**Code Reference**:
```python
def finish_processing_directory(self):
    if not self.config.enabled or not self.results:
        return

    logger.info(f"\n{'=' * 60}")
    logger.info("Starting file organization...")
    logger.info(f"{'=' * 60}")

    # Create organized base directory
    self.organized_dir.mkdir(parents=True, exist_ok=True)

    # Process each result sequentially (no concurrency issues!)
    for result in self.results:
        self._organize_single_file(result)

    # Print summary
    self._print_summary()
```

## Flow 4: Single File Organization

### Input
```python
result = {
    "context": ProcessingContext,
    "output_path": "/path/to/CheckedOMRs/image_CHECKED.jpg",
    "score": 95,
    "omr_response": {"roll": "12345", "code": "A"},
    "is_multi_marked": False
}
```

### Organization Flow

```
_organize_single_file(result)
    ↓
1. Extract output_path and validate existence
   if not exists → log warning and skip
    ↓
2. Build formatting_fields dictionary
   {
       "file_path": str(context.file_path),
       "file_name": Path(context.file_path).name,
       "file_stem": Path(context.file_path).stem,
       "original_name": Path(output_path).name,
       "score": str(score),
       "is_multi_marked": str(is_multi_marked),
       **omr_response  # All OMR fields (roll, code, etc.)
   }
    ↓
3. Find matching rule by priority
   matched_rule = _find_matching_rule(formatting_fields)
    ↓
4. Determine pattern, action, collision strategy
   if matched_rule:
       use rule's pattern/action/collision
   else:
       use default_pattern, "symlink", "skip"
    ↓
5. Resolve pattern to destination path
   dest_path = pattern_resolver.resolve_pattern(
       pattern, fields, source_path, collision_strategy
   )
    ↓
6. Handle collision result
   if dest_path is None → skip file
    ↓
7. Create parent directories
   dest_path.parent.mkdir(parents=True, exist_ok=True)
    ↓
8. Perform file operation (symlink or copy)
   if action == "symlink" → try symlink (with fallback)
   if action == "copy" → copy file
    ↓
9. Track operation for summary
   file_operations.append({...})
```

**Code Reference**:
```python
def _organize_single_file(self, result: dict) -> None:
    context = result["context"]
    output_path = result.get("output_path")

    # Validate file exists
    if not output_path or not Path(output_path).exists():
        logger.warning(f"Output file not found, skipping: {output_path}")
        return

    # Build formatting fields
    formatting_fields = {
        "file_path": str(context.file_path),
        "file_name": Path(context.file_path).name,
        "file_stem": Path(context.file_path).stem,
        "original_name": Path(output_path).name,
        "score": str(result["score"]),
        "is_multi_marked": str(result["is_multi_marked"]),
        **result["omr_response"],
    }

    # Find matching rule
    matched_rule = self._find_matching_rule(formatting_fields)

    # Determine pattern/action/collision strategy
    if matched_rule:
        pattern = matched_rule.destination_pattern
        action = matched_rule.action
        collision_strategy = matched_rule.collision_strategy
        rule_name = matched_rule.name
    else:
        pattern = self.config.default_pattern
        action = "symlink"
        collision_strategy = "skip"
        rule_name = "default"

    # Resolve pattern
    source_path = Path(output_path)
    dest_path = self.pattern_resolver.resolve_pattern(
        pattern=pattern,
        fields=formatting_fields,
        original_path=source_path,
        collision_strategy=collision_strategy,
    )

    # Handle collision skip
    if dest_path is None:
        logger.warning(f"Skipping due to collision or pattern error: {source_path.name}")
        self.file_operations.append({
            "source": str(source_path),
            "destination": "N/A",
            "action": "skipped",
            "rule": rule_name,
            "reason": "collision or pattern error",
        })
        return

    # Create directories
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Perform operation
    try:
        if action == "symlink":
            # Create relative symlink if possible
            try:
                rel_source = os.path.relpath(source_path, dest_path.parent)
                dest_path.symlink_to(rel_source)
            except (OSError, NotImplementedError):
                # Fallback to absolute symlink or copy on Windows
                if os.name == "nt":
                    shutil.copy2(str(source_path), str(dest_path))
                    action = "copy"
                else:
                    dest_path.symlink_to(source_path.absolute())

        elif action == "copy":
            shutil.copy2(str(source_path), str(dest_path))

        # Track success
        self.file_operations.append({
            "source": str(source_path),
            "destination": str(dest_path),
            "action": action,
            "rule": rule_name,
        })

    except Exception as e:
        logger.error(f"Failed to {action} {source_path.name}: {e}")
        self.file_operations.append({
            "source": str(source_path),
            "destination": str(dest_path) if dest_path else "N/A",
            "action": "failed",
            "rule": rule_name,
            "error": str(e),
        })
```

## Flow 5: Rule Matching

### Input
```python
formatting_fields = {
    "roll": "12345",
    "code": "A",
    "score": "95"
}

rules = [
    {priority: 1, matcher: {"formatString": "{code}", "matchRegex": "^[A-D]$"}},
    {priority: 2, matcher: {"formatString": "{score}", "matchRegex": "^(9[0-9]|100)$"}}
]
```

### Matching Flow

```
_find_matching_rule(formatting_fields)
    ↓
1. Iterate rules in priority order (already sorted)
   for rule in config.rules:
    ↓
2. Extract matcher configuration
   format_string = rule.matcher["format_string"]
   match_regex = rule.matcher["match_regex"]
    ↓
3. Format the string using fields
   try:
       formatted = format_string.format(**formatting_fields)
   catch KeyError:
       log warning and continue to next rule
    ↓
4. Apply regex pattern
   if re.search(match_regex, formatted):
       return rule  # First match wins!
    ↓
5. No match found
   return None  # Use default pattern
```

**Code Reference**:
```python
def _find_matching_rule(self, formatting_fields: dict) -> GroupingRule | None:
    for rule in self.config.rules:
        try:
            format_string = rule.matcher["format_string"]
            match_regex = rule.matcher["match_regex"]

            formatted_string = format_string.format(**formatting_fields)

            if re.search(match_regex, formatted_string):
                return rule

        except KeyError as e:
            logger.warning(f"Rule '{rule.name}' references undefined field: {e}")
        except Exception as e:
            logger.warning(f"Error evaluating rule '{rule.name}': {e}")

    return None
```

**Example**:
```python
# Rule 1: Match codes A-D
format_string = "{code}"         # → "A"
match_regex = "^[A-D]$"          # → MATCH!
return rule  # Stop here, don't check rule 2
```

## Flow 6: Pattern Resolution

### Input
```python
pattern = "booklet_{code}/roll_{roll}"
fields = {"code": "A", "roll": "12345"}
original_path = Path("CheckedOMRs/image_CHECKED.jpg")
collision_strategy = "increment"
```

### Resolution Flow

```
FilePatternResolver.resolve_pattern(...)
    ↓
1. Format pattern with fields
   formatted = pattern.format(**fields)
   # Result: "booklet_A/roll_12345"
    ↓
2. Sanitize path (remove invalid characters)
   sanitized = _sanitize_path(formatted)
   # Result: "booklet_A/roll_12345" (no changes needed)
    ↓
3. Create Path object
   resolved_path = Path(sanitized)
    ↓
4. Preserve extension if not in pattern
   if not resolved_path.suffix:
       original_ext = original_path.suffix  # ".jpg"
       resolved_path = resolved_path.with_suffix(original_ext)
   # Result: Path("booklet_A/roll_12345.jpg")
    ↓
5. Apply base directory
   if base_dir:
       resolved_path = base_dir / resolved_path
   # Result: Path("outputs/organized/booklet_A/roll_12345.jpg")
    ↓
6. Handle collisions
   return _handle_collision(resolved_path, collision_strategy)
```

**Code Reference**:
```python
def resolve_pattern(
    self,
    pattern: str,
    fields: dict[str, Any],
    original_path: Path | str | None = None,
    collision_strategy: str = "skip",
) -> Path | None:
    try:
        # Format the pattern with fields
        formatted = pattern.format(**fields)

        # Sanitize the path
        sanitized = self._sanitize_path(formatted)

        # Create Path object
        resolved_path = Path(sanitized)

        # Handle extension preservation
        if original_path and not resolved_path.suffix:
            original_ext = Path(original_path).suffix
            resolved_path = resolved_path.with_suffix(original_ext)

        # Apply base directory if set
        if self.base_dir:
            resolved_path = self.base_dir / resolved_path

        # Handle collisions
        return self._handle_collision(resolved_path, collision_strategy)

    except KeyError as e:
        logger.warning(f"Pattern references undefined field: {e}")
        return None
    except Exception as e:
        logger.error(f"Error resolving pattern '{pattern}': {e}")
        return None
```

## Flow 7: Path Sanitization

### Input
```python
path_str = "booklet_A/B/roll:12345?"
```

### Sanitization Flow

```
_sanitize_path(path_str)
    ↓
1. Split by directory separator
   parts = path_str.split("/")
   # ["booklet_A/B", "roll:12345?"]
    ↓
2. For each part:
    a. Remove invalid filename characters (<>:"|?*\\)
       sanitized = re.sub(r'[<>:"|?*\\]', "_", part)
       # "roll:12345?" → "roll_12345_"

    b. Collapse multiple underscores
       sanitized = re.sub(r"_+", "_", sanitized)
       # "roll__12345___" → "roll_12345_"

    c. Strip leading/trailing underscores and spaces
       sanitized = sanitized.strip("_ ")
       # "roll_12345_" → "roll_12345"

    d. Add to results if non-empty
       if sanitized:
           sanitized_parts.append(sanitized)
    ↓
3. Join with forward slashes
   return "/".join(sanitized_parts)
   # "booklet_A/B/roll_12345"
```

**Code Reference**:
```python
def _sanitize_path(self, path_str: str) -> str:
    parts = path_str.split("/")
    sanitized_parts = []

    for part in parts:
        # Sanitize each path component
        sanitized = re.sub(r'[<>:"|?*\\]', "_", part)
        # Remove any double underscores
        sanitized = re.sub(r"_+", "_", sanitized)
        # Strip leading/trailing underscores and spaces
        sanitized = sanitized.strip("_ ")
        if sanitized:
            sanitized_parts.append(sanitized)

    return "/".join(sanitized_parts)
```

## Flow 8: Collision Handling

### Input
```python
path = Path("outputs/organized/booklet_A/roll_12345.jpg")
strategy = "increment"
```

### Collision Flow

```
_handle_collision(path, strategy)
    ↓
1. Check if file exists
   if not path.exists() → return path (no collision)
    ↓
2. Apply strategy

   STRATEGY: "skip"
   ↓
   Log debug message
   return None  # Signal to skip file

   STRATEGY: "overwrite"
   ↓
   Log debug message
   return path  # Will overwrite existing file

   STRATEGY: "increment"
   ↓
   Extract stem and suffix
   stem = "roll_12345"
   suffix = ".jpg"
   parent = Path("outputs/organized/booklet_A")
   counter = 1
   ↓
   While counter < 9999:
       new_name = f"{stem}_{counter:03d}{suffix}"
       # "roll_12345_001.jpg"
       new_path = parent / new_name
       if not new_path.exists():
           return new_path  # Found available name
       counter += 1
   ↓
   If counter reaches 9999:
       log error
       return None  # Too many collisions, give up
```

**Code Reference**:
```python
def _handle_collision(self, path: Path, strategy: str) -> Path | None:
    if not path.exists():
        return path

    if strategy == "skip":
        logger.debug(f"File exists, skipping: {path.name}")
        return None

    if strategy == "overwrite":
        logger.debug(f"File exists, will overwrite: {path.name}")
        return path

    if strategy == "increment":
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        counter = 1

        while counter < 9999:
            new_name = f"{stem}_{counter:03d}{suffix}"
            new_path = parent / new_name
            if not new_path.exists():
                logger.debug(f"File exists, using incremented name: {new_name}")
                return new_path
            counter += 1

            if counter == 9999:
                logger.error(f"Too many collisions for {stem}, giving up")
                return None

    logger.warning(f"Unknown collision strategy '{strategy}', skipping")
    return None
```

## Flow 9: Summary Reporting

### Input
```python
file_operations = [
    {"rule": "Sort by Booklet", "action": "symlink"},
    {"rule": "Sort by Booklet", "action": "symlink"},
    {"rule": "default", "action": "copy"},
    {"rule": "Sort by Score", "action": "skipped"},
]
```

### Summary Flow

```
_print_summary()
    ↓
1. Print header
   logger.info("File Organization Summary")
    ↓
2. Count total operations
   total = len(file_operations)
    ↓
3. Group operations by rule and action
   by_rule = {}
   skipped = 0
   failed = 0

   for op in file_operations:
       if action == "skipped": skipped += 1
       elif action == "failed": failed += 1
       else: by_rule[rule].append(op)
    ↓
4. Print counts by rule
   for rule_name, ops in by_rule.items():
       logger.info(f"  {rule_name}: {len(ops)} files")
    ↓
5. Print warnings/errors
   if skipped > 0:
       logger.warning(f"  Skipped: {skipped} files")
   if failed > 0:
       logger.error(f"  Failed: {failed} files")
    ↓
6. Print organized directory location
   logger.info(f"\nOrganized files location: {organized_dir}")
```

**Code Reference**:
```python
def _print_summary(self) -> None:
    logger.info(f"\n{'=' * 60}")
    logger.info("File Organization Summary")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total files processed: {len(self.file_operations)}")

    # Group by rule and action
    by_rule = {}
    skipped = 0
    failed = 0

    for op in self.file_operations:
        rule = op["rule"]
        action = op["action"]

        if action == "skipped":
            skipped += 1
        elif action == "failed":
            failed += 1
        else:
            by_rule.setdefault(rule, []).append(op)

    for rule_name, ops in by_rule.items():
        logger.info(f"  {rule_name}: {len(ops)} files")

    if skipped > 0:
        logger.warning(f"  Skipped (collisions/errors): {skipped} files")
    if failed > 0:
        logger.error(f"  Failed: {failed} files")

    logger.info(f"\nOrganized files location: {self.organized_dir}")
    logger.info(f"{'=' * 60}\n")
```

**Example Output**:
```
============================================================
File Organization Summary
============================================================
Total files processed: 10
  Sort by Booklet: 8 files
  default: 1 files
  Skipped (collisions/errors): 1 files

Organized files location: /path/to/outputs/organized/
============================================================
```

## Complete End-to-End Flow

```
1. INITIALIZATION
   entry.py → FileOrganizerProcessor.__init__()
   ↓
2. COLLECTION PHASE (Parallel)
   Worker Thread 1: process(context_1) → results.append()
   Worker Thread 2: process(context_2) → results.append()
   Worker Thread N: process(context_N) → results.append()
   ↓
3. ORGANIZATION PHASE (Sequential)
   finish_processing_directory()
   ↓
   For each result:
       ↓
       Build formatting_fields
       ↓
       Find matching rule
       ↓
       Resolve pattern → dest_path
       ↓
       Create directories
       ↓
       Create symlink/copy
       ↓
       Track operation
   ↓
4. SUMMARY
   Print organization summary
```

## Browser Implementation Flow

```typescript
// 1. INITIALIZATION
class FileOrganizer {
  private results: OrganizedFile[] = [];
  private operations: FileOperation[] = [];

  constructor(
    private config: FileGroupingConfig,
    private outputDir: string
  ) {
    // Sort rules by priority
    this.config.rules.sort((a, b) => a.priority - b.priority);
  }
}

// 2. COLLECTION PHASE (from Web Workers)
async process(context: ProcessingContext): Promise<void> {
  if (!this.config.enabled) return;

  // Collect result (no lock needed in single-threaded JS)
  this.results.push({
    context,
    outputBlob: context.metadata.outputBlob,
    score: context.score,
    omrResponse: {...context.omrResponse},
    isMultiMarked: context.isMultiMarked
  });
}

// 3. ORGANIZATION PHASE
async finishProcessing(): Promise<void> {
  const zip = new JSZip();

  for (const result of this.results) {
    const fields = this.buildFormattingFields(result);
    const rule = this.findMatchingRule(fields);
    const pattern = rule?.destinationPattern || this.config.defaultPattern;

    const destPath = this.resolvePattern(pattern, fields, result.outputBlob.name);
    if (!destPath) continue;

    // Add to ZIP
    zip.file(destPath, result.outputBlob);

    this.operations.push({
      source: result.context.filePath,
      destination: destPath,
      action: "copy",
      rule: rule?.name || "default"
    });
  }

  // Download ZIP
  const zipBlob = await zip.generateAsync({type: "blob"});
  this.downloadBlob(zipBlob, "organized_files.zip");

  this.printSummary();
}
```

## Performance Characteristics

| Phase | Complexity | Parallelism |
|-------|-----------|-------------|
| Initialization | O(r log r) for sorting rules | Sequential |
| Collection | O(1) per file | Parallel (thread-safe) |
| Organization | O(n * r) where n=files, r=rules | Sequential |
| Pattern Resolution | O(p) where p=pattern length | Sequential |
| Collision Handling | O(1) to O(k) where k=collisions | Sequential |
| Summary | O(n) | Sequential |

**Total**: O(n * r) dominated by organization phase
