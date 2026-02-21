# File Organization - Integration & Pipeline

## Overview

This document details how the File Organization system integrates with the OMRChecker pipeline, including lifecycle hooks, configuration loading, and interaction with other processors.

## Pipeline Integration Points

### 1. Configuration Loading

**Location**: `entry.py:152-183`

**Flow**:
```python
# In process_directory_wise() after template loaded
file_grouping_config = tuning_config.outputs.file_grouping

if file_grouping_config.enabled:
    # Validate configuration
    has_evaluation = evaluation_config is not None
    validation_errors = file_grouping_config.validate(
        template=template,
        has_evaluation=has_evaluation
    )

    if validation_errors:
        # Log errors and disable
        logger.error("File grouping configuration has errors.")
        for error in validation_errors:
            logger.error(f"  - {error}")
        logger.error("File organization will be DISABLED.")
    else:
        # Create and add processor
        from src.processors.organization import FileOrganizerProcessor

        organizer = FileOrganizerProcessor(
            file_grouping_config,
            output_dir=template.path_utils.output_dir
        )

        # Add to pipeline
        template.pipeline.add_processor(organizer)

        logger.info("File organization enabled with dynamic patterns")
```

**Key Points**:
- Configuration loaded from `config.json` via `tuning_config.outputs.file_grouping`
- Validation runs BEFORE adding to pipeline
- Validation errors cause organization to be disabled (fail-safe)
- Processor dynamically added to pipeline if enabled

---

### 2. Pipeline Position

**Typical Pipeline Order**:
```
1. PreprocessingCoordinator     # Image preprocessing
2. AlignmentProcessor           # Template alignment
3. ReadOMRProcessor             # Bubble detection
4. EvaluationProcessor          # Scoring (if enabled)
5. FileOrganizerProcessor       # ← Added here if enabled
```

**Rationale**:
- Organization must run AFTER detection (needs OMR responses)
- Organization must run AFTER evaluation (if using `{score}` field)
- Organization is last step (doesn't modify context for other processors)

**Code**:
```python
# In template.py or pipeline.py
pipeline.add_processor(preprocessing_coordinator)
pipeline.add_processor(alignment_processor)
pipeline.add_processor(read_omr_processor)
if evaluation_config:
    pipeline.add_processor(evaluation_processor)
if file_grouping_enabled:
    pipeline.add_processor(organizer)  # ← Last processor
```

---

### 3. Processor Interface

**Base Class**: `Processor` (from `src/processors/base.py`)

**Required Methods**:
```python
class FileOrganizerProcessor(Processor):
    def get_name(self) -> str:
        """Return processor name for logging."""
        return "FileOrganizer"

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Collect results during parallel processing."""
        # Thread-safe collection
        with self._lock:
            self.results.append({...})
        return context  # Return unchanged

    def finish_processing_directory(self):
        """Organize files after all processing complete."""
        # Sequential organization
        for result in self.results:
            self._organize_single_file(result)
```

**Key Points**:
- Implements standard `Processor` interface
- `process()` called for each file during parallel processing
- `finish_processing_directory()` called once after all files processed
- Returns context unchanged (doesn't modify for downstream processors)

---

### 4. Lifecycle Hooks

**Phase 1: Directory Start**
```python
# entry.py
template.reset_and_setup_for_directory(output_dir)
# ↓ Creates output directories
# ↓ Resets template state
# ↓ FileOrganizerProcessor initialized here
```

**Phase 2: Parallel Processing**
```python
# For each file in parallel:
context = template.process_file(file_path, gray_image, colored_image)
# ↓ Runs through pipeline processors
# ↓ FileOrganizerProcessor.process() collects results
```

**Phase 3: Directory Finish**
```python
# After all files processed:
template.finish_processing_directory()
# ↓ Calls finish_processing_directory() on all processors
# ↓ FileOrganizerProcessor organizes files
```

**Code Flow**:
```python
# In entry.py:process_directory_files()
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = {
        executor.submit(
            template.process_file, file_path, gray_image, colored_image
        ): file_path
        for file_path in omr_files
    }

    for future in as_completed(futures):
        context = future.result()
        # FileOrganizerProcessor.process() was called during this

# After all futures complete:
template.finish_processing_directory()
# ↓ FileOrganizerProcessor.finish_processing_directory() runs here
```

---

## Configuration Structure

### Config Hierarchy

**Global Config** (`config.json`):
```json
{
  "outputs": {
    "fileGrouping": {
      "enabled": true,
      "defaultPattern": "ungrouped/{original_name}",
      "rules": [...]
    }
  }
}
```

**Loaded Via**:
```python
# In entry.py
tuning_config = open_config_with_defaults(local_config_path, args)
# ↓ Merges default config + global config + local config
# ↓ Returns Config dataclass instance

file_grouping_config = tuning_config.outputs.file_grouping
# ↓ Type: FileGroupingConfig dataclass
```

**Config Dataclasses**:
```python
@dataclass
class Config:
    outputs: OutputsConfig
    # ... other sections

@dataclass
class OutputsConfig:
    file_grouping: FileGroupingConfig = field(default_factory=FileGroupingConfig)
    # ... other output settings

@dataclass
class FileGroupingConfig:
    enabled: bool = False
    rules: list[GroupingRule] = field(default_factory=list)
    default_pattern: str = "ungrouped/{original_name}"
```

---

### Validation Integration

**Validation Trigger**:
```python
# In entry.py before adding processor
validation_errors = file_grouping_config.validate(
    template=template,
    has_evaluation=(evaluation_config is not None)
)
```

**Validation Logic** (`config.py:FileGroupingConfig.validate()`):
```python
def validate(self, template=None, *, has_evaluation: bool = False) -> list[str]:
    errors = []

    # Validate default pattern
    errors.extend(self._validate_pattern(
        self.default_pattern, "default_pattern", template, has_evaluation
    ))

    # Validate each rule
    for i, rule in enumerate(self.rules, 1):
        errors.extend(self._validate_rule(rule, i, template, has_evaluation))

    # Check for duplicate priorities
    priorities = [rule.priority for rule in self.rules]
    if len(priorities) != len(set(priorities)):
        errors.append("Duplicate rule priorities found")

    return errors
```

**Pattern Validation**:
```python
def _validate_pattern(self, pattern, pattern_name, template, has_evaluation):
    # Extract field names from pattern
    field_names = {
        field_name
        for _, field_name, _, _ in string.Formatter().parse(pattern)
        if field_name
    }

    # Check each field exists
    for field_name in field_names:
        if field_name in BUILTIN_FIELDS:
            continue
        if field_name in EVALUATION_FIELDS and not has_evaluation:
            errors.append(f"{field_name} requires evaluation.json")
        if template and field_name not in template.all_fields:
            available = sorted(BUILTIN_FIELDS | EVALUATION_FIELDS | template.all_fields)
            errors.append(f"Field '{field_name}' not found. Available: {available}")

    return errors
```

---

## Context Flow

### ProcessingContext Fields Used

**Input Fields** (Set by earlier processors):
```python
@dataclass
class ProcessingContext:
    # Used by FileOrganizer:
    file_path: Path | str              # Original input file path
    omr_response: dict[str, str]       # Detected OMR values
    is_multi_marked: bool              # Multi-mark detection flag
    score: float                       # Evaluation score (if evaluation enabled)
    metadata: dict[str, Any]           # Contains "output_path"
```

**Metadata Field**:
```python
# Set by template.process_file() after saving output
context.metadata["output_path"] = str(checked_omr_path)
# ↓ FileOrganizer uses this to find file to organize
```

**Example Context**:
```python
context = ProcessingContext(
    file_path="/path/to/input/sheet1.jpg",
    gray_image=...,
    colored_image=...,
    template=template,
    omr_response={"roll": "12345", "code": "A", "batch": "morning"},
    is_multi_marked=False,
    score=95.5,
    metadata={
        "output_path": "/path/to/outputs/CheckedOMRs/sheet1_CHECKED.jpg"
    }
)
```

---

## Interaction with Other Processors

### Dependency on ReadOMRProcessor

**Requirement**: Must run AFTER `ReadOMRProcessor` to access `omr_response`

**Flow**:
```python
# ReadOMRProcessor populates omr_response
context.omr_response = {
    "roll": "12345",
    "code": "A",
    "q1": "B",
    "q2": "C"
}

# FileOrganizerProcessor reads omr_response
organizing_fields = {
    **context.omr_response,  # Includes all detected fields
    "original_name": Path(output_path).name
}
```

**Edge Case**:
- If ReadOMRProcessor fails → `omr_response` is empty `{}`
- Patterns using OMR fields will fail (KeyError)
- File skipped with warning

---

### Dependency on EvaluationProcessor

**Requirement**: If using `{score}` field, must run AFTER `EvaluationProcessor`

**Flow**:
```python
# EvaluationProcessor populates score
context.score = 95.5

# FileOrganizerProcessor reads score
organizing_fields = {
    "score": str(context.score),  # Converted to string
    ...
}
```

**Validation**:
```python
# Validation catches this BEFORE processing
if "{score}" in pattern and not has_evaluation:
    errors.append("Field '{score}' requires evaluation.json")
```

**Edge Case**:
- If evaluation disabled but pattern uses `{score}` → Validation error
- Organization disabled to prevent runtime errors

---

### Independence from Preprocessing/Alignment

**No Dependencies**:
- FileOrganizer doesn't care about preprocessing or alignment
- Only cares about final detected values in `omr_response`

**Benefits**:
- Can enable/disable preprocessing without affecting organization
- Organization logic is decoupled from image processing

---

## Output Directory Structure

### Created Directories

**Base Output** (`template.path_utils.output_dir`):
```
outputs/
└── <directory_name>/
    ├── CheckedOMRs/          # Created by template
    │   └── (processed images)
    ├── Results/              # Created by template
    │   └── Results.csv
    └── organized/            # Created by FileOrganizer
        └── (organized view)
```

**Organized Directory**:
```python
self.organized_dir = output_dir / "organized"
# ↓ Created in finish_processing_directory()
self.organized_dir.mkdir(parents=True, exist_ok=True)
```

**Dynamic Subdirectories**:
```python
# Created based on pattern resolution
dest_path = organized_dir / "booklet_A" / "roll_12345.jpg"
dest_path.parent.mkdir(parents=True, exist_ok=True)
# ↓ Creates "booklet_A" directory if it doesn't exist
```

---

### File Placement Examples

**Example 1: Booklet-based Organization**

**Config**:
```json
{
  "destinationPattern": "booklet_{code}/{roll}",
  "matcher": {"formatString": "{code}", "matchRegex": "^[A-D]$"}
}
```

**Result**:
```
organized/
├── booklet_A/
│   ├── 12345.jpg → ../../CheckedOMRs/sheet1_CHECKED.jpg
│   └── 12346.jpg → ../../CheckedOMRs/sheet3_CHECKED.jpg
└── booklet_B/
    ├── 12347.jpg → ../../CheckedOMRs/sheet2_CHECKED.jpg
    └── 12348.jpg → ../../CheckedOMRs/sheet4_CHECKED.jpg
```

**Example 2: Score-based Organization**

**Config**:
```json
{
  "rules": [
    {
      "name": "High Scorers",
      "priority": 1,
      "destinationPattern": "high_scores/{roll}",
      "matcher": {"formatString": "{score}", "matchRegex": "^(9[0-9]|100)$"}
    },
    {
      "name": "Medium Scorers",
      "priority": 2,
      "destinationPattern": "medium_scores/{roll}",
      "matcher": {"formatString": "{score}", "matchRegex": "^[6-8][0-9]$"}
    }
  ],
  "defaultPattern": "low_scores/{roll}"
}
```

**Result**:
```
organized/
├── high_scores/    # scores 90-100
│   ├── 12345.jpg
│   └── 12347.jpg
├── medium_scores/  # scores 60-89
│   ├── 12346.jpg
│   └── 12349.jpg
└── low_scores/     # scores < 60
    └── 12348.jpg
```

---

## Parallel Processing Integration

### Thread Safety Design

**Collection Phase** (Parallel):
```python
# Multiple worker threads calling this simultaneously
def process(self, context: ProcessingContext) -> ProcessingContext:
    with self._lock:  # ← Thread-safe
        self.results.append({
            "context": context,
            "output_path": context.metadata.get("output_path"),
            "score": context.score,
            "omr_response": context.omr_response.copy(),  # ← Copy to avoid mutation
            "is_multi_marked": context.is_multi_marked,
        })
    return context
```

**Why Copy `omr_response`?**
- Context object may be reused or mutated by other processors
- Copying ensures we have snapshot of values at this point
- Prevents race conditions if context is modified

**Organization Phase** (Sequential):
```python
# Single thread calling this after all workers finished
def finish_processing_directory(self):
    # No lock needed, sequential execution
    for result in self.results:
        self._organize_single_file(result)
```

**Why Sequential?**
- File system operations are NOT thread-safe
- Collision detection requires atomic check-and-create
- Simpler error handling (no need for thread-local errors)

---

### Worker Pool Integration

**Entry Point** (`entry.py:process_directory_files()`):
```python
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    # Submit all files for processing
    futures = {
        executor.submit(
            template.process_file,
            file_path,
            gray_image,
            colored_image
        ): file_path
        for file_path in omr_files
    }

    # Wait for completion
    for future in as_completed(futures):
        try:
            context = future.result()
            # FileOrganizer.process() was called during this
        except Exception as e:
            logger.error(f"Error processing file: {e}")

# All workers finished
template.finish_processing_directory()
# ↓ FileOrganizer.finish_processing_directory() runs here
```

**Key Points**:
- `num_workers` set by `config.processing.max_parallel_workers`
- Default: `1` (sequential processing)
- FileOrganizer works with any worker count (thread-safe collection)

---

## Error Handling & Recovery

### Validation Errors

**Behavior**: Disable organization if validation fails

**Flow**:
```python
if validation_errors:
    logger.error("File grouping configuration has errors.")
    for error in validation_errors:
        logger.error(f"  - {error}")
    logger.error("File organization will be DISABLED.")
    # ↓ Don't add processor to pipeline
    # ↓ Processing continues normally, just no organization
```

**User Experience**:
- Processing completes successfully
- Files are in `CheckedOMRs/` but not organized
- CSV results are generated
- User sees errors in log and can fix config

---

### Runtime Errors

**File Not Found**:
```python
if not output_path or not Path(output_path).exists():
    logger.warning(f"Output file not found, skipping: {output_path}")
    return  # Skip this file, continue with others
```

**Pattern Resolution Errors**:
```python
try:
    formatted = pattern.format(**fields)
except KeyError as e:
    logger.warning(f"Pattern references undefined field: {e}")
    return None  # Skip this file
```

**File Operation Errors**:
```python
try:
    shutil.copy2(source_path, dest_path)
except Exception as e:
    logger.error(f"Failed to copy {source_path.name}: {e}")
    self.file_operations.append({
        "action": "failed",
        "error": str(e)
    })
```

**Key Principle**: Errors in organization don't stop processing. Files are skipped, logged, and processing continues.

---

### Summary Reporting

**Always Prints Summary** (even if errors):
```python
def _print_summary(self):
    logger.info("File Organization Summary")
    logger.info(f"Total files processed: {len(self.file_operations)}")

    # Group by rule
    for rule_name, ops in by_rule.items():
        logger.info(f"  {rule_name}: {len(ops)} files")

    # Show warnings
    if skipped > 0:
        logger.warning(f"  Skipped: {skipped} files")
    if failed > 0:
        logger.error(f"  Failed: {failed} files")

    logger.info(f"Organized files location: {self.organized_dir}")
```

**User Benefits**:
- See how many files were organized
- See which rules matched
- See how many errors occurred
- Know where to find organized files

---

## Future Integration Points

### Potential Enhancements

1. **Multiple Organizers**:
   ```python
   # Could add multiple organizers with different configs
   pipeline.add_processor(organizer_by_booklet)
   pipeline.add_processor(organizer_by_score)
   # Each creates different organized/ subdirectory
   ```

2. **Conditional Organization**:
   ```python
   # Only organize if confidence > threshold
   if context.metadata.get("confidence", 1.0) > 0.8:
       organizer.process(context)
   ```

3. **External Hook Integration**:
   ```python
   # Call external script after organization
   subprocess.run(["upload_to_cloud.sh", organized_dir])
   ```

4. **Incremental Organization**:
   ```python
   # Re-organize only new files in subsequent runs
   organizer.organize_incremental(new_files_only=True)
   ```

---

## Browser Integration Strategy

### Config Loading
```typescript
// Load from localStorage or JSON file
const config = await loadConfig();
const fileGrouping = config.outputs.fileGrouping;

if (fileGrouping.enabled) {
  const organizer = new FileOrganizer(fileGrouping);
  pipeline.addProcessor(organizer);
}
```

### Collection Phase
```typescript
// In Web Worker processing
self.onmessage = async (e) => {
  const context = await processFile(e.data);
  // Send results back to main thread
  self.postMessage({type: "result", context});
};

// In main thread
worker.onmessage = (e) => {
  if (e.data.type === "result") {
    organizer.addResult(e.data.context);
  }
};
```

### Organization Phase
```typescript
// After all workers complete
await organizer.finishProcessing();
// ↓ Creates ZIP or IndexedDB entries
// ↓ Downloads ZIP or shows organized view in UI
```

---

## Summary

**Integration Checklist**:
- ✅ Config loaded from `config.json`
- ✅ Validation runs before pipeline execution
- ✅ Processor added to pipeline if enabled and valid
- ✅ `process()` collects results during parallel processing (thread-safe)
- ✅ `finish_processing_directory()` organizes files sequentially
- ✅ Errors in organization don't stop processing
- ✅ Summary always printed
- ✅ Organized files in separate `organized/` directory
- ✅ Works with any number of parallel workers
- ✅ Compatible with optional evaluation processor
