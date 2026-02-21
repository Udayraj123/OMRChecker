# File Organization - Constraints & Edge Cases

## Overview

This document catalogs all edge cases, constraints, and error conditions in the File Organization system, along with how they're handled.

## Configuration Constraints

### C1: Enabled Flag Must Be True

**Constraint**: Organization only runs if `fileGrouping.enabled = true`

**Validation**:
```python
if not self.config.enabled:
    return context  # Skip collection entirely
```

**Edge Cases**:
- ✅ Enabled=false → Processor does nothing, no overhead
- ✅ Enabled=true but no rules → Uses defaultPattern for all files
- ✅ Enabled dynamically via args → Not supported, must be in config

---

### C2: Priority Must Be Unique

**Constraint**: No two rules can have the same priority number

**Validation**:
```python
priorities = [rule.priority for rule in self.rules]
if len(priorities) != len(set(priorities)):
    duplicates = [p for p in priorities if priorities.count(p) > 1]
    errors.append(f"Duplicate rule priorities: {set(duplicates)}")
```

**Edge Cases**:
- ❌ Two rules with priority=1 → Validation error at config load
- ✅ Priorities [1, 3, 5] (gaps allowed) → Valid
- ✅ Negative priorities → Valid (lower still means higher priority)

**Error Message**:
```
Duplicate rule priorities found: {1, 3}.
Each rule should have a unique priority.
```

---

### C3: Fields Must Exist in Template or Built-ins

**Constraint**: All `{field}` placeholders must reference:
- Built-in fields: `file_path`, `file_name`, `file_stem`, `original_name`, `is_multi_marked`
- Template fields: Any field defined in template.json
- Evaluation fields: `score` (only if evaluation.json exists)

**Validation**:
```python
# Extract field names from pattern
field_names = {
    field_name
    for _, field_name, _, _ in string.Formatter().parse(pattern)
    if field_name
}

# Check each field
for field_name in field_names:
    if field_name not in (BUILTIN_FIELDS | template_fields | EVALUATION_FIELDS):
        errors.append(f"Field '{{{field_name}}}' not found")
```

**Edge Cases**:
- ❌ Pattern `{booklet}` but field is `booklet_code` → Validation error
- ❌ Pattern `{score}` but no evaluation.json → Validation error
- ✅ Pattern `{roll}` and template has roll field → Valid
- ✅ Pattern `{file_stem}_{roll}` → Valid (both built-in and template field)

**Error Messages**:
```
Field '{booklet}' not found in template.
Available fields: {code}, {roll}, {batch}, {file_name}, {score}

Field '{score}' requires evaluation.json to be present.
Either add evaluation.json or remove this field from the pattern.
```

---

### C4: Regex Must Be Valid

**Constraint**: `matcher.matchRegex` must be a valid regex pattern

**Validation**:
```python
try:
    re.compile(rule.matcher.get("matchRegex", ""))
except re.error as e:
    errors.append(f"Invalid regex pattern: {e}")
```

**Edge Cases**:
- ❌ `matchRegex: "(unclosed"` → Validation error
- ❌ `matchRegex: "[z-a]"` → Invalid range, validation error
- ✅ `matchRegex: ".*"` → Valid (matches everything)
- ✅ `matchRegex: ""` → Valid (empty string matches empty)

**Error Message**:
```
Rule 'Sort by Score': Invalid regex pattern in matcher.matchRegex:
unterminated character set at position 5
```

---

### C5: Action Must Be symlink or copy

**Constraint**: `action` field must be exactly `"symlink"` or `"copy"`

**Validation**:
```python
if rule.action not in ("symlink", "copy"):
    errors.append(f"Invalid action '{rule.action}'. Must be 'symlink' or 'copy'.")
```

**Edge Cases**:
- ❌ `action: "link"` → Validation error (must be "symlink")
- ❌ `action: "move"` → Validation error (not supported)
- ✅ `action: "symlink"` → Valid
- ✅ `action: "copy"` → Valid

---

### C6: Collision Strategy Must Be Valid

**Constraint**: `collisionStrategy` must be `"skip"`, `"increment"`, or `"overwrite"`

**Validation**:
```python
if rule.collision_strategy not in ("skip", "increment", "overwrite"):
    errors.append(
        f"Invalid collision_strategy '{rule.collision_strategy}'. "
        "Must be 'skip', 'increment', or 'overwrite'."
    )
```

**Edge Cases**:
- ❌ `collisionStrategy: "replace"` → Validation error (use "overwrite")
- ❌ `collisionStrategy: "rename"` → Validation error (use "increment")
- ✅ `collisionStrategy: "skip"` → Valid
- ✅ `collisionStrategy: "increment"` → Valid
- ✅ `collisionStrategy: "overwrite"` → Valid

---

## Runtime Constraints

### R1: Output File Must Exist

**Constraint**: File referenced by `context.metadata["output_path"]` must exist

**Handling**:
```python
if not output_path or not Path(output_path).exists():
    logger.warning(f"Output file not found, skipping: {output_path}")
    return  # Skip this file
```

**Edge Cases**:
- ⚠️ `output_path` is None → Skip file, log warning
- ⚠️ `output_path` exists but is deleted before organization → Skip, log warning
- ✅ `output_path` exists → Proceed with organization

**Warning Message**:
```
Output file not found, skipping: /path/to/CheckedOMRs/image_CHECKED.jpg
```

---

### R2: Field Values Must Be Strings

**Constraint**: All field values used in pattern formatting must be strings

**Handling**:
```python
formatting_fields = {
    "score": str(result["score"]),           # Convert to string
    "is_multi_marked": str(result["is_multi_marked"]),
    **result["omr_response"],  # Assume already strings
}
```

**Edge Cases**:
- ⚠️ `score` is `95.5` (float) → Converted to `"95.5"`
- ⚠️ `is_multi_marked` is `True` (bool) → Converted to `"True"`
- ✅ All OMR fields are strings → Used directly

---

### R3: Missing Fields in Pattern

**Constraint**: If a field referenced in pattern is missing at runtime, pattern resolution fails

**Handling**:
```python
try:
    formatted = pattern.format(**fields)
except KeyError as e:
    logger.warning(f"Pattern references undefined field: {e}")
    return None  # Pattern resolution fails
```

**Edge Cases**:
- ⚠️ Pattern `{roll}` but `roll=""` (empty) → Produces empty segment
- ❌ Pattern `{roll}` but `roll` key missing → KeyError, skip file
- ✅ All fields present → Pattern resolves successfully

**Warning Message**:
```
Pattern references undefined field: 'roll'
Skipping due to collision or pattern error: image_CHECKED.jpg
```

---

### R4: Invalid Characters in Field Values

**Constraint**: Field values may contain invalid filename characters

**Handling** (Automatic Sanitization):
```python
# Invalid characters: < > : " / \ | ? *
sanitized = re.sub(r'[<>:"|?*\\]', "_", part)
sanitized = re.sub(r"_+", "_", sanitized)
sanitized = sanitized.strip("_ ")
```

**Edge Cases**:
- ✅ `code="A/B"` → Sanitized to `"A_B"`
- ✅ `roll="12:45"` → Sanitized to `"12_45"`
- ✅ `name="Test?"` → Sanitized to `"Test"`
- ✅ `batch="Morning  Batch"` → Sanitized to `"Morning_Batch"`

**Examples**:
```python
"Code: A/B?" → "Code_A_B"
"Roll: 12:34|56?" → "Roll_12_34_56"
"<Name>" → "Name"
```

---

### R5: Empty Path Components

**Constraint**: Sanitization may produce empty path components

**Handling**:
```python
if sanitized:  # Only add non-empty parts
    sanitized_parts.append(sanitized)
```

**Edge Cases**:
- ⚠️ Pattern `{code}//{roll}` → Empty middle component dropped
- ⚠️ Field value is all invalid chars `":::"` → Empty, component dropped
- ⚠️ Pattern produces `"folder//file"` → Becomes `"folder/file"`

**Example**:
```python
# Pattern: "{batch}/{section}/{roll}"
# Fields: {"batch": "A", "section": "", "roll": "12345"}
# Result: "A/12345" (section component dropped)
```

---

### R6: File Already Exists (Collision)

**Constraint**: Destination file may already exist

**Handling** (Strategy-Dependent):

**Skip Strategy**:
```python
if path.exists():
    logger.debug(f"File exists, skipping: {path.name}")
    return None
```

**Increment Strategy**:
```python
counter = 1
while counter < 9999:
    new_name = f"{stem}_{counter:03d}{suffix}"
    new_path = parent / new_name
    if not new_path.exists():
        return new_path
    counter += 1
```

**Overwrite Strategy**:
```python
if path.exists():
    logger.debug(f"File exists, will overwrite: {path.name}")
    return path  # Allow overwrite
```

**Edge Cases**:
- ✅ File doesn't exist → Use path as-is
- ⚠️ File exists + skip → Return None, skip file
- ⚠️ File exists + increment → Try `_001`, `_002`, ..., up to `_9998`
- ⚠️ File exists + overwrite → Use path, file will be overwritten
- ❌ 9999+ collisions with increment → Give up, return None

**Warning Messages**:
```
File exists, skipping: roll_12345.jpg
File exists, using incremented name: roll_12345_001.jpg
Too many collisions for roll_12345, giving up
```

---

### R7: Symlink Creation Fails

**Constraint**: Symlink creation may fail (Windows permissions, unsupported filesystem)

**Handling** (Automatic Fallback):
```python
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
```

**Edge Cases**:
- ✅ Unix/Mac + permissions → Creates relative symlink
- ⚠️ Windows + no admin → Falls back to copy
- ⚠️ Network filesystem doesn't support symlinks → Falls back to copy
- ✅ Relative symlink works → Uses relative path
- ⚠️ Relative symlink fails → Uses absolute path

**Platform-Specific**:
```python
# Windows
try:
    symlink()  # Requires admin or Developer Mode
except OSError:
    copy2()    # Automatic fallback

# Unix/Mac
symlink()  # Always works
```

---

### R8: Copy Operation Fails

**Constraint**: File copy may fail (permissions, disk space, etc.)

**Handling**:
```python
try:
    shutil.copy2(str(source_path), str(dest_path))
except Exception as e:
    logger.error(f"Failed to copy {source_path.name}: {e}")
    self.file_operations.append({
        "action": "failed",
        "error": str(e)
    })
```

**Edge Cases**:
- ❌ Disk full → Copy fails, logged as error
- ❌ Permission denied → Copy fails, logged as error
- ❌ Source file deleted mid-copy → Copy fails, logged as error
- ✅ Copy succeeds → Tracked as successful operation

**Error Message**:
```
Failed to copy image_CHECKED.jpg: [Errno 28] No space left on device
```

---

### R9: Extension Preservation Logic

**Constraint**: Pattern may or may not include file extension

**Handling**:
```python
if original_path and not resolved_path.suffix:
    original_ext = Path(original_path).suffix
    resolved_path = resolved_path.with_suffix(original_ext)
```

**Edge Cases**:
- ✅ Pattern: `{roll}`, original: `.jpg` → Result: `12345.jpg`
- ✅ Pattern: `{roll}.png`, original: `.jpg` → Result: `12345.png` (override)
- ✅ Pattern: `{roll}.`, original: `.jpg` → Result: `12345..jpg` (double dot)
- ⚠️ Original has no extension → No extension added
- ⚠️ Original: `.tar.gz` → Only preserves `.gz` (last extension)

**Examples**:
```python
# Pattern: "output/{roll}"
# Original: "image.jpg"
# Result: "output/12345.jpg"

# Pattern: "output/{roll}.png"
# Original: "image.jpg"
# Result: "output/12345.png" (forced PNG)

# Pattern: "output/{roll}"
# Original: "image.tar.gz"
# Result: "output/12345.gz" (only last extension)
```

---

### R10: Directory Creation Fails

**Constraint**: Parent directory creation may fail

**Handling**:
```python
try:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
except Exception as e:
    logger.error(f"Failed to create directory: {e}")
```

**Edge Cases**:
- ❌ Permission denied → Fails, logged as error
- ❌ Path is too long (Windows MAX_PATH) → Fails, logged as error
- ✅ Directory already exists (`exist_ok=True`) → No error
- ✅ Creates nested directories (`parents=True`) → All created

---

### R11: Rule Matcher Fails

**Constraint**: Matcher evaluation may fail (missing field, regex error, etc.)

**Handling**:
```python
try:
    formatted_string = format_string.format(**formatting_fields)
    if re.search(match_regex, formatted_string):
        return rule
except KeyError as e:
    logger.warning(f"Rule '{rule.name}' references undefined field: {e}")
except Exception as e:
    logger.warning(f"Error evaluating rule '{rule.name}': {e}")
```

**Edge Cases**:
- ⚠️ formatString references missing field → Log warning, skip rule
- ⚠️ Regex has error (shouldn't happen if validated) → Log warning, skip rule
- ✅ All fields present, regex valid → Match succeeds or fails

**Warning Messages**:
```
Rule 'Sort by Booklet' references undefined field: 'booklet_code'
Error evaluating rule 'Sort by Score': invalid literal for int()
```

---

### R12: No Results Collected

**Constraint**: `finish_processing_directory()` called but no results collected

**Handling**:
```python
if not self.config.enabled or not self.results:
    return  # Early exit, nothing to do
```

**Edge Cases**:
- ✅ No files processed → Early exit, no overhead
- ✅ All files failed detection → No results, early exit
- ✅ Organization disabled mid-processing → Early exit

---

### R13: Thread Safety During Collection

**Constraint**: Multiple threads appending to `results` list

**Handling**:
```python
with self._lock:
    self.results.append({...})
```

**Edge Cases**:
- ✅ 10 threads appending simultaneously → Lock serializes, all succeed
- ✅ Thread crashes during append → Lock released, other threads continue
- ⚠️ Very high thread count → Lock contention (negligible for append)

---

## Performance Constraints

### P1: Maximum Increment Limit

**Constraint**: Increment collision strategy has hard limit of 9999

**Rationale**: Safety limit to prevent infinite loops

**Handling**:
```python
if counter == 9999:
    logger.error(f"Too many collisions for {stem}, giving up")
    return None
```

**Edge Cases**:
- ⚠️ 9999 existing files with same name → 10000th file skipped
- ✅ < 9999 collisions → All files organized
- ⚠️ User creates pattern that generates same name for all files → Hits limit

---

### P2: Memory Usage for Results

**Constraint**: All results stored in memory until organization phase

**Impact**:
- Each result: ~1KB (context, paths, response)
- 10,000 files → ~10MB memory
- 100,000 files → ~100MB memory

**Mitigation**:
- Only store necessary fields (not full images)
- Copy `omr_response` to avoid holding context references

**Edge Cases**:
- ⚠️ Processing millions of files → High memory usage
- ✅ Typical use (< 10,000 files) → Negligible impact

---

### P3: Filesystem Limits

**Constraint**: Filesystems have limits on path length, files per directory, etc.

**Platform Limits**:
- **Windows**: 260 character path limit (MAX_PATH)
- **Linux/Mac**: 4096 character path limit
- **Files per directory**: Varies by filesystem (ext4: unlimited, FAT32: 65,536)

**Handling**:
```python
# No explicit handling, will fail with OS error
try:
    dest_path.symlink_to(source_path)
except OSError as e:
    logger.error(f"Failed to create link: {e}")
```

**Edge Cases**:
- ❌ Path > 260 chars on Windows → Fails
- ❌ > 65,536 files in one directory (FAT32) → Fails
- ⚠️ Long field values → May produce long paths

**Mitigation**:
- Users should keep patterns short
- Sanitization reduces path length (removes invalid chars)

---

## Browser-Specific Constraints

### B1: No Symlinks in Browser

**Constraint**: File API doesn't support symbolic links

**Handling**:
```typescript
// Always use "copy" action in browser
if (action === "symlink") {
  action = "copy";  // Force copy for browser
}
```

**Edge Cases**:
- ⚠️ Config specifies `symlink` → Ignored, always copies
- ✅ Config specifies `copy` → Works as expected

---

### B2: File System Access API Limited

**Constraint**: File System Access API only in Chrome/Edge

**Handling**:
```typescript
if ("showDirectoryPicker" in window) {
  // Use File System Access API
} else {
  // Fallback to ZIP download
}
```

**Edge Cases**:
- ⚠️ Firefox/Safari → No directory picker, use ZIP
- ⚠️ HTTP (not HTTPS) → API not available, use ZIP
- ✅ Chrome/Edge + HTTPS → Can use directory picker

---

### B3: IndexedDB Storage Limits

**Constraint**: IndexedDB has storage quotas (varies by browser)

**Limits**:
- **Chrome**: ~60% of disk space (min 400MB)
- **Firefox**: ~50% of disk space (max 2GB)
- **Safari**: 1GB max

**Handling**:
```typescript
try {
  await db.organizedFiles.put({blob, path});
} catch (e) {
  if (e.name === "QuotaExceededError") {
    alert("Storage quota exceeded");
  }
}
```

**Edge Cases**:
- ❌ Organizing 10,000 large images → May hit quota
- ✅ Organizing small images → No issues
- ⚠️ User has low disk space → Hit quota sooner

---

### B4: ZIP Size Limits

**Constraint**: In-memory ZIP generation limited by browser memory

**Limits**:
- Typical browser: 2-4GB heap
- Must fit all files in memory to generate ZIP

**Handling**:
```typescript
const zip = new JSZip();
for (const file of organizedFiles) {
  zip.file(file.path, file.blob);  // All in memory
}
const zipBlob = await zip.generateAsync({type: "blob"});
```

**Edge Cases**:
- ❌ Organizing 10GB of images → Out of memory
- ✅ Organizing < 1GB → Works fine
- ⚠️ Mobile browser → Lower memory limits

**Mitigation**:
- Stream ZIP generation with jszip-utils
- Download in batches
- Warn user about size limits

---

## Summary of All Constraints

| ID | Constraint | Type | Severity | Handling |
|----|-----------|------|----------|----------|
| C1 | Enabled flag | Config | Info | Early exit if disabled |
| C2 | Unique priorities | Config | Error | Validation fails |
| C3 | Valid fields | Config | Error | Validation fails |
| C4 | Valid regex | Config | Error | Validation fails |
| C5 | Valid action | Config | Error | Validation fails |
| C6 | Valid collision strategy | Config | Error | Validation fails |
| R1 | Output file exists | Runtime | Warning | Skip file |
| R2 | Fields are strings | Runtime | Info | Auto-convert |
| R3 | Missing fields | Runtime | Warning | Skip file |
| R4 | Invalid chars | Runtime | Info | Auto-sanitize |
| R5 | Empty components | Runtime | Info | Auto-drop |
| R6 | File collision | Runtime | Info | Strategy-based |
| R7 | Symlink fails | Runtime | Info | Fallback to copy |
| R8 | Copy fails | Runtime | Error | Log and track |
| R9 | Extension preservation | Runtime | Info | Auto-add |
| R10 | Directory creation fails | Runtime | Error | Log and fail |
| R11 | Matcher fails | Runtime | Warning | Skip rule |
| R12 | No results | Runtime | Info | Early exit |
| R13 | Thread safety | Runtime | Critical | Lock-protected |
| P1 | Increment limit | Performance | Warning | Hard limit 9999 |
| P2 | Memory usage | Performance | Info | Store minimal data |
| P3 | Filesystem limits | Performance | Error | OS-dependent |
| B1 | No symlinks | Browser | Info | Force copy |
| B2 | API availability | Browser | Info | Fallback to ZIP |
| B3 | Storage quota | Browser | Error | Catch and alert |
| B4 | ZIP size | Browser | Error | OOM |
