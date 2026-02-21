# File Organization - Design Decisions

## Overview

This document explains the key design decisions made in the File Organization system, including the rationale behind architectural choices, trade-offs considered, and alternatives rejected.

## Decision 1: Two-Phase Processing (Collection vs Organization)

### Decision
Separate file processing into two distinct phases:
1. **Collection Phase**: Thread-safe collection during parallel processing
2. **Organization Phase**: Sequential organization after all files processed

### Rationale

**Why Not Organize During Processing?**
- File system operations are NOT thread-safe (race conditions on mkdir, symlink, collision detection)
- Rule matching could be expensive; don't want to slow down parallel processing
- Collision detection requires checking file existence (not atomic across threads)

**Benefits of Two-Phase**:
- **Eliminates Race Conditions**: No concurrent file system access
- **Predictable Behavior**: Collision handling is deterministic
- **Simple Mental Model**: Collection is parallel, organization is sequential
- **Performance**: Lightweight collection doesn't slow down processing pipeline

**Trade-offs**:
- Memory usage: Must store all results in memory until organization phase
- Delayed feedback: Users don't see organization until all processing completes

**Alternatives Considered**:
1. ❌ **Organize during processing with locks** - Too slow, locks would serialize organization
2. ❌ **Organize in separate thread pool** - Complex coordination, still has race conditions
3. ✅ **Current approach** - Simple, safe, performant

### Browser Impact
```typescript
// Browser can use same pattern (no Web Worker complexity for organization)
async function processAllFiles() {
  // Phase 1: Parallel processing with Web Workers
  const results = await Promise.all(
    files.map(file => processInWorker(file))
  );

  // Phase 2: Sequential organization in main thread
  await organizeFiles(results);
}
```

---

## Decision 2: Dynamic Patterns with Field Placeholders

### Decision
Use Python's `str.format()` syntax for patterns instead of custom DSL or regex-based substitution.

**Pattern Example**: `booklet_{code}/roll_{roll}.jpg`

### Rationale

**Why str.format()?**
- **Familiar Syntax**: Python developers already know it
- **Built-in Validation**: KeyError if field missing
- **Simple Implementation**: One line of code (`pattern.format(**fields)`)
- **Clear Error Messages**: Automatic exception with missing field name

**Benefits**:
- **Readability**: `{field}` is clearer than `$field` or `#{field}`
- **IDE Support**: Syntax highlighting for format strings
- **Type Safety**: Can validate fields at config load time

**Trade-offs**:
- Limited to string substitution (no expressions like `{score + 10}`)
- No conditional logic in patterns (handled by rules instead)

**Alternatives Considered**:
1. ❌ **Regex-based substitution** - Hard to validate, error-prone
2. ❌ **Template engine (Jinja2)** - Overkill, adds dependency
3. ❌ **Custom DSL** - More complexity, learning curve
4. ✅ **str.format()** - Simple, familiar, built-in

### Browser Implementation
```typescript
// JavaScript equivalent
function resolvePattern(pattern: string, fields: Record<string, string>): string {
  return pattern.replace(/\{(\w+)\}/g, (match, key) => {
    if (!(key in fields)) {
      throw new Error(`Field '${key}' not found`);
    }
    return fields[key];
  });
}
```

---

## Decision 3: Priority-Based Rule Matching (First Match Wins)

### Decision
- Rules are sorted by priority (lower number = higher priority)
- First matching rule is used (stop on first match)
- If no rules match, use `defaultPattern`

### Rationale

**Why First Match Wins?**
- **Deterministic**: Always same result for same input
- **Performance**: Stop searching after first match
- **Intuitive**: Users can reason about rule order

**Why Priority Numbers Instead of Order?**
- **Explicit Intent**: Priority is visible in config
- **Refactoring Safe**: Can reorder rules in JSON without changing behavior
- **Validation**: Can detect duplicate priorities

**Benefits**:
- **Clear Semantics**: High-priority rules take precedence
- **Efficient**: O(r) matching where r = number of rules (early exit)
- **Flexible**: Can have catch-all rule with low priority

**Trade-offs**:
- Must be careful with priority numbering
- Can't have multiple rules apply to same file (intentional)

**Alternatives Considered**:
1. ❌ **All matching rules apply** - Confusing, what if conflicts?
2. ❌ **Last match wins** - Counterintuitive
3. ❌ **Array order determines priority** - Fragile during refactoring
4. ✅ **Explicit priority + first match** - Clear and deterministic

### Example
```json
{
  "rules": [
    {
      "name": "High Scorers",
      "priority": 1,
      "matcher": {"formatString": "{score}", "matchRegex": "^(9[0-9]|100)$"}
    },
    {
      "name": "All Students",
      "priority": 2,
      "matcher": {"formatString": "{roll}", "matchRegex": ".*"}
    }
  ]
}
```
A file with `score=95` matches Rule 1 (priority 1) and stops. Never checks Rule 2.

---

## Decision 4: Matcher System (formatString + matchRegex)

### Decision
Each rule has a matcher with two components:
- `formatString`: Template to format using fields
- `matchRegex`: Regular expression to match against formatted string

### Rationale

**Why Two-Step Matching?**
- **Flexibility**: Can match on single field (`{code}`) or combined (`{batch}_{section}`)
- **Powerful**: Regex supports complex patterns (ranges, alternatives, etc.)
- **Composable**: formatString handles field access, regex handles logic

**Benefits**:
- **Expressive**: Can express complex rules like "scores 90-100" or "codes A-D"
- **Validated**: formatString checked at config load, regex compiled once
- **Debuggable**: Can see formatted string in logs if match fails

**Trade-offs**:
- Two-step process is slightly more complex than single regex
- Users must understand both Python format syntax and regex

**Alternatives Considered**:
1. ❌ **Direct regex on fields** - How to combine multiple fields?
2. ❌ **JavaScript-style expressions** - Security risk, hard to validate
3. ❌ **Custom query language** - Too complex for simple use case
4. ✅ **formatString + regex** - Balance of power and simplicity

### Example Matchers
```json
// Match booklet codes A-D
{
  "formatString": "{code}",
  "matchRegex": "^[A-D]$"
}

// Match scores 90-100
{
  "formatString": "{score}",
  "matchRegex": "^(9[0-9]|100)$"
}

// Match batch + section combination
{
  "formatString": "{batch}_{section}",
  "matchRegex": "^morning_(A|B|C)$"
}
```

---

## Decision 5: Symlink by Default, Copy as Fallback

### Decision
- Default action is `symlink` (symbolic link)
- Automatically falls back to `copy` on Windows if symlink fails
- Users can explicitly choose `copy` action if needed

### Rationale

**Why Symlink by Default?**
- **Space Efficient**: No data duplication (organized view is just links)
- **Fast**: Creating symlink is O(1), copying is O(file size)
- **Reflects Relationship**: User can see organized files point to originals
- **Safe**: Original files in `CheckedOMRs/` are never modified

**Why Automatic Fallback?**
- **Windows Compatibility**: Symlinks require admin privileges on Windows
- **User Experience**: Works out of the box on all platforms
- **Transparent**: Logs indicate when fallback happens

**Benefits**:
- **Best Default**: Efficient for most users (Unix/Mac developers)
- **Universal**: Works on all platforms
- **Flexible**: Users can opt for copy if they prefer

**Trade-offs**:
- Symlinks don't work if original files are moved/deleted
- Broken links if `CheckedOMRs/` is cleaned up

**Alternatives Considered**:
1. ❌ **Copy by default** - Wastes disk space, slower
2. ❌ **Platform detection** - Brittle, what about non-admin Windows?
3. ❌ **Force users to choose** - Extra configuration burden
4. ✅ **Symlink with fallback** - Best of both worlds

### Browser Reality
```typescript
// Browser: Always "copy" (store blob in IndexedDB or ZIP)
// No symlink concept in browser file systems
const organized = {
  path: "booklet_A/roll_12345.jpg",
  blob: imageBlob.slice(0)  // Copy blob
};
```

---

## Decision 6: Collision Strategies (Skip, Increment, Overwrite)

### Decision
Three collision strategies when destination file exists:
1. `skip`: Don't create file, log warning
2. `increment`: Append `_001`, `_002`, etc.
3. `overwrite`: Replace existing file

### Rationale

**Why Three Options?**
- **Different Use Cases**:
  - `skip` for deduplication
  - `increment` for keeping all versions
  - `overwrite` for updating with latest
- **User Control**: Let users decide what happens on collision
- **Safe Default**: `skip` prevents accidental overwrites

**Why Increment Format `_001`?**
- **Sortable**: Alphabetical sort matches numeric order
- **Fixed Width**: Up to 999 duplicates (enough for most cases, limit at 9999)
- **Readable**: Clear that these are duplicates

**Benefits**:
- **Covers Common Scenarios**: Most users need one of these three
- **Predictable**: Users know exactly what will happen
- **Debuggable**: Summary shows how many skipped/incremented

**Trade-offs**:
- Limited to 9999 duplicates (intentional safety limit)
- No custom increment patterns (e.g., timestamps)

**Alternatives Considered**:
1. ❌ **Timestamp suffixes** - Not sortable, clutters filenames
2. ❌ **UUID suffixes** - Unreadable, overkill
3. ❌ **Always overwrite** - Dangerous default
4. ✅ **Three simple strategies** - Covers 99% of use cases

---

## Decision 7: Extension Preservation

### Decision
If pattern doesn't include extension, preserve original file's extension.

**Example**:
- Pattern: `booklet_{code}/{roll}`
- Original: `image_CHECKED.jpg`
- Result: `booklet_A/12345.jpg` (`.jpg` preserved)

### Rationale

**Why Preserve Extension?**
- **User Expectation**: Files should keep their type
- **Convenience**: Don't force users to specify `.jpg` in every pattern
- **Flexibility**: Can override by including extension in pattern

**Benefits**:
- **Less Verbose Patterns**: `{roll}` instead of `{roll}.jpg`
- **Works with Mixed Extensions**: Same pattern works for `.jpg` and `.png`
- **Explicit Override**: Pattern `{roll}.png` forces PNG extension

**Trade-offs**:
- Slightly magical behavior (might surprise users initially)
- Documentation burden to explain the rule

**Alternatives Considered**:
1. ❌ **Always require extension in pattern** - Verbose, error-prone
2. ❌ **Drop extension entirely** - Breaks file type association
3. ❌ **Add separate `extension` field** - More configuration complexity
4. ✅ **Smart preservation** - Convenient with escape hatch

---

## Decision 8: Organized Directory Outside CheckedOMRs

### Decision
Create `outputs/organized/` **separate** from `outputs/CheckedOMRs/`.

**Structure**:
```
outputs/
  ├── CheckedOMRs/      # Original processed files
  └── organized/        # Organized view (symlinks/copies)
```

### Rationale

**Why Separate Directory?**
- **Non-Destructive**: Original structure is preserved
- **Optional Feature**: Can disable organization without affecting core output
- **Clear Separation**: Users know `CheckedOMRs/` is authoritative
- **Cleanup Safe**: Can delete `organized/` and regenerate

**Benefits**:
- **Safety**: Can't accidentally corrupt original outputs
- **Flexibility**: Multiple organization schemes possible (future: multiple rules)
- **Performance**: Can skip organization entirely if disabled

**Trade-offs**:
- Duplication if using `copy` action (disk space)
- Two places to look for files

**Alternatives Considered**:
1. ❌ **Organize inside CheckedOMRs/** - Mixes original and organized
2. ❌ **Replace CheckedOMRs/** - Destructive, can't regenerate
3. ❌ **Separate directory per rule** - Too fragmented
4. ✅ **Single organized/ directory** - Clear, safe, simple

---

## Decision 9: Thread-Safe Collection with Lock

### Decision
Use `threading.Lock()` to protect `results.append()` during parallel processing.

### Rationale

**Why Lock?**
- **List append is NOT atomic** in Python's CPython (despite GIL)
- **Data race possible**: Multiple threads appending simultaneously
- **Simple Solution**: Lock is lightweight for append-only operation

**Benefits**:
- **Correct**: No data races, guaranteed thread safety
- **Simple**: Single lock, minimal contention
- **Fast**: Lock held for microseconds (just during append)

**Trade-offs**:
- Tiny serialization point (but append is O(1) so negligible)

**Alternatives Considered**:
1. ❌ **No lock (rely on GIL)** - Not safe, GIL doesn't protect all operations
2. ❌ **Queue.Queue** - Overkill for simple append, harder to iterate later
3. ❌ **Thread-local storage** - Complex to merge, hard to reason about
4. ✅ **Lock-protected list** - Simplest correct solution

### Browser Equivalent
```typescript
// JavaScript: Single-threaded, no lock needed
// If using Web Workers, use message passing instead
class FileOrganizer {
  private results: Result[] = [];

  // Called from main thread (after worker messages received)
  addResult(result: Result): void {
    this.results.push(result);  // No lock needed
  }
}
```

---

## Decision 10: Validation at Config Load Time

### Decision
Validate configuration when loading `config.json`, **before** processing any files.

**Validated**:
- All fields referenced in patterns exist in template
- Regex patterns are valid
- Priority numbers are unique
- `score` field only used if evaluation enabled

### Rationale

**Why Fail Early?**
- **Better UX**: Users see errors immediately, not after processing 1000 files
- **Clear Messages**: Can provide helpful error with available fields
- **Fast Iteration**: Fix config and retry without waiting for processing

**Benefits**:
- **Prevents Silent Failures**: Won't silently skip organization due to bad config
- **Helpful Errors**: Points to exact problem (field name, rule name, etc.)
- **Config Confidence**: If validation passes, organization will work

**Trade-offs**:
- Slightly slower startup (negligible)
- Must keep validation logic in sync with runtime logic

**Alternatives Considered**:
1. ❌ **Lazy validation (fail on first use)** - Poor UX, wastes processing time
2. ❌ **Best-effort (skip invalid rules)** - Silent failures, confusing
3. ❌ **No validation** - Runtime errors, hard to debug
4. ✅ **Fail-fast validation** - Best user experience

### Example Validation Error
```
ERROR: File grouping configuration has errors:
  - Rule 'Sort by Booklet' destination_pattern: Field '{booklet_code}'
    not found in template. Available fields: {code}, {roll}, {batch}
  - Rule 'Sort by Score' pattern uses {score} but evaluation.json not found.
    Either add evaluation.json or remove {score} from pattern.

File organization will be DISABLED for this directory.
```

---

## Decision Summary Table

| Decision | Choice | Key Rationale | Browser Adaptation |
|----------|--------|---------------|-------------------|
| Processing Phases | Collection + Organization | Avoid race conditions | Same pattern works |
| Pattern Syntax | `str.format()` | Familiar, built-in | Use `replace()` |
| Rule Matching | Priority + First Match | Deterministic, efficient | Same logic |
| Matcher System | formatString + regex | Flexible, powerful | Same logic |
| Default Action | Symlink | Space efficient | Always copy (blobs) |
| Collision Handling | Skip/Increment/Overwrite | Covers all use cases | Same strategies |
| Extension | Auto-preserve | Convenience | Same logic |
| Output Location | Separate `organized/` | Non-destructive | IndexedDB or ZIP |
| Thread Safety | Lock for append | Correctness | N/A (single-threaded) |
| Validation | Fail-fast at load | Better UX | Same validation |

---

## Future Considerations

### Potential Enhancements
1. **Multiple Organization Schemes**: Allow multiple organizer instances with different rule sets
2. **Dynamic defaultPattern**: Per-rule defaults instead of global
3. **Custom Collision Format**: User-defined increment pattern (e.g., timestamps)
4. **Conditional Patterns**: `{score if score > 90 else "low"}`
5. **Hierarchical Validation**: Warn on overlapping rules

### Not Planned
- ❌ **In-place organization**: Too risky, violates non-destructive principle
- ❌ **External script hooks**: Security risk, platform-dependent
- ❌ **Database-backed organization**: Overkill for simple file grouping
