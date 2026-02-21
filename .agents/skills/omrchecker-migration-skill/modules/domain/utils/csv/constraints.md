# CSV Utils Constraints

**Module**: Domain - Utils - CSV
**Python Reference**: `src/utils/csv.py`
**Last Updated**: 2026-02-21

---

## Input Constraints

### file_path Parameter

**Type**: `Path | IO[str]`
**Description**: Target CSV file for writing

**Constraints**:
```python
# Valid inputs
file_path = Path("outputs/Results/Results.csv")    # Path object
file_path = "outputs/Results/Results.csv"          # String path
file_path = open("results.csv", "a")               # File handle

# Invariants
assert file_path is not None
# If Path: directory must exist (file created if missing)
# If IO: must be opened in append mode ("a" or "a+")
```

**File Creation**:
```python
# If file doesn't exist → pandas creates it
thread_safe_csv_append("new_file.csv", ["data"])
# Result: new_file.csv created with one row

# If directory doesn't exist → raises FileNotFoundError
thread_safe_csv_append("nonexistent/dir/file.csv", ["data"])
# ❌ Error: Parent directory must exist
```

**File Permissions**:
- Must have write permission to file and directory
- File must not be locked by another process
- Concurrent writes from other processes will cause corruption

**Path Format**:
```python
# Prefer absolute paths
file_path = Path("/absolute/path/to/results.csv")  # ✓ Good

# Relative paths work but context-dependent
file_path = Path("results.csv")  # ✓ Works (relative to CWD)

# Cross-platform paths
file_path = Path("outputs") / "Results" / "Results.csv"  # ✓ Good
```

### data_line Parameter

**Type**: `list`
**Description**: Row data to append (columns)

**Constraints**:
```python
# Valid
data_line = ["col1", "col2", "col3"]              # Strings
data_line = ["text", 123, 45.6]                   # Mixed types
data_line = []                                     # Empty (valid)

# Edge cases
data_line = ["value with,comma"]                  # ✓ Quoted in CSV
data_line = ["value with\"quote"]                 # ✓ Escaped in CSV
data_line = ["value\nwith\nnewlines"]             # ✓ Quoted in CSV
data_line = [None]                                 # ✓ Becomes empty string
data_line = ["", "", ""]                           # ✓ Empty strings valid

# Invalid
data_line = "not a list"                           # ❌ TypeError
data_line = None                                   # ❌ TypeError
```

**Type Conversion**:
```python
# All values converted to strings via dtype=str
data_line = [123, 45.6, True, None]
# Becomes: ["123", "45.6", "True", ""]

# Preserves string representations
data_line = ["01234"]  # Leading zero preserved
# Output: "01234" (not 1234)
```

**Length Constraints**:
```python
# No enforced maximum length
len(data_line) >= 0  # Can be empty

# Typical lengths
len(data_line) = 4-50   # Common range
len(data_line) = 100+   # Large but valid

# Should match CSV column count (not enforced)
# First row: 5 columns
# Second row: 7 columns  # Valid but creates unaligned CSV
```

**Special Characters**:
```python
# Commas, quotes, newlines handled automatically
data_line = [
    'text with,comma',      # → "text with,comma"
    'text with"quote',      # → "text with""quote" (escaped)
    'text with\nnewline',   # → "text with\nnewline" (quoted)
]

# All properly escaped in CSV output
```

### quoting Parameter

**Type**: `int` (from csv module)
**Default**: `csv.QUOTE_NONNUMERIC`

**Valid Values**:
```python
from csv import QUOTE_NONNUMERIC, QUOTE_ALL, QUOTE_MINIMAL, QUOTE_NONE

# QUOTE_NONNUMERIC (default, recommended)
quoting = QUOTE_NONNUMERIC
# Quotes all non-numeric values
# "text",123,"more text"

# QUOTE_ALL
quoting = QUOTE_ALL
# Quotes everything
# "text","123","more text"

# QUOTE_MINIMAL
quoting = QUOTE_MINIMAL
# Quotes only when necessary (commas, quotes)
# text,123,more text

# QUOTE_NONE
quoting = QUOTE_NONE
# No quoting (dangerous - can corrupt CSV)
# ❌ Not recommended
```

**Recommendation**:
```python
# Always use QUOTE_NONNUMERIC (default)
thread_safe_csv_append(file_path, data_line)
# Safe, consistent, readable
```

---

## Output Constraints

### CSV File Format

**Encoding**: UTF-8 (pandas default)
**Line Ending**: Platform-dependent (`\n` on Unix, `\r\n` on Windows)
**Quoting**: Non-numeric values quoted by default

**Example Output**:
```csv
"file1.jpg","/input/file1.jpg","/output/file1.jpg","95","A","B","C"
"file2.jpg","/input/file2.jpg","/output/file2.jpg","80","A","C","D"
```

**Invariants**:
- Each `thread_safe_csv_append()` call adds exactly one row
- Rows are newline-terminated
- Column count per row can vary (CSV spec allows, but not ideal)

---

## Threading Constraints

### CSV_WRITE_LOCK

**Type**: `threading.Lock`
**Scope**: Module-level global
**Purpose**: Serialize CSV writes across threads

**Constraints**:
```python
# Lock is global to csv.py module
from src.utils.csv import CSV_WRITE_LOCK

# Lock behavior
CSV_WRITE_LOCK.acquire()  # Blocks if locked by another thread
# ... perform CSV write ...
CSV_WRITE_LOCK.release()  # Unlock

# With context manager (automatic)
with CSV_WRITE_LOCK:
    # Only one thread executes here at a time
    df.to_csv(...)
```

**Invariants**:
- Lock must be acquired before CSV write
- Lock automatically released after write (context manager)
- No nested locking (not reentrant)
- No timeout (blocks indefinitely if needed)

### Thread Safety Guarantees

**Safe**:
```python
# Multiple threads calling thread_safe_csv_append()
# on the SAME file
thread1: thread_safe_csv_append(file, data1)
thread2: thread_safe_csv_append(file, data2)
thread3: thread_safe_csv_append(file, data3)
# ✓ All writes succeed, no corruption
```

**Unsafe**:
```python
# Direct pandas write without lock
thread1: pd.DataFrame(data1).to_csv(file, mode="a")
thread2: pd.DataFrame(data2).to_csv(file, mode="a")
# ❌ POTENTIAL CORRUPTION: Interleaved writes
```

**Unsafe**:
```python
# Multiple processes writing to same file
process1: thread_safe_csv_append(file, data1)
process2: thread_safe_csv_append(file, data2)
# ❌ CORRUPTION: Lock doesn't work across processes
# Use file locking (fcntl) for multi-process safety
```

### Deadlock Prevention

**No Deadlock Risk**: Single lock, no lock ordering issues.

```python
# Safe pattern (only one lock)
with CSV_WRITE_LOCK:
    thread_safe_csv_append(file1, data)  # Nested call OK
    # (Lock is reentrant in this implementation via context manager)

# Actually, Lock is NOT reentrant - avoid nested calls
with CSV_WRITE_LOCK:
    thread_safe_csv_append(file, data)  # ❌ DEADLOCK
    # Don't call thread_safe_csv_append() while holding lock
```

**Correct Pattern**:
```python
# Don't call thread_safe_csv_append() from within lock
# thread_safe_csv_append() handles locking internally

# Just call the function directly
thread_safe_csv_append(file1, data1)
thread_safe_csv_append(file2, data2)
# ✓ Safe, no deadlock
```

---

## Performance Constraints

### Time Complexity

**Per Write Operation**:
```python
# O(n) where n = len(data_line)
# Dominated by:
# 1. DataFrame creation: O(n)
# 2. Transpose: O(n)
# 3. CSV serialization: O(n)
# 4. File I/O: O(1) amortized
```

**Typical Timings**:
```python
# Small row (5 columns): ~0.5ms
# Medium row (20 columns): ~1ms
# Large row (100 columns): ~3ms

# Lock overhead: ~0.01ms (negligible)
```

### Memory Usage

**Per Write Operation**:
```python
# Memory allocation breakdown:
data_line = ["col1", "col2", ...]       # ~100 bytes (5 strings)
df = pd.DataFrame(data_line)            # ~500 bytes
df_T = df.T                             # ~500 bytes (copy)
csv_string = df_T.to_csv(...)           # ~200 bytes

# Total: ~1.3KB per write
# Freed after write completes
```

**Peak Memory** (parallel processing):
```python
# Worst case: All workers writing simultaneously
workers = 4
peak_memory = workers × 1.3KB = 5.2KB

# Negligible compared to:
# - Image data: ~1MB per image
# - Template data: ~100KB
# - Processing context: ~50KB
```

### Lock Contention

**Scenario Analysis**:

**Low Contention** (typical):
```python
# 4 workers, file processing takes 500ms, CSV write takes 1ms
contention = (1ms write time) / (500ms processing time) = 0.2%
# Workers rarely wait for lock
```

**Medium Contention**:
```python
# 8 workers, fast processing (100ms), 1ms write
contention = (1ms × 8 workers) / 100ms = 8%
# Occasional waits, minimal impact
```

**High Contention** (unusual):
```python
# 16 workers, very fast processing (10ms), 1ms write
contention = (1ms × 16) / 10ms = 160% (queue forms)
# Workers frequently wait
# Solution: Reduce workers or batch CSV writes
```

**Mitigation**:
```python
# If high contention observed:
# 1. Reduce worker count
max_workers = 4  # Instead of 16

# 2. Batch CSV writes
results = [process_file(f) for f in files]
for result in results:
    thread_safe_csv_append(file, result)
# Write after all processing (no contention)
```

---

## File System Constraints

### File Size Limits

**CSV File Growth**:
```python
# Example: 1000 files, 20 columns each, 50 bytes per column
rows = 1000
columns = 20
avg_bytes_per_value = 50
file_size = rows × columns × avg_bytes_per_value
          = 1000 × 20 × 50
          = 1,000,000 bytes
          = ~1MB

# Typical CSV sizes: 1MB-100MB
# Large batches: 100MB-1GB

# File system limits:
# - Most systems: 2TB+ (not a concern)
# - Excel import limit: ~1M rows (may need to split)
```

### Disk I/O Constraints

**Write Patterns**:
```python
# Append mode: Efficient for sequential writes
# pandas.to_csv(mode="a") → O(1) seek to end + write

# Performance factors:
# - SSD: ~500MB/s write (very fast)
# - HDD: ~100MB/s write (slower)
# - Network drive: Variable (can be bottleneck)
```

**I/O Throughput**:
```python
# CSV write: ~1KB per row
# Throughput: 1000 rows/s = 1MB/s

# Well below disk limits:
# SSD: 500MB/s (500× faster than CSV writes)
# HDD: 100MB/s (100× faster than CSV writes)

# Conclusion: I/O not a bottleneck for typical use
```

### File Locking

**OS-Level Locking**:
```python
# Python threading.Lock: In-process only
# Does NOT prevent:
# - Other processes writing to same file
# - User opening file in Excel during processing
# - Network file system race conditions

# For multi-process safety, use fcntl (Unix) or msvcrt (Windows)
import fcntl  # Unix

with open(file_path, "a") as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
    f.write(csv_data)
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Unlock
```

**Current Implementation**:
```python
# OMRChecker: Single process + multi-threading
# threading.Lock is sufficient
# No inter-process locking needed
```

---

## Data Integrity Constraints

### Row Ordering

**Parallel Processing**:
```python
# Files complete in non-deterministic order
# CSV must maintain input file order

# WRONG: Write as completed
for future in as_completed(futures):
    result = future.result()
    thread_safe_csv_append(file, result["data"])
# ❌ CSV order doesn't match input order

# CORRECT: Sort then write
results = [future.result() for future in as_completed(futures)]
results.sort(key=lambda r: r["file_counter"])
for result in results:
    thread_safe_csv_append(file, result["data"])
# ✓ CSV order matches input order
```

**Invariant**: CSV row N corresponds to input file N.

### Atomic Writes

**Threading Atomicity**:
```python
# Within Python process:
with CSV_WRITE_LOCK:
    df.to_csv(file, mode="a", ...)
# ✓ Atomic: Lock ensures complete row written

# No partial rows possible from threading
```

**File System Atomicity**:
```python
# File system: Writes may not be atomic at byte level
# Small writes (<4KB): Usually atomic on most systems
# Large writes (>4KB): May be split across disk sectors

# CSV row size: Typically <1KB
# → Effectively atomic on modern systems
```

**Power Failure**:
```python
# If power lost during write:
# - Partial row may be written
# - File may be truncated
# - No transaction guarantees

# Mitigation: Write to temp file, then rename
temp_file = "results.csv.tmp"
final_file = "results.csv"

# Write all data to temp
for row in data:
    thread_safe_csv_append(temp_file, row)

# Atomic rename (POSIX guarantee)
os.rename(temp_file, final_file)
```

### Data Loss Prevention

**Buffer Flushing**:
```python
# pandas .to_csv() automatically flushes buffers
# No explicit flush needed

# For file handles, ensure flush:
with open(file_path, "a") as f:
    writer = csv.writer(f)
    writer.writerow(data)
    f.flush()  # Ensure written to disk
```

**Crash Recovery**:
```python
# If processing crashes:
# - CSV contains all rows written before crash
# - Last row may be partial (rare)
# - No automatic recovery (resume processing manually)

# Best practice: Checkpoint progress
checkpoint_file = "processing_checkpoint.json"
{
    "last_processed_index": 542,
    "total_files": 1000
}
# Resume from checkpoint after crash
```

---

## Browser Migration Constraints

### No Direct File Append

**Python** (file append):
```python
# Append to existing file on disk
thread_safe_csv_append("results.csv", data)
# File grows incrementally
```

**Browser** (no file append):
```javascript
// Cannot append to files on user's disk
// Must build in-memory and download at end

const csvData = [];
csvData.push(["col1", "col2", "col3"]);
csvData.push(["val1", "val2", "val3"]);

// Download as CSV
downloadCSV(csvData, "results.csv");
```

### Memory Constraints

**Python** (streaming writes):
```python
# Low memory: Write rows as processed
for file in large_batch:
    result = process_file(file)
    thread_safe_csv_append(csv_file, result)
# Memory usage: O(1) per file
```

**Browser** (in-memory accumulation):
```javascript
// Must store all results in memory
const results = [];
for (const file of files) {
    const result = await processFile(file);
    results.push(result);  // Accumulates in RAM
}
// Memory usage: O(n) where n = file count

// Constraint: Browser memory limit (~2GB typical)
// Large batches (1000+ files) may cause issues
```

**Mitigation Strategies**:

1. **Progressive Download**:
```javascript
// Download in chunks
const CHUNK_SIZE = 100;
for (let i = 0; i < files.length; i += CHUNK_SIZE) {
    const chunk = files.slice(i, i + CHUNK_SIZE);
    const results = await processFiles(chunk);
    downloadCSV(results, `results_${i}.csv`);
}
// Multiple CSV files instead of one large file
```

2. **IndexedDB Storage**:
```javascript
// Store results in IndexedDB (no memory limit)
async function saveResult(result) {
    await db.add('results', result);
}

// Export all at once
async function exportCSV() {
    const results = await db.getAll('results');
    downloadCSV(results);
    await db.clear('results');
}
```

3. **Streaming Download** (advanced):
```javascript
// Stream CSV directly to download (no memory accumulation)
const stream = new ReadableStream({
    async start(controller) {
        for (const file of files) {
            const result = await processFile(file);
            const csvRow = formatCSV(result);
            controller.enqueue(csvRow);
        }
        controller.close();
    }
});

// Trigger streaming download
const blob = await new Response(stream).blob();
downloadBlob(blob, "results.csv");
```

### CSV Format Constraints

**Python** (pandas handles escaping):
```python
# Automatic escaping
data = ['value with,comma']
# Output: "value with,comma"
```

**Browser** (manual escaping):
```javascript
// Must implement CSV escaping
function escapeCSV(value) {
    // Convert to string
    value = String(value);

    // Escape quotes
    value = value.replace(/"/g, '""');

    // Quote if contains special chars
    if (value.includes(',') || value.includes('"') || value.includes('\n')) {
        value = `"${value}"`;
    }

    return value;
}

// Use in CSV generation
const csvRow = data.map(escapeCSV).join(',');
```

**Recommendation**: Use a CSV library like `papaparse`:
```javascript
import Papa from 'papaparse';

const csv = Papa.unparse(data, {
    quotes: true,  // Quote all fields
    header: true   // Include header
});

downloadCSV(csv);
```

### Download Constraints

**File Size Limits**:
```javascript
// Browser download limits:
// - Chrome: ~2GB
// - Firefox: ~2GB
// - Safari: ~1GB

// Large CSV (10,000 files, 50 cols, 50 bytes/col):
size = 10000 × 50 × 50 = 25MB
// ✓ Well within limits

// Very large CSV (100,000 files):
size = 100000 × 50 × 50 = 250MB
// ✓ Still OK

// Extreme CSV (1,000,000 files):
size = 1000000 × 50 × 50 = 2.5GB
// ⚠️ May fail - use chunking
```

**Download Trigger**:
```javascript
// Must trigger download programmatically
function downloadCSV(csv, filename) {
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();

    // Cleanup
    URL.revokeObjectURL(url);
}
```

### No Lock Needed

**Python** (multi-threading):
```python
with CSV_WRITE_LOCK:  # Required
    thread_safe_csv_append(file, data)
```

**Browser** (single-threaded main, Web Workers communicate via messages):
```javascript
// Main thread
const results = [];
worker.onmessage = (e) => {
    results.push(e.data);  // No lock needed
};

// Web Workers: Message passing is serialized
// No shared memory write → no race conditions → no lock needed
```

**Exception**: SharedArrayBuffer (rare):
```javascript
// If using SharedArrayBuffer between workers:
const lock = new Atomics.Lock(sharedBuffer);
Atomics.wait(lock, 0, 0);  // Acquire lock
// ... write to shared buffer ...
Atomics.notify(lock, 0);   // Release lock

// Generally not needed for CSV generation
```

---

## Testing Constraints

### Test Data

**Valid Test Cases**:
```python
# Empty line
test_case_1 = (file_path, [])

# Single column
test_case_2 = (file_path, ["value"])

# Multiple columns
test_case_3 = (file_path, ["col1", "col2", "col3"])

# Special characters
test_case_4 = (file_path, ["comma,test", 'quote"test', "newline\ntest"])

# Mixed types
test_case_5 = (file_path, ["text", 123, 45.6, None, True])

# Large row
test_case_6 = (file_path, ["col" + str(i) for i in range(100)])
```

### Determinism

**Thread-Safe Writes Are Deterministic**:
```python
# Same input always produces same CSV output
for data in test_data:
    thread_safe_csv_append(file, data)

# Result: Deterministic CSV file (same every time)
```

**Parallel Processing Is Non-Deterministic**:
```python
# File completion order varies
# But CSV output is deterministic (if sorted before write)

results.sort(key=lambda r: r["file_counter"])
for result in results:
    thread_safe_csv_append(file, result["data"])

# Result: Deterministic CSV (same order every time)
```

### Concurrency Testing

**Test Concurrent Writes**:
```python
import threading

def write_rows(file_path, rows, thread_id):
    for i, row in enumerate(rows):
        thread_safe_csv_append(file_path, [thread_id, i, *row])

# Launch multiple threads
threads = []
for tid in range(4):
    t = threading.Thread(
        target=write_rows,
        args=(file_path, test_data, tid)
    )
    threads.append(t)
    t.start()

for t in threads:
    t.join()

# Verify: No corruption, all rows present
df = pd.read_csv(file_path, header=None)
assert len(df) == 4 * len(test_data)
```

---

## Summary of Critical Constraints

| Constraint | Value | Rationale |
|------------|-------|-----------|
| Lock scope | Module-level global | Serialize writes across threads |
| Lock type | threading.Lock (non-reentrant) | Simple, sufficient for single process |
| File mode | Append ("a") | Preserve existing rows |
| Quoting | QUOTE_NONNUMERIC | Safe, readable CSV |
| Atomicity | Per-row (lock held) | Prevent partial writes |
| Ordering | Maintained via sorting | Match input file order |
| Memory | ~1KB per write | Minimal overhead |
| Thread safety | Lock guarantees | In-process only |
| Multi-process | Not supported | Use file locking if needed |
| Browser | No file append | Build in-memory + download |

---

## Related Constraints

- **Parallel Processing**: `docs/v2/Parallel-Processing.md`
- **Threading Patterns**: `modules/technical/concurrency/concurrency-patterns.md`
- **File I/O**: `modules/integration/file-io/file-io.md`
- **Browser Downloads**: `modules/migration/browser-adaptations.md`
