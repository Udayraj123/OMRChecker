# CSV Utils Flows

**Module**: Domain - Utils - CSV
**Python Reference**: `src/utils/csv.py`
**Last Updated**: 2026-02-21

---

## Overview

CSV utilities provide thread-safe CSV writing operations for parallel OMR processing. The module ensures result integrity when multiple threads write to the same CSV file, maintaining proper ordering of results.

**Key Features**:
- Thread-safe CSV append operations
- Automatic quoting of non-numeric values
- Integration with pandas for reliable CSV handling
- Support for both file paths and file handles

---

## Flow 1: Thread-Safe CSV Append

**File**: `src/utils/csv.py`
**Function**: `thread_safe_csv_append()`
**Purpose**: Append a single row to a CSV file in a thread-safe manner

### Input

```python
file_path: Path | IO[str]     # CSV file path or file handle
data_line: list               # Row data to append [col1, col2, ...]
quoting: int                  # CSV quoting mode (default: QUOTE_NONNUMERIC)
```

### Processing Steps

```
START: thread_safe_csv_append(file_path, data_line, quoting)
│
├─► Step 1: Acquire Lock
│   with CSV_WRITE_LOCK:
│       │ Global threading.Lock() instance
│       │ Ensures only one thread writes at a time
│       │ Prevents interleaved writes and corruption
│
├─► Step 2: Convert Data to DataFrame
│   df = pd.DataFrame(data_line, dtype=str)
│   │
│   │ Example data_line: ["file1.jpg", "A,B,C", "85"]
│   │
│   │ DataFrame created:
│   │     0
│   │ 0   file1.jpg
│   │ 1   A,B,C
│   │ 2   85
│   │
│   │ dtype=str ensures all values treated as strings
│   │ (pandas will apply quoting rules during CSV write)
│
├─► Step 3: Transpose DataFrame
│   df_transposed = df.T
│   │
│   │ Result:
│   │          0        1    2
│   │ 0  file1.jpg  A,B,C   85
│   │
│   │ Converts column vector to row vector
│
├─► Step 4: Write to CSV
│   df_transposed.to_csv(
│       file_path,
│       mode="a",              # Append mode
│       quoting=quoting,       # QUOTE_NONNUMERIC by default
│       header=False,          # No column headers
│       index=False            # No row index
│   )
│   │
│   │ Written to file: "file1.jpg","A,B,C","85"\n
│   │                   ↑________↑ ↑____↑ ↑__↑
│   │                   Quoted     Quoted Quoted (all non-numeric)
│
└─► Step 5: Release Lock
    Lock released automatically (context manager)

END
```

### Output

**File Contents** (after append):
```csv
"file1.jpg","A,B,C","85"
```

**Characteristics**:
- One new line added to CSV file
- Values quoted according to quoting mode
- Atomic operation (lock ensures completion)
- Thread-safe (no race conditions)

---

## Flow 2: Parallel CSV Writing Pattern

**File**: `src/entry.py`
**Purpose**: Write CSV results from parallel processing in correct order

### Context

When processing OMR files in parallel using ThreadPoolExecutor:
- Files complete in non-deterministic order (based on processing speed)
- CSV output must maintain input file order for traceability
- Solution: Collect results, sort by input order, then write sequentially

### Processing Steps

```
START: process_directory_files() with max_workers > 1
│
├─► Step 1: Prepare File Tasks
│   file_tasks = [
│       (file1.jpg, counter=0, ...),
│       (file2.jpg, counter=1, ...),
│       (file3.jpg, counter=2, ...),
│       ...
│   ]
│   │
│   │ Each task includes:
│   │ - file_path: Path to OMR image
│   │ - file_counter: Original input order (0, 1, 2, ...)
│   │ - other context (template, config, etc.)
│
├─► Step 2: Submit Tasks to Thread Pool
│   with ThreadPoolExecutor(max_workers=4) as executor:
│       futures = {
│           executor.submit(process_single_file, task): task
│           for task in file_tasks
│       }
│   │
│   │ All files submitted for parallel processing
│   │ Threads process files concurrently
│
├─► Step 3: Collect Results (Unsorted)
│   results = []
│   for future in as_completed(futures):
│       result = future.result()
│       │
│       │ Result structure:
│       │ {
│       │     "file_counter": 2,              # Original position
│       │     "status": "success",
│       │     "csv_writes": [
│       │         {
│       │             "file": results.csv,
│       │             "data": ["file3.jpg", "B,C", "90"]
│       │         }
│       │     ]
│       │ }
│       │
│       results.append(result)
│   │
│   │ Results arrive in completion order, NOT input order:
│   │ results = [
│   │     {file_counter: 2, ...},  # file3 finished first
│   │     {file_counter: 0, ...},  # file1 finished second
│   │     {file_counter: 1, ...},  # file2 finished third
│   │ ]
│
├─► Step 4: Sort by Input Order
│   results.sort(key=lambda r: r["file_counter"])
│   │
│   │ Sorted results:
│   │ [
│   │     {file_counter: 0, data: ["file1.jpg", ...]},
│   │     {file_counter: 1, data: ["file2.jpg", ...]},
│   │     {file_counter: 2, data: ["file3.jpg", ...]},
│   │ ]
│   │
│   │ Now matches original input order
│
├─► Step 5: Write CSV in Order
│   for result in results:
│       for csv_write in result["csv_writes"]:
│           thread_safe_csv_append(
│               csv_write["file"],    # results.csv or errors.csv
│               csv_write["data"]     # Row data
│           )
│   │
│   │ CSV written sequentially in correct order:
│   │ Row 1: file1.jpg data
│   │ Row 2: file2.jpg data
│   │ Row 3: file3.jpg data
│   │
│   │ Order preserved despite parallel processing
│
└─► Step 6: Finalize
    CSV files complete with properly ordered results

END
```

### Result

**results.csv**:
```csv
"file1.jpg","/path/to/file1.jpg","output1.jpg","95","A","B","C"
"file2.jpg","/path/to/file2.jpg","output2.jpg","80","A","C","D"
"file3.jpg","/path/to/file3.jpg","output3.jpg","90","B","C","A"
```

**Order matches input file list** (deterministic output).

---

## Flow 3: Sequential CSV Writing Pattern

**Purpose**: Write CSV results immediately in sequential processing mode

### Processing Steps

```
START: process_directory_files() with max_workers = 1
│
├─► Step 1: Process Files Sequentially
│   for task in file_tasks:
│       result = process_single_file(task)
│       │
│       │ Files processed one at a time
│       │ No parallel execution
│
├─► Step 2: Write CSV Immediately
│       for csv_write in result["csv_writes"]:
│           thread_safe_csv_append(
│               csv_write["file"],
│               csv_write["data"]
│           )
│   │
│   │ CSV written immediately after each file
│   │ No sorting needed (already in order)
│   │ Lock still used for consistency
│
└─► Step 3: Continue to Next File
    Loop continues with next file

END
```

**Benefits**:
- Simpler flow (no result collection)
- Immediate output (good for monitoring)
- Lower memory usage (one file at a time)
- Automatically ordered (sequential processing)

---

## Flow 4: Result Data Formatting

**File**: `src/entry.py`, `src/processors/template/template.py`
**Purpose**: Format OMR processing results into CSV row data

### Processing Steps

```
START: Format results for CSV writing
│
├─► Step 1: Process Single File
│   result = process_single_file(file_info)
│   │
│   │ Extracts OMR responses, evaluates answers
│   │ Generates scores and metadata
│
├─► Step 2: Build Results Line (Success Case)
│   If processing successful:
│       │
│       ├─ Get basic file info
│       │  file_name = "student001.jpg"
│       │  posix_file_path = "/input/batch1/student001.jpg"
│       │  output_file_path = "/output/batch1/student001.jpg"
│       │  score = "85"
│       │
│       ├─ Format OMR response array
│       │  omr_response_array = template.append_output_omr_response(
│       │      file_name,
│       │      output_omr_response
│       │  )
│       │  │
│       │  │ Example output_omr_response:
│       │  │ {
│       │  │     "Roll": "12345",
│       │  │     "Q1": "A",
│       │  │     "Q2": "B,C",
│       │  │     "Q3": "D"
│       │  │ }
│       │  │
│       │  │ omr_response_array = ["12345", "A", "B,C", "D"]
│       │  │ (Ordered by omr_response_columns from template)
│       │
│       └─ Combine into CSV row
│          results_line = [
│              file_name,           # "student001.jpg"
│              posix_file_path,     # "/input/batch1/student001.jpg"
│              output_file_path,    # "/output/batch1/student001.jpg"
│              score,               # "85"
│              *omr_response_array  # ["12345", "A", "B,C", "D"]
│          ]
│          │
│          │ Final results_line:
│          │ [
│          │     "student001.jpg",
│          │     "/input/batch1/student001.jpg",
│          │     "/output/batch1/student001.jpg",
│          │     "85",
│          │     "12345",
│          │     "A",
│          │     "B,C",
│          │     "D"
│          │ ]
│
├─► Step 3: Build Error Line (Failure Case)
│   If marker detection failed:
│       error_file_line = [
│           file_name,
│           posix_file_path,
│           output_file_path,
│           "NA",                        # No score
│           *template.get_empty_response_array()  # Empty responses
│       ]
│       │
│       │ Example:
│       │ ["student002.jpg", "/path/...", "NA", "", "", "", ""]
│
├─► Step 4: Store CSV Write Instructions
│   result["csv_writes"].append({
│       "file": template.get_results_file(),  # Path to results.csv
│       "data": results_line
│   })
│   │
│   │ Or for errors:
│   result["csv_writes"].append({
│       "file": template.get_errors_file(),  # Path to errors.csv
│       "data": error_file_line
│   })
│
└─► Step 5: Return Result
    return result  # Contains csv_writes list

END
```

### Output Structure

**Result Dictionary**:
```python
{
    "file_counter": 42,           # Position in input list
    "status": "success",          # or "error"
    "error": None,                # or error code like "NO_MARKER_ERR"
    "csv_writes": [
        {
            "file": Path("outputs/Results/Results.csv"),
            "data": ["file.jpg", "/path", "output.jpg", "85", "A", "B", "C"]
        }
    ]
}
```

---

## Flow 5: CSV File Initialization

**File**: `src/processors/template/directory_handler.py`
**Purpose**: Create CSV files with headers before processing

### Processing Steps

```
START: Initialize output CSV files
│
├─► Step 1: Determine Output Columns
│   # Fixed columns
│   fixed_columns = ["file_name", "input_path", "output_path", "score"]
│   │
│   # Dynamic columns from template
│   omr_response_columns = [
│       "Roll",  # Field labels from template
│       "Q1",
│       "Q2",
│       "Q3",
│       ...
│   ]
│   │
│   # Combine
│   csv_columns = fixed_columns + omr_response_columns
│
├─► Step 2: Create Results CSV
│   results_csv = output_dir / "Results" / "Results.csv"
│   │
│   # Write header row
│   pd.DataFrame(columns=csv_columns).to_csv(
│       results_csv,
│       index=False,
│       quoting=QUOTE_NONNUMERIC
│   )
│   │
│   │ Results.csv created:
│   │ "file_name","input_path","output_path","score","Roll","Q1","Q2","Q3"
│
├─► Step 3: Create Errors CSV
│   errors_csv = output_dir / "Errors" / "Errors.csv"
│   │
│   # Same columns as Results.csv
│   pd.DataFrame(columns=csv_columns).to_csv(
│       errors_csv,
│       index=False,
│       quoting=QUOTE_NONNUMERIC
│   )
│
└─► Step 4: Store File Paths
    output_files = {
        "Results": results_csv,
        "Errors": errors_csv,
        # ... other output files
    }

END
```

### Output

**Results.csv** (header only):
```csv
"file_name","input_path","output_path","score","Roll","Q1","Q2","Q3"
```

**Ready for data rows via `thread_safe_csv_append()`**.

---

## Decision Points

### Decision 1: Use Lock vs Lock-Free Queue

```
Need to write CSV data from multiple threads?
│
├─► Option A: Use Lock (Current Implementation)
│   with CSV_WRITE_LOCK:
│       df.to_csv(file_path, mode="a", ...)
│   │
│   │ ✓ Simple implementation
│   │ ✓ Familiar threading primitive
│   │ ✓ Works with pandas .to_csv()
│   │ ✓ Minimal overhead for our use case
│   │ ✗ Theoretical contention (not observed)
│
└─► Option B: Lock-Free Queue
    queue.put(csv_data)
    # Separate writer thread consumes queue
    │
    │ ✓ No lock contention
    │ ✗ More complex implementation
    │ ✗ Additional thread overhead
    │ ✗ Overkill for 4-8 workers
    │
    │ DECISION: Use lock (simpler, sufficient)
```

### Decision 2: Collect Results vs Write Immediately

```
Running in parallel mode?
│
├─► Yes (max_workers > 1)
│   │ Must preserve input order
│   │ Files complete in random order
│   │
│   └─► Collect all results → Sort → Write
│       results.sort(key=lambda r: r["file_counter"])
│       for result in results:
│           write_csv(result)
│
└─► No (max_workers = 1)
    │ Sequential processing
    │ Already in correct order
    │
    └─► Write immediately after each file
        for task in tasks:
            result = process(task)
            write_csv(result)  # Immediate write
```

### Decision 3: Use pandas vs csv module

```
Need to write CSV data?
│
├─► Option A: pandas (Current Implementation)
│   pd.DataFrame(data).T.to_csv(...)
│   │
│   │ ✓ Consistent with rest of codebase
│   │ ✓ Handles quoting automatically
│   │ ✓ Type inference (if needed)
│   │ ✓ Well-tested library
│   │ ✗ Slightly higher overhead
│
└─► Option B: csv.writer
    writer = csv.writer(file, quoting=QUOTE_NONNUMERIC)
    writer.writerow(data)
    │
    │ ✓ Lightweight
    │ ✓ Standard library
    │ ✗ Less consistent with codebase
    │ ✗ Manual type handling
    │
    │ DECISION: Use pandas (consistency wins)
```

---

## Error Handling

### Empty Data Line

```python
# Edge case: empty data line
thread_safe_csv_append(file_path, [])

# Result: Empty line in CSV (valid but unusual)
# No error raised
```

### File Not Found

```python
# CSV file doesn't exist yet
thread_safe_csv_append("nonexistent.csv", ["data"])

# Result: File created automatically by pandas
# First write creates new file
```

### Lock Timeout

**Not applicable**: `threading.Lock()` has no timeout by default.

```python
# If lock held indefinitely → thread blocks forever
# Mitigation: Ensure all operations complete quickly
# CSV write is fast (~1ms), timeout not needed
```

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Lock acquire | O(1) | Fast mutex operation |
| DataFrame creation | O(n) | n = row length (~10 cols) |
| CSV write | O(n) | Single row write |
| Total per row | O(n) | Dominated by I/O |

**Typical Values**:
- n (columns): 10-50
- Write time: 0.5-2ms per row
- Lock contention: Negligible with 4-8 workers

### Memory Usage

```python
# Per write operation
data_line = ["col1", "col2", ...]     # ~100 bytes
df = pd.DataFrame(data_line)          # ~500 bytes
df_transposed = df.T                  # ~500 bytes (copy)
# Total: ~1KB per write (minimal)
```

### Lock Contention

**Scenario**: 4 workers, 100 files, 2ms write time

```
Total CSV write time: 100 files × 2ms = 200ms
Processing time per file: ~500ms (average)
Contention: 200ms / (4 workers × 500ms) = 10%

Conclusion: Low contention, minimal impact
```

---

## Browser Migration Notes

### Challenge: No Threading in Browser

**Python**:
```python
with CSV_WRITE_LOCK:  # Thread synchronization
    df.to_csv(file_path, mode="a")
```

**Browser**: No threads, different approach needed.

### Solution 1: Sequential Processing Only

```javascript
// Browser processes files sequentially
for (const file of files) {
  const result = await processFile(file);
  csvData.push(result);  // Build array
}

// Download as CSV at end
downloadCSV(csvData);
```

**No lock needed**: Single-threaded execution.

### Solution 2: Web Workers + Message Passing

```javascript
// Main thread
const workers = [worker1, worker2, worker3, worker4];
const results = [];

workers.forEach((worker, i) => {
  worker.postMessage({ file: files[i] });

  worker.onmessage = (e) => {
    results.push(e.data);  // Collect results

    if (results.length === files.length) {
      // All done - sort and download
      results.sort((a, b) => a.fileCounter - b.fileCounter);
      downloadCSV(results);
    }
  };
});
```

**No lock needed**: Message passing is inherently serialized.

### Solution 3: IndexedDB for Incremental Storage

```javascript
// Worker writes result to IndexedDB
async function saveResult(result) {
  const db = await openDB('omr-results');
  await db.add('results', result);
}

// Main thread reads all results
async function exportCSV() {
  const db = await openDB('omr-results');
  const results = await db.getAll('results');
  results.sort((a, b) => a.fileCounter - b.fileCounter);
  downloadCSV(results);
}
```

**No lock needed**: IndexedDB handles concurrency internally.

### Recommended Approach

```javascript
// Simple: Build in-memory, download at end
const csvBuilder = {
  data: [],

  addRow(row) {
    this.data.push(row);
  },

  exportCSV() {
    // Sort by file counter
    this.data.sort((a, b) => a.fileCounter - b.fileCounter);

    // Convert to CSV string
    const csv = this.data.map(row =>
      row.map(col => `"${col}"`).join(',')
    ).join('\n');

    // Trigger download
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'results.csv';
    a.click();
  }
};
```

**Key Differences from Python**:
- No file append (build in memory instead)
- Single download at end (not incremental writes)
- No lock needed (single-threaded or message passing)
- Browser File API for download

---

## Related Flows

- **Parallel Processing**: `docs/v2/Parallel-Processing.md`
- **Template Directory Handler**: `modules/domain/template/directory-handler/` (pending)
- **File Organization**: `modules/domain/organization/` (pending)

---

## Summary

CSV utils provide:

1. **Thread-Safe Writing**: Global lock ensures atomic CSV append
2. **Result Ordering**: Parallel results sorted before writing
3. **pandas Integration**: Reliable CSV handling with automatic quoting
4. **Dual Mode Support**: Sequential (immediate write) and parallel (collect → sort → write)
5. **Simple API**: Single function for all CSV append needs

**Critical Pattern**: In parallel mode, always collect results, sort by `file_counter`, then write to preserve input order.

**Browser Migration**: Replace file append with in-memory array building + single download at end.
