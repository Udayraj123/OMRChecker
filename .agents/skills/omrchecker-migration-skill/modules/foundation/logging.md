# Logging Patterns

**Module**: Foundation
**Python Reference**: `src/utils/logger.py`
**Last Updated**: 2026-02-20

---

## Overview

OMRChecker uses a custom Logger class built on Python's `logging` module with Rich terminal integration for colored, formatted output. The logging system supports dynamic log level filtering and thread-safe operation.

**Key Features**:
1. **Rich terminal UI**: Colored, formatted console output using Rich library
2. **Dynamic log levels**: Enable/disable log levels per message type at runtime
3. **Thread-safe**: Lock-protected logging for parallel processing
4. **Auto-stringify**: Automatic conversion of non-string arguments
5. **Stack-aware**: Correct caller information in log messages
6. **Configurable**: Log levels controlled via config.json

---

## Logger Architecture

### Logger Class

**Code Reference**: `src/utils/logger.py:41-88`

**Attributes**:
- `log` (logging.Logger): Underlying Python logger instance
- `show_logs_by_type` (dict[str, bool]): Map of log level to enabled/disabled
- `_lock` (threading.Lock): Thread-safety lock

**Methods**:
- `set_log_levels(show_logs_by_type)`: Update log level filtering
- `reset_log_levels()`: Reset to defaults
- `debug(*msg, sep=" ")`: Log debug message
- `info(*msg, sep=" ")`: Log info message
- `warning(*msg, sep=" ")`: Log warning message
- `error(*msg, sep=" ")`: Log error message
- `critical(*msg, sep=" ")`: Log critical message

---

## Log Levels

### Default Log Level Map

**Code Reference**: `src/utils/logger.py:17-23`

```python
DEFAULT_LOG_LEVEL_MAP = {
    "critical": True,
    "error": True,
    "warning": True,
    "info": True,
    "debug": False,  # Disabled by default
}
```

**Behavior**:
- `True`: Messages at this level are logged
- `False`: Messages at this level are suppressed

**Default Config** (`src/schemas/defaults/config.py:37-43`):
```python
show_logs_by_type={
    "critical": True,
    "error": True,
    "warning": True,
    "info": True,
    "debug": False,
}
```

---

## Log Level Usage Patterns

### 1. Debug Level

**When to Use**: Detailed diagnostic information for debugging

**Examples**:
```python
logger.debug(f"Executing processor: {processor_name}")
logger.debug(f"SIFT keypoints detected: {len(keypoints)}")
logger.debug(f"Bubble value: {bubble_value}, threshold: {threshold}")
```

**Typical Content**:
- Processor execution flow
- Algorithm intermediate values
- Feature detection counts
- Threshold calculations
- Variable state dumps

**Enabled**: Only when `show_logs_by_type["debug"] = True` in config

---

### 2. Info Level

**When to Use**: General informational messages about normal operation

**Examples** (from `src/entry.py`):
```python
logger.info("")  # Blank line for formatting
logger.info(f"({file_counter}) Opening image: \t'{file_path}'\tResolution: {gray_image.shape}")
logger.info(f"Read Response: \n{concatenated_omr_response}")
logger.info(f"(/{file_counter}) Graded with score: {round(score, 2)}")
logger.info("File organization enabled with dynamic patterns")
logger.info(f"\nProcessing {len(omr_files)} files with {max_workers} worker thread(s)...")
```

**Typical Content**:
- File processing progress
- Detected responses
- Scores and evaluation results
- Configuration summaries
- Processing statistics

**Enabled**: By default

---

### 3. Warning Level

**When to Use**: Non-critical issues that don't prevent execution

**Examples** (from `src/entry.py`):
```python
logger.warning(f"Found an evaluation file without a parent template file: {local_evaluation_path}")
logger.warning(f"Marker detection failed for '{file_path}'")
logger.warning(f"[{file_counter}] Error in {file_path.name}: {error}")
logger.warning("Continuing without alignment")
```

**Typical Content**:
- Missing optional files
- Failed non-critical operations
- Fallback to defaults
- Validation warnings
- Processing anomalies

**Enabled**: By default

---

### 4. Error Level

**When to Use**: Errors that prevent specific operations but allow continuation

**Examples** (from `src/entry.py`):
```python
logger.error(f"Found images, but no template in the directory tree of '{curr_dir}'")
logger.error("File grouping configuration has errors. Please fix the following issues:")
for error in validation_errors:
    logger.error(f"  - {error}")
logger.error(f"Error processing {file_path}: {e}")
```

**Typical Content**:
- Missing required files
- Validation failures
- Configuration errors
- Processing failures
- Exception messages

**Enabled**: By default

---

### 5. Critical Level

**When to Use**: Critical errors that may cause application termination

**Examples**:
```python
logger.critical("Failed to initialize OpenCV")
logger.critical(f"Unrecoverable error in template: {template_path}")
```

**Typical Content**:
- System-level failures
- Unrecoverable errors
- Fatal configuration issues

**Enabled**: Always (cannot be disabled)

---

## Rich Terminal Integration

### RichHandler Configuration

**Code Reference**: `src/utils/logger.py:10-15`

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
```

**Features**:
- **Colored output**: Automatic color coding by log level
  - DEBUG: Cyan
  - INFO: Default/white
  - WARNING: Yellow
  - ERROR: Red
  - CRITICAL: Bold red
- **Rich tracebacks**: Enhanced exception formatting with syntax highlighting
- **Timestamp formatting**: `[%X]` shows time in HH:MM:SS format
- **Markup support**: Rich markup in log messages

---

## Console Object

**Code Reference**: `src/utils/logger.py:91`

```python
console = Console()
```

**Purpose**: Direct Rich Console access for tables, panels, and formatted output

**Usage Examples**:

**Tables** (`src/entry.py:47-71`):
```python
from src.utils.logger import console
from rich.table import Table

table = Table(title="Current Configurations", show_header=False, show_lines=False)
table.add_column("Key", style="cyan", no_wrap=True)
table.add_column("Value", style="magenta")
table.add_row("Directory Path", f"{curr_dir}")
table.add_row("Count of Images", f"{len(omr_files)}")
table.add_row("Debug Mode", "ON" if args["debug"] else "OFF")
console.print(table, justify="center")
```

**Direct printing**:
```python
console.print("[bold green]Success![/bold green]")
console.print("[red]Error occurred[/red]")
```

---

## Thread-Safe Logging

### Lock Mechanism

**Code Reference**: `src/utils/logger.py:52,86-87`

```python
class Logger:
    def __init__(self, ...):
        self._lock = Lock()

    def logutil(self, method_type, *msg, sep=" "):
        # ... filtering logic ...
        with self._lock:
            return logger_func(sep.join(msg), stacklevel=4)
```

**Purpose**: Prevent race conditions during parallel processing

**Why Needed**:
- Multiple threads call logger simultaneously
- Terminal output could interleave without locking
- Ensures clean, sequential log messages

**Performance Impact**: Minimal (logging is not bottleneck)

---

## Auto-Stringify Decorator

### Stringify Pattern

**Code Reference**: `src/utils/logger.py:27-38`

```python
def stringify(func: Callable) -> Callable:
    def inner(self, method_type: str, *msg: object, sep=" ") -> None:
        nmsg = []
        for v in msg:
            if not isinstance(v, str):
                nmsg.append(str(v))
            else:
                nmsg.append(v)
        return func(self, method_type, *nmsg, sep=sep)
    return inner
```

**Purpose**: Automatically convert non-string arguments to strings

**Allows**:
```python
logger.info("Score:", 95.5, "Total:", 100)
# Automatically converts to: "Score: 95.5 Total: 100"

logger.info("Processing image:", Path("/path/to/image.jpg"))
# Converts Path to string automatically
```

**Without decorator**:
```python
logger.info("Score: " + str(95.5) + " Total: " + str(100))  # Manual conversion
```

---

## Dynamic Log Level Control

### Runtime Configuration

**Set Levels** (`src/entry.py:89`):
```python
# Load from config.json
logger.set_log_levels(tuning_config.outputs.show_logs_by_type)
```

**Reset Levels** (`src/entry.py:635`):
```python
# Reset to defaults after processing
logger.reset_log_levels()
```

### Configuration via config.json

**Example config.json**:
```json
{
  "outputs": {
    "show_logs_by_type": {
      "critical": true,
      "error": true,
      "warning": true,
      "info": true,
      "debug": true
    }
  }
}
```

**Enable debug logging**:
```json
{
  "outputs": {
    "show_logs_by_type": {
      "debug": true
    }
  }
}
```

**Suppress info messages** (keep only warnings/errors):
```json
{
  "outputs": {
    "show_logs_by_type": {
      "info": false,
      "debug": false
    }
  }
}
```

---

## Stack Level Management

### Correct Caller Attribution

**Code Reference**: `src/utils/logger.py:76-87`

```python
# set stack level to 3 so that the caller of this function is logged, not this function itself.
# stack-frame - self.log.debug - logutil - stringify - log method - caller
@stringify
def logutil(self, method_type: str, *msg: object, sep=" ") -> None:
    # ...
    with self._lock:
        return logger_func(sep.join(msg), stacklevel=4)
```

**Stack Trace**:
1. Caller (actual code location) ← **Want this**
2. `logger.info()` method
3. `logutil()` method
4. `stringify()` decorator
5. `self.log.info()` Python logging method

**stacklevel=4**: Skips 3 intermediate frames to attribute log to actual caller

**Why Important**: Log messages show correct file/line number where logger was called

---

## Common Logging Patterns

### 1. Progress Logging

**Pattern**: Log file processing progress

```python
logger.info("")
logger.info(f"({file_counter}) Opening image: \t'{file_path}'\tResolution: {gray_image.shape}")
# ... processing ...
logger.info(f"(/{file_counter}) Processed file: '{file_id}'")
```

**Format**: `(N)` for start, `(/N)` for completion

---

### 2. Error with Context

**Pattern**: Log error with relevant context

```python
try:
    template = Template(template_path, config, args)
except TemplateValidationError as e:
    logger.error(f"Template validation failed: {e}")
    for error in e.errors:
        logger.error(f"  - {error}")
```

---

### 3. Conditional Logging

**Pattern**: Log only when condition met

```python
if tuning_config.outputs.show_image_level <= 1:
    logger.info(f"\nTip: To see awesome visuals, increase `outputs.showImageLevel`")
```

---

### 4. Debug Diagnostics

**Pattern**: Detailed debugging info (only when enabled)

```python
logger.debug(f"Template loaded: {template.path}")
logger.debug(f"Field blocks: {list(template.field_blocks.keys())}")
logger.debug(f"Pre-processors: {template.get_pre_processor_names()}")
```

---

### 5. Statistics Reporting

**Pattern**: Log processing statistics

```python
log = logger.info
log("")
log(f"{'Total file(s) moved': <27}: {STATS.files_moved}")
log(f"{'Total file(s) not moved': <27}: {STATS.files_not_moved}")
log("--------------------------------")
log(f"{'Total file(s) processed': <27}: {files_counter}")
log(f"{'OMR Processing Rate': <27}: \t ~ {round(time_checking / files_counter, 2)} seconds/OMR")
log(f"{'OMR Processing Speed': <27}: \t ~ {round((files_counter * 60) / time_checking, 2)} OMRs/minute")
```

**Code Reference**: `src/entry.py:652-675`

---

## Browser Migration Notes

### JavaScript Console Equivalents

**Python Logger** → **Browser Console**

```javascript
// Console API mapping
const logger = {
    debug: (...msg) => console.debug(...msg),
    info: (...msg) => console.info(...msg),
    warning: (...msg) => console.warn(...msg),
    error: (...msg) => console.error(...msg),
    critical: (...msg) => console.error('[CRITICAL]', ...msg),
};
```

### Dynamic Log Level Control

```javascript
class Logger {
    constructor() {
        this.showLogsByType = {
            critical: true,
            error: true,
            warning: true,
            info: true,
            debug: false,
        };
    }

    setLogLevels(config) {
        this.showLogsByType = { ...this.showLogsByType, ...config };
    }

    resetLogLevels() {
        this.setLogLevels({
            critical: true,
            error: true,
            warning: true,
            info: true,
            debug: false,
        });
    }

    info(...msg) {
        if (this.showLogsByType.info) {
            console.info(...msg);
        }
    }

    debug(...msg) {
        if (this.showLogsByType.debug) {
            console.debug(...msg);
        }
    }

    // ... other levels
}

const logger = new Logger();
export { logger };
```

### UI Logging (Browser Alternative)

Instead of terminal output, log to UI elements:

```javascript
class UILogger extends Logger {
    constructor(logContainerId) {
        super();
        this.container = document.getElementById(logContainerId);
    }

    info(...msg) {
        if (this.showLogsByType.info) {
            this.appendToUI('info', msg.join(' '));
            console.info(...msg);
        }
    }

    appendToUI(level, message) {
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry log-${level}`;
        logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        this.container.appendChild(logEntry);
        this.container.scrollTop = this.container.scrollHeight;
    }
}
```

### Progress Logging in Browser

Use progress bars instead of console logs:

```javascript
function updateProgress(current, total, message) {
    const percentage = (current / total) * 100;
    document.getElementById('progress-bar').style.width = `${percentage}%`;
    document.getElementById('progress-text').textContent =
        `(${current}/${total}) ${message}`;
    logger.info(`Processing ${current}/${total}: ${message}`);
}
```

### Rich Tables in Browser

Use HTML tables instead of Rich tables:

```javascript
function displayConfigSummary(config) {
    const table = document.createElement('table');
    table.className = 'config-table';

    const rows = [
        ['Directory Path', config.directory],
        ['Count of Images', config.imageCount],
        ['Debug Mode', config.debug ? 'ON' : 'OFF'],
        ['Processing Speed', `${config.speed} OMRs/minute`],
    ];

    rows.forEach(([key, value]) => {
        const row = table.insertRow();
        row.innerHTML = `<td>${key}</td><td>${value}</td>`;
    });

    document.getElementById('config-summary').appendChild(table);
}
```

---

## Testing Logging

### Unit Tests

**Code Reference**: `src/tests/utils/__tests__/test_logger.py`

**Test Coverage**:
- Logger initialization
- Log level filtering
- Dynamic log level changes
- Thread safety
- Message formatting

**Example Test**:
```python
def test_set_log_levels():
    log = Logger(__name__)
    log.set_log_levels({"debug": False, "info": True})
    assert log.show_logs_by_type["debug"] is False
    assert log.show_logs_by_type["info"] is True
```

---

## Summary

**Architecture**: Custom Logger class wrapping Python logging with Rich integration
**Log Levels**: 5 (debug, info, warning, error, critical)
**Thread Safety**: Lock-protected for parallel processing
**Dynamic Control**: Runtime log level filtering via config
**Auto-Stringify**: Automatic argument conversion to strings
**Stack-Aware**: Correct caller attribution (stacklevel=4)

**Browser Migration**:
- Use console API for basic logging
- Implement dynamic log level filtering
- Replace terminal output with UI logging
- Use HTML tables instead of Rich tables
- Add progress bars for visual feedback

**Key Takeaway**: OMRChecker's logging is well-structured with dynamic control and rich formatting. Browser version should maintain log level filtering but use UI elements instead of terminal output.
