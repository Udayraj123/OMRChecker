# FILE_MAPPING.json Verification and Update - 2026-01-17

## Summary

Verified and updated FILE_MAPPING.json entries for files that were recently synced between Python and TypeScript.

## Files Updated in FILE_MAPPING.json

### 1. `bubble_field.py` → `bubbleField.ts`
- **Status**: `synced` ✅
- **Updates**:
  - `lastPythonChange`: `null` → `2026-01-17T00:00:00Z`
  - `lastTypescriptChange`: `2026-01-16T00:00:00Z` → `2026-01-17T00:00:00Z`
  - **Notes**: Updated to reflect recent sync work:
    - `BubbleField.setup_scan_boxes` temporarily sets `bubble_dimensions` and `bubble_field_type` before creating `BubblesScanBox` instances
    - Handles initialization order issue (setup_scan_boxes called from base Field constructor before BubbleField sets these properties)
    - BubblesScanBox constructor accesses field.bubble_dimensions and field.bubble_field_type

### 2. `detection_pass.py` → `detectionPass.ts`
- **Status**: `synced` ✅
- **Updates**:
  - `lastPythonChange`: `null` → `2026-01-17T00:00:00Z`
  - `lastTypescriptChange`: Already updated to `2026-01-17T00:00:00Z`
  - **Notes**: Already updated in previous sync (includes files_by_label_count, error handling, etc.)

### 3. `interpretation_pass.py` → `interpretationPass.ts`
- **Status**: `synced` ✅
- **Updates**:
  - `lastPythonChange`: `null` → `2026-01-17T00:00:00Z`
  - `lastTypescriptChange`: Already updated to `2026-01-17T00:00:00Z`
  - **Notes**: Already updated in previous sync (includes files_by_label_count, read_response_flags, etc.)

## Why `detect_python_changes.py` Didn't Capture These Changes

### Root Cause

The `detect_python_changes.py` script only detects changes in **staged files** (files added to git index with `git add`).

**Line 154-155 in `detect_python_changes.py`:**
```python
result = subprocess.run(
    ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
    ...
)
```

The `--cached` flag means it only looks at staged changes, not unstaged modifications.

### Current Status

When we ran the verification:
- `bubble_field.py`: Modified but **not staged** (`M` in `git status`)
- `interpretation_pass.py`: Modified but **not staged** (`M` in `git status`)
- `detection_pass.py`: Not modified (already committed or not changed)

### Additional Limitations

The script also has these limitations:

1. **Only detects method signature changes**: The AST comparison only checks:
   - Method arguments (`args`)
   - Decorators (`decorators`)
   - Method additions/deletions
   - **Does NOT detect changes to method bodies**

2. **No body content analysis**: Changes to method implementations (like adding temporary property assignments in `setup_scan_boxes`) are not detected because the method signature didn't change.

3. **Requires staging**: Files must be staged with `git add` before the script can detect them.

## Recommendations

### 1. Improve `detect_python_changes.py` to detect unstaged changes

Add an option to check unstaged files:

```python
def get_changed_python_files(self, base_ref: str = "HEAD", include_unstaged: bool = False) -> list[str]:
    """Get list of Python files changed in git."""
    files = []

    # Get staged files
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        files.extend(result.stdout.strip().split("\n"))
    except subprocess.CalledProcessError:
        pass

    # Optionally get unstaged files
    if include_unstaged:
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=ACM"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
            unstaged = result.stdout.strip().split("\n")
            files.extend(unstaged)
        except subprocess.CalledProcessError:
            pass

    # Deduplicate and filter
    files = list(set(files))
    return [f for f in files if f.endswith(".py") and f.startswith("src/")]
```

### 2. Add method body change detection

Use AST diffing or line-based comparison to detect changes in method bodies:

```python
def detect_method_body_changes(self, old_method, new_method):
    """Detect if method body has changed."""
    # Compare line ranges or use AST diffing
    if old_method["lineno"] != new_method["lineno"]:
        return True
    if old_method["end_lineno"] != new_method["end_lineno"]:
        return True
    # Could also compare AST body nodes
    return False
```

### 3. Add manual update option

Add a flag to manually mark files as changed:

```python
parser.add_argument(
    "--manual-files",
    nargs="+",
    help="Manually specify Python files to check (even if not in git diff)",
)
```

## Verification Checklist

- [x] `bubble_field.py` entry updated with correct timestamps and notes
- [x] `detection_pass.py` entry updated with correct timestamps
- [x] `interpretation_pass.py` entry updated with correct timestamps
- [x] All entries marked as `synced`
- [x] JSON file is valid
- [x] Notes accurately reflect recent sync work

## Next Steps

1. **Stage the Python files** if you want the script to detect them:
   ```bash
   git add src/processors/layout/field/bubble_field.py
   git add src/processors/detection/base/interpretation_pass.py
   ```

2. **Consider improving the script** to detect unstaged changes and method body modifications

3. **Run the script with staged files** to verify it works:
   ```bash
   python3 scripts/detect_python_changes.py --update-timestamps
   ```

