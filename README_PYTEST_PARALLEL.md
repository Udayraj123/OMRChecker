# Pytest Parallel Execution

## Status

✅ **Parallel execution is now configured** using `pytest-xdist`.

## Configuration

### Installation
`pytest-xdist` has been added to `pyproject.toml` in the `dev` dependency group.

### Configuration
The `pytest.ini` file has been updated with `-n auto` in `addopts`, which automatically detects the number of CPU cores and runs tests in parallel.

## Usage

### Automatic (Recommended)
```bash
# Uses all available CPU cores automatically
uv run pytest

# Or explicitly
uv run pytest -n auto
```

### Manual Worker Count
```bash
# Use 4 workers
uv run pytest -n 4

# Use 8 workers
uv run pytest -n 8
```

### Disable Parallel Execution
```bash
# Run sequentially (useful for debugging)
uv run pytest -n 0

# Or comment out `-n auto` in pytest.ini
```

## Performance

With 8 CPU cores, you can expect:
- **2-4x speedup** for CPU-bound tests
- **Variable speedup** for I/O-bound tests (file operations, network)
- **Best results** with independent tests (no shared state)

## Considerations

### Test Isolation
- Tests should be independent (no shared state between tests)
- Use fixtures properly to avoid race conditions
- File I/O tests may need special handling

### Coverage Reports
- Coverage collection works with parallel execution
- Use `pytest-cov` (already configured)

### Debugging
- If tests fail only in parallel mode, run with `-n 0` to debug
- Use `--dist=worksteal` for better load balancing
- Use `-v` for verbose output to see which worker ran which test

## Advanced Options

```bash
# Use work-stealing scheduler (better for uneven test durations)
uv run pytest -n auto --dist=worksteal

# Limit to specific number of workers
uv run pytest -n 4

# Run tests in groups (loadscope)
uv run pytest -n auto --dist=loadscope
```

## Troubleshooting

### Tests fail only in parallel
- Check for shared state between tests
- Ensure fixtures are properly scoped
- Use `-n 0` to verify sequential execution works

### Coverage issues
- Coverage should work automatically with `pytest-cov`
- If issues occur, try `--cov-append` flag

### Performance not improving
- Some tests may be I/O bound (file operations)
- Tests with shared resources may serialize anyway
- Check CPU usage during test runs

