# Virtual Environment Setup

This guide documents a reproducible local setup for OMRChecker using `uv`.

## Setup

Sync project dependencies:

```bash
uv sync
```

Install contributor tooling too:

```bash
uv sync --group dev
```

`uv` automatically manages the project environment, so no manual activation step is required.

## Run Commands

Run the application:

```bash
uv run main.py
```

Run quality checks:

```bash
uv run pre-commit run -a
uv run pytest -rfpsxEX --disable-warnings --verbose
```