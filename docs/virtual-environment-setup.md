# Virtual Environment Setup

This guide documents a reproducible local setup for OMRChecker.

## Recommended: uv workflow

```bash
uv sync
```

Install contributor tooling too:

```bash
uv sync --group dev
```

Run commands inside the managed environment:

```bash
uv run main.py --help
uv run pre-commit run -a
uv run pytest -rfpsxEX --disable-warnings --verbose
```

## Fallback: pip + venv workflow

Create an isolated environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
# Linux/macOS
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements.dev.txt
```

Run the application:

```bash
python main.py
```
