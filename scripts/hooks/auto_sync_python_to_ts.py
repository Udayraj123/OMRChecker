#!/usr/bin/env python3
"""
Pre-commit hook for automatically syncing Python changes to TypeScript.

This script:
1. Detects staged Python file changes
2. Analyzes changes using AST
3. Applies structural synchronization to corresponding TypeScript files
4. Stages the updated TypeScript files
5. Provides rich terminal output with actionable guidance
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def load_file_mapping(repo_root: Path) -> dict:
    """Load FILE_MAPPING.json."""
    mapping_file = repo_root / "FILE_MAPPING.json"
    if not mapping_file.exists():
        console.print("[yellow]⚠️  FILE_MAPPING.json not found[/yellow]")
        return {"mappings": []}

    with open(mapping_file) as f:
        return json.load(f)


def get_changed_python_files(repo_root: Path) -> list[str]:
    """Get list of Python files changed in git (staged)."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        files = result.stdout.strip().split("\n")
        # Filter for Python files in src/ directory
        return [f for f in files if f and f.endswith(".py") and f.startswith("src/")]
    except subprocess.CalledProcessError:
        return []


def run_change_detection(repo_root: Path, output_file: Path) -> bool:
    """Run change detection script."""
    try:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/detect_python_changes.py",
                "--repo-root",
                str(repo_root),
                "--output",
                str(output_file),
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error detecting changes:[/red]\n{e.stderr}")
        return False


def run_sync_script(repo_root: Path, changes_json: Path, stage: bool = True) -> bool:
    """Run TypeScript sync script."""
    try:
        cmd = [
            sys.executable,
            "scripts/sync_ts_from_python.py",
            "--repo-root",
            str(repo_root),
            "--changes-json",
            str(changes_json),
        ]
        if stage:
            cmd.append("--stage")

        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        console.print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error syncing TypeScript files:[/red]\n{e.stderr}")
        return False


def main() -> int:
    """Main entry point."""
    repo_root = Path.cwd()

    # Check if we're in the repository root
    if not (repo_root / ".git").exists():
        console.print("[red]Error: Not in a git repository[/red]", file=sys.stderr)
        return 1

    # Get changed Python files
    changed_py_files = get_changed_python_files(repo_root)

    if not changed_py_files:
        # No Python files changed, nothing to do
        return 0

    # Load file mapping
    file_mapping = load_file_mapping(repo_root)

    # Filter for mapped files (excluding future phase)
    mapped_files = []
    for py_file in changed_py_files:
        for mapping in file_mapping.get("mappings", []):
            if mapping["python"] == py_file:
                phase = mapping.get("phase", "unknown")
                # Skip future phase files
                if phase == "future":
                    continue
                ts_file = mapping.get("typescript")
                if (
                    ts_file
                    and ts_file != "Not mapped"
                    and not ts_file.startswith("N/A")
                ):
                    mapped_files.append((py_file, ts_file))
                break

    if not mapped_files:
        # No mapped files to sync
        return 0

    console.print("\n🔄 [bold cyan]Auto-syncing Python → TypeScript...[/bold cyan]\n")

    # Show what will be processed
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Python File", style="yellow", no_wrap=False)
    table.add_column("TypeScript File", style="cyan", no_wrap=False)

    for py_file, ts_file in mapped_files:
        short_py = py_file.replace("src/", "")
        short_ts = ts_file.replace("omrchecker-js/packages/core/src/", "")
        table.add_row(short_py, short_ts)

    console.print(table)
    console.print()

    # Create temporary file for changes
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        # Run change detection
        if not run_change_detection(repo_root, tmp_path):
            console.print(
                "[yellow]⚠️  Change detection failed, skipping auto-sync[/yellow]"
            )
            return 0

        # Run sync script (with staging)
        if not run_sync_script(repo_root, tmp_path, stage=True):
            console.print("[yellow]⚠️  Sync failed, skipping auto-sync[/yellow]")
            return 0

        # Success message
        console.print(
            Panel(
                """[bold green]✅ TypeScript files auto-updated and staged![/bold green]

[yellow]Next steps:[/yellow]
  1. Review the TypeScript changes in your editor
  2. Manually fix any implementation details, types, or logic
  3. Run tests: [cyan]cd omrchecker-js && pnpm test[/cyan]
  4. Stage any additional fixes: [cyan]git add <files>[/cyan]
  5. Retry commit

[dim]Note: Only structural changes (classes/methods) were auto-synced.
You must manually implement the logic and update types.[/dim]
""",
                title="[bold]Auto-sync Complete[/bold]",
                border_style="green",
                expand=False,
            )
        )

    finally:
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
