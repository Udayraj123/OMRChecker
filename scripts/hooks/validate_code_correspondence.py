#!/usr/bin/env python3
"""
Pre-commit hook for validating Python-TypeScript code correspondence.

This script:
1. Detects Python file changes
2. Checks if corresponding TypeScript files exist and are updated
3. Provides rich terminal output with actionable guidance
4. Blocks commit if out of sync (unless --no-verify is used)
"""

import json
import subprocess
import sys
from pathlib import Path

from rich import box

# Rich library is already a dependency
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


def get_ts_file_status(repo_root: Path, ts_file: str | None) -> str:
    """Check if TypeScript file has changes staged."""
    if not ts_file:
        return "not_mapped"

    ts_path = repo_root / ts_file
    if not ts_path.exists():
        return "not_created"

    try:
        # Check if TS file is also staged
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        staged_files = result.stdout.strip().split("\n")
        if ts_file in staged_files:
            return "synced"
        return "out_of_sync"
    except subprocess.CalledProcessError:
        return "unknown"


def check_correspondence(repo_root: Path) -> tuple[list[dict], bool]:
    """Check Python-TypeScript correspondence."""
    changed_py_files = get_changed_python_files(repo_root)

    if not changed_py_files:
        return [], True

    file_mapping = load_file_mapping(repo_root)
    results = []
    all_synced = True

    for py_file in changed_py_files:
        # Find mapping
        mapping = None
        for m in file_mapping.get("mappings", []):
            if m["python"] == py_file:
                mapping = m
                break

        if not mapping:
            # Python file not in mapping (might be CLI, test, or other non-mapped file)
            continue

        ts_file = mapping.get("typescript")
        status = get_ts_file_status(repo_root, ts_file)
        priority = mapping.get("priority", "medium")
        phase = mapping.get("phase", "unknown")

        # Skip validation for "future" phase files - they're not part of current port
        if phase == "future":
            continue

        result = {
            "python": py_file,
            "typescript": ts_file,
            "status": status,
            "priority": priority,
            "phase": phase,
            "mapping_status": mapping.get("status", "not_started"),
        }
        results.append(result)

        if status != "synced":
            all_synced = False

    return results, all_synced


def display_results(results: list[dict], all_synced: bool):
    """Display rich terminal output."""
    if not results:
        console.print("✅ [green]No mapped Python files changed[/green]")
        return

    console.print("\n🔍 [bold]Analyzing Python changes...[/bold]")

    # Create table
    table = Table(
        title="Python ↔ TypeScript Sync Status",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Python File", style="yellow", no_wrap=False, width=35)
    table.add_column("Status", justify="center", width=12)
    table.add_column("Priority", justify="center", width=8)
    table.add_column("Phase", justify="center", width=6)

    out_of_sync_files = []

    for result in results:
        py_file = result["python"]
        status = result["status"]
        priority = result["priority"]
        phase = result["phase"]

        # Status emoji and color
        if status == "synced":
            status_display = "[green]✅ SYNCED[/green]"
        elif status == "not_created":
            status_display = "[red]❌ NOT CREATED[/red]"
            out_of_sync_files.append(result)
        elif status == "out_of_sync":
            status_display = "[red]❌ OUT OF SYNC[/red]"
            out_of_sync_files.append(result)
        elif status == "not_mapped":
            status_display = "[gray]⚠️  NOT MAPPED[/gray]"
        else:
            status_display = "[gray]❓ UNKNOWN[/gray]"

        # Priority color
        if priority == "high":
            priority_display = "[red]HIGH[/red]"
        elif priority == "medium":
            priority_display = "[yellow]MED[/yellow]"
        else:
            priority_display = "[gray]LOW[/gray]"

        # Shorten path for display
        short_py = py_file.replace("src/", "")

        table.add_row(short_py, status_display, priority_display, str(phase))

    console.print(table)

    if not all_synced:
        console.print("\n[red]⚠️  SYNC CHECK FAILED[/red]\n")

        # Show detailed info for each out-of-sync file
        for result in out_of_sync_files:
            py_file = result["python"]
            ts_file = result["typescript"]
            status = result["status"]

            panel_content = f"""[yellow]Python:[/yellow]     {py_file}
[cyan]TypeScript:[/cyan] {ts_file or "Not mapped"}

[bold]Status:[/bold] {"TypeScript file needs to be created and updated" if status == "not_created" else "TypeScript file exists but not staged in this commit"}

[bold yellow]Action required:[/bold yellow]
  1. Update the TypeScript file: [cyan]{ts_file}[/cyan]
  2. Stage the TypeScript changes: [green]git add {ts_file}[/green]
  3. Retry commit

[bold]Or use change propagation tool:[/bold]
  [green]pnpm run change-tool[/green]

[dim]Bypass (not recommended):[/dim] [dim]git commit --no-verify[/dim]
"""

            console.print(
                Panel(
                    panel_content,
                    title=f"[red]❌ {py_file.split('/')[-1]}[/red]",
                    border_style="red",
                    expand=False,
                )
            )

        console.print()

    else:
        console.print(
            "\n✅ [green bold]All Python changes have corresponding TypeScript updates![/green bold]\n"
        )


def main() -> int:
    """Main entry point."""
    repo_root = Path.cwd()

    # Check if we're in the repository root
    if not (repo_root / ".git").exists():
        console.print("[red]Error: Not in a git repository[/red]", file=sys.stderr)
        return 1

    results, all_synced = check_correspondence(repo_root)
    display_results(results, all_synced)

    if not all_synced:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
