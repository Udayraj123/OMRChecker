#!/usr/bin/env python3
"""
Sync Tool CLI - Interactive command-line tool for managing Python-TypeScript synchronization.

This tool provides commands for:
- Checking sync status
- Detecting changes
- Generating TypeScript suggestions
- Marking files as synced
- Generating reports
- Watching for changes
"""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from rich import box
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def load_file_mapping(repo_root: Path) -> dict:
    """Load FILE_MAPPING.json."""
    mapping_file = repo_root / "FILE_MAPPING.json"
    if not mapping_file.exists():
        console.print("[red]Error: FILE_MAPPING.json not found[/red]")
        sys.exit(1)

    with open(mapping_file) as f:
        return json.load(f)


def save_file_mapping(repo_root: Path, mapping: dict):
    """Save FILE_MAPPING.json."""
    mapping_file = repo_root / "FILE_MAPPING.json"
    with open(mapping_file, "w") as f:
        json.dump(mapping, f, indent=2)


def cmd_status(args):
    """Show current sync status."""
    repo_root = Path(args.repo_root)
    mapping = load_file_mapping(repo_root)

    mappings = mapping.get("mappings", [])

    # Calculate statistics
    stats = {
        "synced": 0,
        "partial": 0,
        "not_started": 0,
        "total": len(mappings),
    }

    for m in mappings:
        status = m.get("status", "not_started")
        if status == "synced":
            stats["synced"] += 1
        elif status == "partial":
            stats["partial"] += 1
        else:
            stats["not_started"] += 1

    # Display summary
    console.print("\n[bold cyan]Sync Status Report[/bold cyan]")
    console.print("=" * 50)

    total = stats["total"]
    synced_pct = (stats["synced"] / total * 100) if total > 0 else 0

    console.print(f"Total mapped files: {total}")
    console.print(f"✅ In sync: {stats['synced']} ({synced_pct:.1f}%)")
    console.print(
        f"⚠️  Partially synced: {stats['partial']} ({stats['partial'] / total * 100:.1f}%)"
    )
    console.print(
        f"❌ Not started: {stats['not_started']} ({stats['not_started'] / total * 100:.1f}%)"
    )

    # Show details if verbose
    if args.verbose:
        console.print("\n[bold]Details:[/bold]\n")

        table = Table(box=box.ROUNDED)
        table.add_column("Python File", style="yellow")
        table.add_column("TypeScript File", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Phase", justify="center")

        for m in mappings:
            status = m.get("status", "not_started")
            if status == "synced":
                status_display = "[green]✅ Synced[/green]"
            elif status == "partial":
                status_display = "[yellow]⚠️ Partial[/yellow]"
            else:
                status_display = "[red]❌ Not Started[/red]"

            py_file = m["python"].replace("src/", "")
            ts_file = (
                m.get("typescript", "").replace("omrchecker-js/packages/core/src/", "")
                if m.get("typescript")
                else "N/A"
            )
            phase = str(m.get("phase", "?"))

            table.add_row(py_file, ts_file, status_display, phase)

        console.print(table)


def cmd_detect(args):
    """Detect changes requiring updates."""
    repo_root = Path(args.repo_root)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing changes...", total=None)

        # Run change detector
        result = subprocess.run(
            [
                sys.executable,
                "scripts/detect_python_changes.py",
                "--repo-root",
                str(repo_root),
                "--output",
                "/tmp/changes.json",
            ],
            check=False,
            cwd=repo_root,
            capture_output=True,
            text=True,
        )

        progress.remove_task(task)

    if result.returncode != 0:
        console.print(f"[red]Error detecting changes:[/red]\n{result.stderr}")
        sys.exit(1)

    # Load and display results
    with open("/tmp/changes.json") as f:
        changes = json.load(f)

    total = changes.get("total_files_changed", 0)

    if total == 0:
        console.print("[green]✅ No Python changes detected[/green]")
        return

    console.print(
        f"\n[yellow]Found {total} file(s) with changes requiring TypeScript updates[/yellow]\n"
    )

    for change in changes.get("changes", []):
        py_file = change["pythonFile"]
        ts_file = change.get("typescriptFile", "Not mapped")
        status = change.get("status", "unknown")

        if status in ("not_started", "partial"):
            console.print(f"  {py_file} → {ts_file}")

    # Suggest running auto-sync
    if args.interactive and total > 0:
        console.print()
        console.print(
            "[bold cyan]Run auto-sync to apply structural changes:[/bold cyan]"
        )
        console.print("  [green]uv run python scripts/sync_tool.py auto-sync[/green]")
        console.print(
            "\n[dim]Or commit your changes to trigger the pre-commit hook[/dim]"
        )


def cmd_suggest(args):
    """Generate TypeScript suggestions for a file."""
    repo_root = Path(args.repo_root)
    py_file = args.file

    console.print(f"\n[cyan]Analyzing {py_file}...[/cyan]\n")

    # Run suggestion generator
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_ts_suggestions.py",
            "--file",
            py_file,
            "--output",
            f".sync-suggestions/{Path(py_file).stem}.ts.patch",
        ],
        check=False,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        console.print(f"[red]Error generating suggestions:[/red]\n{result.stderr}")
        sys.exit(1)

    console.print(result.stdout)


def cmd_mark_synced(args):
    """Mark a file as synced."""
    repo_root = Path(args.repo_root)
    py_file = args.file

    mapping = load_file_mapping(repo_root)

    found = False
    for m in mapping.get("mappings", []):
        if m["python"] == py_file:
            m["status"] = "synced"
            m["lastSyncedCommit"] = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=False,
                cwd=repo_root,
                capture_output=True,
                text=True,
            ).stdout.strip()
            m["lastTypescriptChange"] = datetime.now(UTC).isoformat()
            found = True
            break

    if not found:
        console.print(f"[red]Error: {py_file} not found in FILE_MAPPING.json[/red]")
        sys.exit(1)

    save_file_mapping(repo_root, mapping)
    console.print(f"[green]✅ Marked {py_file} as synced[/green]")


def cmd_report(args):
    """Generate HTML sync report."""
    repo_root = Path(args.repo_root)
    mapping = load_file_mapping(repo_root)

    # Generate HTML report
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_file = repo_root / f"reports/sync-status-{timestamp}.html"
    output_file.parent.mkdir(exist_ok=True)

    mappings = mapping.get("mappings", [])
    stats = {
        "synced": sum(1 for m in mappings if m.get("status") == "synced"),
        "partial": sum(1 for m in mappings if m.get("status") == "partial"),
        "not_started": sum(1 for m in mappings if m.get("status") == "not_started"),
        "total": len(mappings),
    }

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>OMRChecker Sync Status Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 8px; flex: 1; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat-value {{ font-size: 48px; font-weight: bold; }}
        .synced {{ color: #27ae60; }}
        .partial {{ color: #f39c12; }}
        .not-started {{ color: #e74c3c; }}
        table {{ width: 100%; background: white; border-collapse: collapse; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #34495e; color: white; }}
        .status-badge {{ padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
        .badge-synced {{ background: #d4edda; color: #155724; }}
        .badge-partial {{ background: #fff3cd; color: #856404; }}
        .badge-not-started {{ background: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>OMRChecker Sync Status Report</h1>
        <p>Generated: {datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-value synced">{stats["synced"]}</div>
            <div>Synced Files</div>
        </div>
        <div class="stat-card">
            <div class="stat-value partial">{stats["partial"]}</div>
            <div>Partially Synced</div>
        </div>
        <div class="stat-card">
            <div class="stat-value not-started">{stats["not_started"]}</div>
            <div>Not Started</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats["total"]}</div>
            <div>Total Files</div>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th>Python File</th>
                <th>TypeScript File</th>
                <th>Status</th>
                <th>Phase</th>
                <th>Priority</th>
            </tr>
        </thead>
        <tbody>
"""

    for m in mappings:
        status = m.get("status", "not_started")
        badge_class = f"badge-{status.replace('_', '-')}"

        html_content += f"""
            <tr>
                <td>{m["python"]}</td>
                <td>{m.get("typescript", "N/A")}</td>
                <td><span class="status-badge {badge_class}">{status}</span></td>
                <td>{m.get("phase", "?")}</td>
                <td>{m.get("priority", "medium")}</td>
            </tr>
"""

    html_content += """
        </tbody>
    </table>
</body>
</html>
"""

    with open(output_file, "w") as f:
        f.write(html_content)

    console.print(f"[green]✅ Report generated: {output_file}[/green]")

    # Open in browser if requested
    if args.open:
        import webbrowser

        webbrowser.open(f"file://{output_file.absolute()}")


def cmd_watch(_args):
    """Watch for changes and notify."""
    console.print("[yellow]Watch mode not yet implemented[/yellow]")
    console.print("Use 'git diff' hooks or IDE extensions for now")


def cmd_auto_sync(args):
    """Run auto-sync manually without committing."""
    repo_root = Path(args.repo_root)

    console.print("\n🔄 [bold cyan]Running auto-sync...[/bold cyan]\n")

    # Run the auto-sync hook script
    result = subprocess.run(
        [sys.executable, "scripts/hooks/auto_sync_python_to_ts.py"],
        cwd=repo_root,
        check=False,
    )

    if result.returncode != 0:
        console.print("[red]❌ Auto-sync failed[/red]")
        sys.exit(1)

    console.print("[green]✅ Auto-sync complete[/green]")


def main():
    parser = argparse.ArgumentParser(
        description="OMRChecker Sync Tool - Manage Python-TypeScript synchronization"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root directory",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # status command
    status_parser = subparsers.add_parser("status", help="Show current sync status")
    status_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed status"
    )

    # detect command
    detect_parser = subparsers.add_parser(
        "detect", help="Detect changes requiring updates"
    )
    detect_parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        default=True,
        help="Interactive mode",
    )

    # suggest command
    suggest_parser = subparsers.add_parser(
        "suggest", help="Generate TypeScript suggestions for file"
    )
    suggest_parser.add_argument("file", help="Python file to analyze")

    # mark-synced command
    mark_parser = subparsers.add_parser("mark-synced", help="Mark file as synced")
    mark_parser.add_argument("file", help="Python file to mark as synced")

    # report command
    report_parser = subparsers.add_parser("report", help="Generate HTML sync report")
    report_parser.add_argument(
        "--open", "-o", action="store_true", help="Open report in browser"
    )

    # watch command
    subparsers.add_parser("watch", help="Watch for changes and notify")

    # auto-sync command
    subparsers.add_parser("auto-sync", help="Run auto-sync manually")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate command
    commands = {
        "status": cmd_status,
        "detect": cmd_detect,
        "suggest": cmd_suggest,
        "mark-synced": cmd_mark_synced,
        "report": cmd_report,
        "watch": cmd_watch,
        "auto-sync": cmd_auto_sync,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
