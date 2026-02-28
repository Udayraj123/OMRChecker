#!/usr/bin/env python3
"""
Batch Migration Orchestrator

Orchestrates migration of multiple Python files to TypeScript.
Reads module lists from files or phase definitions, runs migration,
collects errors, tracks progress, and updates FILE_MAPPING.json.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class MigrationTask:
    """Single migration task."""

    python_file: str
    typescript_file: str
    phase: int | None = None
    priority: str | None = None
    status: str = "pending"
    error: str | None = None
    validation_score: float | None = None


@dataclass
class BatchReport:
    """Batch migration report."""

    total_tasks: int
    completed: int
    failed: int
    skipped: int
    average_score: float
    tasks: list[MigrationTask] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""

    def print_summary(self) -> None:
        """Print batch report summary."""
        print(f"\n{'=' * 70}")
        print("Batch Migration Report")
        print(f"{'=' * 70}")
        print(f"Total tasks:    {self.total_tasks}")
        print(f"✅ Completed:   {self.completed}")
        print(f"❌ Failed:      {self.failed}")
        print(f"⏭️  Skipped:     {self.skipped}")
        print(f"📊 Avg Score:   {self.average_score:.1f}%")
        print(f"⏱️  Duration:    {self._duration()}")
        print(f"{'=' * 70}\n")

        if self.failed > 0:
            print("❌ Failed Migrations:")
            for task in self.tasks:
                if task.status == "failed":
                    print(f"  - {task.python_file}")
                    if task.error:
                        print(f"    Error: {task.error}")
            print()

    def _duration(self) -> str:
        """Calculate duration."""
        if not self.start_time or not self.end_time:
            return "N/A"
        start = datetime.fromisoformat(self.start_time)
        end = datetime.fromisoformat(self.end_time)
        duration = end - start
        minutes = int(duration.total_seconds() / 60)
        seconds = int(duration.total_seconds() % 60)
        return f"{minutes}m {seconds}s"


class BatchMigrator:
    """Orchestrate batch Python to TypeScript migration."""

    def __init__(
        self,
        repo_root: Path,
        file_mapping_path: Path,
        patterns_path: Path,
        skip_validation: bool = False,
        dry_run: bool = False,
    ):
        self.repo_root = repo_root
        self.file_mapping_path = file_mapping_path
        self.patterns_path = patterns_path
        self.skip_validation = skip_validation
        self.dry_run = dry_run
        self.file_mapping = self._load_file_mapping()
        self.exclusions = self._load_exclusions()

    def _load_file_mapping(self) -> dict:
        """Load FILE_MAPPING.json."""
        if not self.file_mapping_path.exists():
            return {"mappings": []}
        with open(self.file_mapping_path) as f:
            return json.load(f)

    def _save_file_mapping(self) -> None:
        """Save FILE_MAPPING.json."""
        with open(self.file_mapping_path, "w") as f:
            json.dump(self.file_mapping, f, indent=2)

    def _load_exclusions(self) -> set[str]:
        """Load migration exclusions."""
        exclusion_file = self.repo_root / ".ts-migration-exclude"
        if not exclusion_file.exists():
            return set()

        exclusions = set()
        with open(exclusion_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    exclusions.add(line)
        return exclusions

    def _is_excluded(self, python_file: str) -> bool:
        """Check if file is excluded from migration."""
        for exclusion in self.exclusions:
            if exclusion.endswith("/"):
                # Directory exclusion
                if python_file.startswith(exclusion):
                    return True
            else:
                # File exclusion
                if python_file == exclusion or python_file.endswith(exclusion):
                    return True
        return False

    def load_tasks_from_file(self, task_file: Path) -> list[MigrationTask]:
        """Load migration tasks from file.
        
        File format (one per line):
        python_file|typescript_file|phase|priority
        """
        tasks = []
        with open(task_file) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split("|")
                if len(parts) < 2:
                    print(f"⚠️  Line {line_num}: Invalid format, skipping")
                    continue

                python_file = parts[0].strip()
                typescript_file = parts[1].strip()
                phase = int(parts[2].strip()) if len(parts) > 2 else None
                priority = parts[3].strip() if len(parts) > 3 else None

                # Check exclusions
                if self._is_excluded(python_file):
                    print(f"⏭️  Skipping excluded file: {python_file}")
                    continue

                tasks.append(
                    MigrationTask(
                        python_file=python_file,
                        typescript_file=typescript_file,
                        phase=phase,
                        priority=priority,
                    )
                )

        return tasks

    def load_tasks_from_mapping(
        self, phase: int | None = None, status_filter: str | None = None
    ) -> list[MigrationTask]:
        """Load migration tasks from FILE_MAPPING.json."""
        tasks = []

        for mapping in self.file_mapping.get("mappings", []):
            python_file = mapping.get("python")
            typescript_file = mapping.get("typescript")

            if not python_file or not typescript_file:
                continue

            # Apply filters
            if phase and mapping.get("phase") != phase:
                continue

            if status_filter and mapping.get("status") != status_filter:
                continue

            # Check exclusions
            if self._is_excluded(python_file):
                continue

            tasks.append(
                MigrationTask(
                    python_file=python_file,
                    typescript_file=typescript_file,
                    phase=mapping.get("phase"),
                    priority=mapping.get("priority"),
                )
            )

        return tasks

    def migrate_single(self, task: MigrationTask) -> bool:
        """Migrate a single file."""
        python_path = self.repo_root / task.python_file
        typescript_path = self.repo_root / task.typescript_file

        if not python_path.exists():
            task.status = "failed"
            task.error = f"Python file not found: {python_path}"
            return False

        print(f"📝 Migrating: {task.python_file} → {task.typescript_file}")

        if self.dry_run:
            print("   [DRY RUN] Would migrate file")
            task.status = "completed"
            return True

        # Run migration script
        try:
            # Ensure output directory exists
            typescript_path.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                "uv",
                "run",
                "scripts/generate_ts_suggestions.py",
                "--file",
                str(python_path),
                "--output",
                str(typescript_path),
                "--repo-root",
                str(self.repo_root),
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, cwd=self.repo_root
            )

            print(f"   ✅ Generated TypeScript")

            # Run validation if not skipped
            if not self.skip_validation:
                validation_result = self._validate_migration(task)
                if validation_result:
                    print(f"   ✅ Validation score: {task.validation_score:.1f}%")
                else:
                    print(f"   ⚠️  Validation failed: {task.validation_score:.1f}%")

            task.status = "completed"
            return True

        except subprocess.CalledProcessError as e:
            task.status = "failed"
            task.error = e.stderr if e.stderr else str(e)
            print(f"   ❌ Migration failed: {task.error}")
            return False
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            print(f"   ❌ Unexpected error: {task.error}")
            return False

    def _validate_migration(self, task: MigrationTask) -> bool:
        """Validate migrated file."""
        try:
            cmd = [
                "uv",
                "run",
                "scripts/validate_ts_migration.py",
                "--python-file",
                str(self.repo_root / task.python_file),
                "--typescript-file",
                str(self.repo_root / task.typescript_file),
                "--json",
                "--min-score",
                "70",
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False, cwd=self.repo_root
            )

            if result.returncode == 0:
                output = json.loads(result.stdout)
                task.validation_score = output.get("score", 0)
                return True
            else:
                # Parse output even on failure
                try:
                    output = json.loads(result.stdout)
                    task.validation_score = output.get("score", 0)
                except:
                    task.validation_score = 0
                return False

        except Exception as e:
            print(f"   ⚠️  Validation error: {e}")
            task.validation_score = 0
            return False

    def update_mapping_status(self, task: MigrationTask) -> None:
        """Update FILE_MAPPING.json with migration status."""
        now = datetime.now(UTC).isoformat()

        for mapping in self.file_mapping.get("mappings", []):
            if mapping.get("python") == task.python_file:
                mapping["status"] = "synced" if task.status == "completed" else "failed"
                mapping["lastTypescriptChange"] = now
                if task.validation_score:
                    mapping["validationScore"] = task.validation_score
                break

    def migrate_batch(self, tasks: list[MigrationTask]) -> BatchReport:
        """Migrate batch of files."""
        report = BatchReport(
            total_tasks=len(tasks),
            completed=0,
            failed=0,
            skipped=0,
            average_score=0.0,
            tasks=tasks,
            start_time=datetime.now(UTC).isoformat(),
        )

        print(f"\n🚀 Starting batch migration of {len(tasks)} files\n")

        for i, task in enumerate(tasks, 1):
            print(f"[{i}/{len(tasks)}] ", end="")

            if self._is_excluded(task.python_file):
                print(f"⏭️  Skipping: {task.python_file} (excluded)")
                task.status = "skipped"
                report.skipped += 1
                continue

            success = self.migrate_single(task)

            if success:
                report.completed += 1
                if not self.dry_run:
                    self.update_mapping_status(task)
            else:
                report.failed += 1

            print()  # Blank line between tasks

        report.end_time = datetime.now(UTC).isoformat()

        # Calculate average score
        scores = [t.validation_score for t in tasks if t.validation_score is not None]
        report.average_score = sum(scores) / len(scores) if scores else 0

        # Save updated mapping
        if not self.dry_run and report.completed > 0:
            self._save_file_mapping()
            print("💾 Updated FILE_MAPPING.json\n")

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Batch Python to TypeScript migration orchestrator"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root directory",
    )
    parser.add_argument(
        "--task-file",
        type=Path,
        help="File with migration tasks (python|typescript|phase|priority per line)",
    )
    parser.add_argument(
        "--from-mapping",
        action="store_true",
        help="Load tasks from FILE_MAPPING.json",
    )
    parser.add_argument(
        "--phase",
        type=int,
        help="Filter by phase number",
    )
    parser.add_argument(
        "--status",
        choices=["not_started", "partial", "synced", "deviating"],
        help="Filter by status in FILE_MAPPING.json",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation step (faster)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without doing it",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save report to JSON file",
    )

    args = parser.parse_args()

    # Initialize migrator
    migrator = BatchMigrator(
        repo_root=args.repo_root,
        file_mapping_path=args.repo_root / "FILE_MAPPING.json",
        patterns_path=args.repo_root / "CHANGE_PATTERNS.yaml",
        skip_validation=args.skip_validation,
        dry_run=args.dry_run,
    )

    # Load tasks
    tasks = []
    if args.task_file:
        if not args.task_file.exists():
            print(f"❌ Task file not found: {args.task_file}", file=sys.stderr)
            sys.exit(1)
        tasks = migrator.load_tasks_from_file(args.task_file)
    elif args.from_mapping:
        tasks = migrator.load_tasks_from_mapping(
            phase=args.phase, status_filter=args.status
        )
    else:
        print("❌ Must specify --task-file or --from-mapping", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    if not tasks:
        print("⚠️  No tasks to migrate")
        sys.exit(0)

    # Run migration
    report = migrator.migrate_batch(tasks)

    # Print summary
    report.print_summary()

    # Save report if requested
    if args.output:
        output_data = {
            "total_tasks": report.total_tasks,
            "completed": report.completed,
            "failed": report.failed,
            "skipped": report.skipped,
            "average_score": report.average_score,
            "start_time": report.start_time,
            "end_time": report.end_time,
            "tasks": [
                {
                    "python_file": t.python_file,
                    "typescript_file": t.typescript_file,
                    "status": t.status,
                    "error": t.error,
                    "validation_score": t.validation_score,
                }
                for t in report.tasks
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"💾 Report saved to {args.output}")

    # Exit with error if any failed
    sys.exit(1 if report.failed > 0 else 0)


if __name__ == "__main__":
    main()
