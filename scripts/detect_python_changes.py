#!/usr/bin/env python3
"""
Detect Python code changes and map them to TypeScript files.

This script:
1. Parses git diffs to find changed Python files
2. Uses Python AST to understand semantic changes (not just line diffs)
3. Maps changes to corresponding TypeScript files using FILE_MAPPING.json
4. Generates structured change report in JSON format
"""

import argparse
import ast
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class MethodChange:
    """Represents a change to a method."""

    type: str  # "added", "modified", "deleted"
    name: str
    line_range: tuple[int, int] | None = None
    change_details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassChange:
    """Represents a change to a class."""

    type: str  # "added", "modified", "deleted"
    name: str
    line_range: tuple[int, int] | None = None
    methods: list[MethodChange] = field(default_factory=list)
    change_details: dict[str, Any] = field(default_factory=dict)


@dataclass
class FileChange:
    """Represents changes to a Python file."""

    python_file: str
    typescript_file: str | None
    status: str  # "synced", "out_of_sync", "not_mapped"
    phase: int | str | None
    priority: str | None
    classes: list[ClassChange] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)


class PythonASTAnalyzer(ast.NodeVisitor):
    """Analyze Python AST to extract semantic information."""

    def __init__(self) -> None:
        self.classes: dict[str, dict] = {}
        self.functions: list[str] = []
        self.imports: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definition."""
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(
                    {
                        "name": item.name,
                        "lineno": item.lineno,
                        "end_lineno": item.end_lineno,
                        "args": [arg.arg for arg in item.args.args],
                        "decorators": [
                            self._get_decorator_name(d) for d in item.decorator_list
                        ],
                    }
                )

        self.classes[node.name] = {
            "name": node.name,
            "lineno": node.lineno,
            "end_lineno": node.end_lineno,
            "methods": methods,
            "bases": [self._get_name(base) for base in node.bases],
        }
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definition (module-level)."""
        # Only track module-level functions
        if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(node)):
            self.functions.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        """Visit import statement."""
        for alias in node.names:
            self.imports.append(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Visit from...import statement."""
        if node.module:
            for alias in node.names:
                self.imports.append(f"{node.module}.{alias.name}")

    @staticmethod
    def _get_decorator_name(decorator):
        """Extract decorator name."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        if isinstance(decorator, ast.Call):
            return PythonASTAnalyzer._get_name(decorator.func)
        return str(decorator)

    @staticmethod
    def _get_name(node):
        """Extract name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{PythonASTAnalyzer._get_name(node.value)}.{node.attr}"
        return str(node)


class ChangeDetector:
    """Detect changes in Python files and map to TypeScript."""

    def __init__(self, repo_root: Path, mapping_file: Path) -> None:
        self.repo_root = repo_root
        self.mapping_file = mapping_file
        self.file_mappings = self._load_file_mappings()

    def _load_file_mappings(self) -> dict:
        """Load FILE_MAPPING.json."""
        if not self.mapping_file.exists():
            return {"mappings": []}

        with open(self.mapping_file) as f:
            return json.load(f)

    def _save_file_mappings(self) -> None:
        """Save FILE_MAPPING.json."""
        with open(self.mapping_file, "w") as f:
            json.dump(self.file_mappings, f, indent=2)

    def get_changed_python_files(self, base_ref: str = "HEAD") -> list[str]:
        """Get list of Python files changed in git."""
        try:
            # Get staged files
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
            files = result.stdout.strip().split("\n")
            return [f for f in files if f.endswith(".py") and f.startswith("src/")]
        except subprocess.CalledProcessError:
            return []

    def analyze_file(self, file_path: Path) -> PythonASTAnalyzer | None:
        """Analyze Python file using AST."""
        try:
            with open(file_path) as f:
                source = f.read()

            tree = ast.parse(source, filename=str(file_path))
            analyzer = PythonASTAnalyzer()
            analyzer.visit(tree)
            return analyzer
        except Exception:
            return None

    def find_mapping(self, python_file: str) -> dict | None:
        """Find TypeScript mapping for Python file."""
        for mapping in self.file_mappings.get("mappings", []):
            if mapping["python"] == python_file:
                return mapping
        return None

    def compare_files(
        self,
        old_analysis: PythonASTAnalyzer | None,
        new_analysis: PythonASTAnalyzer | None,
    ) -> FileChange:
        """Compare old and new file analyses to detect changes."""
        changes = FileChange(
            python_file="",
            typescript_file=None,
            status="unknown",
            phase=None,
            priority=None,
        )

        if old_analysis is None and new_analysis is not None:
            # File added
            for class_name, class_info in new_analysis.classes.items():
                changes.classes.append(
                    ClassChange(
                        type="added",
                        name=class_name,
                        line_range=(class_info["lineno"], class_info["end_lineno"]),
                    )
                )
            changes.functions = new_analysis.functions
        elif old_analysis is not None and new_analysis is not None:
            # File modified - compare classes
            old_classes = set(old_analysis.classes.keys())
            new_classes = set(new_analysis.classes.keys())

            # Deleted classes
            for class_name in old_classes - new_classes:
                changes.classes.append(ClassChange(type="deleted", name=class_name))

            # Added classes
            for class_name in new_classes - old_classes:
                class_info = new_analysis.classes[class_name]
                changes.classes.append(
                    ClassChange(
                        type="added",
                        name=class_name,
                        line_range=(class_info["lineno"], class_info["end_lineno"]),
                    )
                )

            # Modified classes
            for class_name in old_classes & new_classes:
                old_class = old_analysis.classes[class_name]
                new_class = new_analysis.classes[class_name]

                old_methods = {m["name"]: m for m in old_class["methods"]}
                new_methods = {m["name"]: m for m in new_class["methods"]}

                method_changes = []

                # Check for deleted methods
                for method_name in old_methods.keys() - new_methods.keys():
                    method_changes.append(
                        MethodChange(type="deleted", name=method_name)
                    )

                # Check for added methods
                for method_name in new_methods.keys() - old_methods.keys():
                    method = new_methods[method_name]
                    method_changes.append(
                        MethodChange(
                            type="added",
                            name=method_name,
                            line_range=(method["lineno"], method["end_lineno"]),
                        )
                    )

                # Check for modified methods
                for method_name in old_methods.keys() & new_methods.keys():
                    old_method = old_methods[method_name]
                    new_method = new_methods[method_name]

                    details = {}
                    if old_method["args"] != new_method["args"]:
                        details["args_changed"] = {
                            "old": old_method["args"],
                            "new": new_method["args"],
                        }
                    if old_method["decorators"] != new_method["decorators"]:
                        details["decorators_changed"] = {
                            "old": old_method["decorators"],
                            "new": new_method["decorators"],
                        }

                    if details:
                        method_changes.append(
                            MethodChange(
                                type="modified",
                                name=method_name,
                                line_range=(
                                    new_method["lineno"],
                                    new_method["end_lineno"],
                                ),
                                change_details=details,
                            )
                        )

                if method_changes:
                    changes.classes.append(
                        ClassChange(
                            type="modified",
                            name=class_name,
                            line_range=(new_class["lineno"], new_class["end_lineno"]),
                            methods=method_changes,
                        )
                    )

        return changes

    def detect_changes(self) -> list[FileChange]:
        """Detect all changes and map to TypeScript files."""
        changed_files = self.get_changed_python_files()
        file_changes = []

        for python_file in changed_files:
            file_path = self.repo_root / python_file

            # Analyze current file
            current_analysis = self.analyze_file(file_path)

            # Get mapping
            mapping = self.find_mapping(python_file)

            # Get old version for comparison
            try:
                old_content = subprocess.run(
                    ["git", "show", f"HEAD:{python_file}"],
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout

                old_tree = ast.parse(old_content, filename=python_file)
                old_analyzer = PythonASTAnalyzer()
                old_analyzer.visit(old_tree)
            except:
                old_analyzer = None

            # Compare
            change = self.compare_files(old_analyzer, current_analysis)
            change.python_file = python_file

            if mapping:
                change.typescript_file = mapping.get("typescript")
                change.status = mapping.get("status", "not_mapped")
                change.phase = mapping.get("phase")
                change.priority = mapping.get("priority")
            else:
                change.status = "not_mapped"

            file_changes.append(change)

        return file_changes

    def generate_report(self, changes: list[FileChange]) -> dict:
        """Generate JSON report of changes."""
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "total_files_changed": len(changes),
            "changes": [
                {
                    "pythonFile": change.python_file,
                    "typescriptFile": change.typescript_file,
                    "status": change.status,
                    "phase": change.phase,
                    "priority": change.priority,
                    "classes": [
                        {
                            "type": cls.type,
                            "name": cls.name,
                            "lineRange": list(cls.line_range)
                            if cls.line_range
                            else None,
                            "methods": [
                                {
                                    "type": method.type,
                                    "name": method.name,
                                    "lineRange": list(method.line_range)
                                    if method.line_range
                                    else None,
                                    "changeDetails": method.change_details,
                                }
                                for method in cls.methods
                            ],
                        }
                        for cls in change.classes
                    ],
                    "functions": change.functions,
                }
                for change in changes
            ],
        }

    def update_mapping_timestamps(self, changes: list[FileChange]):
        """Update timestamps in FILE_MAPPING.json."""
        now = datetime.now(UTC).isoformat()

        for change in changes:
            for mapping in self.file_mappings.get("mappings", []):
                if mapping["python"] == change.python_file:
                    mapping["lastPythonChange"] = now
                    break

        self._save_file_mappings()


def main():
    parser = argparse.ArgumentParser(
        description="Detect Python code changes and map to TypeScript"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Root directory of repository",
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=Path("FILE_MAPPING.json"),
        help="Path to FILE_MAPPING.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file (default: print to stdout)",
    )
    parser.add_argument(
        "--ci-mode",
        action="store_true",
        help="CI mode: exit with error if out of sync",
    )
    parser.add_argument(
        "--update-timestamps",
        action="store_true",
        help="Update lastPythonChange timestamps in FILE_MAPPING.json",
    )

    args = parser.parse_args()

    detector = ChangeDetector(args.repo_root, args.repo_root / args.mapping_file)
    changes = detector.detect_changes()
    report = detector.generate_report(changes)

    # Output report
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
    else:
        pass

    # Update timestamps if requested
    if args.update_timestamps:
        detector.update_mapping_timestamps(changes)

    # CI mode: check for out-of-sync files
    if args.ci_mode:
        out_of_sync = [
            c
            for c in changes
            if c.status in ("not_started", "partial") and c.typescript_file
        ]
        if out_of_sync:
            for _change in out_of_sync:
                pass
            sys.exit(1)
        else:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
