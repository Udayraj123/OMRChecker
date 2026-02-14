#!/usr/bin/env python3
"""
Sync TypeScript files from Python changes.

This script applies structural synchronization (add/remove class and method stubs)
to TypeScript files based on detected Python changes.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    if name.startswith("_"):
        # Private method
        return "_" + snake_to_camel(name[1:])
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def load_file_mapping(repo_root: Path) -> dict:
    """Load FILE_MAPPING.json."""
    mapping_file = repo_root / "FILE_MAPPING.json"
    if not mapping_file.exists():
        return {"mappings": []}

    with open(mapping_file) as f:
        return json.load(f)


def get_method_name_mapping(
    file_mapping: dict, python_class: str, python_method: str
) -> str | None:
    """Get TypeScript method name from FILE_MAPPING if available."""
    for mapping in file_mapping.get("mappings", []):
        for cls in mapping.get("classes", []):
            if cls.get("python") == python_class:
                for method in cls.get("methods", []):
                    if method.get("python") == python_method:
                        return method.get("typescript")
    return None


class TypeScriptFileSync:
    """Sync TypeScript file based on Python changes."""

    def __init__(
        self, ts_file_path: Path, file_mapping: dict, python_file: str
    ) -> None:
        self.ts_file_path = ts_file_path
        self.file_mapping = file_mapping
        self.python_file = python_file
        self.lines: list[str] = []
        self.modified = False

        # Load existing TS file if it exists
        if ts_file_path.exists():
            with open(ts_file_path) as f:
                self.lines = f.read().splitlines()

    def find_class(self, class_name: str) -> tuple[int, int] | None:
        """Find class definition and return (start_line, end_line)."""
        class_pattern = re.compile(
            rf"^\s*export\s+(?:abstract\s+)?class\s+{re.escape(class_name)}\s*"
        )

        for i, line in enumerate(self.lines):
            if class_pattern.match(line):
                # Find matching closing brace
                brace_count = 0
                for j in range(i, len(self.lines)):
                    brace_count += self.lines[j].count("{") - self.lines[j].count("}")
                    if brace_count == 0 and "{" in self.lines[i]:
                        return (i, j)
                return (i, len(self.lines) - 1)
        return None

    def find_method_in_class(
        self, class_start: int, class_end: int, method_name: str
    ) -> tuple[int, int] | None:
        """Find method definition within a class and return (start_line, end_line)."""
        # Match method patterns: methodName(, private methodName(, static methodName(, etc.
        method_pattern = re.compile(
            rf"^\s*(?:(?:public|private|protected|static|async|abstract)\s+)*{re.escape(method_name)}\s*\("
        )

        for i in range(class_start, class_end + 1):
            if method_pattern.match(self.lines[i]):
                # Find matching closing brace
                brace_count = 0
                in_method = False
                for j in range(i, class_end + 1):
                    if "{" in self.lines[j]:
                        in_method = True
                    if in_method:
                        brace_count += self.lines[j].count("{") - self.lines[j].count(
                            "}"
                        )
                        if brace_count == 0:
                            return (i, j)
                # Abstract method or just signature
                if ";" in self.lines[i] or "abstract" in self.lines[i]:
                    return (i, i)
                return (i, i)
        return None

    def get_class_indent(self, class_start: int) -> str:
        """Get indentation for methods inside a class."""
        # Look at the first line inside the class to determine indent
        for i in range(class_start + 1, min(class_start + 10, len(self.lines))):
            line = self.lines[i]
            if line.strip() and not line.strip().startswith("//"):
                # Extract leading whitespace
                match = re.match(r"^(\s+)", line)
                if match:
                    return match.group(1)
        # Default to 2 spaces
        return "  "

    def add_class_stub(self, class_name: str, base_class: str | None = None) -> None:
        """Add a new class stub to the file."""
        extends = f" extends {base_class}" if base_class else ""
        class_lines = [
            "",
            f"export class {class_name}{extends} {{",
            "  // TODO: Implement class members",
            "}",
            "",
        ]

        # Add at the end of file
        self.lines.extend(class_lines)
        self.modified = True

    def add_method_stub(
        self,
        class_name: str,
        method_name: str,
        is_static: bool = False,
        is_abstract: bool = False,
    ) -> None:
        """Add a new method stub to a class."""
        class_range = self.find_class(class_name)
        if not class_range:
            # Class doesn't exist, skip
            return

        class_start, class_end = class_range
        indent = self.get_class_indent(class_start)

        # Find a good place to insert the method (before the closing brace)
        insert_line = class_end

        # Build method signature
        modifiers = []
        if is_static:
            modifiers.append("static")
        if is_abstract:
            modifiers.append("abstract")

        modifier_str = " ".join(modifiers) + " " if modifiers else ""

        if is_abstract:
            method_lines = [
                f"{indent}{modifier_str}{method_name}(): any; // TODO: Add parameters and return type"
            ]
        else:
            method_lines = [
                "",
                f"{indent}{modifier_str}{method_name}(): any {{ // TODO: Add parameters and return type",
                f"{indent}  // TODO: Implement",
                f"{indent}}}",
            ]

        # Insert before closing brace
        self.lines[insert_line:insert_line] = method_lines
        self.modified = True

    def remove_method(self, class_name: str, method_name: str) -> None:
        """Comment out or remove a method from a class."""
        class_range = self.find_class(class_name)
        if not class_range:
            return

        class_start, class_end = class_range
        method_range = self.find_method_in_class(class_start, class_end, method_name)
        if not method_range:
            return

        method_start, method_end = method_range

        # Comment out the method instead of removing it
        for i in range(method_start, method_end + 1):
            if not self.lines[i].strip().startswith("//"):
                self.lines[i] = "// " + self.lines[i]

        self.modified = True

    def remove_class(self, class_name: str) -> None:
        """Comment out or remove a class."""
        class_range = self.find_class(class_name)
        if not class_range:
            return

        class_start, class_end = class_range

        # Comment out the class instead of removing it
        for i in range(class_start, class_end + 1):
            if not self.lines[i].strip().startswith("//"):
                self.lines[i] = "// " + self.lines[i]

        self.modified = True

    def apply_changes(self, change_report: dict[str, Any]) -> None:
        """Apply changes from change report to the TypeScript file."""
        for cls in change_report.get("classes", []):
            class_name = cls["name"]
            change_type = cls["type"]

            if change_type == "added":
                # Check if class already exists
                if not self.find_class(class_name):
                    self.add_class_stub(class_name)

            elif change_type == "deleted":
                self.remove_class(class_name)

            elif change_type == "modified":
                # Handle method changes
                for method in cls.get("methods", []):
                    method_name = method["name"]
                    method_type = method["type"]

                    # Convert Python method name to TypeScript (check FILE_MAPPING first)
                    ts_method_name = get_method_name_mapping(
                        self.file_mapping, class_name, method_name
                    )
                    if not ts_method_name:
                        # Fall back to snake_to_camel conversion
                        if method_name == "__init__":
                            ts_method_name = "constructor"
                        else:
                            ts_method_name = snake_to_camel(method_name)

                    if method_type == "added":
                        # Check if method already exists
                        class_range = self.find_class(class_name)
                        if class_range:
                            if not self.find_method_in_class(
                                class_range[0], class_range[1], ts_method_name
                            ):
                                self.add_method_stub(class_name, ts_method_name)

                    elif method_type == "deleted":
                        self.remove_method(class_name, ts_method_name)

                    elif method_type == "modified":
                        # For modified methods, we could add a comment or log
                        # For now, we don't auto-update implementation
                        pass

    def save(self) -> bool:
        """Save changes to file if modified."""
        if not self.modified:
            return False

        # Ensure parent directory exists
        self.ts_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.ts_file_path, "w") as f:
            f.write("\n".join(self.lines) + "\n")

        return True


def sync_file(
    repo_root: Path,
    python_file: str,
    typescript_file: str,
    change_report: dict[str, Any],
    file_mapping: dict,
) -> bool:
    """Sync a single TypeScript file based on Python changes."""
    ts_path = repo_root / typescript_file

    syncer = TypeScriptFileSync(ts_path, file_mapping, python_file)
    syncer.apply_changes(change_report)

    return syncer.save()


def main():
    parser = argparse.ArgumentParser(
        description="Sync TypeScript files from Python changes"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root directory",
    )
    parser.add_argument(
        "--changes-json",
        type=Path,
        required=True,
        help="JSON file with detected changes",
    )
    parser.add_argument(
        "--stage",
        action="store_true",
        help="Stage modified TypeScript files in git",
    )

    args = parser.parse_args()

    # Load FILE_MAPPING
    file_mapping = load_file_mapping(args.repo_root)

    # Load change report
    with open(args.changes_json) as f:
        change_data = json.load(f)

    modified_files = []

    for change in change_data.get("changes", []):
        python_file = change["pythonFile"]
        typescript_file = change.get("typescriptFile")

        if not typescript_file or typescript_file == "Not mapped":
            continue

        # Sync the file
        if sync_file(
            args.repo_root, python_file, typescript_file, change, file_mapping
        ):
            modified_files.append(typescript_file)

    # Print results
    if modified_files:
        print(f"\n✅ Updated {len(modified_files)} TypeScript file(s):")
        for f in modified_files:
            print(f"   - {f}")

        # Stage files if requested
        if args.stage:
            import subprocess

            for f in modified_files:
                subprocess.run(
                    ["git", "add", f],
                    cwd=args.repo_root,
                    check=False,
                    capture_output=True,
                )
            print("\n✅ Staged modified TypeScript files")
    else:
        print("\n✅ No TypeScript files needed updating")

    return 0


if __name__ == "__main__":
    sys.exit(main())
