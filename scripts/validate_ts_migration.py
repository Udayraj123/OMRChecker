#!/usr/bin/env python3
"""
TypeScript Migration Validation Script

Validates that a TypeScript file matches the structure of its Python source.
Checks:
- Class structure (names, presence)
- Method counts and names
- Type annotation coverage
- Import completeness
- Migration quality score
"""

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ValidationResult:
    """Result of validation check."""

    passed: bool
    message: str
    details: dict[str, Any] | None = None


@dataclass
class MigrationReport:
    """Complete migration validation report."""

    python_file: str
    typescript_file: str
    overall_score: float
    results: list[ValidationResult]

    def print_report(self) -> None:
        """Print formatted validation report."""
        print(f"\n{'=' * 70}")
        print(f"Migration Validation Report")
        print(f"{'=' * 70}")
        print(f"Python:     {self.python_file}")
        print(f"TypeScript: {self.typescript_file}")
        print(f"Score:      {self.overall_score:.1f}%")
        print(f"{'=' * 70}\n")

        for result in self.results:
            icon = "✅" if result.passed else "❌"
            print(f"{icon} {result.message}")
            if result.details:
                for key, value in result.details.items():
                    print(f"   {key}: {value}")
            print()

        if self.overall_score >= 90:
            print("🎉 Excellent migration quality!")
        elif self.overall_score >= 70:
            print("⚠️  Good migration, minor issues to address")
        else:
            print("🔴 Significant issues found - review needed")


class PythonAnalyzer:
    """Analyze Python source file."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.classes: dict[str, dict] = {}
        self.functions: list[str] = []

    def analyze(self) -> None:
        """Parse and analyze Python file."""
        with open(self.file_path) as f:
            source = f.read()

        tree = ast.parse(source, filename=str(self.file_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self._analyze_class(node)
            elif isinstance(node, ast.FunctionDef) and not self._is_in_class(node):
                self.functions.append(node.name)

    def _analyze_class(self, node: ast.ClassDef) -> None:
        """Analyze a class definition."""
        methods = []
        properties = []

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = {
                    "name": item.name,
                    "is_static": any(
                        isinstance(d, ast.Name) and d.id == "staticmethod"
                        for d in item.decorator_list
                    ),
                    "is_abstract": any(
                        isinstance(d, ast.Name) and d.id == "abstractmethod"
                        for d in item.decorator_list
                    ),
                    "has_type_hints": item.returns is not None
                    or any(arg.annotation for arg in item.args.args),
                }
                methods.append(method_info)
            elif isinstance(item, ast.AnnAssign):
                # Class-level properties
                if isinstance(item.target, ast.Name):
                    properties.append(item.target.id)

        self.classes[node.name] = {
            "methods": methods,
            "properties": properties,
            "method_count": len(methods),
            "public_method_count": len(
                [m for m in methods if not m["name"].startswith("_")]
            ),
        }

    def _is_in_class(self, node: ast.FunctionDef) -> bool:
        """Check if function is inside a class."""
        # Simplified check - in practice would need parent tracking
        return False


class TypeScriptAnalyzer:
    """Analyze TypeScript file."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.classes: dict[str, dict] = {}
        self.functions: list[str] = []
        self.imports: list[str] = []
        self.has_any_types: bool = False
        self.any_type_count: int = 0

    def analyze(self) -> None:
        """Parse and analyze TypeScript file."""
        if not self.file_path.exists():
            return

        with open(self.file_path) as f:
            content = f.read()

        # Extract imports
        import_pattern = re.compile(r"^import\s+(?:{[^}]+}|[\w]+)\s+from\s+['\"]([^'\"]+)['\"];?", re.MULTILINE)
        self.imports = import_pattern.findall(content)

        # Count 'any' types
        self.any_type_count = len(re.findall(r":\s*any\b", content))
        self.has_any_types = self.any_type_count > 0

        # Extract classes
        class_pattern = re.compile(
            r"export\s+(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?\s*{([^}]*(?:{[^}]*}[^}]*)*)}"
        )

        for match in class_pattern.finditer(content):
            class_name = match.group(1)
            class_body = match.group(2)

            methods = self._extract_methods(class_body)

            self.classes[class_name] = {
                "methods": methods,
                "method_count": len(methods),
                "public_method_count": len(
                    [m for m in methods if not m["name"].startswith("_")]
                ),
            }

    def _extract_methods(self, class_body: str) -> list[dict]:
        """Extract method information from class body."""
        methods = []

        # Match method signatures (simplified)
        method_pattern = re.compile(
            r"(?:(static|private|protected|public|abstract)\s+)?"
            r"(\w+)\s*\([^)]*\)\s*:\s*(\w+)"
        )

        for match in method_pattern.finditer(class_body):
            modifier = match.group(1)
            name = match.group(2)
            return_type = match.group(3)

            if name in ["constructor"]:
                continue  # Skip constructor in method count

            methods.append(
                {
                    "name": name,
                    "is_static": modifier == "static",
                    "return_type": return_type,
                }
            )

        return methods


class MigrationValidator:
    """Validate Python to TypeScript migration."""

    def __init__(self, python_file: Path, typescript_file: Path):
        self.python_file = python_file
        self.typescript_file = typescript_file
        self.python_analyzer = PythonAnalyzer(python_file)
        self.typescript_analyzer = TypeScriptAnalyzer(typescript_file)

    def validate(self) -> MigrationReport:
        """Run all validation checks."""
        self.python_analyzer.analyze()
        self.typescript_analyzer.analyze()

        results = [
            self._check_file_exists(),
            self._check_class_structure(),
            self._check_method_counts(),
            self._check_type_coverage(),
            self._check_imports(),
        ]

        # Calculate overall score
        passed_count = sum(1 for r in results if r.passed)
        total_checks = len(results)
        overall_score = (passed_count / total_checks) * 100 if total_checks > 0 else 0

        return MigrationReport(
            python_file=str(self.python_file),
            typescript_file=str(self.typescript_file),
            overall_score=overall_score,
            results=results,
        )

    def _check_file_exists(self) -> ValidationResult:
        """Check if TypeScript file exists."""
        if self.typescript_file.exists():
            return ValidationResult(
                passed=True,
                message="TypeScript file exists",
            )
        else:
            return ValidationResult(
                passed=False,
                message="TypeScript file not found",
                details={"path": str(self.typescript_file)},
            )

    def _check_class_structure(self) -> ValidationResult:
        """Check if class structure matches."""
        py_classes = set(self.python_analyzer.classes.keys())
        ts_classes = set(self.typescript_analyzer.classes.keys())

        if py_classes == ts_classes:
            return ValidationResult(
                passed=True,
                message=f"Class structure matches ({len(py_classes)} classes)",
                details={"classes": list(py_classes)},
            )
        else:
            missing = py_classes - ts_classes
            extra = ts_classes - py_classes
            return ValidationResult(
                passed=False,
                message="Class structure mismatch",
                details={
                    "missing_in_ts": list(missing),
                    "extra_in_ts": list(extra),
                },
            )

    def _check_method_counts(self) -> ValidationResult:
        """Check if method counts match."""
        mismatches = []

        for class_name in self.python_analyzer.classes:
            if class_name not in self.typescript_analyzer.classes:
                continue

            py_count = self.python_analyzer.classes[class_name]["public_method_count"]
            ts_count = self.typescript_analyzer.classes[class_name]["public_method_count"]

            if py_count != ts_count:
                mismatches.append(
                    {
                        "class": class_name,
                        "python": py_count,
                        "typescript": ts_count,
                    }
                )

        if not mismatches:
            return ValidationResult(
                passed=True,
                message="Method counts match",
            )
        else:
            return ValidationResult(
                passed=False,
                message="Method count mismatches found",
                details={"mismatches": mismatches},
            )

    def _check_type_coverage(self) -> ValidationResult:
        """Check type annotation coverage."""
        total_params = 0
        typed_params = 0

        for class_info in self.python_analyzer.classes.values():
            for method in class_info["methods"]:
                if method["has_type_hints"]:
                    typed_params += 1
                total_params += 1

        coverage = (typed_params / total_params * 100) if total_params > 0 else 100
        any_ratio = self.typescript_analyzer.any_type_count

        details = {
            "python_type_coverage": f"{coverage:.0f}%",
            "typescript_any_count": any_ratio,
        }

        if any_ratio > 10:
            return ValidationResult(
                passed=False,
                message=f"⚠️  Type annotations: {any_ratio} 'any' types found",
                details=details,
            )
        elif any_ratio > 5:
            return ValidationResult(
                passed=True,
                message=f"Type annotations: {any_ratio} 'any' types (acceptable)",
                details=details,
            )
        else:
            return ValidationResult(
                passed=True,
                message=f"✨ Type annotations: only {any_ratio} 'any' types",
                details=details,
            )

    def _check_imports(self) -> ValidationResult:
        """Check if TypeScript has imports."""
        import_count = len(self.typescript_analyzer.imports)

        if import_count == 0:
            return ValidationResult(
                passed=False,
                message="⚠️  No imports found (verify if needed)",
                details={"import_count": 0},
            )
        else:
            return ValidationResult(
                passed=True,
                message=f"Imports present ({import_count} modules)",
                details={"modules": self.typescript_analyzer.imports[:5]},
            )


def main():
    parser = argparse.ArgumentParser(
        description="Validate Python to TypeScript migration"
    )
    parser.add_argument(
        "--python-file",
        type=Path,
        required=True,
        help="Path to Python source file",
    )
    parser.add_argument(
        "--typescript-file",
        type=Path,
        required=True,
        help="Path to TypeScript target file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=70.0,
        help="Minimum passing score (default: 70)",
    )

    args = parser.parse_args()

    if not args.python_file.exists():
        print(f"❌ Python file not found: {args.python_file}", file=sys.stderr)
        sys.exit(1)

    validator = MigrationValidator(args.python_file, args.typescript_file)
    report = validator.validate()

    if args.json:
        output = {
            "python_file": report.python_file,
            "typescript_file": report.typescript_file,
            "score": report.overall_score,
            "passed": report.overall_score >= args.min_score,
            "results": [
                {
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details,
                }
                for r in report.results
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        report.print_report()

    # Exit with error if score below threshold
    if report.overall_score < args.min_score:
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
