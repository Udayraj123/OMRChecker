#!/usr/bin/env python3
"""
TypeScript Suggestion Generator

This script generates TypeScript code suggestions based on Python code changes.
Uses CHANGE_PATTERNS.yaml to translate common Python patterns to TypeScript.
"""

import argparse
import ast
import re
import sys
from pathlib import Path

import yaml


def load_patterns(repo_root: Path) -> dict:
    """Load CHANGE_PATTERNS.yaml."""
    patterns_file = repo_root / "CHANGE_PATTERNS.yaml"
    if not patterns_file.exists():
        sys.exit(1)

    with open(patterns_file) as f:
        return yaml.safe_load(f)


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def translate_type(python_type: str, patterns: dict) -> str:
    """Translate Python type to TypeScript type."""
    type_mappings = {
        m["python_type"]: m["typescript_type"]
        for m in patterns.get("type_mappings", [])
    }

    # Direct mapping
    if python_type in type_mappings:
        return type_mappings[python_type]

    # Handle generic types
    if "[" in python_type:
        base = python_type.split("[")[0]
        inner = python_type[python_type.index("[") + 1 : python_type.rindex("]")]

        if base == "list":
            return f"{translate_type(inner, patterns)}[]"
        if base == "dict":
            key, value = [t.strip() for t in inner.split(",")]
            return f"Record<{translate_type(key, patterns)}, {translate_type(value, patterns)}>"
        if base == "Optional":
            return f"{translate_type(inner, patterns)} | null | undefined"

    # Default to any if unknown
    return "any"


def translate_docstring(docstring: str) -> str:
    """Convert Python docstring to JSDoc."""
    if not docstring:
        return ""

    lines = docstring.strip().split("\n")
    jsdoc = ["/**"]

    current_section = None
    for line in lines:
        line = line.strip()

        if line.startswith("Args:"):
            current_section = "params"
            continue
        if line.startswith("Returns:"):
            current_section = "returns"
            jsdoc.append(" *")
            continue
        if line.startswith("Raises:"):
            current_section = "throws"
            jsdoc.append(" *")
            continue
        if not line:
            jsdoc.append(" *")
            continue

        if current_section == "params":
            # Parse "param_name: description"
            match = re.match(r"(\w+):\s*(.+)", line)
            if match:
                param_name, desc = match.groups()
                jsdoc.append(f" * @param {snake_to_camel(param_name)} - {desc}")
        elif current_section == "returns":
            jsdoc.append(f" * @returns {line}")
        elif current_section == "throws":
            # Parse "ExceptionType: description"
            match = re.match(r"(\w+):\s*(.+)", line)
            if match:
                exc_type, desc = match.groups()
                jsdoc.append(f" * @throws {{{exc_type}}} {desc}")
        else:
            jsdoc.append(f" * {line}")

    jsdoc.append(" */")
    return "\n".join(jsdoc)


class TypeScriptGenerator(ast.NodeVisitor):
    """Generate TypeScript code from Python AST."""

    def __init__(self, patterns: dict) -> None:
        self.patterns = patterns
        self.output = []
        self.indent_level = 0

    def indent(self) -> str:
        return "  " * self.indent_level

    def visit_ClassDef(self, node: ast.ClassDef):
        """Generate TypeScript class."""
        # Class docstring
        docstring = ast.get_docstring(node)
        if docstring:
            self.output.append(translate_docstring(docstring))

        # Class declaration
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)

        extends = f" extends {bases[0]}" if bases else ""
        self.output.append(f"export class {node.name}{extends} {{")
        self.indent_level += 1

        # Process methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                self.visit_FunctionDef(item, is_method=True)
                self.output.append("")

        self.indent_level -= 1
        self.output.append("}")

    def visit_FunctionDef(self, node: ast.FunctionDef, is_method: bool = False):
        """Generate TypeScript function/method."""
        # Skip __init__, convert to constructor
        if node.name == "__init__":
            self.generate_constructor(node)
            return

        # Method docstring
        docstring = ast.get_docstring(node)
        if docstring:
            for line in translate_docstring(docstring).split("\n"):
                self.output.append(self.indent() + line)

        # Function name (convert snake_case to camelCase)
        ts_name = snake_to_camel(node.name) if is_method else node.name

        # Check for decorators
        is_static = any(
            isinstance(d, ast.Name) and d.id == "staticmethod"
            for d in node.decorator_list
        )
        is_abstract = any(
            isinstance(d, ast.Name) and d.id == "abstractmethod"
            for d in node.decorator_list
        )

        # Parameters
        params = []
        for arg in node.args.args:
            if arg.arg in {"self", "cls"}:
                continue

            param_name = snake_to_camel(arg.arg)
            param_type = "any"

            # Try to get type annotation
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    param_type = translate_type(arg.annotation.id, self.patterns)
                elif isinstance(arg.annotation, ast.Subscript):
                    # Handle generic types like list[int]
                    param_type = "any"  # Simplified

            # Check for default value
            defaults_offset = len(node.args.args) - len(node.args.defaults)
            arg_index = node.args.args.index(arg)
            has_default = arg_index >= defaults_offset

            if has_default:
                param_name += "?"

            params.append(f"{param_name}: {param_type}")

        params_str = ", ".join(params)

        # Return type
        return_type = "void"
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return_type = translate_type(node.returns.id, self.patterns)
            elif isinstance(node.returns, ast.Subscript):
                return_type = "any"  # Simplified

        # Generate signature
        modifiers = []
        if is_static:
            modifiers.append("static")
        if is_abstract:
            modifiers.append("abstract")

        modifier_str = " ".join(modifiers) + " " if modifiers else ""

        if is_abstract:
            self.output.append(
                f"{self.indent()}{modifier_str}{ts_name}({params_str}): {return_type};"
            )
        else:
            self.output.append(
                f"{self.indent()}{modifier_str}{ts_name}({params_str}): {return_type} {{"
            )
            self.indent_level += 1
            self.output.append(f"{self.indent()}// TODO: Implement")
            self.indent_level -= 1
            self.output.append(f"{self.indent()}}}")

    def generate_constructor(self, node: ast.FunctionDef):
        """Generate TypeScript constructor from __init__."""
        self.output.append(f"{self.indent()}constructor(")
        self.indent_level += 1

        # Parameters (skip self)
        params = []
        for arg in node.args.args[1:]:  # Skip 'self'
            param_name = snake_to_camel(arg.arg)
            param_type = "any"

            if arg.annotation and isinstance(arg.annotation, ast.Name):
                param_type = translate_type(arg.annotation.id, self.patterns)

            params.append(f"{self.indent()}{param_name}: {param_type}")

        if params:
            self.output.append(",\n".join(params))

        self.indent_level -= 1
        self.output.append(f"{self.indent()}) {{")
        self.indent_level += 1
        self.output.append(f"{self.indent()}// TODO: Initialize properties")
        self.indent_level -= 1
        self.output.append(f"{self.indent()}}}")

    def get_output(self) -> str:
        return "\n".join(self.output)


def generate_typescript_suggestions(python_file: Path, repo_root: Path) -> str:
    """Generate TypeScript code suggestions from Python file."""
    patterns = load_patterns(repo_root)

    # Read Python file
    with open(python_file) as f:
        source = f.read()

    # Parse AST
    try:
        tree = ast.parse(source, filename=str(python_file))
    except SyntaxError:
        sys.exit(1)

    # Generate TypeScript
    generator = TypeScriptGenerator(patterns)

    # Process top-level nodes
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            generator.visit_ClassDef(node)
            generator.output.append("")
        elif isinstance(node, ast.FunctionDef):
            generator.visit_FunctionDef(node, is_method=False)
            generator.output.append("")

    return generator.get_output()


def main():
    parser = argparse.ArgumentParser(
        description="Generate TypeScript suggestions from Python code"
    )
    parser.add_argument("--file", required=True, help="Python file to analyze")
    parser.add_argument("--output", help="Output file for suggestions")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root directory",
    )

    args = parser.parse_args()

    python_file = Path(args.file)
    if not python_file.exists():
        sys.exit(1)

    # Generate suggestions
    typescript_code = generate_typescript_suggestions(python_file, args.repo_root)

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(typescript_code)
    else:
        pass


if __name__ == "__main__":
    main()
