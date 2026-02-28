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
        if base == "Union":
            types = [t.strip() for t in inner.split(",")]
            return " | ".join([translate_type(t, patterns) for t in types])
        if base == "tuple":
            types = [t.strip() for t in inner.split(",")]
            return f"[{', '.join([translate_type(t, patterns) for t in types])}]"

    # Default to any if unknown
    return "any"


def parse_subscript_annotation(node: ast.Subscript, patterns: dict) -> str:
    """Parse complex subscript type annotations (List[int], Dict[str, Any], etc.)."""
    if isinstance(node.value, ast.Name):
        base_type = node.value.id
        
        # Handle different slice types
        if isinstance(node.slice, ast.Name):
            # Simple generic: List[int]
            inner_type = translate_type(node.slice.id, patterns)
        elif isinstance(node.slice, ast.Subscript):
            # Nested generic: List[Dict[str, int]]
            inner_type = parse_subscript_annotation(node.slice, patterns)
        elif isinstance(node.slice, ast.Tuple):
            # Multiple args: Dict[str, int], Union[str, int]
            inner_types = []
            for elt in node.slice.elts:
                if isinstance(elt, ast.Name):
                    inner_types.append(translate_type(elt.id, patterns))
                elif isinstance(elt, ast.Subscript):
                    inner_types.append(parse_subscript_annotation(elt, patterns))
                elif isinstance(elt, ast.Constant):
                    inner_types.append(str(elt.value))
                else:
                    inner_types.append("any")
            
            if base_type == "dict" or base_type == "Dict":
                if len(inner_types) >= 2:
                    return f"Record<{inner_types[0]}, {inner_types[1]}>"
            elif base_type == "Union":
                return " | ".join(inner_types)
            elif base_type == "tuple" or base_type == "Tuple":
                return f"[{', '.join(inner_types)}]"
            else:
                return f"{base_type}<{', '.join(inner_types)}>"
        elif isinstance(node.slice, ast.Constant):
            # Literal type
            inner_type = str(node.slice.value)
        else:
            inner_type = "any"
        
        # Apply type transformations
        if base_type in ["list", "List"]:
            return f"{inner_type}[]"
        elif base_type in ["dict", "Dict"]:
            return f"Record<string, {inner_type}>"
        elif base_type in ["Optional"]:
            return f"{inner_type} | null | undefined"
        elif base_type in ["set", "Set"]:
            return f"Set<{inner_type}>"
        else:
            return f"{base_type}<{inner_type}>"
    
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

    def __init__(self, patterns: dict, file_mapping: dict = None) -> None:
        self.patterns = patterns
        self.file_mapping = file_mapping or {}
        self.output = []
        self.imports: set[str] = set()
        self.import_sources: dict[str, set[str]] = {}  # module -> {symbols}
        self.indent_level = 0

    def indent(self) -> str:
        return "  " * self.indent_level
    
    def add_import(self, symbol: str, from_module: str = None) -> None:
        """Track imports needed for generated code."""
        if from_module:
            if from_module not in self.import_sources:
                self.import_sources[from_module] = set()
            self.import_sources[from_module].add(symbol)
        else:
            self.imports.add(symbol)
    
    def generate_imports(self) -> str:
        """Generate import statements."""
        import_lines = []
        
        # Generate named imports
        for module, symbols in sorted(self.import_sources.items()):
            symbols_str = ", ".join(sorted(symbols))
            # Convert Python module path to TypeScript relative path
            ts_module = self._python_module_to_ts(module)
            import_lines.append(f"import {{ {symbols_str} }} from '{ts_module}';")
        
        # Add standalone imports
        for imp in sorted(self.imports):
            import_lines.append(imp)
        
        if import_lines:
            return "\n".join(import_lines) + "\n\n"
        return ""
    
    def _python_module_to_ts(self, python_module: str) -> str:
        """Convert Python module path to TypeScript relative path."""
        # Remove 'src.' prefix if present
        if python_module.startswith("src."):
            python_module = python_module[4:]
        
        # Convert module separators
        ts_path = python_module.replace(".", "/")
        
        # Make it a relative import
        if not ts_path.startswith("."):
            ts_path = "./" + ts_path
        
        return ts_path

    def visit_ClassDef(self, node: ast.ClassDef):
        """Generate TypeScript class."""
        # Class docstring
        docstring = ast.get_docstring(node)
        if docstring:
            self.output.append(translate_docstring(docstring))

        # Class declaration
        bases = []
        interfaces = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_name = base.id
                bases.append(base_name)
                # Track import for base class (assuming same module structure)
                # This is a simplification - in practice, would need FILE_MAPPING lookup
                if base_name not in ["object", "ABC"]:
                    self.add_import(base_name, f"../{snake_to_camel(base_name)}")

        # TypeScript: first base is extends, rest are implements
        extends = f" extends {bases[0]}" if bases else ""
        implements = f" implements {', '.join(bases[1:])}" if len(bases) > 1 else ""
        self.output.append(f"export class {node.name}{extends}{implements} {{")
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
                    # Handle generic types like list[int], Dict[str, Any]
                    param_type = parse_subscript_annotation(arg.annotation, self.patterns)

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
                return_type = parse_subscript_annotation(node.returns, self.patterns)
            elif isinstance(node.returns, ast.Constant) and node.returns.value is None:
                return_type = "void"

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

    def extract_properties_from_init(self, node: ast.FunctionDef) -> list[dict]:
        """Extract property assignments from __init__ body."""
        properties = []
        
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        if target.value.id == "self":
                            prop_name = target.attr
                            prop_type = self._infer_type_from_value(stmt.value)
                            properties.append({
                                "name": snake_to_camel(prop_name),
                                "type": prop_type,
                                "private": prop_name.startswith("_"),
                                "original_name": prop_name
                            })
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Attribute):
                # Annotated assignment: self.prop: Type = value
                if isinstance(stmt.target.value, ast.Name) and stmt.target.value.id == "self":
                    prop_name = stmt.target.attr
                    if stmt.annotation:
                        if isinstance(stmt.annotation, ast.Name):
                            prop_type = translate_type(stmt.annotation.id, self.patterns)
                        elif isinstance(stmt.annotation, ast.Subscript):
                            prop_type = parse_subscript_annotation(stmt.annotation, self.patterns)
                        else:
                            prop_type = "any"
                    else:
                        prop_type = self._infer_type_from_value(stmt.value) if stmt.value else "any"
                    properties.append({
                        "name": snake_to_camel(prop_name),
                        "type": prop_type,
                        "private": prop_name.startswith("_"),
                        "original_name": prop_name
                    })
        
        return properties
    
    def _infer_type_from_value(self, value_node) -> str:
        """Infer TypeScript type from Python value."""
        if isinstance(value_node, ast.Constant):
            if isinstance(value_node.value, bool):
                return "boolean"
            elif isinstance(value_node.value, int):
                return "number"
            elif isinstance(value_node.value, float):
                return "number"
            elif isinstance(value_node.value, str):
                return "string"
            elif value_node.value is None:
                return "null"
        elif isinstance(value_node, ast.List):
            return "any[]"
        elif isinstance(value_node, ast.Dict):
            return "Record<string, any>"
        elif isinstance(value_node, ast.Set):
            return "Set<any>"
        elif isinstance(value_node, ast.Call):
            # Constructor call
            if isinstance(value_node.func, ast.Name):
                return value_node.func.id
        return "any"

    def generate_constructor(self, node: ast.FunctionDef):
        """Generate TypeScript constructor from __init__ with property declarations."""
        # Extract properties from __init__ body
        properties = self.extract_properties_from_init(node)
        
        # Declare properties before constructor
        if properties:
            for prop in properties:
                visibility = "private" if prop["private"] else "public"
                self.output.append(f"{self.indent()}{visibility} {prop['name']}: {prop['type']};")
            self.output.append("")  # Blank line
        
        # Generate constructor
        self.output.append(f"{self.indent()}constructor(")
        self.indent_level += 1

        # Parameters (skip self)
        params = []
        for arg in node.args.args[1:]:  # Skip 'self'
            param_name = snake_to_camel(arg.arg)
            param_type = "any"

            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    param_type = translate_type(arg.annotation.id, self.patterns)
                elif isinstance(arg.annotation, ast.Subscript):
                    param_type = parse_subscript_annotation(arg.annotation, self.patterns)

            params.append(f"{self.indent()}{param_name}: {param_type}")

        if params:
            self.output.append(",\n".join(params))

        self.indent_level -= 1
        self.output.append(f"{self.indent()}) {{")
        self.indent_level += 1
        
        # Generate property assignments
        if properties:
            for prop in properties:
                # Try to match constructor parameter
                matching_param = None
                for arg in node.args.args[1:]:
                    if snake_to_camel(arg.arg) == prop['name']:
                        matching_param = prop['name']
                        break
                
                if matching_param:
                    self.output.append(f"{self.indent()}this.{prop['name']} = {matching_param};")
                else:
                    self.output.append(f"{self.indent()}// TODO: Initialize {prop['name']}")
        else:
            self.output.append(f"{self.indent()}// TODO: Initialize properties")
        
        self.indent_level -= 1
        self.output.append(f"{self.indent()}}}")

    def get_output(self, python_file: str = None) -> str:
        """Generate final TypeScript output with imports and header."""
        result = []
        
        # Add header comment
        if python_file:
            result.append(f"/**")
            result.append(f" * AUTO-GENERATED from Python source: {python_file}")
            result.append(f" * DO NOT EDIT MANUALLY - Use migration scripts")
            result.append(f" *")
            result.append(f" * Manual review required for:")
            result.append(f" * - TODO items marked below")
            result.append(f" * - Import paths (verify correctness)")
            result.append(f" * - Type annotations (review 'any' types)")
            result.append(f" * - OpenCV memory management (add try/finally where needed)")
            result.append(f" */")
            result.append("")
        
        # Add imports
        imports = self.generate_imports()
        if imports:
            result.append(imports)
        
        # Add generated code
        result.append("\n".join(self.output))
        
        return "\n".join(result)


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

    return generator.get_output(str(python_file))


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
