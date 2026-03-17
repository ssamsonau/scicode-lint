"""AST utilities for resolving names to code locations.

Provides robust name-based location resolution for LLM detection results.
LLMs are good at identifying function/class names but unreliable at counting
line numbers. This module bridges the gap by resolving names to actual lines.
"""

import ast
from dataclasses import dataclass
from typing import Literal


@dataclass
class ResolvedLocation:
    """Result of resolving a name to a code location."""

    name: str
    location_type: Literal["function", "class", "method", "module"]
    start_line: int
    end_line: int
    snippet: str

    @property
    def lines(self) -> list[int]:
        """Get all line numbers in the range."""
        return list(range(self.start_line, self.end_line + 1))


class DefinitionFinder(ast.NodeVisitor):
    """AST visitor that finds all function/class/method definitions."""

    def __init__(self, code_lines: list[str]) -> None:
        self.code_lines = code_lines
        self.definitions: list[ResolvedLocation] = []
        self._class_stack: list[str] = []  # Track nested class names

    def _extract_snippet(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
        """Extract the code snippet for a node."""
        start = node.lineno - 1  # 0-indexed
        end = node.end_lineno if node.end_lineno else node.lineno
        return "\n".join(self.code_lines[start:end])

    def _get_qualified_name(self, name: str) -> str:
        """Get fully qualified name including class prefix if nested."""
        if self._class_stack:
            return f"{'.'.join(self._class_stack)}.{name}"
        return name

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Visit a function or async function definition."""
        qualified_name = self._get_qualified_name(node.name)
        location_type: Literal["function", "method"] = "method" if self._class_stack else "function"

        self.definitions.append(
            ResolvedLocation(
                name=qualified_name,
                location_type=location_type,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                snippet=self._extract_snippet(node),
            )
        )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a function definition."""
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit an async function definition."""
        self._visit_func(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition."""
        qualified_name = self._get_qualified_name(node.name)

        self.definitions.append(
            ResolvedLocation(
                name=qualified_name,
                location_type="class",
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                snippet=self._extract_snippet(node),
            )
        )
        # Track class name for method resolution
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()


def find_all_definitions(code: str) -> list[ResolvedLocation]:
    """Find all function, class, and method definitions in code.

    Args:
        code: Python source code

    Returns:
        List of ResolvedLocation objects for all definitions found.
        Returns empty list if code cannot be parsed.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    code_lines = code.splitlines()
    finder = DefinitionFinder(code_lines)
    finder.visit(tree)
    return finder.definitions


def resolve_name(
    code: str,
    name: str,
    location_type: Literal["function", "class", "method", "module"] | None = None,
    near_line: int | None = None,
) -> ResolvedLocation | None:
    """Find a function/class/method by name in code.

    Supports both simple names ("train_model") and qualified names ("Trainer.train").
    Uses near_line to disambiguate when multiple definitions have the same name.

    Args:
        code: Python source code
        name: Name to find (e.g., "train_model" or "Trainer.train")
        location_type: Optional filter by type ("function", "class", "method")
        near_line: Optional hint line number to pick closest match when duplicates exist

    Returns:
        ResolvedLocation if found, None otherwise

    Examples:
        >>> code = '''
        ... def train_model(data):
        ...     pass
        ... '''
        >>> loc = resolve_name(code, "train_model")
        >>> loc.start_line
        2
    """
    # Handle module-level location
    if location_type == "module":
        code_lines = code.splitlines()
        if near_line and 1 <= near_line <= len(code_lines):
            # Return context around near_line (±3 lines)
            start = max(1, near_line - 3)
            end = min(len(code_lines), near_line + 3)
            snippet = "\n".join(code_lines[start - 1 : end])
            return ResolvedLocation(
                name="<module>",
                location_type="module",
                start_line=start,
                end_line=end,
                snippet=snippet,
            )
        # No near_line hint, return first 10 lines as fallback
        snippet = "\n".join(code_lines[:10])
        return ResolvedLocation(
            name="<module>",
            location_type="module",
            start_line=1,
            end_line=min(10, len(code_lines)),
            snippet=snippet,
        )

    definitions = find_all_definitions(code)
    if not definitions:
        return None

    # Find matching definitions
    matches: list[ResolvedLocation] = []
    for defn in definitions:
        # Check type filter
        if location_type and defn.location_type != location_type:
            # Special case: "function" type should also match standalone functions
            # even if pattern says "method"
            if not (location_type == "method" and defn.location_type == "function"):
                continue

        # Check name match
        # Support both exact match and partial match for qualified names
        if defn.name == name:
            # Exact match
            matches.append(defn)
        elif name in defn.name or defn.name.endswith(f".{name}"):
            # Partial match (e.g., "train" matches "Trainer.train")
            matches.append(defn)
        elif "." in name:
            # Qualified name: try matching the method part
            parts = name.split(".")
            if defn.name.endswith(parts[-1]):
                matches.append(defn)

    if not matches:
        return None

    if len(matches) == 1:
        return matches[0]

    # Multiple matches: use near_line to pick closest
    if near_line:
        # Sort by distance to near_line
        matches.sort(key=lambda d: abs(d.start_line - near_line))

    # Return first (closest) match
    return matches[0]


def resolve_name_with_fallback(
    code: str,
    name: str | None,
    location_type: Literal["function", "class", "method", "module"] | None = None,
    near_line: int | None = None,
    context_lines: int = 3,
) -> ResolvedLocation | None:
    """Resolve name with graceful fallback when AST resolution fails.

    Tries AST resolution first. If that fails (no name provided, parse error,
    or name not found), falls back to using near_line with surrounding context.

    Args:
        code: Python source code
        name: Name to find (can be None for module-level issues)
        location_type: Optional filter by type
        near_line: Line number hint (required for fallback)
        context_lines: Number of context lines around near_line for fallback

    Returns:
        ResolvedLocation if found or fallback worked, None otherwise
    """
    # Try AST resolution first
    if name:
        result = resolve_name(code, name, location_type, near_line)
        if result:
            return result

    # Fallback: use near_line with context
    if near_line:
        code_lines = code.splitlines()
        if 1 <= near_line <= len(code_lines):
            start = max(1, near_line - context_lines)
            end = min(len(code_lines), near_line + context_lines)
            snippet = "\n".join(code_lines[start - 1 : end])
            return ResolvedLocation(
                name=name or "<unknown>",
                location_type=location_type or "module",
                start_line=start,
                end_line=end,
                snippet=snippet,
            )

    return None
