#!/usr/bin/env python3
"""Comprehensive pattern validation - deterministic quality checks.

Usage:
    python scripts/validate_pattern_tests.py           # Check all patterns
    python scripts/validate_pattern_tests.py --fix     # Auto-fix what's possible
    python scripts/validate_pattern_tests.py ml-002    # Check specific pattern
    python scripts/validate_pattern_tests.py --strict  # Fail on warnings too

Checks performed:
1. TOML/file sync - every test file has TOML entry and vice versa
2. Schema validation - pattern.toml matches Pydantic model
3. Data leakage - no BUG/CORRECT/WRONG hints in test files
4. Test file count - minimum 3 positive, 3 negative (warning)
5. TODO markers - no unfinished placeholders in TOML
6. Detection question format - ends with YES/NO conditions
7. Test file syntax - all .py files are valid Python
8. Empty fields - required fields have content
9. Test file diversity - detect copy-paste (AST similarity)
"""

import argparse
import ast
import hashlib
import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import tomllib

# Data leakage patterns to detect in test files
DATA_LEAKAGE_PATTERNS = [
    (r"#\s*BUG:", "BUG marker comment"),
    (r"#\s*CORRECT:", "CORRECT marker comment"),
    (r"#\s*WRONG", "WRONG marker comment"),
    (r"#\s*FIXME:", "FIXME marker comment"),
    (r"#\s*This is (wrong|incorrect|buggy|the bug)", "Hint comment"),
    (r"#\s*This (leaks|causes|creates)", "Hint comment explaining issue"),
    (r'""".*?(bug|issue|problem|incorrect|wrong).*?"""', "Docstring with hint"),
]

# Required YES/NO pattern at end of detection question
YES_NO_PATTERN = re.compile(r"YES\s*=.*\n\s*NO\s*=", re.IGNORECASE | re.MULTILINE)


@dataclass
class ValidationIssue:
    """A single validation issue."""

    level: str  # "error" or "warning"
    check: str  # which check found this
    message: str
    file: str = ""


@dataclass
class ValidationResult:
    """Result of validating a pattern."""

    pattern_id: str
    category: str
    issues: list[ValidationIssue] = field(default_factory=list)
    fixed: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.level == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.level == "warning" for i in self.issues)


def get_test_files_on_disk(pattern_dir: Path) -> dict[str, set[str]]:
    """Get all test files on disk for a pattern."""
    result: dict[str, set[str]] = {"positive": set(), "negative": set(), "context_dependent": set()}

    for test_type in result.keys():
        test_dir = pattern_dir / f"test_{test_type}"
        if test_dir.exists():
            result[test_type] = {
                f.name for f in test_dir.glob("*.py") if not f.name.startswith("_")
            }

    return result


def get_test_files_in_toml(toml_data: dict[str, Any]) -> dict[str, set[str]]:
    """Get all test files referenced in pattern.toml."""
    result: dict[str, set[str]] = {"positive": set(), "negative": set(), "context_dependent": set()}

    tests = toml_data.get("tests", {})
    for test_type in result.keys():
        for entry in tests.get(test_type, []):
            file_path = entry.get("file", "")
            result[test_type].add(Path(file_path).name)

    return result


# =============================================================================
# CHECK 1: TOML/File Sync
# =============================================================================


def check_toml_file_sync(
    pattern_dir: Path, toml_data: dict[str, Any], result: ValidationResult
) -> dict[str, dict[str, list[str]]]:
    """Check that test files on disk match TOML entries."""
    on_disk = get_test_files_on_disk(pattern_dir)
    in_toml = get_test_files_in_toml(toml_data)

    issues: dict[str, dict[str, list[str]]] = {"missing_in_toml": {}, "missing_on_disk": {}}

    for test_type in ["positive", "negative", "context_dependent"]:
        missing_in_toml = on_disk[test_type] - in_toml[test_type]
        missing_on_disk = in_toml[test_type] - on_disk[test_type]

        if missing_in_toml:
            issues["missing_in_toml"][test_type] = sorted(missing_in_toml)
            for f in missing_in_toml:
                result.issues.append(
                    ValidationIssue(
                        "error",
                        "toml_sync",
                        f"File on disk not in TOML: test_{test_type}/{f}",
                    )
                )

        if missing_on_disk:
            issues["missing_on_disk"][test_type] = sorted(missing_on_disk)
            for f in missing_on_disk:
                result.issues.append(
                    ValidationIssue(
                        "error",
                        "toml_sync",
                        f"TOML entry but file missing: test_{test_type}/{f}",
                    )
                )

    return issues


# =============================================================================
# CHECK 2: Schema Validation
# =============================================================================


def check_schema(pattern_dir: Path, toml_data: dict[str, Any], result: ValidationResult) -> bool:
    """Validate pattern.toml against Pydantic schema."""
    try:
        from scicode_lint.detectors.pattern_models import PatternTOML

        PatternTOML(**toml_data)
        return True
    except Exception as e:
        result.issues.append(ValidationIssue("error", "schema", f"Schema validation failed: {e}"))
        return False


# =============================================================================
# CHECK 3: Data Leakage Detection
# =============================================================================


def check_data_leakage(pattern_dir: Path, result: ValidationResult) -> None:
    """Check test files for hint comments that leak answers."""
    for test_type in ["positive", "negative", "context_dependent"]:
        test_dir = pattern_dir / f"test_{test_type}"
        if not test_dir.exists():
            continue

        for py_file in test_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                content = py_file.read_text()
            except Exception:
                continue

            for pattern, desc in DATA_LEAKAGE_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE):
                    result.issues.append(
                        ValidationIssue(
                            "error",
                            "data_leakage",
                            f"{desc} found",
                            file=f"test_{test_type}/{py_file.name}",
                        )
                    )


# =============================================================================
# CHECK 4: Test File Count
# =============================================================================


def check_test_count(pattern_dir: Path, result: ValidationResult) -> None:
    """Check minimum test file counts (3 positive, 3 negative recommended)."""
    on_disk = get_test_files_on_disk(pattern_dir)

    pos_count = len(on_disk["positive"])
    neg_count = len(on_disk["negative"])

    if pos_count < 3:
        result.issues.append(
            ValidationIssue(
                "warning",
                "test_count",
                f"Only {pos_count} positive tests (recommend 3+)",
            )
        )

    if neg_count < 3:
        result.issues.append(
            ValidationIssue(
                "warning",
                "test_count",
                f"Only {neg_count} negative tests (recommend 3+)",
            )
        )


# =============================================================================
# CHECK 5: TODO Markers
# =============================================================================


def check_todo_markers(toml_path: Path, result: ValidationResult) -> None:
    """Check for unfinished TODO placeholders in TOML."""
    content = toml_path.read_text()

    # Find TODO patterns
    todo_matches = re.findall(r"TODO[:\s].*", content, re.IGNORECASE)
    for match in todo_matches:
        result.issues.append(
            ValidationIssue("error", "todo_marker", f"Unfinished: {match.strip()}")
        )


# =============================================================================
# CHECK 6: Detection Question Format
# =============================================================================


def check_detection_question(toml_data: dict[str, Any], result: ValidationResult) -> None:
    """Check detection question ends with YES/NO conditions."""
    detection = toml_data.get("detection", {})
    question = detection.get("question", "")

    if not question:
        result.issues.append(
            ValidationIssue("error", "detection_format", "Missing detection question")
        )
        return

    if not YES_NO_PATTERN.search(question):
        result.issues.append(
            ValidationIssue(
                "warning",
                "detection_format",
                "Detection question should end with 'YES = ...' and 'NO = ...' conditions",
            )
        )


# =============================================================================
# CHECK 7: Test File Syntax
# =============================================================================


def check_test_syntax(pattern_dir: Path, result: ValidationResult) -> None:
    """Check all test files are valid Python."""
    for test_type in ["positive", "negative", "context_dependent"]:
        test_dir = pattern_dir / f"test_{test_type}"
        if not test_dir.exists():
            continue

        for py_file in test_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                source = py_file.read_text()
                ast.parse(source)
            except SyntaxError as e:
                result.issues.append(
                    ValidationIssue(
                        "error",
                        "syntax",
                        f"Invalid Python syntax: {e}",
                        file=f"test_{test_type}/{py_file.name}",
                    )
                )


# =============================================================================
# CHECK 8: Empty Fields
# =============================================================================


def check_empty_fields(toml_data: dict[str, Any], result: ValidationResult) -> None:
    """Check required fields have content."""
    # Check meta fields
    meta = toml_data.get("meta", {})
    for field_name in ["description", "explanation"]:
        value = meta.get(field_name, "")
        if not value or not value.strip():
            result.issues.append(
                ValidationIssue("error", "empty_field", f"meta.{field_name} is empty")
            )

    # Check detection fields
    detection = toml_data.get("detection", {})
    for field_name in ["question", "warning_message"]:
        value = detection.get(field_name, "")
        if not value or not value.strip():
            result.issues.append(
                ValidationIssue("error", "empty_field", f"detection.{field_name} is empty")
            )

    # Check test entries
    tests = toml_data.get("tests", {})
    for test_type in ["positive", "negative", "context_dependent"]:
        for i, entry in enumerate(tests.get(test_type, [])):
            if not entry.get("description", "").strip():
                result.issues.append(
                    ValidationIssue(
                        "error",
                        "empty_field",
                        f"tests.{test_type}[{i}].description is empty",
                    )
                )
            if test_type == "positive" and not entry.get("expected_issue", "").strip():
                result.issues.append(
                    ValidationIssue(
                        "error",
                        "empty_field",
                        f"tests.{test_type}[{i}].expected_issue is empty",
                    )
                )


# =============================================================================
# CHECK 9: Test File Diversity (AST similarity)
# =============================================================================


def get_ast_hash(file_path: Path) -> str | None:
    """Get a normalized AST hash for a Python file."""
    try:
        source = file_path.read_text()
        tree = ast.parse(source)

        # Normalize: remove docstrings, normalize names
        for node in ast.walk(tree):
            # Remove docstrings
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    node.body = node.body[1:]

        # Convert to string and hash
        ast_str = ast.dump(tree, annotate_fields=False)
        return hashlib.md5(ast_str.encode()).hexdigest()[:8]
    except Exception:
        return None


def check_test_diversity(pattern_dir: Path, result: ValidationResult) -> None:
    """Check test files aren't too similar (copy-paste detection)."""
    for test_type in ["positive", "negative"]:
        test_dir = pattern_dir / f"test_{test_type}"
        if not test_dir.exists():
            continue

        files = [f for f in test_dir.glob("*.py") if not f.name.startswith("_")]
        if len(files) < 2:
            continue

        # Get AST hashes
        hashes: dict[str, list[str]] = {}
        for f in files:
            h = get_ast_hash(f)
            if h:
                hashes.setdefault(h, []).append(f.name)

        # Check for duplicates
        for h, filenames in hashes.items():
            if len(filenames) > 1:
                result.issues.append(
                    ValidationIssue(
                        "warning",
                        "diversity",
                        f"Nearly identical files in test_{test_type}: {', '.join(filenames)}",
                    )
                )


# =============================================================================
# FIX FUNCTIONS
# =============================================================================


def find_best_match(wrong_name: str, available_names: set[str]) -> str | None:
    """Find the best matching filename from available names."""
    best_match = None
    best_ratio = 0.6

    for name in available_names:
        ratio = SequenceMatcher(None, wrong_name, name).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = name

    return best_match


def extract_first_function_or_class(file_path: Path) -> tuple[str, str, str]:
    """Extract first function/class name and a code snippet."""
    try:
        source = file_path.read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                snippet = ""
                for stmt in node.body:
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                        continue
                    if hasattr(stmt, "lineno"):
                        lines = source.split("\n")
                        if stmt.lineno <= len(lines):
                            snippet = lines[stmt.lineno - 1].strip()
                            break
                return "function", node.name, snippet or f"def {node.name}"
            elif isinstance(node, ast.ClassDef):
                return "class", node.name, f"class {node.name}"

        return "module", file_path.stem, ""
    except Exception:
        return "module", file_path.stem, ""


def fix_toml_sync(
    pattern_dir: Path, sync_issues: dict[str, dict[str, list[str]]], result: ValidationResult
) -> None:
    """Fix TOML/file sync issues."""
    toml_path = pattern_dir / "pattern.toml"
    content = toml_path.read_text()

    missing_on_disk = sync_issues.get("missing_on_disk", {})
    missing_in_toml = sync_issues.get("missing_in_toml", {})

    # Try to rename mismatched entries
    for test_type in ["positive", "negative", "context_dependent"]:
        wrong_names = list(missing_on_disk.get(test_type, []))
        available_names = set(missing_in_toml.get(test_type, []))

        for wrong_name in wrong_names:
            best_match = find_best_match(wrong_name, available_names)
            if best_match:
                old_path = f"test_{test_type}/{wrong_name}"
                new_path = f"test_{test_type}/{best_match}"
                content = content.replace(f'"{old_path}"', f'"{new_path}"')
                result.fixed.append(f"Renamed {wrong_name} -> {best_match}")
                available_names.discard(best_match)
            else:
                # Remove orphaned entry
                lines = content.split("\n")
                new_lines: list[str] = []
                skip_block = False
                old_path = f"test_{test_type}/{wrong_name}"

                for line in lines:
                    if f'"{old_path}"' in line:
                        j = len(new_lines) - 1
                        while j >= 0 and not new_lines[j].strip().startswith("[[tests."):
                            j -= 1
                        new_lines = new_lines[:j]
                        skip_block = True
                        result.fixed.append(f"Removed orphaned entry: {wrong_name}")
                    elif skip_block:
                        starts_new = line.strip().startswith(("[[tests.", "[tests."))
                        if starts_new:
                            skip_block = False
                            new_lines.append(line)
                    else:
                        new_lines.append(line)
                content = "\n".join(new_lines)

    # Re-read to get updated missing_in_toml
    toml_path.write_text(content)

    # Re-check what's still missing
    with open(toml_path, "rb") as f:
        updated_toml = tomllib.load(f)
    on_disk = get_test_files_on_disk(pattern_dir)
    in_toml = get_test_files_in_toml(updated_toml)

    # Add missing entries
    new_entries = []
    for test_type in ["positive", "negative", "context_dependent"]:
        still_missing = on_disk[test_type] - in_toml[test_type]
        for filename in sorted(still_missing):
            if test_type == "positive":
                file_path = pattern_dir / "test_positive" / filename
                loc_type, loc_name, snippet = extract_first_function_or_class(file_path)
                snippet = snippet.replace('"', '\\"')
                new_entries.append(f'''
[[tests.positive]]
file = "test_positive/{filename}"
description = "TODO: Add description"
expected_issue = "TODO: Add expected issue"
min_confidence = 0.85

[tests.positive.expected_location]
type = "{loc_type}"
name = "{loc_name}"
snippet = "{snippet}"
''')
            elif test_type == "negative":
                new_entries.append(f"""
[[tests.negative]]
file = "test_negative/{filename}"
description = "TODO: Add description"
""")
            else:
                new_entries.append(f"""
[[tests.context_dependent]]
file = "test_context_dependent/{filename}"
description = "TODO: Add description"
context_notes = "TODO: Add context notes"
allow_detection = true
allow_skip = true
""")
            result.fixed.append(f"Added entry for {test_type}/{filename}")

    if new_entries:
        content = toml_path.read_text()
        with open(toml_path, "a") as f:
            f.write("\n# === Auto-generated entries (review and update) ===")
            for entry in new_entries:
                f.write(entry)


# =============================================================================
# MAIN VALIDATION
# =============================================================================


def validate_pattern(pattern_dir: Path, fix: bool = False) -> ValidationResult:
    """Run all validation checks on a pattern."""
    result = ValidationResult(
        pattern_id=pattern_dir.name,
        category=pattern_dir.parent.name,
    )

    toml_path = pattern_dir / "pattern.toml"
    if not toml_path.exists():
        result.issues.append(ValidationIssue("error", "missing", "pattern.toml not found"))
        return result

    # Load TOML
    try:
        with open(toml_path, "rb") as f:
            toml_data = tomllib.load(f)
    except Exception as e:
        result.issues.append(ValidationIssue("error", "toml_parse", f"Failed to parse TOML: {e}"))
        return result

    # Run checks
    sync_issues = check_toml_file_sync(pattern_dir, toml_data, result)
    check_schema(pattern_dir, toml_data, result)
    check_data_leakage(pattern_dir, result)
    check_test_count(pattern_dir, result)
    check_todo_markers(toml_path, result)
    check_detection_question(toml_data, result)
    check_test_syntax(pattern_dir, result)
    check_empty_fields(toml_data, result)
    check_test_diversity(pattern_dir, result)

    # Apply fixes if requested
    if fix and (sync_issues.get("missing_in_toml") or sync_issues.get("missing_on_disk")):
        fix_toml_sync(pattern_dir, sync_issues, result)

    return result


def find_all_patterns(patterns_dir: Path) -> list[Path]:
    """Find all pattern directories."""
    patterns = []
    for category in patterns_dir.iterdir():
        if not category.is_dir() or category.name.startswith((".", "_")):
            continue
        for pattern_dir in category.iterdir():
            if not pattern_dir.is_dir() or pattern_dir.name.startswith((".", "_")):
                continue
            if (pattern_dir / "pattern.toml").exists():
                patterns.append(pattern_dir)
    return sorted(patterns)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pattern", nargs="?", help="Specific pattern ID (e.g., ml-002)")
    parser.add_argument("--fix", action="store_true", help="Auto-fix what's possible")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only show issues")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    args = parser.parse_args()

    patterns_dir = Path("patterns")
    if not patterns_dir.exists():
        print("Error: patterns/ directory not found", file=sys.stderr)
        return 1

    # Find patterns
    if args.pattern:
        matches = list(patterns_dir.glob(f"*/{args.pattern}"))
        if not matches:
            print(f"Error: Pattern '{args.pattern}' not found", file=sys.stderr)
            return 1
        patterns = matches
    else:
        patterns = find_all_patterns(patterns_dir)

    # Validate
    error_count = 0
    warning_count = 0
    fixed_count = 0

    for pattern_dir in patterns:
        result = validate_pattern(pattern_dir, fix=args.fix)

        has_errors = result.has_errors
        has_warnings = result.has_warnings

        if has_errors:
            error_count += 1
        if has_warnings:
            warning_count += 1
        if result.fixed:
            fixed_count += 1

        # Print results
        if result.issues or result.fixed:
            prefix = "❌" if has_errors else ("⚠" if has_warnings else "✓")
            print(f"\n{prefix} {result.category}/{result.pattern_id}")

            for issue in result.issues:
                icon = "✗" if issue.level == "error" else "⚡"
                file_info = f" [{issue.file}]" if issue.file else ""
                print(f"   {icon} [{issue.check}]{file_info} {issue.message}")

            for fix_msg in result.fixed:
                print(f"   ✓ Fixed: {fix_msg}")

        elif not args.quiet:
            print(f"✓ {result.category}/{result.pattern_id}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Checked {len(patterns)} patterns")
    print(f"  Errors: {error_count}")
    print(f"  Warnings: {warning_count}")
    if args.fix:
        print(f"  Fixed: {fixed_count}")

    if args.strict:
        return 1 if (error_count > 0 or warning_count > 0) else 0
    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
