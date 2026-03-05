#!/usr/bin/env python3
"""
Validate pattern.toml files and directory structure.

Checks for:
- Valid TOML syntax and structure
- Required directories (positive, negative, context_dependent)
- Test file references exist
- ID matches directory name
- Category matches parent directory
"""

import argparse
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from scicode_lint.detectors.pattern_loader import PatternLoader


class PatternValidator:
    """Validate pattern directories and TOML files."""

    def __init__(self, patterns_dir: Path):
        """
        Initialize validator.

        Args:
            patterns_dir: Root patterns directory containing categories
        """
        self.patterns_dir = Path(patterns_dir)
        self.loader = PatternLoader(patterns_dir)
        self.errors: dict[str, list[str]] = {}

    def validate_pattern(self, pattern_dir: Path) -> tuple[bool, list[str]]:
        """
        Validate a single pattern directory.

        Args:
            pattern_dir: Path to pattern directory

        Returns:
            tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check pattern.toml exists
        toml_file = pattern_dir / "pattern.toml"
        if not toml_file.exists():
            return False, ["Missing pattern.toml"]

        # Load and validate TOML structure
        try:
            pattern = self.loader.load_pattern_toml(pattern_dir)
        except ValidationError as e:
            return False, [f"TOML validation failed: {e}"]
        except Exception as e:
            return False, [f"Failed to parse TOML: {e}"]

        # Validate required directories exist
        for dirname in ["positive", "negative", "context_dependent"]:
            dir_path = pattern_dir / dirname
            if not dir_path.exists():
                errors.append(f"Missing {dirname}/ directory")

        # Validate test file references (skip placeholder files)
        for test in pattern.tests.positive:
            # Skip validation for placeholder test files
            if test.file == "positive/example.py":
                continue
            test_path = pattern_dir / test.file
            if not test_path.exists():
                errors.append(f"Positive test file not found: {test.file}")

        for test in pattern.tests.negative:
            # Skip validation for placeholder test files
            if test.file == "negative/example.py":
                continue
            test_path = pattern_dir / test.file
            if not test_path.exists():
                errors.append(f"Negative test file not found: {test.file}")

        for test in pattern.tests.context_dependent:
            # Skip validation for placeholder test files
            if test.file == "context_dependent/example.py":
                continue
            test_path = pattern_dir / test.file
            if not test_path.exists():
                errors.append(f"Context-dependent test file not found: {test.file}")

        # Validate ID matches directory name
        if pattern.meta.id not in pattern_dir.name:
            errors.append(
                f"ID '{pattern.meta.id}' doesn't match directory name '{pattern_dir.name}'"
            )

        # Validate category matches parent directory
        parent_category = pattern_dir.parent.name
        if pattern.meta.category != parent_category:
            errors.append(
                f"Category '{pattern.meta.category}' doesn't match "
                f"parent directory '{parent_category}'"
            )

        # Check severity is valid
        if pattern.meta.severity not in ["critical", "high", "medium"]:
            errors.append(f"Invalid severity: {pattern.meta.severity}")

        # Warn if no test cases defined
        if (
            not pattern.tests.positive
            and not pattern.tests.negative
            and not pattern.tests.context_dependent
        ):
            errors.append("Warning: No test cases defined")

        return len(errors) == 0, errors

    def validate_all(self, verbose: bool = True) -> dict[str, Any]:
        """
        Validate all patterns in the patterns directory.

        Args:
            verbose: Print progress for each pattern

        Returns:
            dictionary with validation results
        """
        results = {"total": 0, "valid": 0, "invalid": 0, "errors": {}}

        if not self.patterns_dir.exists():
            print(f"ERROR: Patterns directory not found: {self.patterns_dir}")
            return results

        # Iterate through categories
        for category_dir in sorted(self.patterns_dir.iterdir()):
            if not category_dir.is_dir() or category_dir.name.startswith("_"):
                continue

            if verbose:
                print(f"\n{category_dir.name}/")
                print("─" * 60)

            # Iterate through patterns in category
            for pattern_dir in sorted(category_dir.iterdir()):
                if not pattern_dir.is_dir():
                    continue

                results["total"] += 1
                is_valid, errors = self.validate_pattern(pattern_dir)

                if is_valid:
                    results["valid"] += 1
                    if verbose:
                        print(f"  ✓ {pattern_dir.name}")
                else:
                    results["invalid"] += 1
                    results["errors"][str(pattern_dir)] = errors
                    if verbose:
                        print(f"  ✗ {pattern_dir.name}")
                        for err in errors:
                            print(f"      - {err}")

        return results

    def print_summary(self, results: dict[str, Any]):
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total patterns:    {results['total']}")
        print(f"Valid:             {results['valid']}")
        print(f"Invalid:           {results['invalid']}")
        print("=" * 60)

        if results["invalid"] > 0:
            print(f"\n{results['invalid']} pattern(s) have errors. See details above.")
            return 1
        else:
            print("\n✓ All patterns are valid!")
            return 0


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate pattern.toml files and directory structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate all patterns
  python -m scicode_lint.tools.validate_pattern

  # Validate specific pattern
  python -m scicode_lint.tools.validate_pattern \\
    --pattern patterns/ai-training/ml-001-scaler-leakage

  # Quiet mode (only show errors)
  python -m scicode_lint.tools.validate_pattern --quiet
""",
    )

    parser.add_argument(
        "--patterns-dir",
        type=Path,
        default=Path("patterns"),
        help="Path to patterns directory (default: patterns)",
    )
    parser.add_argument(
        "--pattern",
        type=Path,
        help="Validate a specific pattern directory",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show summary and errors",
    )

    args = parser.parse_args()

    # Validate specific pattern
    if args.pattern:
        validator = PatternValidator(args.patterns_dir)
        is_valid, errors = validator.validate_pattern(args.pattern)

        if is_valid:
            print(f"✓ {args.pattern.name} is valid")
            return 0
        else:
            print(f"✗ {args.pattern.name} has errors:")
            for err in errors:
                print(f"  - {err}")
            return 1

    # Validate all patterns
    validator = PatternValidator(args.patterns_dir)
    results = validator.validate_all(verbose=not args.quiet)
    validator.print_summary(results)

    return 0 if results["invalid"] == 0 else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
