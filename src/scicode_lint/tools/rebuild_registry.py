#!/usr/bin/env python3
"""
Build pattern registry (_registry.toml) from pattern directories.

The registry provides fast lookup of pattern metadata without parsing
all pattern.toml files.
"""

import argparse
import datetime
from pathlib import Path
from typing import Any

from scicode_lint.detectors.pattern_loader import PatternLoader


class RegistryBuilder:
    """Build pattern registry from TOML patterns."""

    def __init__(self, patterns_dir: Path):
        """
        Initialize registry builder.

        Args:
            patterns_dir: Root patterns directory
        """
        self.patterns_dir = Path(patterns_dir)
        self.loader = PatternLoader(patterns_dir)

    def _find_pattern_dir(self, pattern_id: str) -> Path | None:
        """
        Find pattern directory by ID.

        Args:
            pattern_id: Pattern ID (e.g., "ml-001")

        Returns:
            Path to pattern directory, or None if not found
        """
        for category_dir in self.patterns_dir.iterdir():
            if not category_dir.is_dir() or category_dir.name.startswith("_"):
                continue

            for pattern_dir in category_dir.iterdir():
                if not pattern_dir.is_dir():
                    continue

                # Check if pattern ID is in directory name
                if pattern_id in pattern_dir.name:
                    return pattern_dir

        return None

    def build_registry(self) -> str:
        """
        Build registry content from all patterns.

        Returns:
            Registry content as TOML string
        """
        patterns = self.loader.load_all_patterns()

        lines = [
            "# Pattern Registry - Auto-generated",
            f"# Generated: {datetime.datetime.now().isoformat()}",
            f"# Total patterns: {len(patterns)}",
            "",
            'version = "1.0"',
            f"total_patterns = {len(patterns)}",
            "",
        ]

        # Sort patterns by ID for consistent ordering
        for pattern in sorted(patterns, key=lambda p: p.meta.id):
            pattern_dir = self._find_pattern_dir(pattern.meta.id)
            if not pattern_dir:
                continue

            rel_path = pattern_dir.relative_to(self.patterns_dir)

            lines.extend(
                [
                    f'[patterns."{pattern.meta.id}"]',
                    f'path = "{rel_path}"',
                    f'name = "{pattern.meta.name}"',
                    f'category = "{pattern.meta.category}"',
                    f'severity = "{pattern.meta.severity}"',
                    f'version = "{pattern.meta.version}"',
                    "",
                ]
            )

        return "\n".join(lines)

    def write_registry(self, output_path: Path | None = None) -> None:
        """
        Write registry to _registry.toml.

        Args:
            output_path: Optional custom output path
        """
        if output_path is None:
            output_path = self.patterns_dir / "_registry.toml"

        content = self.build_registry()
        output_path.write_text(content)
        print(f"✓ Registry written to: {output_path}")
        print(f"  Total patterns: {len(self.loader.load_all_patterns())}")

    def get_stats(self) -> dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            dictionary with pattern counts by category
        """
        patterns = self.loader.load_all_patterns()
        stats: dict[str, Any] = {
            "total": len(patterns),
            "by_category": {},
            "by_severity": {},
        }

        by_category: dict[str, int] = stats["by_category"]
        by_severity: dict[str, int] = stats["by_severity"]

        for pattern in patterns:
            # Count by category
            cat = pattern.meta.category
            by_category[cat] = by_category.get(cat, 0) + 1

            # Count by severity
            sev = pattern.meta.severity
            by_severity[sev] = by_severity.get(sev, 0) + 1

        return stats

    def print_stats(self) -> None:
        """Print registry statistics."""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("PATTERN REGISTRY STATISTICS")
        print("=" * 60)
        print(f"Total patterns: {stats['total']}")
        print("\nBy category:")
        for cat, count in sorted(stats["by_category"].items()):
            print(f"  {cat:30} {count:3}")

        print("\nBy severity:")
        for sev, count in sorted(stats["by_severity"].items()):
            print(f"  {sev:30} {count:3}")
        print("=" * 60)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build pattern registry from TOML patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build registry from default patterns directory
  python -m scicode_lint.tools.rebuild_registry

  # Build from custom directory
  python -m scicode_lint.tools.rebuild_registry \\
    --patterns-dir patterns

  # Show statistics only
  python -m scicode_lint.tools.rebuild_registry --stats-only
""",
    )

    parser.add_argument(
        "--patterns-dir",
        type=Path,
        default=Path("patterns"),
        help="Path to patterns directory (default: patterns)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Custom output path (default: <patterns-dir>/_registry.toml)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print statistics without writing registry",
    )

    args = parser.parse_args()

    if not args.patterns_dir.exists():
        print(f"ERROR: Patterns directory not found: {args.patterns_dir}")
        return 1

    builder = RegistryBuilder(args.patterns_dir)

    if args.stats_only:
        builder.print_stats()
    else:
        builder.write_registry(args.output)
        builder.print_stats()

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
