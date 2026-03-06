#!/usr/bin/env python3
"""
Generate project statistics for SciCode-Lint.

Counts lines of code across implementation, patterns, tests, and documentation.
Outputs human-readable Markdown and/or machine-readable JSON.

Examples:
    # Generate stats (creates PROJECT_STATS.md)
    python scripts/project_stats_generate.py

    # With historical snapshots
    python scripts/project_stats_generate.py --history

    # JSON output for CI/automation
    python scripts/project_stats_generate.py --format json

    # Both formats with history
    python scripts/project_stats_generate.py --format both --history

Output files are gitignored - run when you need stats for documentation or reports.
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, cast


def run_command(cmd: str) -> str:
    """Run shell command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()


def count_lines_in_path(path: str, pattern: str = "*.py") -> int:
    """Count total lines in files matching pattern."""
    cmd = f'find {path} -name "{pattern}" -type f -exec wc -l {{}} + 2>/dev/null | tail -1'
    output = run_command(cmd)
    if output and 'total' in output:
        return int(output.split()[0])
    return 0


def count_files(path: str, pattern: str = "*.py") -> int:
    """Count files matching pattern."""
    cmd = f'find {path} -name "{pattern}" -type f 2>/dev/null | wc -l'
    output = run_command(cmd)
    return int(output) if output else 0


def get_pattern_stats() -> dict[str, Any]:
    """Get statistics for pattern files by category."""
    pattern_dir = Path("./patterns")
    categories = {}

    for category_path in sorted(pattern_dir.glob("*/")):
        if category_path.name.startswith('.'):
            continue

        category_name = category_path.name
        num_patterns = count_files(str(category_path), "*.py")
        num_lines = count_lines_in_path(str(category_path), "*.py")

        categories[category_name] = {
            "patterns": num_patterns,
            "lines": num_lines
        }

    return categories


def get_module_stats() -> dict[str, int]:
    """Get line counts for main modules."""
    modules = {}
    src_path = Path("./src/scicode_lint")

    for module_path in sorted(src_path.glob("*/")):
        if module_path.name.startswith('.') or module_path.name == '__pycache__':
            continue

        module_name = module_path.name
        lines = count_lines_in_path(str(module_path), "*.py")
        if lines > 0:
            modules[module_name] = lines

    return modules


def collect_stats() -> dict[str, Any]:
    """Collect all project statistics."""
    stats = {
        "timestamp": datetime.now().isoformat(),
        "python_code": {
            "implementation": {
                "total_lines": count_lines_in_path("./src", "*.py"),
                "total_files": count_files("./src", "*.py"),
                "modules": get_module_stats()
            },
            "patterns": {
                "total_lines": count_lines_in_path("./patterns", "*.py"),
                "total_patterns": count_files("./patterns", "*.py"),
                "categories": get_pattern_stats()
            },
            "tests": {
                "total_lines": count_lines_in_path("./tests", "*.py"),
                "total_files": count_files("./tests", "*.py")
            },
            "evaluations": {
                "total_lines": count_lines_in_path("./evals", "*.py"),
                "total_files": count_files("./evals", "*.py")
            }
        },
        "documentation": {
            "total_lines": count_lines_in_path(".", "*.md"),
            "total_files": count_files(".", "*.md")
        }
    }

    # Calculate totals
    python_code = cast(dict[str, Any], stats["python_code"])
    impl_lines: int = python_code["implementation"]["total_lines"]
    pattern_lines: int = python_code["patterns"]["total_lines"]
    test_lines: int = python_code["tests"]["total_lines"]
    eval_lines: int = python_code["evaluations"]["total_lines"]
    documentation = cast(dict[str, Any], stats["documentation"])
    doc_lines: int = documentation["total_lines"]

    python_total = impl_lines + pattern_lines + test_lines + eval_lines
    stats["totals"] = {
        "python_lines": python_total,
        "documentation_lines": doc_lines,
        "total_lines": python_total + doc_lines,
    }

    return stats


def format_markdown(stats: dict[str, Any]) -> str:
    """Format statistics as Markdown."""
    md = [
        "# SciCode-Lint Project Statistics",
        "",
        f"*Generated: {datetime.fromisoformat(stats['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Summary",
        "",
        f"- **Total Lines:** {stats['totals']['total_lines']:,}",
        f"- **Python Code:** {stats['totals']['python_lines']:,} lines",
        f"- **Documentation:** {stats['totals']['documentation_lines']:,} lines",
        "",
        "## Python Code",
        "",
        "### Implementation (src/)",
        "",
        f"- **Total Lines:** {stats['python_code']['implementation']['total_lines']:,}",
        f"- **Files:** {stats['python_code']['implementation']['total_files']}",
        "",
        "**Breakdown by Module:**",
        ""
    ]

    # Add module breakdown
    for module, lines in sorted(stats['python_code']['implementation']['modules'].items(),
                                 key=lambda x: x[1], reverse=True):
        md.append(f"- {module}/: {lines:,} lines")

    md.extend([
        "",
        "### Pattern Definitions (patterns/)",
        "",
        f"- **Total Lines:** {stats['python_code']['patterns']['total_lines']:,}",
        f"- **Total Patterns:** {stats['python_code']['patterns']['total_patterns']}",
        f"- **Categories:** {len(stats['python_code']['patterns']['categories'])}",
        "",
        "**Breakdown by Category:**",
        ""
    ])

    # Add pattern breakdown
    for category, data in sorted(stats['python_code']['patterns']['categories'].items(),
                                  key=lambda x: x[1]['lines'], reverse=True):
        md.append(f"- **{category}**: {data['patterns']} patterns ({data['lines']:,} lines)")

    md.extend([
        "",
        "### Tests (tests/)",
        "",
        f"- **Total Lines:** {stats['python_code']['tests']['total_lines']:,}",
        f"- **Files:** {stats['python_code']['tests']['total_files']}",
        "",
        "### Evaluations (evals/)",
        "",
        f"- **Total Lines:** {stats['python_code']['evaluations']['total_lines']:,}",
        f"- **Files:** {stats['python_code']['evaluations']['total_files']}",
        "",
        "## Documentation",
        "",
        f"- **Total Lines:** {stats['documentation']['total_lines']:,}",
        f"- **Files:** {stats['documentation']['total_files']} markdown files",
        ""
    ])

    return "\n".join(md)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SciCode-Lint project statistics",
        epilog="""
Examples:
  %(prog)s                              # Generate PROJECT_STATS.md
  %(prog)s --history                     # Save timestamped copy to stats/
  %(prog)s --format json                 # Generate JSON for automation
  %(prog)s --format both --history       # Both formats with history
  %(prog)s --output docs/STATS.md        # Custom output location
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", default="PROJECT_STATS.md",
                       help="Output file path (default: %(default)s)")
    parser.add_argument("--format", choices=["markdown", "json", "both"], default="markdown",
                       help="Output format: markdown, json, or both (default: %(default)s)")
    parser.add_argument("--history", action="store_true",
                       help="Store timestamped copy in stats/ directory")
    args = parser.parse_args()

    print("Collecting project statistics...")
    stats = collect_stats()

    # Save main output
    if args.format in ["markdown", "both"]:
        md_content = format_markdown(stats)
        output_path = Path(args.output)
        output_path.write_text(md_content)
        print(f"✓ Markdown stats written to {output_path}")

    if args.format in ["json", "both"]:
        json_path = Path(args.output).with_suffix('.json')
        json_path.write_text(json.dumps(stats, indent=2))
        print(f"✓ JSON stats written to {json_path}")

    # Save historical copy
    if args.history:
        stats_dir = Path("stats")
        stats_dir.mkdir(exist_ok=True)

        timestamp = datetime.fromisoformat(stats['timestamp']).strftime('%Y%m%d_%H%M%S')

        if args.format in ["markdown", "both"]:
            history_md = stats_dir / f"stats_{timestamp}.md"
            history_md.write_text(format_markdown(stats))
            print(f"✓ Historical markdown saved to {history_md}")

        if args.format in ["json", "both"]:
            history_json = stats_dir / f"stats_{timestamp}.json"
            history_json.write_text(json.dumps(stats, indent=2))
            print(f"✓ Historical JSON saved to {history_json}")

    # Print summary
    print("\nProject Summary:")
    print(f"  Total lines: {stats['totals']['total_lines']:,}")
    print(f"  Python: {stats['totals']['python_lines']:,}")
    print(f"  Patterns: {stats['python_code']['patterns']['total_patterns']} files")
    print(f"  Documentation: {stats['totals']['documentation_lines']:,}")


if __name__ == "__main__":
    main()
