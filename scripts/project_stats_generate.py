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
import re
import subprocess
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, cast


def run_command(cmd: str) -> str:
    """Run shell command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()


# Directories to exclude from all counts (external/downloaded/generated content)
EXCLUDE_DIRS = [
    "real_world_demo/cloned_repos",
    "real_world_demo/collected_code",
    "pattern_verification/semantic/reports",
    "pattern_verification/deterministic/doc_cache",
]


def _build_exclude_args() -> str:
    """Build find command exclusion arguments."""
    excludes = []
    for d in EXCLUDE_DIRS:
        excludes.append(f'-path "./{d}" -prune -o')
    return " ".join(excludes)


def count_lines_in_path(path: str, pattern: str = "*.py") -> int:
    """Count total lines in files matching pattern."""
    exclude_args = _build_exclude_args()
    cmd = (
        f'find {path} {exclude_args} -name "{pattern}" -type f -print '
        f"-exec wc -l {{}} + 2>/dev/null | tail -1"
    )
    output = run_command(cmd)
    if output and "total" in output:
        return int(output.split()[0])
    return 0


def count_files(path: str, pattern: str = "*.py") -> int:
    """Count files matching pattern."""
    exclude_args = _build_exclude_args()
    cmd = f'find {path} {exclude_args} -name "{pattern}" -type f -print 2>/dev/null | wc -l'
    output = run_command(cmd)
    return int(output) if output else 0


def count_pattern_dirs(category_path: Path) -> int:
    """Count pattern directories (e.g., pt-001-xxx) in a category."""
    count = 0
    for item in category_path.iterdir():
        # Pattern dirs have naming convention like pt-001-*, ml-001-*, etc.
        if item.is_dir() and not item.name.startswith("."):
            count += 1
    return count


def get_pattern_stats() -> dict[str, Any]:
    """Get statistics for pattern files by category."""
    pattern_dir = Path("./patterns")
    categories = {}

    for category_path in sorted(pattern_dir.glob("*/")):
        if category_path.name.startswith("."):
            continue

        category_name = category_path.name
        num_patterns = count_pattern_dirs(category_path)
        num_test_files = count_files(str(category_path), "*.py")
        num_lines = count_lines_in_path(str(category_path), "*.py")

        categories[category_name] = {
            "patterns": num_patterns,
            "test_files": num_test_files,
            "lines": num_lines,
        }

    return categories


def get_module_stats() -> dict[str, int]:
    """Get line counts for main modules."""
    modules = {}
    src_path = Path("./src/scicode_lint")

    for module_path in sorted(src_path.glob("*/")):
        if module_path.name.startswith(".") or module_path.name == "__pycache__":
            continue

        module_name = module_path.name
        lines = count_lines_in_path(str(module_path), "*.py")
        if lines > 0:
            modules[module_name] = lines

    return modules


def get_git_stats() -> dict[str, Any]:
    """Get git repository statistics."""
    # Total commits
    total_commits = run_command("git log --oneline 2>/dev/null | wc -l")

    # Branch count
    branch_count = run_command("git branch -a 2>/dev/null | wc -l")

    # First commit date
    first_commit = run_command("git log --reverse --format='%ci' 2>/dev/null | head -1")

    # Latest commit date
    latest_commit = run_command("git log --format='%ci' 2>/dev/null | head -1")

    # Calculate project age in days
    age_days = 0
    if first_commit:
        try:
            first_date = datetime.fromisoformat(first_commit.split()[0])
            age_days = (datetime.now() - first_date).days
        except (ValueError, IndexError):
            pass

    return {
        "total_commits": int(total_commits) if total_commits else 0,
        "branch_count": int(branch_count) if branch_count else 0,
        "first_commit": first_commit or "N/A",
        "latest_commit": latest_commit or "N/A",
        "age_days": age_days,
    }


def get_gpu_info() -> dict[str, Any] | None:
    """Detect GPU information using nvidia-smi."""
    # Check if nvidia-smi is available
    check = run_command("which nvidia-smi 2>/dev/null")
    if not check:
        return None

    # Get GPU name
    gpu_name = run_command(
        "nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1"
    )
    if not gpu_name:
        return None

    # Get total VRAM in MB, convert to GB
    vram_mb = run_command(
        "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1"
    )
    vram_gb = round(int(vram_mb) / 1024, 1) if vram_mb else 0

    # Get driver version
    driver = run_command(
        "nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1"
    )

    # Get CUDA version from nvidia-smi header
    header = run_command("nvidia-smi 2>/dev/null | grep 'CUDA Version'")
    cuda = None
    if header and "CUDA Version:" in header:
        cuda = header.split("CUDA Version:")[1].strip().split()[0]

    # Count GPUs
    gpu_count = run_command("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l")

    return {
        "name": gpu_name,
        "vram_gb": vram_gb,
        "count": int(gpu_count) if gpu_count else 1,
        "driver": driver or "N/A",
        "cuda": cuda or "N/A",
    }


def get_tech_stack() -> dict[str, Any]:
    """Extract tech stack from pyproject.toml."""
    pyproject_path = Path("./pyproject.toml")
    if not pyproject_path.exists():
        return {}

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    project = data.get("project", {})
    optional_deps = project.get("optional-dependencies", {})

    # Extract Python version
    python_version = project.get("requires-python", "unknown")

    # Extract build system
    build_backend = data.get("build-system", {}).get("build-backend", "unknown")

    # Core dependencies
    core_deps = project.get("dependencies", [])

    # Dev tools from dev dependencies
    dev_deps = optional_deps.get("dev", [])

    # Categorize dev tools
    linting = [d for d in dev_deps if "ruff" in d.lower()]
    type_checking = [d for d in dev_deps if "mypy" in d.lower()]
    testing = [d for d in dev_deps if "pytest" in d.lower()]
    security = [d for d in dev_deps if any(s in d.lower() for s in ["bandit", "safety", "audit"])]

    # LLM/ML dependencies
    vllm_deps = optional_deps.get("vllm-server", [])

    # Extract version from dependency string (e.g., "ruff>=0.15.5" -> "ruff >=0.15.5")
    def parse_dep(dep: str) -> str:
        match = re.match(r"([a-zA-Z0-9_-]+)(.*)", dep)
        if match:
            name, version = match.groups()
            return f"{name}{version}" if version else name
        return dep

    return {
        "python_version": python_version,
        "build_backend": build_backend,
        "core_dependencies": [parse_dep(d) for d in core_deps],
        "dev_tools": {
            "linting": [parse_dep(d) for d in linting],
            "type_checking": [parse_dep(d) for d in type_checking],
            "testing": [parse_dep(d) for d in testing],
            "security": [parse_dep(d) for d in security],
        },
        "llm_server": [parse_dep(d) for d in vllm_deps],
    }


def collect_stats() -> dict[str, Any]:
    """Collect all project statistics."""
    stats = {
        "timestamp": datetime.now().isoformat(),
        "git": get_git_stats(),
        "gpu": get_gpu_info(),
        "tech_stack": get_tech_stack(),
        "python_code": {
            "implementation": {
                "total_lines": count_lines_in_path("./src", "*.py"),
                "total_files": count_files("./src", "*.py"),
                "modules": get_module_stats(),
            },
            "patterns": {
                "total_lines": count_lines_in_path("./patterns", "*.py"),
                "total_test_files": count_files("./patterns", "*.py"),
                "categories": get_pattern_stats(),
            },
            "tests": {
                "total_lines": count_lines_in_path("./tests", "*.py"),
                "total_files": count_files("./tests", "*.py"),
            },
            "evaluations": {
                "total_lines": count_lines_in_path("./evals", "*.py"),
                "total_files": count_files("./evals", "*.py"),
            },
        },
        "documentation": {
            "total_lines": count_lines_in_path(".", "*.md"),
            "total_files": count_files(".", "*.md"),
        },
    }

    # Calculate total patterns from categories
    python_code = cast(dict[str, Any], stats["python_code"])
    categories = python_code["patterns"]["categories"]
    total_patterns = sum(cat["patterns"] for cat in categories.values())
    python_code["patterns"]["total_patterns"] = total_patterns

    # Calculate totals
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
    git = stats.get("git", {})
    gpu = stats.get("gpu")
    tech = stats.get("tech_stack", {})

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
        "## Git Statistics",
        "",
        f"- **Total Commits:** {git.get('total_commits', 'N/A')}",
        f"- **Branches:** {git.get('branch_count', 'N/A')}",
        f"- **First Commit:** {git.get('first_commit', 'N/A')}",
        f"- **Latest Commit:** {git.get('latest_commit', 'N/A')}",
        f"- **Project Age:** {git.get('age_days', 'N/A')} days",
        "",
    ]

    # GPU section (only if GPU detected)
    if gpu:
        gpu_name = gpu.get("name", "Unknown")
        if gpu.get("count", 1) > 1:
            gpu_name = f"{gpu['count']}x {gpu_name}"
        md.extend(
            [
                "## GPU",
                "",
                f"- **Model:** {gpu_name}",
                f"- **VRAM:** {gpu.get('vram_gb', 'N/A')} GB",
                f"- **Driver:** {gpu.get('driver', 'N/A')}",
                f"- **CUDA:** {gpu.get('cuda', 'N/A')}",
                "",
            ]
        )

    md.extend(
        [
            "## Tech Stack",
            "",
            f"- **Python:** {tech.get('python_version', 'N/A')}",
            f"- **Build Backend:** {tech.get('build_backend', 'N/A')}",
            "",
            "**Core Dependencies:**",
            "",
        ]
    )

    for dep in tech.get("core_dependencies", []):
        md.append(f"- {dep}")

    md.extend(["", "**Dev Tools:**", ""])

    dev_tools = tech.get("dev_tools", {})
    if dev_tools.get("linting"):
        md.append(f"- Linting: {', '.join(dev_tools['linting'])}")
    if dev_tools.get("type_checking"):
        md.append(f"- Type Checking: {', '.join(dev_tools['type_checking'])}")
    if dev_tools.get("testing"):
        md.append(f"- Testing: {', '.join(dev_tools['testing'])}")
    if dev_tools.get("security"):
        md.append(f"- Security: {', '.join(dev_tools['security'])}")

    if tech.get("llm_server"):
        md.extend(["", "**LLM Server:**", ""])
        for dep in tech["llm_server"]:
            md.append(f"- {dep}")

    md.extend(
        [
            "",
            "## Python Code",
            "",
            "### Implementation (src/)",
            "",
            f"- **Total Lines:** {stats['python_code']['implementation']['total_lines']:,}",
            f"- **Files:** {stats['python_code']['implementation']['total_files']}",
            "",
            "**Breakdown by Module:**",
            "",
        ]
    )

    # Add module breakdown
    for module, lines in sorted(
        stats["python_code"]["implementation"]["modules"].items(), key=lambda x: x[1], reverse=True
    ):
        md.append(f"- {module}/: {lines:,} lines")

    md.extend(
        [
            "",
            "### Pattern Definitions (patterns/)",
            "",
            f"- **Total Lines:** {stats['python_code']['patterns']['total_lines']:,}",
            f"- **Total Patterns:** {stats['python_code']['patterns']['total_patterns']}",
            f"- **Categories:** {len(stats['python_code']['patterns']['categories'])}",
            "",
            "**Breakdown by Category:**",
            "",
        ]
    )

    # Add pattern breakdown
    for category, data in sorted(
        stats["python_code"]["patterns"]["categories"].items(),
        key=lambda x: x[1]["lines"],
        reverse=True,
    ):
        md.append(f"- **{category}**: {data['patterns']} patterns ({data['lines']:,} lines)")

    md.extend(
        [
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
            "",
        ]
    )

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
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", default="PROJECT_STATS.md", help="Output file path (default: %(default)s)"
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "both"],
        default="markdown",
        help="Output format: markdown, json, or both (default: %(default)s)",
    )
    parser.add_argument(
        "--history", action="store_true", help="Store timestamped copy in stats/ directory"
    )
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
        json_path = Path(args.output).with_suffix(".json")
        json_path.write_text(json.dumps(stats, indent=2))
        print(f"✓ JSON stats written to {json_path}")

    # Save historical copy
    if args.history:
        stats_dir = Path("stats")
        stats_dir.mkdir(exist_ok=True)

        timestamp = datetime.fromisoformat(stats["timestamp"]).strftime("%Y%m%d_%H%M%S")

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
    print(f"  Patterns: {stats['python_code']['patterns']['total_patterns']}")
    print(f"  Documentation: {stats['totals']['documentation_lines']:,}")


if __name__ == "__main__":
    main()
