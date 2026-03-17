#!/usr/bin/env python3
"""Semantic pattern validation using Claude Code agents.

Usage:
    python pattern_verification/semantic/semantic_validate.py pt-001           # Single pattern
    python pattern_verification/semantic/semantic_validate.py --category ai-training  # Category
    python pattern_verification/semantic/semantic_validate.py --all            # All patterns

Output structure (auto-generated for batch runs):
    reports/
    └── YYYYMMDD_HHMMSS_<scope>/
        ├── summary.md          # Overall summary
        ├── progress.log        # One line per pattern completion
        └── patterns/
            ├── pt-001.log      # Raw Claude agent output (written as each completes)
            ├── pt-003.log
            └── ...

Background execution with real-time monitoring:
    python pattern_verification/semantic/semantic_validate.py --all &
    tail -f pattern_verification/semantic/reports/*/progress.log
    tail -f pattern_verification/semantic/reports/*/patterns/*.log  # Individual pattern logs

Options:
    --model MODEL    Claude model: sonnet, opus (default from config.toml)
    --timeout SECS   Timeout per pattern in seconds (default: 300)
    --quiet          Suppress stdout progress

Rate limiting: Controlled globally via [claude_cli] in config.toml.

Architecture:
    Uses async write queue to serialize disk I/O. Multiple Claude processes run in
    parallel, but pattern logs are written one at a time through a single writer.
    This prevents disk I/O spikes that can freeze WSL with many concurrent writes.

Requires: Claude CLI installed and configured.

What it checks (LLM-powered, scripts can't do this):
1. Does each test file's description accurately describe the code?
2. Does expected_issue align with detection question's YES condition?
3. Does snippet in expected_location actually exist in the test file?
4. Do positive tests actually contain the bug described?
5. Do negative tests actually avoid the bug?
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Add project root to sys.path so dev_lib can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dev_lib.claude_cli import DEFAULT_DISALLOWED_TOOLS, ClaudeCLI, ClaudeCLIError  # noqa: E402
from dev_lib.config import load_project_config  # noqa: E402
from dev_lib.run_output import RunOutput, write_worker  # noqa: E402


def get_default_semantic_model() -> str:
    """Get default model from config.toml (sonnet or opus).

    Raises:
        RuntimeError: If semantic_model is missing or invalid.
    """
    config = load_project_config()
    pv_config = config["pattern_verification"]
    if not isinstance(pv_config, dict):
        raise RuntimeError("Invalid [pattern_verification] section in config.toml")

    value = pv_config.get("semantic_model")
    if value is None:
        raise RuntimeError("Missing semantic_model in [pattern_verification] config")
    if not isinstance(value, str) or value not in ("sonnet", "opus"):
        raise RuntimeError(f"semantic_model must be 'sonnet' or 'opus', got: {value!r}")
    return value


# Categories in the patterns directory
CATEGORIES = [
    "ai-inference",
    "ai-training",
    "scientific-numerical",
    "scientific-performance",
    "scientific-reproducibility",
]

# Default reports directory (relative to script location)
REPORTS_DIR = Path(__file__).parent / "reports"


@dataclass
class SemanticResult:
    """Result of semantic validation for a pattern."""

    pattern_id: str
    category: str
    status: str = "pending"  # pending, running, ok, issues, error
    issues: list[str] = field(default_factory=list)
    error: str = ""
    output: str = ""


def find_patterns_dir() -> Path:
    """Find the patterns directory."""
    # Try relative to script (pattern_verification/semantic/ -> root -> src/scicode_lint/patterns)
    script_dir = Path(__file__).parent.parent.parent
    patterns_dir = script_dir / "src" / "scicode_lint" / "patterns"
    if patterns_dir.exists():
        return patterns_dir

    # Try current directory
    if Path("src/scicode_lint/patterns").exists():
        return Path("src/scicode_lint/patterns")

    raise FileNotFoundError("src/scicode_lint/patterns/ directory not found")


def get_all_pattern_ids(patterns_dir: Path) -> list[tuple[str, str]]:
    """Get all pattern IDs with their categories.

    Returns:
        List of (pattern_id, category) tuples
    """
    patterns = []
    for category_dir in sorted(patterns_dir.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith((".", "_")):
            continue
        for pattern_dir in sorted(category_dir.iterdir()):
            if not pattern_dir.is_dir() or pattern_dir.name.startswith((".", "_")):
                continue
            if (pattern_dir / "pattern.toml").exists():
                patterns.append((pattern_dir.name, category_dir.name))
    return patterns


def get_patterns_by_category(patterns_dir: Path, category: str) -> list[tuple[str, str]]:
    """Get all pattern IDs in a specific category."""
    patterns: list[tuple[str, str]] = []
    category_dir = patterns_dir / category
    if not category_dir.exists():
        return patterns
    for pattern_dir in sorted(category_dir.iterdir()):
        if not pattern_dir.is_dir() or pattern_dir.name.startswith((".", "_")):
            continue
        if (pattern_dir / "pattern.toml").exists():
            patterns.append((pattern_dir.name, category))
    return patterns


def find_pattern(patterns_dir: Path, pattern_id: str) -> tuple[str, str] | None:
    """Find a pattern by ID, returning (pattern_id, category) or None."""
    for category_dir in patterns_dir.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith((".", "_")):
            continue
        for pattern_dir in category_dir.iterdir():
            if pattern_dir.name == pattern_id or pattern_id in pattern_dir.name:
                if (pattern_dir / "pattern.toml").exists():
                    return (pattern_dir.name, category_dir.name)
    return None


async def run_semantic_check(
    pattern_id: str,
    category: str,
    timeout: int,
    cli: ClaudeCLI,
) -> SemanticResult:
    """Run semantic validation for a single pattern using Claude CLI.

    Rate limiting is handled globally by ClaudeCLI.

    Args:
        pattern_id: Pattern ID to validate
        category: Category of the pattern
        timeout: Timeout in seconds
        cli: Claude CLI wrapper instance

    Returns:
        SemanticResult with validation results
    """
    result = SemanticResult(pattern_id=pattern_id, category=category)
    result.status = "running"
    prompt = f"Review {pattern_id}"

    try:
        cli_result = await cli.arun(
            prompt,
            agent="pattern-reviewer",
            disallowed_tools=DEFAULT_DISALLOWED_TOOLS,
            timeout=timeout,
        )
        output = cli_result.stdout
        result.output = output

        # Parse output to determine if there are issues
        output_lower = output.lower()
        if any(
            term in output_lower
            for term in ["issue found", "inconsisten", "mismatch", "incorrect", "missing"]
        ):
            result.status = "issues"
            # Extract issue lines (simple heuristic)
            for line in output.split("\n"):
                line_lower = line.lower()
                if any(
                    term in line_lower
                    for term in ["issue", "error", "mismatch", "incorrect", "missing"]
                ):
                    result.issues.append(line.strip())
        else:
            result.status = "ok"

    except ClaudeCLIError as e:
        result.status = "error"
        result.error = str(e)

    return result


async def run_all_checks(
    patterns: list[tuple[str, str]],
    timeout: int,
    model: str,
    run_output: RunOutput | None = None,
    quiet: bool = False,
) -> list[SemanticResult]:
    """Run semantic validation on multiple patterns.

    Rate limiting is handled globally by ClaudeCLI (semaphore + RPM limiter).
    Pattern logs are written incrementally as results come in, using an async
    queue to serialize disk writes and avoid I/O contention.

    Args:
        patterns: List of (pattern_id, category) tuples
        timeout: Timeout per pattern in seconds
        model: Claude model to use (sonnet, opus)
        run_output: Optional output directory for writing logs
        quiet: If True, suppress stdout progress

    Returns:
        List of SemanticResult
    """
    cli = ClaudeCLI(model=model, effort="medium")
    write_queue: asyncio.Queue[tuple[Path, str] | None] = asyncio.Queue()

    # Start the write worker
    writer_task = asyncio.create_task(write_worker(write_queue))

    # Open progress log file
    progress_file = run_output.log.open("a") if run_output else None

    tasks = []
    for pattern_id, category in patterns:
        task = asyncio.create_task(run_semantic_check(pattern_id, category, timeout, cli))
        tasks.append(task)

    results = []
    total = len(patterns)

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed = len(results)

        # Progress output
        icon = {"ok": "✓", "issues": "⚠", "error": "✗"}.get(result.status, "?")
        msg = f"[{completed}/{total}] {icon} {result.category}/{result.pattern_id}"

        if not quiet:
            print(msg, flush=True)

        if progress_file:
            progress_file.write(msg + "\n")
            progress_file.flush()

        # Queue pattern log write (non-blocking, serialized by worker)
        if run_output and result.output:
            pattern_file = run_output.item_file(result.pattern_id)
            content = format_pattern_log(result)
            await write_queue.put((pattern_file, content))
            result.output = ""  # Free memory after queuing

    # Signal writer to stop and wait for completion
    await write_queue.put(None)
    await writer_task

    if progress_file:
        progress_file.close()

    # Sort by category and pattern_id
    results.sort(key=lambda r: (r.category, r.pattern_id))
    return results


def format_results_markdown(results: list[SemanticResult]) -> str:
    """Format results as markdown.

    Args:
        results: List of SemanticResult

    Returns:
        Markdown formatted string
    """
    lines = [
        "# Semantic Validation Results",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        f"- Total patterns: {len(results)}",
        f"- OK: {sum(1 for r in results if r.status == 'ok')}",
        f"- Issues found: {sum(1 for r in results if r.status == 'issues')}",
        f"- Errors: {sum(1 for r in results if r.status == 'error')}",
        "",
    ]

    # Group by status
    issues = [r for r in results if r.status == "issues"]
    errors = [r for r in results if r.status == "error"]
    ok = [r for r in results if r.status == "ok"]

    if issues:
        lines.extend(["## Issues Found", ""])
        for r in issues:
            lines.append(f"### {r.category}/{r.pattern_id}")
            lines.append("")
            for issue in r.issues:
                lines.append(f"- {issue}")
            lines.append("")

    if errors:
        lines.extend(["## Errors", ""])
        for r in errors:
            lines.append(f"### {r.category}/{r.pattern_id}")
            lines.append(f"Error: {r.error}")
            lines.append("")

    if ok:
        lines.extend(["## Patterns OK", ""])
        current_category = ""
        for r in ok:
            if r.category != current_category:
                current_category = r.category
                lines.append(f"### {current_category}")
            lines.append(f"- {r.pattern_id}")
        lines.append("")

    return "\n".join(lines)


def format_pattern_log(result: SemanticResult) -> str:
    """Format a single pattern result as log output.

    Args:
        result: SemanticResult for one pattern

    Returns:
        Log formatted string with raw Claude output
    """
    lines = [
        f"Pattern: {result.pattern_id}",
        f"Category: {result.category}",
        f"Status: {result.status}",
        "=" * 60,
        "",
    ]

    if result.status == "issues":
        lines.append("Issues Found:")
        for issue in result.issues:
            lines.append(f"  - {issue}")
        lines.append("")

    if result.status == "error":
        lines.append(f"Error: {result.error}")
        lines.append("")

    if result.output:
        lines.append("Claude Agent Output:")
        lines.append("-" * 40)
        lines.append(result.output)

    return "\n".join(lines)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pattern",
        nargs="?",
        help="Specific pattern ID (e.g., ml-002, pt-001)",
    )
    parser.add_argument(
        "--category",
        choices=CATEGORIES,
        help="Validate all patterns in category",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_patterns",
        help="Validate all patterns",
    )
    try:
        default_model = get_default_semantic_model()
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=300,
        help="Timeout per pattern in seconds (default: 300)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--model",
        "-m",
        choices=["sonnet", "opus"],
        default=default_model,
        help=f"Claude model to use (default: {default_model}, from config.toml)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.pattern and not args.category and not args.all_patterns:
        parser.error("Specify a pattern ID, --category, or --all")

    try:
        patterns_dir = find_patterns_dir()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Build pattern list
    patterns: list[tuple[str, str]] = []

    # Determine scope and build pattern list
    scope = ""
    if args.pattern:
        from pattern_verification.utils import resolve_pattern

        resolved = resolve_pattern(patterns_dir, args.pattern)
        if not resolved:
            print(f"Error: Pattern '{args.pattern}' not found", file=sys.stderr)
            return 1
        # Convert Path to (pattern_id, category) tuple
        p = resolved[0]
        found = (p.name, p.parent.name)
        patterns = [found]
        scope = found[0]  # pattern_id

    elif args.category:
        patterns = get_patterns_by_category(patterns_dir, args.category)
        if not patterns:
            print(f"Error: No patterns found in category '{args.category}'", file=sys.stderr)
            return 1
        scope = args.category

    elif args.all_patterns:
        patterns = get_all_pattern_ids(patterns_dir)
        scope = "all"

    # Create run output directory for batch runs
    run_output: RunOutput | None = None
    if len(patterns) > 1:
        run_output = RunOutput.create(REPORTS_DIR, scope, items_dirname="patterns")
        print(f"Output directory: {run_output.run_dir}")
        print(f"  Summary:  {run_output.summary}")
        print(f"  Log:      {run_output.log}")
        print(f"  Patterns: {run_output.items_dir}/")
        run_output.init_log()

    print(f"Validating {len(patterns)} pattern(s)...")
    print(f"Model: {args.model}")
    print(f"Timeout: {args.timeout}s per pattern")
    if run_output:
        print(f"Monitor progress: tail -f {run_output.log}")
        print(f"Pattern logs appear in: {run_output.items_dir}/")
    print()

    # Run validation (pattern logs written incrementally via async queue)
    results = asyncio.run(
        run_all_checks(
            patterns,
            timeout=args.timeout,
            model=args.model,
            run_output=run_output,
            quiet=args.quiet,
        )
    )

    # Format and output results
    markdown = format_results_markdown(results)

    if run_output:
        run_output.summary.write_text(markdown)
        print(f"Results written to: {run_output.run_dir}")
    else:
        print("\n" + "=" * 60)
        print(markdown)

    # Summary
    issues_count = sum(1 for r in results if r.status == "issues")
    errors_count = sum(1 for r in results if r.status == "error")

    if issues_count > 0 or errors_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
