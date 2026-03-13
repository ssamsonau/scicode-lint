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
    --parallel N     Run N concurrent Claude processes (default from config.toml)
    --model MODEL    Claude model: haiku, sonnet, opus (default from config.toml)
    --timeout SECS   Timeout per pattern in seconds (default: 300)
    --quiet          Suppress stdout progress

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
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles

if TYPE_CHECKING:
    from asyncio.subprocess import Process

# Default parallel count (used if config.toml cannot be loaded)
DEFAULT_SEMANTIC_PARALLEL = 4
DEFAULT_SEMANTIC_MODEL = "haiku"


def _load_config() -> dict[str, object]:
    """Load config.toml from project root."""
    config_path = Path(__file__).parent.parent.parent / "config.toml"
    if config_path.exists():
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    return {}


def get_default_semantic_parallel() -> int:
    """Get default parallel count from config.toml."""
    config = _load_config()
    pv_config = config.get("pattern_verification", {})
    if isinstance(pv_config, dict):
        value = pv_config.get("semantic_parallel", DEFAULT_SEMANTIC_PARALLEL)
        if isinstance(value, int):
            return value
    return DEFAULT_SEMANTIC_PARALLEL


def get_default_semantic_model() -> str:
    """Get default model from config.toml (haiku, sonnet, or opus)."""
    config = _load_config()
    pv_config = config.get("pattern_verification", {})
    if isinstance(pv_config, dict):
        value = pv_config.get("semantic_model", DEFAULT_SEMANTIC_MODEL)
        if isinstance(value, str) and value in ("haiku", "sonnet", "opus"):
            return value
    return DEFAULT_SEMANTIC_MODEL


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
class RunOutput:
    """Output paths for a validation run."""

    run_dir: Path
    summary: Path
    log: Path
    patterns_dir: Path

    @classmethod
    def create(cls, scope: str) -> RunOutput:
        """Create a new run output directory structure.

        Args:
            scope: What's being validated (e.g., "all", "ai-training", "pt-001")

        Returns:
            RunOutput with all paths created
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = REPORTS_DIR / f"{timestamp}_{scope}"
        patterns_dir = run_dir / "patterns"

        # Create directories
        run_dir.mkdir(parents=True, exist_ok=True)
        patterns_dir.mkdir(exist_ok=True)

        return cls(
            run_dir=run_dir,
            summary=run_dir / "summary.md",
            log=run_dir / "progress.log",
            patterns_dir=patterns_dir,
        )

    def pattern_file(self, pattern_id: str) -> Path:
        """Get path for individual pattern log (raw Claude output)."""
        return self.patterns_dir / f"{pattern_id}.log"


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
    semaphore: asyncio.Semaphore,
    model: str,
) -> SemanticResult:
    """Run semantic validation for a single pattern using Claude CLI.

    Args:
        pattern_id: Pattern ID to validate
        category: Category of the pattern
        timeout: Timeout in seconds
        semaphore: Semaphore for concurrency control
        model: Claude model to use (haiku, sonnet, opus)

    Returns:
        SemanticResult with validation results
    """
    result = SemanticResult(pattern_id=pattern_id, category=category)

    async with semaphore:
        result.status = "running"

        # Build the prompt for the pattern-reviewer agent
        prompt = f"Review {pattern_id}"

        try:
            # Run claude CLI with pattern-reviewer agent
            # Explicitly restrict tools to prevent Task spawning (memory explosion)
            # Use "--" to separate options from prompt (--disallowed-tools is variadic)
            proc: Process = await asyncio.create_subprocess_exec(
                "claude",
                "--agent",
                "pattern-reviewer",
                "--model",
                model,
                "--print",
                "--disallowed-tools",
                "Task,WebSearch,WebFetch,Bash,Write,Edit,NotebookEdit",
                "--",  # End of options
                prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                result.status = "error"
                result.error = f"Timeout after {timeout}s"
                return result

            if proc.returncode != 0:
                result.status = "error"
                result.error = stderr.decode() if stderr else f"Exit code {proc.returncode}"
                return result

            output = stdout.decode() if stdout else ""
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

        except FileNotFoundError:
            result.status = "error"
            result.error = "Claude CLI not found. Install: npm install -g @anthropic-ai/claude-code"
        except Exception as e:
            result.status = "error"
            result.error = str(e)

    return result


async def _write_worker(
    queue: asyncio.Queue[tuple[Path, str] | None],
) -> None:
    """Worker that writes pattern logs sequentially from a queue.

    Serializes disk writes to avoid I/O contention. Stops when None is received.
    """
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        path, content = item
        async with aiofiles.open(path, "w") as f:
            await f.write(content)
        queue.task_done()


async def run_all_checks(
    patterns: list[tuple[str, str]],
    parallel: int,
    timeout: int,
    model: str,
    run_output: RunOutput | None = None,
    quiet: bool = False,
) -> list[SemanticResult]:
    """Run semantic validation on multiple patterns.

    Pattern logs are written incrementally as results come in, using an async
    queue to serialize disk writes and avoid I/O contention.

    Args:
        patterns: List of (pattern_id, category) tuples
        parallel: Max concurrent validations
        timeout: Timeout per pattern in seconds
        model: Claude model to use (haiku, sonnet, opus)
        run_output: Optional output directory for writing logs
        quiet: If True, suppress stdout progress

    Returns:
        List of SemanticResult
    """
    semaphore = asyncio.Semaphore(parallel)
    write_queue: asyncio.Queue[tuple[Path, str] | None] = asyncio.Queue()

    # Start the write worker
    writer_task = asyncio.create_task(_write_worker(write_queue))

    # Open progress log file
    progress_file = run_output.log.open("a") if run_output else None

    tasks = []
    for pattern_id, category in patterns:
        task = asyncio.create_task(
            run_semantic_check(pattern_id, category, timeout, semaphore, model)
        )
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
            pattern_file = run_output.pattern_file(result.pattern_id)
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
        help="Validate all 64 patterns",
    )
    default_parallel = get_default_semantic_parallel()
    parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        default=default_parallel,
        help=f"Max concurrent validations (default: {default_parallel}, from config.toml)",
    )
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
    default_model = get_default_semantic_model()
    parser.add_argument(
        "--model",
        "-m",
        choices=["haiku", "sonnet", "opus"],
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
        found = find_pattern(patterns_dir, args.pattern)
        if not found:
            print(f"Error: Pattern '{args.pattern}' not found", file=sys.stderr)
            return 1
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
        run_output = RunOutput.create(scope)
        print(f"Output directory: {run_output.run_dir}")
        print(f"  Summary:  {run_output.summary}")
        print(f"  Log:      {run_output.log}")
        print(f"  Patterns: {run_output.patterns_dir}/")
        run_output.log.write_text("")  # Initialize log file

    print(f"Validating {len(patterns)} pattern(s) with {args.parallel} concurrent processes...")
    print(f"Model: {args.model}")
    print(f"Timeout: {args.timeout}s per pattern")
    if run_output:
        print(f"Monitor progress: tail -f {run_output.log}")
        print(f"Pattern logs appear in: {run_output.patterns_dir}/")
    print()

    # Run validation (pattern logs written incrementally via async queue)
    results = asyncio.run(
        run_all_checks(
            patterns,
            parallel=args.parallel,
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
