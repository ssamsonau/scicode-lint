#!/usr/bin/env python3
"""Semantic diversity check for pattern test files using Claude CLI.

Detects conceptually similar test files WITHIN each pattern that AST hash check misses.
Compares files within the same pattern directory only (not across patterns).

Usage:
    python pattern_verification/deterministic/diversity_check.py           # Check all patterns
    python pattern_verification/deterministic/diversity_check.py ml-001    # Specific pattern
    python pattern_verification/deterministic/diversity_check.py --category ai-training  # Category
    python pattern_verification/deterministic/diversity_check.py --verbose  # Verbose output
Requires: Claude CLI installed and configured (`claude login`).
Rate limiting: Controlled globally via [claude_cli] in config.toml.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tomllib
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Literal

from loguru import logger

# Add project root to sys.path so dev_lib can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dev_lib.claude_cli import (  # noqa: E402
    DISALLOWED_TOOLS_ALL,
    ClaudeCLI,
    ClaudeCLIError,
)
from dev_lib.config import load_project_config  # noqa: E402
from dev_lib.run_output import RunOutput, write_worker  # noqa: E402

# Prompt template for batch diversity comparison (one call per pattern)
BATCH_DIVERSITY_PROMPT = """You are checking test file diversity for a detection pattern.

PATTERN: {pattern_name}
DETECTION QUESTION:
{detection_question}

Compare each pair of test files below and determine if the second file adds testing value beyond the first:
- DIVERSE: Different code pattern, library, structure, or real-world approach
- REDUNDANT: Same demonstration with cosmetic changes (variable renames, minor edits)

{pairs_section}

Return ONLY a JSON array with one object per pair:
[{{"pair_id": 1, "verdict": "diverse"}}, {{"pair_id": 2, "verdict": "redundant"}}, ...]
"""  # noqa: E501

# Template for a same-type pair within the batch prompt
SAME_TYPE_PAIR_TEMPLATE = """--- PAIR {pair_id} ({test_type}-{test_type} comparison) ---
FILE 1: {filename1}
```python
{code1}
```

FILE 2: {filename2}
```python
{code2}
```
"""

# Template for a cross-type pair within the batch prompt
CROSS_TYPE_PAIR_TEMPLATE = """--- PAIR {pair_id} (positive-negative comparison) ---
POSITIVE TEST (contains the bug): {pos_filename}
```python
{positive_code}
```

NEGATIVE TEST (correct code): {neg_filename}
```python
{negative_code}
```
Question: Is the negative genuinely different from the positive, or same structure with bug removed?
"""


def get_default_diversity_model() -> str:
    """Get default model from config.toml (sonnet or opus).

    Raises:
        RuntimeError: If diversity_model is missing or invalid.
    """
    config = load_project_config()
    pv_config = config["pattern_verification"]
    if not isinstance(pv_config, dict):
        raise RuntimeError("Invalid [pattern_verification] section in config.toml")

    value = pv_config.get("diversity_model")
    if value is None:
        raise RuntimeError("Missing diversity_model in [pattern_verification] config")
    if not isinstance(value, str) or value not in ("sonnet", "opus"):
        raise RuntimeError(f"diversity_model must be 'sonnet' or 'opus', got: {value!r}")
    return value


# Default reports directory (relative to script location)
REPORTS_DIR = Path(__file__).parent / "reports"


def find_all_patterns(patterns_dir: Path) -> list[Path]:
    """Find all pattern directories.

    Delegates to shared utility in pattern_verification.utils.
    """
    from pattern_verification.utils import find_all_patterns as _find_all_patterns

    return _find_all_patterns(patterns_dir)


@dataclass
class PatternDiversityResult:
    """Result of diversity check for a single pattern."""

    pattern_id: str
    category: str
    pattern_dir: Path | None = None
    redundant_positive_pairs: list[tuple[str, str]] = field(default_factory=list)
    redundant_negative_pairs: list[tuple[str, str]] = field(default_factory=list)
    fixed_copy_negatives: list[tuple[str, str]] = field(default_factory=list)
    total_comparisons: int = 0
    error: str = ""
    raw_output: str = ""

    @property
    def has_issues(self) -> bool:
        """Return True if any diversity issues were found."""
        return bool(
            self.redundant_positive_pairs
            or self.redundant_negative_pairs
            or self.fixed_copy_negatives
        )


def load_pattern_info(pattern_dir: Path) -> tuple[str, str]:
    """Load pattern name and detection question from pattern.toml.

    Args:
        pattern_dir: Path to pattern directory containing pattern.toml

    Returns:
        Tuple of (pattern_name, detection_question)
    """
    with open(pattern_dir / "pattern.toml", "rb") as f:
        toml = tomllib.load(f)
    name = toml["meta"]["name"]
    question = toml["detection"]["question"]
    return name, question


def load_test_files(test_dir: Path) -> dict[str, str]:
    """Load all test files from directory.

    Args:
        test_dir: Path to test directory (e.g., test_positive/)

    Returns:
        Dict mapping filename to code content
    """
    if not test_dir.exists():
        return {}
    return {
        f.name: f.read_text(encoding="utf-8")
        for f in test_dir.glob("*.py")
        if not f.name.startswith("_")
    }


@dataclass
class _PairInfo:
    """Tracks a comparison pair for mapping results back to files."""

    pair_id: int
    kind: Literal["positive", "negative", "cross"]
    file1: str
    file2: str


def _build_pairs_section(
    positive_files: dict[str, str],
    negative_files: dict[str, str],
) -> tuple[str, list[_PairInfo]]:
    """Build the pairs section of the batch prompt.

    Args:
        positive_files: Dict of positive test {filename: code}
        negative_files: Dict of negative test {filename: code}

    Returns:
        Tuple of (pairs_text, pair_info_list)
    """
    pairs: list[_PairInfo] = []
    sections: list[str] = []
    pair_id = 1

    # Same-type positive pairs
    pos_names = sorted(positive_files.keys())
    for f1, f2 in combinations(pos_names, 2):
        sections.append(
            SAME_TYPE_PAIR_TEMPLATE.format(
                pair_id=pair_id,
                test_type="positive",
                filename1=f1,
                code1=positive_files[f1],
                filename2=f2,
                code2=positive_files[f2],
            )
        )
        pairs.append(_PairInfo(pair_id=pair_id, kind="positive", file1=f1, file2=f2))
        pair_id += 1

    # Same-type negative pairs
    neg_names = sorted(negative_files.keys())
    for f1, f2 in combinations(neg_names, 2):
        sections.append(
            SAME_TYPE_PAIR_TEMPLATE.format(
                pair_id=pair_id,
                test_type="negative",
                filename1=f1,
                code1=negative_files[f1],
                filename2=f2,
                code2=negative_files[f2],
            )
        )
        pairs.append(_PairInfo(pair_id=pair_id, kind="negative", file1=f1, file2=f2))
        pair_id += 1

    # Cross-type pairs (negative vs positive)
    for neg_name in sorted(negative_files.keys()):
        for pos_name in sorted(positive_files.keys()):
            sections.append(
                CROSS_TYPE_PAIR_TEMPLATE.format(
                    pair_id=pair_id,
                    pos_filename=pos_name,
                    positive_code=positive_files[pos_name],
                    neg_filename=neg_name,
                    negative_code=negative_files[neg_name],
                )
            )
            pairs.append(_PairInfo(pair_id=pair_id, kind="cross", file1=neg_name, file2=pos_name))
            pair_id += 1

    return "\n".join(sections), pairs


def _parse_verdicts(output: str, pairs: list[_PairInfo]) -> dict[int, str] | None:
    """Parse JSON array of verdicts from Claude output.

    Args:
        output: Raw Claude CLI stdout
        pairs: List of pair info for validation

    Returns:
        Dict mapping pair_id to verdict string, or None on parse failure
    """
    # Find JSON array in output (may have surrounding text)
    text = output.strip()

    # Try to find JSON array boundaries
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        logger.error(f"No JSON array found in Claude output: {text[:200]}")
        return None

    json_str = text[start : end + 1]
    try:
        verdicts_list = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from Claude output: {e}\nJSON: {json_str[:200]}")
        return None

    verdicts: dict[int, str] = {}
    for item in verdicts_list:
        if isinstance(item, dict) and "pair_id" in item and "verdict" in item:
            pid = int(item["pair_id"])
            verdict = item["verdict"].lower()
            if verdict in ("diverse", "redundant"):
                verdicts[pid] = verdict

    # Validate we got verdicts for all pairs
    expected_ids = {p.pair_id for p in pairs}
    missing = expected_ids - set(verdicts.keys())
    if missing:
        logger.warning(
            f"Missing verdicts for pair IDs {sorted(missing)} (got {len(verdicts)}/{len(pairs)})"
        )

    return verdicts


async def check_pattern_diversity(
    pattern_dir: Path,
    cli: ClaudeCLI,
    verbose: bool = False,
) -> PatternDiversityResult:
    """Run all diversity checks for a single pattern via Claude CLI.

    Rate limiting is handled globally by ClaudeCLI.

    Args:
        pattern_dir: Path to pattern directory
        cli: Claude CLI wrapper instance
        verbose: Print verbose output

    Returns:
        PatternDiversityResult with any issues found
    """
    result = PatternDiversityResult(
        pattern_id=pattern_dir.name,
        category=pattern_dir.parent.name,
        pattern_dir=pattern_dir,
    )

    # Load pattern info
    try:
        pattern_name, detection_question = load_pattern_info(pattern_dir)
    except Exception as e:
        logger.error(f"Failed to load pattern info for {pattern_dir.name}: {e}")
        return result

    # Load test files
    positive_files = load_test_files(pattern_dir / "test_positive")
    negative_files = load_test_files(pattern_dir / "test_negative")

    if verbose:
        logger.info(
            f"Checking {pattern_dir.name}: "
            f"{len(positive_files)} positive, {len(negative_files)} negative files"
        )

    # Calculate total comparisons
    n_pos = len(positive_files)
    n_neg = len(negative_files)
    pos_pairs = n_pos * (n_pos - 1) // 2 if n_pos >= 2 else 0
    neg_pairs = n_neg * (n_neg - 1) // 2 if n_neg >= 2 else 0
    cross_pairs = n_pos * n_neg
    result.total_comparisons = pos_pairs + neg_pairs + cross_pairs

    if result.total_comparisons == 0:
        return result

    # Build batch prompt
    pairs_section, pairs_info = _build_pairs_section(positive_files, negative_files)
    prompt = BATCH_DIVERSITY_PROMPT.format(
        pattern_name=pattern_name,
        detection_question=detection_question,
        pairs_section=pairs_section,
    )

    # Call Claude CLI (rate limiting handled globally by ClaudeCLI)
    try:
        cli_result = await cli.arun(prompt, disallowed_tools=DISALLOWED_TOOLS_ALL)
        output = cli_result.stdout
        result.raw_output = output
    except ClaudeCLIError as e:
        result.error = f"Claude CLI error for {pattern_dir.name}: {e}"
        logger.error(result.error)
        return result

    # Parse verdicts
    verdicts = _parse_verdicts(output, pairs_info)

    if verdicts is None:
        result.error = f"Failed to parse Claude JSON output for {pattern_dir.name}"
        logger.error(result.error)
        return result

    # Map verdicts back to pair types
    for pair in pairs_info:
        verdict = verdicts.get(pair.pair_id)
        if verdict != "redundant":
            continue
        if pair.kind == "positive":
            result.redundant_positive_pairs.append((pair.file1, pair.file2))
        elif pair.kind == "negative":
            result.redundant_negative_pairs.append((pair.file1, pair.file2))
        elif pair.kind == "cross":
            result.fixed_copy_negatives.append((pair.file1, pair.file2))

    return result


def format_results(results: list[PatternDiversityResult]) -> str:
    """Format diversity check results for display.

    Args:
        results: List of PatternDiversityResult objects

    Returns:
        Formatted string for terminal output
    """
    lines: list[str] = []
    lines.append("Diversity Check Results")
    lines.append("=" * 23)
    lines.append("")

    # Count patterns with issues
    patterns_with_issues = [r for r in results if r.has_issues]
    total_redundant_pairs = sum(
        len(r.redundant_positive_pairs) + len(r.redundant_negative_pairs) for r in results
    )
    total_fixed_copies = sum(len(r.fixed_copy_negatives) for r in results)

    if not patterns_with_issues:
        lines.append("No diversity issues found.")
        lines.append("")
    else:
        lines.append("ISSUES FOUND:")
        lines.append("")

        for r in patterns_with_issues:
            base = r.pattern_dir if r.pattern_dir else Path(f"{r.category}/{r.pattern_id}")
            lines.append(f"[diversity] {r.category}/{r.pattern_id}")
            lines.append("")

            if r.redundant_positive_pairs:
                lines.append("   Redundant positive pairs:")
                for f1, f2 in r.redundant_positive_pairs:
                    lines.append(f"   - {base / 'test_positive' / f1}")
                    lines.append(f"     {base / 'test_positive' / f2}")
                lines.append("")

            if r.redundant_negative_pairs:
                lines.append("   Redundant negative pairs:")
                for f1, f2 in r.redundant_negative_pairs:
                    lines.append(f"   - {base / 'test_negative' / f1}")
                    lines.append(f"     {base / 'test_negative' / f2}")
                lines.append("")

            if r.fixed_copy_negatives:
                lines.append("   Non-diverse negatives (same code as positive, with bug removed):")
                for neg, pos in r.fixed_copy_negatives:
                    lines.append(f"   - {base / 'test_negative' / neg}")
                    lines.append(f"     (same structure as {base / 'test_positive' / pos})")
                lines.append("")

            lines.append("---")
            lines.append("")

    # Report errors (failed to parse Claude output)
    errors = [r for r in results if r.error]
    if errors:
        lines.append("ERRORS (could not parse Claude output):")
        lines.append("")
        for r in errors:
            lines.append(f"   {r.category}/{r.pattern_id}: {r.error}")
        lines.append("")

    # Summary
    total_patterns = len(results)
    total_comparisons = sum(r.total_comparisons for r in results)
    issues = len(patterns_with_issues)
    error_count = len(errors)
    lines.append(f"Summary: {issues} pattern(s) with issues out of {total_patterns} checked")
    lines.append(f"         {total_redundant_pairs} redundant, {total_fixed_copies} non-diverse")
    lines.append(f"         {total_comparisons} total comparisons made")
    if error_count:
        lines.append(f"         {error_count} error(s) - rerun failed patterns")

    return "\n".join(lines)


def format_pattern_log(result: PatternDiversityResult) -> str:
    """Format a single pattern result as log output.

    Args:
        result: PatternDiversityResult for one pattern

    Returns:
        Log formatted string with raw Claude output
    """
    lines = [
        f"Pattern: {result.pattern_id}",
        f"Category: {result.category}",
        f"Comparisons: {result.total_comparisons}",
        f"Has issues: {result.has_issues}",
        "=" * 60,
        "",
    ]

    if result.redundant_positive_pairs:
        lines.append("Redundant positive pairs:")
        for f1, f2 in result.redundant_positive_pairs:
            lines.append(f"  - {f1} <-> {f2}")
        lines.append("")

    if result.redundant_negative_pairs:
        lines.append("Redundant negative pairs:")
        for f1, f2 in result.redundant_negative_pairs:
            lines.append(f"  - {f1} <-> {f2}")
        lines.append("")

    if result.fixed_copy_negatives:
        lines.append("Non-diverse negatives:")
        for neg, pos in result.fixed_copy_negatives:
            lines.append(f"  - {neg} (same structure as {pos})")
        lines.append("")

    if result.error:
        lines.append(f"Error: {result.error}")
        lines.append("")

    if result.raw_output:
        lines.append("Claude Output:")
        lines.append("-" * 40)
        lines.append(result.raw_output)

    return "\n".join(lines)


async def run_diversity_check(
    patterns: list[Path],
    model: str = "sonnet",
    timeout: int = 120,
    verbose: bool = False,
    run_output: RunOutput | None = None,
    quiet: bool = False,
) -> list[PatternDiversityResult]:
    """Run diversity check on all patterns.

    Rate limiting is handled globally by ClaudeCLI (semaphore + RPM limiter).
    Pattern logs are written incrementally as results come in, using an async
    queue to serialize disk writes and avoid I/O contention.

    Args:
        patterns: List of pattern directory paths
        model: Claude model to use (sonnet, opus)
        timeout: Timeout per pattern in seconds
        verbose: Print verbose output
        run_output: Optional output directory for writing logs
        quiet: If True, suppress stdout progress

    Returns:
        List of PatternDiversityResult for all patterns
    """
    cli = ClaudeCLI(model=model, effort="low", timeout=timeout)
    write_queue: asyncio.Queue[tuple[Path, str] | None] = asyncio.Queue()

    # Start the write worker
    writer_task = asyncio.create_task(write_worker(write_queue))

    # Open progress log file
    progress_file = run_output.log.open("a") if run_output else None

    # Check all patterns
    tasks = [
        asyncio.create_task(check_pattern_diversity(pattern_dir, cli, verbose))
        for pattern_dir in patterns
    ]

    results: list[PatternDiversityResult] = []
    total = len(patterns)

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed = len(results)

        # Progress output
        if result.error:
            icon = "✗"
        elif result.has_issues:
            icon = "⚠"
        else:
            icon = "✓"
        msg = f"[{completed}/{total}] {icon} {result.category}/{result.pattern_id}"

        if not quiet:
            print(msg, flush=True)

        if progress_file:
            progress_file.write(msg + "\n")
            progress_file.flush()

        # Queue pattern log write (non-blocking, serialized by worker)
        if run_output and (result.raw_output or result.error):
            pattern_file = run_output.item_file(result.pattern_id)
            content = format_pattern_log(result)
            await write_queue.put((pattern_file, content))
            result.raw_output = ""  # Free memory after queuing

    # Signal writer to stop and wait for completion
    await write_queue.put(None)
    await writer_task

    if progress_file:
        progress_file.close()

    return results


def main() -> int:
    """Main entry point for diversity check."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pattern", nargs="?", help="Specific pattern ID (e.g., ml-001)")
    parser.add_argument(
        "--category",
        help="Check patterns in specific category (e.g., ai-training)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    try:
        default_model = get_default_diversity_model()
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1
    parser.add_argument(
        "--model",
        "-m",
        choices=["sonnet", "opus"],
        default=default_model,
        help=f"Claude model to use (default: {default_model})",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=120,
        help="Timeout per pattern in seconds (default: 120)",
    )
    args = parser.parse_args()

    patterns_dir = Path("src/scicode_lint/patterns")
    if not patterns_dir.exists():
        print("Error: src/scicode_lint/patterns/ directory not found", file=sys.stderr)
        return 1

    # Find patterns to check
    if args.pattern:
        from pattern_verification.utils import resolve_pattern

        patterns = resolve_pattern(patterns_dir, args.pattern)
        if not patterns:
            print(f"Error: Pattern '{args.pattern}' not found", file=sys.stderr)
            return 1
    elif args.category:
        category_dir = patterns_dir / args.category
        if not category_dir.exists():
            print(f"Error: Category '{args.category}' not found", file=sys.stderr)
            return 1
        patterns = [p for p in find_all_patterns(patterns_dir) if p.parent.name == args.category]
    else:
        patterns = find_all_patterns(patterns_dir)

    if not patterns:
        print("No patterns found to check", file=sys.stderr)
        return 1

    if args.verbose:
        logger.enable("scicode_lint")

    # Determine scope for output directory naming
    if args.pattern:
        scope = args.pattern
    elif args.category:
        scope = args.category
    else:
        scope = "all"

    # Create run output directory for batch runs (>1 pattern)
    run_output: RunOutput | None = None
    if len(patterns) > 1:
        run_output = RunOutput.create(REPORTS_DIR, scope, items_dirname="patterns")
        print(f"Output directory: {run_output.run_dir}")
        print(f"  Summary:  {run_output.summary}")
        print(f"  Log:      {run_output.log}")
        print(f"  Patterns: {run_output.items_dir}/")
        run_output.init_log()

    print(f"Checking {len(patterns)} pattern(s)...")
    print(f"Model: {args.model}, Timeout: {args.timeout}s")
    if run_output:
        print(f"Monitor progress: tail -f {run_output.log}")
    print()

    # Run async check (pattern logs written incrementally via async queue)
    results = asyncio.run(
        run_diversity_check(
            patterns,
            model=args.model,
            timeout=args.timeout,
            verbose=args.verbose,
            run_output=run_output,
        )
    )

    # Format and output results
    summary = format_results(results)

    if run_output:
        run_output.summary.write_text(summary)
        print()
        print(summary)
        print(f"\nResults written to: {run_output.run_dir}")
    else:
        print()
        print(summary)

    # Return exit code (fail on issues OR parse errors)
    has_issues = any(r.has_issues for r in results)
    has_errors = any(r.error for r in results)
    return 1 if has_issues or has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
