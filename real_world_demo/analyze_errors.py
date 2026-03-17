"""Analyze false positives from verification results using Claude (Opus).

Automates Stage 3 of the Meta Improvement Loop: groups invalid findings by pattern,
sends verification reasoning to Claude for theme extraction, and produces actionable
recommendations for improving detection questions.

Usage:
    python -m real_world_demo.analyze_errors --run-id 56
    python -m real_world_demo.analyze_errors --run-id 56 --min-invalid 3
    python -m real_world_demo.analyze_errors --run-id 56 --patterns perf-004 par-005

Output:
    reports/ERROR_ANALYSIS_<date>.md
    + Per-pattern analysis saved to reports/error_analysis/<pattern>.md

Requires: Claude CLI installed and authenticated (claude login).
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from dev_lib.claude_cli import DISALLOWED_TOOLS_ALL, ClaudeCLI, ClaudeCLIError

from .config import REPORTS_DIR
from .database import get_db_path, init_db

# Defaults
DEFAULT_MODEL = "sonnet"
DEFAULT_TIMEOUT = 300
DEFAULT_MIN_INVALID = 2


@dataclass
class PatternStats:
    """Aggregated stats for a single pattern."""

    pattern_id: str
    category: str
    valid: int = 0
    invalid: int = 0
    uncertain: int = 0
    total: int = 0
    precision: float = 0.0
    invalid_reasonings: list[str] = field(default_factory=list)
    valid_reasonings: list[str] = field(default_factory=list)
    detection_question: str = ""


@dataclass
class PatternAnalysis:
    """Claude's analysis of a pattern's false positives."""

    pattern_id: str
    category: str
    stats: PatternStats
    analysis: str = ""
    error: str = ""


def load_pattern_stats(
    conn: sqlite3.Connection,
    run_id: int,
    min_invalid: int = DEFAULT_MIN_INVALID,
    patterns: list[str] | None = None,
) -> list[PatternStats]:
    """Load per-pattern verification stats with reasoning texts.

    Args:
        conn: Database connection.
        run_id: Analysis run ID.
        min_invalid: Minimum invalid count to include pattern.
        patterns: Optional list of specific pattern IDs to analyze.

    Returns:
        List of PatternStats, sorted by invalid count descending.
    """
    # Get pattern-level stats
    query = """
        SELECT
            fn.pattern_id,
            fn.category,
            SUM(CASE WHEN fv.status = 'valid' THEN 1 ELSE 0 END) as valid,
            SUM(CASE WHEN fv.status = 'invalid' THEN 1 ELSE 0 END) as invalid,
            SUM(CASE WHEN fv.status = 'uncertain' THEN 1 ELSE 0 END) as uncertain,
            COUNT(*) as total
        FROM finding_verifications fv
        JOIN findings fn ON fn.id = fv.finding_id
        JOIN file_analyses fa ON fn.file_analysis_id = fa.id
        WHERE fa.run_id = ?
        GROUP BY fn.pattern_id, fn.category
        HAVING SUM(CASE WHEN fv.status = 'invalid' THEN 1 ELSE 0 END) >= ?
        ORDER BY invalid DESC
    """
    params: list[int] = [run_id, min_invalid]
    rows = conn.execute(query, params).fetchall()

    result = []
    for row in rows:
        pattern_id = row[0]

        # Filter by specific patterns if requested
        if patterns and pattern_id not in patterns:
            continue

        valid_count = row[2]
        invalid_count = row[3]
        evaluated = valid_count + invalid_count
        precision = valid_count / evaluated * 100 if evaluated > 0 else 0.0

        stats = PatternStats(
            pattern_id=pattern_id,
            category=row[1],
            valid=valid_count,
            invalid=invalid_count,
            uncertain=row[4],
            total=row[5],
            precision=precision,
        )

        # Load invalid reasonings
        inv_rows = conn.execute(
            """
            SELECT fv.reasoning
            FROM finding_verifications fv
            JOIN findings fn ON fn.id = fv.finding_id
            JOIN file_analyses fa ON fn.file_analysis_id = fa.id
            WHERE fa.run_id = ? AND fn.pattern_id = ? AND fv.status = 'invalid'
            """,
            [run_id, pattern_id],
        ).fetchall()
        stats.invalid_reasonings = [r[0] for r in inv_rows if r[0]]

        # Load valid reasonings (for context)
        val_rows = conn.execute(
            """
            SELECT fv.reasoning
            FROM finding_verifications fv
            JOIN findings fn ON fn.id = fv.finding_id
            JOIN file_analyses fa ON fn.file_analysis_id = fa.id
            WHERE fa.run_id = ? AND fn.pattern_id = ? AND fv.status = 'valid'
            """,
            [run_id, pattern_id],
        ).fetchall()
        stats.valid_reasonings = [r[0] for r in val_rows if r[0]]

        # Load detection question from pattern.toml
        stats.detection_question = _load_detection_question(pattern_id, stats.category)

        result.append(stats)

    return result


def _load_detection_question(pattern_id: str, category: str) -> str:
    """Load detection question from pattern.toml.

    Args:
        pattern_id: Pattern ID (e.g., 'ml-009').
        category: Pattern category (e.g., 'ai-training').

    Returns:
        Detection question text, or empty string if not found.
    """
    patterns_dir = Path(__file__).parent.parent / "src" / "scicode_lint" / "patterns"
    category_dir = patterns_dir / category

    if not category_dir.exists():
        return ""

    # Find pattern directory (format: <id>-<name>)
    for d in category_dir.iterdir():
        if d.is_dir() and d.name.startswith(f"{pattern_id}-"):
            toml_path = d / "pattern.toml"
            if toml_path.exists():
                return _extract_question_from_toml(toml_path)

    return ""


def _extract_question_from_toml(toml_path: Path) -> str:
    """Extract detection question from pattern.toml file.

    Args:
        toml_path: Path to pattern.toml.

    Returns:
        Detection question text.
    """
    content = toml_path.read_text()

    # Find question = \"\"\" ... \"\"\" block
    start_marker = 'question = """'
    end_marker = '"""'

    start = content.find(start_marker)
    if start == -1:
        return ""

    start += len(start_marker)
    end = content.find(end_marker, start)
    if end == -1:
        return ""

    return content[start:end].strip()


ANALYSIS_PROMPT = """You are analyzing false positive patterns from a scientific code linter (scicode-lint).

## Pattern: {pattern_id} ({category})

**Precision:** {valid}/{total} valid ({precision:.1f}%)
- Valid findings: {valid}
- Invalid (false positive): {invalid}
- Uncertain: {uncertain}

## Current Detection Question

```
{detection_question}
```

## False Positive Reasoning (from Claude Sonnet verifier)

Each entry below explains WHY a detection was a false positive:

{invalid_reasoning_text}

## Valid Detection Reasoning (for contrast)

These cases WERE correctly detected:

{valid_reasoning_text}

## Your Task

Analyze the false positives and produce:

1. **COMMON THEMES**: Group the false positives into 2-5 distinct failure modes. For each theme:
   - Name it concisely (e.g., "Fires on pre-split data loaded from separate directories")
   - Count how many FPs match this theme
   - Quote a representative reasoning snippet

2. **ROOT CAUSE**: What is fundamentally wrong with the detection question that causes these FPs? Is it:
   - Too broad (fires on any code mentioning X)?
   - Missing exclusion criteria (doesn't check for safe patterns)?
   - Wrong abstraction level (flags call sites instead of implementations)?
   - Something else?

3. **SPECIFIC RECOMMENDATIONS**: Concrete changes to the detection question text. For each recommendation:
   - What exclusion/check to add
   - Example wording to add to the "NO" / "correct patterns" section
   - What the detection question should say differently

4. **CONTRAST WITH VALID DETECTIONS**: What distinguishes true positives from false positives? This helps define the boundary the detection question needs to draw.

Be specific and actionable. The detection question will be read by Qwen3-8B (a capable but small model), so recommendations must be clear and explicit — the model cannot infer subtle distinctions.
"""


async def analyze_pattern(
    stats: PatternStats,
    cli: ClaudeCLI,
    timeout: int,
) -> PatternAnalysis:
    """Analyze false positives for a single pattern using Claude.

    Args:
        stats: Pattern statistics with reasoning texts.
        cli: Claude CLI wrapper.
        timeout: Timeout in seconds.

    Returns:
        PatternAnalysis with Claude's analysis.
    """
    result = PatternAnalysis(
        pattern_id=stats.pattern_id,
        category=stats.category,
        stats=stats,
    )

    # Format reasoning texts
    invalid_lines = []
    for i, reasoning in enumerate(stats.invalid_reasonings, 1):
        # Truncate very long reasonings
        text = reasoning[:800] if len(reasoning) > 800 else reasoning
        invalid_lines.append(f"### FP #{i}\n{text}")

    valid_lines = []
    for i, reasoning in enumerate(stats.valid_reasonings, 1):
        text = reasoning[:500] if len(reasoning) > 500 else reasoning
        valid_lines.append(f"### TP #{i}\n{text}")

    invalid_text = "\n\n".join(invalid_lines) if invalid_lines else "(none)"
    valid_text = "\n\n".join(valid_lines) if valid_lines else "(none)"

    prompt = ANALYSIS_PROMPT.format(
        pattern_id=stats.pattern_id,
        category=stats.category,
        valid=stats.valid,
        invalid=stats.invalid,
        uncertain=stats.uncertain,
        total=stats.total,
        precision=stats.precision,
        detection_question=stats.detection_question or "(detection question not found)",
        invalid_reasoning_text=invalid_text,
        valid_reasoning_text=valid_text,
    )

    try:
        cli_result = await cli.arun(
            prompt,
            disallowed_tools=DISALLOWED_TOOLS_ALL,
            timeout=timeout,
        )
        result.analysis = cli_result.stdout.strip()
    except ClaudeCLIError as e:
        result.error = str(e)
        logger.error(f"Error analyzing {stats.pattern_id}: {e}")

    return result


async def analyze_all_patterns(
    pattern_stats: list[PatternStats],
    model: str,
    timeout: int,
    quiet: bool = False,
) -> list[PatternAnalysis]:
    """Analyze false positives for all patterns in parallel.

    Rate limiting is handled globally by ClaudeCLI.

    Args:
        pattern_stats: List of pattern statistics.
        model: Claude model to use.
        timeout: Timeout per pattern.
        quiet: Suppress progress output.

    Returns:
        List of PatternAnalysis results.
    """
    cli = ClaudeCLI(model=model, effort="high")

    tasks = []
    for stats in pattern_stats:
        task = asyncio.create_task(analyze_pattern(stats, cli, timeout))
        tasks.append(task)

    results = []
    total = len(pattern_stats)

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed = len(results)

        if not quiet:
            status = "OK" if not result.error else "ERR"
            logger.info(
                f"[{completed}/{total}] {status} {result.pattern_id} "
                f"({result.stats.invalid} FPs, {result.stats.precision:.0f}% precision)"
            )

    # Sort by invalid count descending
    results.sort(key=lambda r: r.stats.invalid, reverse=True)
    return results


def generate_report(
    analyses: list[PatternAnalysis],
    run_id: int,
    run_date: str,
) -> str:
    """Generate markdown summary report.

    Args:
        analyses: List of pattern analyses.
        run_id: Analysis run ID.
        run_date: Run date string.

    Returns:
        Markdown report string.
    """
    lines = []

    # Header
    lines.append("# Error Analysis Report")
    lines.append("")
    lines.append(f"Analysis of false positives from run {run_id} ({run_date}).")
    lines.append(f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M')} UTC")
    lines.append("")

    # Overall stats
    total_valid = sum(a.stats.valid for a in analyses)
    total_invalid = sum(a.stats.invalid for a in analyses)
    total_all = total_valid + total_invalid
    overall_precision = total_valid / total_all * 100 if total_all > 0 else 0

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Patterns analyzed:** {len(analyses)}")
    lines.append(f"- **Total false positives:** {total_invalid}")
    lines.append(f"- **Overall precision (analyzed patterns):** {overall_precision:.1f}%")
    lines.append("")

    # Pattern table
    lines.append("## Patterns by False Positive Count")
    lines.append("")
    lines.append("| Pattern | Category | Valid | Invalid | Precision | Status |")
    lines.append("|---------|----------|-------|---------|-----------|--------|")
    for a in analyses:
        s = a.stats
        status = "analyzed" if not a.error else "error"
        lines.append(
            f"| {s.pattern_id} | {s.category} | {s.valid} | {s.invalid} "
            f"| {s.precision:.1f}% | {status} |"
        )
    lines.append("")

    # Per-pattern analysis
    lines.append("## Per-Pattern Analysis")
    lines.append("")

    for a in analyses:
        lines.append(f"### {a.pattern_id} ({a.category})")
        lines.append("")
        lines.append(
            f"**Stats:** {a.stats.valid} valid, {a.stats.invalid} invalid, "
            f"{a.stats.precision:.1f}% precision"
        )
        lines.append("")

        if a.error:
            lines.append(f"**Error:** {a.error}")
        elif a.analysis:
            lines.append(a.analysis)
        lines.append("")
        lines.append("---")
        lines.append("")

    # Footer
    lines.append(
        f"*Error analysis conducted: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M')} UTC "
        f"using Claude {DEFAULT_MODEL}*"
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze false positives from verification results"
    )
    parser.add_argument(
        "--run-id",
        type=int,
        help="Analysis run ID (default: latest completed with verifications)",
    )
    parser.add_argument(
        "--min-invalid",
        type=int,
        default=DEFAULT_MIN_INVALID,
        help=f"Minimum invalid findings to analyze a pattern (default: {DEFAULT_MIN_INVALID})",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        help="Specific pattern IDs to analyze (e.g., perf-004 par-005)",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout per pattern in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"Claude model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path (default: reports/error_analysis/ERROR_ANALYSIS_<date>.md)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    # Connect to database
    db_path = get_db_path()
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise SystemExit(1)

    conn = init_db(db_path)

    # Get run_id
    run_id = args.run_id
    if run_id is None:
        row = conn.execute(
            "SELECT id FROM analysis_runs WHERE status = 'completed' "
            "ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            logger.error("No completed analysis runs found")
            conn.close()
            raise SystemExit(1)
        run_id = row[0]

    # Get run date
    row = conn.execute("SELECT started_at FROM analysis_runs WHERE id = ?", [run_id]).fetchone()
    run_date = row[0] if row else "unknown"

    # Check that verifications exist for this run
    ver_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM finding_verifications fv
        JOIN findings fn ON fn.id = fv.finding_id
        JOIN file_analyses fa ON fn.file_analysis_id = fa.id
        WHERE fa.run_id = ?
        """,
        [run_id],
    ).fetchone()[0]

    if ver_count == 0:
        logger.error(f"No verifications found for run {run_id}. Run verify_findings first.")
        conn.close()
        raise SystemExit(1)

    # Load pattern stats
    pattern_stats = load_pattern_stats(
        conn, run_id, min_invalid=args.min_invalid, patterns=args.patterns
    )

    if not pattern_stats:
        logger.info(f"No patterns with >= {args.min_invalid} invalid findings in run {run_id}")
        conn.close()
        raise SystemExit(0)

    total_fps = sum(s.invalid for s in pattern_stats)
    logger.info(
        f"Analyzing {len(pattern_stats)} patterns with {total_fps} total false positives "
        f"(run {run_id}, model: {args.model})"
    )

    # Run analysis
    analyses = asyncio.run(
        analyze_all_patterns(
            pattern_stats,
            model=args.model,
            timeout=args.timeout,
            quiet=args.quiet,
        )
    )

    # Generate summary report
    report = generate_report(analyses, run_id, run_date)

    # Determine output path
    output_dir = REPORTS_DIR / "error_analysis"
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.output:
        summary_path = args.output
    else:
        date_str = datetime.now(UTC).strftime("%Y-%m-%d_%H%M")
        summary_path = output_dir / f"ERROR_ANALYSIS_{date_str}.md"

    summary_path.write_text(report)
    logger.info(f"Summary report saved to {summary_path}")

    # Save per-pattern analysis files
    for a in analyses:
        if a.analysis:
            pattern_path = output_dir / f"{a.pattern_id}.md"
            pattern_content = (
                f"# {a.pattern_id} ({a.category}) — Error Analysis\n\n"
                f"**Run:** {run_id} | **Date:** {run_date}\n"
                f"**Stats:** {a.stats.valid} valid, {a.stats.invalid} invalid, "
                f"{a.stats.precision:.1f}% precision\n\n"
                f"## Detection Question\n\n```\n{a.stats.detection_question}\n```\n\n"
                f"## Analysis\n\n{a.analysis}\n"
            )
            pattern_path.write_text(pattern_content)
            logger.debug(f"Pattern analysis saved to {pattern_path}")

    # Print summary
    logger.info(f"Analysis complete: {len(analyses)} patterns analyzed")
    for a in analyses:
        status = "OK" if not a.error else "ERR"
        logger.info(
            f"  {status} {a.pattern_id}: {a.stats.invalid} FPs, {a.stats.precision:.0f}% precision"
        )

    conn.close()


if __name__ == "__main__":
    main()
