"""Verify findings using Claude CLI (Opus model).

Evaluates whether scicode-lint findings are real issues by having Claude
review the actual code in context.

Usage:
    python -m real_world_demo.verify_findings                    # Verify all findings
    python -m real_world_demo.verify_findings --limit 10         # Verify first 10
    python -m real_world_demo.verify_findings --category data-leakage  # By category
    python -m real_world_demo.verify_findings --run-id 1         # Specific run

Output:
    reports/VERIFICATION_REPORT_<date>.md

Requires: Claude CLI installed and authenticated (claude login).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from .config import COLLECTED_DIR, LEAKAGE_PAPER_COLLECTED_DIR, REPORTS_DIR
from .database import get_db_path, get_run_stats, init_db, save_verification
from .utils import extract_code_from_notebook

if TYPE_CHECKING:
    from asyncio.subprocess import Process

# Defaults
DEFAULT_PARALLEL = 3
DEFAULT_MODEL = "opus"
DEFAULT_TIMEOUT = 120


@dataclass
class Finding:
    """A finding to verify."""

    finding_id: int
    pattern_id: str
    category: str
    severity: str
    confidence: float
    issue: str
    explanation: str
    snippet: str
    lines: str  # JSON array
    file_path: str
    original_path: str
    repo_name: str
    repo_url: str
    domain: str


@dataclass
class VerificationResult:
    """Result of finding verification."""

    finding: Finding
    status: str = "pending"  # pending, valid, invalid, uncertain, error
    reasoning: str = ""
    error: str = ""


def extract_reasoning(raw_output: str) -> str:
    """Extract reasoning from Claude output, handling various formats.

    Expected format: verdict on line 1, reasoning on subsequent lines.
    But handles cases where Claude puts reasoning on the same line.

    Args:
        raw_output: Raw Claude output.

    Returns:
        Extracted reasoning text.
    """
    if not raw_output:
        return ""

    lines = raw_output.strip().split("\n")

    # Collect reasoning from all lines
    reasoning_parts = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        if i == 0:
            # First line contains verdict - extract any reasoning after it
            # Skip standard verdict prefixes
            upper = line.upper()
            for verdict in ["VALID:", "INVALID:", "UNCERTAIN:"]:
                if upper.startswith(verdict):
                    after = line[len(verdict) :].strip()
                    # Skip if it's just the template text
                    if after and not any(
                        after.upper().startswith(t)
                        for t in [
                            "THIS IS A REAL ISSUE",
                            "THIS IS A FALSE POSITIVE",
                            "CANNOT DETERMINE",
                        ]
                    ):
                        reasoning_parts.append(after)
                    break
        else:
            # Subsequent lines are reasoning
            reasoning_parts.append(line)

    return " ".join(reasoning_parts).strip()


VERIFICATION_PROMPT = """You are reviewing a potential code issue detected by an automated linter.

**Pattern:** {pattern_id} ({category})
**Severity:** {severity}
**Confidence:** {confidence:.0%}

**Issue detected:** {issue}

**Explanation:** {explanation}

**Repository:** {repo_name} (domain: {domain})
**File:** {original_path}

**Code snippet flagged:**
```python
{snippet}
```

**Surrounding code context:**
```python
{context}
```

**Your task:** Determine if this is a REAL issue that could cause problems in practice.

Consider:
1. Is the detected pattern actually present in the code?
2. Could this realistically cause the problem described?
3. Is there surrounding context that mitigates the issue?
4. Is this a false positive?

Respond with ONE of these verdicts on the first line:
- VALID: This is a real issue that should be fixed
- INVALID: This is a false positive, the code is fine
- UNCERTAIN: Cannot determine without more context

Then explain your reasoning briefly (2-3 sentences max).
"""


def load_findings_to_verify(
    conn: sqlite3.Connection,
    run_id: int | None = None,
    category: str | None = None,
    limit: int | None = None,
    skip_verified: bool = False,
) -> list[Finding]:
    """Load findings from database for verification.

    Args:
        conn: Database connection.
        run_id: Specific run ID, or None for latest.
        category: Filter by category.
        limit: Max findings to return.
        skip_verified: Skip findings that already have verifications.

    Returns:
        List of Finding objects.
    """
    # Get run_id if not specified
    if run_id is None:
        cursor = conn.execute("SELECT id FROM analysis_runs ORDER BY started_at DESC LIMIT 1")
        row = cursor.fetchone()
        if not row:
            return []
        run_id = row[0]

    query = """
        SELECT
            fn.id,
            fn.pattern_id,
            fn.category,
            fn.severity,
            fn.confidence,
            fn.issue,
            fn.explanation,
            fn.snippet,
            fn.lines,
            f.file_path,
            f.original_path,
            r.repo_name,
            r.repo_url,
            r.domain
        FROM findings fn
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        JOIN files f ON f.id = fa.file_id
        JOIN repos r ON r.id = f.repo_id
        WHERE fa.run_id = ?
    """
    params: list[Any] = [run_id]

    if skip_verified:
        query += " AND fn.id NOT IN (SELECT finding_id FROM finding_verifications)"

    if category:
        query += " AND fn.category = ?"
        params.append(category)

    query += " ORDER BY fn.confidence DESC"

    if limit:
        query += " LIMIT ?"
        params.append(limit)

    cursor = conn.execute(query, params)
    findings = []
    for row in cursor.fetchall():
        findings.append(
            Finding(
                finding_id=row[0],
                pattern_id=row[1],
                category=row[2],
                severity=row[3],
                confidence=row[4],
                issue=row[5],
                explanation=row[6],
                snippet=row[7] or "",
                lines=row[8] or "[]",
                file_path=row[9],
                original_path=row[10] or "",
                repo_name=row[11],
                repo_url=row[12] or "",
                domain=row[13],
            )
        )
    return findings


def get_code_context(finding: Finding, context_lines: int = 30) -> str:
    """Get surrounding code context for a finding.

    Args:
        finding: Finding to get context for.
        context_lines: Number of lines before/after snippet.

    Returns:
        Code context string.
    """
    # Try to find the file in collected_code
    file_path = COLLECTED_DIR / finding.file_path
    if not file_path.exists():
        # Try leakage paper directory
        file_path = LEAKAGE_PAPER_COLLECTED_DIR / finding.file_path
    if not file_path.exists():
        # Try with just the filename part
        parts = finding.file_path.split("/")
        if len(parts) >= 2:
            file_path = COLLECTED_DIR / "files" / parts[-2] / parts[-1]

    if not file_path.exists():
        return finding.snippet or "(file not found)"

    try:
        # Handle Jupyter notebooks by extracting Python code
        if file_path.suffix == ".ipynb":
            content = extract_code_from_notebook(file_path)
        else:
            content = file_path.read_text()
        lines = content.split("\n")

        # Parse line numbers from finding
        try:
            line_nums = json.loads(finding.lines) if finding.lines else []
        except (json.JSONDecodeError, TypeError):
            line_nums = []

        if not line_nums:
            # Return first part of file if no line numbers
            return "\n".join(lines[:50])

        # Get context around the flagged lines
        start = max(0, min(line_nums) - context_lines - 1)
        end = min(len(lines), max(line_nums) + context_lines)

        context_lines_list = []
        for i, line in enumerate(lines[start:end], start=start + 1):
            marker = ">>>" if i in line_nums else "   "
            context_lines_list.append(f"{marker} {i:4d} | {line}")

        return "\n".join(context_lines_list)

    except Exception as e:
        return f"(error reading file: {e})"


async def verify_finding(
    finding: Finding,
    timeout: int,
    semaphore: asyncio.Semaphore,
    model: str,
) -> VerificationResult:
    """Verify a single finding using Claude CLI.

    Args:
        finding: Finding to verify.
        timeout: Timeout in seconds.
        semaphore: Semaphore for concurrency control.
        model: Claude model to use.

    Returns:
        VerificationResult with verdict.
    """
    result = VerificationResult(finding=finding)

    async with semaphore:
        result.status = "running"

        # Get code context
        context = get_code_context(finding)

        # Build prompt
        prompt = VERIFICATION_PROMPT.format(
            pattern_id=finding.pattern_id,
            category=finding.category,
            severity=finding.severity,
            confidence=finding.confidence,
            issue=finding.issue,
            explanation=finding.explanation,
            repo_name=finding.repo_name,
            domain=finding.domain,
            original_path=finding.original_path,
            snippet=finding.snippet,
            context=context,
        )

        try:
            proc: Process = await asyncio.create_subprocess_exec(
                "claude",
                "--model",
                model,
                "--print",
                "--disallowed-tools",
                "Task,WebSearch,WebFetch,Bash,Write,Edit,NotebookEdit,Read,Glob,Grep",
                "--",
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

            output = stdout.decode().strip() if stdout else ""
            result.reasoning = output

            # Parse verdict from first line
            first_line = output.split("\n")[0].upper() if output else ""
            if "VALID" in first_line and "INVALID" not in first_line:
                result.status = "valid"
            elif "INVALID" in first_line:
                result.status = "invalid"
            elif "UNCERTAIN" in first_line:
                result.status = "uncertain"
            else:
                result.status = "uncertain"

        except FileNotFoundError:
            result.status = "error"
            result.error = "Claude CLI not found. Run: claude login"
        except Exception as e:
            result.status = "error"
            result.error = str(e)

    return result


async def verify_all_findings(
    findings: list[Finding],
    parallel: int,
    timeout: int,
    model: str,
    quiet: bool = False,
    conn: sqlite3.Connection | None = None,
) -> list[VerificationResult]:
    """Verify multiple findings in parallel.

    Args:
        findings: List of findings to verify.
        parallel: Max concurrent verifications.
        timeout: Timeout per finding.
        model: Claude model to use.
        quiet: Suppress progress output.
        conn: Database connection for incremental saves. If provided,
            results are saved immediately after each verification completes.

    Returns:
        List of VerificationResult.
    """
    semaphore = asyncio.Semaphore(parallel)

    tasks = []
    for finding in findings:
        task = asyncio.create_task(verify_finding(finding, timeout, semaphore, model))
        tasks.append(task)

    results = []
    total = len(findings)

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed = len(results)

        # Save incrementally to database (prevents data loss on interruption)
        if conn is not None and result.status not in ("pending", "running"):
            # For errors, store the error message in reasoning field
            reasoning_to_save = result.error if result.status == "error" else result.reasoning
            save_verification(
                conn,
                finding_id=result.finding.finding_id,
                status=result.status,
                reasoning=reasoning_to_save,
                model=model,
            )

        icon = {
            "valid": "Y",
            "invalid": "N",
            "uncertain": "?",
            "error": "X",
        }.get(result.status, "?")

        if not quiet:
            f = result.finding
            logger.info(f"[{completed}/{total}] {icon} {f.pattern_id} in {f.repo_name}")

    # Sort by status then pattern
    order = {"valid": 0, "invalid": 1, "uncertain": 2, "error": 3}
    results.sort(key=lambda r: (order.get(r.status, 4), r.finding.pattern_id))
    return results


def update_findings_report(
    findings_report_path: Path,
    results: list[VerificationResult],
    analysis_date: datetime,
) -> str:
    """Update existing findings report with verification marks.

    Args:
        findings_report_path: Path to existing FINDINGS_REPORT.md.
        results: Verification results.
        analysis_date: When analysis was conducted.

    Returns:
        Updated markdown content.
    """
    if not findings_report_path.exists():
        return ""

    content = findings_report_path.read_text()
    lines = content.split("\n")

    # Build lookup: pattern_id + repo_name -> result
    result_lookup: dict[str, VerificationResult] = {}
    for r in results:
        key = f"{r.finding.pattern_id}|{r.finding.repo_name}"
        result_lookup[key] = r

    # Calculate stats
    valid = sum(1 for r in results if r.status == "valid")
    invalid = sum(1 for r in results if r.status == "invalid")
    uncertain = sum(1 for r in results if r.status == "uncertain")
    total = valid + invalid + uncertain
    precision = valid / total * 100 if total > 0 else 0

    # Build verification banner
    verification_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    banner = [
        "",
        f"> **Verification (Claude Opus, {verification_date}):** "
        f"{valid}/{total} findings confirmed as real issues ({precision:.0f}% precision)",
        "",
    ]

    # Find where to insert banner (after intro paragraph, before ## Analysis Summary)
    new_lines = []
    banner_inserted = False
    i = 0

    while i < len(lines):
        line = lines[i]

        # Insert banner before Analysis Summary
        if line.startswith("## Analysis Summary") and not banner_inserted:
            new_lines.extend(banner)
            banner_inserted = True

        # Check if this is a finding header: **pattern-id** (severity, confidence)
        if line.startswith("**") and "(" in line and "confidence" in line.lower():
            # Extract pattern_id from **pattern-id**
            pattern_match = line.split("**")[1] if "**" in line else ""

            # Look ahead to find repo_name in the next few lines
            repo_name = ""
            for j in range(i + 1, min(i + 6, len(lines))):
                if "in [" in lines[j] and "__" in lines[j]:
                    # Extract repo_name from: in [owner__repo](url)
                    start = lines[j].find("in [") + 4
                    end = lines[j].find("]", start)
                    if start > 3 and end > start:
                        repo_name = lines[j][start:end]
                        break

            # Look up verification result
            key = f"{pattern_match}|{repo_name}"
            verification = result_lookup.get(key)

            if verification:
                # Add verification mark to the header
                mark = {
                    "valid": "[VALID]",
                    "invalid": "[FALSE POSITIVE]",
                    "uncertain": "[?]",
                }.get(verification.status, "")

                # Reconstruct header with mark
                if mark:
                    # Insert mark after pattern_id: **pt-001** [VALID] (high, 95%)
                    parts = line.split("**")
                    if len(parts) >= 3:
                        new_line = f"**{parts[1]}** {mark} {parts[2]}"
                        new_lines.append(new_line)
                        i += 1
                        continue

        new_lines.append(line)
        i += 1

    # Update footer with verification info
    result = "\n".join(new_lines)
    old_footer = f"*Analysis conducted: {analysis_date.strftime('%Y-%m-%d')}"
    if old_footer in result:
        new_footer = (
            f"*Analysis conducted: {analysis_date.strftime('%Y-%m-%d')} | "
            f"Verified: {verification_date} ({precision:.0f}% precision)*"
        )
        result = result.replace(
            old_footer + result.split(old_footer)[1].split("*")[0] + "*",
            new_footer,
        )

    return result


def generate_verification_report(
    results: list[VerificationResult],
    analysis_date: datetime,
) -> str:
    """Generate markdown verification report.

    Args:
        results: List of verification results.
        analysis_date: When the original analysis was conducted.

    Returns:
        Markdown report string.
    """
    lines = []

    # Header
    lines.append("# Findings Verification Report")
    lines.append("")
    lines.append("Claude (Opus) evaluation of whether detected issues are real problems.")
    lines.append("")

    # Dates
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Analysis Date:** {analysis_date.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **Verification Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **Findings Reviewed:** {len(results)}")
    lines.append("")

    # Stats
    valid = sum(1 for r in results if r.status == "valid")
    invalid = sum(1 for r in results if r.status == "invalid")
    uncertain = sum(1 for r in results if r.status == "uncertain")
    errors = sum(1 for r in results if r.status == "error")

    total_evaluated = valid + invalid + uncertain
    precision = valid / total_evaluated * 100 if total_evaluated > 0 else 0

    lines.append("| Verdict | Count | Percentage |")
    lines.append("|---------|-------|------------|")
    lines.append(f"| Valid (real issues) | {valid} | {valid / len(results) * 100:.1f}% |")
    lines.append(f"| Invalid (false positives) | {invalid} | {invalid / len(results) * 100:.1f}% |")
    lines.append(f"| Uncertain | {uncertain} | {uncertain / len(results) * 100:.1f}% |")
    if errors:
        lines.append(f"| Errors | {errors} | {errors / len(results) * 100:.1f}% |")
    lines.append("")
    lines.append(f"**Estimated Precision:** {precision:.1f}% (valid / evaluated)")
    lines.append("")

    # By category
    categories: dict[str, dict[str, int]] = {}
    for r in results:
        cat = r.finding.category
        if cat not in categories:
            categories[cat] = {"valid": 0, "invalid": 0, "uncertain": 0, "error": 0}
        categories[cat][r.status] = categories[cat].get(r.status, 0) + 1

    lines.append("## Results by Category")
    lines.append("")
    lines.append("| Category | Valid | Invalid | Uncertain | Precision |")
    lines.append("|----------|-------|---------|-----------|-----------|")
    for cat, counts in sorted(categories.items()):
        v, inv, unc = counts["valid"], counts["invalid"], counts["uncertain"]
        total = v + inv + unc
        prec = v / total * 100 if total > 0 else 0
        lines.append(f"| {cat} | {v} | {inv} | {unc} | {prec:.0f}% |")
    lines.append("")

    # Detailed results - Valid findings
    valid_results = [r for r in results if r.status == "valid"]
    if valid_results:
        lines.append("## Valid Findings (Real Issues)")
        lines.append("")
        for r in valid_results:
            f = r.finding
            lines.append(f"### {f.pattern_id} - {f.repo_name}")
            lines.append("")
            lines.append(f"- **File:** {f.original_path}")
            lines.append(f"- **Issue:** {f.issue}")
            lines.append(f"- **Severity:** {f.severity}")
            lines.append(f"- **Domain:** {f.domain}")
            lines.append("")
            if f.explanation:
                lines.append(f"**Detector reasoning:** {f.explanation[:500]}")
                lines.append("")
            if f.snippet:
                lines.append("```python")
                lines.append(f.snippet.strip())
                lines.append("```")
                lines.append("")
            verdict_line = r.reasoning.split(chr(10))[0] if r.reasoning else ""
            lines.append(f"**Verification:** {verdict_line}")
            lines.append("")

    # Invalid findings (false positives)
    invalid_results = [r for r in results if r.status == "invalid"]
    if invalid_results:
        lines.append("## Invalid Findings (False Positives)")
        lines.append("")
        for r in invalid_results:
            f = r.finding
            lines.append(f"### {f.pattern_id} - {f.repo_name}")
            lines.append("")
            lines.append(f"- **File:** {f.original_path}")
            lines.append(f"- **Issue:** {f.issue}")
            lines.append(f"- **Severity:** {f.severity}")
            lines.append(f"- **Domain:** {f.domain}")
            lines.append("")
            if f.explanation:
                lines.append(f"**Detector reasoning:** {f.explanation[:500]}")
                lines.append("")
            if f.snippet:
                lines.append("```python")
                lines.append(f.snippet.strip())
                lines.append("```")
                lines.append("")
            # Get reasoning - may be on same line as verdict or subsequent lines
            reasoning = extract_reasoning(r.reasoning)
            if not reasoning:
                reasoning = "(No explanation provided by reviewer)"
            lines.append(f"**Why false positive:** {reasoning[:500]}")
            lines.append("")

    # Uncertain findings
    uncertain_results = [r for r in results if r.status == "uncertain"]
    if uncertain_results:
        lines.append("## Uncertain Findings (Needs Review)")
        lines.append("")
        for r in uncertain_results:
            f = r.finding
            lines.append(f"### {f.pattern_id} - {f.repo_name}")
            lines.append("")
            lines.append(f"- **File:** {f.original_path}")
            lines.append(f"- **Issue:** {f.issue}")
            lines.append(f"- **Severity:** {f.severity}")
            lines.append(f"- **Domain:** {f.domain}")
            lines.append("")
            if f.explanation:
                lines.append(f"**Detector reasoning:** {f.explanation[:500]}")
                lines.append("")
            if f.snippet:
                lines.append("```python")
                lines.append(f.snippet.strip())
                lines.append("```")
                lines.append("")
            reasoning = extract_reasoning(r.reasoning)
            if reasoning:
                lines.append(f"**Reviewer notes:** {reasoning[:500]}")
                lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append(
        f"*Verification conducted: {datetime.now().strftime('%Y-%m-%d %H:%M')} using Claude Opus*"
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Verify findings using Claude CLI")
    parser.add_argument(
        "--run-id",
        type=int,
        help="Analysis run ID (default: latest)",
    )
    parser.add_argument(
        "--category",
        help="Filter by category",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Max findings to verify",
    )
    parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        default=DEFAULT_PARALLEL,
        help=f"Max concurrent verifications (default: {DEFAULT_PARALLEL})",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout per finding in seconds (default: {DEFAULT_TIMEOUT})",
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
        help="Output file path",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--skip-verified",
        action="store_true",
        help="Skip findings that already have verifications (for resuming)",
    )
    args = parser.parse_args()

    # Connect to database
    db_path = get_db_path()
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise SystemExit(1)

    conn = init_db(db_path)

    # Get analysis stats for date
    stats = get_run_stats(conn, args.run_id)
    if stats.run_id == 0:
        logger.error("No analysis runs found")
        conn.close()
        raise SystemExit(1)

    # Load findings
    findings = load_findings_to_verify(
        conn,
        run_id=args.run_id,
        category=args.category,
        limit=args.limit,
        skip_verified=args.skip_verified,
    )

    if not findings:
        if args.skip_verified:
            logger.info("No unverified findings remaining")
        else:
            logger.error("No findings to verify")
        conn.close()
        raise SystemExit(0 if args.skip_verified else 1)

    logger.info(f"Verifying {len(findings)} findings with Claude {args.model}")
    logger.info(f"Parallel: {args.parallel}, Timeout: {args.timeout}s")

    # Run verification (saves results incrementally to database)
    results = asyncio.run(
        verify_all_findings(
            findings,
            parallel=args.parallel,
            timeout=args.timeout,
            model=args.model,
            quiet=args.quiet,
            conn=conn,
        )
    )
    logger.info(f"Verification complete, {len(results)} results saved to database")

    # Update existing findings report with verification marks
    # Include time to avoid overwriting same-day reports
    date_str = stats.run_date.strftime("%Y-%m-%d_%H%M")
    findings_report_path = REPORTS_DIR / f"FINDINGS_REPORT_{date_str}.md"

    if findings_report_path.exists():
        logger.info(f"Updating findings report: {findings_report_path}")
        updated_content = update_findings_report(findings_report_path, results, stats.run_date)
        if updated_content:
            findings_report_path.write_text(updated_content)
            logger.info("Findings report updated with verification marks")
    else:
        logger.warning(f"Findings report not found: {findings_report_path}")

    # Also generate standalone verification report
    report = generate_verification_report(results, stats.run_date)
    if args.output:
        output_path = args.output
    else:
        output_path = REPORTS_DIR / f"VERIFICATION_REPORT_{date_str}.md"

    output_path.parent.mkdir(exist_ok=True, parents=True)
    output_path.write_text(report)
    logger.info(f"Verification report saved to {output_path}")

    # Summary
    valid = sum(1 for r in results if r.status == "valid")
    invalid = sum(1 for r in results if r.status == "invalid")
    total = valid + invalid + sum(1 for r in results if r.status == "uncertain")
    precision = valid / total * 100 if total > 0 else 0

    logger.info(f"Results: {valid} valid, {invalid} invalid, precision: {precision:.1f}%")

    conn.close()


if __name__ == "__main__":
    main()
