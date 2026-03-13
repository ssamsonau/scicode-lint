"""Generate markdown report from SQLite analysis database.

Creates a committable report with statistics and example findings
from real-world scientific ML code analysis.
"""

import argparse
import json
import sqlite3
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from .config import REPORTS_DIR
from .database import get_db_path, get_run_stats, get_verification_stats, init_db, list_runs


class DistributionStats:
    """Statistics for a distribution of values."""

    def __init__(self, values: list[int]) -> None:
        self.count = len(values)
        if values:
            self.min = min(values)
            self.max = max(values)
            self.mean = statistics.mean(values)
            self.stdev = statistics.stdev(values) if len(values) > 1 else 0.0
            self.median = statistics.median(values)
        else:
            self.min = self.max = 0
            self.mean = self.stdev = self.median = 0.0


def get_findings_distribution(
    conn: sqlite3.Connection, run_id: int, *, verified_only: bool = False
) -> dict[str, DistributionStats]:
    """Get distribution stats for findings per paper and per severity.

    Args:
        conn: Database connection.
        run_id: Analysis run ID.
        verified_only: If True, only include findings with completed verification.

    Returns:
        Dict with 'per_paper' and 'per_paper_by_severity' distribution stats.
    """
    result: dict[str, DistributionStats] = {}

    # Build verification join clause
    verification_join = (
        "INNER JOIN finding_verifications fv ON fv.finding_id = fn.id AND fv.status != 'error'"
        if verified_only
        else ""
    )

    # Findings per paper (total)
    cursor = conn.execute(
        f"""
        SELECT r.paper_id, COUNT(fn.id) as finding_count
        FROM findings fn
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        JOIN files f ON f.id = fa.file_id
        JOIN repos r ON r.id = f.repo_id
        {verification_join}
        WHERE fa.run_id = ? AND r.paper_id IS NOT NULL
        GROUP BY r.paper_id
        """,
        (run_id,),
    )
    counts = [row[1] for row in cursor.fetchall()]
    result["per_paper"] = DistributionStats(counts)

    # Findings per paper by severity
    for severity in ["critical", "high", "medium", "low"]:
        cursor = conn.execute(
            f"""
            SELECT r.paper_id, COUNT(fn.id) as finding_count
            FROM findings fn
            JOIN file_analyses fa ON fa.id = fn.file_analysis_id
            JOIN files f ON f.id = fa.file_id
            JOIN repos r ON r.id = f.repo_id
            {verification_join}
            WHERE fa.run_id = ? AND r.paper_id IS NOT NULL AND fn.severity = ?
            GROUP BY r.paper_id
            """,
            (run_id, severity),
        )
        counts = [row[1] for row in cursor.fetchall()]
        result[severity] = DistributionStats(counts)

    return result


def get_papers_by_severity(
    conn: sqlite3.Connection, run_id: int, *, verified_only: bool = False
) -> dict[str, int]:
    """Get paper counts by severity (papers with at least one finding of that severity).

    Args:
        conn: Database connection.
        run_id: Analysis run ID.
        verified_only: If True, only include findings with completed verification.

    Returns:
        Dict mapping severity to paper count. A paper with both critical and medium
        findings will be counted in both categories.
    """
    verification_join = (
        "INNER JOIN finding_verifications fv ON fv.finding_id = fn.id AND fv.status != 'error'"
        if verified_only
        else ""
    )

    # Count distinct papers per severity level
    cursor = conn.execute(
        f"""
        SELECT
            fn.severity,
            COUNT(DISTINCT r.paper_id) as paper_count
        FROM findings fn
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        JOIN files f ON f.id = fa.file_id
        JOIN repos r ON r.id = f.repo_id
        {verification_join}
        WHERE fa.run_id = ? AND r.paper_id IS NOT NULL
        GROUP BY fn.severity
        """,
        (run_id,),
    )

    counts: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for row in cursor.fetchall():
        sev = row[0] or "low"
        if sev in counts:
            counts[sev] = row[1]

    return counts


def get_verification_by_severity(
    conn: sqlite3.Connection, run_id: int
) -> dict[str, dict[str, int]]:
    """Get verification stats broken down by severity.

    Args:
        conn: Database connection.
        run_id: Analysis run ID.

    Returns:
        Dict with severities as keys, each containing {valid, invalid, uncertain, pending}.
    """
    # Get all findings with their verification status (if any)
    cursor = conn.execute(
        """
        SELECT
            fn.severity,
            fv.status as verification_status,
            COUNT(*) as count
        FROM findings fn
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        LEFT JOIN finding_verifications fv ON fv.finding_id = fn.id
        WHERE fa.run_id = ?
        GROUP BY fn.severity, fv.status
        """,
        (run_id,),
    )

    result: dict[str, dict[str, int]] = {}
    for sev in ["critical", "high", "medium", "low"]:
        result[sev] = {"valid": 0, "invalid": 0, "uncertain": 0, "pending": 0, "total": 0}

    for row in cursor.fetchall():
        sev = row[0] or "low"
        status = row[1]  # None if not verified
        count = row[2]

        if sev not in result:
            result[sev] = {"valid": 0, "invalid": 0, "uncertain": 0, "pending": 0, "total": 0}

        result[sev]["total"] += count
        if status is None:
            result[sev]["pending"] += count
        elif status in result[sev]:
            result[sev][status] += count

    return result


def get_example_findings(
    conn: sqlite3.Connection,
    run_id: int,
    limit_per_category: int = 3,
    *,
    verified_only: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Get example findings grouped by category.

    Args:
        conn: Database connection.
        run_id: Analysis run ID.
        limit_per_category: Max findings per category.
        verified_only: If True, only include findings with completed verification.

    Returns:
        Dict mapping category to list of finding dicts.
    """
    verification_join = (
        "INNER JOIN finding_verifications fv ON fv.finding_id = fn.id AND fv.status != 'error'"
        if verified_only
        else ""
    )

    cursor = conn.execute(
        f"""
        SELECT DISTINCT fn.category
        FROM findings fn
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        {verification_join}
        WHERE fa.run_id = ?
        ORDER BY fn.category
        """,
        (run_id,),
    )
    categories = [row[0] for row in cursor.fetchall()]

    examples: dict[str, list[dict[str, Any]]] = {}
    for category in categories:
        cursor = conn.execute(
            f"""
            SELECT
                fn.id,
                fn.pattern_id,
                fn.severity,
                fn.confidence,
                fn.issue,
                fn.explanation,
                fn.suggestion,
                fn.snippet,
                fn.lines,
                f.file_path,
                f.original_path,
                r.repo_name,
                r.repo_url,
                r.domain,
                p.title as paper_title,
                p.arxiv_id,
                p.authors as paper_authors
            FROM findings fn
            JOIN file_analyses fa ON fa.id = fn.file_analysis_id
            JOIN files f ON f.id = fa.file_id
            JOIN repos r ON r.id = f.repo_id
            LEFT JOIN papers p ON p.id = r.paper_id
            {verification_join}
            WHERE fa.run_id = ? AND fn.category = ?
            ORDER BY fn.confidence DESC
            LIMIT ?
            """,
            (run_id, category, limit_per_category),
        )
        examples[category] = [
            {
                "finding_id": row[0],
                "pattern_id": row[1],
                "severity": row[2],
                "confidence": row[3],
                "issue": row[4],
                "explanation": row[5],
                "suggestion": row[6],
                "snippet": row[7],
                "lines": row[8],
                "file_path": row[9],
                "original_path": row[10],
                "repo_name": row[11],
                "repo_url": row[12],
                "domain": row[13],
                "paper_title": row[14],
                "arxiv_id": row[15],
                "paper_authors": row[16],
            }
            for row in cursor.fetchall()
        ]

    return examples


def get_top_patterns(
    conn: sqlite3.Connection, run_id: int, limit: int = 10, *, verified_only: bool = False
) -> list[dict[str, Any]]:
    """Get most frequent patterns.

    Args:
        conn: Database connection.
        run_id: Analysis run ID.
        limit: Max patterns to return.
        verified_only: If True, only include findings with completed verification.

    Returns:
        List of pattern stats dicts.
    """
    verification_join = (
        "INNER JOIN finding_verifications fv ON fv.finding_id = fn.id AND fv.status != 'error'"
        if verified_only
        else ""
    )

    cursor = conn.execute(
        f"""
        SELECT
            fn.pattern_id,
            fn.category,
            COUNT(*) as count,
            COUNT(DISTINCT fa.file_id) as unique_files,
            COUNT(DISTINCT f.repo_id) as unique_repos,
            AVG(fn.confidence) as avg_confidence
        FROM findings fn
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        JOIN files f ON f.id = fa.file_id
        {verification_join}
        WHERE fa.run_id = ?
        GROUP BY fn.pattern_id
        ORDER BY count DESC
        LIMIT ?
        """,
        (run_id, limit),
    )
    return [
        {
            "pattern_id": row[0],
            "category": row[1],
            "count": row[2],
            "unique_files": row[3],
            "unique_repos": row[4],
            "avg_confidence": row[5],
        }
        for row in cursor.fetchall()
    ]


def generate_markdown_report(
    conn: sqlite3.Connection, run_id: int | None = None, *, verified_only: bool = False
) -> str:
    """Generate markdown report for an analysis run.

    Args:
        conn: Database connection.
        run_id: Specific run ID, or None for latest.
        verified_only: If True, only include findings with completed verification.

    Returns:
        Markdown report string.
    """
    stats = get_run_stats(conn, run_id)
    if stats.run_id == 0:
        return "# No Analysis Data\n\nNo analysis runs found in database.\n"

    run_id = stats.run_id
    examples = get_example_findings(conn, run_id, verified_only=verified_only)
    top_patterns = get_top_patterns(conn, run_id, verified_only=verified_only)
    papers_by_severity = get_papers_by_severity(conn, run_id, verified_only=verified_only)
    verification_by_severity = get_verification_by_severity(conn, run_id)
    verification_stats = get_verification_stats(conn, run_id)
    findings_distribution = get_findings_distribution(conn, run_id, verified_only=verified_only)

    lines = []

    # Header with data source-specific description
    title = "# Real-World Scientific ML Code Analysis Report"
    if verified_only:
        title += " (Verified Findings Only)"
    lines.append(title)
    lines.append("")
    if stats.data_source == "leakage_paper":
        lines.append(
            "Analysis of 100 Python files with **ground truth labels** for data leakage issues, "
            "from Yang et al. ASE'22 study. scicode-lint findings are compared against "
            "human-verified labels to measure detection accuracy."
        )
    else:
        lines.append(
            "Analysis of Python code from **AI applications to scientific domains**. "
            "Papers sourced from PapersWithCode, filtered to include only scientific domains "
            "(biology, chemistry, physics, medicine, earth science, astronomy, materials, etc.) "
            "where ML/AI is applied to scientific discovery and domain-specific research."
        )
    lines.append("")

    # Run metadata
    lines.append("## Analysis Summary")
    lines.append("")
    import scicode_lint

    lines.append(f"- **Analysis Date:** {stats.run_date.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **scicode-lint Version:** {scicode_lint.__version__}")
    # Skip papers/repos stats for leakage_paper (single source, not meaningful)
    if stats.data_source != "leakage_paper":
        if stats.total_papers > 0:
            paper_pct = 100 * stats.papers_with_findings / stats.total_papers
            lines.append(
                f"- **Papers with Findings:** {stats.papers_with_findings:,} / "
                f"{stats.total_papers:,} ({paper_pct:.1f}%)"
            )
        if stats.total_repos > 0:
            repo_pct = 100 * stats.repos_with_findings / stats.total_repos
            lines.append(
                f"- **Repos with Findings:** {stats.repos_with_findings:,} / "
                f"{stats.total_repos:,} ({repo_pct:.1f}%)"
            )
    lines.append(f"- **Files Analyzed:** {stats.analyzed_files:,} / {stats.total_files:,}")
    finding_pct = stats.finding_rate
    lines.append(f"- **Files with Findings:** {stats.files_with_findings:,} ({finding_pct:.1f}%)")
    lines.append(f"- **Total Findings:** {stats.total_findings:,}")
    lines.append("")

    # Papers by severity (skip for leakage_paper - single source)
    show_papers_by_severity = (
        stats.data_source != "leakage_paper"
        and stats.total_papers > 0
        and any(papers_by_severity.values())
    )
    if show_papers_by_severity:
        lines.append("## Papers by Severity")
        lines.append("")
        lines.append(
            "Papers with at least one finding of each severity level "
            "(a paper may appear in multiple rows):"
        )
        lines.append("")
        lines.append("| Severity | Papers | % of Papers Analyzed |")
        lines.append("|----------|--------|----------------------|")
        for sev in ["critical", "high", "medium", "low"]:
            count = papers_by_severity.get(sev, 0)
            if count > 0:
                pct = 100 * count / stats.total_papers if stats.total_papers > 0 else 0
                lines.append(f"| {sev.capitalize()} | {count:,} | {pct:.1f}% |")
        lines.append("")
        lines.append(f"*Total papers analyzed: {stats.total_papers:,}*")
        lines.append("")

    # Findings distribution per paper (skip for leakage_paper - single source)
    show_dist = (
        stats.data_source != "leakage_paper"
        and findings_distribution
        and findings_distribution["per_paper"].count > 0
    )
    if show_dist:
        lines.append("## Findings Distribution (per paper)")
        lines.append("")
        lines.append("| Metric | All | Critical | High | Medium | Low |")
        lines.append("|--------|-----|----------|------|--------|-----|")

        # Papers with findings row
        dist = findings_distribution
        papers_row = f"| Papers | {dist['per_paper'].count}"
        for sev in ["critical", "high", "medium", "low"]:
            papers_row += f" | {dist[sev].count}"
        papers_row += " |"
        lines.append(papers_row)

        # Min row
        min_row = f"| Min | {dist['per_paper'].min}"
        for sev in ["critical", "high", "medium", "low"]:
            min_row += f" | {dist[sev].min}" if dist[sev].count > 0 else " | -"
        min_row += " |"
        lines.append(min_row)

        # Max row
        max_row = f"| Max | {dist['per_paper'].max}"
        for sev in ["critical", "high", "medium", "low"]:
            max_row += f" | {dist[sev].max}" if dist[sev].count > 0 else " | -"
        max_row += " |"
        lines.append(max_row)

        # Mean row
        mean_row = f"| Mean | {dist['per_paper'].mean:.1f}"
        for sev in ["critical", "high", "medium", "low"]:
            mean_row += f" | {dist[sev].mean:.1f}" if dist[sev].count > 0 else " | -"
        mean_row += " |"
        lines.append(mean_row)

        # Median row
        median_row = f"| Median | {dist['per_paper'].median:.1f}"
        for sev in ["critical", "high", "medium", "low"]:
            median_row += f" | {dist[sev].median:.1f}" if dist[sev].count > 0 else " | -"
        median_row += " |"
        lines.append(median_row)

        # Std dev row
        std_row = f"| Std Dev | {dist['per_paper'].stdev:.1f}"
        for sev in ["critical", "high", "medium", "low"]:
            std_row += f" | {dist[sev].stdev:.1f}" if dist[sev].count > 1 else " | -"
        std_row += " |"
        lines.append(std_row)

        lines.append("")

    # Verification summary (if any verifications exist)
    total_verified = verification_stats.get("total_verified", 0)
    total_pending = sum(v["pending"] for v in verification_by_severity.values())

    if total_verified > 0 or total_pending > 0:
        lines.append("## Verification Summary")
        lines.append("")

        if total_verified > 0:
            valid = verification_stats.get("valid", 0)
            invalid = verification_stats.get("invalid", 0)
            uncertain = verification_stats.get("uncertain", 0)
            precision = verification_stats.get("precision", 0)
            prec_str = f"{precision:.1f}% ({valid:,} valid / {total_verified:,} verified)"
            lines.append(f"**Overall Precision:** {prec_str}")
            lines.append("")
            lines.append("| Status | Count | % |")
            lines.append("|--------|-------|---|")
            valid_pct = 100 * valid / total_verified
            invalid_pct = 100 * invalid / total_verified
            uncertain_pct = 100 * uncertain / total_verified
            lines.append(f"| Valid (confirmed) | {valid:,} | {valid_pct:.1f}% |")
            lines.append(f"| Invalid (false positive) | {invalid:,} | {invalid_pct:.1f}% |")
            lines.append(f"| Uncertain | {uncertain:,} | {uncertain_pct:.1f}% |")
            if total_pending > 0:
                lines.append(f"| Pending verification | {total_pending:,} | - |")
            lines.append("")

        elif total_pending > 0:
            lines.append(f"**{total_pending:,} findings pending verification.**")
            lines.append("")

        # Verification by severity table
        if total_verified > 0:
            lines.append("### Verified Findings by Severity")
            lines.append("")
            lines.append("| Severity | Total | Valid | Invalid | Uncertain | Pending | Precision |")
            lines.append("|----------|-------|-------|---------|-----------|---------|-----------|")
            for sev in ["critical", "high", "medium", "low"]:
                v = verification_by_severity.get(sev, {})
                total = v.get("total", 0)
                if total == 0:
                    continue
                valid = v.get("valid", 0)
                invalid = v.get("invalid", 0)
                uncertain = v.get("uncertain", 0)
                pending = v.get("pending", 0)
                verified = valid + invalid + uncertain
                prec = f"{100 * valid / verified:.0f}%" if verified > 0 else "-"
                sev_cap = sev.capitalize()
                row = f"| {sev_cap} | {total:,} | {valid:,} | {invalid:,} "
                row += f"| {uncertain:,} | {pending:,} | {prec} |"
                lines.append(row)
            lines.append("")

    # Findings by domain (skip for leakage_paper - all data_science)
    if stats.data_source != "leakage_paper" and stats.by_domain:
        lines.append("## Findings by Scientific Domain")
        lines.append("")
        lines.append("| Domain | Files Analyzed | With Findings | Finding Rate | Total Findings |")
        lines.append("|--------|---------------|---------------|--------------|----------------|")
        for d in sorted(stats.by_domain, key=lambda x: x.total_findings, reverse=True):
            lines.append(
                f"| {d.domain} | {d.analyzed_files:,} | {d.files_with_findings:,} | "
                f"{d.finding_rate:.1f}% | {d.total_findings:,} |"
            )
        lines.append("")

    # Findings by category
    if stats.by_category:
        lines.append("## Findings by Category")
        lines.append("")
        # Skip Unique Repos for leakage_paper (always 1)
        if stats.data_source == "leakage_paper":
            lines.append("| Category | Count | Unique Files |")
            lines.append("|----------|-------|--------------|")
            for c in stats.by_category:
                row = f"| {c.category} | {c.count:,} | {c.unique_files:,} |"
                lines.append(row)
        else:
            lines.append("| Category | Count | Unique Files | Unique Repos |")
            lines.append("|----------|-------|--------------|--------------|")
            for c in stats.by_category:
                row = f"| {c.category} | {c.count:,} | {c.unique_files:,} | {c.unique_repos:,} |"
                lines.append(row)
        lines.append("")

    # Findings by severity (as table with percentages)
    if stats.by_severity:
        lines.append("## Findings by Severity")
        lines.append("")
        lines.append("| Severity | Count | % of Total |")
        lines.append("|----------|-------|------------|")
        total = stats.total_findings
        for sev in ["critical", "high", "medium", "low"]:
            if sev in stats.by_severity:
                count = stats.by_severity[sev]
                pct = 100 * count / total if total > 0 else 0
                lines.append(f"| {sev.capitalize()} | {count:,} | {pct:.1f}% |")
        lines.append("")

    # Top patterns
    if top_patterns:
        lines.append("## Most Common Patterns")
        lines.append("")
        # Simplified columns for leakage_paper (no repos/confidence)
        if stats.data_source == "leakage_paper":
            lines.append("| Pattern | Category | Count | Files |")
            lines.append("|---------|----------|-------|-------|")
            for p in top_patterns:
                lines.append(
                    f"| {p['pattern_id']} | {p['category']} | "
                    f"{p['count']:,} | {p['unique_files']:,} |"
                )
        else:
            lines.append("| Pattern | Category | Count | Files | Repos | Avg Confidence |")
            lines.append("|---------|----------|-------|-------|-------|----------------|")
            for p in top_patterns:
                conf = p["avg_confidence"] or 0
                lines.append(
                    f"| {p['pattern_id']} | {p['category']} | {p['count']:,} | "
                    f"{p['unique_files']:,} | {p['unique_repos']:,} | {conf:.0%} |"
                )
        lines.append("")

    # Example findings
    if examples:
        lines.append("## Example Findings")
        lines.append("")
        lines.append("Representative findings from each category (with links to source):")
        lines.append("")

        for category, findings in examples.items():
            if not findings:
                continue

            lines.append(f"### {category}")
            lines.append("")

            for f in findings:
                conf = f["confidence"]
                header = f"**{f['pattern_id']}** ({f['severity']}, {conf:.0%} confidence)"
                lines.append(header)
                lines.append("")

                # Build GitHub link to file with line numbers
                repo_url = f["repo_url"] or ""
                original_path = f["original_path"] or ""
                finding_lines = f["lines"] or "[]"

                if repo_url and original_path:
                    # Parse line numbers from JSON string
                    try:
                        line_nums = json.loads(finding_lines) if finding_lines else []
                    except (json.JSONDecodeError, TypeError):
                        line_nums = []

                    # Build GitHub blob URL
                    github_url = f"{repo_url}/blob/main/{original_path}"
                    if line_nums:
                        if len(line_nums) == 1:
                            github_url += f"#L{line_nums[0]}"
                        else:
                            github_url += f"#L{min(line_nums)}-L{max(line_nums)}"

                    file_link = f"[{original_path}]({github_url})"
                    lines.append(f"- **File:** {file_link} in [{f['repo_name']}]({repo_url})")
                else:
                    lines.append(f"- **Repo:** {f['repo_name']} ({f['domain']})")

                if f["paper_title"]:
                    paper_ref = f["paper_title"]
                    if f["arxiv_id"]:
                        paper_ref += (
                            f" ([arXiv:{f['arxiv_id']}](https://arxiv.org/abs/{f['arxiv_id']}))"
                        )
                    lines.append(f"- **Paper:** {paper_ref}")

                    # Add authors if available
                    if f.get("paper_authors"):
                        try:
                            authors = json.loads(f["paper_authors"])
                            if authors:
                                if len(authors) > 2:
                                    authors_str = f"{authors[0]} et al."
                                else:
                                    authors_str = ", ".join(authors)
                                lines.append(f"- **Authors:** {authors_str}")
                        except (json.JSONDecodeError, TypeError):
                            pass
                lines.append(f"- **Issue:** {f['issue']}")
                if f["explanation"]:
                    lines.append(f"- **Explanation:** {f['explanation']}")
                if f["suggestion"]:
                    lines.append(f"- **Suggestion:** {f['suggestion']}")
                if f["snippet"]:
                    lines.append("")
                    lines.append("```python")
                    lines.append(f["snippet"].strip())
                    lines.append("```")
                lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    analysis_date = stats.run_date.strftime("%Y-%m-%d")
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"*Analysis conducted: {analysis_date} | Report generated: {report_date}*")
    lines.append("")

    return "\n".join(lines)


def save_report(content: str, analysis_date: datetime, output_path: Path | None = None) -> Path:
    """Save report to file.

    Args:
        content: Markdown content.
        analysis_date: Date when analysis was conducted.
        output_path: Output path, or None for default (includes date).

    Returns:
        Path to saved file.
    """
    if output_path is None:
        # Include time to avoid overwriting same-day reports
        date_str = analysis_date.strftime("%Y-%m-%d_%H%M")
        output_path = REPORTS_DIR / f"FINDINGS_REPORT_{date_str}.md"

    output_path.parent.mkdir(exist_ok=True, parents=True)
    output_path.write_text(content)
    return output_path


def _query_valid_findings(
    conn: sqlite3.Connection, run_id: int, severity: str
) -> list[dict[str, Any]]:
    """Query valid findings for a given severity level.

    Args:
        conn: Database connection.
        run_id: Analysis run ID.
        severity: Severity level ('critical' or 'high').

    Returns:
        List of finding dicts ordered by confidence.
    """
    cursor = conn.execute(
        """
        SELECT
            fn.id,
            fn.pattern_id,
            fn.severity,
            fn.confidence,
            fn.issue,
            fn.explanation,
            fn.suggestion,
            fn.snippet,
            fn.lines,
            f.file_path,
            f.original_path,
            r.repo_name,
            r.repo_url,
            r.domain,
            r.paper_id,
            p.title as paper_title,
            p.arxiv_id,
            p.authors as paper_authors,
            fv.reasoning as verification_reasoning
        FROM findings fn
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        JOIN files f ON f.id = fa.file_id
        JOIN repos r ON r.id = f.repo_id
        LEFT JOIN papers p ON p.id = r.paper_id
        INNER JOIN finding_verifications fv ON fv.finding_id = fn.id
        WHERE fa.run_id = ?
          AND fn.severity = ?
          AND fv.status = 'valid'
          AND r.paper_id IS NOT NULL
        GROUP BY r.paper_id
        ORDER BY fn.confidence DESC
        """,
        (run_id, severity),
    )

    return [
        {
            "finding_id": row[0],
            "pattern_id": row[1],
            "severity": row[2],
            "confidence": row[3],
            "issue": row[4],
            "explanation": row[5],
            "suggestion": row[6],
            "snippet": row[7],
            "lines": row[8],
            "file_path": row[9],
            "original_path": row[10],
            "repo_name": row[11],
            "repo_url": row[12],
            "domain": row[13],
            "paper_id": row[14],
            "paper_title": row[15],
            "arxiv_id": row[16],
            "paper_authors": row[17],
            "verification_reasoning": row[18],
        }
        for row in cursor.fetchall()
    ]


def get_valid_critical_findings(
    conn: sqlite3.Connection, run_id: int, limit: int = 10
) -> list[dict[str, Any]]:
    """Get valid findings with pattern diversity across severity levels.

    Selection strategy for maximum diversity:
    1. Critical findings: pick one per unique pattern
    2. High findings: pick one per unique pattern (not already seen)
    3. Fill remaining slots with duplicates by confidence

    Args:
        conn: Database connection.
        run_id: Analysis run ID.
        limit: Max findings to return.

    Returns:
        List of finding dicts with paper info.
    """
    selected: list[dict[str, Any]] = []
    seen_patterns: set[str] = set()
    remaining: list[dict[str, Any]] = []

    # Process critical findings first, then high
    for severity in ("critical", "high"):
        findings = _query_valid_findings(conn, run_id, severity)

        for f in findings:
            if len(selected) >= limit:
                break

            pattern = f["pattern_id"]
            if pattern not in seen_patterns:
                selected.append(f)
                seen_patterns.add(pattern)
            else:
                remaining.append(f)

    # Fill remaining slots with duplicates (by confidence, critical first)
    for f in remaining:
        if len(selected) >= limit:
            break
        selected.append(f)

    return selected


def generate_valid_critical_report(conn: sqlite3.Connection, run_id: int) -> str:
    """Generate markdown report of valid findings for quick verification.

    Args:
        conn: Database connection.
        run_id: Analysis run ID.

    Returns:
        Markdown report string.
    """
    findings = get_valid_critical_findings(conn, run_id, limit=10)

    # Count by severity
    critical_count = sum(1 for f in findings if f["severity"] == "critical")
    high_count = sum(1 for f in findings if f["severity"] == "high")

    lines = []
    lines.append("# Valid Findings - Quick Verification Sample")
    lines.append("")

    # Build summary with counts
    counts = []
    if critical_count:
        counts.append(f"{critical_count} critical")
    if high_count:
        counts.append(f"{high_count} high")
    count_str = " + ".join(counts) if counts else "0"

    lines.append(
        f"**{len(findings)} verified findings** ({count_str}) "
        "with pattern diversity for fast manual verification."
    )
    lines.append("")

    for i, f in enumerate(findings, 1):
        lines.append(f"## {i}. {f['pattern_id']} ({f['severity']})")
        lines.append("")

        # Build GitHub link
        repo_url = f["repo_url"] or ""
        original_path = f["original_path"] or ""
        finding_lines = f["lines"] or "[]"

        if repo_url and original_path:
            try:
                line_nums = json.loads(finding_lines) if finding_lines else []
            except (json.JSONDecodeError, TypeError):
                line_nums = []

            github_url = f"{repo_url}/blob/main/{original_path}"
            if line_nums:
                if len(line_nums) == 1:
                    github_url += f"#L{line_nums[0]}"
                else:
                    github_url += f"#L{min(line_nums)}-L{max(line_nums)}"

            lines.append(f"**File:** [{original_path}]({github_url})")
        else:
            lines.append(f"**File:** {f['original_path']}")

        lines.append(f"**Repo:** [{f['repo_name']}]({repo_url})")

        if f["paper_title"]:
            paper_ref = f["paper_title"]
            if f["arxiv_id"]:
                paper_ref += f" ([arXiv:{f['arxiv_id']}](https://arxiv.org/abs/{f['arxiv_id']}))"
            lines.append(f"**Paper:** {paper_ref}")

            # Add authors if available
            if f.get("paper_authors"):
                try:
                    authors = json.loads(f["paper_authors"])
                    if authors:
                        # Format: "First Author et al." for >2 authors, otherwise list all
                        if len(authors) > 2:
                            authors_str = f"{authors[0]} et al."
                        else:
                            authors_str = ", ".join(authors)
                        lines.append(f"**Authors:** {authors_str}")
                except (json.JSONDecodeError, TypeError):
                    pass

        lines.append("")
        lines.append(f"**Issue:** {f['issue']}")
        lines.append("")

        if f["explanation"]:
            lines.append(f"**Explanation:** {f['explanation']}")
            lines.append("")

        if f["snippet"]:
            lines.append("**Code:**")
            lines.append("```python")
            lines.append(f["snippet"].strip())
            lines.append("```")
            lines.append("")

        if f["verification_reasoning"]:
            lines.append(f"**Verification reasoning:** {f['verification_reasoning']}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate markdown report from analysis database")
    parser.add_argument("--run-id", type=int, help="Specific run ID (default: latest)")
    parser.add_argument("--output", "-o", type=Path, help="Output file path")
    parser.add_argument("--db", type=Path, help="Database path (default: data/analysis.db)")
    parser.add_argument(
        "--data-source", type=str, help="Filter by data source (e.g., 'leakage_paper')"
    )
    parser.add_argument("--list-runs", action="store_true", help="List available runs and exit")
    parser.add_argument(
        "--verified-only",
        action="store_true",
        help="Only include findings with completed verification",
    )
    args = parser.parse_args()

    # Connect to database
    db_path = args.db or get_db_path()
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        logger.info("Run the pipeline first: python -m real_world_demo.run_pipeline --pilot 100")
        raise SystemExit(1)

    conn = init_db(db_path)

    # List runs mode
    if args.list_runs:
        runs = list_runs(conn, data_source=args.data_source)
        if not runs:
            msg = "No analysis runs found"
            if args.data_source:
                msg += f" for data_source='{args.data_source}'"
            logger.info(msg)
        else:
            logger.info("Available runs:")
            for run in runs:
                status = run.status.value if run.status else "unknown"
                logger.info(
                    f"  #{run.run_id}: [{run.data_source}] "
                    f"{run.started_at.strftime('%Y-%m-%d %H:%M')} - "
                    f"{run.analyzed_files}/{run.total_files} files, "
                    f"{run.total_findings} findings ({status})"
                )
        conn.close()
        return

    # Generate report
    logger.info(f"Generating report from {db_path}")

    # Get stats first to extract analysis date
    stats = get_run_stats(conn, args.run_id)
    if stats.run_id == 0:
        logger.error("No analysis runs found in database")
        conn.close()
        raise SystemExit(1)

    report = generate_markdown_report(conn, args.run_id, verified_only=args.verified_only)

    # Save with analysis date in filename
    output_path = save_report(report, stats.run_date, args.output)
    logger.info(f"Report saved to {output_path}")

    # Generate valid critical findings sample when using --verified-only
    if args.verified_only:
        critical_report = generate_valid_critical_report(conn, stats.run_id)
        date_str = stats.run_date.strftime("%Y-%m-%d_%H%M")
        critical_path = REPORTS_DIR / f"VALID_FINDINGS_SAMPLE_{date_str}.md"
        critical_path.write_text(critical_report)
        logger.info(f"Valid findings sample saved to {critical_path}")

    conn.close()


if __name__ == "__main__":
    main()
