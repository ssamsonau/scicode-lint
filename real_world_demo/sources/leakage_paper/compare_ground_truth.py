"""Compare scicode-lint findings with leakage paper ground truth.

Computes precision/recall for each leakage type by comparing our pattern
detections against the manual labels from Yang et al. ASE'22.

Pattern mapping:
    Paper Label    scicode-lint Patterns
    -----------    --------------------
    pre            ml-001, ml-007
    overlap        ml-009
    multi          ml-010

Usage:
    python -m real_world_demo.sources.leakage_paper.compare_ground_truth
    python -m real_world_demo.sources.leakage_paper.compare_ground_truth --detailed
    python -m real_world_demo.sources.leakage_paper.compare_ground_truth --run-id 8
"""

import argparse
import csv
import re
import sqlite3
from dataclasses import dataclass

from loguru import logger

from real_world_demo.config import LEAKAGE_PAPER_DATA_DIR
from real_world_demo.database import get_db_path, get_latest_run_id, init_db

# Pattern mapping: paper label -> scicode-lint patterns
PATTERN_MAP = {
    "pre": ["ml-001", "ml-007"],
    "overlap": ["ml-009"],
    "multi": ["ml-010"],
}


@dataclass
class ComparisonResult:
    """Result of comparing one leakage type."""

    label: str
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)


def load_ground_truth() -> dict[str, dict[str, bool]]:
    """Load ground truth labels from CSV.

    Returns:
        Dict mapping notebook ID (e.g., "nb_1239") to labels dict.
        Labels dict has keys: 'model', 'pre', 'overlap', 'multi'.
    """
    gt_path = LEAKAGE_PAPER_DATA_DIR / "ground_truth.csv"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    ground_truth = {}

    with open(gt_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract notebook ID from path like "GitHub-data/notebooks/2021-09-05/nb_1244.py"
            nb_path = row.get("nb", "")
            match = re.search(r"(nb_\d+)", nb_path)
            if not match:
                continue

            nb_id = match.group(1)

            # Parse labels - Y means positive, anything else is negative
            # Handle cases like "Y (MinMaxScaler)", "Y [unsupported lib]", "N [human errors]"
            def is_positive(value: str) -> bool:
                return value.strip().upper().startswith("Y")

            ground_truth[nb_id] = {
                "model": is_positive(row.get("model", "")),
                "pre": is_positive(row.get("pre", "")),
                "overlap": is_positive(row.get("overlap", "")),
                "multi": is_positive(row.get("multi", "")),
            }

    return ground_truth


def get_findings_by_notebook(run_id: int) -> dict[str, set[str]]:
    """Get pattern detections grouped by notebook.

    Args:
        run_id: Analysis run ID.

    Returns:
        Dict mapping notebook ID to set of detected pattern IDs.
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)

    query = """
        SELECT fi.file_path, f.pattern_id
        FROM findings f
        JOIN file_analyses fa ON f.file_analysis_id = fa.id
        JOIN files fi ON fa.file_id = fi.id
        WHERE fa.run_id = ?
          AND fa.status = 'success'
          AND f.pattern_id IN ('ml-001', 'ml-007', 'ml-009', 'ml-010')
    """

    findings: dict[str, set[str]] = {}
    for row in conn.execute(query, (run_id,)):
        file_path, pattern_id = row
        # Extract notebook ID from path like "files/nb_1239.ipynb"
        match = re.search(r"(nb_\d+)", file_path)
        if match:
            nb_id = match.group(1)
            if nb_id not in findings:
                findings[nb_id] = set()
            findings[nb_id].add(pattern_id)

    conn.close()
    return findings


def get_analyzed_notebooks(run_id: int) -> set[str]:
    """Get set of successfully analyzed notebook IDs.

    Args:
        run_id: Analysis run ID.

    Returns:
        Set of notebook IDs that were successfully analyzed.
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)

    query = """
        SELECT fi.file_path
        FROM file_analyses fa
        JOIN files fi ON fa.file_id = fi.id
        WHERE fa.run_id = ? AND fa.status = 'success'
    """

    notebooks = set()
    for row in conn.execute(query, (run_id,)):
        file_path = row[0]
        match = re.search(r"(nb_\d+)", file_path)
        if match:
            notebooks.add(match.group(1))

    conn.close()
    return notebooks


def get_timed_out_patterns(run_id: int) -> dict[str, set[str]]:
    """Get patterns that timed out per notebook.

    Args:
        run_id: Analysis run ID.

    Returns:
        Dict mapping notebook ID to set of pattern IDs that timed out.
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)

    # Check if pattern_runs table exists
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='pattern_runs'"
    )
    if not cursor.fetchone():
        conn.close()
        return {}  # Table doesn't exist yet

    query = """
        SELECT fi.file_path, pr.pattern_id
        FROM pattern_runs pr
        JOIN file_analyses fa ON pr.file_analysis_id = fa.id
        JOIN files fi ON fa.file_id = fi.id
        WHERE fa.run_id = ?
          AND pr.status = 'timeout'
          AND pr.pattern_id IN ('ml-001', 'ml-007', 'ml-009', 'ml-010')
    """

    timeouts: dict[str, set[str]] = {}
    for row in conn.execute(query, (run_id,)):
        file_path, pattern_id = row
        match = re.search(r"(nb_\d+)", file_path)
        if match:
            nb_id = match.group(1)
            if nb_id not in timeouts:
                timeouts[nb_id] = set()
            timeouts[nb_id].add(pattern_id)

    conn.close()
    return timeouts


def get_latest_leakage_paper_run_id() -> int:
    """Get the most recent leakage paper analysis run ID."""
    conn = init_db()
    run_id = get_latest_run_id(conn, data_source="leakage_paper")
    conn.close()
    return run_id


def get_run_date(run_id: int) -> str | None:
    """Get the start date of an analysis run.

    Args:
        run_id: Analysis run ID.

    Returns:
        Run date as string, or None if not found.
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT started_at FROM analysis_runs WHERE id = ?",
        (run_id,),
    )
    row = cursor.fetchone()
    conn.close()
    if row and row[0]:
        # Parse and format the date
        try:
            from datetime import datetime

            dt = datetime.fromisoformat(row[0])
            return dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            return str(row[0])
    return None


def compare(run_id: int) -> tuple[list[ComparisonResult], int]:
    """Compare findings with ground truth.

    Args:
        run_id: Analysis run ID.

    Returns:
        Tuple of (list of ComparisonResult for each leakage type, count of excluded due to timeout).
    """
    ground_truth = load_ground_truth()
    findings = get_findings_by_notebook(run_id)
    analyzed = get_analyzed_notebooks(run_id)
    timeouts = get_timed_out_patterns(run_id)

    results = []
    total_excluded = 0

    for label, patterns in PATTERN_MAP.items():
        tp = fp = fn = tn = 0
        excluded = 0

        for nb_id in analyzed:
            if nb_id not in ground_truth:
                logger.warning(f"Notebook {nb_id} not in ground truth, skipping")
                continue

            gt_labels = ground_truth[nb_id]

            # Skip notebooks without ML model (can't have leakage)
            if not gt_labels["model"]:
                continue

            # Skip if any relevant pattern timed out (result unknown)
            nb_timeouts = timeouts.get(nb_id, set())
            if any(p in nb_timeouts for p in patterns):
                excluded += 1
                continue

            gt_positive = gt_labels[label]
            detected = any(p in findings.get(nb_id, set()) for p in patterns)

            if gt_positive and detected:
                tp += 1
            elif not gt_positive and detected:
                fp += 1
            elif gt_positive and not detected:
                fn += 1
            else:
                tn += 1

        total_excluded = max(total_excluded, excluded)

        results.append(
            ComparisonResult(
                label=label,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                true_negatives=tn,
            )
        )

    return results, total_excluded


def print_results(
    results: list[ComparisonResult],
    run_id: int,
    excluded_count: int = 0,
    run_date: str | None = None,
) -> None:
    """Print comparison results in markdown format."""
    from datetime import datetime

    import scicode_lint

    print("# Ground Truth Comparison")
    print()
    print("Comparison against Yang et al. ASE'22 ground truth labels.")
    print()
    if run_date:
        print(f"**Analysis Date:** {run_date}")
    print(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"**scicode-lint version:** {scicode_lint.__version__}")

    if excluded_count > 0:
        print(f"**Excluded:** {excluded_count} notebooks (pattern timeouts)")

    print()
    print("## Results")
    print()
    print("| Label | TP | FP | FN | TN | Precision | Recall | F1 |")
    print("|-------|---:|---:|---:|---:|----------:|-------:|---:|")

    for r in results:
        print(
            f"| {r.label} | {r.true_positives} | {r.false_positives} | "
            f"{r.false_negatives} | {r.true_negatives} | "
            f"{r.precision:.1%} | {r.recall:.1%} | {r.f1:.1%} |"
        )

    print()
    print("## Accuracy")
    print()
    print("| Label | Correct | Total | Accuracy |")
    print("|-------|--------:|------:|---------:|")

    for r in results:
        total = r.true_positives + r.false_positives + r.false_negatives + r.true_negatives
        if total > 0:
            correct = r.true_positives + r.true_negatives
            print(f"| {r.label} | {correct} | {total} | {correct / total:.1%} |")


def print_detailed_comparison(run_id: int) -> None:
    """Print per-notebook comparison."""
    ground_truth = load_ground_truth()
    findings = get_findings_by_notebook(run_id)
    analyzed = get_analyzed_notebooks(run_id)

    print(f"\n{'=' * 60}")
    print("Per-Notebook Comparison")
    print(f"{'=' * 60}\n")

    header = (
        f"{'Notebook':<12} {'GT:pre':<8} {'Det:pre':<8} "
        f"{'GT:ovlp':<8} {'Det:ovlp':<8} {'GT:multi':<8} {'Det:multi':<8}"
    )
    print(header)
    print("-" * 80)

    for nb_id in sorted(analyzed):
        if nb_id not in ground_truth:
            continue

        gt = ground_truth[nb_id]
        if not gt["model"]:
            continue  # Skip non-ML notebooks

        det = findings.get(nb_id, set())

        def mark(gt_val: bool, detected: bool) -> str:
            if gt_val and detected:
                return "✅ TP"
            elif not gt_val and detected:
                return "❌ FP"
            elif gt_val and not detected:
                return "⚠️ FN"
            else:
                return "- TN"

        pre_det = any(p in det for p in ["ml-001", "ml-007"])
        ovlp_det = "ml-009" in det
        multi_det = "ml-010" in det

        print(
            f"{nb_id:<12} "
            f"{'Y' if gt['pre'] else 'N':<8} {mark(gt['pre'], pre_det):<8} "
            f"{'Y' if gt['overlap'] else 'N':<8} {mark(gt['overlap'], ovlp_det):<8} "
            f"{'Y' if gt['multi'] else 'N':<8} {mark(gt['multi'], multi_det):<8}"
        )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare scicode-lint findings with leakage paper ground truth"
    )
    parser.add_argument(
        "--run-id",
        type=int,
        help="Analysis run ID (default: latest)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show per-notebook comparison",
    )
    args = parser.parse_args()

    run_id = args.run_id or get_latest_leakage_paper_run_id()
    if not run_id:
        logger.error("No leakage_paper analysis runs found. Run analysis first.")
        return

    run_date = get_run_date(run_id)
    results, excluded_count = compare(run_id)
    print_results(results, run_id, excluded_count, run_date)

    if args.detailed:
        print_detailed_comparison(run_id)


if __name__ == "__main__":
    main()
