"""SQLite database for tracking analysis results.

Stores repos, files, findings, and analysis runs for easy querying and reporting.
Supports multiple analysis runs across different dates.
"""

import json
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from .config import DATA_DIR
from .models import (
    AnalysisRun,
    AnalysisStats,
    CategoryStats,
    DomainStats,
    PatternStats,
    RunStatus,
)


def get_db_path() -> Path:
    """Get path to SQLite database."""
    return DATA_DIR / "analysis.db"


def init_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Initialize database with schema.

    Args:
        db_path: Optional path to database file. Uses default if not provided.

    Returns:
        Connection to the database.
    """
    if db_path is None:
        db_path = get_db_path()

    db_path.parent.mkdir(exist_ok=True, parents=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Create tables with comprehensive schema
    conn.executescript("""
        -- Papers from PapersWithCode
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_url TEXT UNIQUE NOT NULL,
            title TEXT,
            arxiv_id TEXT,
            abstract TEXT,
            authors TEXT,  -- JSON array of author names
            tasks TEXT,  -- JSON array
            matched_domain TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Repositories or data sources containing files
        CREATE TABLE IF NOT EXISTS repos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo_url TEXT UNIQUE NOT NULL,
            repo_name TEXT NOT NULL,  -- owner__repo format or data source name
            data_source TEXT DEFAULT 'papersWithCode',  -- data source identifier
            paper_id INTEGER,
            domain TEXT,
            clone_status TEXT DEFAULT 'pending',
            clone_error TEXT,
            cloned_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (paper_id) REFERENCES papers(id)
        );

        -- Python files collected from repos
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,  -- relative to collected_code/files/
            original_path TEXT,       -- path within original repo
            is_notebook BOOLEAN DEFAULT FALSE,
            file_size INTEGER,
            line_count INTEGER,
            ml_imports TEXT,          -- comma-separated
            scientific_imports TEXT,  -- comma-separated
            prefilter_passed BOOLEAN DEFAULT TRUE,
            prefilter_response TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (repo_id) REFERENCES repos(id),
            UNIQUE(repo_id, file_path)
        );

        -- Analysis runs (can have multiple per day)
        CREATE TABLE IF NOT EXISTS analysis_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            status TEXT DEFAULT 'running',
            data_source TEXT DEFAULT 'papers_with_code',  -- data source identifier
            total_files INTEGER DEFAULT 0,
            analyzed_files INTEGER DEFAULT 0,
            files_with_findings INTEGER DEFAULT 0,
            total_findings INTEGER DEFAULT 0,
            config TEXT,              -- JSON config (max_files, max_concurrent, etc.)
            git_commit TEXT,          -- scicode-lint git commit
            model_name TEXT,          -- LLM model used
            notes TEXT                -- optional run notes
        );

        -- File analysis results (one per file per run)
        CREATE TABLE IF NOT EXISTS file_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            file_id INTEGER NOT NULL,
            status TEXT NOT NULL,     -- success, error, timeout, skipped
            error TEXT,               -- error message if failed
            duration_seconds REAL,
            analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES analysis_runs(id),
            FOREIGN KEY (file_id) REFERENCES files(id),
            UNIQUE(run_id, file_id)
        );

        -- Individual findings
        CREATE TABLE IF NOT EXISTS findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_analysis_id INTEGER NOT NULL,
            pattern_id TEXT NOT NULL,
            category TEXT,
            severity TEXT,
            confidence REAL,
            issue TEXT,
            explanation TEXT,
            suggestion TEXT,
            reasoning TEXT,
            lines TEXT,               -- JSON array of line numbers
            snippet TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (file_analysis_id) REFERENCES file_analyses(id)
        );

        -- Finding verifications (Claude evaluation of whether findings are real)
        CREATE TABLE IF NOT EXISTS finding_verifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            finding_id INTEGER NOT NULL,
            status TEXT NOT NULL,     -- valid, invalid, uncertain, error
            reasoning TEXT,
            model TEXT,               -- Claude model used
            verified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (finding_id) REFERENCES findings(id),
            UNIQUE(finding_id)        -- One verification per finding
        );

        -- Individual pattern run results (tracks success, timeout, etc.)
        CREATE TABLE IF NOT EXISTS pattern_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_analysis_id INTEGER NOT NULL,
            pattern_id TEXT NOT NULL,
            status TEXT NOT NULL,     -- success, timeout, context_length, api_error
            detected TEXT,            -- yes, no, context-dependent (null if failed)
            confidence REAL,          -- null if failed
            reasoning TEXT,
            error_message TEXT,       -- error details if failed
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (file_analysis_id) REFERENCES file_analyses(id),
            UNIQUE(file_analysis_id, pattern_id)
        );

        -- Indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_pattern_runs_analysis ON pattern_runs(file_analysis_id);
        CREATE INDEX IF NOT EXISTS idx_pattern_runs_pattern ON pattern_runs(pattern_id);
        CREATE INDEX IF NOT EXISTS idx_pattern_runs_status ON pattern_runs(status);
        CREATE INDEX IF NOT EXISTS idx_repos_domain ON repos(domain);
        CREATE INDEX IF NOT EXISTS idx_verifications_finding ON finding_verifications(finding_id);
        CREATE INDEX IF NOT EXISTS idx_verifications_status ON finding_verifications(status);
        CREATE INDEX IF NOT EXISTS idx_repos_name ON repos(repo_name);
        CREATE INDEX IF NOT EXISTS idx_files_repo ON files(repo_id);
        CREATE INDEX IF NOT EXISTS idx_file_analyses_run ON file_analyses(run_id);
        CREATE INDEX IF NOT EXISTS idx_file_analyses_file ON file_analyses(file_id);
        CREATE INDEX IF NOT EXISTS idx_findings_analysis ON findings(file_analysis_id);
        CREATE INDEX IF NOT EXISTS idx_findings_pattern ON findings(pattern_id);
        CREATE INDEX IF NOT EXISTS idx_findings_category ON findings(category);
        CREATE INDEX IF NOT EXISTS idx_findings_severity ON findings(severity);
        CREATE INDEX IF NOT EXISTS idx_analysis_runs_date ON analysis_runs(started_at);
    """)

    # Migration: Add data_source column if missing (for existing databases)
    cursor = conn.execute("PRAGMA table_info(analysis_runs)")
    columns = {row[1] for row in cursor.fetchall()}
    if "data_source" not in columns:
        conn.execute(
            "ALTER TABLE analysis_runs ADD COLUMN data_source TEXT DEFAULT 'papers_with_code'"
        )
        logger.info("Migrated analysis_runs: added data_source column")

    # Migration: Add authors column to papers if missing
    cursor = conn.execute("PRAGMA table_info(papers)")
    paper_columns = {row[1] for row in cursor.fetchall()}
    if "authors" not in paper_columns:
        conn.execute("ALTER TABLE papers ADD COLUMN authors TEXT")
        logger.info("Migrated papers: added authors column")

    conn.commit()
    return conn


# ============================================================================
# Paper operations
# ============================================================================


def insert_paper(conn: sqlite3.Connection, paper_data: dict[str, Any]) -> int:
    """Insert or get paper record.

    Args:
        conn: Database connection.
        paper_data: Paper metadata dict.

    Returns:
        Paper ID.
    """
    paper_url = paper_data.get("paper_url", "")
    if not paper_url:
        return 0

    # Try to get existing
    cursor = conn.execute("SELECT id FROM papers WHERE paper_url = ?", (paper_url,))
    row = cursor.fetchone()
    if row:
        return int(row["id"])

    # Insert new
    tasks = paper_data.get("tasks", [])
    authors = paper_data.get("authors", [])
    cursor = conn.execute(
        """
        INSERT INTO papers (paper_url, title, arxiv_id, abstract, authors, tasks, matched_domain)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            paper_url,
            paper_data.get("paper_title", ""),
            paper_data.get("arxiv_id", ""),
            paper_data.get("abstract", ""),
            json.dumps(authors) if isinstance(authors, list) else authors,
            json.dumps(tasks) if isinstance(tasks, list) else tasks,
            paper_data.get("domain", ""),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid) if cursor.lastrowid else 0


# ============================================================================
# Repo operations
# ============================================================================


def get_or_create_repo(conn: sqlite3.Connection, repo_data: dict[str, Any]) -> int:
    """Get existing repo or create new one.

    Args:
        conn: Database connection.
        repo_data: Dict with repo metadata.

    Returns:
        Repo ID.
    """
    repo_name = repo_data.get("repo_name", "")
    if not repo_name:
        return 0

    # Try to get existing
    cursor = conn.execute("SELECT id FROM repos WHERE repo_name = ?", (repo_name,))
    row = cursor.fetchone()
    if row:
        return int(row["id"])

    # Get or create paper first
    paper_id = insert_paper(conn, repo_data) if repo_data.get("paper_url") else None

    # Create repo
    cursor = conn.execute(
        """
        INSERT INTO repos (repo_url, repo_name, data_source, paper_id, domain, clone_status)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            repo_data.get("repo_url", ""),
            repo_name,
            repo_data.get("data_source", "papersWithCode"),
            paper_id,
            repo_data.get("domain", ""),
            "success",  # assume cloned if we're analyzing
        ),
    )
    conn.commit()
    return int(cursor.lastrowid) if cursor.lastrowid else 0


# ============================================================================
# File operations
# ============================================================================


def insert_file(conn: sqlite3.Connection, repo_id: int, file_data: dict[str, Any]) -> int:
    """Insert or update file record.

    Args:
        conn: Database connection.
        repo_id: Parent repo ID.
        file_data: Dict with file metadata.

    Returns:
        File ID.
    """
    file_path = file_data.get("file_path", "")

    # Check for existing
    cursor = conn.execute(
        "SELECT id FROM files WHERE repo_id = ? AND file_path = ?",
        (repo_id, file_path),
    )
    row = cursor.fetchone()
    if row:
        return int(row["id"])

    # Parse imports
    ml_imports = file_data.get("ml_imports", "")
    if isinstance(ml_imports, list):
        ml_imports = ",".join(ml_imports)

    scientific_imports = file_data.get("scientific_imports", "")
    if isinstance(scientific_imports, list):
        scientific_imports = ",".join(scientific_imports)

    cursor = conn.execute(
        """
        INSERT INTO files
        (repo_id, file_path, original_path, is_notebook, file_size, line_count,
         ml_imports, scientific_imports, prefilter_passed, prefilter_response)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            repo_id,
            file_path,
            file_data.get("original_path", ""),
            file_data.get("is_notebook", False),
            file_data.get("file_size", 0),
            file_data.get("line_count", 0),
            ml_imports,
            scientific_imports,
            file_data.get("prefilter_passed", True),
            file_data.get("prefilter_response", ""),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid) if cursor.lastrowid else 0


def get_file_id(conn: sqlite3.Connection, file_path: str) -> int | None:
    """Get file ID by path.

    Args:
        conn: Database connection.
        file_path: File path to look up.

    Returns:
        File ID or None.
    """
    # Try exact match
    cursor = conn.execute("SELECT id FROM files WHERE file_path = ?", (file_path,))
    row = cursor.fetchone()
    if row:
        return int(row["id"])

    # Try partial match (for absolute paths)
    if "/" in file_path:
        # Extract relative path
        parts = file_path.split("files/")
        if len(parts) > 1:
            relative_path = "files/" + parts[-1]
            cursor = conn.execute("SELECT id FROM files WHERE file_path = ?", (relative_path,))
            row = cursor.fetchone()
            if row:
                return int(row["id"])

    return None


# ============================================================================
# Analysis run operations
# ============================================================================


def get_current_git_commit() -> str:
    """Get current git commit hash of scicode-lint."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def start_analysis_run(
    conn: sqlite3.Connection,
    total_files: int,
    data_source: str = "papers_with_code",
    config: dict[str, Any] | None = None,
    model_name: str = "",
    notes: str = "",
) -> int:
    """Start a new analysis run.

    Args:
        conn: Database connection.
        total_files: Total files to analyze.
        data_source: Data source identifier (e.g., 'papers_with_code', 'leakage_paper').
        config: Run configuration dict.
        model_name: LLM model being used.
        notes: Optional run notes.

    Returns:
        Run ID.
    """
    cursor = conn.execute(
        """
        INSERT INTO analysis_runs (total_files, data_source, config, git_commit, model_name, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            total_files,
            data_source,
            json.dumps(config or {}),
            get_current_git_commit(),
            model_name,
            notes,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid) if cursor.lastrowid else 0


def get_latest_run_id(
    conn: sqlite3.Connection,
    data_source: str | None = None,
) -> int:
    """Get the most recent analysis run ID.

    Args:
        conn: Database connection.
        data_source: Optional filter by data source.

    Returns:
        Run ID, or 0 if no runs found.
    """
    if data_source:
        result = conn.execute(
            "SELECT MAX(id) FROM analysis_runs WHERE data_source = ?",
            (data_source,),
        ).fetchone()
    else:
        result = conn.execute("SELECT MAX(id) FROM analysis_runs").fetchone()
    return result[0] if result and result[0] else 0


def complete_analysis_run(
    conn: sqlite3.Connection,
    run_id: int,
    analyzed: int,
    with_findings: int,
    total_findings: int,
    status: str = "completed",
) -> None:
    """Complete an analysis run.

    Args:
        conn: Database connection.
        run_id: Run ID.
        analyzed: Number of files analyzed.
        with_findings: Number of files with findings.
        total_findings: Total findings count.
        status: Final status.
    """
    conn.execute(
        """
        UPDATE analysis_runs
        SET completed_at = ?, analyzed_files = ?, files_with_findings = ?,
            total_findings = ?, status = ?
        WHERE id = ?
        """,
        (datetime.now().isoformat(), analyzed, with_findings, total_findings, status, run_id),
    )
    conn.commit()


# ============================================================================
# File analysis operations
# ============================================================================


def insert_file_analysis(
    conn: sqlite3.Connection,
    run_id: int,
    file_id: int,
    status: str,
    error: str | None = None,
    duration: float = 0,
) -> int:
    """Insert file analysis result.

    Args:
        conn: Database connection.
        run_id: Analysis run ID.
        file_id: File ID.
        status: Analysis status.
        error: Error message if failed.
        duration: Analysis duration in seconds.

    Returns:
        File analysis ID.
    """
    cursor = conn.execute(
        """
        INSERT OR REPLACE INTO file_analyses
        (run_id, file_id, status, error, duration_seconds)
        VALUES (?, ?, ?, ?, ?)
        """,
        (run_id, file_id, status, error, duration),
    )
    conn.commit()
    return int(cursor.lastrowid) if cursor.lastrowid else 0


def get_timed_out_patterns(
    conn: sqlite3.Connection,
    run_id: int,
) -> list[dict[str, Any]]:
    """Get patterns that timed out in a run.

    Args:
        conn: Database connection.
        run_id: Analysis run ID.

    Returns:
        List of dicts with file_path, file_id, file_analysis_id, and pattern_id.
    """
    query = """
        SELECT f.file_path, f.id as file_id, fa.id as file_analysis_id, pr.pattern_id
        FROM pattern_runs pr
        JOIN file_analyses fa ON pr.file_analysis_id = fa.id
        JOIN files f ON fa.file_id = f.id
        WHERE fa.run_id = ? AND pr.status = 'timeout'
        ORDER BY f.file_path, pr.pattern_id
    """
    results = []
    for row in conn.execute(query, (run_id,)):
        results.append(
            {
                "file_path": row[0],
                "file_id": row[1],
                "file_analysis_id": row[2],
                "pattern_id": row[3],
            }
        )
    return results


def update_pattern_run(
    conn: sqlite3.Connection,
    file_analysis_id: int,
    pattern_id: str,
    status: str,
    detected: str | None = None,
    confidence: float | None = None,
    reasoning: str | None = None,
    error_message: str | None = None,
) -> None:
    """Update an existing pattern run result.

    Args:
        conn: Database connection.
        file_analysis_id: Parent file analysis ID.
        pattern_id: Pattern ID.
        status: New status (success, timeout, etc.).
        detected: Detection result (yes, no, context-dependent).
        confidence: Confidence score.
        reasoning: LLM reasoning.
        error_message: Error message if failed.
    """
    conn.execute(
        """
        UPDATE pattern_runs
        SET status = ?, detected = ?, confidence = ?, reasoning = ?, error_message = ?
        WHERE file_analysis_id = ? AND pattern_id = ?
        """,
        (status, detected, confidence, reasoning, error_message, file_analysis_id, pattern_id),
    )
    conn.commit()


def insert_pattern_runs(
    conn: sqlite3.Connection,
    file_analysis_id: int,
    checked_patterns: list[dict[str, Any]],
    failed_patterns: list[dict[str, Any]],
) -> int:
    """Insert pattern run results for a file analysis.

    Args:
        conn: Database connection.
        file_analysis_id: Parent file analysis ID.
        checked_patterns: List of successful pattern check results.
        failed_patterns: List of failed pattern results.

    Returns:
        Number of pattern runs inserted.
    """
    count = 0

    # Insert successful pattern runs
    for pattern in checked_patterns:
        conn.execute(
            """
            INSERT OR REPLACE INTO pattern_runs
            (file_analysis_id, pattern_id, status, detected, confidence, reasoning)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                file_analysis_id,
                pattern.get("pattern_id", ""),
                "success",
                pattern.get("detected", ""),
                pattern.get("confidence", 0),
                pattern.get("reasoning", ""),
            ),
        )
        count += 1

    # Insert failed pattern runs
    for pattern in failed_patterns:
        conn.execute(
            """
            INSERT OR REPLACE INTO pattern_runs
            (file_analysis_id, pattern_id, status, error_message)
            VALUES (?, ?, ?, ?)
            """,
            (
                file_analysis_id,
                pattern.get("pattern_id", ""),
                pattern.get("error_type", "api_error"),
                pattern.get("error_message", ""),
            ),
        )
        count += 1

    conn.commit()
    return count


def insert_findings(
    conn: sqlite3.Connection,
    file_analysis_id: int,
    findings: list[dict[str, Any]],
) -> int:
    """Insert findings for a file analysis.

    Args:
        conn: Database connection.
        file_analysis_id: Parent file analysis ID.
        findings: List of finding dicts from scicode-lint.

    Returns:
        Number of findings inserted.
    """
    count = 0
    for finding in findings:
        location = finding.get("location", {})
        lines = location.get("lines", [])

        conn.execute(
            """
            INSERT INTO findings
            (file_analysis_id, pattern_id, category, severity, confidence,
             issue, explanation, suggestion, reasoning, lines, snippet)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_analysis_id,
                finding.get("pattern_id", finding.get("id", "")),
                finding.get("category", ""),
                finding.get("severity", ""),
                finding.get("confidence", 0),
                finding.get("issue", ""),
                finding.get("explanation", ""),
                finding.get("suggestion", ""),
                finding.get("reasoning", ""),
                json.dumps(lines),
                location.get("snippet", ""),
            ),
        )
        count += 1

    conn.commit()
    return count


# ============================================================================
# Finding verification operations
# ============================================================================


def save_verification(
    conn: sqlite3.Connection,
    finding_id: int,
    status: str,
    reasoning: str,
    model: str,
) -> None:
    """Save or update a finding verification result.

    Args:
        conn: Database connection.
        finding_id: Finding ID being verified.
        status: Verification status (valid, invalid, uncertain, error).
        reasoning: Claude's reasoning.
        model: Model used for verification.
    """
    conn.execute(
        """
        INSERT OR REPLACE INTO finding_verifications
        (finding_id, status, reasoning, model)
        VALUES (?, ?, ?, ?)
        """,
        (finding_id, status, reasoning, model),
    )
    conn.commit()


def get_verification_stats(conn: sqlite3.Connection, run_id: int | None = None) -> dict[str, Any]:
    """Get verification statistics for a run.

    Args:
        conn: Database connection.
        run_id: Analysis run ID, or None for latest.

    Returns:
        Dict with verification stats.
    """
    if run_id is None:
        cursor = conn.execute("SELECT id FROM analysis_runs ORDER BY started_at DESC LIMIT 1")
        row = cursor.fetchone()
        if not row:
            return {}
        run_id = row["id"]

    # Get counts by status
    cursor = conn.execute(
        """
        SELECT fv.status, COUNT(*) as count
        FROM finding_verifications fv
        JOIN findings fn ON fn.id = fv.finding_id
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        WHERE fa.run_id = ?
        GROUP BY fv.status
        """,
        (run_id,),
    )
    stats: dict[str, Any] = {"by_status": {}}
    for row in cursor.fetchall():
        stats["by_status"][row["status"]] = row["count"]

    # Calculate totals
    valid = stats["by_status"].get("valid", 0)
    invalid = stats["by_status"].get("invalid", 0)
    uncertain = stats["by_status"].get("uncertain", 0)
    total = valid + invalid + uncertain

    stats["total_verified"] = total
    stats["valid"] = valid
    stats["invalid"] = invalid
    stats["uncertain"] = uncertain
    stats["precision"] = (valid / total * 100) if total > 0 else 0

    # Get verification date
    cursor = conn.execute(
        """
        SELECT MAX(fv.verified_at) as last_verified, fv.model
        FROM finding_verifications fv
        JOIN findings fn ON fn.id = fv.finding_id
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        WHERE fa.run_id = ?
        """,
        (run_id,),
    )
    row = cursor.fetchone()
    if row and row["last_verified"]:
        stats["verified_at"] = row["last_verified"]
        stats["model"] = row["model"]

    return stats


def get_finding_verification(conn: sqlite3.Connection, finding_id: int) -> dict[str, Any] | None:
    """Get verification result for a specific finding.

    Args:
        conn: Database connection.
        finding_id: Finding ID.

    Returns:
        Dict with status and reasoning, or None if not verified.
    """
    cursor = conn.execute(
        "SELECT status, reasoning, model FROM finding_verifications WHERE finding_id = ?",
        (finding_id,),
    )
    row = cursor.fetchone()
    if row:
        return {"status": row["status"], "reasoning": row["reasoning"], "model": row["model"]}
    return None


# ============================================================================
# Statistics queries
# ============================================================================


def get_run_stats(conn: sqlite3.Connection, run_id: int | None = None) -> AnalysisStats:
    """Get statistics for a specific run or the latest run.

    Args:
        conn: Database connection.
        run_id: Specific run ID, or None for latest.

    Returns:
        AnalysisStats with aggregated data.
    """
    # Get run info
    if run_id is None:
        cursor = conn.execute("SELECT * FROM analysis_runs ORDER BY started_at DESC LIMIT 1")
    else:
        cursor = conn.execute("SELECT * FROM analysis_runs WHERE id = ?", (run_id,))

    run_row = cursor.fetchone()
    if not run_row:
        return AnalysisStats(run_id=0, run_date=datetime.now())

    run_id = run_row["id"]

    # Compute actual counts from file_analyses (more reliable than run record)
    cursor = conn.execute(
        """
        SELECT
            COUNT(DISTINCT fa.id) as analyzed_files,
            COUNT(DISTINCT CASE WHEN fn.id IS NOT NULL THEN fa.file_id END) as files_with_findings,
            COUNT(fn.id) as total_findings
        FROM file_analyses fa
        LEFT JOIN findings fn ON fn.file_analysis_id = fa.id
        WHERE fa.run_id = ?
        """,
        (run_id,),
    )
    counts = cursor.fetchone()

    stats = AnalysisStats(
        run_id=run_id,
        run_date=datetime.fromisoformat(run_row["started_at"]),
        data_source=run_row["data_source"] or "papers_with_code",
        total_files=run_row["total_files"],
        analyzed_files=counts["analyzed_files"] or 0,
        files_with_findings=counts["files_with_findings"] or 0,
        total_findings=counts["total_findings"] or 0,
    )

    # Total repos and repos with findings for this run
    cursor = conn.execute(
        """
        SELECT
            COUNT(DISTINCT f.repo_id) as total_repos,
            COUNT(DISTINCT CASE WHEN fn.id IS NOT NULL THEN f.repo_id END) as repos_with_findings
        FROM files f
        JOIN file_analyses fa ON fa.file_id = f.id
        LEFT JOIN findings fn ON fn.file_analysis_id = fa.id
        WHERE fa.run_id = ?
        """,
        (run_id,),
    )
    repo_counts = cursor.fetchone()
    stats.total_repos = repo_counts["total_repos"] or 0
    stats.repos_with_findings = repo_counts["repos_with_findings"] or 0

    # Total papers and papers with findings (scoped to this run)
    cursor = conn.execute(
        """
        SELECT
            COUNT(DISTINCT r.paper_id) as total_papers,
            COUNT(DISTINCT CASE WHEN fn.id IS NOT NULL THEN r.paper_id END) as papers_with_findings
        FROM repos r
        JOIN files f ON f.repo_id = r.id
        JOIN file_analyses fa ON fa.file_id = f.id
        LEFT JOIN findings fn ON fn.file_analysis_id = fa.id
        WHERE fa.run_id = ? AND r.paper_id IS NOT NULL
        """,
        (run_id,),
    )
    paper_counts = cursor.fetchone()
    stats.total_papers = paper_counts["total_papers"] or 0
    stats.papers_with_findings = paper_counts["papers_with_findings"] or 0

    # Success rate
    if stats.total_files > 0:
        stats.analysis_success_rate = 100 * stats.analyzed_files / stats.total_files

    # Finding rate
    if stats.analyzed_files > 0:
        stats.finding_rate = 100 * stats.files_with_findings / stats.analyzed_files

    # By domain
    cursor = conn.execute(
        """
        SELECT r.domain,
               COUNT(DISTINCT f.id) as total_files,
               COUNT(DISTINCT CASE WHEN fa.status = 'success' THEN f.id END) as analyzed,
               COUNT(DISTINCT CASE WHEN fn.id IS NOT NULL THEN f.id END) as with_findings,
               COUNT(fn.id) as total_findings
        FROM repos r
        JOIN files f ON f.repo_id = r.id
        JOIN file_analyses fa ON fa.file_id = f.id AND fa.run_id = ?
        LEFT JOIN findings fn ON fn.file_analysis_id = fa.id
        GROUP BY r.domain
        ORDER BY total_files DESC
        """,
        (run_id,),
    )
    for row in cursor.fetchall():
        analyzed = row["analyzed"] or 0
        stats.by_domain.append(
            DomainStats(
                domain=row["domain"] or "unknown",
                total_files=row["total_files"],
                analyzed_files=analyzed,
                files_with_findings=row["with_findings"],
                total_findings=row["total_findings"],
                finding_rate=100 * row["with_findings"] / analyzed if analyzed > 0 else 0,
            )
        )

    # By category
    cursor = conn.execute(
        """
        SELECT category,
               COUNT(*) as count,
               COUNT(DISTINCT fa.file_id) as unique_files,
               COUNT(DISTINCT f.repo_id) as unique_repos
        FROM findings fn
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        JOIN files f ON f.id = fa.file_id
        WHERE fa.run_id = ?
        GROUP BY category
        ORDER BY count DESC
        """,
        (run_id,),
    )
    for row in cursor.fetchall():
        stats.by_category.append(
            CategoryStats(
                category=row["category"],
                count=row["count"],
                unique_files=row["unique_files"],
                unique_repos=row["unique_repos"],
            )
        )

    # By pattern (top 15)
    cursor = conn.execute(
        """
        SELECT pattern_id, category,
               COUNT(*) as count,
               COUNT(DISTINCT fa.file_id) as unique_files,
               AVG(confidence) as avg_confidence
        FROM findings fn
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        WHERE fa.run_id = ?
        GROUP BY pattern_id
        ORDER BY count DESC
        LIMIT 15
        """,
        (run_id,),
    )
    for row in cursor.fetchall():
        stats.by_pattern.append(
            PatternStats(
                pattern_id=row["pattern_id"],
                category=row["category"],
                count=row["count"],
                unique_files=row["unique_files"],
                avg_confidence=row["avg_confidence"],
            )
        )

    # By severity
    cursor = conn.execute(
        """
        SELECT severity, COUNT(*) as count
        FROM findings fn
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        WHERE fa.run_id = ?
        GROUP BY severity
        """,
        (run_id,),
    )
    stats.by_severity = {row["severity"]: row["count"] for row in cursor.fetchall()}

    return stats


def list_runs(
    conn: sqlite3.Connection,
    limit: int = 10,
    data_source: str | None = None,
) -> list[AnalysisRun]:
    """List recent analysis runs.

    Args:
        conn: Database connection.
        limit: Maximum runs to return.
        data_source: Optional filter by data source.

    Returns:
        List of AnalysisRun models.
    """
    if data_source:
        cursor = conn.execute(
            """
            SELECT * FROM analysis_runs
            WHERE data_source = ?
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (data_source, limit),
        )
    else:
        cursor = conn.execute(
            """
            SELECT * FROM analysis_runs
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,),
        )
    runs = []
    for row in cursor.fetchall():
        runs.append(
            AnalysisRun(
                run_id=row["id"],
                started_at=datetime.fromisoformat(row["started_at"]),
                completed_at=(
                    datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
                ),
                status=RunStatus(row["status"]) if row["status"] else RunStatus.RUNNING,
                data_source=row["data_source"] or "papers_with_code",
                total_files=row["total_files"],
                analyzed_files=row["analyzed_files"],
                files_with_findings=row["files_with_findings"],
                total_findings=row["total_findings"],
                config=json.loads(row["config"]) if row["config"] else {},
                git_commit=row["git_commit"] or "",
                model_name=row["model_name"] or "",
            )
        )
    return runs


def get_analyzed_file_ids(conn: sqlite3.Connection, run_id: int) -> set[int]:
    """Get set of file IDs already analyzed in a run.

    Args:
        conn: Database connection.
        run_id: Analysis run ID.

    Returns:
        Set of file IDs that have been analyzed.
    """
    cursor = conn.execute(
        "SELECT file_id FROM file_analyses WHERE run_id = ?",
        (run_id,),
    )
    return {row["file_id"] for row in cursor.fetchall()}


def get_incomplete_run(conn: sqlite3.Connection) -> int | None:
    """Get the latest incomplete (running) analysis run.

    Args:
        conn: Database connection.

    Returns:
        Run ID if there's an incomplete run, None otherwise.
    """
    cursor = conn.execute(
        """
        SELECT id FROM analysis_runs
        WHERE status = 'running'
        ORDER BY started_at DESC
        LIMIT 1
        """
    )
    row = cursor.fetchone()
    return row["id"] if row else None


def print_stats(conn: sqlite3.Connection, run_id: int | None = None) -> None:
    """Print statistics from database.

    Args:
        conn: Database connection.
        run_id: Specific run ID, or None for latest.
    """
    stats = get_run_stats(conn, run_id)

    logger.info("=" * 60)
    logger.info(f"Analysis Run #{stats.run_id} - {stats.run_date.strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 60)
    logger.info(f"  Total repos: {stats.total_repos:,}")
    logger.info(f"  Total files: {stats.total_files:,}")
    pct = stats.analysis_success_rate
    logger.info(f"  Analyzed successfully: {stats.analyzed_files:,} ({pct:.1f}%)")
    logger.info(f"  Files with findings: {stats.files_with_findings:,} ({stats.finding_rate:.1f}%)")
    logger.info(f"  Total findings: {stats.total_findings:,}")

    if stats.by_category:
        logger.info("  By category:")
        for cat in stats.by_category:
            logger.info(f"    {cat.category}: {cat.count:,} ({cat.unique_files} files)")

    if stats.by_severity:
        logger.info("  By severity:")
        for sev in ["critical", "high", "medium", "low"]:
            if sev in stats.by_severity:
                logger.info(f"    {sev}: {stats.by_severity[sev]:,}")

    if stats.by_domain:
        logger.info("  By domain:")
        for domain in stats.by_domain:
            logger.info(
                f"    {domain.domain}: {domain.files_with_findings}/{domain.analyzed_files} "
                f"({domain.finding_rate:.1f}%)"
            )
