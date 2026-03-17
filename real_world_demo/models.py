"""Pydantic models for real-world demo data structures.

Defines schemas for repos, files, findings, and analysis runs.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class CloneStatus(StrEnum):
    """Clone operation status."""

    PENDING = "pending"
    SUCCESS = "success"
    NOT_FOUND = "not_found"
    PRIVATE = "private"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    ERROR = "error"


class AnalysisStatus(StrEnum):
    """File analysis status."""

    PENDING = "pending"
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    SKIPPED = "skipped"


class Severity(StrEnum):
    """Finding severity level."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RunStatus(StrEnum):
    """Analysis run status."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FileClassification(StrEnum):
    """Self-contained file classification."""

    SELF_CONTAINED = "self_contained"
    FRAGMENT = "fragment"
    UNCERTAIN = "uncertain"
    SKIPPED = "skipped"  # No ML indicators


# ============================================================================
# Core domain models
# ============================================================================


class Paper(BaseModel):
    """Scientific paper metadata from PapersWithCode."""

    paper_url: str = Field(description="URL on PapersWithCode")
    title: str = Field(default="", description="Paper title")
    arxiv_id: str = Field(default="", description="ArXiv identifier")
    abstract: str = Field(default="", description="Paper abstract")
    tasks: list[str] = Field(default_factory=list, description="PWC task categories")
    matched_domain: str = Field(default="", description="Scientific domain we matched")


class Repo(BaseModel):
    """GitHub repository linked to a paper."""

    repo_url: str = Field(description="GitHub repository URL")
    repo_name: str = Field(description="Normalized name (owner__repo)")
    paper_url: str = Field(default="", description="Associated paper URL")
    paper_title: str = Field(default="", description="Associated paper title")
    domain: str = Field(default="", description="Scientific domain")
    arxiv_id: str = Field(default="", description="ArXiv ID of associated paper")
    clone_status: CloneStatus = Field(default=CloneStatus.PENDING)
    clone_error: str | None = Field(default=None, description="Clone error message")
    cloned_at: datetime | None = Field(default=None)


class FileRecord(BaseModel):
    """Python file from a cloned repository."""

    file_path: str = Field(description="Path relative to collected_code/files/")
    original_path: str = Field(description="Path within original repo")
    repo_name: str = Field(description="Parent repo name")
    is_notebook: bool = Field(default=False, description="Is Jupyter notebook")
    file_size: int = Field(default=0, description="File size in bytes")
    line_count: int = Field(default=0, description="Number of lines")
    ml_imports: list[str] = Field(default_factory=list, description="ML libraries used")
    scientific_imports: list[str] = Field(
        default_factory=list, description="Scientific libraries used"
    )
    prefilter_passed: bool = Field(default=True, description="Passed LLM prefilter")
    prefilter_response: str = Field(default="", description="Prefilter LLM response")


class FindingLocation(BaseModel):
    """Location of a finding in source code."""

    name: str | None = Field(default=None, description="Function/class/method name")
    location_type: str | None = Field(default=None, description="function, method, class, module")
    lines: list[int] = Field(default_factory=list, description="Line numbers (full range)")
    focus_line: int | None = Field(default=None, description="Specific line with the issue")
    snippet: str = Field(default="", description="Code snippet")


class Finding(BaseModel):
    """Single scicode-lint finding."""

    pattern_id: str = Field(description="Pattern ID (e.g., rep-002)")
    category: str = Field(description="Category (e.g., scientific-reproducibility)")
    severity: Severity = Field(description="Finding severity")
    confidence: float = Field(ge=0, le=1, description="Detection confidence")
    issue: str = Field(description="Short issue description")
    explanation: str = Field(default="", description="Detailed explanation")
    suggestion: str = Field(default="", description="How to fix")
    reasoning: str = Field(default="", description="LLM reasoning for detection")
    location: FindingLocation = Field(default_factory=FindingLocation)


class FileAnalysis(BaseModel):
    """Analysis result for a single file."""

    file_path: str = Field(description="Path to analyzed file")
    status: AnalysisStatus = Field(description="Analysis status")
    error: str | None = Field(default=None, description="Error message if failed")
    findings: list[Finding] = Field(default_factory=list)
    duration_seconds: float = Field(default=0, description="Analysis duration")
    analyzed_at: datetime = Field(default_factory=datetime.now)


class AnalysisRun(BaseModel):
    """A single analysis run (can have multiple per day)."""

    run_id: int = Field(description="Auto-generated run ID")
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = Field(default=None)
    status: RunStatus = Field(default=RunStatus.RUNNING)
    data_source: str = Field(default="papers_with_code", description="Data source identifier")
    total_files: int = Field(default=0, description="Files to analyze")
    analyzed_files: int = Field(default=0, description="Files successfully analyzed")
    files_with_findings: int = Field(default=0)
    total_findings: int = Field(default=0)
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Run configuration (max_files, max_concurrent, etc.)",
    )
    git_commit: str = Field(default="", description="Git commit of scicode-lint")
    model_name: str = Field(default="", description="LLM model used")


class RepoScan(BaseModel):
    """A repository scan run for finding self-contained ML files."""

    scan_id: int = Field(description="Auto-generated scan ID")
    repo_id: int = Field(description="Repository being scanned")
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = Field(default=None)
    status: RunStatus = Field(default=RunStatus.RUNNING)
    total_files: int = Field(default=0, description="Total Python files found")
    passed_ml_import_filter: int = Field(default=0, description="Files with ML imports")
    self_contained: int = Field(default=0, description="Files classified as self-contained")
    fragments: int = Field(default=0, description="Files classified as fragments")
    uncertain: int = Field(default=0, description="Files with uncertain classification")
    skipped: int = Field(default=0, description="Files skipped (no ML indicators)")
    duration_seconds: float = Field(default=0, description="Scan duration")
    model_name: str = Field(default="", description="LLM model used for classification")


class ScanStats(BaseModel):
    """Aggregated scan statistics."""

    total_scans: int = 0
    total_repos_scanned: int = 0
    total_files_scanned: int = 0
    total_self_contained: int = 0
    total_fragments: int = 0
    total_skipped: int = 0
    avg_self_contained_per_repo: float = 0
    avg_scan_duration: float = 0


class PrefilterRun(BaseModel):
    """A prefilter run for classifying files as self-contained vs fragments."""

    run_id: int = Field(description="Auto-generated run ID")
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = Field(default=None)
    status: RunStatus = Field(default=RunStatus.RUNNING)
    data_source: str = Field(default="papers_with_code", description="Data source identifier")
    total_files: int = Field(default=0, description="Total files to classify")
    self_contained: int = Field(default=0, description="Files classified as self-contained")
    fragments: int = Field(default=0, description="Files classified as fragments")
    uncertain: int = Field(default=0, description="Files with uncertain classification")
    errors: int = Field(default=0, description="Files that errored during classification")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    config: dict[str, Any] = Field(default_factory=dict, description="Run configuration")
    model_name: str = Field(default="", description="LLM model used for classification")
    git_commit: str = Field(default="", description="Git commit of scicode-lint")
    parent_run_id: int | None = Field(default=None, description="ID of parent run if reusing files")


class PrefilterFileResult(BaseModel):
    """Classification result for a single file in a prefilter run."""

    file_id: int = Field(description="File ID in files table")
    file_path: str = Field(description="Path to the file")
    repo_id: int | None = Field(default=None, description="Repository ID")
    classification: str = Field(description="Classification: self_contained, fragment, uncertain")
    confidence: float | None = Field(default=None, description="Classification confidence")
    reasoning: str | None = Field(default=None, description="LLM reasoning for classification")


# ============================================================================
# Aggregated statistics
# ============================================================================


class DomainStats(BaseModel):
    """Statistics for a scientific domain."""

    domain: str
    total_repos: int = 0
    total_files: int = 0
    analyzed_files: int = 0
    files_with_findings: int = 0
    total_findings: int = 0
    finding_rate: float = Field(default=0, description="% files with findings")


class CategoryStats(BaseModel):
    """Statistics for a finding category."""

    category: str
    count: int = 0
    unique_files: int = 0
    unique_repos: int = 0


class PatternStats(BaseModel):
    """Statistics for a specific pattern."""

    pattern_id: str
    category: str = ""
    count: int = 0
    unique_files: int = 0
    avg_confidence: float = 0


class AnalysisStats(BaseModel):
    """Aggregated analysis statistics."""

    # Run info
    run_id: int
    run_date: datetime
    data_source: str = "papers_with_code"

    # Totals
    total_repos: int = 0
    repos_with_findings: int = 0
    total_papers: int = 0
    papers_with_findings: int = 0
    total_files: int = 0
    analyzed_files: int = 0
    files_with_findings: int = 0
    total_findings: int = 0

    # Rates
    analysis_success_rate: float = Field(default=0, description="% files analyzed OK")
    finding_rate: float = Field(default=0, description="% analyzed files with findings")

    # Breakdowns
    by_domain: list[DomainStats] = Field(default_factory=list)
    by_category: list[CategoryStats] = Field(default_factory=list)
    by_pattern: list[PatternStats] = Field(default_factory=list)
    by_severity: dict[str, int] = Field(default_factory=dict)
