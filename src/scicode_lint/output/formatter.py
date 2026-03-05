"""Format linter findings for output."""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from scicode_lint.config import Severity


class Location(BaseModel):
    """Location of a finding in code - line numbers with optional snippet."""

    lines: list[int] = Field(
        description="Line numbers where the issue occurs", examples=[[15, 16, 17], [42]]
    )
    snippet: str = Field(default="", description="Actual code snippet from those lines")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "lines": [15, 16, 17],
                    "snippet": "all_data = torch.cat([train_data, test_data])\\nmean = all_data.mean()",
                }
            ]
        }
    )


class Finding(BaseModel):
    """A single linter finding.

    Designed for both human and GenAI agent consumption.
    Includes structured location data and actionable explanations.
    """

    id: str = Field(description="Pattern ID (e.g., 'ml-001')")
    category: str = Field(description="Pattern category (e.g., 'ai-training')")
    severity: Severity = Field(description="Issue severity level")
    location: Location = Field(description="Where the issue was found")
    issue: str = Field(description="Brief issue description")
    explanation: str = Field(description="Detailed explanation for humans and agents")
    suggestion: str = Field(description="Actionable suggestion to fix the issue")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    reasoning: str = Field(default="", description="LLM's reasoning for this detection")
    detection_type: Literal["yes", "context-dependent"] = Field(
        default="yes", description="Whether this is a definite issue or context-dependent"
    )

    model_config = ConfigDict(
        use_enum_values=True  # Serialize Severity enum as string value
    )


class LintError(BaseModel):
    """An error that occurred during linting.

    Designed for both human and GenAI agent consumption.
    """

    file: Path = Field(description="File where error occurred")
    error_type: str = Field(description="Type of error (e.g., 'ContextLengthError')")
    message: str = Field(description="Human-readable error message")
    details: dict[str, Any] | None = Field(
        default=None, description="Structured error details (optional)"
    )

    model_config = ConfigDict(
        # Automatically serialize Path as string
        json_encoders={Path: str}
    )


class LintResult(BaseModel):
    """Result of linting a file."""

    file: Path = Field(description="Path to the linted file")
    findings: list[Finding] = Field(
        default_factory=list, description="List of findings detected in the file"
    )
    error: LintError | None = Field(
        default=None, description="Error that occurred during linting (if any)"
    )

    model_config = ConfigDict(json_encoders={Path: str}, use_enum_values=True)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def summary(self) -> dict[str, Any]:
        """Generate summary statistics (computed on-the-fly)."""
        severity_counts = Counter(f.severity for f in self.findings)
        category_counts = Counter(f.category for f in self.findings)

        return {
            "total_findings": len(self.findings),
            "by_severity": {
                (k.value if hasattr(k, "value") else k): v for k, v in severity_counts.items()
            },
            "by_category": dict(category_counts),
        }


def format_findings(
    results: list[LintResult],
    output_format: str = "text",
) -> str:
    """
    Format linter results for output.

    Supports both human-readable text and structured JSON output.
    Designed for dual audience (humans and GenAI agents).

    Args:
        results: List of lint results
        output_format: 'text' or 'json'

    Returns:
        Formatted output string

    Example:
        >>> results = [LintResult(...)]
        >>> print(format_findings(results, "json"))
    """
    if output_format == "json":
        return _format_json(results)
    else:
        return _format_text(results)


def _format_json(results: list[LintResult]) -> str:
    """Format as JSON for GenAI agent consumption.

    Includes both findings and errors in structured format.

    Args:
        results: List of lint results

    Returns:
        JSON string with findings and errors
    """
    data = [r.model_dump(mode="json") for r in results]
    return json.dumps(data, indent=2)


def _format_text(results: list[LintResult]) -> str:
    """Format as human-readable text.

    Includes visual indicators (emojis) and clear formatting.

    Args:
        results: List of lint results

    Returns:
        Human-readable text output
    """
    lines = []

    for result in results:
        # Show errors first if present
        if result.error:
            lines.append(f"⚠️  {result.file} — Error during linting")
            lines.append(f"   {result.error.error_type}: {result.error.message}")
            lines.append("")
            continue

        if not result.findings:
            continue

        lines.append(f"{result.file} — {len(result.findings)} issues found\n")

        for finding in result.findings:
            # Severity icon (add '?' for context-dependent findings)
            severity_icon = {
                Severity.CRITICAL: "🔴 CRITICAL",
                Severity.HIGH: "🟠 HIGH",
                Severity.MEDIUM: "🟡 MEDIUM",
            }[finding.severity]

            # Add indicator for context-dependent findings
            if finding.detection_type == "context-dependent":
                icon = f"{severity_icon}?"
            else:
                icon = severity_icon

            # Location description - show line numbers
            loc = finding.location
            if loc.lines:
                if len(loc.lines) == 1:
                    location = f"line {loc.lines[0]}"
                else:
                    location = f"lines {loc.lines[0]}-{loc.lines[-1]}"
            else:
                location = "unknown location"

            lines.append(f"{icon} [{location}] {finding.issue}")
            lines.append(f"   {finding.explanation}")
            if finding.reasoning:
                lines.append(f"   Reasoning: {finding.reasoning}")
            if finding.location.snippet:
                lines.append("   Code:")
                for snippet_line in finding.location.snippet.split("\n"):
                    lines.append(f"      {snippet_line}")
            lines.append("")

    return "\n".join(lines)


def get_json_schemas() -> dict[str, Any]:
    """
    Get JSON schemas for all output models.

    Useful for API documentation, OpenAPI specs, and GenAI agent integration.

    Returns:
        Dictionary with schema for each model type

    Example:
        >>> from scicode_lint.output.formatter import get_json_schemas
        >>> schemas = get_json_schemas()
        >>> print(schemas["Finding"]["properties"]["confidence"])
    """
    return {
        "Location": Location.model_json_schema(),
        "Finding": Finding.model_json_schema(),
        "LintError": LintError.model_json_schema(),
        "LintResult": LintResult.model_json_schema(),
    }
