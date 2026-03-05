"""
Pydantic models for TOML pattern files.

These models define the structure of pattern.toml files and provide
validation for pattern data.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PatternMeta(BaseModel):
    """Metadata about the pattern."""

    id: str
    name: str
    category: str  # Category: ai-*, scientific-*
    severity: Literal["critical", "high", "medium"]
    version: str = "1.0.0"
    created: str
    updated: str
    author: str = "scicode-lint"
    description: str
    explanation: str
    research_impact: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    related_patterns: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)


class DetectionConfig(BaseModel):
    """Configuration for pattern detection."""

    question: str
    warning_message: str
    suggestion: Optional[str] = None
    min_confidence: float = 0.85
    code_patterns: list[str] = Field(default_factory=list)
    false_positive_risks: list[str] = Field(default_factory=list)


class TestLocation(BaseModel):
    """Expected location of a finding in test code."""

    type: Literal["function", "class", "method", "module"]
    name: str
    snippet: str


class PositiveTest(BaseModel):
    """Test case that should trigger detection (True Positive)."""

    file: str
    description: str
    expected_location: TestLocation
    expected_issue: str
    min_confidence: float = 0.85


class NegativeTest(BaseModel):
    """Test case that should NOT trigger detection (True Negative)."""

    file: str
    description: str
    max_false_positives: int = 0
    notes: Optional[str] = None


class ContextDependentTest(BaseModel):
    """Test case where detection is acceptable either way (ambiguous context)."""

    file: str
    description: str
    allow_detection: bool = True
    allow_skip: bool = True
    context_notes: str
    rationale_for_detection: Optional[str] = None
    rationale_against_detection: Optional[str] = None


class TestCases(BaseModel):
    """Collection of test cases for the pattern."""

    positive: list[PositiveTest] = Field(default_factory=list)
    negative: list[NegativeTest] = Field(default_factory=list)
    context_dependent: list[ContextDependentTest] = Field(default_factory=list)


class QualityMetrics(BaseModel):
    """Quality metrics for pattern performance."""

    target_precision: float = 0.90
    target_recall: float = 0.80
    target_f1: float = 0.85
    actual_precision: Optional[float] = None
    actual_recall: Optional[float] = None
    actual_f1: Optional[float] = None
    last_evaluated: Optional[str] = None


class AIScience(BaseModel):
    """AI/science-specific metadata."""

    domains: list[str] = Field(default_factory=list)
    audience: list[str] = Field(default_factory=list)
    paper_sections: list[str] = Field(default_factory=list)
    impact_severity: Optional[str] = None
    impact_magnitude: Optional[str] = None
    prevalence: Optional[str] = None
    educational_notes: Optional[str] = None


class PatternTOML(BaseModel):
    """Complete pattern definition from TOML file."""

    meta: PatternMeta
    detection: DetectionConfig
    tests: TestCases
    quality: Optional[QualityMetrics] = Field(default_factory=QualityMetrics)
    ai_science: Optional[AIScience] = None
