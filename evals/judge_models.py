"""Pydantic models for LLM-as-judge evaluation."""

from typing import Literal

from pydantic import BaseModel, Field


class JudgeVerdict(BaseModel):
    """Judge's verdict on whether linter output matches expected behavior."""

    verdict: Literal["yes", "no", "partial"] = Field(
        description="Whether linter output matches test case intent"
    )
    reasoning: str = Field(description="One sentence explaining the verdict")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Judge's confidence in this verdict (0.0-1.0)"
    )


class TestCaseEvaluation(BaseModel):
    """Evaluation result for a single test case."""

    test_file: str
    test_type: Literal["positive", "negative", "context_dependent"]
    expected_behavior: str

    # New linter output format (line-based with reasoning)
    linter_detected: Literal["yes", "no", "context-dependent"]
    linter_lines: list[int]
    linter_snippet: str
    linter_reasoning: str
    linter_issue: str | None
    linter_confidence: float

    # Judge evaluation
    judge_verdict: Literal["yes", "no", "partial"]
    judge_reasoning: str
    judge_confidence: float


class PatternJudgeMetrics(BaseModel):
    """LLM-as-judge metrics for a single pattern."""

    pattern_id: str
    total_tests: int

    # By test type
    positive_accuracy: float  # % where verdict = "yes"
    negative_accuracy: float
    context_accuracy: float  # % where verdict = "yes" or "partial"

    # Overall
    overall_accuracy: float  # (correct + 0.5*partial) / total

    # Detailed counts
    correct_count: int  # verdict = "yes"
    partial_count: int  # verdict = "partial"
    incorrect_count: int  # verdict = "no"

    # Evaluations
    evaluations: list[TestCaseEvaluation]


class OverallJudgeMetrics(BaseModel):
    """Aggregated LLM-as-judge metrics across all patterns."""

    total_patterns: int
    total_tests: int

    # By test type
    positive_accuracy: float
    negative_accuracy: float
    context_accuracy: float

    # Overall
    overall_accuracy: float

    # Per-pattern metrics
    patterns: list[PatternJudgeMetrics]

    # Summary statistics
    avg_judge_confidence: float
    patterns_above_threshold: int  # accuracy >= 0.85
