"""Pydantic models for structured LLM output."""

from typing import Literal

from pydantic import BaseModel, Field


class DetectionResult(BaseModel):
    """
    Three-way detection result: yes/no/context-dependent with reasoning.

    This format allows the LLM to express uncertainty when the answer
    depends on context, coding style, or interpretation.
    """

    detected: Literal["yes", "no", "context-dependent"] = Field(
        description="Whether the issue was detected: 'yes' (definite issue), "
        "'no' (no issue), or 'context-dependent' (depends on context/style)"
    )
    lines: list[int] = Field(
        default_factory=list,
        description="Line numbers where the issue occurs (empty if not detected). "
        "Example: [15, 16, 17] means the issue is on lines 15-17.",
    )
    confidence: float = Field(
        description="Confidence level from 0.0 to 1.0.",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        description="Brief explanation (1-2 sentences) of why this decision was made. "
        "Explain what pattern was detected or why it's not an issue.",
    )
