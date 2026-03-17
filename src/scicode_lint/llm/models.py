"""Pydantic models for structured LLM output."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class NamedLocation(BaseModel):
    """Name-based location for detected issues.

    LLMs are good at identifying function/class names but unreliable at counting
    line numbers. This schema captures what LLMs do well, and we resolve to actual
    lines using AST parsing.
    """

    name: str = Field(
        description="Name of the function, class, or method where issue occurs. "
        "Use qualified names for methods (e.g., 'Trainer.train'). "
        "Use '<module>' for module-level code."
    )
    location_type: Literal["function", "class", "method", "module"] = Field(
        description="Type of code construct: 'function' for standalone functions, "
        "'method' for class methods, 'class' for class definitions, "
        "'module' for module-level code."
    )
    near_line: int | None = Field(
        default=None,
        description="Approximate line number where issue occurs (optional hint). "
        "Used to disambiguate when multiple definitions have the same name.",
    )


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
    # Name-based location instead of line numbers. LLMs are better at names than lines.
    # For detected="no", use null. For detected="yes", provide name-based location.
    location: NamedLocation | None = Field(
        default=None,
        description="Location of the issue. REQUIRED when detected='yes' or 'context-dependent'. "
        "Use null when detected='no'.",
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
    thinking: str | None = Field(
        default=None,
        description="Model's internal reasoning/thinking (extracted from <think> tags). "
        "Not part of the structured output schema - populated automatically if present.",
    )

    @model_validator(mode="after")
    def validate_location_when_detected(self) -> "DetectionResult":
        """Require location when issue is detected.

        Raises ValueError if detected="yes" or "context-dependent" but location is None.
        This triggers retry logic in the client and prevents storing invalid findings.
        """
        if self.detected in ("yes", "context-dependent") and not self.location:
            raise ValueError(
                f"Location required when detected='{self.detected}'. "
                "Model must provide function/class name where issue occurs."
            )
        return self
