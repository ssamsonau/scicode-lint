"""Validation logic for comparing linter findings against ground truth.

Uses function/class-level location matching instead of line numbers to avoid
LLM hallucination issues.
"""

import re
from difflib import SequenceMatcher
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Location(BaseModel):
    """Function/class-level location specification."""

    type: Literal["function", "class", "method", "module"]
    name: str
    snippet: str

    def normalize_name(self) -> str:
        """Normalize function/class name for comparison."""
        # Remove common prefixes/suffixes
        name = self.name.strip()
        # Handle method names (ClassName.method_name)
        if "." in name:
            parts = name.split(".")
            return parts[-1]  # Use just the method name
        return name

    def normalize_snippet(self) -> str:
        """Normalize code snippet for comparison."""
        # Remove whitespace, comments, and normalize
        snippet = self.snippet.strip()
        # Remove inline comments
        snippet = re.sub(r"#.*$", "", snippet, flags=re.MULTILINE)
        # Normalize whitespace
        snippet = re.sub(r"\s+", " ", snippet)
        return snippet.lower()


class ExpectedFinding(BaseModel):
    """Expected finding from ground truth."""

    location: Location
    issue: str
    min_confidence: float = Field(default=0.80, ge=0.0, le=1.0)


class ActualLocation(BaseModel):
    """Line-based location from linter output."""

    lines: list[int] = Field(default_factory=list)


class ActualFinding(BaseModel):
    """Actual finding from linter output."""

    id: str
    category: str
    severity: str
    location: ActualLocation
    issue: str
    explanation: str
    suggestion: str
    confidence: float = Field(ge=0.0, le=1.0)


class LocationMatcher:
    """Fuzzy matching for function/class locations."""

    def __init__(
        self,
        name_exact_threshold: float = 1.0,
        name_fuzzy_threshold: float = 0.85,
        snippet_threshold: float = 0.70,
    ):
        """
        Initialize matcher with thresholds.

        Args:
            name_exact_threshold: Score for exact name match (1.0)
            name_fuzzy_threshold: Minimum score for fuzzy name match (0.85)
            snippet_threshold: Minimum Jaccard similarity for snippet match (0.70)
        """
        self.name_exact_threshold = name_exact_threshold
        self.name_fuzzy_threshold = name_fuzzy_threshold
        self.snippet_threshold = snippet_threshold

    def match_name(self, expected: str, actual: str) -> float:
        """
        Match function/class names.

        Returns:
            1.0 for exact match, 0.85+ for fuzzy match, 0.0 for no match
        """
        expected_norm = expected.lower().strip()
        actual_norm = actual.lower().strip()

        # Exact match
        if expected_norm == actual_norm:
            return 1.0

        # Fuzzy match using SequenceMatcher
        ratio = SequenceMatcher(None, expected_norm, actual_norm).ratio()

        return ratio if ratio >= self.name_fuzzy_threshold else 0.0

    def match_snippet(self, expected: str, actual: str) -> float:
        """
        Match code snippets using token-based Jaccard similarity.

        Returns:
            Similarity score 0.0 to 1.0
        """
        # Tokenize by splitting on non-alphanumeric characters
        expected_tokens = set(re.findall(r"\w+", expected.lower()))
        actual_tokens = set(re.findall(r"\w+", actual.lower()))

        if not expected_tokens or not actual_tokens:
            return 0.0

        # Jaccard similarity
        intersection = expected_tokens & actual_tokens
        union = expected_tokens | actual_tokens

        return len(intersection) / len(union)

    def match_location(self, expected: Location, actual: Location) -> tuple[bool, float]:
        """
        Match two locations.

        Returns:
            (is_match, confidence_score)
        """
        # Type must match exactly
        if expected.type != actual.type:
            return False, 0.0

        # Match names
        name_score = self.match_name(expected.normalize_name(), actual.normalize_name())

        if name_score == 0.0:
            return False, 0.0

        # Match snippets
        snippet_score = self.match_snippet(expected.normalize_snippet(), actual.normalize_snippet())

        # Combined score (weighted average: 60% name, 40% snippet)
        combined_score = 0.6 * name_score + 0.4 * snippet_score

        # Match if snippet meets threshold
        is_match = snippet_score >= self.snippet_threshold

        return is_match, combined_score


class ValidationResult(BaseModel):
    """Result of validating findings for a single test case."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    matched_findings: list[tuple[ExpectedFinding, ActualFinding]] = Field(default_factory=list)
    unmatched_expected: list[ExpectedFinding] = Field(default_factory=list)
    unmatched_actual: list[ActualFinding] = Field(default_factory=list)


class FindingValidator:
    """Validates linter findings against ground truth."""

    def __init__(self, location_matcher: Optional[LocationMatcher] = None):
        """Initialize validator with optional custom location matcher."""
        self.location_matcher = location_matcher or LocationMatcher()

    def validate_positive_case(
        self, expected: list[ExpectedFinding], actual: list[ActualFinding]
    ) -> ValidationResult:
        """
        Validate a positive test case (buggy code that should be detected).

        Three-way detection logic:
        - "yes" = True Positive (definite detection)
        - "context-dependent" = True Positive (flagged potential issue)
        - "no" = False Negative (missed the issue)

        Args:
            expected: Expected findings from ground truth
            actual: Actual findings from linter (should have detected field)

        Returns:
            ValidationResult with TP/FP/FN counts
        """
        result = ValidationResult()

        # For positive cases, we expect at least one detection
        if expected and actual:
            # Check if linter detected the issue
            # Note: actual findings come from LintResult which converts detected field
            # For now, if we have any actual findings, it means detected != "no"
            result.true_positives = 1
            # Any extra actual findings beyond the first are false positives
            if len(actual) > 1:
                result.false_positives = len(actual) - 1
        elif expected and not actual:
            # We expected to detect but didn't (detected="no") - count as FN
            result.false_negatives = len(expected)
        elif not expected and actual:
            # Shouldn't happen (expected should always have items for positive cases)
            result.false_positives = len(actual)

        return result

    def validate_negative_case(self, actual: list[ActualFinding]) -> ValidationResult:
        """
        Validate a negative test case (correct code that should NOT be flagged).

        Args:
            actual: Actual findings from linter

        Returns:
            ValidationResult (all actual findings are FPs)
        """
        result = ValidationResult()
        result.false_positives = len(actual)
        result.unmatched_actual = actual
        return result

    def validate_context_dependent_case(
        self,
        actual: list[ActualFinding],
        allow_detection: bool = True,
        allow_skip: bool = True,
    ) -> ValidationResult:
        """
        Validate a context-dependent test case (edge case where either outcome is OK).

        These are test cases where detection depends on context, coding style,
        or interpretation. Both detecting and not detecting can be acceptable.

        Args:
            actual: Actual findings from linter
            allow_detection: Whether detection is acceptable
            allow_skip: Whether not detecting is acceptable

        Returns:
            ValidationResult (informational only, doesn't affect metrics)
        """
        result = ValidationResult()

        # Ambiguous cases are always "valid" - we just record what happened
        # These are for analysis purposes only
        if actual:
            # Something was detected
            if allow_detection:
                result.true_positives = len(actual)
            else:
                result.false_positives = len(actual)
        else:
            # Nothing detected
            if not allow_skip:
                result.false_negatives = 1

        return result

    # Backward compatibility alias
    def validate_ambiguous_case(
        self,
        actual: list[ActualFinding],
        allow_detection: bool = True,
        allow_skip: bool = True,
    ) -> ValidationResult:
        """
        Deprecated: Use validate_context_dependent_case() instead.

        This is a backward compatibility alias.
        """
        return self.validate_context_dependent_case(actual, allow_detection, allow_skip)
