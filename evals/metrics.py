"""Metrics calculation for evaluation framework.

Calculates precision, recall, F1 scores and aggregates by category and severity.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass
class PatternMetrics:
    """Metrics for a single detection pattern."""

    pattern_id: str
    category: str
    severity: Literal["critical", "high", "medium"]

    # Counts
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    # Test case counts
    positive_cases: int = 0
    negative_cases: int = 0
    context_dependent_cases: int = 0

    # Computed metrics
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None

    # Pass/fail status
    passes_thresholds: bool = False

    def calculate_metrics(self, min_precision: float = 1.0, min_recall: float = 1.0) -> None:
        """Calculate precision, recall, F1 and check thresholds."""
        # Precision = TP / (TP + FP)
        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)
        else:
            self.precision = 1.0  # No predictions = perfect precision

        # Recall = TP / (TP + FN)
        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        else:
            self.recall = 1.0  # No ground truth = perfect recall

        # F1 = 2 * (precision * recall) / (precision + recall)
        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1_score = 0.0

        # Check thresholds
        self.passes_thresholds = self.precision >= min_precision and self.recall >= min_recall

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pattern_id": self.pattern_id,
            "category": self.category,
            "severity": self.severity,
            "counts": {
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
            },
            "test_cases": {
                "positive": self.positive_cases,
                "negative": self.negative_cases,
                "ambiguous": self.context_dependent_cases,
            },
            "metrics": {
                "precision": round(self.precision, 3) if self.precision is not None else None,
                "recall": round(self.recall, 3) if self.recall is not None else None,
                "f1_score": round(self.f1_score, 3) if self.f1_score is not None else None,
            },
            "passes_thresholds": self.passes_thresholds,
        }


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple patterns."""

    name: str  # e.g., "overall", "ml-correctness", "critical"
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    pattern_count: int = 0

    def calculate_metrics(self) -> None:
        """Calculate aggregated precision, recall, F1."""
        # Precision = TP / (TP + FP)
        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)
        else:
            self.precision = 1.0

        # Recall = TP / (TP + FN)
        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        else:
            self.recall = 1.0

        # F1 = 2 * (precision * recall) / (precision + recall)
        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1_score = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "pattern_count": self.pattern_count,
            "counts": {
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
            },
            "metrics": {
                "precision": round(self.precision, 3) if self.precision is not None else None,
                "recall": round(self.recall, 3) if self.recall is not None else None,
                "f1_score": round(self.f1_score, 3) if self.f1_score is not None else None,
            },
        }


@dataclass
class OverallMetrics:
    """Complete metrics report across all patterns."""

    overall: AggregatedMetrics = field(default_factory=lambda: AggregatedMetrics("overall"))
    by_category: dict[str, AggregatedMetrics] = field(default_factory=dict)
    by_severity: dict[str, AggregatedMetrics] = field(default_factory=dict)
    patterns: list[PatternMetrics] = field(default_factory=list)

    # Threshold checks
    meets_overall_thresholds: bool = False
    meets_critical_threshold: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall": self.overall.to_dict(),
            "by_category": {cat: metrics.to_dict() for cat, metrics in self.by_category.items()},
            "by_severity": {sev: metrics.to_dict() for sev, metrics in self.by_severity.items()},
            "patterns": [p.to_dict() for p in self.patterns],
            "threshold_checks": {
                "meets_overall_thresholds": self.meets_overall_thresholds,
                "meets_critical_threshold": self.meets_critical_threshold,
            },
        }


class MetricsCalculator:
    """Calculates and aggregates metrics across patterns."""

    def __init__(
        self,
        overall_precision_threshold: float = 1.0,
        overall_recall_threshold: float = 1.0,
        critical_precision_threshold: float = 0.95,
    ):
        """
        Initialize calculator with threshold values.

        Args:
            overall_precision_threshold: Minimum overall precision (default 1.0)
            overall_recall_threshold: Minimum overall recall (default 1.0)
            critical_precision_threshold: Minimum precision for critical severity (default 0.95)
        """
        self.overall_precision_threshold = overall_precision_threshold
        self.overall_recall_threshold = overall_recall_threshold
        self.critical_precision_threshold = critical_precision_threshold

    def calculate_pattern_metrics(
        self,
        pattern_id: str,
        category: str,
        severity: Literal["critical", "high", "medium"],
        true_positives: int,
        false_positives: int,
        false_negatives: int,
        positive_cases: int = 0,
        negative_cases: int = 0,
        context_dependent_cases: int = 0,
    ) -> PatternMetrics:
        """
        Calculate metrics for a single pattern.

        Args:
            pattern_id: Pattern identifier
            category: Pattern category
            severity: Pattern severity level
            true_positives: Number of true positives
            false_positives: Number of false positives
            false_negatives: Number of false negatives
            positive_cases: Number of positive test cases
            negative_cases: Number of negative test cases
            context_dependent_cases: Number of ambiguous test cases

        Returns:
            PatternMetrics with calculated values
        """
        metrics = PatternMetrics(
            pattern_id=pattern_id,
            category=category,
            severity=severity,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            positive_cases=positive_cases,
            negative_cases=negative_cases,
            context_dependent_cases=context_dependent_cases,
        )

        metrics.calculate_metrics(
            min_precision=self.overall_precision_threshold,
            min_recall=self.overall_recall_threshold,
        )

        return metrics

    def aggregate_metrics(self, patterns: list[PatternMetrics]) -> OverallMetrics:
        """
        Aggregate metrics across all patterns.

        Args:
            patterns: List of PatternMetrics to aggregate

        Returns:
            OverallMetrics with aggregated values
        """
        overall = OverallMetrics()
        overall.patterns = patterns

        # Aggregate overall counts
        for pattern in patterns:
            overall.overall.true_positives += pattern.true_positives
            overall.overall.false_positives += pattern.false_positives
            overall.overall.false_negatives += pattern.false_negatives
            overall.overall.pattern_count += 1

        # Aggregate by category
        category_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"tp": 0, "fp": 0, "fn": 0, "count": 0}
        )
        for pattern in patterns:
            category_counts[pattern.category]["tp"] += pattern.true_positives
            category_counts[pattern.category]["fp"] += pattern.false_positives
            category_counts[pattern.category]["fn"] += pattern.false_negatives
            category_counts[pattern.category]["count"] += 1

        for category, counts in category_counts.items():
            metrics = AggregatedMetrics(
                name=category,
                true_positives=counts["tp"],
                false_positives=counts["fp"],
                false_negatives=counts["fn"],
                pattern_count=counts["count"],
            )
            metrics.calculate_metrics()
            overall.by_category[category] = metrics

        # Aggregate by severity
        severity_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"tp": 0, "fp": 0, "fn": 0, "count": 0}
        )
        for pattern in patterns:
            severity_counts[pattern.severity]["tp"] += pattern.true_positives
            severity_counts[pattern.severity]["fp"] += pattern.false_positives
            severity_counts[pattern.severity]["fn"] += pattern.false_negatives
            severity_counts[pattern.severity]["count"] += 1

        for severity, counts in severity_counts.items():
            metrics = AggregatedMetrics(
                name=severity,
                true_positives=counts["tp"],
                false_positives=counts["fp"],
                false_negatives=counts["fn"],
                pattern_count=counts["count"],
            )
            metrics.calculate_metrics()
            overall.by_severity[severity] = metrics

        # Calculate overall metrics
        overall.overall.calculate_metrics()

        # Check thresholds
        overall.meets_overall_thresholds = (
            overall.overall.precision is not None
            and overall.overall.precision >= self.overall_precision_threshold
            and overall.overall.recall is not None
            and overall.overall.recall >= self.overall_recall_threshold
        )

        # Check critical severity threshold
        if "critical" in overall.by_severity:
            critical_precision = overall.by_severity["critical"].precision
            overall.meets_critical_threshold = (
                critical_precision is not None
                and critical_precision >= self.critical_precision_threshold
            )
        else:
            overall.meets_critical_threshold = True  # No critical patterns

        return overall
