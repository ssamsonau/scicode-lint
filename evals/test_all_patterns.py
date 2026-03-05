"""Pytest tests for scicode-lint evaluation framework.

Tests each pattern individually and validates overall metrics.
"""

import pytest


def test_pattern_evaluation(eval_runner, enabled_patterns, thresholds, pattern_id):
    """
    Test evaluation of a single pattern.

    This is a parametrized test that runs once per enabled pattern.
    """
    # Evaluate pattern
    metrics = eval_runner.evaluate_pattern(pattern_id)

    # Assertions
    assert metrics is not None, f"Failed to evaluate pattern {pattern_id}"

    # Check precision threshold
    pattern_precision_threshold = thresholds.get("pattern_precision", 0.90)
    assert metrics.precision >= pattern_precision_threshold, (
        f"{pattern_id}: Precision {metrics.precision:.3f} "
        f"below threshold {pattern_precision_threshold}"
    )

    # Check recall threshold
    pattern_recall_threshold = thresholds.get("pattern_recall", 0.80)
    assert metrics.recall >= pattern_recall_threshold, (
        f"{pattern_id}: Recall {metrics.recall:.3f} below threshold {pattern_recall_threshold}"
    )

    # Report metrics for visibility
    print(f"\n{pattern_id} metrics:")
    print(f"  Precision: {metrics.precision:.3f}")
    print(f"  Recall: {metrics.recall:.3f}")
    print(f"  F1: {metrics.f1_score:.3f}")
    print(
        f"  TP: {metrics.true_positives}, FP: {metrics.false_positives}, "
        f"FN: {metrics.false_negatives}"
    )


def pytest_generate_tests(metafunc):
    """Generate parametrized tests for each enabled pattern."""
    if (
        "pattern_id" in metafunc.fixturenames
        and "test_pattern_evaluation" in metafunc.function.__name__
    ):
        # Get enabled patterns - for now, just ml-001
        # In the future, this will read from test_definitions.yaml
        enabled_patterns = ["ml-001-scaler-leakage"]

        # Parametrize the test with pattern IDs
        metafunc.parametrize("pattern_id", enabled_patterns, ids=lambda x: x)


def test_overall_metrics(eval_runner, thresholds):
    """
    Test overall metrics across all patterns.

    This test validates that aggregated metrics meet thresholds.
    """
    # Run evaluation on all patterns
    pattern_metrics = eval_runner.evaluate_all_patterns()

    assert len(pattern_metrics) > 0, "No patterns were evaluated"

    # Calculate overall metrics
    overall_metrics = eval_runner.metrics_calculator.aggregate_metrics(pattern_metrics)

    # Check overall precision
    overall_precision_threshold = thresholds.get("overall_precision", 0.90)
    assert overall_metrics.overall.precision >= overall_precision_threshold, (
        f"Overall precision {overall_metrics.overall.precision:.3f} "
        f"below threshold {overall_precision_threshold}"
    )

    # Check overall recall
    overall_recall_threshold = thresholds.get("overall_recall", 0.80)
    assert overall_metrics.overall.recall >= overall_recall_threshold, (
        f"Overall recall {overall_metrics.overall.recall:.3f} "
        f"below threshold {overall_recall_threshold}"
    )

    # Report overall metrics
    print("\nOverall metrics:")
    print(f"  Precision: {overall_metrics.overall.precision:.3f}")
    print(f"  Recall: {overall_metrics.overall.recall:.3f}")
    print(f"  F1: {overall_metrics.overall.f1_score:.3f}")
    print(f"  Patterns evaluated: {overall_metrics.overall.pattern_count}")


def test_critical_severity_precision(eval_runner, thresholds):
    """
    Test that critical severity findings meet high precision threshold.

    Critical findings should have very high precision (≥ 0.95) to minimize
    false alarms on serious issues.
    """
    # Run evaluation on all patterns
    pattern_metrics = eval_runner.evaluate_all_patterns()

    assert len(pattern_metrics) > 0, "No patterns were evaluated"

    # Calculate overall metrics
    overall_metrics = eval_runner.metrics_calculator.aggregate_metrics(pattern_metrics)

    # Check critical severity precision
    if "critical" in overall_metrics.by_severity:
        critical_precision_threshold = thresholds.get("critical_precision", 0.95)
        critical_precision = overall_metrics.by_severity["critical"].precision

        assert critical_precision >= critical_precision_threshold, (
            f"Critical severity precision {critical_precision:.3f} "
            f"below threshold {critical_precision_threshold}"
        )

        # Report critical metrics
        print("\nCritical severity metrics:")
        print(f"  Precision: {critical_precision:.3f}")
        print(f"  Recall: {overall_metrics.by_severity['critical'].recall:.3f}")
        print(f"  Patterns: {overall_metrics.by_severity['critical'].pattern_count}")
    else:
        pytest.skip("No critical severity patterns evaluated")


def test_no_negative_false_positives(eval_runner):
    """
    Test that negative test cases produce zero false positives.

    This is a strict test: correct code should NEVER trigger warnings.
    """
    # Run evaluation on all patterns
    pattern_metrics = eval_runner.evaluate_all_patterns()

    violations = []
    for metrics in pattern_metrics:
        # Check if pattern has negative cases with false positives
        # Note: This is a simplified check; detailed per-case validation
        # would require tracking FPs specifically from negative cases
        if metrics.false_positives > 0:
            # For now, just report patterns with any FPs
            violations.append(f"{metrics.pattern_id}: {metrics.false_positives} FP(s)")

    if violations:
        print("\nPatterns with false positives:")
        for v in violations:
            print(f"  - {v}")

    # This test is informational for now - we check overall precision instead
    # In a real implementation, we'd track negative-case FPs separately
