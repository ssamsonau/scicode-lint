"""LLM-as-judge evaluation for scicode-lint patterns.

Evaluates pattern detection quality using LLM to judge semantic correctness
of linter outputs against test case expectations.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from scicode_lint.llm.client import LLMClient

# Handle both module and script execution
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from evals.judge_models import (
        JudgeVerdict,
        OverallJudgeMetrics,
        PatternJudgeMetrics,
        TestCaseEvaluation,
    )
    from evals.prompts.judge_system_prompt import (
        JUDGE_SYSTEM_PROMPT,
        generate_judge_prompt,
    )
else:
    from .judge_models import (
        JudgeVerdict,
        OverallJudgeMetrics,
        PatternJudgeMetrics,
        TestCaseEvaluation,
    )
    from .prompts.judge_system_prompt import (
        JUDGE_SYSTEM_PROMPT,
        generate_judge_prompt,
    )

# Import scicode_lint components
try:
    from scicode_lint import SciCodeLinter
    from scicode_lint.config import (
        LinterConfig,
        LLMConfig,
        get_default_patterns_dir,
        load_config_from_toml,
        load_llm_config,
    )
    from scicode_lint.llm.client import create_client
    from scicode_lint.vllm import VLLMMetricsMonitor
except ImportError:
    logger.error("Failed to import scicode_lint. Is it installed?")
    sys.exit(1)


class LLMJudgeEvaluator:
    """Evaluate linter outputs using LLM as judge."""

    def __init__(
        self,
        llm_client: LLMClient,
        patterns_dir: Path,
        max_concurrent: int = 150,
        skip_judge: bool = False,
    ):
        """
        Initialize evaluator.

        Args:
            llm_client: LLM client for judge evaluations
            patterns_dir: Directory containing patterns
            max_concurrent: Max concurrent test evaluations (from config.toml).
                           Higher values improve GPU utilization but use more memory.
            skip_judge: If True, skip LLM judge evaluation (only compute direct metrics)
        """
        self.llm = llm_client
        self.patterns_dir = Path(patterns_dir)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.skip_judge = skip_judge

    async def run_linter(self, file_path: Path, linter: SciCodeLinter) -> dict[str, Any]:
        """
        Run linter on file using async API directly.

        Args:
            file_path: Path to test file
            linter: SciCodeLinter instance (pre-configured with pattern filter)

        Returns:
            Dictionary with detection results
        """
        try:
            # Use linter's async API directly (no subprocess overhead)
            result = await linter._check_file_async(file_path)

            # Process findings
            if result.findings:
                finding = result.findings[0]
                location = finding.location
                return {
                    "detected": finding.detection_type,  # "yes"/"context-dependent"
                    "lines": location.lines if location else [],
                    "snippet": location.snippet if location else "",
                    "reasoning": finding.reasoning or "",
                    "issue": finding.issue or "",
                    "confidence": finding.confidence,
                    "explanation": finding.explanation or "",
                    "thinking": finding.thinking,
                }

            # No findings - check checked_patterns for reasoning
            if result.checked_patterns:
                check_result = result.checked_patterns[0]
                return {
                    "detected": check_result.detected,
                    "lines": [],
                    "snippet": "",
                    "reasoning": check_result.reasoning or "",
                    "issue": None,
                    "confidence": check_result.confidence,
                    "explanation": None,
                    "thinking": check_result.thinking,
                }

            return {
                "detected": "no",
                "lines": [],
                "snippet": "",
                "reasoning": "",
                "issue": None,
                "confidence": 0.0,
                "explanation": None,
                "thinking": None,
            }

        except Exception as e:
            logger.error(f"Error running linter on {file_path}: {e}")
            return {
                "detected": "no",
                "lines": [],
                "snippet": "",
                "reasoning": "",
                "issue": None,
                "confidence": 0.0,
                "explanation": f"Error: {e}",
                "thinking": None,
            }

    def _evaluate_direct(
        self,
        test_type: Literal["positive", "negative", "context_dependent"],
        linter_output: dict[str, Any],
        min_confidence: float = 0.80,
    ) -> tuple[bool, str]:
        """
        Evaluate using direct metrics logic.

        Returns:
            (passed, reason) tuple
        """
        detected = linter_output.get("detected", "no")
        confidence = linter_output.get("confidence", 0.0)

        if test_type == "positive":
            # Should detect the bug
            if detected == "no":
                return False, "Bug not detected (expected detection)"
            if confidence < min_confidence:
                return False, f"Confidence {confidence:.2f} below threshold {min_confidence}"
            return True, "Bug detected with sufficient confidence"

        elif test_type == "negative":
            # Should NOT detect a bug
            if detected in ("yes", "context-dependent"):
                return False, "False positive - bug detected in clean code"
            return True, "Correctly identified as clean code"

        else:  # context_dependent
            # Either detection or non-detection is acceptable
            return True, f"Context-dependent: detected={detected}"

    def _determine_alignment(
        self,
        direct_passed: bool,
        judge_verdict: Literal["yes", "no", "partial"],
    ) -> Literal["both_pass", "both_fail", "quality_issue", "overly_strict"]:
        """Determine alignment category between direct metrics and judge."""
        judge_passed = judge_verdict in ("yes", "partial")

        if direct_passed and judge_passed:
            return "both_pass"
        elif not direct_passed and not judge_passed:
            return "both_fail"
        elif direct_passed and not judge_passed:
            return "quality_issue"
        else:  # not direct_passed and judge_passed
            return "overly_strict"

    async def evaluate_test_file(
        self,
        test_file: Path,
        test_type: Literal["positive", "negative", "context_dependent"],
        pattern_id: str,
        expected_behavior: str,
        linter: SciCodeLinter,
        min_confidence: float = 0.80,
    ) -> TestCaseEvaluation:
        """
        Evaluate a single test file using LLM judge.

        Args:
            test_file: Path to test file
            test_type: "positive", "negative", or "context_dependent"
            pattern_id: Pattern identifier
            expected_behavior: Expected behavior from pattern.toml
            linter: SciCodeLinter instance (pre-configured with pattern filter)

        Returns:
            TestCaseEvaluation with judge verdict
        """
        # Semaphore limits concurrent evaluations to avoid overwhelming vLLM/system
        async with self._semaphore:
            # Run linter (linter is pre-configured with pattern filter)
            linter_output = await self.run_linter(test_file, linter)

            # Evaluate with direct metrics
            direct_passed, direct_reason = self._evaluate_direct(
                test_type, linter_output, min_confidence
            )

            # Get judge verdict (skip if --skip-judge flag is set)
            if self.skip_judge:
                # Use direct metrics result as proxy for judge verdict
                verdict = JudgeVerdict(
                    verdict="yes" if direct_passed else "no",
                    reasoning="Judge skipped (--skip-judge flag)",
                    confidence=1.0 if direct_passed else 0.0,
                )
            else:
                # Generate judge prompt
                judge_prompt = generate_judge_prompt(
                    test_file_path=str(test_file.relative_to(self.patterns_dir.parent)),
                    test_type=test_type,
                    expected_behavior=expected_behavior,
                    linter_output=linter_output,
                )

                # Get judge verdict using async for better concurrency
                try:
                    verdict = await self.llm.async_complete_structured(
                        system_prompt=JUDGE_SYSTEM_PROMPT,
                        user_prompt=judge_prompt,
                        schema=JudgeVerdict,
                    )
                except Exception as e:
                    logger.error(f"Judge evaluation failed for {test_file}: {e}")
                    # Default to "no" verdict on error
                    verdict = JudgeVerdict(
                        verdict="no",
                        reasoning=f"Evaluation failed: {e}",
                        confidence=0.0,
                    )

            # Determine alignment
            alignment = self._determine_alignment(direct_passed, verdict.verdict)

            # Build evaluation result
            return TestCaseEvaluation(
                test_file=str(test_file.relative_to(self.patterns_dir.parent)),
                test_type=test_type,
                expected_behavior=expected_behavior,
                linter_detected=linter_output["detected"],
                linter_lines=linter_output.get("lines", []),
                linter_snippet=linter_output.get("snippet", ""),
                linter_reasoning=linter_output.get("reasoning", ""),
                linter_issue=linter_output.get("issue", ""),
                linter_confidence=linter_output["confidence"],
                linter_thinking=linter_output.get("thinking"),
                judge_verdict=verdict.verdict,
                judge_reasoning=verdict.reasoning,
                judge_confidence=verdict.confidence,
                judge_thinking=verdict.thinking,
                direct_passed=direct_passed,
                direct_reason=direct_reason,
                alignment=alignment,
            )

    async def evaluate_pattern(self, pattern_id: str) -> PatternJudgeMetrics | None:
        """
        Evaluate all test files for a pattern.

        Args:
            pattern_id: Pattern identifier (e.g., "ml-001-scaler-leakage")

        Returns:
            PatternJudgeMetrics or None if pattern not found
        """
        # Find pattern directory
        pattern_dir = None
        for category_dir in self.patterns_dir.iterdir():
            if category_dir.is_dir():
                candidate = category_dir / pattern_id
                if candidate.exists():
                    pattern_dir = candidate
                    break

        if not pattern_dir:
            logger.error(f"Pattern not found: {pattern_id}")
            return None

        # Load and validate pattern using Pydantic models
        from scicode_lint.detectors.pattern_loader import PatternLoader

        loader = PatternLoader(self.patterns_dir)
        try:
            pattern_toml_obj = loader.load_pattern_toml(pattern_dir)
        except Exception as e:
            logger.error(f"Failed to load pattern {pattern_id}: {e}")
            return None

        actual_pattern_id = pattern_toml_obj.meta.id
        logger.debug(f"Pattern directory: {pattern_id}, actual ID: {actual_pattern_id}")

        # Build mapping of test file paths to (expected_behavior, min_confidence)
        # Uses TOML descriptions to avoid "data leakage"
        test_expectations: dict[str, tuple[str, float]] = {}

        for pos_test in pattern_toml_obj.tests.positive:
            description = pos_test.description
            expected_issue = pos_test.expected_issue
            min_confidence = pos_test.min_confidence
            # Combine description and expected_issue for positive tests
            expected = f"{description}\n\nExpected issue: {expected_issue}"
            test_expectations[pos_test.file] = (expected, min_confidence)

        for neg_test in pattern_toml_obj.tests.negative:
            test_expectations[neg_test.file] = (neg_test.description, 0.80)

        for ctx_test in pattern_toml_obj.tests.context_dependent:
            test_expectations[ctx_test.file] = (ctx_test.description, 0.80)

        # Collect all test files with (path, type, expected_behavior, min_confidence)
        TestType = Literal["positive", "negative", "context_dependent"]
        test_files: list[tuple[Path, TestType, str, float]] = []

        # Positive tests
        positive_dir = pattern_dir / "test_positive"
        if positive_dir.exists():
            for py_file in positive_dir.glob("**/*.py"):
                rel_path = str(py_file.relative_to(pattern_dir))
                expected, min_conf = test_expectations.get(rel_path, ("", 0.80))
                if not expected:
                    logger.warning(f"No description in pattern.toml for {py_file}, using default")
                    expected = f"Positive test case for {pattern_id}"
                test_files.append((py_file, "positive", expected, min_conf))

        # Negative tests
        negative_dir = pattern_dir / "test_negative"
        if negative_dir.exists():
            for py_file in negative_dir.glob("**/*.py"):
                rel_path = str(py_file.relative_to(pattern_dir))
                expected, min_conf = test_expectations.get(rel_path, ("", 0.80))
                if not expected:
                    logger.warning(f"No description in pattern.toml for {py_file}, using default")
                    expected = f"Negative test case for {pattern_id}"
                test_files.append((py_file, "negative", expected, min_conf))

        # Context-dependent tests
        context_dir = pattern_dir / "test_context_dependent"
        if context_dir.exists():
            for py_file in context_dir.glob("**/*.py"):
                rel_path = str(py_file.relative_to(pattern_dir))
                expected, min_conf = test_expectations.get(rel_path, ("", 0.80))
                if not expected:
                    logger.warning(f"No description in pattern.toml for {py_file}, using default")
                    expected = f"Context-dependent test case for {pattern_id}"
                test_files.append((py_file, "context_dependent", expected, min_conf))

        if not test_files:
            logger.warning(f"No test files found for {pattern_id}")
            return None

        logger.info(f"Evaluating {len(test_files)} test files for {pattern_id}")

        # Create linter with pattern filter (only checks this one pattern)
        linter_config = LinterConfig(
            patterns_dir=self.patterns_dir,
            llm_config=load_llm_config(),
            enabled_patterns={actual_pattern_id},
        )
        linter = SciCodeLinter(config=linter_config)

        # Evaluate all test files in parallel - vLLM handles concurrent requests efficiently
        # with continuous batching, PagedAttention, and KV cache sharing
        evaluation_tasks = [
            self.evaluate_test_file(
                test_file, test_type, actual_pattern_id, expected_behavior, linter, min_conf
            )
            for test_file, test_type, expected_behavior, min_conf in test_files
        ]
        evaluations = await asyncio.gather(*evaluation_tasks)

        # Calculate metrics
        return self._calculate_pattern_metrics(pattern_id, evaluations)

    def _calculate_pattern_metrics(
        self, pattern_id: str, evaluations: list[TestCaseEvaluation]
    ) -> PatternJudgeMetrics:
        """Calculate metrics from evaluations."""
        # Count by verdict
        correct = sum(1 for e in evaluations if e.judge_verdict == "yes")
        partial = sum(1 for e in evaluations if e.judge_verdict == "partial")
        incorrect = sum(1 for e in evaluations if e.judge_verdict == "no")

        # Calculate accuracy by test type
        positive_evals = [e for e in evaluations if e.test_type == "positive"]
        negative_evals = [e for e in evaluations if e.test_type == "negative"]
        context_evals = [e for e in evaluations if e.test_type == "context_dependent"]

        positive_accuracy = (
            sum(1 for e in positive_evals if e.judge_verdict == "yes") / len(positive_evals)
            if positive_evals
            else 0.0
        )

        negative_accuracy = (
            sum(1 for e in negative_evals if e.judge_verdict == "yes") / len(negative_evals)
            if negative_evals
            else 0.0
        )

        # For context tests, both "yes" and "partial" are acceptable
        context_accuracy = (
            sum(1 for e in context_evals if e.judge_verdict in ["yes", "partial"])
            / len(context_evals)
            if context_evals
            else 1.0  # Default to 1.0 if no context tests
        )

        # Overall accuracy: correct + 0.5 * partial
        overall_accuracy = (correct + 0.5 * partial) / len(evaluations)

        # Calculate alignment metrics (direct vs judge)
        aligned_count = sum(1 for e in evaluations if e.aligned)
        both_pass_count = sum(1 for e in evaluations if e.alignment == "both_pass")
        both_fail_count = sum(1 for e in evaluations if e.alignment == "both_fail")
        quality_issue_count = sum(1 for e in evaluations if e.alignment == "quality_issue")
        overly_strict_count = sum(1 for e in evaluations if e.alignment == "overly_strict")

        return PatternJudgeMetrics(
            pattern_id=pattern_id,
            total_tests=len(evaluations),
            positive_accuracy=positive_accuracy,
            negative_accuracy=negative_accuracy,
            context_accuracy=context_accuracy,
            overall_accuracy=overall_accuracy,
            correct_count=correct,
            partial_count=partial,
            incorrect_count=incorrect,
            aligned_count=aligned_count,
            both_pass_count=both_pass_count,
            both_fail_count=both_fail_count,
            quality_issue_count=quality_issue_count,
            overly_strict_count=overly_strict_count,
            evaluations=evaluations,
        )


def _compute_variance_report(
    all_run_results: list[list[PatternJudgeMetrics]], pattern_ids: list[str]
) -> dict[str, Any]:
    """
    Compute variance across multiple runs to identify unstable patterns.

    Returns a report with:
    - Per-pattern variance in accuracy
    - Unstable patterns (where results differ across runs)
    - Overall variance statistics
    """
    import statistics

    # Build pattern_id -> list of accuracies across runs
    pattern_accuracies: dict[str, list[float]] = {pid: [] for pid in pattern_ids}
    overall_accuracies: list[float] = []

    for run_results in all_run_results:
        run_total_correct = 0
        run_total_tests = 0
        for pm in run_results:
            pattern_accuracies[pm.pattern_id].append(pm.overall_accuracy)
            run_total_correct += pm.correct_count
            run_total_tests += pm.total_tests

        if run_total_tests > 0:
            overall_accuracies.append(run_total_correct / run_total_tests)

    # Compute per-pattern stats
    pattern_stats: list[dict[str, Any]] = []
    unstable_patterns: list[str] = []

    for pid, accuracies in pattern_accuracies.items():
        if len(accuracies) < 2:
            continue

        mean_acc = statistics.mean(accuracies)
        stdev_acc = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
        min_acc = min(accuracies)
        max_acc = max(accuracies)
        variance = max_acc - min_acc

        stat = {
            "pattern_id": pid,
            "mean_accuracy": mean_acc,
            "stdev": stdev_acc,
            "min": min_acc,
            "max": max_acc,
            "variance": variance,
            "accuracies": accuracies,
        }
        pattern_stats.append(stat)

        # Flag as unstable if variance > 10% or stdev > 5%
        if variance > 0.10 or stdev_acc > 0.05:
            unstable_patterns.append(pid)

    # Sort by variance (most unstable first)
    pattern_stats.sort(key=lambda x: x["variance"], reverse=True)

    # Overall stats
    overall_mean = statistics.mean(overall_accuracies) if overall_accuracies else 0.0
    overall_stdev = statistics.stdev(overall_accuracies) if len(overall_accuracies) > 1 else 0.0

    return {
        "num_runs": len(all_run_results),
        "overall_mean_accuracy": overall_mean,
        "overall_stdev": overall_stdev,
        "overall_accuracies": overall_accuracies,
        "pattern_stats": pattern_stats,
        "unstable_patterns": unstable_patterns,
    }


def _print_variance_report(report: dict[str, Any], output_dir: Path) -> None:
    """Print and save variance report from multi-run evaluation."""
    lines = [
        "",
        "=" * 70,
        f"VARIANCE REPORT ({report['num_runs']} runs)",
        "=" * 70,
        "",
        f"Overall Mean Accuracy: {report['overall_mean_accuracy']:.2%}",
        f"Overall Std Dev:       {report['overall_stdev']:.2%}",
        f"Per-run accuracies:    {[f'{a:.2%}' for a in report['overall_accuracies']]}",
        "",
    ]

    if report["unstable_patterns"]:
        lines.extend(
            [
                "UNSTABLE PATTERNS (variance > 10% or stdev > 5%)",
                "-" * 70,
            ]
        )
        for pid in report["unstable_patterns"]:
            stat = next(s for s in report["pattern_stats"] if s["pattern_id"] == pid)
            accs = [f"{a:.0%}" for a in stat["accuracies"]]
            lines.append(
                f"  {pid}: {stat['mean_accuracy']:.1%} mean, "
                f"{stat['variance']:.0%} range [{stat['min']:.0%}-{stat['max']:.0%}], "
                f"runs: {accs}"
            )
        lines.extend(
            [
                "",
                ">> These patterns have ambiguous detection questions.",
                ">> Run with --verbose to see LLM reasoning differences.",
                "",
            ]
        )
    else:
        lines.append("All patterns are stable across runs (variance < 10%).")
        lines.append("")

    lines.append("=" * 70)

    # Print to console
    print("\n".join(lines))

    # Save variance report
    variance_path = output_dir / "variance_report.json"
    variance_path.parent.mkdir(parents=True, exist_ok=True)
    with open(variance_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Variance report saved to {variance_path}")


class JudgeReportGenerator:
    """Generate reports from LLM-as-judge evaluations."""

    @staticmethod
    def generate_summary_text(metrics: OverallJudgeMetrics) -> str:
        """Generate human-readable summary."""
        lines = [
            "",
            "=" * 70,
            "LLM-AS-JUDGE EVALUATION SUMMARY",
            "=" * 70,
            "",
            f"Total Patterns: {metrics.total_patterns}",
            f"Total Tests: {metrics.total_tests}",
            "",
            "ACCURACY BY TEST TYPE",
            "-" * 70,
            f"Positive Tests:  {metrics.positive_accuracy:.2%}",
            f"Negative Tests:  {metrics.negative_accuracy:.2%}",
            f"Context Tests:   {metrics.context_accuracy:.2%}",
            "",
            f"Overall Accuracy: {metrics.overall_accuracy:.2%}",
            f"Avg Judge Confidence: {metrics.avg_judge_confidence:.2f}",
            "",
            f"Patterns Above Threshold (≥85%): "
            f"{metrics.patterns_above_threshold}/{metrics.total_patterns}",
            "",
            "ALIGNMENT METRICS (Direct vs Judge)",
            "-" * 70,
            f"Semantic Alignment:           {metrics.semantic_alignment:.1%}",
        ]
        # Handle conditional formatting for counts
        if metrics.total_tests > 0:
            both_pass_pct = metrics.both_pass_count / metrics.total_tests
            both_fail_pct = metrics.both_fail_count / metrics.total_tests
            cnt = metrics.both_pass_count
            lines.append(f"  - Both Pass:                {cnt} ({both_pass_pct:.1%})")
            cnt = metrics.both_fail_count
            lines.append(f"  - Both Fail:                {cnt} ({both_fail_pct:.1%})")
        else:
            lines.append("  - Both Pass:                0 (0.0%)")
            lines.append("  - Both Fail:                0 (0.0%)")

        lines.extend(
            [
                "",
                f"Quality Issue Rate:           {metrics.quality_issue_rate:.1%} "
                f"({metrics.quality_issue_count} cases)",
                "  (Direct passes, Judge fails - right location, wrong explanation)",
                "",
                f"Ground Truth Strictness Rate: {metrics.ground_truth_strictness_rate:.1%} "
                f"({metrics.overly_strict_count} cases)",
                "  (Direct fails, Judge passes - ground truth too rigid)",
                "",
                "=" * 70,
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def save_json_report(metrics: OverallJudgeMetrics, output_path: Path) -> None:
        """Save metrics as JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics.model_dump(), f, indent=2)
        logger.info(f"JSON report saved to {output_path}")

    @staticmethod
    def save_markdown_report(metrics: OverallJudgeMetrics, output_path: Path) -> None:
        """Save metrics as Markdown."""
        lines = [
            "# LLM-as-Judge Evaluation Report",
            "",
            f"**Total Patterns:** {metrics.total_patterns}  ",
            f"**Total Tests:** {metrics.total_tests}  ",
            f"**Overall Accuracy:** {metrics.overall_accuracy:.2%}  ",
            "",
            "## Accuracy by Test Type",
            "",
            "| Test Type | Accuracy |",
            "|-----------|----------|",
            f"| Positive | {metrics.positive_accuracy:.2%} |",
            f"| Negative | {metrics.negative_accuracy:.2%} |",
            f"| Context-Dependent | {metrics.context_accuracy:.2%} |",
            "",
            "## Alignment Metrics (Direct vs Judge)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Semantic Alignment | {metrics.semantic_alignment:.1%} |",
            f"| Quality Issue Rate | {metrics.quality_issue_rate:.1%} |",
            f"| Ground Truth Strictness Rate | {metrics.ground_truth_strictness_rate:.1%} |",
            "",
            "**Interpretation:**",
            "- **Quality Issues**: Direct metrics pass but judge fails "
            "(right location, wrong explanation)",
            "- **Overly Strict**: Direct metrics fail but judge passes (ground truth too rigid)",
            "",
            "## Per-Pattern Results",
            "",
            "| Pattern ID | Tests | Accuracy | Aligned | Quality Issues | Overly Strict |",
            "|------------|-------|----------|---------|----------------|---------------|",
        ]

        for pattern in sorted(metrics.patterns, key=lambda p: p.overall_accuracy, reverse=True):
            lines.append(
                f"| {pattern.pattern_id} | {pattern.total_tests} | "
                f"{pattern.overall_accuracy:.2%} | {pattern.alignment_rate:.0%} | "
                f"{pattern.quality_issue_count} | {pattern.overly_strict_count} |"
            )

        # Add divergent cases section if any exist
        divergent_patterns = [
            p for p in metrics.patterns if p.quality_issue_count > 0 or p.overly_strict_count > 0
        ]
        if divergent_patterns:
            lines.extend(
                [
                    "",
                    "## Divergent Cases (Need Attention)",
                    "",
                ]
            )
            for pattern in divergent_patterns:
                divergent_cases = [e for e in pattern.evaluations if not e.aligned]
                if divergent_cases:
                    lines.append(f"### {pattern.pattern_id}")
                    lines.append("")
                    for case in divergent_cases:
                        emoji = "!!" if case.alignment == "quality_issue" else ">>"
                        lines.append(f"- {emoji} **[{case.alignment}]** `{case.test_file}`")
                        direct_status = "PASS" if case.direct_passed else "FAIL"
                        lines.append(f"  - Direct: {direct_status} - {case.direct_reason}")
                        reason_truncated = case.judge_reasoning[:80]
                        lines.append(f"  - Judge: {case.judge_verdict} - {reason_truncated}...")
                    lines.append("")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        logger.info(f"Markdown report saved to {output_path}")


async def main() -> int:
    """Main evaluation runner."""
    # Load config for defaults
    config = load_config_from_toml()
    default_max_concurrent = config.get("performance", {}).get("max_concurrent_evals", 150)

    parser = argparse.ArgumentParser(
        description="Run LLM-as-judge evaluation for scicode-lint patterns"
    )
    parser.add_argument(
        "--pattern",
        "-p",
        action="append",
        dest="patterns",
        help="Evaluate specific pattern(s) only (can be used multiple times)",
    )
    parser.add_argument(
        "--category",
        "-c",
        action="append",
        dest="categories",
        help="Evaluate specific category/categories only (can be used multiple times). "
        "Available: ai-inference, ai-training, scientific-numerical, "
        "scientific-performance, scientific-reproducibility",
    )
    parser.add_argument(
        "--patterns-dir",
        type=Path,
        default=get_default_patterns_dir(),
        help="Directory containing pattern test cases",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for reports (default: evals/reports/judge)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "all"],
        default="all",
        help="Report format",
    )
    parser.add_argument(
        "--llm-base-url",
        default="http://localhost:5001",
        help="LLM API base URL (without /v1 suffix)",
    )
    parser.add_argument(
        "--no-auto-server",
        action="store_true",
        help="Don't auto-start vLLM server (assume already running)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=default_max_concurrent,
        help=f"Max concurrent test evaluations (default: {default_max_concurrent} from config). "
        "Higher = better GPU utilization but more memory.",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=5.0,
        help="vLLM metrics monitoring interval in seconds (default: 5.0, set to 0 to disable).",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM judge evaluation (only compute direct metrics). "
        "Faster but no semantic evaluation.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of evaluation runs for consistency checking (default: 1). "
        "Use 3 to detect LLM variance and identify ambiguous patterns.",
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")

    # Auto-start vLLM server if needed
    vllm_server = None
    if not args.no_auto_server:
        try:
            from scicode_lint.vllm import VLLMServer, get_server_info

            # Extract port from base URL
            port = int(args.llm_base_url.split(":")[-1])

            # Check if server is already running
            server_info = get_server_info(base_url=args.llm_base_url)
            if server_info.is_running:
                logger.info(f"vLLM server already running at {server_info.base_url}")
            else:
                logger.info("Starting vLLM server...")
                vllm_server = VLLMServer(port=port, wait_timeout=120)
                vllm_server.__enter__()
                logger.info(f"vLLM server ready at {vllm_server.base_url}")
        except ImportError:
            logger.warning("vLLM utilities not available, assuming server is running")
        except Exception as e:
            logger.error(f"Failed to start vLLM server: {e}")
            logger.info("Try running with --no-auto-server if server is already running")
            return 1

    try:
        # Setup LLM client for judge
        # Use LLMConfig defaults (temperature=0.6, top_p=0.95 for Qwen3 thinking mode)
        llm_config = LLMConfig(base_url=args.llm_base_url)
        llm_client = create_client(llm_config)

        # Create evaluator with concurrency limit
        evaluator = LLMJudgeEvaluator(
            llm_client,
            args.patterns_dir,
            max_concurrent=args.max_concurrent,
            skip_judge=args.skip_judge,
        )

        # Determine which patterns to evaluate
        if args.patterns:
            pattern_ids = args.patterns
        else:
            # Find all patterns, optionally filtered by category
            pattern_ids = []
            for category_dir in args.patterns_dir.iterdir():
                if category_dir.is_dir() and not category_dir.name.startswith("."):
                    # Filter by category if specified
                    if args.categories and category_dir.name not in args.categories:
                        continue
                    for pattern_dir in category_dir.iterdir():
                        if pattern_dir.is_dir() and (pattern_dir / "pattern.toml").exists():
                            pattern_ids.append(pattern_dir.name)

        if args.categories:
            logger.info(
                f"Evaluating {len(pattern_ids)} patterns from {len(args.categories)} "
                f"category(s): {', '.join(args.categories)}"
            )
        else:
            logger.info(f"Evaluating {len(pattern_ids)} patterns (all categories)")

        # Multi-run consistency checking
        num_runs = args.runs
        all_run_results: list[list[PatternJudgeMetrics]] = []

        for run_idx in range(num_runs):
            if num_runs > 1:
                logger.info(f"\n=== Run {run_idx + 1}/{num_runs} ===")

            # Start metrics monitor if enabled (only on first run)
            monitor = None
            if args.monitor_interval > 0 and run_idx == 0:
                monitor = VLLMMetricsMonitor(
                    base_url=args.llm_base_url,
                    interval=args.monitor_interval,
                    output_file="evals/reports/vllm_metrics.log",
                    console=True,
                )
                monitor.start()

            eval_start = time.time()
            try:
                # Evaluate ALL patterns in parallel - vLLM handles batching efficiently
                pattern_tasks = [evaluator.evaluate_pattern(pid) for pid in pattern_ids]
                all_results = await asyncio.gather(*pattern_tasks)
                pattern_metrics = [m for m in all_results if m is not None]
                all_run_results.append(pattern_metrics)
            finally:
                if monitor:
                    await monitor.stop()
                eval_elapsed = time.time() - eval_start
                logger.info(f"Run {run_idx + 1} completed in {eval_elapsed:.1f}s")

        if not all_run_results or not all_run_results[0]:
            logger.error("No patterns were evaluated")
            return 1

        # Use first run for primary metrics, but compute variance if multiple runs
        pattern_metrics = all_run_results[0]

        # Calculate overall metrics from first run
        total_tests = sum(p.total_tests for p in pattern_metrics)
        total_correct = sum(p.correct_count for p in pattern_metrics)
        total_partial = sum(p.partial_count for p in pattern_metrics)

        # Aggregate by test type
        all_positive = [
            e for p in pattern_metrics for e in p.evaluations if e.test_type == "positive"
        ]
        all_negative = [
            e for p in pattern_metrics for e in p.evaluations if e.test_type == "negative"
        ]
        all_context = [
            e for p in pattern_metrics for e in p.evaluations if e.test_type == "context_dependent"
        ]

        positive_accuracy = (
            sum(1 for e in all_positive if e.judge_verdict == "yes") / len(all_positive)
            if all_positive
            else 0.0
        )
        negative_accuracy = (
            sum(1 for e in all_negative if e.judge_verdict == "yes") / len(all_negative)
            if all_negative
            else 0.0
        )
        context_accuracy = (
            sum(1 for e in all_context if e.judge_verdict in ["yes", "partial"]) / len(all_context)
            if all_context
            else 1.0
        )

        overall_accuracy = (total_correct + 0.5 * total_partial) / total_tests

        # Average judge confidence
        all_evals = [e for p in pattern_metrics for e in p.evaluations]
        avg_confidence = sum(e.judge_confidence for e in all_evals) / len(all_evals)

        # Patterns above threshold
        patterns_above_threshold = sum(1 for p in pattern_metrics if p.overall_accuracy >= 0.85)

        # Aggregate alignment metrics
        total_aligned = sum(p.aligned_count for p in pattern_metrics)
        total_both_pass = sum(p.both_pass_count for p in pattern_metrics)
        total_both_fail = sum(p.both_fail_count for p in pattern_metrics)
        total_quality_issues = sum(p.quality_issue_count for p in pattern_metrics)
        total_overly_strict = sum(p.overly_strict_count for p in pattern_metrics)

        semantic_alignment = total_aligned / total_tests if total_tests > 0 else 0.0
        quality_issue_rate = total_quality_issues / total_tests if total_tests > 0 else 0.0
        ground_truth_strictness_rate = total_overly_strict / total_tests if total_tests > 0 else 0.0

        overall_metrics = OverallJudgeMetrics(
            total_patterns=len(pattern_metrics),
            total_tests=total_tests,
            positive_accuracy=positive_accuracy,
            negative_accuracy=negative_accuracy,
            context_accuracy=context_accuracy,
            overall_accuracy=overall_accuracy,
            patterns=pattern_metrics,
            avg_judge_confidence=avg_confidence,
            patterns_above_threshold=patterns_above_threshold,
            semantic_alignment=semantic_alignment,
            quality_issue_rate=quality_issue_rate,
            ground_truth_strictness_rate=ground_truth_strictness_rate,
            aligned_count=total_aligned,
            both_pass_count=total_both_pass,
            both_fail_count=total_both_fail,
            quality_issue_count=total_quality_issues,
            overly_strict_count=total_overly_strict,
        )

        # Compute variance across runs if multiple runs
        variance_report: dict[str, Any] | None = None
        if num_runs > 1:
            variance_report = _compute_variance_report(all_run_results, pattern_ids)

        # Determine output directory
        if args.output_dir is None:
            base_output = args.patterns_dir.parent / "evals" / "reports" / "judge"
            # Use category-specific subdirectory if filtering by category
            if args.categories and len(args.categories) == 1:
                output_dir = base_output / args.categories[0]
            else:
                output_dir = base_output
        else:
            output_dir = args.output_dir

        # Generate reports
        reporter = JudgeReportGenerator()

        if args.format in ["json", "all"]:
            reporter.save_json_report(overall_metrics, output_dir / "llm_judge_report.json")

        if args.format in ["markdown", "all"]:
            reporter.save_markdown_report(overall_metrics, output_dir / "llm_judge_report.md")

        # Print summary
        print(reporter.generate_summary_text(overall_metrics))

        # Print variance report if multiple runs
        if variance_report:
            _print_variance_report(variance_report, output_dir)

        return 0

    finally:
        # Cleanup vLLM server if we started it
        if vllm_server is not None:
            try:
                vllm_server.__exit__(None, None, None)
                logger.info("vLLM server stopped")
            except Exception:
                pass


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
