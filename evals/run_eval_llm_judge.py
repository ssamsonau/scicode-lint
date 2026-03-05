"""LLM-as-judge evaluation for scicode-lint patterns.

Evaluates pattern detection quality using LLM to judge semantic correctness
of linter outputs against test case expectations.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

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
    from scicode_lint.config import LLMConfig, get_default_patterns_dir
    from scicode_lint.llm.client import create_client
except ImportError:
    logger.error("Failed to import scicode_lint. Is it installed?")
    sys.exit(1)


class DocstringExtractor:
    """Extract expected behavior from test file docstrings."""

    @staticmethod
    def extract_module_docstring(file_path: Path) -> str:
        """
        Extract module-level docstring from Python file.

        Args:
            file_path: Path to Python test file

        Returns:
            Module docstring or empty string if not found
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Simple docstring extraction (module-level)
            lines = content.split("\n")
            in_docstring = False
            docstring_lines = []
            quote_type = None

            for line in lines:
                stripped = line.strip()

                # Start of docstring
                if not in_docstring:
                    if stripped.startswith('"""'):
                        quote_type = '"""'
                        in_docstring = True
                        # Check if docstring starts and ends on same line
                        if stripped.count('"""') >= 2 and len(stripped) > 6:
                            return stripped[3:-3].strip()
                        docstring_lines.append(stripped[3:])
                    elif stripped.startswith("'''"):
                        quote_type = "'''"
                        in_docstring = True
                        if stripped.count("'''") >= 2 and len(stripped) > 6:
                            return stripped[3:-3].strip()
                        docstring_lines.append(stripped[3:])
                # End of docstring
                elif in_docstring and quote_type in stripped:
                    docstring_lines.append(stripped.replace(quote_type, ""))
                    break
                # Inside docstring
                elif in_docstring:
                    docstring_lines.append(line)

            return "\n".join(docstring_lines).strip()

        except Exception as e:
            logger.warning(f"Failed to extract docstring from {file_path}: {e}")
            return ""


class LLMJudgeEvaluator:
    """Evaluate linter outputs using LLM as judge."""

    def __init__(self, llm_client, patterns_dir: Path):
        """
        Initialize evaluator.

        Args:
            llm_client: LLM client for judge evaluations
            patterns_dir: Directory containing patterns
        """
        self.llm = llm_client
        self.patterns_dir = Path(patterns_dir)
        self.docstring_extractor = DocstringExtractor()

    async def run_linter(self, file_path: Path, pattern_id: str | None = None) -> dict[str, Any]:
        """
        Run linter on file and parse output (async to avoid blocking event loop).

        Args:
            file_path: Path to test file
            pattern_id: Optional pattern ID to filter to single pattern

        Returns:
            Dictionary with detection results
        """
        try:
            cmd = ["python", "-m", "scicode_lint", "check", str(file_path), "--format", "json"]
            # Add pattern filter to check only the relevant pattern (speeds up evaluation)
            if pattern_id:
                cmd.extend(["--pattern", pattern_id])

            # Use asyncio.create_subprocess_exec for truly async subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait for process with timeout
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120.0)
                stdout_text = stdout.decode()
                _stderr_text = stderr.decode()
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise RuntimeError(f"Linter timed out after 120s on {file_path}")

            if stdout_text:
                output = json.loads(stdout_text)
                # Linter returns a list of file results
                if isinstance(output, list) and len(output) > 0:
                    file_result = output[0]
                    findings = file_result.get("findings", [])
                    if findings:
                        # Return first finding with new format (lines, snippet, reasoning)
                        finding = findings[0]
                        location = finding.get("location", {})
                        return {
                            "detected": finding.get(
                                "detection_type", "yes"
                            ),  # "yes"/"context-dependent"
                            "lines": location.get("lines", []),
                            "snippet": location.get("snippet", ""),
                            "reasoning": finding.get("reasoning", ""),
                            "issue": finding.get("issue", ""),
                            "confidence": finding.get("confidence", 0.0),
                            "explanation": finding.get("explanation", ""),
                        }

            return {
                "detected": "no",
                "lines": [],
                "snippet": "",
                "reasoning": "",
                "issue": None,
                "confidence": 0.0,
                "explanation": None,
            }

        except FileNotFoundError:
            logger.warning("Linter not found. Skipping linter execution.")
            return {
                "detected": "no",
                "lines": [],
                "snippet": "",
                "reasoning": "",
                "issue": None,
                "confidence": 0.0,
                "explanation": "Linter not available",
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
            }

    async def evaluate_test_file(
        self, test_file: Path, test_type: str, pattern_id: str
    ) -> TestCaseEvaluation:
        """
        Evaluate a single test file using LLM judge.

        Args:
            test_file: Path to test file
            test_type: "positive", "negative", or "context_dependent"
            pattern_id: Pattern identifier

        Returns:
            TestCaseEvaluation with judge verdict
        """
        # Extract expected behavior from docstring
        expected_behavior = self.docstring_extractor.extract_module_docstring(test_file)

        if not expected_behavior:
            logger.warning(f"No docstring found in {test_file}, using default")
            expected_behavior = f"{test_type.title()} test case for {pattern_id}"

        # Run linter (filter to just this pattern for speed)
        linter_output = await self.run_linter(test_file, pattern_id=pattern_id)

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
            judge_verdict=verdict.verdict,
            judge_reasoning=verdict.reasoning,
            judge_confidence=verdict.confidence,
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

        # Read actual pattern ID from pattern.toml (directory name may differ)
        pattern_toml = pattern_dir / "pattern.toml"
        if not pattern_toml.exists():
            raise FileNotFoundError(f"pattern.toml not found in {pattern_dir}")

        import tomli

        with open(pattern_toml, "rb") as f:
            pattern_data = tomli.load(f)
            if "meta" not in pattern_data or "id" not in pattern_data["meta"]:
                raise ValueError(f"Pattern ID not found in {pattern_toml} under [meta] section")
            actual_pattern_id = pattern_data["meta"]["id"]

        logger.debug(f"Pattern directory: {pattern_id}, actual ID: {actual_pattern_id}")

        # Collect all test files
        test_files = []

        # Positive tests
        positive_dir = pattern_dir / "positive"
        if positive_dir.exists():
            for py_file in positive_dir.glob("*.py"):
                test_files.append((py_file, "positive"))

        # Negative tests
        negative_dir = pattern_dir / "negative"
        if negative_dir.exists():
            for py_file in negative_dir.glob("*.py"):
                test_files.append((py_file, "negative"))

        # Context-dependent tests
        context_dir = pattern_dir / "context_dependent"
        if context_dir.exists():
            for py_file in context_dir.glob("*.py"):
                test_files.append((py_file, "context_dependent"))

        if not test_files:
            logger.warning(f"No test files found for {pattern_id}")
            return None

        logger.info(f"Evaluating {len(test_files)} test files for {pattern_id}")

        # Evaluate all test files in parallel - vLLM handles concurrent requests efficiently
        # with continuous batching, PagedAttention, and KV cache sharing
        evaluation_tasks = [
            self.evaluate_test_file(test_file, test_type, actual_pattern_id)
            for test_file, test_type in test_files
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
            evaluations=evaluations,
        )


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
            f"Patterns Above Threshold (≥85%): {metrics.patterns_above_threshold}/{metrics.total_patterns}",
            "",
            "=" * 70,
        ]
        return "\n".join(lines)

    @staticmethod
    def save_json_report(metrics: OverallJudgeMetrics, output_path: Path):
        """Save metrics as JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics.model_dump(), f, indent=2)
        logger.info(f"JSON report saved to {output_path}")

    @staticmethod
    def save_markdown_report(metrics: OverallJudgeMetrics, output_path: Path):
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
            "## Per-Pattern Results",
            "",
            "| Pattern ID | Tests | Accuracy | Correct | Partial | Incorrect |",
            "|------------|-------|----------|---------|---------|-----------|",
        ]

        for pattern in sorted(metrics.patterns, key=lambda p: p.overall_accuracy, reverse=True):
            lines.append(
                f"| {pattern.pattern_id} | {pattern.total_tests} | "
                f"{pattern.overall_accuracy:.2%} | {pattern.correct_count} | "
                f"{pattern.partial_count} | {pattern.incorrect_count} |"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        logger.info(f"Markdown report saved to {output_path}")


async def main():
    """Main evaluation runner."""
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

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")

    # Setup LLM client for judge
    llm_config = LLMConfig(base_url=args.llm_base_url, temperature=0.0)
    llm_client = create_client(llm_config)

    # Create evaluator
    evaluator = LLMJudgeEvaluator(llm_client, args.patterns_dir)

    # Determine which patterns to evaluate
    if args.patterns:
        pattern_ids = args.patterns
    else:
        # Find all patterns
        pattern_ids = []
        for category_dir in args.patterns_dir.iterdir():
            if category_dir.is_dir() and not category_dir.name.startswith("."):
                for pattern_dir in category_dir.iterdir():
                    if pattern_dir.is_dir() and (pattern_dir / "pattern.toml").exists():
                        pattern_ids.append(pattern_dir.name)

    logger.info(f"Evaluating {len(pattern_ids)} patterns")

    # Evaluate patterns
    pattern_metrics = []
    for pattern_id in pattern_ids:
        logger.info(f"Evaluating pattern: {pattern_id}")
        metrics = await evaluator.evaluate_pattern(pattern_id)
        if metrics:
            pattern_metrics.append(metrics)

    if not pattern_metrics:
        logger.error("No patterns were evaluated")
        return 1

    # Calculate overall metrics
    total_tests = sum(p.total_tests for p in pattern_metrics)
    total_correct = sum(p.correct_count for p in pattern_metrics)
    total_partial = sum(p.partial_count for p in pattern_metrics)

    # Aggregate by test type
    all_positive = [e for p in pattern_metrics for e in p.evaluations if e.test_type == "positive"]
    all_negative = [e for p in pattern_metrics for e in p.evaluations if e.test_type == "negative"]
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
    )

    # Determine output directory
    if args.output_dir is None:
        output_dir = args.patterns_dir.parent / "evals" / "reports" / "judge"
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

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
