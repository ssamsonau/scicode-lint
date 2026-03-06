"""Main evaluation runner for scicode-lint patterns.

Orchestrates running the linter on test cases, validating findings,
calculating metrics, and generating reports.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger

# Handle both module and script execution
if __name__ == "__main__" and __package__ is None:
    # Running as script - add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from evals.metrics import MetricsCalculator, PatternMetrics
    from evals.report_generator import ReportGenerator
    from evals.validators import (
        ActualFinding,
        ExpectedFinding,
        FindingValidator,
        LocationMatcher,
    )
else:
    # Running as module
    from .metrics import MetricsCalculator, PatternMetrics
    from .report_generator import ReportGenerator
    from .validators import (
        ActualFinding,
        ExpectedFinding,
        FindingValidator,
        LocationMatcher,
    )


class EvalRunner:
    """Main orchestration class for running evaluations."""

    def __init__(self, patterns_dir: Path, test_definitions_path: Path, linter_timeout: int = 30):
        """
        Initialize evaluation runner.

        Args:
            patterns_dir: Directory containing pattern test cases
            test_definitions_path: Path to test_definitions.yaml
            linter_timeout: Timeout for linter execution in seconds
        """
        self.patterns_dir = Path(patterns_dir)
        self.test_definitions_path = Path(test_definitions_path)
        self.linter_timeout = linter_timeout

        # Load test definitions
        with open(self.test_definitions_path) as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.validator = FindingValidator(LocationMatcher())
        self.metrics_calculator = MetricsCalculator(
            overall_precision_threshold=self.config.get("thresholds", {}).get(
                "overall_precision", 0.90
            ),
            overall_recall_threshold=self.config.get("thresholds", {}).get("overall_recall", 0.80),
            critical_precision_threshold=self.config.get("thresholds", {}).get(
                "critical_precision", 0.95
            ),
        )

    def run_linter(self, file_path: Path, pattern_id: Optional[str] = None) -> list[ActualFinding]:
        """
        Run the linter on a file and parse findings.

        Args:
            file_path: Path to Python file to lint
            pattern_id: Optional pattern ID to check (e.g., "ml-001-scaler-leakage")
                       If not provided, checks all patterns

        Returns:
            List of ActualFinding objects

        Note:
            This will fail until the actual linter is implemented.
            For now, it returns an empty list.
        """
        try:
            # Build linter command
            cmd = [
                "python",
                "-m",
                "scicode_lint",
                "check",
                str(file_path),
                "--format",
                "json",
            ]

            # Add pattern filter if specified
            if pattern_id:
                cmd.extend(["--pattern", pattern_id])

            # Run linter
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.linter_timeout,
                check=False,
            )

            if result.returncode != 0:
                # Linter doesn't exist yet or file has findings
                # Try to parse JSON output anyway
                pass

            # Parse JSON output
            if result.stdout:
                output = json.loads(result.stdout)
                findings = []

                # Handle list format (linter returns list of file results)
                if isinstance(output, list) and len(output) > 0:
                    # Get findings from first file result
                    for finding_data in output[0].get("findings", []):
                        findings.append(ActualFinding.model_validate(finding_data))
                # Handle dict format (legacy or alternative format)
                elif isinstance(output, dict):
                    for finding_data in output.get("findings", []):
                        findings.append(ActualFinding.model_validate(finding_data))

                return findings
            else:
                return []

        except FileNotFoundError:
            # Linter not installed yet - expected during initial development
            logger.warning(f"Linter not found. Skipping {file_path}")
            return []
        except subprocess.TimeoutExpired:
            logger.error(f"Linter timed out on {file_path}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse linter output for {file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error running linter on {file_path}: {e}")
            return []

    def load_ground_truth(self, pattern_dir: Path) -> dict[str, Any]:
        """
        Load ground truth from pattern.toml or evaluation.yaml.

        Tries pattern.toml first (new format), falls back to evaluation.yaml (legacy).

        Args:
            pattern_dir: Directory containing pattern test cases

        Returns:
            Ground truth configuration dictionary
        """
        # Try new TOML format first
        pattern_toml = pattern_dir / "pattern.toml"
        if pattern_toml.exists():
            return self._load_from_toml(pattern_toml)

        # Fall back to legacy YAML format
        ground_truth_yaml = pattern_dir / "evaluation.yaml"
        if ground_truth_yaml.exists():
            with open(ground_truth_yaml) as f:
                result = yaml.safe_load(f)
                if not isinstance(result, dict):
                    raise ValueError(f"Invalid ground truth file: {ground_truth_yaml}")
                return result

        raise FileNotFoundError(f"No pattern.toml or evaluation.yaml found in {pattern_dir}")

    def _load_from_toml(self, toml_path: Path) -> dict[str, Any]:
        """
        Convert TOML pattern to evaluation format.

        Args:
            toml_path: Path to pattern.toml

        Returns:
            Dictionary in the format expected by evaluate_pattern()
        """
        import sys

        if sys.version_info >= (3, 11):
            import tomllib
        else:
            try:
                import tomli as tomllib
            except ImportError:
                raise ImportError("tomli is required for Python < 3.11")

        with open(toml_path, "rb") as f:
            pattern_data = tomllib.load(f)

        # Convert to evaluation format
        result = {
            "pattern_id": pattern_data["meta"]["id"],
            "category": pattern_data["meta"]["category"],
            "severity": pattern_data["meta"]["severity"],
            "positive_cases": [],
            "negative_cases": [],
            "context_dependent_cases": [],  # Keep legacy name for now, will rename later
        }

        # Convert positive test cases
        for test in pattern_data.get("tests", {}).get("positive", []):
            result["positive_cases"].append(
                {
                    "file": test["file"],
                    "expected_findings": [
                        {
                            "location": {
                                "type": test["expected_location"]["type"],
                                "name": test["expected_location"]["name"],
                                "snippet": test["expected_location"].get("snippet", ""),
                            },
                            "issue": test.get("expected_issue", ""),
                            "min_confidence": test.get("min_confidence", 0.85),
                        }
                    ],
                }
            )

        # Convert negative test cases
        for test in pattern_data.get("tests", {}).get("negative", []):
            result["negative_cases"].append(
                {
                    "file": test["file"],
                    "max_false_positives": test.get("max_false_positives", 0),
                }
            )

        # Convert context-dependent test cases (formerly ambiguous)
        for test in pattern_data.get("tests", {}).get("context_dependent", []):
            result["context_dependent_cases"].append(
                {
                    "file": test["file"],
                    "allow_detection": test.get("allow_detection", True),
                    "allow_skip": test.get("allow_skip", True),
                }
            )

        return result

    def evaluate_pattern(self, pattern_id: str) -> Optional[PatternMetrics]:
        """
        Evaluate a single pattern.

        Args:
            pattern_id: Pattern identifier (e.g., "ml-001")

        Returns:
            PatternMetrics or None if pattern not found
        """
        # Search for pattern in category subdirectories
        pattern_dir = None
        for category_dir in self.patterns_dir.iterdir():
            if category_dir.is_dir():
                candidate = category_dir / pattern_id
                if candidate.exists():
                    pattern_dir = candidate
                    break

        if pattern_dir is None:
            logger.error(f"Pattern directory not found: {pattern_id}")
            return None

        # Load ground truth
        try:
            ground_truth = self.load_ground_truth(pattern_dir)
        except FileNotFoundError:
            logger.error(f"evaluation.yaml not found for {pattern_id}")
            return None
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse evaluation.yaml for {pattern_id}: {e}")
            return None

        # Extract metadata
        category = ground_truth.get("category", "unknown")
        severity = ground_truth.get("severity", "medium")

        # Get the actual pattern ID from ground truth (e.g., "ml-001" not "ml-001-scaler-leakage")
        actual_pattern_id = ground_truth.get("pattern_id", pattern_id)

        # Initialize counters
        total_tp = 0
        total_fp = 0
        total_fn = 0
        positive_count = 0
        negative_count = 0
        context_dependent_count = 0

        # Process positive cases
        for case in ground_truth.get("positive_cases", []):
            positive_count += 1
            case_file = pattern_dir / case["file"]

            if not case_file.exists():
                logger.warning(f"Positive case file not found: {case_file}")
                continue

            # Run linter on this specific pattern only
            actual_findings = self.run_linter(case_file, pattern_id=actual_pattern_id)

            # Parse expected findings
            expected_findings = [
                ExpectedFinding.model_validate(ef) for ef in case.get("expected_findings", [])
            ]

            # Validate
            result = self.validator.validate_positive_case(expected_findings, actual_findings)

            total_tp += result.true_positives
            total_fp += result.false_positives
            total_fn += result.false_negatives

        # Process negative cases
        for case in ground_truth.get("negative_cases", []):
            negative_count += 1
            case_file = pattern_dir / case["file"]

            if not case_file.exists():
                logger.warning(f"Negative case file not found: {case_file}")
                continue

            # Run linter on this specific pattern only
            actual_findings = self.run_linter(case_file, pattern_id=actual_pattern_id)

            # Validate (all findings are FPs)
            max_fps = case.get("max_false_positives", 0)
            result = self.validator.validate_negative_case(actual_findings, max_fps)

            total_fp += result.false_positives

        # Process ambiguous cases (informational only)
        context_dependent_count = len(ground_truth.get("context_dependent_cases", []))

        # Calculate metrics
        metrics = self.metrics_calculator.calculate_pattern_metrics(
            pattern_id=pattern_id,
            category=category,
            severity=severity,
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
            positive_cases=positive_count,
            negative_cases=negative_count,
            context_dependent_cases=context_dependent_count,
        )

        return metrics

    def evaluate_all_patterns(
        self, pattern_filter: Optional[list[str]] = None
    ) -> list[PatternMetrics]:
        """
        Evaluate all patterns (or filtered subset).

        Args:
            pattern_filter: Optional list of pattern IDs to evaluate

        Returns:
            List of PatternMetrics
        """
        # Get all pattern directories from category subdirectories
        pattern_dirs = []
        for category_dir in self.patterns_dir.iterdir():
            if category_dir.is_dir():
                for pattern_dir in category_dir.iterdir():
                    if pattern_dir.is_dir() and (
                        (pattern_dir / "pattern.toml").exists()
                        or (pattern_dir / "evaluation.yaml").exists()
                    ):
                        pattern_dirs.append(pattern_dir)

        # Filter if requested
        if pattern_filter:
            pattern_dirs = [d for d in pattern_dirs if d.name in pattern_filter]

        # Evaluate each pattern
        all_metrics = []
        for pattern_dir in sorted(pattern_dirs):
            pattern_id = pattern_dir.name
            logger.info(f"Evaluating pattern: {pattern_id}")

            metrics = self.evaluate_pattern(pattern_id)
            if metrics:
                all_metrics.append(metrics)

        return all_metrics

    def run(
        self,
        pattern_filter: Optional[list[str]] = None,
        output_dir: Optional[Path] = None,
        output_format: str = "all",
    ) -> int:
        """
        Run full evaluation and generate reports.

        Args:
            pattern_filter: Optional list of pattern IDs to evaluate
            output_dir: Directory for reports (default: evals/reports)
            output_format: Output format: "json", "markdown", or "all"

        Returns:
            Exit code (0 = success, 1 = failure)
        """
        # Determine output directory
        if output_dir is None:
            output_dir = self.patterns_dir.parent / "evals" / "reports"

        # Run evaluations
        logger.info("Starting evaluation...")
        logger.info(f"Patterns directory: {self.patterns_dir}")
        logger.info("")

        pattern_metrics = self.evaluate_all_patterns(pattern_filter)

        if not pattern_metrics:
            logger.error("No patterns evaluated")
            return 1

        # Calculate overall metrics
        overall_metrics = self.metrics_calculator.aggregate_metrics(pattern_metrics)

        # Generate reports
        logger.info("\nGenerating reports...")
        generator = ReportGenerator(output_dir)

        if output_format in ["json", "all"]:
            json_path = generator.generate_json_report(overall_metrics)
            logger.info(f"JSON report: {json_path}")

        if output_format in ["markdown", "all"]:
            md_path = generator.generate_markdown_report(overall_metrics)
            logger.info(f"Markdown report: {md_path}")

        # Print summary
        logger.info("\n" + generator.generate_summary_text(overall_metrics))

        # Return exit code based on thresholds
        if overall_metrics.meets_overall_thresholds and overall_metrics.meets_critical_threshold:
            return 0
        else:
            return 1


def main() -> None:
    """CLI entry point."""
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")

    parser = argparse.ArgumentParser(description="Run scicode-lint evaluation framework")
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
        default=Path(__file__).parent.parent / "patterns",
        help="Directory containing pattern test cases",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "test_definitions.yaml",
        help="Path to test_definitions.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for reports (default: evals/reports)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "all"],
        default="all",
        help="Report format",
    )
    parser.add_argument("--timeout", type=int, default=30, help="Linter timeout in seconds")
    parser.add_argument(
        "--no-auto-server",
        action="store_true",
        help="Don't auto-start vLLM server (assume already running)",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=5001,
        help="vLLM server port (default: 5001)",
    )

    args = parser.parse_args()

    def run_evaluation() -> int:
        """Run the actual evaluation."""
        runner = EvalRunner(
            patterns_dir=args.patterns_dir,
            test_definitions_path=args.config,
            linter_timeout=args.timeout,
        )
        return runner.run(
            pattern_filter=args.patterns,
            output_dir=args.output_dir,
            output_format=args.format,
        )

    # Auto-start vLLM server if needed
    if args.no_auto_server:
        exit_code = run_evaluation()
    else:
        try:
            from scicode_lint.vllm import VLLMServer, get_server_info

            # Check if server is already running
            base_url = f"http://localhost:{args.vllm_port}"
            server_info = get_server_info(base_url=base_url)
            if server_info.is_running:
                logger.info(f"vLLM server already running at {server_info.base_url}")
                exit_code = run_evaluation()
            else:
                logger.info("Starting vLLM server automatically...")
                with VLLMServer(port=args.vllm_port, wait_timeout=120) as server:
                    logger.info(f"vLLM server ready at {server.base_url}")
                    exit_code = run_evaluation()
        except ImportError:
            logger.warning("vLLM utilities not available, running without auto-start")
            exit_code = run_evaluation()
        except Exception as e:
            logger.error(f"Failed to start vLLM server: {e}")
            logger.info("Try running with --no-auto-server if server is already running")
            sys.exit(1)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
